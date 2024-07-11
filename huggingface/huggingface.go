// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package huggingface

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/schollz/progressbar/v3"
)

// Model is a model stored on https://huggingface.co
type Model struct {
	// Repo is in the form <user>/<project>.
	Repo string
	// Upstream is the upstream repo when the model is based on another one.
	Upstream string
	// PackagingType is the file format used in the model. It can be one of
	// "safetensors", "gguf" or "llamafile".
	PackagingType string
	// Basename is the base filename when PackagingType is one of "gguf" or
	// "llamafile".
	Basename string
	// Tensor is the native quantization of the weight. Frequently "BF16" for
	// "bfloat16" type. This is found in config.json in Upstream.
	TensorType string
	// Number of weights. Has direct impact on performance and memory usage.
	NumWeight int
	// ContentLength is the number of tokens that the LLM can take as context
	// when relevant. Has impact on performance and memory usage. Not relevant
	// for image generators.
	ContextLength int
	// License is the license of the weights, for whatever that means. Use the
	// name for well known licences (e.g. "Apache v2.0" or "MIT") or an URL for
	// custom licenses.
	License string

	_ struct{}
}

// URL returns the Model's canonical URL.
func (m *Model) URL() string {
	return "https://huggingface.co/" + m.Repo
}

// Client is the client for https://huggingface.co/.
type Client struct {
	Cache string

	// serverBase is mocked in test.
	serverBase string
	token      string
}

// New returns a new *Client client to download files and list repositories.
//
// It uses the endpoints as described at https://huggingface.co/docs/hub/api.
func New(token string, cache string) (*Client, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, err
	}
	tokenFile := filepath.Join(home, ".cache", "huggingface", "token")
	if token == "" {
		if t, err := os.ReadFile(tokenFile); err == nil {
			token = strings.TrimSpace(string(t))
			slog.Info("hf", "message", "found token from cache", "file", tokenFile)
		}
	} else {
		if _, err := os.Stat(tokenFile); os.IsNotExist(err) {
			if err = os.WriteFile(tokenFile, []byte(token), 0o644); err != nil {
				return nil, err
			}
			slog.Info("hf", "message", "saved token to cache", "file", tokenFile)
		}
	}
	if token != "" && !strings.HasPrefix(token, "hf_") {
		return nil, errors.New("token is invalid, it must have prefix 'hf_'")
	}
	return &Client{serverBase: "https://huggingface.co", token: token, Cache: cache}, nil
}

// https://huggingface.co/docs/hub/api#get-apimodelsrepoid-or-apimodelsrepoidrevisionrevision
type modelInfoResponse struct {
	LastModified time.Time `json:"lastModified"`
	Siblings     []struct {
		Filename string `json:"rfilename"`
	}
	CardData struct {
		License    string
		LicenseURL string `json:"license_link"`
	} `json:"cardData"`
	CreatedAt   time.Time `json:"createdAt"`
	SafeTensors struct {
		Parameters map[string]int64
		Total      int64
	} `json:"safetensors"`
}

// ModelInfo contains information retrieved from the repo.
type ModelInfo struct {
	Files      []string
	Created    time.Time
	Modified   time.Time
	Tensor     string
	Size       int64
	License    string
	LicenseURL string

	_ struct{}
}

// ListRepo returns the list of files in the repo.
//
// TODO: Support split files.
func (c *Client) ListRepo(ctx context.Context, repo string) (*ModelInfo, error) {
	url := c.serverBase + "/api/models/" + repo + "/revision/main"
	resp, err := authGet(ctx, url, c.token)
	if err != nil {
		return nil, fmt.Errorf("failed to list repo %s: %w", repo, err)
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to list repo %s: %w", repo, err)
	}
	r := modelInfoResponse{}
	if err := json.Unmarshal(b, &r); err != nil {
		return nil, fmt.Errorf("failed to parse list repo %s response: %w", repo, err)
	}
	out := &ModelInfo{
		Files:      make([]string, len(r.Siblings)),
		Created:    r.CreatedAt,
		Modified:   r.LastModified,
		License:    r.CardData.License,
		LicenseURL: r.CardData.LicenseURL,
	}
	for i := range r.Siblings {
		out.Files[i] = r.Siblings[i].Filename
	}
	for k, m := range r.SafeTensors.Parameters {
		if m > out.Size {
			out.Tensor = k
			out.Size = m
		}
	}
	if out.Size == 0 {
		out.Size = r.SafeTensors.Total
	}
	return out, nil
}

// EnsureFile ensures the file is available, downloads it otherwise.
//
// TODO: Support split files.
func (c *Client) EnsureFile(ctx context.Context, repo string, filename string, mode os.FileMode) (string, error) {
	dst := filepath.Join(c.Cache, filename)
	if _, err := os.Stat(dst); err == nil {
		return dst, err
	}
	url := c.serverBase + "/" + repo + "/resolve/main/" + filename + "?download=true"
	return dst, DownloadFile(ctx, url, dst, c.token, mode)
}

// DownloadFile downloads a file optionally with a bearer token.
//
// It prints a progress bar.
func DownloadFile(ctx context.Context, url, dst string, token string, mode os.FileMode) error {
	slog.Info("hf", "downloading", url)
	resp, err := authGet(ctx, url, token)
	if err != nil {
		return fmt.Errorf("failed to download %q: %w", dst, err)
	}
	defer resp.Body.Close()
	// Only then create the file.
	f, err := os.OpenFile(dst, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, mode)
	if err != nil {
		return fmt.Errorf("failed to download %q: %w", dst, err)
	}
	defer f.Close()
	// This is iffy to spam the user but necessary for large files.
	// TODO: check if resp.ContentLength is small and skip output in this case.
	bar := progressbar.DefaultBytes(resp.ContentLength, "downloading")
	_, err = io.Copy(io.MultiWriter(f, bar), resp.Body)
	return err
}

// authGet does an authenticated HTTP request with a Bearer token.
func authGet(ctx context.Context, url, token string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		// Unlikely.
		return nil, err
	}
	if token != "" {
		req.Header.Add("Authorization", "Bearer "+token)
	}
	resp, err := http.DefaultClient.Do(req)
	if resp.StatusCode != 200 {
		_, _ = io.Copy(io.Discard, resp.Body)
		_ = resp.Body.Close()
		if resp.StatusCode == 401 {
			if token != "" {
				return nil, fmt.Errorf("double check if your token is valid: %s", resp.Status)
			}
			return nil, fmt.Errorf("a valid token is likely required: %s", resp.Status)
		}
		return nil, fmt.Errorf("request status: %s", resp.Status)
	}
	return resp, err
}
