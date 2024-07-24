// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package huggingface is the best library to fetch files from an huggingface
// repository.
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

// PackedFileRef is a packed reference to a file in an hugging face repository.
//
// The form is "hf:<author>/<repo>/HEAD/<file>"
//
// It doesn't have allowance for a branch or commit at the moment, HEAD is
// required until further implementation is done.
type PackedFileRef string

// RepoID returns the canonical "<author>/<repo>" for this repository.
func (p PackedFileRef) RepoID() string {
	if i := strings.LastIndexByte(string(p), '/'); i != -1 {
		if i = strings.LastIndexByte(string(p[:i]), '/'); i != -1 {
			return strings.TrimPrefix(string(p)[:i], "hf:")
		}
	}
	return ""
}

// Author returns the <author> part of the packed reference.
func (p PackedFileRef) Author() string {
	if i := strings.IndexByte(string(p), '/'); i != -1 {
		return strings.TrimPrefix(string(p)[:i], "hf:")
	}
	return ""
}

// Repo returns the <repo> part of the packed reference.
func (p PackedFileRef) Repo() string {
	if i := strings.IndexByte(string(p), '/'); i != -1 {
		if j := strings.IndexByte(string(p)[i+1:], '/'); j != -1 {
			return string(p)[i+1 : i+1+j]
		}
	}
	return ""
}

// ModelRef returns the ModelRef reference to the repo containing this file.
func (p PackedFileRef) ModelRef() ModelRef {
	return ModelRef{Author: p.Author(), Repo: p.Repo()}
}

// Basename returns the basename part of this reference.
func (p PackedFileRef) Basename() string {
	if i := strings.LastIndexByte(string(p), '/'); i != -1 {
		return string(p)[i+1:]
	}
	return ""
}

// RepoURL returns the canonical URL for this repository.
func (p PackedFileRef) RepoURL() string {
	return "https://huggingface.co/" + p.RepoID()
}

// Validate checks for obvious errors in the string.
func (p PackedFileRef) Validate() error {
	if !strings.HasPrefix(string(p), "hf:") {
		return fmt.Errorf("invalid file ref %q", p)
	}
	parts := strings.Split(string(p)[4:], "/")
	if len(parts) != 4 {
		return fmt.Errorf("invalid file ref %q", p)
	}
	if parts[2] != "HEAD" {
		// Add allowance for commit later. I'm looking at you, Microsoft.
		return fmt.Errorf("invalid file ref %q", p)
	}
	for _, p := range parts {
		if len(p) < 3 {
			return fmt.Errorf("invalid file ref %q", p)
		}
	}
	return nil
}

// PackedRepoRef is a packed reference to an hugging face repository.
//
// The form is "hf:<author>/<repo>"
type PackedRepoRef string

// RepoID returns the canonical "<author>/<repo>" for this repository.
func (p PackedRepoRef) RepoID() string {
	return strings.TrimPrefix(string(p), "hf:")
}

// ModelRef converts to a ModelRef reference.
func (p PackedRepoRef) ModelRef() ModelRef {
	out := ModelRef{}
	if parts := strings.SplitN(p.RepoID(), "/", 2); len(parts) == 2 {
		out.Author = parts[0]
		out.Repo = parts[1]
	}
	return out
}

// RepoURL returns the canonical URL for this repository.
func (p PackedRepoRef) RepoURL() string {
	return "https://huggingface.co/" + strings.TrimPrefix(string(p), "hf:")
}

// Validate checks for obvious errors in the string.
func (p PackedRepoRef) Validate() error {
	if strings.Count(string(p), "/") != 1 {
		return fmt.Errorf("invalid repo %q", p)
	}
	if !strings.HasPrefix(string(p), "hf:") {
		return fmt.Errorf("invalid repo %q", p)
	}
	parts := strings.Split(string(p)[3:], "/")
	if len(parts) != 2 {
		return fmt.Errorf("invalid repo %q", p)
	}
	for _, p := range parts {
		if len(p) < 3 {
			return fmt.Errorf("invalid repo %q", p)
		}
	}
	return nil
}

// ModelRef is a reference to a model stored on https://huggingface.co
type ModelRef struct {
	// Author is the owner, either a person or an organization.
	Author string
	// Repo is the name of the repository owned by the Author.
	Repo string

	_ struct{}
}

// RepoID is a shorthand to return .m.Author + "/" + m.Repo
func (m *ModelRef) RepoID() string {
	return m.Author + "/" + m.Repo
}

// URL returns the Model's canonical URL.
func (m *ModelRef) URL() string {
	return "https://huggingface.co/" + m.RepoID()
}

// Model is a model stored on https://huggingface.co
type Model struct {
	ModelRef
	// Upstream is the upstream repo when the model is based on another one.
	Upstream ModelRef

	// Information filled by GetModel():

	// Tensor is the native quantization of the weight. Frequently "BF16" for
	// "bfloat16" type. This is found in config.json in Upstream.
	TensorType string
	// Number of weights. Has direct impact on performance and memory usage.
	NumWeights int64
	// ContentLength is the number of tokens that the LLM can take as context
	// when relevant. Has impact on performance and memory usage. Not relevant
	// for image generators.
	ContextLength int
	// License is the license of the weights, for whatever that means. Use the
	// name for well known licences (e.g. "Apache v2.0" or "MIT") or an URL for
	// custom licenses.
	License string
	// LicenseURL is the URL to the license file.
	LicenseURL string
	// Files is the list of files in the repository.
	Files []string
	// Created is the time the repository was created. It can be at the earliest
	// 2022-03-02 as documented at
	// https://huggingface.co/docs/hub/api#repo-listing-api.
	Created time.Time
	// Modified is the last time the repository was modified.
	Modified time.Time

	_ struct{}
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
		BaseModel  string `json:"base_model"`
		License    string
		LicenseURL string `json:"license_link"`
	} `json:"cardData"`
	CreatedAt   time.Time `json:"createdAt"`
	SafeTensors struct {
		Parameters map[string]int64
		Total      int64
	} `json:"safetensors"`
}

// GetModelInfo fills the supplied Model with information from the HuggingFace Hub.
func (c *Client) GetModelInfo(ctx context.Context, m *Model) error {
	slog.Info("hf", "model", m.RepoID())
	url := c.serverBase + "/api/models/" + m.RepoID() + "/revision/HEAD"
	resp, err := authGet(ctx, url, c.token)
	if err != nil {
		return fmt.Errorf("failed to list repoID %s: %w", m.RepoID(), err)
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to list repoID %s: %w", m.RepoID(), err)
	}
	r := modelInfoResponse{}
	if err := json.Unmarshal(b, &r); err != nil {
		return fmt.Errorf("failed to parse list repoID %s response: %w", m.RepoID(), err)
	}
	m.Files = make([]string, len(r.Siblings))
	m.Created = r.CreatedAt
	m.Modified = r.LastModified
	parts := strings.Split(r.CardData.BaseModel, "/")
	if len(parts) == 2 {
		m.Upstream.Author = parts[0]
		m.Upstream.Repo = parts[1]
	}
	m.License = r.CardData.License
	m.LicenseURL = r.CardData.LicenseURL
	for i := range r.Siblings {
		m.Files[i] = r.Siblings[i].Filename
	}
	for k, s := range r.SafeTensors.Parameters {
		if s > m.NumWeights {
			m.TensorType = k
			m.NumWeights = s
		}
	}
	if m.NumWeights == 0 {
		m.NumWeights = r.SafeTensors.Total
	}
	return nil
}

// EnsureFile ensures the file is available, downloads it otherwise.
//
// TODO: Support split files.
func (c *Client) EnsureFile(ctx context.Context, ref PackedFileRef, mode os.FileMode) (string, error) {
	dst := filepath.Join(c.Cache, ref.Basename())
	if _, err := os.Stat(dst); err == nil {
		return dst, err
	}
	url := c.serverBase + "/" + ref.RepoID() + "/resolve/HEAD/" + ref.Basename() + "?download=true"
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
	for i := 0; i < 10; i++ {
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
			if resp.StatusCode == 429 {
				// Sleep and retry.
				time.Sleep(time.Duration(i+1) * time.Second)
				continue
			}
			return nil, fmt.Errorf("request status: %s", resp.Status)
		}
		return resp, err
	}
	return nil, errors.New("failed retrying on 429")
}
