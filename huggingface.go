// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package sillybot

import (
	"archive/zip"
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

type HuggingFace struct {
	// serverBase is mocked in test.
	serverBase string
	token      string
	cache      string
}

// newHuggingFace returns a new HuggingFace client to download files.
//
// It uses the endpoints as described at https://huggingface.co/docs/hub/api.
func newHuggingFace(token string, cache string) (*HuggingFace, error) {
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
	return &HuggingFace{serverBase: "https://huggingface.co", token: token, cache: cache}, nil
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
func (h *HuggingFace) ListRepo(ctx context.Context, repo string) (*ModelInfo, error) {
	url := h.serverBase + "/api/models/" + repo + "/revision/main"
	resp, err := authGet(ctx, url, h.token)
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

// ensureFile ensures the file is available, downloads it otherwise.
//
// TODO: Support split files.
func (h *HuggingFace) ensureFile(ctx context.Context, repo string, filename string, mode os.FileMode) (string, error) {
	dst := filepath.Join(h.cache, filename)
	if _, err := os.Stat(dst); err == nil {
		return dst, err
	}
	url := h.serverBase + "/" + repo + "/resolve/main/" + filename + "?download=true"
	return dst, downloadFile(ctx, url, dst, h.token, mode)
}

// ensureGGUFFromLlamafile retrieves a file from an HuggingFace repository if missing.
func (h *HuggingFace) ensureGGUFFromLlamafile(ctx context.Context, repo string, model string) (string, error) {
	gguf := model + ".gguf"
	dstgguf := filepath.Join(h.cache, gguf)
	if _, err := os.Stat(dstgguf); err == nil {
		return dstgguf, err
	}
	// Get the llamafile first.
	src, err := h.ensureFile(ctx, repo, model+".llamafile", 0o755)
	if err != nil {
		return dstgguf, err
	}
	z, err := zip.OpenReader(src)
	if err != nil {
		return dstgguf, err
	}
	defer z.Close()
	for _, i := range z.File {
		if i.Name == gguf {
			s, err := i.Open()
			if err != nil {
				return dstgguf, err
			}
			defer s.Close()
			d, err := os.OpenFile(dstgguf, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
			if err != nil {
				return dstgguf, err
			}
			defer d.Close()
			_, err = io.CopyN(d, s, int64(i.UncompressedSize64))
			return dstgguf, err
		}
	}
	return dstgguf, errors.New("gguf not found")
}

// downloadFile downloads a file optionally with a bearer token.
func downloadFile(ctx context.Context, url, dst string, token string, mode os.FileMode) error {
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
