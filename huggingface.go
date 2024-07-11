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

type huggingface struct {
	// serverBase is mocked in test.
	serverBase string
	token      string
	cache      string
}

// newHuggingFace returns a new huggingface client to download files.
//
// It uses the endpoints as described at https://huggingface.co/docs/hub/api.
func newHuggingFace(token string, cache string) (*huggingface, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, err
	}
	tokenFile := filepath.Join(home, ".cache", "huggingface", "token")
	if token == "" {
		t, err := os.ReadFile(tokenFile)
		if err != nil {
			token = strings.TrimSpace(string(t))
		}
	} else {
		if _, err := os.Stat(tokenFile); os.IsNotExist(err) {
			if err = os.WriteFile(tokenFile, []byte(token), 0o644); err != nil {
				return nil, err
			}
		}
	}
	return &huggingface{serverBase: "https://huggingface.co", token: token, cache: cache}, nil
}

// https://huggingface.co/docs/hub/api#get-apimodelsrepoid-or-apimodelsrepoidrevisionrevision
type modelInfoResponse struct {
	Siblings []struct {
		Filename string `json:"rfilename"`
	}
	LastModified time.Time `json:"lastModified"`
	CreatedAt    time.Time `json:"createdAt"`
	SafeTensors  struct {
		Parameters map[string]int64
		Total      int64
	} `json:"safetensors"`
}

type modelInfo struct {
	files    []string
	created  time.Time
	modified time.Time
	tensor   string
	size     int64
}

// listRepo returns the list of files in the repo.
//
// TODO: Support split files.
func (h *huggingface) listRepo(ctx context.Context, repo string) (*modelInfo, error) {
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
	out := &modelInfo{
		files:    make([]string, len(r.Siblings)),
		created:  r.CreatedAt,
		modified: r.LastModified,
	}
	for i := range r.Siblings {
		out.files[i] = r.Siblings[i].Filename
	}
	for k, m := range r.SafeTensors.Parameters {
		if m > out.size {
			out.tensor = k
			out.size = m
		}
	}
	if out.size == 0 {
		out.size = r.SafeTensors.Total
	}
	return out, nil
}

// ensureFile ensures the file is available, downloads it otherwise.
//
// TODO: Support split files.
func (h *huggingface) ensureFile(ctx context.Context, repo string, filename string, mode os.FileMode) (string, error) {
	dst := filepath.Join(h.cache, filename)
	if _, err := os.Stat(dst); err == nil {
		return dst, err
	}
	url := h.serverBase + "/" + repo + "/resolve/main/" + filename + "?download=true"
	return dst, downloadFile(ctx, url, dst, h.token, mode)
}

// ensureGGUFFromLlamafile retrieves a file from an HuggingFace repository if missing.
func (h *huggingface) ensureGGUFFromLlamafile(ctx context.Context, repo string, model string) (string, error) {
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
