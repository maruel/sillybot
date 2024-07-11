// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package sillybot

import (
	"archive/zip"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/schollz/progressbar/v3"
)

type huggingface struct {
	token string
	cache string
}

// newHuggingFace returns a new huggingface client to download files.
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
	return &huggingface{token: token, cache: cache}, nil
}

// ensure ensures the file is available, downloads it otherwise.
//
// TODO: Support split files.
func (h *huggingface) ensure(ctx context.Context, repo string, filename string, mode os.FileMode) (string, error) {
	dst := filepath.Join(h.cache, filename)
	if _, err := os.Stat(dst); err == nil {
		return dst, err
	}
	url := "https://huggingface.co/" + repo + "/resolve/main/" + filename + "?download=true"
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
	src, err := h.ensure(ctx, repo, model+".llamafile", 0o755)
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
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return fmt.Errorf("failed to download %q: %w", dst, err)
	}
	if token != "" {
		req.Header.Add("Authorization", "Bearer "+token)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to download %q: %w", dst, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode == 401 {
		if token != "" {
			return errors.New("double check if your token is valid")
		}
		return errors.New("a valid token is likely required")
	}
	if resp.StatusCode != 200 {
		return fmt.Errorf("request status: %s", resp.Status)
	}
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
