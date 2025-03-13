// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package llamacppsrv is a tiny package to help use llama-server from
// llama.cpp, directly from GitHub releases.
package llamacppsrv

import (
	"archive/zip"
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"

	"golang.org/x/sys/cpu"
)

// DownloadRelease downloads a specific release from GitHub into the specified
// directory and returns the file path to llama.cpp executable.
//
// Returns the file path to the executable and true if it is llamafile, false
// if it is llama-server from llama.cpp.
func DownloadRelease(ctx context.Context, cache string, version int) (string, error) {
	execSuffix := ""
	if runtime.GOOS == "windows" {
		execSuffix = ".exe"
	}
	llamaserver := filepath.Join(cache, "llama-server"+execSuffix)

	build := "b" + strconv.Itoa(version)
	url := "https://github.com/ggerganov/llama.cpp/releases/download/" + build + "/"
	zipname := ""
	wantedFiles := []string{filepath.Base(llamaserver)}
	switch runtime.GOOS {
	case "darwin":
		zipname = "llama-" + build + "-bin-macos-arm64.zip"
		wantedFiles = append(wantedFiles, "*.dylib", "*.metal")
	case "linux":
		zipname = "llama-" + build + "-bin-ubuntu-x64.zip"
		wantedFiles = append(wantedFiles, "*.so")
	case "windows":
		_, err := exec.Command("nvcc", "--version").CombinedOutput()
		if err == nil {
			// This is tricky because in the case of image generation, we may want to
			// run on the CPU instead.
			zipname = "llama-" + build + "-bin-win-cuda-cu12.2.0-x64.zip"
		} else if cpu.X86.HasAVX512BF16 {
			zipname = "llama-" + build + "-bin-win-avx512-x64.zip"
		} else if cpu.X86.HasAVX2 {
			zipname = "llama-" + build + "-bin-win-avx2-x64.zip"
		} else {
			zipname = "llama-" + build + "-bin-win-avx-x64.zip"
		}
		wantedFiles = append(wantedFiles, "ggml.dll", "llama.dll")
	}
	zippath := filepath.Join(cache, zipname)
	if _, err := os.Stat(zippath); err == nil {
		if _, err := os.Stat(llamaserver); err == nil {
			// It both the zip and the executable are present, we are done. There's a
			// small risk that the zip is corrupted or extraction was partial. In this
			// case the user will have to clean up their cache.
			return llamaserver, nil
		}
	}
	if err := downloadFile(ctx, url+zipname, zippath); err != nil {
		return "", fmt.Errorf("failed to download %s from github: %w", zipname, err)
	}

	z, err := zip.OpenReader(zippath)
	if err != nil {
		return "", err
	}
	defer z.Close()
	for _, f := range z.File {
		// Files are under build/bin/; ignore path.
		n := filepath.Base(f.Name)
		for _, desired := range wantedFiles {
			if ok, _ := filepath.Match(desired, n); ok {
				var src io.ReadCloser
				if src, err = f.Open(); err != nil {
					return "", err
				}
				var dst io.WriteCloser
				if dst, err = os.OpenFile(filepath.Join(cache, n), os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0o755); err == nil {
					_, err = io.CopyN(dst, src, int64(f.UncompressedSize64))
				}
				if err2 := src.Close(); err == nil {
					err = err2
				}
				if err2 := dst.Close(); err == nil {
					err = err2
				}
				if err != nil {
					return "", fmt.Errorf("failed to write %q: %w", n, err)
				}
			}
		}
	}
	return llamaserver, err
}

func downloadFile(ctx context.Context, url, dst string) error {
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	// Only then create the file.
	f, err := os.OpenFile(dst, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o666)
	if err != nil {
		return err
	}
	_, err = io.Copy(f, resp.Body)
	if err2 := f.Close(); err == nil {
		err = err2
	}
	return err
}
