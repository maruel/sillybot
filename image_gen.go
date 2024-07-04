// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package sillybot

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"time"
)

var (
	//go:embed py/image_gen.py
	imageGenPy []byte
	//go:embed py/setup.bat
	setupBat []byte
	//go:embed py/setup.sh
	setupSh []byte
)

// ImageGen manages an image generation server.
type ImageGen struct {
	done    <-chan error
	cancel  func() error
	port    int
	steps   int
	loading bool
}

func imageGenNeedRecreate(cache string) bool {
	if _, err := os.Stat(filepath.Join(cache, "venv", "pyvenv.cfg")); err != nil {
		return true
	}
	if b, err := os.ReadFile(filepath.Join(cache, "image_gen.py")); err != nil || !bytes.Equal(b, imageGenPy) {
		return true
	}
	name := "setup.sh"
	content := setupSh
	if runtime.GOOS == "windows" {
		name = "setup.bat"
		content = setupBat
	}
	if b, err := os.ReadFile(filepath.Join(cache, name)); err != nil || !bytes.Equal(b, content) {
		return true
	}
	return false
}

func imageGenRecreate(ctx context.Context, cache string) error {
	if err := os.WriteFile(filepath.Join(cache, "image_gen.py"), imageGenPy, 0o755); err != nil {
		return err
	}
	name := "setup.sh"
	content := setupSh
	if runtime.GOOS == "windows" {
		name = "setup.bat"
		content = setupBat
	}
	if err := os.WriteFile(filepath.Join(cache, name), content, 0o755); err != nil {
		return err
	}
	c := exec.CommandContext(ctx, filepath.Join(cache, name))
	c.Stdout = os.Stdout
	c.Stderr = os.Stderr
	return c.Run()
}

// NewImageGen initializes a new image generation server.
func NewImageGen(ctx context.Context, cache string) (*ImageGen, error) {
	if imageGenNeedRecreate(cache) {
		if err := imageGenRecreate(ctx, cache); err != nil {
			return nil, err
		}
	}

	port := findFreePort()
	cmd := []string{filepath.Join(cache, "image_gen.py"), "--port", strconv.Itoa(port)}
	done, cancel, err := runPython(ctx, filepath.Join(cache, "venv"), cmd, cache, filepath.Join(cache, "imagegen.log"))
	if err != nil {
		return nil, err
	}
	ig := &ImageGen{
		done:    done,
		cancel:  cancel,
		port:    port,
		steps:   1,
		loading: true,
	}
	slog.Info("ig", "state", "started", "port", ig.port, "message", "Please be patient, it can take several minutes to download everything")
	for ctx.Err() == nil {
		if _, err = ig.GenImage("cat"); err == nil {
			break
		}
		select {
		case err := <-ig.done:
			return nil, fmt.Errorf("failed to start: %w", err)
		case <-ctx.Done():
		case <-time.After(100 * time.Millisecond):
		}
	}
	ig.steps = 4
	slog.Info("ig", "state", "ready")
	ig.loading = false
	return ig, nil
}

func (ig *ImageGen) Close() error {
	slog.Info("ig", "state", "terminating")
	ig.cancel()
	return <-ig.done
}

// GenImage returns a PNG encoded image based on the prompt.
func (ig *ImageGen) GenImage(prompt string) ([]byte, error) {
	start := time.Now()
	if !ig.loading {
		// Otherwise it storms on startup.
		slog.Info("ig", "prompt", prompt)
	}
	data := struct {
		Message string `json:"message"`
		Steps   int    `json:"steps"`
		Seed    int    `json:"seed"`
	}{Message: prompt, Steps: ig.steps, Seed: 1}
	b, _ := json.Marshal(data)
	url := fmt.Sprintf("http://localhost:%d/", ig.port)
	resp, err := http.Post(url, "application/json", bytes.NewReader(b))
	if err != nil {
		if !ig.loading {
			// Otherwise it storms on startup.
			slog.Error("ig", "prompt", prompt, "error", err, "duration", time.Since(start).Round(time.Millisecond))
		}
		return nil, err
	}
	d := json.NewDecoder(resp.Body)
	d.DisallowUnknownFields()
	r := struct {
		Image []byte `json:"image"`
	}{}
	err = d.Decode(&r)
	_ = resp.Body.Close()
	if err != nil {
		slog.Error("ig", "prompt", prompt, "error", err, "duration", time.Since(start).Round(time.Millisecond))
		return nil, err
	}
	slog.Info("ig", "prompt", prompt, "duration", time.Since(start).Round(time.Millisecond))
	return r.Image, nil
}
