// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package sillybot

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"path/filepath"
	"strconv"
	"time"
)

// ImageGen manages an image generation server.
type ImageGen struct {
	done    <-chan error
	cancel  func() error
	port    int
	steps   int
	loading bool
}

// NewImageGen initializes a new image generation server.
func NewImageGen(ctx context.Context, cache string) (*ImageGen, error) {
	port := findFreePort()
	cmd := []string{filepath.Join("py", "main.py"), "--port", strconv.Itoa(port)}
	done, cancel, err := runPython(ctx, filepath.Join("py", "venv"), cmd, cache, filepath.Join(cache, "imagegen.log"))
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
	err := ig.cancel()
	if err2 := <-ig.done; err2 != nil {
		err = err2
	}
	return err
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
