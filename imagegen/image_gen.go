// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package imagegen runs an image generator.
package imagegen

import (
	"context"
	_ "embed"
	"fmt"
	"image"
	"log/slog"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"github.com/maruel/httpjson"
	"github.com/maruel/sillybot/internal"
	"github.com/maruel/sillybot/py"
)

// Options for New.
type Options struct {
	// Remote is the host:port of a pre-existing server to use instead of
	// starting our own.
	Remote string
	// Model specifies a model to use. Use "python" to use the python backend.
	// "python" is currently the only supported value.
	Model string

	_ struct{}
}

// Session manages an image generation server.
type Session struct {
	baseURL string
	done    <-chan error
	cancel  func() error

	steps int64
}

// New initializes a new image generation server.
func New(ctx context.Context, cache string, opts *Options) (*Session, error) {
	// Using few steps assumes using a LoRA from Latent Consistency. See
	// https://huggingface.co/blog/lcm_lora for more information.
	ig := &Session{steps: 8}
	if opts.Remote == "" {
		if opts.Model != "python" {
			return nil, fmt.Errorf("unknown model %q", opts.Model)
		}
		cachePy := filepath.Join(cache, "py")
		if err := os.MkdirAll(cachePy, 0o755); err != nil {
			return nil, fmt.Errorf("failed to create the directory to cache python: %w", err)
		}
		if err := py.RecreateVirtualEnvIfNeeded(ctx, cachePy); err != nil {
			return nil, fmt.Errorf("failed to load image_gen: %w", err)
		}
		port := internal.FindFreePort(8032)
		cmd := []string{filepath.Join(cachePy, "image_gen.py"), "--port", strconv.Itoa(port)}
		var err error
		ig.done, ig.cancel, err = py.Run(ctx, filepath.Join(cachePy, "venv"), cmd, cachePy, filepath.Join(cachePy, "image_gen.log"))
		if err != nil {
			return nil, err
		}
		ig.baseURL = fmt.Sprintf("http://localhost:%d", port)
	} else {
		if !internal.IsHostPort(opts.Remote) {
			return nil, fmt.Errorf("invalid remote %q; use form 'host:port'", opts.Remote)
		}
		ig.baseURL = "http://" + opts.Remote
	}

	slog.Info("ig", "state", "started", "url", ig.baseURL, "message", "Please be patient, it can take several minutes to download everything")
	for ctx.Err() == nil {
		r := struct {
			Status string
		}{}
		c := httpjson.DefaultClient
		c.Compress = ""
		if err := c.Get(ctx, ig.baseURL+"/health", nil, &r); err == nil && r.Status == "ok" {
			break
		}
		select {
		case err := <-ig.done:
			return nil, fmt.Errorf("failed to start: %w", err)
		case <-ctx.Done():
		case <-time.After(100 * time.Millisecond):
		}
	}
	slog.Info("ig", "state", "ready")
	return ig, nil
}

func (ig *Session) Close() error {
	if ig.cancel == nil {
		return nil
	}
	slog.Info("ig", "state", "terminating")
	_ = ig.cancel()
	return <-ig.done
}

// GenImage returns an image based on the prompt.
//
// Use a non-zero seed to get deterministic output (without strong guarantees).
func (ig *Session) GenImage(ctx context.Context, prompt string, seed int64) (*image.NRGBA, error) {
	start := time.Now()
	slog.Info("ig", "prompt", prompt)
	// If you feel this API is subpar, I hear you. If you got this far to read
	// this comment, please send a PR to make this a proper API and update
	// image_gen.py. ❤
	data := struct {
		Message string `json:"message"`
		Steps   int64  `json:"steps"`
		Seed    int64  `json:"seed"`
	}{Message: prompt, Steps: ig.steps, Seed: seed}
	r := struct {
		Image []byte `json:"image"`
	}{}
	c := httpjson.DefaultClient
	c.Compress = ""
	if err := c.Post(ctx, ig.baseURL+"/api/generate", nil, data, &r); err != nil {
		slog.Error("ig", "prompt", prompt, "error", err, "duration", time.Since(start).Round(time.Millisecond))
		return nil, fmt.Errorf("failed to create image request: %w", err)
	}
	slog.Info("ig", "prompt", prompt, "duration", time.Since(start).Round(time.Millisecond))

	img, err := decodePNG(r.Image)
	if err != nil {
		return nil, err
	}
	addWatermark(img)
	return img, nil
}
