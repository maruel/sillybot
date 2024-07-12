// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package imagegen runs an image generator.
package imagegen

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"image"
	"image/draw"
	"image/png"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"time"

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
	url    string
	done   <-chan error
	cancel func() error

	steps   int
	loading bool
}

// New initializes a new image generation server.
func New(ctx context.Context, cache string, opts *Options) (*Session, error) {
	ig := &Session{
		steps:   1,
		loading: true,
	}
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
		port := py.FindFreePort()
		cmd := []string{filepath.Join(cachePy, "image_gen.py"), "--port", strconv.Itoa(port)}
		var err error
		ig.done, ig.cancel, err = py.Run(ctx, filepath.Join(cachePy, "venv"), cmd, cachePy, filepath.Join(cachePy, "image_gen.log"))
		if err != nil {
			return nil, err
		}
		ig.url = fmt.Sprintf("http://localhost:%d/", port)
	} else {
		if !py.IsHostPort(opts.Remote) {
			return nil, fmt.Errorf("invalid remote %q; use form 'host:port'", opts.Remote)
		}
		ig.url = "http://" + opts.Remote + "/"
	}

	slog.Info("ig", "state", "started", "url", ig.url, "message", "Please be patient, it can take several minutes to download everything")
	for ctx.Err() == nil {
		if _, err := ig.GenImage(ctx, "cat", 1); err == nil {
			break
		}
		select {
		case err := <-ig.done:
			return nil, fmt.Errorf("failed to start: %w", err)
		case <-ctx.Done():
		case <-time.After(100 * time.Millisecond):
		}
	}
	// Using few steps assumes using a LoRA from Latent Consistency. See
	// https://huggingface.co/blog/lcm_lora for more information.
	ig.steps = 8
	slog.Info("ig", "state", "ready")
	ig.loading = false
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
func (ig *Session) GenImage(ctx context.Context, prompt string, seed int) (*image.NRGBA, error) {
	start := time.Now()
	if !ig.loading || slog.Default().Enabled(ctx, slog.LevelDebug) {
		// Otherwise it storms on startup.
		slog.Info("ig", "prompt", prompt)
	}
	// If you feel this API is subpar, I hear you. If you got this far to read
	// this comment, please send a PR to make this a proper API and update
	// image_gen.py. ❤
	data := struct {
		Message string `json:"message"`
		Steps   int    `json:"steps"`
		Seed    int    `json:"seed"`
	}{Message: prompt, Steps: ig.steps, Seed: seed}
	b, err := json.Marshal(data)
	if err != nil {
		return nil, fmt.Errorf("internal error: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, "POST", ig.url, bytes.NewReader(b))
	if err != nil {
		return nil, fmt.Errorf("failed to create HTTP request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		if !ig.loading || slog.Default().Enabled(ctx, slog.LevelDebug) {
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

	img, err := decodePNG(r.Image)
	if err != nil {
		return nil, err
	}
	addWatermark(img)
	return img, nil
}

//

//go:embed mascot.png
var mascotPNG []byte

var mascot = func() *image.NRGBA {
	img, err := decodePNG(mascotPNG)
	if err != nil {
		panic(err)
	}
	for i := 0; i < len(img.Pix); i += 4 {
		img.Pix[i+3] = img.Pix[i+3] >> 2
	}
	return img
}()

func addWatermark(img *image.NRGBA) {
	d := img.Bounds()
	m := mascot.Bounds()
	draw.Draw(img, m.Add(image.Pt(0, d.Dy()-m.Dy())), mascot, image.Point{}, draw.Over)
}

func decodePNG(b []byte) (*image.NRGBA, error) {
	img, err := png.Decode(bytes.NewReader(b))
	if err != nil {
		return nil, fmt.Errorf("failed to decode PNG: %w", err)
	}
	switch n := img.(type) {
	case *image.NRGBA:
		return n, nil
	case *image.RGBA:
		// Convert.
		b := n.Bounds()
		dst := image.NewNRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
		draw.Draw(dst, dst.Bounds(), n, b.Min, draw.Src)
		return dst, nil
	default:
		return nil, fmt.Errorf("failed to decode PNG: expected NRGBA, got %T", img)
	}
}
