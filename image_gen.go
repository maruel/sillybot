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
	"regexp"
	"strconv"
	"time"
)

// ImageGenOptions for NewImageGen.
type ImageGenOptions struct {
	// Remote is the host:port of a pre-existing server to use instead of
	// starting our own.
	Remote string
	// Model specifies a model to use. Use "python" to use the python backend.
	// "python" is currently the only supported value.
	Model string

	_ struct{}
}

// ImageGen manages an image generation server.
type ImageGen struct {
	url    string
	done   <-chan error
	cancel func() error

	steps   int
	loading bool
}

// NewImageGen initializes a new image generation server.
func NewImageGen(ctx context.Context, cache string, opts *ImageGenOptions) (*ImageGen, error) {
	ig := &ImageGen{
		steps:   1,
		loading: true,
	}
	if opts.Remote == "" {
		if opts.Model != "python" {
			return nil, fmt.Errorf("unknown model %q", opts.Model)
		}
		if pyNeedRecreate(cache) {
			if err := pyRecreate(ctx, cache); err != nil {
				return nil, err
			}
		}
		port := findFreePort()
		cmd := []string{filepath.Join(cache, "image_gen.py"), "--port", strconv.Itoa(port)}
		var err error
		ig.done, ig.cancel, err = runPython(ctx, filepath.Join(cache, "venv"), cmd, cache, filepath.Join(cache, "image_gen.log"))
		if err != nil {
			return nil, err
		}
		ig.url = fmt.Sprintf("http://localhost:%d/", port)
	} else {
		if !isHostPort(opts.Remote) {
			return nil, fmt.Errorf("invalid remote %q; use form 'host:port'", opts.Remote)
		}
		ig.url = "http://" + opts.Remote + "/"
	}

	slog.Info("ig", "state", "started", "url", ig.url, "message", "Please be patient, it can take several minutes to download everything")
	for ctx.Err() == nil {
		if _, err := ig.GenImage("cat", 1); err == nil {
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

func (ig *ImageGen) Close() error {
	if ig.cancel == nil {
		return nil
	}
	slog.Info("ig", "state", "terminating")
	_ = ig.cancel()
	return <-ig.done
}

// GenImage returns a PNG encoded image based on the prompt.
func (ig *ImageGen) GenImage(prompt string, seed int) ([]byte, error) {
	start := time.Now()
	if !ig.loading {
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
	b, _ := json.Marshal(data)
	resp, err := http.Post(ig.url, "application/json", bytes.NewReader(b))
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

// isHostPort returns true if the string seems like a valid "host:port" string.
func isHostPort(s string) bool {
	// Simplified regexp that supports IPv4, IPv6 and hostname and requires a port.
	ipv4 := `\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}`
	ipv6 := `\[[a-fA-F0-9:]+\]`
	hostname := `[a-zA-Z0-9\-\.]{2,}`
	r := `^(?:` + ipv4 + `|` + ipv6 + `|` + hostname + `):\d{1,5}$`
	ok, err := regexp.MatchString(r, s)
	if err != nil {
		panic(err)
	}
	return ok
}
