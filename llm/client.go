// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package llm

import (
	"context"
	"errors"
	"log/slog"
	"runtime/trace"
	"time"

	"github.com/maruel/genai"
)

// Client wraps a ProviderGen client and logs its calls.
type Client struct {
	genai.ProviderGen
}

// GenSync implements genai.ProviderGen
func (c *Client) GenSync(ctx context.Context, msgs []genai.Message, opts genai.Options) (genai.Result, error) {
	r := trace.StartRegion(ctx, "llm.GenSync")
	defer r.End()
	if len(msgs) == 0 {
		return genai.Result{}, errors.New("input required")
	}
	start := time.Now()
	slog.Info("llm", "num_msgs", len(msgs), "msg", msgs[len(msgs)-1], "type", "blocking")
	result, err := c.ProviderGen.GenSync(ctx, msgs, opts)
	if _, ok := err.(*genai.UnsupportedContinuableError); ok {
		err = nil
	}
	if err != nil {
		slog.Error("llm", "msgs", msgs, "err", err, "dur", time.Since(start).Round(time.Millisecond))
	} else {
		slog.Info("llm", "reply", result.AsText(), "dur", time.Since(start).Round(time.Millisecond))
	}
	return result, err
}

// GenStream implements genai.ProviderGen
func (c *Client) GenStream(ctx context.Context, msgs []genai.Message, chunks chan<- genai.ContentFragment, opts genai.Options) (genai.Result, error) {
	r := trace.StartRegion(ctx, "llm.GenStream")
	defer r.End()
	if len(msgs) == 0 {
		return genai.Result{}, errors.New("input required")
	}
	start := time.Now()
	slog.Info("llm", "num_msgs", len(msgs), "msg", msgs[len(msgs)-1], "type", "streaming")
	result, err := c.ProviderGen.GenStream(ctx, msgs, chunks, opts)
	if _, ok := err.(*genai.UnsupportedContinuableError); ok {
		err = nil
	}
	if err != nil {
		slog.Error("llm", "err", err, "dur", time.Since(start).Round(time.Millisecond))
	} else {
		slog.Info("llm", "duration", time.Since(start).Round(time.Millisecond), "usage", result)
	}
	return result, err
}

func (c *Client) Unwrap() genai.Provider {
	return c.ProviderGen
}
