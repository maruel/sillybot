// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package llm

import (
	"context"
	"errors"
	"iter"
	"log/slog"
	"runtime/trace"
	"time"

	"github.com/maruel/genai"
)

// Client wraps a Provider client and logs its calls.
type Client struct {
	genai.Provider
}

// GenSync implements genai.Provider
func (c *Client) GenSync(ctx context.Context, msgs []genai.Message, opts ...genai.Options) (genai.Result, error) {
	r := trace.StartRegion(ctx, "llm.GenSync")
	defer r.End()
	if len(msgs) == 0 {
		return genai.Result{}, errors.New("input required")
	}
	start := time.Now()
	slog.Info("llm", "num_msgs", len(msgs), "msg", msgs[len(msgs)-1], "type", "blocking")
	res, err := c.Provider.GenSync(ctx, msgs, opts...)
	if _, ok := err.(*genai.UnsupportedContinuableError); ok {
		err = nil
	}
	if err != nil {
		slog.Error("llm", "msgs", msgs, "err", err, "dur", time.Since(start).Round(time.Millisecond))
	} else {
		slog.Info("llm", "reply", res.String(), "dur", time.Since(start).Round(time.Millisecond), "usage", res.Usage)
	}
	return res, err
}

// GenStream implements genai.Provider
func (c *Client) GenStream(ctx context.Context, msgs []genai.Message, opts ...genai.Options) (iter.Seq[genai.ReplyFragment], func() (genai.Result, error)) {
	r := trace.StartRegion(ctx, "llm.GenStream")
	defer r.End()
	if len(msgs) == 0 {
		return yieldNoFragment, func() (genai.Result, error) {
			return genai.Result{}, errors.New("input required")
		}
	}
	start := time.Now()
	slog.Info("llm", "num_msgs", len(msgs), "msg", msgs[len(msgs)-1], "type", "streaming")
	fragments, finish := c.Provider.GenStream(ctx, msgs, opts...)
	return fragments, func() (genai.Result, error) {
		res, err := finish()
		if _, ok := err.(*genai.UnsupportedContinuableError); ok {
			err = nil
		}
		if err != nil {
			slog.Error("llm", "err", err, "dur", time.Since(start).Round(time.Millisecond))
		} else {
			slog.Info("llm", "dur", time.Since(start).Round(time.Millisecond), "usage", res.Usage)
		}
		return res, err
	}
}

func (c *Client) Unwrap() genai.Provider {
	return c.Provider
}

func yieldNoFragment(yield func(genai.ReplyFragment) bool) {
}
