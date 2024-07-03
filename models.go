// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package sillybot implements the common code used by both discord-bot and
// slack-bot.
package sillybot

import (
	"context"
	"log/slog"
	"net"
	"time"

	"golang.org/x/sync/errgroup"
)

func findFreePort() int {
	l, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		panic(err)
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port
}

// LoadModels loads the LLMInstruct and ImageGen models.
//
// Both take a while to start, so load them in parallel for faster initialization.
func LoadModels(ctx context.Context, cache string, llm string, ig bool) (*LLMInstruct, *ImageGen, error) {
	start := time.Now()
	slog.Info("models", "state", "initializing")
	eg := errgroup.Group{}
	var l *LLMInstruct
	var s *ImageGen
	eg.Go(func() error {
		var err error
		if llm != "" {
			if l, err = NewLLMInstruct(ctx, cache, llm); err != nil {
				slog.Info("llm", "state", "failed", "err", err, "duration", time.Since(start).Round(time.Millisecond), "message", "Try running 'tail -f cache/llm.log'")
			}
		}
		return err
	})
	eg.Go(func() error {
		var err error
		if ig {
			if s, err = NewImageGen(ctx, cache); err != nil {
				slog.Info("ig", "state", "failed", "err", err, "duration", time.Since(start).Round(time.Millisecond), "message", "Try running 'tail -f cache/imagegen.log'")
			}
		}
		return err
	})
	err := eg.Wait()
	if err == nil {
		err = ctx.Err()
	}
	slog.Info("models", "state", "ready", "error", err, "duration", time.Since(start).Round(time.Millisecond))
	return l, s, err
}
