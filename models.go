// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package sillybot implements the common code used by both discord-bot and
// slack-bot.
package sillybot

import (
	"bytes"
	"context"
	_ "embed"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"time"

	"github.com/maruel/sillybot/imagegen"
	"github.com/maruel/sillybot/llm"
	"golang.org/x/sync/errgroup"
	"gopkg.in/yaml.v3"
)

// Default configuration with well known models and sane presets.
//
//go:embed default_config.yml
var DefaultConfig []byte

// Config defines the configuration format.
type Config struct {
	Bot struct {
		LLM      llm.Options
		ImageGen imagegen.Options `yaml:"image_gen"`
	}
	KnownLLMs []llm.KnownLLM
}

// Validate checks for obvious errors in the fields.
func (c *Config) Validate() error {
	for i := range c.KnownLLMs {
		if err := c.KnownLLMs[i].Validate(); err != nil {
			return err
		}
	}
	return nil
}

// LoadOrDefault loads a config or write the default to disk.
func (c *Config) LoadOrDefault(config string) error {
	b, err := os.ReadFile(config)
	if os.IsNotExist(err) {
		// Write the default config but hack it to skip the knownllms, I'm changing
		// this section too often for now and it's merged in when missing anyway.
		b = DefaultConfig
		if i := bytes.Index(b, []byte("\n# You can remove this section.")); i != -1 {
			b = b[:i]
		}
		if err = os.WriteFile(config, b, 0o644); err != nil {
			return fmt.Errorf("failed to write default config: %w", err)
		}
	}
	d := yaml.NewDecoder(bytes.NewReader(b))
	d.KnownFields(true)
	if err = d.Decode(c); err != nil {
		return fmt.Errorf("failed to read %q: %w", config, err)
	}
	if len(c.KnownLLMs) == 0 {
		// Load the default knownllms from the default config if it was stripped
		// out.
		d := yaml.NewDecoder(bytes.NewReader(DefaultConfig))
		d.KnownFields(true)
		defaultCfg := Config{}
		if err = d.Decode(&defaultCfg); err == nil {
			slog.Info("models", "message", "using default_config.yml's knownllms")
			c.KnownLLMs = defaultCfg.KnownLLMs
		}
	}
	return c.Validate()
}

// LoadModels loads the LLM and ImageGen models.
//
// Both take a while to start, so load them in parallel for faster initialization.
func LoadModels(ctx context.Context, cache string, cfg *Config) (*llm.Session, *imagegen.Session, error) {
	start := time.Now()
	slog.Info("models", "state", "initializing")

	// Hack, since both may create <cache>/py and it would be racy, create it here.
	if cfg.Bot.LLM.Model == "python" || cfg.Bot.ImageGen.Model == "python" {
		cachePy := filepath.Join(cache, "py")
		if err := os.MkdirAll(cachePy, 0o755); err != nil {
			return nil, nil, fmt.Errorf("failed to create the directory to cache python: %w", err)
		}
	}

	eg := errgroup.Group{}
	var l *llm.Session
	var s *imagegen.Session
	eg.Go(func() error {
		if cfg.Bot.LLM.Remote == "" && cfg.Bot.LLM.Model == "" {
			slog.Info("models", "message", "no llm requested")
			return nil
		}
		var err error
		if l, err = llm.New(ctx, cache, &cfg.Bot.LLM, cfg.KnownLLMs); err != nil {
			slog.Info("llm", "state", "failed", "err", err, "duration", time.Since(start).Round(time.Millisecond), "message", "Try running 'tail -f cache/llm.log'")
		}
		return err
	})
	eg.Go(func() error {
		if cfg.Bot.ImageGen.Remote == "" && cfg.Bot.ImageGen.Model == "" {
			slog.Info("models", "message", "no image_gen requested")
			return nil
		}
		var err error
		if s, err = imagegen.New(ctx, cache, &cfg.Bot.ImageGen); err != nil {
			slog.Info("ig", "state", "failed", "err", err, "duration", time.Since(start).Round(time.Millisecond), "message", "Try running 'tail -f cache/image_gen.log'")
		}
		return err
	})
	var err error
	if err = eg.Wait(); err == nil {
		err = ctx.Err()
	}
	slog.Info("models", "state", "ready", "error", err, "duration", time.Since(start).Round(time.Millisecond))
	return l, s, err
}
