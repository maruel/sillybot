// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package sillybot

import (
	"path/filepath"
	"testing"
)

func TestConfig(t *testing.T) {
	cfg := Config{}
	config := filepath.Join(t.TempDir(), "config.yml")
	if err := cfg.LoadOrDefault(config); err != nil {
		t.Fatal(err)
	}
	if len(cfg.KnownLLMs) < 5 {
		t.Fatal("missing known_llms")
	}
	if cfg.Bot.ImageGen.Model != "" {
		t.Fatalf("Oh no, I forgot to disable the image generation in config.yml: %s", cfg.Bot.ImageGen.Model)
	}
}
