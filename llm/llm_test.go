// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package llm

import (
	"bytes"
	"context"
	"flag"
	"log/slog"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/lmittmann/tint"
	"github.com/mattn/go-colorable"
	"github.com/mattn/go-isatty"
	"gopkg.in/yaml.v3"
)

func TestLLM(t *testing.T) {
	re := regexp.MustCompile(`-(\d+)[bBk]-`)
	for _, k := range loadKnownLLMs(t) {
		t.Run(k.Basename, func(t *testing.T) {
			b := re.FindStringSubmatch(k.Basename)
			if len(b) != 2 {
				t.Skip("skipping complex")
			}
			size, err := strconv.Atoi(b[1])
			if err != nil {
				t.Fatal(err)
			}
			if size > 9 {
				t.Skip("too large")
			}
			if !strings.HasPrefix(k.Basename, "Mistral") && testing.Short() {
				t.Skip("skipping this model when -short is used")
			}
			if strings.HasPrefix(k.Basename, "phi-3") {
				// I suspect it's because it's already a small model so it fails at
				// high quantization.
				t.Skip("phi-3-mini is misbehaving. TODO: investigate")
			}
			quant := "Q3_K_M"
			// Hack: naming conventions are not figured out.
			if strings.Contains(strings.ToLower(k.Basename), "qwen") {
				quant = strings.ToLower(quant)
			}
			// I tested with Q2_K and results are unreliable.
			t.Run("Blocking", func(t *testing.T) {
				testModelBlocking(t, k.Basename+quant)
			})
			t.Run("Stream", func(t *testing.T) {
				testModelStreaming(t, k.Basename+quant)
			})
		})
	}
}

func testModelBlocking(t *testing.T, model string) {
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	opts := Options{
		Model:        model,
		SystemPrompt: "You are an AI assistant. You strictly follow orders. Do not add extraneous words. Only reply with what is asked of you.",
	}
	l, err := New(ctx, filepath.Join(filepath.Dir(wd), "cache"), &opts, loadKnownLLMs(t))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err2 := l.Close(); err2 != nil {
			t.Error(err2)
		}
	})
	msgs := []Message{
		{Role: System, Content: opts.SystemPrompt},
		{Role: User, Content: "reply with \"ok chief\""},
	}
	got, err := l.Prompt(ctx, msgs, 1, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	// Work around various non-determinism.
	if want := "ok chief"; !strings.Contains(strings.ToLower(got), want) {
		if runtime.GOOS == "darwin" && os.Getenv("CI") == "true" && os.Getenv("GITHUB_ACTION") != "" {
			t.Log("TODO: Figure out why macOS GitHub hosted runner return an empty string")
		} else {
			t.Fatalf("expected %q, got %q", want, got)
		}
	}
}

func testModelStreaming(t *testing.T, model string) {
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	opts := Options{
		Model:        model,
		SystemPrompt: "You are an AI assistant. You strictly follow orders. Do not add extraneous words. Only reply with what is asked of you.",
	}
	l, err := New(ctx, filepath.Join(filepath.Dir(wd), "cache"), &opts, loadKnownLLMs(t))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err2 := l.Close(); err2 != nil {
			t.Error(err2)
		}
	})
	msgs := []Message{
		{Role: System, Content: opts.SystemPrompt},
		{Role: User, Content: "reply with \"ok chief\""},
	}
	words := make(chan string, 10)
	got := ""
	wg := sync.WaitGroup{}
	wg.Add(1)
	go func() {
		for c := range words {
			got += c
		}
		wg.Done()
	}()
	err = l.PromptStreaming(ctx, msgs, 1, 0.1, words)
	close(words)
	wg.Wait()
	if err != nil {
		t.Fatal(err)
	}
	// Work around various non-determinism.
	if want := "ok chief"; !strings.Contains(strings.ToLower(got), want) {
		if runtime.GOOS == "darwin" && os.Getenv("CI") == "true" && os.Getenv("GITHUB_ACTION") != "" {
			t.Log("TODO: Figure out why macOS GitHub hosted runner return an empty string")
		} else {
			t.Fatalf("expected %q, got %q", want, got)
		}
	}
}

func loadKnownLLMs(t *testing.T) []KnownLLM {
	b, err := os.ReadFile("../default_config.yml")
	if err != nil {
		t.Fatal(err)
	}
	c := struct {
		KnownLLMs []KnownLLM
	}{}
	d := yaml.NewDecoder(bytes.NewReader(b))
	if err = d.Decode(&c); err != nil {
		t.Fatal(err)
	}
	if len(c.KnownLLMs) < 5 {
		t.Fatalf("Expected more known LLMs\n%# v", c.KnownLLMs)
	}
	return c.KnownLLMs
}

// TestMain sets up the verbose logging.
func TestMain(m *testing.M) {
	flag.Parse()
	l := slog.LevelWarn
	if os.Getenv("LLM_TEST_VERBOSE") == "true" {
		l = slog.LevelDebug
	}
	logger := slog.New(tint.NewHandler(colorable.NewColorable(os.Stderr), &tint.Options{
		Level:      l,
		TimeFormat: time.TimeOnly,
		NoColor:    !isatty.IsTerminal(os.Stderr.Fd()),
	}))
	slog.SetDefault(logger)
	os.Exit(m.Run())
}
