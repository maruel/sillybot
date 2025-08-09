// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package llm

import (
	"context"
	"flag"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/lmittmann/tint"
	"github.com/maruel/genai"
	"github.com/mattn/go-colorable"
	"github.com/mattn/go-isatty"
)

func TestLLM(t *testing.T) {
	// Run with -v to list the model sizes.
	const systemPrompt = "You are an AI assistant. You strictly follow orders. Reply exactly with what is asked of you.\n"
	t.Run("qwen3", func(t *testing.T) {
		testModel(t, "llama-server", "hf:Qwen/Qwen3-0.6B-GGUF/HEAD/Qwen3-0.6B-Q8_0", systemPrompt)
	})
	t.Run("python", func(t *testing.T) {
		if testing.Short() {
			t.Skip("skipping test case in short mode")
		}
		testModel(t, "python", "", systemPrompt)
	})
}

func testModel(t *testing.T, backend string, model PackedFileRef, systemPrompt string) {
	p := loadBackend(t, backend, model)
	ctx := t.Context()
	const prompt = "reply with \"ok chief\""
	t.Run("Blocking", func(t *testing.T) {
		t.Parallel()
		msgs := genai.Messages{
			genai.NewTextMessage(prompt),
		}
		opts := genai.OptionsText{Seed: 1, SystemPrompt: systemPrompt}
		result, err2 := p.GenSync(ctx, msgs, &opts)
		if err2 != nil {
			if _, ok := err2.(*genai.UnsupportedContinuableError); !ok {
				t.Fatal(err2)
			}
		}
		t.Logf("generated: %4d tokens; returned: %4d", result.InputTokens, result.OutputTokens)
		checkAnswer(t, result.AsText())
	})
	t.Run("Streaming", func(t *testing.T) {
		t.Parallel()
		msgs := []genai.Message{
			genai.NewTextMessage(prompt),
		}
		chunks := make(chan genai.ContentFragment)
		got := ""
		wg := sync.WaitGroup{}
		wg.Add(1)
		go func() {
			for pkt := range chunks {
				if pkt.TextFragment == "" {
					t.Errorf("expected Text type, got %#v", pkt)
				}
				got += pkt.TextFragment
			}
			wg.Done()
		}()
		opts := genai.OptionsText{Seed: 1, SystemPrompt: systemPrompt}
		result, err2 := p.GenStream(ctx, msgs, chunks, &opts)
		close(chunks)
		wg.Wait()
		if err2 != nil {
			if _, ok := err2.(*genai.UnsupportedContinuableError); !ok {
				t.Fatal(err2)
			}
		}
		t.Logf("generated: %4d tokens; returned: %4d", result.InputTokens, result.OutputTokens)
		checkAnswer(t, got)
	})
}

//

// loadBackend returns the models in ../default_config.yml to ensure they are valid.
func loadBackend(t *testing.T, backend string, model PackedFileRef) genai.ProviderGen {
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	opts := Options{Backend: backend, Model: model}
	l, err := New(ctx, filepath.Join(filepath.Dir(wd), "cache"), &opts)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err2 := l.Close(); err2 != nil {
			t.Error(err2)
		}
	})
	return l.Client()
}

func checkAnswer(t *testing.T, got string) {
	// Work around various non-determinism.
	processed := strings.ToLower(got)
	// Accept ok, chief.
	processed = strings.Replace(processed, ",", "", 1)
	if want := "ok chief"; !strings.Contains(processed, want) {
		if runtime.GOOS == "darwin" && os.Getenv("CI") == "true" && os.Getenv("GITHUB_ACTION") != "" {
			t.Log("TODO: Figure out why macOS GitHub hosted runner return an empty string")
		} else {
			t.Helper()
			t.Fatalf("expected %q, got %q", want, got)
		}
	}
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
