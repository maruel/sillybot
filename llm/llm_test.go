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
	"github.com/maruel/genai/llamacpp"
	"github.com/mattn/go-colorable"
	"github.com/mattn/go-isatty"
)

func TestLLM(t *testing.T) {
	// Run with -v to list the model sizes.
	const systemPrompt = "You are an AI assistant. You strictly follow orders. Reply exactly with what is asked of you.\n"
	var totalSize int64
	t.Run("qwen3", func(t *testing.T) {
		start := time.Now()
		modelFile := testModel(t, "hf:Qwen/Qwen3-0.6B-GGUF/HEAD/Qwen3-0.6B-Q8_0", systemPrompt)
		i, err := os.Stat(modelFile)
		if err != nil {
			t.Fatalf("%q: %s", modelFile, err)
		}
		// Note: duration is highly relative to the CPU's temperature and thermal
		// throttling. So the first runs will be super fast and then performance
		// will lower significantly. So take the numbers with a grain of salt.
		t.Logf("model %.1fGiB, took %s", float64(i.Size())*0.000000001, time.Since(start).Round(time.Second/10))
		// This only works because the tests are not run in parallel.
		totalSize += i.Size()
	})
	t.Logf("processed %.1fGiB of model", float64(totalSize)*0.000000001)
	t.Run("python", func(t *testing.T) {
		l := loadModel(t, "python")
		testModelInner(t, l, systemPrompt)
	})
}

func testModel(t *testing.T, model PackedFileRef, systemPrompt string) string {
	l := loadModel(t, model)
	testModelInner(t, l, systemPrompt)
	m := llamacpp.Metrics{}
	if err := l.GetMetrics(context.Background(), &m); err != nil {
		t.Fatal(err)
	}
	t.Logf("prompt:    %4d tokens; % 8.2f tok/s", m.Prompt.Count, m.Prompt.Rate())
	t.Logf("generated: %4d tokens; % 8.2f tok/s", m.Generated.Count, m.Generated.Rate())
	return l.modelFile
}

func testModelInner(t *testing.T, l *Session, systemPrompt string) {
	ctx := context.Background()
	const prompt = "reply with \"ok chief\""
	t.Run("Blocking", func(t *testing.T) {
		t.Parallel()
		msgs := genai.Messages{
			genai.NewTextMessage(genai.User, prompt),
		}
		opts := genai.ChatOptions{Seed: 1, SystemPrompt: systemPrompt}
		got, err2 := l.Prompt(ctx, msgs, &opts)
		if err2 != nil {
			t.Fatal(err2)
		}
		checkAnswer(t, got)
	})
	t.Run("Streaming", func(t *testing.T) {
		t.Parallel()
		msgs := []genai.Message{
			genai.NewTextMessage(genai.User, prompt),
		}
		chunks := make(chan genai.MessageFragment)
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
		opts := genai.ChatOptions{Seed: 1, SystemPrompt: systemPrompt}
		err2 := l.PromptStreaming(ctx, msgs, &opts, chunks)
		close(chunks)
		wg.Wait()
		if err2 != nil {
			t.Fatal(err2)
		}
		checkAnswer(t, got)
	})
}

//

// loadModel returns the models in ../default_config.yml to ensure they are valid.
func loadModel(t *testing.T, model PackedFileRef) *Session {
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	opts := Options{Model: model}
	l, err := New(ctx, filepath.Join(filepath.Dir(wd), "cache"), &opts)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err2 := l.Close(); err2 != nil {
			t.Error(err2)
		}
	})
	return l
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
