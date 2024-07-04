// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package sillybot

import (
	"context"
	"flag"
	"log/slog"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/lmittmann/tint"
	"github.com/mattn/go-colorable"
	"github.com/mattn/go-isatty"
)

func TestLLM_Llama_3(t *testing.T) {
	testModel(t, "Meta-Llama-3-8B-Instruct.Q5_K_M")
}

func TestLLM_Gemma_2(t *testing.T) {
	t.Skip("skipping by default; it's a bit slow")
	testModel(t, "gemma-2-27b-it.Q6_K")
}

func TestLLM_Phi_3_Mini(t *testing.T) {
	t.Skip("skipping because it's broken; the model doesn't follow orders")
	testModel(t, "Phi-3-mini-4k-instruct.Q5_K_M")
}

func TestLLM_Stream_Llama_3(t *testing.T) {
	testModelStreaming(t, "Meta-Llama-3-8B-Instruct.Q5_K_M")
}

func TestLLM_Stream_Gemma_2(t *testing.T) {
	t.Skip("skipping by default; it's a bit slow")
	testModelStreaming(t, "gemma-2-27b-it.Q6_K")
}

func TestLLM_Stream_Phi_3_Mini(t *testing.T) {
	t.Skip("skipping because it's broken; the model doesn't follow orders")
	testModelStreaming(t, "Phi-3-mini-4k-instruct.Q5_K_M")
}

func testModel(t *testing.T, model string) {
	if testing.Short() {
		t.Skip("skipping test case in short mode")
	}
	cache, err := filepath.Abs("cache")
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	l, err := NewLLM(ctx, cache, model)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := l.Close(); err != nil {
			t.Error(err)
		}
	})
	msgs := []Message{
		{Role: System, Content: "You are an AI assistant. You strictly follow orders. Do not add punctuation. Do not use uppercase letters."},
		{Role: User, Content: "reply with \"ok chief\""},
	}
	got, err := l.Prompt(ctx, msgs)
	if err != nil {
		t.Fatal(err)
	}
	want := "ok chief"
	if got != want {
		t.Fatalf("expected %s, got %s", want, got)
	}
}

func testModelStreaming(t *testing.T, model string) {
	if testing.Short() {
		t.Skip("skipping test case in short mode")
	}
	cache, err := filepath.Abs("cache")
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	l, err := NewLLM(ctx, cache, model)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := l.Close(); err != nil {
			t.Error(err)
		}
	})
	msgs := []Message{
		{Role: System, Content: "You are an AI assistant. You strictly follow orders. Do not add punctuation. Do not use uppercase letters."},
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
	err = l.PromptStreaming(ctx, msgs, words)
	close(words)
	wg.Wait()
	if err != nil {
		t.Fatal(err)
	}
	want := "ok chief"
	if got != want {
		t.Fatalf("expected %s, got %s", want, got)
	}
}

// TestMain sets up the verbose logging.
func TestMain(m *testing.M) {
	flag.Parse()
	l := slog.LevelWarn
	if testing.Verbose() {
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
