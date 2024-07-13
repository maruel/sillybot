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
	t.Skip("skipping because it's broken when using llamafile")
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
	t.Skip("skipping because it's broken when using llamafile")
	testModelStreaming(t, "Phi-3-mini-4k-instruct.Q5_K_M")
}

func testModel(t *testing.T, model string) {
	if testing.Short() {
		t.Skip("skipping test case in short mode")
	}
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	opts := Options{
		Model:        model,
		SystemPrompt: "Transcript of a never ending dialog, where the User interacts with an Assistant.\nThe Assistant is helpful, kind, honest, good at writing, and never fails to answer the User's requests immediately and with precision.",
	}
	l, err := New(ctx, filepath.Join(filepath.Dir(wd), "cache"), &opts, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err2 := l.Close(); err2 != nil {
			t.Error(err2)
		}
	})
	msgs := []Message{
		{Role: System, Content: "You are an AI assistant. You strictly follow orders. Do not add punctuation. Do not use uppercase letters."},
		{Role: User, Content: "reply with \"ok chief\""},
	}
	got, err := l.Prompt(ctx, msgs, 1, 0.1)
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
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	opts := Options{
		Model:        model,
		SystemPrompt: "Transcript of a never ending dialog, where the User interacts with an Assistant.\nThe Assistant is helpful, kind, honest, good at writing, and never fails to answer the User's requests immediately and with precision.",
	}
	l, err := New(ctx, filepath.Join(filepath.Dir(wd), "cache"), &opts, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err2 := l.Close(); err2 != nil {
			t.Error(err2)
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
	err = l.PromptStreaming(ctx, msgs, 1, 0.1, words)
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
