// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package llamacppsrv

import (
	"context"
	"strconv"
	"strings"
	"testing"

	"github.com/maruel/genai/genaiapi"
	"github.com/maruel/genai/llamacpp"
	"github.com/maruel/huggingface"
	"github.com/maruel/sillybot/internal"
)

func TestNewServer_Query(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}
	ctx := context.Background()
	cache := t.TempDir()
	// It's a bit inefficient to download from github every single time.
	exe, err := DownloadRelease(ctx, cache, 4882)
	if err != nil {
		t.Fatal(err)
	}
	port := internal.FindFreePort(10000)
	hf, err := huggingface.New("")
	if err != nil {
		t.Fatal(err)
	}
	// A really small model.
	modelPath, err := hf.EnsureFile(ctx, huggingface.ModelRef{Author: "Qwen", Repo: "Qwen2-0.5B-Instruct-GGUF"}, "HEAD", "qwen2-0_5b-instruct-q2_k.gguf")
	if err != nil {
		t.Fatal(err)
	}
	srv, err := NewServer(ctx, exe, modelPath, cache, port, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err2 := srv.Close(); err2 != nil {
			t.Error(err2)
		}
	})

	c, err := llamacpp.New("http://localhost:"+strconv.Itoa(port), nil)
	if err != nil {
		t.Fatal(err)
	}
	msgs := []genaiapi.Message{{Role: genaiapi.User, Type: genaiapi.Text, Text: "Say hello. Reply with one word."}}
	opts := genaiapi.CompletionOptions{Seed: 1, MaxTokens: 10, Temperature: 0.01}
	out, err := c.Completion(ctx, msgs, &opts)
	if err != nil {
		t.Fatal(err)
	}
	// This is obviously brittle but it works often enough for now.
	txt := strings.ToLower(out.Text)
	txt = strings.TrimRight(txt, ".!")
	if txt != "hello" {
		t.Fatal("unexpected response:", txt)
	}
}
