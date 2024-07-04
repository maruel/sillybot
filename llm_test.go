// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package sillybot

import (
	"context"
	"path/filepath"
	"testing"
)

func TestLLM(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test case in short mode")
	}
	cache, err := filepath.Abs("cache")
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	l, err := NewLLM(ctx, cache, "Meta-Llama-3-8B-Instruct.Q5_K_M")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := l.Close(); err != nil {
			t.Error(err)
		}
	})
	got, err := l.Prompt(ctx, "reply with \"ok\"")
	if err != nil {
		t.Fatal(err)
	}
	want := "ok"
	if got != want {
		t.Fatalf("expected %s, got %s", want, got)
	}
}
