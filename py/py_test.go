// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package py

import (
	"context"
	"errors"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/maruel/genai/genaiapi"
	"github.com/maruel/httpjson"
	"github.com/maruel/sillybot/internal"
)

func TestNewServer(t *testing.T) {
	if os.Getenv("CI") == "true" {
		// Sadly github is too slow.
		t.Skip("skipping test in CI environment")
	}
	t.Parallel()
	ctx := context.Background()
	// To run from scratch every time, but it's a bit slow:
	// cache := t.TempDir()
	cache, err := filepath.Abs("../cache/py")
	if err != nil {
		t.Fatal(err)
	}
	if err = os.MkdirAll(cache, 0o755); err != nil {
		t.Fatal(err)
	}

	port := strconv.Itoa(internal.FindFreePort(10000))
	// This is a very slow test.
	srv, err := NewServer(ctx, "llm.py", cache, filepath.Join(cache, "py_llm.py"), []string{"--port", port})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err2 := srv.Close(); err2 != nil {
			t.Error(err2)
		}
	})

	client := Client{URL: "http://localhost:" + port}
	start := time.Now()
	for {
		resp, err := client.Completion(ctx, genaiapi.Messages{
			genaiapi.NewTextMessage(genaiapi.User, "Say hello. Reply with only one word."),
		}, nil)
		var v *url.Error
		var h *httpjson.Error
		if errors.As(err, &v) || errors.As(err, &h) {
			// It can be slow when run from scratch, especially on github.
			if time.Since(start) > 5*time.Minute {
				break
			}
			select {
			case <-srv.Done():
				t.Fatal("server died")
			case <-time.After(10 * time.Millisecond):
			}
			continue
		}
		if err != nil {
			t.Fatal(err)
		}
		txt := strings.TrimSpace(resp.Contents[0].Text)
		txt = strings.TrimRight(txt, ".!")
		txt = strings.ToLower(txt)
		if txt != "hello" {
			t.Errorf("got %q, want %q", txt, "hello")
		}
		return
	}
	t.Fatal("too many retries")
}
