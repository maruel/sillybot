// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package sillybot

import (
	"context"
	"image"
	"path/filepath"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/maruel/sillybot/py"
)

func TestImageGen(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test case in short mode")
	}
	cache, err := filepath.Abs("cache")
	if err != nil {
		t.Fatal(err)
	}
	opts := ImageGenOptions{Model: "python"}
	ctx := context.Background()
	s, err := NewImageGen(ctx, cache, &opts)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err2 := s.Close(); err2 != nil {
			t.Error(err2)
		}
	})
	img, err := s.GenImage(ctx, "cat", 1)
	if err != nil {
		t.Fatal(err)
	}
	got := img.Bounds()
	want := image.Rect(0, 0, 1024, 1024)
	if diff := cmp.Diff(want, got); diff != "" {
		t.Fatal(diff)
	}
}

func TestImageGen_Remote_Fail(t *testing.T) {
	cache, err := filepath.Abs("cache")
	if err != nil {
		t.Fatal(err)
	}
	opts := ImageGenOptions{Remote: "host"}
	if _, err = NewImageGen(context.Background(), cache, &opts); err == nil {
		t.Fatal("expected error")
	}
}

func TestIsHostPort(t *testing.T) {
	if py.IsHostPort("a:1") {
		t.Fatal()
	}
	if !py.IsHostPort("aa.bb.ts.net:1") {
		t.Fatal()
	}
}
