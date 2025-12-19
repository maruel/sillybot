// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package imagegen

import (
	"context"
	"flag"
	"image"
	"log/slog"
	"os"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/lmittmann/tint"
	"github.com/mattn/go-colorable"
	"github.com/mattn/go-isatty"
)

func TestImageGen(t *testing.T) {
	// This test is really slow in CPU mode on Intel, close to 2 minutes even on
	// a i7.
	if testing.Short() {
		t.Skip("skipping test case in short mode")
	}
	tmpdir := t.TempDir()
	opts := Options{Model: "python"}
	ctx := context.Background()
	s, err := New(ctx, tmpdir, &opts)
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
	// TODO: Make this configurable. want := image.Rect(0, 0, 1024, 1024)
	want := image.Rect(0, 0, 1216, 832)
	if diff := cmp.Diff(want, got); diff != "" {
		t.Fatal(diff)
	}
}

func TestImageGen_Remote_Fail(t *testing.T) {
	tmpdir := t.TempDir()
	opts := Options{Remote: "host"}
	if _, err := New(context.Background(), tmpdir, &opts); err == nil {
		t.Fatal("expected error")
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
