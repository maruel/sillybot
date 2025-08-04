// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package internaltest is awesome sauce for unit testing.
package internaltest

import (
	"context"
	"flag"
	"log/slog"
	"path/filepath"
	"strings"
	"testing"

	"github.com/maruel/sillybot/internal"
)

// Log returns a slog.Logger that redirects to testing.TB.Log() and adds it to the Context.
func Log(tb testing.TB) (context.Context, *slog.Logger) {
	level := &slog.LevelVar{}
	// Tone down logging by default because it's intense. Need to revisit because tests on CI use -v.
	if false {
		flag.Visit(func(f *flag.Flag) {
			if f.Name == "test.v" {
				level.Set(slog.LevelDebug)
			}
		})
	}
	l := slog.New(slog.NewTextHandler(&testWriter{t: tb}, &slog.HandlerOptions{
		AddSource: true,
		Level:     level,
		ReplaceAttr: func(groups []string, a slog.Attr) slog.Attr {
			switch a.Key {
			case "level":
				a.Key = "l"
				a.Value = slog.StringValue(a.Value.String()[:3])
			case "source":
				a.Key = "s"
				s := a.Value.Any().(*slog.Source)
				s.File = filepath.Base(s.File)
			case "time":
				a = slog.Attr{}
			}
			return a
		},
	}))
	ctx := internal.WithLogger(tb.Context(), l)
	return ctx, l
}

//

// testWriter wraps t.Log() to implement io.Writer
type testWriter struct {
	t testing.TB
}

func (tw *testWriter) Write(p []byte) (n int, err error) {
	// Sadly the log output is attributed to this line.
	tw.t.Log(strings.TrimSpace(string(p)))
	return len(p), nil
}
