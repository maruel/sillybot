// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package internal contains various random shared code.
package internal

import (
	"context"
	"log/slog"
	"net"
	"os"
	"regexp"
	"runtime/debug"
	"strconv"
	"time"

	"github.com/lmittmann/tint"
	"github.com/mattn/go-colorable"
	"github.com/mattn/go-isatty"
)

// General functions I didn't know where to put.

// FindFreePort returns an available TCP port to listen to, first trying
// preferred.
func FindFreePort(preferred ...int) int {
	for _, p := range preferred {
		l, err := net.Listen("tcp", "localhost:"+strconv.Itoa(p))
		if err != nil {
			continue
		}
		defer l.Close()
		return l.Addr().(*net.TCPAddr).Port
	}
	l, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		panic(err)
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port
}

// IsHostPort returns true if the string seems like a valid "host:port" string.
func IsHostPort(s string) bool {
	// Simplified regexp that supports IPv4, IPv6 and hostname and requires a port.
	ipv4 := `\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}`
	ipv6 := `\[[a-fA-F0-9:]+\]`
	hostname := `[a-zA-Z0-9\-\.]{2,}`
	r := `^(?:` + ipv4 + `|` + ipv6 + `|` + hostname + `):\d{1,5}$`
	ok, err := regexp.MatchString(r, s)
	if err != nil {
		panic(err)
	}
	return ok
}

func InitLog(programLevel *slog.LevelVar) {
	logger := slog.New(tint.NewHandler(colorable.NewColorable(os.Stderr), &tint.Options{
		Level:      programLevel,
		TimeFormat: "15:04:05.000", // Like time.TimeOnly plus milliseconds.
		NoColor:    !isatty.IsTerminal(os.Stderr.Fd()),
		ReplaceAttr: func(groups []string, a slog.Attr) slog.Attr {
			switch t := a.Value.Any().(type) {
			case string:
				if t == "" {
					return slog.Attr{}
				}
			case bool:
				if !t {
					return slog.Attr{}
				}
			case uint64:
				if t == 0 {
					return slog.Attr{}
				}
			case int64:
				if t == 0 {
					return slog.Attr{}
				}
			case float64:
				if t == 0 {
					return slog.Attr{}
				}
			case time.Time:
				if t.IsZero() {
					return slog.Attr{}
				}
			case time.Duration:
				if t == 0 {
					return slog.Attr{}
				}
			}
			return a
		},
	}))
	slog.SetDefault(logger)
}

func Commit() string {
	rev := ""
	suffix := ""
	if info, ok := debug.ReadBuildInfo(); ok {
		for _, s := range info.Settings {
			if s.Key == "vcs.revision" {
				rev = s.Value
			} else if s.Key == "vcs.modified" && s.Value == "true" {
				suffix = "-tainted"
			}
		}
	}
	return rev + suffix
}

// Logger retrieves a slog.Logger from the context if any, otherwise returns slog.Default().
func Logger(ctx context.Context) *slog.Logger {
	v := ctx.Value(contextKey{})
	switch v := v.(type) {
	case *slog.Logger:
		return v
	default:
		return slog.Default()
	}
}

// WithLogger injects a slog.Logger into the context. It can be retrieved with Logger().
func WithLogger(ctx context.Context, logger *slog.Logger) context.Context {
	return context.WithValue(ctx, contextKey{}, logger)
}

//

type contextKey struct{}
