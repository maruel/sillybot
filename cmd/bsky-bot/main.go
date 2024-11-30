// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Silly bot to chat with.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"runtime/debug"
	"runtime/pprof"
	"runtime/trace"
	"strings"
	"syscall"
	"time"

	"github.com/lmittmann/tint"
	"github.com/mattn/go-colorable"
	"github.com/mattn/go-isatty"
)

func commit() string {
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

func mainImpl() error {
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM, os.Interrupt)
	defer stop()

	programLevel := &slog.LevelVar{}
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
	go func() {
		<-ctx.Done()
		slog.Info("main", "message", "quitting")
	}()

	handle := flag.String("u", "", "BlueSky username/email/handle")
	password := flag.String("p", "", "BlueSky password; defaults to the content of token_bsky_bot.txt if present")
	verbose := flag.Bool("v", false, "Enable verbose logging")
	version := flag.Bool("version", false, "Print version then exit")
	cpuprofile := flag.String("cpuprofile", "", "file to save trace to. A frequent name is cpu.pprof; you can analyze it with go tool pprof -http=:6060 cpu.pprof")
	tracefile := flag.String("trace", "", "file to save trace to. A frequent name is trace.out; you can analyze it with go tool trace -http=:6060 trace.out")
	flag.Usage = func() {
		o := flag.CommandLine.Output()
		fmt.Fprintf(o, "Usage of %s:\n", os.Args[0])
		flag.PrintDefaults()
	}
	flag.Parse()

	if len(flag.Args()) != 0 {
		return errors.New("unexpected argument")
	}
	if *version {
		fmt.Printf("bsky-bot %s\n", commit())
		return nil
	}
	if *tracefile != "" {
		f, err := os.Create(*tracefile)
		if err != nil {
			return err
		}
		defer f.Close()
		if err := trace.Start(f); err != nil {
			return err
		}
		defer trace.Stop()
	}
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			return err
		}
		defer f.Close()
		if err := pprof.StartCPUProfile(f); err != nil {
			return err
		}
		defer pprof.StopCPUProfile()
	}
	if *verbose {
		programLevel.Set(slog.LevelDebug)
	}

	if *handle == "" {
		return errors.New("-u is required")
	}
	if *password == "" {
		b, err2 := os.ReadFile("token_bsky_bot.txt")
		if err2 != nil || len(b) < 8 {
			return errors.New("-p or a 'token_bsky_bot.txt' is required")
		}
		*password = strings.TrimSpace(string(b))
	}
	c, err := New(ctx, *handle, *password)
	if err != nil {
		return err
	}
	slog.Info("bsky", "state", "connected")

	/*
		if _, _, err = c.Post(ctx, &Post{Text: "Hello world!"});err != nil {
			return err
		}
	*/

	feed, _, err := c.GetTimeline(ctx, "", 5)
	if err != nil {
		return err
	}
	for _, post := range feed {
		b, _ := json.Marshal(post.Post)
		slog.Info("bsky", "context", post.FeedContext, "post", string(b), "reason", post.Reason, "reply", post.Reply)
	}
	/*
		if err = c.SearchPosts(ctx, c.client.Auth.Did); err != nil {
			return err
		}
	*/
	return c.Listen(ctx, "")
}

func main() {
	if err := mainImpl(); err != nil && err != context.Canceled {
		fmt.Fprintf(os.Stderr, "\nbsky-bot: %v\n", err.Error())
		os.Exit(1)
	}
}
