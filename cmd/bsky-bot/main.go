// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Silly bot to chat with.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"runtime/pprof"
	"runtime/trace"
	"strings"
	"sync"
	"syscall"

	"github.com/maruel/sillybot"
	"github.com/maruel/sillybot/internal"
)

func mainImpl() error {
	wg := sync.WaitGroup{}
	defer wg.Wait()
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM, os.Interrupt)
	defer stop()
	programLevel := &slog.LevelVar{}
	internal.InitLog(programLevel)
	wg.Add(1)
	go func() {
		defer wg.Done()
		<-ctx.Done()
		slog.Info("main", "message", "quitting")
	}()

	wd, err2 := os.Getwd()
	if err2 != nil {
		return err2
	}

	cfg := sillybot.Config{}
	handle := flag.String("u", "", "BlueSky username/email/handle")
	password := flag.String("p", "", "BlueSky password; defaults to the content of token_bsky_bot.txt if present")
	cache := flag.String("cache", filepath.Join(wd, "cache"), "Directory where models, python virtualenv and logs are put in")
	config := flag.String("config", "config.yml", "Configuration file. If not present, it is automatically created.")
	version := flag.Bool("version", false, "Print version then exit")
	verbose := flag.Bool("v", false, "Enable verbose logging")
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
		fmt.Printf("bsky-bot %s\n", internal.Commit())
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
	if err := cfg.LoadOrDefault(*config); err != nil {
		return err
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
	if err := os.MkdirAll(*cache, 0o755); err != nil {
		return err
	}
	l, ig, err := sillybot.LoadModels(ctx, *cache, &cfg)
	if l != nil {
		defer l.Close()
	}
	if ig != nil {
		defer ig.Close()
	}
	if err != nil {
		return err
	}

	b, err := newBskyBot(ctx, *handle, *password, l, ig, cfg.Bot.Settings)
	if err != nil {
		return err
	}
	defer b.Close()
	return b.ProcessReplies(ctx)
}

func main() {
	if err := mainImpl(); err != nil && err != context.Canceled {
		fmt.Fprintf(os.Stderr, "\nbsky-bot: %v\n", err.Error())
		os.Exit(1)
	}
}
