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
	"strings"
	"sync"
	"syscall"

	"github.com/maruel/sillybot"
	"github.com/maruel/sillybot/internal"
	"github.com/maruel/sillybot/llm"
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

	wd, err := os.Getwd()
	if err != nil {
		return err
	}

	cfg := sillybot.Config{}
	bottoken := flag.String("bot-token", "", "Bot Token; get one at https://api.slack.com/apps")
	apptoken := flag.String("app-token", "", "App Token; get one at https://api.slack.com/apps")
	cache := flag.String("cache", filepath.Join(wd, "cache"), "Directory where models, python virtualenv and logs are put in")
	verbose := flag.Bool("v", false, "Enable verbose logging")
	config := flag.String("config", "config.yml", "Configuration file. If not present, it is automatically created.")
	version := flag.Bool("version", false, "Print version then exit")
	flag.Usage = func() {
		o := flag.CommandLine.Output()
		fmt.Fprintf(o, "Usage of %s:\n", os.Args[0])
		flag.PrintDefaults()
		if *config != "" {
			if cfg.LoadOrDefault(*config) == nil {
				fmt.Fprintf(o, "\nAvailable LLM models:\n")
				for _, k := range cfg.KnownLLMs {
					fmt.Fprintf(o, "  %s\n", k.Source)
				}
			}
		}
	}
	flag.Parse()

	if len(flag.Args()) != 0 {
		return errors.New("unexpected argument")
	}
	if *version {
		fmt.Printf("discord-bot %s\n", internal.Commit())
		return nil
	}
	if *verbose {
		programLevel.Set(slog.LevelDebug)
	}
	if err = cfg.LoadOrDefault(*config); err != nil {
		return err
	}
	if *bottoken == "" {
		b, err2 := os.ReadFile("token_slack_bot.txt")
		if err2 != nil || len(b) < 10 {
			return errors.New("-bot-token or a 'token_slack_bot.txt' is required")
		}
		*bottoken = strings.TrimSpace(string(b))
	}
	if *apptoken == "" {
		a, err2 := os.ReadFile("token_slack_app.txt")
		if err2 != nil || len(a) < 10 {
			return errors.New("-app-token or a 'token_slack_app.txt' is required")
		}
		*apptoken = strings.TrimSpace(string(a))
	}

	if err = os.MkdirAll(*cache, 0o755); err != nil {
		return err
	}
	memDir := filepath.Join(*cache, "memory")
	if err = os.MkdirAll(memDir, 0o755); err != nil {
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
	// Load memory.
	mem := &llm.Memory{}
	memcache := filepath.Join(memDir, "slack.json")
	f, err := os.Open(memcache)
	if err == nil {
		err = mem.Load(f)
		_ = f.Close()
		if err != nil {
			slog.Error("main", "message", "failed to load memory", "error", err)
			// Continue anyway.
		}
	} else {
		slog.Info("main", "memory", "no memory to load", "error", err)
	}

	s, err := newSlackBot(*apptoken, *bottoken, *verbose, l, mem, ig, cfg.Bot.Settings)
	if err != nil {
		return err
	}
	err = s.Run(ctx)
	// Save memory.
	f, err2 := os.Create(memcache)
	if err2 != nil {
		return err2
	}
	err2 = mem.Save(f)
	err3 := f.Close()
	if err2 != nil {
		return err2
	}
	if err3 != nil {
		return err3
	}
	return err
}

func main() {
	if err := mainImpl(); err != nil && err != context.Canceled {
		fmt.Fprintf(os.Stderr, "\nslack-bot: %v\n", err.Error())
		os.Exit(1)
	}
}
