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
	"log"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"runtime/debug"
	"strings"
	"syscall"
	"time"

	"github.com/lmittmann/tint"
	"github.com/maruel/sillybot"
	"github.com/maruel/sillybot/llm"
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
		TimeFormat: time.TimeOnly,
		NoColor:    !isatty.IsTerminal(os.Stderr.Fd()),
	}))
	slog.SetDefault(logger)
	go func() {
		<-ctx.Done()
		slog.Info("main", "message", "quitting")
	}()

	wd, err := os.Getwd()
	if err != nil {
		return err
	}

	cfg := sillybot.Config{}
	bottoken := flag.String("bot-token", "", "Bot Token; get one at https://discord.com/developers/applications")
	gcptoken := flag.String("gcp-token", "", "Google Cloud Token to enable web search; get one at https://cloud.google.com/docs/authentication/api-keys")
	cxtoken := flag.String("cx-token", "", "Cx Token to enable web search")
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
				l := 0
				for _, k := range cfg.KnownLLMs {
					if m := len(k.Basename) + 1; m > l {
						l = m
					}
				}
				for _, k := range cfg.KnownLLMs {
					fmt.Fprintf(o, "  %-*s: %s\n", l, k.Basename+"*", k.URL())
				}
			}
		}
	}
	flag.Parse()

	if len(flag.Args()) != 0 {
		return errors.New("unexpected argument")
	}
	if *version {
		fmt.Printf("discord-bot %s\n", commit())
		return nil
	}
	if *verbose {
		programLevel.Set(slog.LevelDebug)
	}
	if err = cfg.LoadOrDefault(*config); err != nil {
		return err
	}
	if *bottoken == "" {
		b, err2 := os.ReadFile("token_discord.txt")
		if err2 != nil || len(b) < 10 {
			return errors.New("-bot-token or a 'token_discord.txt' is required")
		}
		*bottoken = strings.TrimSpace(string(b))
	}
	if *gcptoken == "" {
		if b, err2 := os.ReadFile("token_gcp.txt"); err2 == nil {
			*gcptoken = strings.TrimSpace(string(b))
		}
	}
	if *cxtoken == "" {
		if b, err2 := os.ReadFile("token_cx.txt"); err2 == nil {
			*cxtoken = strings.TrimSpace(string(b))
		}
	}
	if err = os.MkdirAll(*cache, 0o755); err != nil {
		log.Fatal(err)
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
	memcache := filepath.Join(*cache, "discord_memory.json")
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

	d, err := newDiscordBot(ctx, *bottoken, *gcptoken, *cxtoken, *verbose, l, mem, cfg.KnownLLMs, ig, cfg.Bot.Settings)
	if err != nil {
		return err
	}
	<-ctx.Done()
	err = d.Close()
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
		fmt.Fprintf(os.Stderr, "\ndiscord-bot: %v\n", err.Error())
		os.Exit(1)
	}
}
