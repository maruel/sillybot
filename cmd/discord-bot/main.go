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
	"strings"
	"syscall"
	"time"

	"github.com/lmittmann/tint"
	"github.com/maruel/sillybot"
	"github.com/mattn/go-colorable"
	"github.com/mattn/go-isatty"
)

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

	flag.Usage = func() {
		o := flag.CommandLine.Output()
		fmt.Fprintf(o, "Usage of %s:\n", os.Args[0])
		flag.PrintDefaults()
		fmt.Fprintf(o, "\nAvailable LLM models:\n")
		l := 0
		for _, k := range sillybot.KnownLLMs {
			if m := len(k.BaseName); m > l {
				l = m
			}
		}
		for _, k := range sillybot.KnownLLMs {
			fmt.Fprintf(o, "  %-*s: %s\n", l, k.BaseName, k.URL)
		}
	}

	bottoken := flag.String("bot-token", "", "Bot Token; get one at https://discord.com/developers/applications or https://api.slack.com/apps")
	cache := flag.String("cache", filepath.Join(wd, "cache"), "Directory where models, python virtualenv and logs are put in")
	verbose := flag.Bool("v", false, "Enable verbose logging")
	llmModel := flag.String("llm", sillybot.KnownLLMs[0].BaseName+".Q5_K_M", "Enable LLM output")
	igUse := flag.Bool("ig", false, "Enable Image Generation output")
	flag.Parse()

	if len(flag.Args()) != 0 {
		return errors.New("unexpected argument")
	}
	if *bottoken == "" {
		b, err := os.ReadFile("token_discord.txt")
		if err != nil || len(b) < 10 {
			return errors.New("-bot-token or a 'token_discord.txt' is required")
		}
		*bottoken = strings.TrimSpace(string(b))
	}
	if *verbose {
		programLevel.Set(slog.LevelDebug)
	}

	if err = os.MkdirAll(*cache, 0o755); err != nil {
		log.Fatal(err)
	}
	l, ig, err := sillybot.LoadModels(ctx, *cache, *llmModel, *igUse)
	if l != nil {
		defer l.Close()
	}
	if ig != nil {
		defer ig.Close()
	}
	if err != nil {
		return err
	}

	d, err := newDiscordBot(ctx, *bottoken, *verbose, l, ig)
	if err != nil {
		return err
	}
	<-ctx.Done()
	return d.Close()
}

func main() {
	if err := mainImpl(); err != nil && err != context.Canceled {
		fmt.Fprintf(os.Stderr, "discord-bot: %v\n", err.Error())
		os.Exit(1)
	}
}
