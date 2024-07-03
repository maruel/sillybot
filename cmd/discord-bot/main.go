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

// Logging configuration.
var (
	programLevel = &slog.LevelVar{}
	logger       = slog.New(tint.NewHandler(colorable.NewColorable(os.Stderr), &tint.Options{
		Level:      programLevel,
		TimeFormat: time.TimeOnly,
		NoColor:    !isatty.IsTerminal(os.Stderr.Fd()),
	}))
)

func mainImpl() error {
	slog.SetDefault(logger)
	wd, err := os.Getwd()
	if err != nil {
		return err
	}
	cache := filepath.Join(wd, "cache")
	if err = os.MkdirAll(cache, 0o755); err != nil {
		log.Fatal(err)
	}

	bottoken := flag.String("bot-token", "", "Bot Token; get one at https://discord.com/developers/applications or https://api.slack.com/apps")
	verbose := flag.Bool("v", false, "Enable verbose logging")
	// Browse at https://huggingface.co/Mozilla for recent models.
	// https://huggingface.co/Mozilla/Meta-Llama-3-70B-Instruct-llamafile/tree/main
	// is too large for my computers. :(

	// https://huggingface.co/Mozilla/Meta-Llama-3-8B-Instruct-llamafile/tree/main
	defaultModel := "Meta-Llama-3-8B-Instruct.Q5_K_M"
	// defaultModel := "Meta-Llama-3-8B-Instruct.BF16" // Doesn't work on M3 Max.
	// defaultModel := "Meta-Llama-3-8B-Instruct.F16" // 3x slower than Q5_K_M.

	// https://huggingface.co/jartine/gemma-2-27b-it-llamafile/tree/main
	//defaultModel := "gemma-2-27b-it.Q6_K"
	llmModel := flag.String("llm", defaultModel, "Enable LLM output")
	sdUse := flag.Bool("sd", false, "Enable Stable Diffusion output")
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
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	l, sd, err := sillybot.LoadModels(ctx, cache, *llmModel, *sdUse)
	if l != nil {
		defer l.Close()
	}
	if sd != nil {
		defer sd.Close()
	}
	if err != nil {
		return err
	}

	c := make(chan os.Signal, 1)
	signal.Notify(c, syscall.SIGINT, syscall.SIGTERM, os.Interrupt)

	d, err := newDiscordBot(*bottoken, *verbose, l, sd)
	if err != nil {
		return err
	}
	<-c
	return d.Close()
}

func main() {
	if err := mainImpl(); err != nil && err != context.Canceled {
		fmt.Fprintf(os.Stderr, "discord-bot: %v\n", err.Error())
		os.Exit(1)
	}
}
