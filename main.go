// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Discord bot to chat with.
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

	"github.com/bwmarrin/discordgo"
	"github.com/lmittmann/tint"
	"github.com/mattn/go-colorable"
	"github.com/mattn/go-isatty"
)

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
	discordgo.Logger = func(msgL, caller int, format string, a ...interface{}) {
		msg := fmt.Sprintf(format, a...)
		switch msgL {
		case discordgo.LogDebug:
			logger.Debug(msg)
		case discordgo.LogInformational:
			logger.Info(msg)
		case discordgo.LogWarning:
			logger.Warn(msg)
		case discordgo.LogError:
			logger.Error(msg)
		}
	}
	wd, err := os.Getwd()
	if err != nil {
		return err
	}
	cache := filepath.Join(wd, "cache")
	if err = os.MkdirAll(cache, 0o755); err != nil {
		log.Fatal(err)
	}

	token := flag.String("t", "", "Bot Token; get one at https://discord.com/developers/applications")
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
	llm := flag.String("llm", defaultModel, "Enable LLM output")
	sd := flag.Bool("sd", false, "Enable Stable Diffusion output")
	flag.Parse()
	if *verbose {
		programLevel.Set(slog.LevelDebug)
	}
	if *token == "" {
		b, err := os.ReadFile("token.txt")
		if err != nil || len(b) < 10 {
			return errors.New("-t or a 'token.txt' is required")
		}
		*token = strings.TrimSpace(string(b))
	}
	dg, err := discordgo.New("Bot " + *token)
	if err != nil {
		return err
	}
	if *verbose {
		// It's very verbose.
		//dg.LogLevel = discordgo.LogDebug
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	bot, err := newBot(ctx, cache, dg, *llm, *sd)
	if err != nil {
		return err
	}
	// Open a websocket connection to Discord and begin listening.
	if err = dg.Open(); err != nil {
		return err
	}
	logger.Info("discord", "state", "running", "info", "Press CTRL-C to exit.")
	c := make(chan os.Signal, 1)
	signal.Notify(c, syscall.SIGINT, syscall.SIGTERM, os.Interrupt)
	<-c
	logger.Info("discord", "state", "terminating")
	err = dg.Close()
	if err2 := bot.Close(); err == nil {
		err = err2
	}
	return err
}

func main() {
	if err := mainImpl(); err != nil {
		fmt.Fprintf(os.Stderr, "discord-bot: %v\n", err.Error())
		os.Exit(1)
	}
}
