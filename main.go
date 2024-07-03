// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package sillybot implements the common code used by both discord-bot and
// slack-bot.
package sillybot

import (
	"context"
	"log/slog"
	"net"
	"os"
	"time"

	"github.com/lmittmann/tint"
	"github.com/mattn/go-colorable"
	"github.com/mattn/go-isatty"
	"golang.org/x/sync/errgroup"
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

func findFreePort() int {
	l, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		panic(err)
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port
}

// LoadModels loads the LLMInstruct and ImageGen models.
//
// Both take a while to start, so load them in parallel for faster initialization.
func LoadModels(ctx context.Context, cache string, llm string, sd bool) (*LLMInstruct, *ImageGen, error) {
	start := time.Now()
	logger.Info("models", "state", "initializing")
	eg := errgroup.Group{}
	var l *LLMInstruct
	var s *ImageGen
	eg.Go(func() error {
		var err error
		if llm != "" {
			if l, err = NewLLMInstruct(ctx, cache, llm); err != nil {
				logger.Info("sd", "state", "failed", "err", err, "duration", time.Since(start).Round(time.Millisecond), "message", "Try running 'tail -f cache/llm.log'")
			}
		}
		return err
	})
	eg.Go(func() error {
		var err error
		if sd {
			if s, err = NewImageGen(ctx, cache); err != nil {
				logger.Info("sd", "state", "failed", "err", err, "duration", time.Since(start).Round(time.Millisecond), "message", "Try running 'tail -f cache/sd.log'")
			}
		}
		return err
	})
	err := eg.Wait()
	logger.Info("models", "state", "ready", "error", err, "duration", time.Since(start).Round(time.Millisecond))
	return l, s, err
}

/*
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
	apptoken := flag.String("app-token", "", "App Token; get one at https://api.slack.com/apps; do not use with discord")
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

	if len(flag.Args()) != 1 {
		return errors.New("must pass command discord or slack")
	}
	botKind := flag.Args()[0]
	switch botKind {
	case "discord":
		if *bottoken == "" {
			b, err := os.ReadFile("token_discord.txt")
			if err != nil || len(b) < 10 {
				return errors.New("-bot-token or a 'token_discord.txt' is required")
			}
			*bottoken = strings.TrimSpace(string(b))
		}
		if *apptoken != "" {
			return errors.New("do not use -app-token with discord")
		}
	case "slack":
		if *bottoken == "" {
			b, err := os.ReadFile("token_slack_bot.txt")
			if err != nil || len(b) < 10 {
				return errors.New("-bot-token or a 'token_slack_bot.txt' is required")
			}
			*bottoken = strings.TrimSpace(string(b))
		}
		if *apptoken == "" {
			a, err := os.ReadFile("token_slack_app.txt")
			if err != nil || len(a) < 10 {
				return errors.New("-app-token or a 'token_slack_app.txt' is required")
			}
			*apptoken = strings.TrimSpace(string(a))
		}
	default:
		return errors.New("must pass command discord or slack")
	}
	if *verbose {
		programLevel.Set(slog.LevelDebug)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	l, sd, err := loadModels(ctx, cache, *llmModel, *sdUse)
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

	switch botKind {
	case "discord":
		d, err := newDiscordBot(*bottoken, *verbose, l, sd)
		if err != nil {
			return err
		}
		<-c
		return d.Close()
	case "slack":
		s, err := newSlackBot(*apptoken, *bottoken, *verbose, l, sd)
		if err != nil {
			return err
		}
		ctx2, cancel2 := context.WithCancel(ctx)
		defer cancel2()
		go func() {
			<-c
			cancel2()
		}()
		return s.Run(ctx2)
	default:
		return errors.New("internal error")
	}
}
*/
