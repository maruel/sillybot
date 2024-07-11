// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package sillybot implements the common code used by both discord-bot and
// slack-bot.
package sillybot

import (
	"bytes"
	"context"
	_ "embed"
	"fmt"
	"log/slog"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"time"

	"golang.org/x/sync/errgroup"
	"gopkg.in/yaml.v3"
)

// Default configuration with well known models and sane presets.
//
//go:embed default_config.yml
var DefaultConfig []byte

// Config defines the configuration format.
type Config struct {
	Bot struct {
		LLM      LLMOptions
		ImageGen ImageGenOptions `yaml:"image_gen"`
	}
	KnownLLMs []KnownLLM
}

// LoadOrDefault loads a config or write the default to disk.
func (c *Config) LoadOrDefault(config string) error {
	b, err := os.ReadFile(config)
	if os.IsNotExist(err) {
		if err = os.WriteFile(config, DefaultConfig, 0o644); err != nil {
			return fmt.Errorf("failed to write default config: %w", err)
		}
		b = DefaultConfig
	}
	d := yaml.NewDecoder(bytes.NewReader(b))
	d.KnownFields(true)
	if err = d.Decode(c); err != nil {
		return fmt.Errorf("failed to read %q: %w", config, err)
	}
	return nil
}

// LoadModels loads the LLM and ImageGen models.
//
// Both take a while to start, so load them in parallel for faster initialization.
func LoadModels(ctx context.Context, cache string, cfg *Config) (*LLM, *ImageGen, error) {
	start := time.Now()
	slog.Info("models", "state", "initializing")

	eg := errgroup.Group{}
	var l *LLM
	var s *ImageGen
	eg.Go(func() error {
		if cfg.Bot.LLM.Remote == "" && cfg.Bot.LLM.Model == "" {
			slog.Info("models", "message", "no llm requested")
			return nil
		}
		var err error
		if l, err = NewLLM(ctx, cache, &cfg.Bot.LLM, cfg.KnownLLMs); err != nil {
			slog.Info("llm", "state", "failed", "err", err, "duration", time.Since(start).Round(time.Millisecond), "message", "Try running 'tail -f cache/llm.log'")
		}
		return err
	})
	eg.Go(func() error {
		if cfg.Bot.ImageGen.Remote == "" && cfg.Bot.ImageGen.Model == "" {
			slog.Info("models", "message", "no image_gen requested")
			return nil
		}
		var err error
		if s, err = NewImageGen(ctx, cache, &cfg.Bot.ImageGen); err != nil {
			slog.Info("ig", "state", "failed", "err", err, "duration", time.Since(start).Round(time.Millisecond), "message", "Try running 'tail -f cache/image_gen.log'")
		}
		return err
	})
	var err error
	if err = eg.Wait(); err == nil {
		err = ctx.Err()
	}
	slog.Info("models", "state", "ready", "error", err, "duration", time.Since(start).Round(time.Millisecond))
	return l, s, err
}

// General utility functions

func findFreePort() int {
	l, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		panic(err)
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port
}

// runPython runs a python subprocess inside a virtualenv.
func runPython(ctx context.Context, venv string, cmd []string, cwd, log string) (<-chan error, func() error, error) {
	l, err := os.OpenFile(log, os.O_WRONLY|os.O_APPEND|os.O_CREATE, 0o644)
	if err != nil {
		return nil, nil, err
	}
	defer l.Close()
	bin := "bin"
	pythonexe := "python3"
	if runtime.GOOS == "windows" {
		bin = "Scripts"
		// They don't put a python3.exe in the virtualenv on Windows... Seriously.
		pythonexe = "python.exe"
	}
	rel := filepath.Join(venv, bin, pythonexe)
	python3, err := filepath.Abs(rel)
	if err != nil {
		slog.Error("exec", "message", "failed to abspath", "pythonexe", rel, "error", err)
		return nil, nil, err
	}
	cmd0, err := filepath.Abs(cmd[0])
	if err != nil {
		slog.Error("exec", "message", "failed to abspath", "cmd[0]", cmd[0], "error", err)
		return nil, nil, err
	}
	cmd[0] = cmd0
	start := time.Now()
	c := exec.CommandContext(ctx, python3, cmd...)
	c.Dir = cwd
	c.Stdout = l
	c.Stderr = l
	doneErr := make(chan error, 1)
	isDone := make(chan struct{}, 1)
	c.Cancel = func() error {
		if runtime.GOOS != "windows" {
			slog.Info("exec", "state", "terminating", "pid", c.Process.Pid, "duration", time.Since(start).Round(time.Millisecond))
			grace := 30 * time.Second
			err := c.Process.Signal(os.Interrupt)
			if err != nil {
				return err
			}
			select {
			case <-isDone:
				return err
			case <-time.After(grace):
				break
			}
		}
		slog.Info("exec", "state", "killing", "pid", c.Process.Pid, "duration", time.Since(start).Round(time.Millisecond))
		return c.Process.Kill()
	}
	if err := c.Start(); err != nil {
		slog.Error("exec", "message", "failed to start", "cmd", cmd, "cwd", cwd, "error", err)
		return nil, nil, err
	}
	slog.Info("exec", "state", "started", "cmd", cmd, "cwd", cwd, "pid", "log", log, c.Process.Pid, "duration", time.Since(start).Round(time.Millisecond))
	go func() {
		doneErr <- c.Wait()
		isDone <- struct{}{}
		slog.Info("exec", "state", "terminated", "pid", c.Process.Pid, "duration", time.Since(start).Round(time.Millisecond))
	}()
	return doneErr, c.Cancel, nil
}

//

var (
	//go:embed py/image_gen.py
	imageGenPy []byte
	//go:embed py/llm.py
	llmPy []byte
	//go:embed py/setup.bat
	setupBat []byte
	//go:embed py/setup.sh
	setupSh []byte
)

func pyNeedRecreate(cache string) bool {
	if _, err := os.Stat(filepath.Join(cache, "venv", "pyvenv.cfg")); err != nil {
		return true
	}
	if b, err := os.ReadFile(filepath.Join(cache, "image_gen.py")); err != nil || !bytes.Equal(b, imageGenPy) {
		return true
	}
	if b, err := os.ReadFile(filepath.Join(cache, "llm.py")); err != nil || !bytes.Equal(b, llmPy) {
		return true
	}
	name := "setup.sh"
	content := setupSh
	if runtime.GOOS == "windows" {
		name = "setup.bat"
		content = setupBat
	}
	if b, err := os.ReadFile(filepath.Join(cache, name)); err != nil || !bytes.Equal(b, content) {
		return true
	}
	return false
}

func pyRecreate(ctx context.Context, cache string) error {
	if err := os.WriteFile(filepath.Join(cache, "image_gen.py"), imageGenPy, 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(cache, "llm.py"), llmPy, 0o755); err != nil {
		return err
	}
	name := "setup.sh"
	content := setupSh
	if runtime.GOOS == "windows" {
		name = "setup.bat"
		content = setupBat
	}
	if err := os.WriteFile(filepath.Join(cache, name), content, 0o755); err != nil {
		return err
	}
	c := exec.CommandContext(ctx, filepath.Join(cache, name))
	c.Stdout = os.Stdout
	c.Stderr = os.Stderr
	return c.Run()
}
