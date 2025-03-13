// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package py manages the python backends.
//
// It is an internal package not meant to be used externally. Breaking changes
// will be done without regards to semver.
package py

import (
	"bufio"
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/maruel/genai/genaiapi"
	"github.com/maruel/httpjson"
)

// RecreateVirtualEnvIfNeeded recreates the virtualenv if needed.
func RecreateVirtualEnvIfNeeded(ctx context.Context, cache string) error {
	if needRecreate(cache) {
		return recreate(ctx, cache)
	}
	return nil
}

// Run runs a python subprocess inside a virtualenv.
func Run(ctx context.Context, venv string, cmd []string, cwd, log string) (<-chan error, func() error, error) {
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

type Server struct {
	Done <-chan error
	Cmd  *exec.Cmd
}

// NewServer creates a new python virtualenv if needed and starts the server in it.
func NewServer(ctx context.Context, script, cacheDir, logDir string, extraArgs []string) (*Server, error) {
	if script == "" || strings.ContainsRune(script, filepath.Separator) {
		return nil, errors.New("script must be a file name, not a path")
	}
	if !filepath.IsAbs(cacheDir) {
		return nil, errors.New("cacheDir must be an absolute path")
	}
	if !filepath.IsAbs(logDir) {
		return nil, errors.New("logDir must be an absolute path")
	}
	if needRecreate(cacheDir) {
		if err := recreate(ctx, cacheDir); err != nil {
			return nil, err
		}
	}

	venv := filepath.Join(cacheDir, "venv")
	bin := "bin"
	pythonexe := "python3"
	if runtime.GOOS == "windows" {
		bin = "Scripts"
		// They don't put a python3.exe in the virtualenv on Windows... Seriously.
		pythonexe = "python.exe"
	}
	rel := filepath.Join(venv, bin, pythonexe)
	args := append([]string{script}, extraArgs...)
	log, err := os.OpenFile(filepath.Join(logDir, "py.log"), os.O_WRONLY|os.O_APPEND|os.O_CREATE, 0o644)
	if err != nil {
		return nil, fmt.Errorf("failed to create log file: %w", err)
	}
	defer log.Close()
	cmd := exec.CommandContext(ctx, rel, args...)
	cmd.Dir = cacheDir
	cmd.Stdout = log
	cmd.Stderr = log
	cmd.Cancel = func() error {
		return cmd.Process.Kill()
	}
	if err = cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to start py: %w", err)
	}
	done := make(chan error)
	go func() {
		done <- cmd.Wait()
		// slog.Info("llm", "state", "terminated")
	}()
	// slog.Info("llm", "state", "started", "pid", l.c.Process.Pid, "port", port)
	return &Server{Done: done, Cmd: cmd}, nil
}

func (s *Server) Close() error {
	select {
	case <-s.Done:
		return nil
	default:
	}
	_ = s.Cmd.Cancel()
	<-s.Done
	return nil
}

type CompletionProvider struct {
	URL string
}

type message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type completionRequest struct {
	Stream   bool      `json:"stream"`
	Messages []message `json:"messages"`
}

func (c *CompletionProvider) Completion(ctx context.Context, msgs []genaiapi.Message, opts any) (genaiapi.Message, error) {
	in := completionRequest{}
	for _, m := range msgs {
		in.Messages = append(in.Messages, message{Role: string(m.Role), Content: m.Text})
	}
	var out struct {
		Choices []struct {
			FinishReason string `json:"finish_reason"`
			Message      struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	msg := genaiapi.Message{}
	if err := httpjson.DefaultClient.Post(ctx, c.URL+"/v1/chat/completions", nil, &in, &out); err != nil {
		return msg, err
	}
	msg.Role = genaiapi.Role(out.Choices[0].Message.Role)
	msg.Type = genaiapi.Text
	msg.Text = out.Choices[0].Message.Content
	return msg, nil
}

func (c *CompletionProvider) CompletionStream(ctx context.Context, msgs []genaiapi.Message, opts any, words chan<- string) error {
	in := completionRequest{Stream: true}
	for _, m := range msgs {
		in.Messages = append(in.Messages, message{Role: string(m.Role), Content: m.Text})
	}
	resp, err := httpjson.DefaultClient.PostRequest(ctx, c.URL+"/v1/chat/completions", nil, &in)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	r := bufio.NewReader(resp.Body)
	for {
		line, err := r.ReadBytes('\n')
		line = bytes.TrimSpace(line)
		if err == io.EOF {
			err = nil
			if len(line) == 0 {
				return nil
			}
		}
		if err != nil {
			return fmt.Errorf("failed to get server response: %w", err)
		}
		if len(line) == 0 {
			continue
		}
		const prefix = "data: "
		if !bytes.HasPrefix(line, []byte(prefix)) {
			return fmt.Errorf("unexpected line. expected \"data: \", got %q", line)
		}
		suffix := string(line[len(prefix):])
		if suffix == "[DONE]" {
			return nil
		}
		d := json.NewDecoder(strings.NewReader(suffix))
		d.DisallowUnknownFields()
		d.UseNumber()
		var msg struct {
			Choices []struct {
				FinishReason string `json:"finish_reason"`
				Delta        struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err = d.Decode(&msg); err != nil {
			return fmt.Errorf("failed to decode server response %q: %w", string(line), err)
		}
		if len(msg.Choices) != 1 {
			return fmt.Errorf("server returned an unexpected number of choices, expected 1, got %d", len(msg.Choices))
		}
		words <- msg.Choices[0].Delta.Content
	}
}

//

var (
	//go:embed image_gen.py
	imageGenPy []byte
	//go:embed llm.py
	llmPy []byte
	//go:embed setup.bat
	setupBat []byte
	//go:embed setup.sh
	setupSh []byte
)

func needRecreate(cache string) bool {
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

func recreate(ctx context.Context, cache string) error {
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
