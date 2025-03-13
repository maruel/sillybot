// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package py manages the python backends.
//
// It is an internal package not meant to be used externally. Breaking changes
// will be done without regards to semver.
package py

import (
	"bytes"
	"context"
	_ "embed"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
)

type Server struct {
	cmd  *exec.Cmd
	done <-chan error
}

// NewServer creates a new python virtualenv if needed and starts the server in it.
func NewServer(ctx context.Context, script, cacheDir, logName string, extraArgs []string) (*Server, error) {
	if script == "" || strings.ContainsRune(script, filepath.Separator) {
		return nil, errors.New("script must be a file name, not a path")
	}
	if !filepath.IsAbs(cacheDir) {
		return nil, errors.New("cacheDir must be an absolute path")
	}
	if !filepath.IsAbs(logName) {
		return nil, errors.New("logName must be an absolute path")
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
	log, err := os.OpenFile(logName, os.O_WRONLY|os.O_APPEND|os.O_CREATE, 0o644)
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
		err2 := cmd.Wait()
		var er *exec.ExitError
		if errors.As(err2, &er) {
			s, ok := er.Sys().(syscall.WaitStatus)
			if ok && s.Signaled() {
				// It was simply killed.
				err2 = nil
			}
			if runtime.GOOS == "windows" {
				// We need to figure out how to differentiate between normal quitting and
				// an error.
				err2 = nil
			}
		}
		slog.Info("py", "state", "terminated", "err", err2)
		done <- err2
		close(done)
	}()
	slog.Info("py", "state", "started", "pid", cmd.Process.Pid)
	return &Server{done: done, cmd: cmd}, nil
}

func (s *Server) Close() error {
	_ = s.cmd.Cancel()
	return <-s.done
}

func (s *Server) Done() <-chan error {
	return s.done
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
