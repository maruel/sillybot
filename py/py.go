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
	"log/slog"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"time"
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

// General functions I didn't know where to put.

// FindFreePort returns an available TCP port to listen to.
func FindFreePort() int {
	l, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		panic(err)
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port
}

// IsHostPort returns true if the string seems like a valid "host:port" string.
func IsHostPort(s string) bool {
	// Simplified regexp that supports IPv4, IPv6 and hostname and requires a port.
	ipv4 := `\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}`
	ipv6 := `\[[a-fA-F0-9:]+\]`
	hostname := `[a-zA-Z0-9\-\.]{2,}`
	r := `^(?:` + ipv4 + `|` + ipv6 + `|` + hostname + `):\d{1,5}$`
	ok, err := regexp.MatchString(r, s)
	if err != nil {
		panic(err)
	}
	return ok
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
