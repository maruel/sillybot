// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package sillybot

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"time"
)

// ImageGen manages an image generation server.
type ImageGen struct {
	c       *exec.Cmd
	done    chan error
	port    int
	steps   int
	loading bool
}

// NewImageGen initializes a new image generation server.
func NewImageGen(ctx context.Context, cache string) (*ImageGen, error) {
	log, err := os.OpenFile(filepath.Join(cache, "imagegen.log"), os.O_WRONLY|os.O_APPEND|os.O_CREATE, 0o644)
	if err != nil {
		return nil, err
	}
	defer log.Close()
	ig := &ImageGen{
		done:    make(chan error),
		port:    findFreePort(),
		steps:   1,
		loading: true,
	}
	bin := "bin"
	pythonexe := "python3"
	if runtime.GOOS == "windows" {
		bin = "Scripts"
		// They don't put a python3.exe in the virtualenv... Seriously.
		pythonexe = "python.exe"
	}
	python3, err := filepath.Abs(filepath.Join("py", "venv", bin, pythonexe))
	if err != nil {
		return nil, err
	}
	main, err := filepath.Abs(filepath.Join("py", "main.py"))
	if err != nil {
		return nil, err
	}
	ig.c = exec.CommandContext(ctx, python3, main, "--port", strconv.Itoa(ig.port))
	ig.c.Dir = cache
	ig.c.Stdout = log
	ig.c.Stderr = log
	ig.c.Cancel = func() error {
		slog.Debug("ig", "state", "killing")
		if runtime.GOOS != "windows" {
			// TODO: Poll for 30s then kill.
			return ig.c.Process.Signal(os.Interrupt)
		}
		return ig.c.Process.Kill()
	}
	slog.Debug("ig", "command", ig.c.Args, "cwd", cache, "log", log.Name())
	if err := ig.c.Start(); err != nil {
		return nil, err
	}
	go func() {
		ig.done <- ig.c.Wait()
		slog.Info("ig", "state", "terminated")
	}()
	slog.Info("ig", "state", "started", "pid", ig.c.Process.Pid, "port", ig.port, "message", "Please be patient, it can take several minutes to download everything")
	for ctx.Err() == nil {
		if _, err = ig.GenImage("cat"); err == nil {
			break
		}
		select {
		case err := <-ig.done:
			return nil, fmt.Errorf("failed to start: %w", err)
		case <-ctx.Done():
		case <-time.After(100 * time.Millisecond):
		}
	}
	ig.steps = 4
	slog.Info("ig", "state", "ready")
	ig.loading = false
	return ig, nil
}

func (ig *ImageGen) Close() error {
	slog.Info("ig", "state", "terminating")
	ig.c.Cancel()
	return <-ig.done
}

// GenImage returns a PNG encoded image based on the prompt.
func (ig *ImageGen) GenImage(prompt string) ([]byte, error) {
	start := time.Now()
	if !ig.loading {
		// Otherwise it storms on startup.
		slog.Info("ig", "prompt", prompt)
	}
	data := struct {
		Message string `json:"message"`
		Steps   int    `json:"steps"`
		Seed    int    `json:"seed"`
	}{Message: prompt, Steps: ig.steps, Seed: 1}
	b, _ := json.Marshal(data)
	url := fmt.Sprintf("http://localhost:%d/", ig.port)
	resp, err := http.Post(url, "application/json", bytes.NewReader(b))
	if err != nil {
		if !ig.loading {
			// Otherwise it storms on startup.
			slog.Error("ig", "prompt", prompt, "error", err, "duration", time.Since(start).Round(time.Millisecond))
		}
		return nil, err
	}
	d := json.NewDecoder(resp.Body)
	d.DisallowUnknownFields()
	r := struct {
		Image []byte `json:"image"`
	}{}
	err = d.Decode(&r)
	_ = resp.Body.Close()
	if err != nil {
		slog.Error("ig", "prompt", prompt, "error", err, "duration", time.Since(start).Round(time.Millisecond))
		return nil, err
	}
	slog.Info("ig", "prompt", prompt, "duration", time.Since(start).Round(time.Millisecond))
	return r.Image, nil
}
