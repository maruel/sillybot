// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"time"
)

type stableDiffusion struct {
	c       *exec.Cmd
	done    chan error
	port    int
	steps   int
	loading bool
}

func newStableDiffusion(ctx context.Context, cache string) (*stableDiffusion, error) {
	log, err := os.OpenFile(filepath.Join(cache, "sd.log"), os.O_WRONLY|os.O_APPEND|os.O_CREATE, 0o644)
	if err != nil {
		return nil, err
	}
	defer log.Close()
	s := &stableDiffusion{
		done:    make(chan error),
		port:    findFreePort(),
		steps:   1,
		loading: true,
	}
	python3, err := filepath.Abs(filepath.Join("py", "venv", "bin", "python3"))
	if err != nil {
		return nil, err
	}
	main, err := filepath.Abs(filepath.Join("py", "main.py"))
	if err != nil {
		return nil, err
	}
	s.c = exec.CommandContext(ctx, python3, main, "--port", strconv.Itoa(s.port))
	s.c.Dir = cache
	s.c.Stdout = log
	s.c.Stderr = log
	s.c.Cancel = func() error {
		if runtime.GOOS != "windows" {
			return s.c.Process.Signal(os.Interrupt)
		}
		return s.c.Process.Kill()
	}
	logger.Info("sd", "command", s.c.Args, "cwd", cache, "log", log.Name())
	if err := s.c.Start(); err != nil {
		return nil, err
	}
	go func() {
		s.done <- s.c.Wait()
		logger.Info("sd", "state", "terminated")
	}()
	logger.Info("sd", "state", "started", "pid", s.c.Process.Pid, "port", s.port)
	for {
		if _, err = s.genImage("cat"); err == nil {
			break
		}
		select {
		case err := <-s.done:
			return nil, fmt.Errorf("failed to start: %w", err)
		default:
		}
	}
	s.steps = 4
	logger.Info("sd", "state", "ready")
	s.loading = false
	return s, nil
}

func (s *stableDiffusion) Close() error {
	logger.Info("sd", "state", "terminating")
	s.c.Cancel()
	return <-s.done
}

func (s *stableDiffusion) genImage(prompt string) ([]byte, error) {
	start := time.Now()
	if !s.loading {
		// Otherwise it storms on startup.
		logger.Info("sd", "prompt", prompt)
	}
	data := struct {
		Message string `json:"message"`
		Steps   int    `json:"steps"`
		Seed    int    `json:"seed"`
	}{Message: prompt, Steps: s.steps, Seed: 1}
	b, _ := json.Marshal(data)
	url := fmt.Sprintf("http://localhost:%d/", s.port)
	resp, err := http.Post(url, "application/json", bytes.NewReader(b))
	if err != nil {
		if !s.loading {
			// Otherwise it storms on startup.
			logger.Error("sd", "prompt", prompt, "error", err, "duration", time.Since(start).Round(time.Millisecond))
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
		logger.Error("sd", "prompt", prompt, "error", err, "duration", time.Since(start).Round(time.Millisecond))
		return nil, err
	}
	logger.Info("sd", "prompt", prompt, "duration", time.Since(start).Round(time.Millisecond))
	return r.Image, nil
}
