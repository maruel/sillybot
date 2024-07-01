// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"time"
)

type stableDiffusion struct {
	c     *exec.Cmd
	done  chan struct{}
	port  int
	steps int
}

func newStableDiffusion(ctx context.Context, cache string) (*stableDiffusion, error) {
	log, err := os.OpenFile(filepath.Join(cache, "sd.log"), os.O_WRONLY|os.O_APPEND|os.O_CREATE, 0o644)
	if err != nil {
		return nil, err
	}
	defer log.Close()
	s := &stableDiffusion{
		done:  make(chan struct{}),
		port:  findFreePort(),
		steps: 1,
	}
	s.c = exec.CommandContext(ctx, filepath.Join("py", "main.sh"), "--port", strconv.Itoa(s.port))
	s.c.Stdout = log
	s.c.Stderr = log
	if err := s.c.Start(); err != nil {
		return nil, err
	}
	go func() {
		s.c.Wait()
		logger.Info("sd", "state", "terminated")
		s.done <- struct{}{}
	}()
	logger.Info("sd", "state", "started", "pid", s.c.Process.Pid, "port", s.port)
	for {
		if _, err = s.genImage("cat"); err == nil {
			break
		}
		select {
		case <-s.done:
			return nil, errors.New("failed to start")
		default:
		}
	}
	s.steps = 28
	logger.Info("sd", "state", "ready")
	return s, nil
}

func (s *stableDiffusion) Close() error {
	logger.Info("sd", "state", "terminating")
	s.c.Cancel()
	<-s.done
	return nil
}

func (s *stableDiffusion) genImage(prompt string) ([]byte, error) {
	data := struct {
		Message string `json:"message"`
		Steps   int    `json:"steps"`
	}{Message: prompt, Steps: s.steps}
	b, _ := json.Marshal(data)
	url := fmt.Sprintf("http://localhost:%d/", s.port)
	start := time.Now()
	resp, err := http.Post(url, "application/json", bytes.NewReader(b))
	if err != nil {
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
		return nil, err
	}
	logger.Info("sd", "prompt", prompt, "duration", time.Since(start).Round(time.Millisecond))
	return r.Image, nil
}
