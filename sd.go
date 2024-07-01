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
	"strconv"
	"time"
)

type stableDiffusion struct {
	c    *exec.Cmd
	port int
}

func newStableDiffusion(ctx context.Context, cache string) (*stableDiffusion, error) {
	log, err := os.OpenFile(filepath.Join(cache, "sd.log"), os.O_WRONLY|os.O_APPEND|os.O_CREATE, 0o644)
	if err != nil {
		return nil, err
	}
	defer log.Close()
	s := &stableDiffusion{port: 8032}
	s.c = exec.CommandContext(ctx, filepath.Join("py", "main.sh"), "--port", strconv.Itoa(s.port))
	s.c.Stdout = log
	s.c.Stderr = log
	if err := s.c.Start(); err != nil {
		return nil, err
	}
	logger.Info("sd: started")
	return s, nil
}

func (s *stableDiffusion) Close() error {
	logger.Info("sd: Terminating")
	s.c.Cancel()
	s.c.Wait()
	return nil
}

func (s *stableDiffusion) genImage(prompt string) ([]byte, error) {
	data := struct {
		Message string `json:"message"`
	}{Message: prompt}
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
