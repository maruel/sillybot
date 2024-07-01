// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package main

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/schollz/progressbar/v3"
)

type llm struct {
	c            *exec.Cmd
	done         chan error
	port         int
	systemPrompt string
}

func newLLM(ctx context.Context, cache, model string) (*llm, error) {
	execSuffix := ""
	if runtime.GOOS == "windows" {
		execSuffix = ".exe"
	}
	llamafile := filepath.Join(cache, "llamafile"+execSuffix)
	if _, err := os.Stat(llamafile); err != nil {
		logger.Info("llm", "llamafile", "", "state", "missing")
		// Download llamafile from GitHub. We always want the latest and greatest
		// as it is very actively developed and the model we download likely use an
		// older version.
		url, name, err := getGitHubLatestRelease("Mozilla-Ocho", "llamafile", "application/octet-stream")
		if err != nil {
			return nil, err
		}
		logger.Info("llm", "llamafile_release", name)
		versioned := filepath.Join(cache, name+execSuffix)
		if err = downloadExec(url, versioned); err != nil {
			return nil, err
		}
		// Copy it as the default executable to use.
		if err = copyFile(llamafile, versioned); err != nil {
			return nil, err
		}
	}

	modelFile := filepath.Join(cache, model+".gguf")
	if _, err := os.Stat(modelFile); err != nil {
		logger.Info("llm", "model", model, "state", "missing")
		// TODO: Hack.
		repo := ""
		if strings.HasPrefix(model, "Meta-Llama-3-8B-Instruct.") {
			repo = "Mozilla/Meta-Llama-3-8B-Instruct-llamafile"
		} else if strings.HasPrefix(model, "gemma-2-27b-it.") {
			repo = "jartine/gemma-2-27b-it-llamafile"
		} else {
			return nil, errors.New("can't guess model's huggingface repo")
		}
		if err = getHfModelGGUFFromLlamafile(cache, repo, model); err != nil {
			return nil, err
		}
	}

	// Create the log file to redirect llamafile's output which is quite verbose.
	log, err := os.OpenFile(filepath.Join(cache, "llm.log"), os.O_WRONLY|os.O_APPEND|os.O_CREATE, 0o644)
	if err != nil {
		return nil, err
	}
	defer log.Close()
	l := &llm{
		done:         make(chan error),
		port:         findFreePort(),
		systemPrompt: "You are a terse assistant. You reply with short answers. You are often joyful, sometimes humorous, sometimes sarcastic.",
	}
	cmd := []string{llamafile, "--model", modelFile, "-ngl", "9999", "--nobrowser", "--port", strconv.Itoa(l.port)}
	single := strings.Join(cmd, " ")
	logger.Info("llm", "command", single, "cwd", cache)
	if runtime.GOOS == "windows" {
		l.c = exec.CommandContext(ctx, cmd[0], cmd[1:]...)
	} else {
		l.c = exec.CommandContext(ctx, "/bin/sh", "-c", single)
	}
	l.c.Dir = cache
	l.c.Stdout = log
	l.c.Stderr = log
	if err = l.c.Start(); err != nil {
		return nil, err
	}
	go func() {
		l.done <- l.c.Wait()
		logger.Info("llm", "state", "terminated")
	}()
	logger.Info("llm", "state", "started", "pid", l.c.Process.Pid, "port", l.port)
	for {
		if _, err = l.prompt("reply with \"ok\""); err == nil {
			break
		}
		select {
		case err := <-l.done:
			return nil, fmt.Errorf("failed to start: %w", err)
		default:
		}
	}
	logger.Info("llm", "state", "ready")
	return l, nil
}

func (l *llm) Close() error {
	logger.Info("llm", "state", "terminating")
	l.c.Cancel()
	return <-l.done
}

func (l *llm) prompt(prompt string) (string, error) {
	data := openAIChatCompletionRequest{
		Model: "llama-3",
		Messages: []message{
			{system, l.systemPrompt},
			{user, prompt},
		},
	}
	b, _ := json.Marshal(data)
	url := fmt.Sprintf("http://localhost:%d/v1/chat/completions", l.port)
	start := time.Now()
	resp, err := http.Post(url, "application/json", bytes.NewReader(b))
	if err != nil {
		return "", err
	}
	d := json.NewDecoder(resp.Body)
	d.DisallowUnknownFields()
	r := openAIChatCompletionsResponse{}
	err = d.Decode(&r)
	_ = resp.Body.Close()
	if err != nil {
		return "", err
	}
	if len(r.Choices) != 1 {
		return "", errors.New("unexpected number of choices")
	}
	// Llama-3
	reply := strings.TrimSuffix(r.Choices[0].Message.Content, "<|eot_id|>")
	// Gemma-2
	reply = strings.TrimSuffix(reply, "<end_of_turn>")
	reply = strings.TrimSpace(reply)
	logger.Info("llm", "prompt", prompt, "reply", reply, "duration", time.Since(start).Round(time.Millisecond))
	return reply, nil
}

// Messages. https://platform.openai.com/docs/api-reference/making-requests

// openAIChatCompletionRequest is documented at
// https://platform.openai.com/docs/api-reference/chat/create
type openAIChatCompletionRequest struct {
	Model    string    `json:"model"`
	Stream   bool      `json:"stream"`
	Messages []message `json:"messages"`
}

type role string

const (
	system    role = "system"
	user      role = "user"
	assistant role = "assistant"
)

type message struct {
	Role    role   `json:"role"`
	Content string `json:"content"`
}

// openAIChatCompletionsResponse is documented at
// https://platform.openai.com/docs/api-reference/chat/object
type openAIChatCompletionsResponse struct {
	Choices []openAIChoices `json:"choices"`
	Created int64           `json:"created"`
	ID      string          `json:"id"`
	Model   string          `json:"model"`
	Object  string          `json:"object"`
	Usage   struct {
		CompletionTokens int64 `json:"completion_tokens"`
		PromptTokens     int64 `json:"prompt_tokens"`
		TotalTokens      int64 `json:"total_tokens"`
	} `json:"usage"`
}

type openAIChoices struct {
	// FinishReason is one of stop, legnth, content_filter or tool_calls.
	FinishReason string  `json:"finish_reason"`
	Index        int     `json:"index"`
	Message      message `json:"message"`
}

// Tools

// getGitHubLatestRelease returns the latest release for a github repository.
func getGitHubLatestRelease(owner, repo, contentType string) (string, string, error) {
	resp, err := http.Get("https://api.github.com/repos/" + owner + "/" + repo + "/releases")
	if err != nil {
		return "", "", err
	}
	// Just enough of the GitHub API response to be able to parse it.
	data := []struct {
		Assets []struct {
			BrowserDownloadURL string `json:"browser_download_url"`
			ContentType        string `json:"content_type"`
			Name               string `json:"name"`
		} `json:"assets"`
		TagName    string `json:"tag_name"`
		Prerelease bool   `json:"prerelease"`
	}{}
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return "", "", err
	}
	resp.Body.Close()
	for _, l := range data {
		if l.Prerelease {
			continue
		}
		for _, asset := range l.Assets {
			if asset.ContentType == contentType {
				return asset.BrowserDownloadURL, asset.Name, nil
			}
		}
	}
	return "", "", nil
}

// downloadExec downloads an executable.
func downloadExec(url, dst string) error {
	if _, err := os.Stat(dst); err == nil || !os.IsNotExist(err) {
		return err
	}
	logger.Info("llm", "downloading", url)
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	f, err := os.OpenFile(dst, os.O_CREATE|os.O_WRONLY, 0o755)
	if err != nil {
		return err
	}
	defer f.Close()
	bar := progressbar.DefaultBytes(resp.ContentLength, "downloading")
	_, err = io.Copy(io.MultiWriter(f, bar), resp.Body)
	return err
}

// copyFile copy a file while keeping the file mode.
func copyFile(dst, src string) error {
	s, err := os.Open(src)
	if err != nil {
		return err
	}
	defer s.Close()
	st, err := s.Stat()
	if err != nil {
		return err
	}
	d, err := os.OpenFile(dst, os.O_CREATE|os.O_WRONLY, st.Mode())
	if err != nil {
		return err
	}
	defer d.Close()
	_, err = io.Copy(d, s)
	return err
}

// getHfModelGGUFFromLlamafile retrieves a file from an HuggingFace repository.
func getHfModelGGUFFromLlamafile(cache, repo, model string) error {
	url := "https://huggingface.co/" + repo + "/resolve/main/" + model + ".llamafile?download=true"
	dst := filepath.Join(cache, model+".llamafile")
	if err := downloadExec(url, dst); err != nil {
		return err
	}
	gguf := model + ".gguf"
	dstgguf := filepath.Join(cache, gguf)
	if _, err := os.Stat(dstgguf); err == nil || !os.IsNotExist(err) {
		return err
	}
	z, err := zip.OpenReader(dst)
	if err != nil {
		return err
	}
	defer z.Close()
	for _, i := range z.File {
		if i.Name == gguf {
			s, err := i.Open()
			if err != nil {
				return err
			}
			defer s.Close()
			d, err := os.OpenFile(dstgguf, os.O_CREATE|os.O_WRONLY, 0o644)
			if err != nil {
				return err
			}
			defer d.Close()
			_, err = io.Copy(d, s)
			return err
		}
	}
	return errors.New("gguf not found")
}
