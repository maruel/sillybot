// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package sillybot

import (
	"archive/zip"
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
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

// KnownLLM is a known model.
type KnownLLM struct {
	URL      string
	Upstream string
	BaseName string
	// Most native format. Normally BF16 or F16 depending on the model. This is
	// found in config.json in Upstream.
	Native string
}

// KnownLLMs is a list of known models for ease of use. This is in no way
// limits what can be used with this system.
var KnownLLMs = []KnownLLM{
	{
		URL:      "https://huggingface.co/Mozilla/Meta-Llama-3-8B-Instruct-llamafile",
		Upstream: "https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct",
		BaseName: "Meta-Llama-3-8B-Instruct",
		Native:   "BF16",
	},
	{
		URL:      "https://huggingface.co/Mozilla/Phi-3-mini-4k-instruct-llamafile",
		Upstream: "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct",
		BaseName: "Phi-3-mini-4k-instruct",
		Native:   "BF16",
	},
	{
		URL:      "https://huggingface.co/Mozilla/Phi-3-medium-128k-instruct-llamafile",
		Upstream: "https://huggingface.co/microsoft/Phi-3-medium-128k-instruct",
		BaseName: "Phi-3-medium-128k-instruct",
		Native:   "BF16",
	},
	{
		URL:      "https://huggingface.co/jartine/gemma-2-27b-it-llamafile",
		Upstream: "https://huggingface.co/google/gemma-2-27b-it",
		BaseName: "gemma-2-27b-it",
		Native:   "BF16",
	},
}

// LLM runs a llamafile server and runs queries on it.
//
// While it is expected that the model is an Instruct form, it is not a
// requirement.
type LLM struct {
	c            *exec.Cmd
	done         chan error
	port         int
	systemPrompt string
	loading      bool

	_ struct{}
}

// NewLLM instantiates a llamafile server.
func NewLLM(ctx context.Context, cache, model string) (*LLM, error) {
	execSuffix := ""
	if runtime.GOOS == "windows" {
		execSuffix = ".exe"
	}
	llamafile := filepath.Join(cache, "llamafile"+execSuffix)
	if _, err := os.Stat(llamafile); err != nil {
		slog.Info("llm", "llamafile", "", "state", "missing")
		// Download llamafile from GitHub. We always want the latest and greatest
		// as it is very actively developed and the model we download likely use an
		// older version.
		url, name, err := getGitHubLatestRelease("Mozilla-Ocho", "llamafile", "application/octet-stream")
		if err != nil {
			return nil, err
		}
		slog.Info("llm", "llamafile_release", name)
		versioned := filepath.Join(cache, name+execSuffix)
		if err = downloadExec(ctx, url, versioned); err != nil {
			return nil, err
		}
		// Copy it as the default executable to use.
		if err = copyFile(llamafile, versioned); err != nil {
			return nil, err
		}
	}

	switch filepath.Ext(model) {
	case ".BF16":
		if runtime.GOOS == "darwin" {
			slog.Warn("llm", "message", "bfloat16 is likely not supported on your Apple Silicon system")
		}
	case ".F16", ".Q8_0", ".Q6_K", ".Q5_K_S", ".Q5_K_M", ".Q5_1", ".Q5_0", ".Q4_K_S", ".Q4_K_M", ".Q4_1", ".Q4_0", ".Q3_K_S", ".Q3_K_M", ".Q3_K_L", ".Q2_K":
	case "":
		return nil, errors.New("you forgot to add a quantization suffix like '.BF16' or '.Q5_K_M'")
	default:
		return nil, errors.New("unknown quantization, did you forget a suffix like '.BF16' or '.Q5_K_M'?")
	}

	modelFile := filepath.Join(cache, model+".gguf")
	if _, err := os.Stat(modelFile); err != nil {
		slog.Info("llm", "model", model, "state", "missing")
		url := ""
		for _, k := range KnownLLMs {
			if strings.HasPrefix(model, k.BaseName) {
				url = k.URL
				break
			}
		}
		if url == "" {
			return nil, errors.New("can't guess model's huggingface repo")
		}
		hf := "https://huggingface.co/"
		if strings.HasPrefix(url, hf) {
			repo := url[len(hf):]
			if err = getHfModelGGUFFromLlamafile(ctx, cache, repo, model); err != nil {
				return nil, err
			}
		} else {
			return nil, errors.New("can't guess model's source")
		}
	}

	// Create the log file to redirect llamafile's output which is quite verbose.
	log, err := os.OpenFile(filepath.Join(cache, "llm.log"), os.O_WRONLY|os.O_APPEND|os.O_CREATE, 0o644)
	if err != nil {
		return nil, err
	}
	defer log.Close()
	l := &LLM{
		done:         make(chan error),
		port:         findFreePort(),
		systemPrompt: "You are a terse assistant. You reply with short answers. You are often joyful, sometimes humorous, sometimes sarcastic.",
		loading:      true,
	}
	cmd := []string{llamafile, "--model", modelFile, "-ngl", "9999", "--nobrowser", "--port", strconv.Itoa(l.port)}
	single := strings.Join(cmd, " ")
	slog.Debug("llm", "command", single, "cwd", cache, "log", log.Name())
	if runtime.GOOS == "windows" {
		l.c = exec.CommandContext(ctx, cmd[0], cmd[1:]...)
	} else {
		l.c = exec.CommandContext(ctx, "/bin/sh", "-c", single)
	}
	l.c.Dir = cache
	l.c.Stdout = log
	l.c.Stderr = log
	l.c.Cancel = func() error {
		slog.Debug("llm", "state", "killing")
		return l.c.Process.Kill()
	}
	if err = l.c.Start(); err != nil {
		return nil, err
	}
	go func() {
		l.done <- l.c.Wait()
		slog.Info("llm", "state", "terminated")
	}()
	slog.Info("llm", "state", "started", "pid", l.c.Process.Pid, "port", l.port)
	for ctx.Err() == nil {
		if _, err = l.Prompt(ctx, "reply with \"ok\""); err == nil {
			break
		}
		select {
		case err := <-l.done:
			return nil, fmt.Errorf("failed to start: %w", err)
		case <-ctx.Done():
		case <-time.After(100 * time.Millisecond):
		}
	}
	slog.Info("llm", "state", "ready")
	l.loading = false
	return l, nil
}

func (l *LLM) Close() error {
	slog.Info("llm", "state", "terminating")
	l.c.Cancel()
	return <-l.done
}

// Prompt prompts the LLM and returns the reply.
func (l *LLM) Prompt(ctx context.Context, prompt string) (string, error) {
	start := time.Now()
	lvl := slog.LevelInfo
	if l.loading {
		// Otherwise it storms on startup.
		lvl = slog.LevelDebug
	}
	slog.Log(ctx, lvl, "llm", "prompt", prompt)
	msgs := []message{
		{system, l.systemPrompt},
		{user, prompt},
	}
	//reply, err := l.promptBlocking(ctx, msgs)
	reply, err := l.promptStreaming(ctx, msgs)
	if err != nil {
		lvl := slog.LevelDebug
		if !l.loading || err == context.Canceled {
			lvl = slog.LevelError
		}
		slog.Log(ctx, lvl, "llm", "prompt", prompt, "error", err, "duration", time.Since(start).Round(time.Millisecond))
		return reply, err
	}
	// Llama-3
	reply = strings.TrimSuffix(reply, "<|eot_id|>")
	// Gemma-2
	reply = strings.TrimSuffix(reply, "<end_of_turn>")
	reply = strings.TrimSpace(reply)
	slog.Info("llm", "prompt", prompt, "reply", reply, "duration", time.Since(start).Round(time.Millisecond))
	return reply, nil
}

func (l *LLM) promptBlocking(ctx context.Context, msgs []message) (string, error) {
	data := openAIChatCompletionRequest{Model: "ignored", Messages: msgs}
	b, _ := json.Marshal(data)
	url := fmt.Sprintf("http://localhost:%d/v1/chat/completions", l.port)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	d := json.NewDecoder(resp.Body)
	d.DisallowUnknownFields()
	msg := openAIChatCompletionsResponse{}
	err = d.Decode(&msg)
	_ = resp.Body.Close()
	if err != nil {
		return "", err
	}
	if len(msg.Choices) != 1 {
		return "", errors.New("unexpected number of choices")
	}
	return msg.Choices[0].Message.Content, nil
}

func (l *LLM) promptStreaming(ctx context.Context, msgs []message) (string, error) {
	data := openAIChatCompletionRequest{Model: "ignored", Messages: msgs, Stream: true}
	b, _ := json.Marshal(data)
	url := fmt.Sprintf("http://localhost:%d/v1/chat/completions", l.port)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	r := bufio.NewReader(resp.Body)
	reply := ""
	for {
		line, err := r.ReadBytes('\n')
		if err == io.EOF {
			err = nil
			break
		}
		if err != nil {
			return "", err
		}
		line = bytes.TrimSpace(line)
		if len(line) == 0 {
			continue
		}
		if !bytes.HasPrefix(line, []byte("data: ")) {
			panic(line)
		}
		d := json.NewDecoder(bytes.NewReader(line[len("data: "):]))
		d.DisallowUnknownFields()
		msg := openAIChatCompletionsStreamResponse{}
		err = d.Decode(&msg)
		if err != nil {
			slog.Error("llm", "data", string(line))
			return "", err
		}
		if len(msg.Choices) != 1 {
			return "", errors.New("unexpected number of choices")
		}
		word := msg.Choices[0].Delta.Content
		slog.Debug("llm", "word", word)
		reply += word
	}
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
	// FinishReason is one of "stop", "length", "content_filter" or "tool_calls".
	FinishReason string  `json:"finish_reason"`
	Index        int     `json:"index"`
	Message      message `json:"message"`
}

// openAIChatCompletionsStreamResponse is not documented?
type openAIChatCompletionsStreamResponse struct {
	Choices []openAIStreamChoices `json:"choices"`
	Created int64                 `json:"created"`
	ID      string                `json:"id"`
	Model   string                `json:"model"`
	Object  string                `json:"object"`
}

type openAIStreamChoices struct {
	Delta openAIStreamDelta `json:"delta"`
	// FinishReason is one of null, "stop", "length", "content_filter" or "tool_calls".
	FinishReason string  `json:"finish_reason"`
	Index        int     `json:"index"`
	Message      message `json:"message"`
}

type openAIStreamDelta struct {
	Content string `json:"content"`
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
func downloadExec(ctx context.Context, url, dst string) error {
	if _, err := os.Stat(dst); err == nil || !os.IsNotExist(err) {
		return err
	}
	slog.Info("llm", "downloading", url)
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
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
func getHfModelGGUFFromLlamafile(ctx context.Context, cache, repo, model string) error {
	url := "https://huggingface.co/" + repo + "/resolve/main/" + model + ".llamafile?download=true"
	dst := filepath.Join(cache, model+".llamafile")
	if err := downloadExec(ctx, url, dst); err != nil {
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
