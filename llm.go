//go:generate go run get_llama.go

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
	"runtime"
	"strconv"
	"strings"
	"time"
)

type llm struct {
	c            *exec.Cmd
	port         int
	systemPrompt string
}

func newLLM(ctx context.Context, cache, model string) (*llm, error) {
	log, err := os.OpenFile(filepath.Join(cache, "server.log"), os.O_WRONLY|os.O_APPEND|os.O_CREATE, 0o644)
	if err != nil {
		return nil, err
	}
	defer log.Close()
	l := &llm{
		port:         8064,
		systemPrompt: "You are a terse assistant. You reply with short answers. You are often joyful, sometimes humorous, sometimes sarcastic.",
	}
	exe := "./llamafile"
	if runtime.GOOS == "windows" {
		exe = ".\\llamafile.exe"
	}
	cmd := []string{
		exe,
		"--model", model + ".gguf",
		"-ngl", "9999",
		"--nobrowser",
		"--port", strconv.Itoa(l.port),
	}
	single := strings.Join(cmd, " ")
	logger.Info("Running", "command", single, "cwd", cache)
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
	logger.Info("Started llama", "pid", l.c.Process.Pid)
	// TODO: Ping the server, since it can take a while to start.
	return l, nil
}

func (l *llm) Close() error {
	logger.Info("Terminating llama")
	l.c.Cancel()
	l.c.Wait()
	return nil
}

func (l *llm) prompt(prompt string) (string, error) {
	data := openAIChatCompletionRequest{
		Model: "llama-3",
		Messages: []openAIMessage{
			{"system", l.systemPrompt},
			{"user", prompt},
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
	Model    string          `json:"model"`
	Stream   bool            `json:"stream"`
	Messages []openAIMessage `json:"messages"`
}

type openAIMessage struct {
	// Role is one of system, user or assistant.
	Role    string `json:"role"`
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
	FinishReason string        `json:"finish_reason"`
	Index        int           `json:"index"`
	Message      openAIMessage `json:"message"`
}
