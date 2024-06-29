//go:generate go run get_llama.go

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

type llm struct {
	c    *exec.Cmd
	port int
}

func newLLM(ctx context.Context) (*llm, error) {
	l := &llm{port: 8064}
	cmd := strings.Join([]string{
		"./Meta-Llama-3-8B-Instruct.Q5_K_M.llamafile",
		"--log-disable",
		"-ngl", "9999",
		"--nobrowser",
		"--port", strconv.Itoa(l.port),
	},
		" ")
	logger.Info("Running", "command", cmd)
	l.c = exec.CommandContext(
		ctx,
		"/bin/sh",
		"-c",
		cmd)
	if err := l.c.Start(); err != nil {
		return nil, err
	}
	logger.Info("Started llama", "pid", l.c.Process.Pid)
	return l, nil
}

func (l *llm) Close() error {
	logger.Info("Terminating llama")
	l.c.Cancel()
	l.c.Wait()
	return nil
}

type openAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type openAIReq struct {
	Model    string          `json:"model"`
	Stream   bool            `json:"stream"`
	Messages []openAIMessage `json:"messages"`
}

type openAIRespChoices struct {
	FinishReason string        `json:"finish_reason"`
	Index        int           `json:"index"`
	Message      openAIMessage `json:"message"`
}

type openAIResp struct {
	Choices []openAIRespChoices `json:"choices"`
	Created int64               `json:"created"`
	ID      string              `json:"id"`
	Model   string              `json:"model"`
	Object  string              `json:"object"`
	Usage   struct {
		CompletionTokens int64 `json:"completion_tokens"`
		PromptTokens     int64 `json:"prompt_tokens"`
		TotalTokens      int64 `json:"total_tokens"`
	} `json:"usage"`
}

func (l *llm) prompt(prompt string) (string, error) {
	data := openAIReq{
		Model: "llama-3",
		Messages: []openAIMessage{
			{"system", "You are a terse assistant. You reply with short and sarcastic answers. You are very blunt."},
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
	r := openAIResp{}
	err = d.Decode(&r)
	_ = resp.Body.Close()
	if err != nil {
		return "", err
	}
	if len(r.Choices) != 1 {
		return "", errors.New("unexpected number of choices")
	}
	reply := strings.TrimSuffix(r.Choices[0].Message.Content, "<|eot_id|>")
	logger.Info("llm", "prompt", prompt, "reply", reply, "duration", time.Since(start).Round(time.Millisecond))
	return reply, nil
}

func runPrompt(prompt string) (string, error) {
	ctx := context.Background()
	cmd := exec.CommandContext(
		ctx,
		"./Meta-Llama-3-8B-Instruct.Q5_K_M.llamafile",
		"-p", "Answer succinctly. "+prompt,
		"--prompt-cache", "prompt.cache",
		"--flash-attn",
		"--log-disable",
		"--seed", "1",
		"-ngl", "9999",
		"--silent-prompt")
	b, err := cmd.Output()
	return string(b), err
}
