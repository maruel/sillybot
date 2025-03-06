// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"strings"
	"time"

	"github.com/maruel/sillybot/internal"
	"github.com/maruel/sillybot/llm/common"
)

// Messages. https://platform.openai.com/docs/api-reference/making-requests

// openAIChatCompletionRequest is documented at
// https://platform.openai.com/docs/api-reference/chat/create
type openAIChatCompletionRequest struct {
	Model       string           `json:"model"`
	MaxTokens   int              `json:"max_tokens,omitempty"`
	Stream      bool             `json:"stream"`
	Messages    []common.Message `json:"messages"`
	Seed        int              `json:"seed,omitempty"`
	Temperature float64          `json:"temperature,omitempty"`
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
	FinishReason string         `json:"finish_reason"`
	Index        int            `json:"index"`
	Message      common.Message `json:"message"`
}

// openAIChatCompletionsStreamResponse is not documented?
type openAIChatCompletionsStreamResponse struct {
	Choices []openAIStreamChoices `json:"choices"`
	Created int64                 `json:"created"`
	ID      string                `json:"id"`
	Model   string                `json:"model"`
	Object  string                `json:"object"`
	Usage   struct {
		CompletionTokens int64 `json:"completion_tokens"`
		PromptTokens     int64 `json:"prompt_tokens"`
		TotalTokens      int64 `json:"total_tokens"`
	} `json:"usage"`
}

type openAIStreamChoices struct {
	Delta openAIStreamDelta `json:"delta"`
	// FinishReason is one of null, "stop", "length", "content_filter" or "tool_calls".
	FinishReason string `json:"finish_reason"`
	Index        int    `json:"index"`
	//Message      common.Message `json:"message"`
}

type openAIStreamDelta struct {
	Content string `json:"content"`
}

type openAIClient struct {
	baseURL string
}

func (l *openAIClient) PromptBlocking(ctx context.Context, msgs []common.Message, maxtoks, seed int, temperature float64) (string, error) {
	data := openAIChatCompletionRequest{
		Model:       "ignored",
		MaxTokens:   maxtoks,
		Messages:    msgs,
		Seed:        seed,
		Temperature: temperature,
	}
	msg := openAIChatCompletionsResponse{}
	if err := internal.JSONPost(ctx, l.baseURL+"/v1/chat/completions", data, &msg); err != nil {
		return "", fmt.Errorf("failed to get llama server chat response: %w", err)
	}
	if len(msg.Choices) != 1 {
		return "", fmt.Errorf("llama server returned an unexpected number of choices, expected 1, got %d", len(msg.Choices))
	}
	return msg.Choices[0].Message.Content, nil
}

func (l *openAIClient) PromptStreaming(ctx context.Context, msgs []common.Message, maxtoks, seed int, temperature float64, words chan<- string) (string, error) {
	start := time.Now()
	data := openAIChatCompletionRequest{
		Model:       "ignored",
		Messages:    msgs,
		MaxTokens:   maxtoks,
		Stream:      true,
		Seed:        seed,
		Temperature: temperature,
	}
	resp, err := internal.JSONPostRequest(ctx, l.baseURL+"/v1/chat/completions", data)
	if err != nil {
		return "", fmt.Errorf("failed to get llama server response: %w", err)
	}
	defer resp.Body.Close()
	r := bufio.NewReader(resp.Body)
	reply := ""
	for {
		line, err := r.ReadBytes('\n')
		line = bytes.TrimSpace(line)
		if err == io.EOF {
			err = nil
			if len(line) == 0 {
				return reply, nil
			}
		}
		if err != nil {
			return reply, fmt.Errorf("failed to get llama server response: %w", err)
		}
		if len(line) == 0 {
			continue
		}
		const prefix = "data: "
		if !bytes.HasPrefix(line, []byte(prefix)) {
			return reply, fmt.Errorf("unexpected line. expected \"data: \", got %q", line)
		}
		suffix := string(line[len(prefix):])
		if suffix == "[DONE]" {
			return reply, nil
		}
		d := json.NewDecoder(strings.NewReader(suffix))
		d.DisallowUnknownFields()
		msg := openAIChatCompletionsStreamResponse{}
		if err = d.Decode(&msg); err != nil {
			return reply, fmt.Errorf("failed to decode llama server response %q: %w", string(line), err)
		}
		if len(msg.Choices) != 1 {
			return reply, fmt.Errorf("llama server returned an unexpected number of choices, expected 1, got %d", len(msg.Choices))
		}
		word := msg.Choices[0].Delta.Content
		slog.Debug("llm", "word", word, "duration", time.Since(start).Round(time.Millisecond))
		// TODO: Remove.
		switch word {
		// Llama-3, Gemma-2, Phi-3
		case "<|eot_id|>", "<end_of_turn>", "<|end|>", "<|endoftext|>":
			return reply, nil
		case "":
		default:
			words <- word
			reply += word
		}
	}
}
