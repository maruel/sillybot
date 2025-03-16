// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package py

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/maruel/genai/genaiapi"
	"github.com/maruel/httpjson"
)

type CompletionProvider struct {
	URL string
}

type message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type completionRequest struct {
	Stream   bool      `json:"stream"`
	Messages []message `json:"messages"`
}

func (c *CompletionProvider) Completion(ctx context.Context, msgs []genaiapi.Message, opts genaiapi.Validatable) (genaiapi.CompletionResult, error) {
	rpcin := completionRequest{}
	for _, m := range msgs {
		rpcin.Messages = append(rpcin.Messages, message{Role: string(m.Role), Content: m.Text})
	}
	var rpcout struct {
		Choices []struct {
			FinishReason string `json:"finish_reason"`
			Message      struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	out := genaiapi.CompletionResult{}
	if err := httpjson.DefaultClient.Post(ctx, c.URL+"/v1/chat/completions", nil, &rpcin, &rpcout); err != nil {
		return out, err
	}
	out.Role = genaiapi.Role(rpcout.Choices[0].Message.Role)
	out.Type = genaiapi.Text
	out.Text = rpcout.Choices[0].Message.Content
	return out, nil
}

func (c *CompletionProvider) CompletionStream(ctx context.Context, msgs []genaiapi.Message, opts genaiapi.Validatable, chunks chan<- genaiapi.MessageChunk) error {
	in := completionRequest{Stream: true}
	for _, m := range msgs {
		in.Messages = append(in.Messages, message{Role: string(m.Role), Content: m.Text})
	}
	resp, err := httpjson.DefaultClient.PostRequest(ctx, c.URL+"/v1/chat/completions", nil, &in)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	r := bufio.NewReader(resp.Body)
	for {
		line, err := r.ReadBytes('\n')
		line = bytes.TrimSpace(line)
		if err == io.EOF {
			err = nil
			if len(line) == 0 {
				return nil
			}
		}
		if err != nil {
			return fmt.Errorf("failed to get server response: %w", err)
		}
		if len(line) == 0 {
			continue
		}
		const prefix = "data: "
		if !bytes.HasPrefix(line, []byte(prefix)) {
			return fmt.Errorf("unexpected line. expected \"data: \", got %q", line)
		}
		suffix := string(line[len(prefix):])
		if suffix == "[DONE]" {
			return nil
		}
		d := json.NewDecoder(strings.NewReader(suffix))
		d.DisallowUnknownFields()
		d.UseNumber()
		var msg struct {
			Choices []struct {
				FinishReason string `json:"finish_reason"`
				Delta        struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err = d.Decode(&msg); err != nil {
			return fmt.Errorf("failed to decode server response %q: %w", string(line), err)
		}
		if len(msg.Choices) != 1 {
			return fmt.Errorf("server returned an unexpected number of choices, expected 1, got %d", len(msg.Choices))
		}
		chunks <- genaiapi.MessageChunk{Role: genaiapi.Assistant, Type: genaiapi.Text, Text: msg.Choices[0].Delta.Content}
	}
}
