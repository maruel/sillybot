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

	"github.com/maruel/genai"
	"github.com/maruel/httpjson"
	"golang.org/x/sync/errgroup"
)

type Client struct {
	URL string
}

type message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

func (m *message) from(in *genai.Message) {
	m.Role = string(in.Role)
	if len(in.Contents) != 1 {
		panic("unexpected number of contents")
	}
	m.Content = in.Contents[0].Text
}

func (m *message) to(out *genai.Message) {
	out.Role = genai.Role(m.Role)
	out.Contents = []genai.Content{{Text: m.Content}}
}

type CompletionRequest struct {
	Stream   bool      `json:"stream"`
	Messages []message `json:"messages"`
}

func (c *CompletionRequest) Init(msgs genai.Messages, opts genai.Validatable) error {
	c.Messages = make([]message, len(msgs))
	for i := range c.Messages {
		c.Messages[i].from(&msgs[i])
	}
	return nil
}

type completionResponse struct {
	Choices []struct {
		FinishReason string  `json:"finish_reason"`
		Message      message `json:"message"`
	} `json:"choices"`
}

type CompletionStreamChunkResponse struct {
	Choices []struct {
		FinishReason string `json:"finish_reason"`
		Delta        struct {
			Content string `json:"content"`
		} `json:"delta"`
	} `json:"choices"`
}

func (c *Client) Chat(ctx context.Context, msgs genai.Messages, opts genai.Validatable) (genai.ChatResult, error) {
	rpcin := CompletionRequest{Messages: make([]message, len(msgs))}
	for i := range rpcin.Messages {
		rpcin.Messages[i].from(&msgs[i])
	}
	out := genai.ChatResult{}
	rpcout := completionResponse{}
	if err := httpjson.DefaultClient.Post(ctx, c.URL+"/v1/chat/completions", nil, &rpcin, &rpcout); err != nil {
		return out, err
	}
	rpcout.Choices[0].Message.to(&out.Message)
	return out, nil
}

func (c *Client) ChatStream(ctx context.Context, msgs genai.Messages, opts genai.Validatable, chunks chan<- genai.MessageFragment) (genai.ChatResult, error) {
	result := genai.ChatResult{}
	in := CompletionRequest{}
	if err := in.Init(msgs, opts); err != nil {
		return result, err
	}
	// TODO:
	ch := make(chan CompletionStreamChunkResponse)
	eg, ctx := errgroup.WithContext(ctx)
	eg.Go(func() error {
		return processStreamPackets(ch, chunks, &result)
	})
	err := c.ChatStreamRaw(ctx, &in, ch)
	close(ch)
	if err2 := eg.Wait(); err2 != nil {
		err = err2
	}
	return result, err
}

func processStreamPackets(ch <-chan CompletionStreamChunkResponse, chunks chan<- genai.MessageFragment, result *genai.ChatResult) error {
	defer func() {
		// We need to empty the channel to avoid blocking the goroutine.
		for range ch {
		}
	}()
	for pkt := range ch {
		if len(pkt.Choices) != 1 {
			return fmt.Errorf("server returned an unexpected number of choices, expected 1, got %d", len(pkt.Choices))
		}
		f := genai.MessageFragment{TextFragment: pkt.Choices[0].Delta.Content}
		if !f.IsZero() {
			chunks <- f
			if err := result.Accumulate(f); err != nil {
				return err
			}
		}
	}
	return nil
}

func (c *Client) ChatStreamRaw(ctx context.Context, in *CompletionRequest, out chan<- CompletionStreamChunkResponse) error {
	in.Stream = true
	resp, err := httpjson.DefaultClient.PostRequest(ctx, c.URL+"/v1/chat/completions", nil, in)
	if err != nil {
		return fmt.Errorf("failed to get server response: %w", err)
	}
	defer resp.Body.Close()
	for r := bufio.NewReader(resp.Body); ; {
		line, err := r.ReadBytes('\n')
		if line = bytes.TrimSpace(line); err == io.EOF {
			if len(line) == 0 {
				return nil
			}
		} else if err != nil {
			return fmt.Errorf("failed to get server response: %w", err)
		}
		if len(line) != 0 {
			if err := parseStreamLine(line, out); err != nil {
				return err
			}
		}
	}
}

func parseStreamLine(line []byte, out chan<- CompletionStreamChunkResponse) error {
	const dataPrefix = "data: "
	if !bytes.HasPrefix(line, []byte(dataPrefix)) {
		return fmt.Errorf("unexpected line. expected \"data: \", got %q", line)
	}
	suffix := string(line[len(dataPrefix):])
	d := json.NewDecoder(strings.NewReader(suffix))
	d.DisallowUnknownFields()
	d.UseNumber()
	msg := CompletionStreamChunkResponse{}
	if err := d.Decode(&msg); err != nil {
		return fmt.Errorf("failed to decode server response %q: %w", string(line), err)
	}
	out <- msg
	return nil
}
