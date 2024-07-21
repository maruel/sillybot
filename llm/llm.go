// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package llm runs a LLM locally via llama.cpp, llamafile, or with a python
// server. It takes care of everything, including fetching gguf packed models
// from hugging face.
package llm

import (
	"archive/zip"
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"html/template"
	"io"
	"log/slog"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/maruel/sillybot/huggingface"
	"github.com/maruel/sillybot/internal"
	"github.com/maruel/sillybot/py"
	"golang.org/x/sys/cpu"
)

// Options for NewLLM.
type Options struct {
	// Remote is the host:port of a pre-existing server to use instead of
	// starting our own.
	Remote string
	// Model specifies a model to use.
	//
	// It will be selected automatically from KnownLLMs.
	//
	// Use "python" to use the integrated python backend.
	Model string

	_ struct{}
}

// KnownLLM is a known model.
//
// Currently assumes the model is hosted on HuggingFace.
type KnownLLM struct {
	// RepoID is the repository in the form <author>/<repo>.
	RepoID string `yaml:"repo"`
	// PackagingType is the file format used in the model. It can be one of
	// "safetensors" or "gguf".
	PackagingType string
	// Basename is the base filename when PackagingType is one of "gguf".
	Basename string
	// UpstreamID is the upstream repo when the model is based on another one.
	UpstreamID string `yaml:"upstream"`
	// PromptEncoding is only used when using llama-server in /completion mode.
	// When not present, llama-server is used in OpenAI compatible API mode.
	PromptEncoding *PromptEncoding `yaml:"prompt_encoding"`

	_ struct{}
}

// PromptEncoding describes how to encode the prompt.
type PromptEncoding struct {
	// Prompt encoding.
	BeginOfText              string `yaml:"begin_of_text"`
	SystemTokenStart         string `yaml:"system_token_start"`
	SystemTokenEnd           string `yaml:"system_token_end"`
	UserTokenStart           string `yaml:"user_token_start"`
	UserTokenEnd             string `yaml:"user_token_end"`
	AssistantTokenStart      string `yaml:"assistant_token_start"`
	AssistantTokenEnd        string `yaml:"assistant_token_end"`
	ToolsAvailableTokenStart string `yaml:"tools_available_token_start"`
	ToolsAvailableTokenEnd   string `yaml:"tools_available_token_end"`
	ToolCallTokenStart       string `yaml:"tool_call_token_start"`
	ToolCallTokenEnd         string `yaml:"tool_call_token_end"`
	ToolCallResultTokenStart string `yaml:"tool_call_result_token_start"`
	ToolCallResultTokenEnd   string `yaml:"tool_call_result_token_end"`

	_ struct{}
}

// URL returns the canonical URL for this repository.
func (k *KnownLLM) URL() string {
	return "https://huggingface.co/" + k.RepoID
}

// Validate checks for obvious errors in the fields.
func (k *KnownLLM) Validate() error {
	if strings.Count(k.RepoID, "/") != 1 {
		return fmt.Errorf("invalid repo %q", k.RepoID)
	}
	if k.PackagingType != "safetensors" && k.PackagingType != "gguf" {
		return fmt.Errorf("invalid packaginetype %q", k.PackagingType)
	}
	if strings.Count(k.UpstreamID, "/") != 1 {
		return fmt.Errorf("invalid upstream %q", k.UpstreamID)
	}
	// TODO: more validation.
	return nil
}

// Session runs a llama.cpp or llamafile server and runs queries on it.
//
// While it is expected that the model is an Instruct form, it is not a
// requirement.
type Session struct {
	HF       *huggingface.Client
	Model    string
	Encoding *PromptEncoding
	baseURL  string
	backend  string

	modelFile string
	c         *exec.Cmd
	done      <-chan error
	cancel    func() error

	_ struct{}
}

// New instantiates a llama.cpp or llamafile server, or optionally uses
// python instead.
func New(ctx context.Context, cache string, opts *Options, knownLLMs []KnownLLM) (*Session, error) {
	cacheModels := filepath.Join(cache, "models")
	if err := os.MkdirAll(cacheModels, 0o755); err != nil {
		return nil, fmt.Errorf("failed to create the directory to cache models: %w", err)
	}
	hf, err := huggingface.New("", cacheModels)
	if err != nil {
		return nil, err
	}
	l := &Session{HF: hf, Model: opts.Model}
	known := -1
	if opts.Model != "python" {
		for i, k := range knownLLMs {
			if strings.HasPrefix(opts.Model, k.Basename) {
				known = i
				l.Encoding = k.PromptEncoding
				break
			}
		}
		if known == -1 {
			return nil, fmt.Errorf("unknown LLM model %q, add to knownllms section first", l.Model)
		}
	}

	cachePy := filepath.Join(cache, "py")
	if opts.Remote == "" {
		llamasrv := ""
		isLlamafile := false
		modelFile := ""
		if opts.Model == "python" {
			if err := os.MkdirAll(cachePy, 0o755); err != nil {
				return nil, fmt.Errorf("failed to create the directory to cache python: %w", err)
			}
			if err := py.RecreateVirtualEnvIfNeeded(ctx, cachePy); err != nil {
				return nil, fmt.Errorf("failed to load llm: %w", err)
			}
			slog.Info("llm", "message", "using python")
			l.backend = "python"
		} else {
			// Make sure the server is available.
			var err error
			if llamasrv, isLlamafile, err = getLlama(ctx, cache); err != nil {
				return nil, fmt.Errorf("failed to load llm: %w", err)
			}
			if l.backend = "llama-server"; isLlamafile {
				l.backend = "llamafile"
			}
			cmd := mangleForLlamafile(isLlamafile, llamasrv, "--version")
			c := exec.CommandContext(ctx, cmd[0], cmd[1:]...)
			d, err := c.CombinedOutput()
			if err != nil {
				return nil, fmt.Errorf("failed to get llm version: %w\n%s", err, string(d))
			}
			slog.Info("llm", "path", llamasrv, "version", strings.TrimSpace(string(d)))

			// Make sure the model is available.
			if modelFile, err = l.ensureModel(ctx, opts.Model, knownLLMs[known]); err != nil {
				return nil, fmt.Errorf("failed to get llm model: %w", err)
			}
		}

		// Create the log file to redirect llamafile's output which is quite verbose.
		port := internal.FindFreePort(8031)
		l.baseURL = fmt.Sprintf("http://localhost:%d", port)
		if opts.Model == "python" {
			cmd := []string{filepath.Join(cachePy, "llm.py"), "--port", strconv.Itoa(port)}
			done, cancel, err := py.Run(ctx, filepath.Join(cachePy, "venv"), cmd, cachePy, filepath.Join(cachePy, "llm.log"))
			if err != nil {
				return nil, fmt.Errorf("failed to start python llm server: %w", err)
			}
			l.done = done
			l.cancel = cancel
		} else {
			done := make(chan error)
			l.done = done
			log, err := os.OpenFile(filepath.Join(cache, "llm.log"), os.O_WRONLY|os.O_APPEND|os.O_CREATE, 0o644)
			if err != nil {
				return nil, fmt.Errorf("failed to create llm server log file: %w", err)
			}
			defer log.Close()
			// Surprisingly llama-server seems to be hardcoded to 8 threads.
			threads := runtime.NumCPU() - 1
			if threads == 0 {
				threads = 1
			}
			// TODO: Investigate using -fa.
			// TODO: Doesn't seem to have any effect, need investigation.
			// "--prompt-cache", filepath.Join(cache, "llm-prompt-cache.bin"), "--prompt-cache-all",
			common := []string{
				llamasrv, "--model", modelFile, "--metrics", "-ngl", "9999", "--threads", strconv.Itoa(threads), "--port", strconv.Itoa(port),
			}
			cmd := mangleForLlamafile(isLlamafile, append(common, "--nobrowser")...)
			if !isLlamafile {
				cmd = mangleForLlamafile(isLlamafile, common...)
			}
			slog.Debug("llm", "command", cmd, "cwd", cache, "log", log.Name())
			l.c = exec.CommandContext(ctx, cmd[0], cmd[1:]...)
			l.c.Dir = cache
			l.c.Stdout = log
			l.c.Stderr = log
			l.c.Cancel = func() error {
				slog.Debug("llm", "state", "killing")
				return l.c.Process.Kill()
			}
			if err = l.c.Start(); err != nil {
				return nil, fmt.Errorf("failed to start llm server: %w", err)
			}
			go func() {
				done <- l.c.Wait()
				slog.Info("llm", "state", "terminated")
			}()
			slog.Info("llm", "state", "started", "pid", l.c.Process.Pid, "port", port)
		}
	} else {
		if !internal.IsHostPort(opts.Remote) {
			return nil, fmt.Errorf("invalid remote %q; use form 'host:port'", opts.Remote)
		}
		// TODO: Support online paid backends:
		// https://platform.openai.com/docs/api-reference/chat/create
		// https://docs.anthropic.com/en/api/messages-examples
		// https://cloud.google.com/vertex-ai/generative-ai/docs/start/quickstarts/quickstart-multimodal
		l.baseURL = "http://" + opts.Remote
		slog.Info("llm", "state", "loading")
		l.backend = "remote"
	}

	for ctx.Err() == nil {
		if status, _ := l.GetHealth(ctx); status == "ok" {
			break
		}
		select {
		case err := <-l.done:
			return nil, fmt.Errorf("starting llm server failed: %w", err)
		case <-ctx.Done():
		case <-time.After(100 * time.Millisecond):
		}
	}
	slog.Info("llm", "state", "ready", "model", opts.Model, "using", l.backend, "url", l.baseURL)
	return l, nil
}

func (l *Session) Close() error {
	slog.Info("llm", "state", "terminating")
	if l.done == nil {
		// Using a remote server.
		return nil
	}
	if l.cancel != nil {
		_ = l.cancel()
	} else {
		_ = l.c.Cancel()
	}
	err := <-l.done
	var er *exec.ExitError
	if errors.As(err, &er) {
		s, ok := er.ProcessState.Sys().(syscall.WaitStatus)
		if ok && s.Signaled() {
			// It was simply killed.
			err = nil
		}
		if runtime.GOOS == "windows" {
			// We need to figure out how to differentiate between normal quitting and
			// an error.
			err = nil
		}
	}
	return err
}

// GetHealth retrieves the heath of the server.
func (l *Session) GetHealth(ctx context.Context) (string, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", l.baseURL+"/health", nil)
	if err != nil {
		return "", fmt.Errorf("failed to create HTTP request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to get health response: %w", err)
	}
	d := json.NewDecoder(resp.Body)
	d.DisallowUnknownFields()
	msg := llamaCPPHealthResponse{}
	err = d.Decode(&msg)
	_ = resp.Body.Close()
	if err != nil {
		return msg.Status, fmt.Errorf("failed to decode health response: %w", err)
	}
	return msg.Status, nil
}

// TokenPerformance is the performance for the metrics
type TokenPerformance struct {
	Count    int
	Duration time.Duration
}

// Rate is the number of token per second.
func (t *TokenPerformance) Rate() float64 {
	if t.Duration == 0 {
		return 0
	}
	return float64(t.Count) / (float64(t.Duration) / float64(time.Second))
}

// Metrics represents the metrics for the LLM server.
type Metrics struct {
	Prompt             TokenPerformance
	Generated          TokenPerformance
	KVCacheUsage       float64
	KVCacheTokens      int
	RequestsProcessing int
	RequestedPending   int
}

// GetMetrics retrieves the performance statistics from the server.
func (l *Session) GetMetrics(ctx context.Context, m *Metrics) error {
	req, err := http.NewRequestWithContext(ctx, "GET", l.baseURL+"/metrics", nil)
	if err != nil {
		return fmt.Errorf("failed to create HTTP request: %w", err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to get metrics response: %w", err)
	}
	b, err := io.ReadAll(resp.Body)
	_ = resp.Body.Close()
	if err != nil {
		return fmt.Errorf("failed to get metrics response: %w", err)
	}
	// We hardcode things here since we know which server we are talking to. See
	// the commit history if you want the generic prometheus style data.
	for _, l := range strings.Split(strings.TrimSpace(string(b)), "\n") {
		if strings.HasPrefix(l, "#") {
			continue
		}
		parts := strings.Split(l, " ")
		if len(parts) != 2 {
			return fmt.Errorf("failed to parse line %q: %w", l, err)
		}
		// Search for these strings in
		// https://github.com/ggerganov/llama.cpp/blob/master/examples/server/server.cpp
		f, err := strconv.ParseFloat(parts[1], 64)
		if err != nil {
			return fmt.Errorf("failed to parse line %q: %w", l, err)
		}
		i, _ := strconv.Atoi(parts[1])
		switch parts[0] {
		case "llamacpp:prompt_tokens_total":
			m.Prompt.Count = i
		case "llamacpp:prompt_seconds_total":
			m.Prompt.Duration = time.Duration(f*1000) * time.Millisecond
		case "llamacpp:tokens_predicted_total":
			m.Generated.Count = i
		case "llamacpp:tokens_predicted_seconds_total":
			m.Generated.Duration = time.Duration(f*1000) * time.Millisecond
		case "llamacpp:prompt_tokens_seconds", "llamacpp:predicted_tokens_seconds":
			// Ignore.
		case "llamacpp:kv_cache_usage_ratio":
			m.KVCacheUsage = f
		case "llamacpp:kv_cache_tokens":
			m.KVCacheTokens = i
		case "llamacpp:requests_processing":
			m.RequestsProcessing = i
		case "llamacpp:requests_deferred":
			m.RequestedPending = i
		default:
			return fmt.Errorf("unknown metric %q", l)
		}
	}
	return nil
}

// Prompt prompts the LLM and returns the reply.
//
// See PromptStreaming for the arguments values.
//
// The first message is assumed to be the system prompt.
func (l *Session) Prompt(ctx context.Context, msgs []Message, seed int, temperature float64) (string, error) {
	start := time.Now()
	msgs = l.processMsgs(msgs)
	slog.Info("llm", "msgs", msgs)
	reply := ""
	var err error
	if l.Encoding == nil {
		reply, err = l.openAIPromptBlocking(ctx, msgs, seed, temperature)
	} else {
		reply, err = l.llamaCPPPromptBlocking(ctx, msgs, seed, temperature)
	}
	if err != nil {
		slog.Error("llm", "msgs", msgs, "error", err, "duration", time.Since(start).Round(time.Millisecond))
		return reply, err
	}
	// TODO: Remove all these.
	// Llama-3
	reply = strings.TrimSuffix(reply, "<|eot_id|>")
	// Gemma-2
	reply = strings.TrimSuffix(reply, "<end_of_turn>")
	// Phi-3
	reply = strings.TrimSuffix(reply, "<|end|>")
	reply = strings.TrimSuffix(reply, "<|endoftext|>")
	reply = strings.TrimSpace(reply)
	slog.Info("llm", "reply", reply, "duration", time.Since(start).Round(time.Millisecond))
	return reply, nil
}

// PromptStreaming prompts the LLM and returns the reply in the supplied channel.
//
// Use a non-zero seed to get deterministic output (without strong guarantees).
//
// Use low temperature (<1.0) to get more deterministic and repetitive output.
//
// Use high temperature (>1.0) to get more creative and random text. High
// values can result in nonsensical responses.
//
// It is recommended to use 1.0 by default, except some models (like
// Mistral-Nemo) requires much lower value <=0.3.
//
// The first message is assumed to be the system prompt.
func (l *Session) PromptStreaming(ctx context.Context, msgs []Message, seed int, temperature float64, words chan<- string) error {
	start := time.Now()
	msgs = l.processMsgs(msgs)
	slog.Info("llm", "msgs", msgs)
	reply := ""
	var err error
	if l.Encoding == nil {
		reply, err = l.openAIPromptStreaming(ctx, msgs, seed, temperature, words)
	} else {
		reply, err = l.llamaCPPPromptStreaming(ctx, msgs, seed, temperature, words)
	}
	if err != nil {
		slog.Error("llm", "reply", reply, "error", err, "duration", time.Since(start).Round(time.Millisecond))
		return err
	}
	slog.Info("llm", "reply", reply, "duration", time.Since(start).Round(time.Millisecond))
	return nil
}

//

func (l *Session) openAIPromptBlocking(ctx context.Context, msgs []Message, seed int, temperature float64) (string, error) {
	data := openAIChatCompletionRequest{
		Model:       "ignored",
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

func (l *Session) openAIPromptStreaming(ctx context.Context, msgs []Message, seed int, temperature float64, words chan<- string) (string, error) {
	data := openAIChatCompletionRequest{
		Model:       "ignored",
		Messages:    msgs,
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
		if !bytes.HasPrefix(line, []byte("data: ")) {
			return reply, fmt.Errorf("unexpected line. expected \"data: \", got %q", line)
		}
		d := json.NewDecoder(bytes.NewReader(line[len("data: "):]))
		d.DisallowUnknownFields()
		msg := openAIChatCompletionsStreamResponse{}
		if err = d.Decode(&msg); err != nil {
			return reply, fmt.Errorf("failed to decode llama server response %q: %w", string(line), err)
		}
		if len(msg.Choices) != 1 {
			return reply, fmt.Errorf("llama server returned an unexpected number of choices, expected 1, got %d", len(msg.Choices))
		}
		word := msg.Choices[0].Delta.Content
		slog.Debug("llm", "word", word)
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

func (l *Session) llamaCPPPromptBlocking(ctx context.Context, msgs []Message, seed int, temperature float64) (string, error) {
	data := llamaCPPCompletionRequest{Seed: int64(seed), Temperature: temperature}
	// Doc mentions it causes non-determinism even if a non-zero seed is
	// specified. Disable if it becomes a problem.
	data.CachePrompt = true
	if err := l.initPrompt(&data, msgs); err != nil {
		return "", err
	}
	msg := llamaCPPCompletionResponse{}
	if err := internal.JSONPost(ctx, l.baseURL+"/completion", data, &msg); err != nil {
		return "", fmt.Errorf("failed to get llama server response: %w", err)
	}
	slog.Debug("llm", "prompt tok", msg.Timings.PromptN, "gen tok", msg.Timings.PredictedN, "prompt tok/ms", msg.Timings.PromptPerTokenMS, "gen tok/ms", msg.Timings.PredictedPerTokenMS)
	return msg.Content, nil
}

func (l *Session) llamaCPPPromptStreaming(ctx context.Context, msgs []Message, seed int, temperature float64, words chan<- string) (string, error) {
	data := llamaCPPCompletionRequest{
		Stream:      true,
		Seed:        int64(seed),
		Temperature: temperature,
	}
	// Doc mentions it causes non-determinism even if a non-zero seed is
	// specified. Disable if it becomes a problem.
	data.CachePrompt = true
	if err := l.initPrompt(&data, msgs); err != nil {
		return "", err
	}
	resp, err := internal.JSONPostRequest(ctx, l.baseURL+"/completion", data)
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
		d := json.NewDecoder(bytes.NewReader(line))
		d.DisallowUnknownFields()
		msg := llamaCPPCompletionResponse{}
		if err = d.Decode(&msg); err != nil {
			return reply, fmt.Errorf("failed to decode llama server response %q: %w", string(line), err)
		}
		word := msg.Content
		slog.Debug("llm", "word", word, "stop", msg.Stop, "prompt tok", msg.Timings.PromptN, "gen tok", msg.Timings.PredictedN, "prompt tok/ms", msg.Timings.PromptPerTokenMS, "gen tok/ms", msg.Timings.PredictedPerTokenMS)
		if word != "" {
			words <- word
			reply += word
		}
		if msg.Stop {
			return reply, nil
		}
	}
}

func (l *Session) initPrompt(data *llamaCPPCompletionRequest, msgs []Message) error {
	// Do a quick validation. 1 == available_tools, 2 = system, 3 = rest
	state := 0
	data.Prompt = l.Encoding.BeginOfText
	for i, m := range msgs {
		switch m.Role {
		case AvailableTools:
			if state != 0 || i != 0 {
				return fmt.Errorf("unexpected available_tools message at index %d; state %d", i, state)
			}
			state = 1
			data.Prompt += l.Encoding.ToolsAvailableTokenStart + m.Content + l.Encoding.ToolsAvailableTokenEnd
		case System:
			if state > 1 {
				return fmt.Errorf("unexpected system message at index %d; state %d", i, state)
			}
			state = 2
			data.Prompt += l.Encoding.SystemTokenStart + m.Content + l.Encoding.SystemTokenEnd
		case User:
			state = 3
			data.Prompt += l.Encoding.UserTokenStart + m.Content + l.Encoding.UserTokenEnd
		case Assistant:
			state = 3
			data.Prompt += l.Encoding.AssistantTokenStart + m.Content + l.Encoding.AssistantTokenEnd
		case ToolCall:
			state = 3
			data.Prompt += l.Encoding.ToolCallTokenStart + m.Content + l.Encoding.ToolCallTokenEnd
		case ToolCallResult:
			state = 3
			data.Prompt += l.Encoding.ToolCallResultTokenStart + m.Content + l.Encoding.ToolCallResultTokenEnd
		default:
			return fmt.Errorf("unexpected role %q", m.Role)
		}
	}
	return nil
}

// ensureModel gets the model if missing.
//
// Currently hard-coded to GGUF files and Hugging Face. Doesn't support split
// files.
func (l *Session) ensureModel(ctx context.Context, model string, k KnownLLM) (string, error) {
	// TODO: This is very "meh".
	// Designed to handle special case like Mistral-7B-Instruct-v0.3-Q3_K_M.
	ext := strings.ToUpper(model)
	if i := strings.LastIndexByte(ext, '-'); i > 0 {
		ext = ext[i+1:]
	}
	if ext2 := filepath.Ext(ext); ext2 != "" {
		ext = strings.TrimLeft(ext2, ".")
	}
	switch ext {
	case "GGUF":
		return "", fmt.Errorf("do not include the .gguf suffix for model %q", model)

	case "BF16":
		if runtime.GOOS == "darwin" {
			slog.Warn("llm", "message", "As of July 2024, bfloat16 was not fully supported on Apple Silicon system. Remove this warning once this is fixed.")
		}

		// Well known quantizations.
	case "F32", "F16", "FP16":
	case "Q8_0", "Q6_K", "Q5_K_S", "Q5_K_M", "Q5_1", "Q5_0", "Q4_K_S", "Q4_K_M", "Q4_K", "Q4_1", "Q4_0", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q2_K":
	case "IQ4_NL", "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M", "IQ1_S", "IQ1_M":

	case "":
		return "", fmt.Errorf("you forgot to add a quantization suffix like 'BF16', 'F16', 'Q8_0' or 'Q5_K_M' when specifying model %q", model)

	default:
		return "", fmt.Errorf("unknown quantization %q for model %q, did you forget a suffix like 'BF16' or 'Q5_K_M'?", ext, model)
	}

	// Hack: quickly check if the file is there, if so, just return this.
	dst := filepath.Join(l.HF.Cache, model+".gguf")
	_, err := os.Stat(dst)
	if err == nil {
		l.modelFile = dst
		return dst, nil
	}
	slog.Info("llm", "model", model, "state", "missing")
	if k.RepoID == "" {
		return "", fmt.Errorf("can't guess model %q huggingface repo", model)
	}
	// Hack: we assume everything is on HuggingFace.
	switch k.PackagingType {
	case "gguf":
		if dst, err = l.HF.EnsureFile(ctx, k.RepoID, model+".gguf", 0o644); err != nil {
			// Get the list of files to help the user.
			parts := strings.SplitN(k.RepoID, "/", 2)
			m := huggingface.Model{ModelRef: huggingface.ModelRef{Author: parts[0], Repo: parts[1]}}
			err = fmt.Errorf("can't find model %q at %s: %w", model, m.URL(), err)
			if err2 := l.HF.GetModelInfo(ctx, &m); err2 != nil {
				return dst, errors.Join(err, err2)
			}
			msg := "Supported quantizations: "
			added := false
			for _, f := range m.Files {
				// TODO: Move this into a common function.
				if !strings.HasPrefix(f, k.Basename) {
					continue
				}
				if strings.Contains(f, "/") {
					// Skip files in subdirectories for now.
					continue
				}
				if strings.HasPrefix(filepath.Ext(f), ".cat") {
					// TODO: Support split files. For now just hide them. They are large
					// anyway so it's only for power users.
					continue
				}
				if added {
					msg += ", "
				}
				msg += strings.TrimSuffix(f[len(k.Basename):], ".gguf")
				added = true
			}
			return dst, fmt.Errorf("%w; %s", err, msg)
		}
		l.modelFile = dst
		return dst, nil
	default:
		return dst, fmt.Errorf("internal error: implement packaging type %s", k.PackagingType)
	}
}

// processMsgs process the system prompt.
func (l *Session) processMsgs(msgs []Message) []Message {
	if len(msgs) == 0 || msgs[0].Role != System {
		return msgs
	}
	t, err := template.New("").Parse(msgs[0].Content)
	if err != nil {
		slog.Error("llm", "message", "invalid system prompt", "system_prompt", msgs[0].Content, "error", err)
		return msgs
	}

	keys := map[string]string{
		"Now":   time.Now().Format("Monday 2006-01-02T15:04:05 MST"),
		"Model": l.Model,
	}
	b := bytes.Buffer{}
	if err = t.Execute(&b, keys); err != nil {
		slog.Error("llm", "message", "invalid system prompt", "system_prompt", msgs[0].Content, "error", err)
		return msgs
	}
	out := make([]Message, len(msgs))
	copy(out, msgs)
	out[0].Content = b.String()
	return out
}

// Messages. https://platform.openai.com/docs/api-reference/making-requests

type errorResponse struct {
	Code    int
	Message string
	Type    string
}

// llamaCPPHealthResponse is documented at
// https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md#api-endpoints
type llamaCPPHealthResponse struct {
	Status          string
	SlotsIdle       int `json:"slots_idle"`
	SlotsProcessing int `json:"slots_processing"`
}

// llamaCPPCompletionRequest is documented at
// https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md#api-endpoints
type llamaCPPCompletionRequest struct {
	SystemPrompt     string      `json:"system_prompt,omitempty"`
	Prompt           string      `json:"prompt"`
	Grammar          string      `json:"grammar,omitempty"`
	JSONSchema       interface{} `json:"json_schema,omitempty"`
	Seed             int64       `json:"seed,omitempty"`
	Temperature      float64     `json:"temperature,omitempty"`
	DynaTempRange    float64     `json:"dynatemp_range,omitempty"`
	DynaTempExponent float64     `json:"dynatemp_exponent,omitempty"`
	CachePrompt      bool        `json:"cache_prompt,omitempty"`
	Stream           bool
	// top_k             float64
	// top_p             float64
	// min_p             float64
	// n_predict         int64
	// n_keep            int64
	// stop              []string
	// tfs_z             float64
	// typical_p         float64
	// repeat_penalty    float64
	// repeat_last_n     int64
	// penalize_nl       bool
	// presence_penalty  float64
	// frequency_penalty float64
	// penalty_prompt    *string
	// mirostat          int32
	// mirostat_tau      float64
	// mirostat_eta      float64
	// ignore_eos   bool
	// logit_bias   []interface{}
	// n_probs      int64
	// min_keep     int64
	// image_data   []byte
	// id_slot      int64
	// samplers     []string
}

// llamaCPPCompletionResponse is documented at
// https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md#result-json
type llamaCPPCompletionResponse struct {
	Content            string
	Stop               bool
	GenerationSettings interface{} `json:"generation_settings"`
	Model              string
	Prompt             string
	StoppedEOS         bool   `json:"stopped_eos"`
	StoppedLimit       bool   `json:"stopped_limit"`
	StoppedWord        bool   `json:"stopped_word"`
	StoppingWord       string `json:"stopping_word"`
	Timings            struct {
		// Undocumented:
		PromptN             int64   `json:"prompt_n"`
		PromptMS            float64 `json:"prompt_ms"`
		PromptPerTokenMS    float64 `json:"prompt_per_token_ms"`
		PromptPerSecond     float64 `json:"prompt_per_second"`
		PredictedN          int64   `json:"predicted_n"`
		PredictedMS         float64 `json:"predicted_ms"`
		PredictedPerTokenMS float64 `json:"predicted_per_token_ms"`
		PredictedPerSecond  float64 `json:"predicted_per_second"`
	}
	TokensCached            int64 `json:"tokens_cached"`
	TokensEvaluated         int64 `json:"tokens_evaluated"`
	Truncated               bool
	CompletionProbabilities []struct {
		Content string
		Probs   []struct {
			Prob   float64
			TokStr string `json:"tok_str"`
		}
	} `json:"completion_probabilities"`
	// Undocumented:
	IDSlot          int64 `json:"id_slot"`
	TokensPredicted int64 `json:"tokens_predicted"`
	// Error case:
	Error errorResponse
}

// openAIChatCompletionRequest is documented at
// https://platform.openai.com/docs/api-reference/chat/create
type openAIChatCompletionRequest struct {
	Model       string    `json:"model"`
	Stream      bool      `json:"stream"`
	Messages    []Message `json:"messages"`
	Seed        int       `json:"seed,omitempty"`
	Temperature float64   `json:"temperature,omitempty"`
}

// Role is one of the LLM known roles.
type Role string

// LLM known roles.
const (
	System    Role = "system"
	User      Role = "user"
	Assistant Role = "assistant"
	// Specific to Mistral models.
	AvailableTools Role = "available_tools"
	ToolCall       Role = "tool_call"
	ToolCallResult Role = "tool_call_result"
)

// Message is a message to send to the LLM as part of the exchange.
type Message struct {
	Role    Role   `json:"role"`
	Content string `json:"content"`

	_ struct{}
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
	Message      Message `json:"message"`
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
	//Message      Message `json:"message"`
}

type openAIStreamDelta struct {
	Content string `json:"content"`
}

// Tools

// mangleForLlamafile hacks the command arguments to make it work for llamafile. Pass
// through otherwise.
func mangleForLlamafile(isLlamafile bool, cmd ...string) []string {
	// This hack is only needed for llamafile, not llama-server.
	if runtime.GOOS == "windows" || !isLlamafile {
		return cmd
	}
	// TODO: Proper escaping.
	return []string{"/bin/sh", "-c", strings.Join(cmd, " ")}
}

// getLlama returns the file path to llama.cpp/llamafile executable.
//
// Returns the file path to the executable and true if it is llamafile, false
// if it is llama-server from llama.cpp.
//
// It first look for llama-server or llamafile if one of them is PATH. Then it
// checks if one of them is s in the cache directory, otherwise downloads an
// hard coded version of llama-server from GitHub.
func getLlama(ctx context.Context, cache string) (string, bool, error) {
	if s, err := exec.LookPath("llama-server"); err == nil {
		return s, false, nil
	}
	if s, err := exec.LookPath("llamafile"); err == nil {
		return s, true, nil
	}
	execSuffix := ""
	if runtime.GOOS == "windows" {
		execSuffix = ".exe"
	}
	llamaserver := filepath.Join(cache, "llama-server"+execSuffix)
	if _, err := os.Stat(llamaserver); err == nil {
		return llamaserver, false, nil
	}
	llamafile := filepath.Join(cache, "llamafile"+execSuffix)
	if _, err := os.Stat(llamafile); err == nil {
		return llamafile, true, nil
	}

	// Time to download!
	// Do not just get the latest version because the odds of it breaking is just
	// too high. This is best effort.
	build := "b3428"
	url := "https://github.com/ggerganov/llama.cpp/releases/download/" + build + "/"
	zipname := ""
	files := []string{filepath.Base(llamaserver)}
	switch runtime.GOOS {
	case "darwin":
		zipname = "llama-" + build + "-bin-macos-arm64.zip"
		files = append(files, "ggml-metal.metal")
	case "linux":
		zipname = "llama-" + build + "-bin-ubuntu-x64.zip"
	case "windows":
		if cpu.X86.HasAVX512BF16 {
			zipname = "llama-" + build + "-bin-win-avx512-x64.zip"
		} else if cpu.X86.HasAVX2 {
			zipname = "llama-" + build + "-bin-win-avx2-x64.zip"
		} else {
			zipname = "llama-" + build + "-bin-win-avx-x64.zip"
		}
		files = append(files, "ggml.dll", "llama.dll")
	}
	zippath := filepath.Join(cache, zipname)
	if _, err := os.Stat(zippath); err != nil {
		if err := huggingface.DownloadFile(ctx, url+zipname, zippath, "", 0o644); err != nil {
			return "", false, fmt.Errorf("failed to download llamafile from github: %w", err)
		}
	}
	z, err := zip.OpenReader(zippath)
	if err != nil {
		return "", false, err
	}
	defer z.Close()
	for _, f := range z.File {
		// Files are under build/bin/
		n := filepath.Base(f.Name)
		for i, desired := range files {
			if n == desired {
				src, err := f.Open()
				if err != nil {
					return "", false, err
				}
				dst, err := os.OpenFile(filepath.Join(cache, desired), os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0o755)
				if err == nil {
					_, err = io.CopyN(dst, src, int64(f.UncompressedSize64))
				}
				if err2 := src.Close(); err == nil {
					err = err2
				}
				if err2 := dst.Close(); err == nil {
					err = err2
				}
				if err != nil {
					return "", false, fmt.Errorf("failed to write %q: %w", desired, err)
				}
				copy(files[i:], files[i+1:])
				if files = files[:len(files)-1]; len(files) == 0 {
					return llamaserver, false, err
				}
			}
		}
	}
	return "", false, fmt.Errorf("failed to find %q in %q", files, zipname)
}
