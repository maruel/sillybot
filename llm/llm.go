// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package llm

import (
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
	"github.com/maruel/sillybot/py"
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
	// Default system prompt to use. Is a Go template as documented at
	// https://pkg.go.dev/text/template. Values provided are:
	// - Now: current time in ISO-8601, including the server's time zone.
	// - Model: the model name.
	SystemPrompt string `yaml:"system_prompt"`

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

// LLM runs a llama.cpp or llamafile server and runs queries on it.
//
// While it is expected that the model is an Instruct form, it is not a
// requirement.
type LLM struct {
	HF        *huggingface.Client
	KnownLLMs []KnownLLM
	model     string

	c       *exec.Cmd
	done    <-chan error
	cancel  func() error
	url     string
	loading bool

	_ struct{}
}

// New instantiates a llama.cpp or llamafile server, or optionally uses
// python instead.
func New(ctx context.Context, cache string, opts *Options, knownLLMs []KnownLLM) (*LLM, error) {
	if opts.SystemPrompt == "" {
		return nil, errors.New("did you forget to specify a system prompt?")
	}
	svr := "remote"
	cacheModels := filepath.Join(cache, "models")
	if err := os.MkdirAll(cacheModels, 0o755); err != nil {
		return nil, fmt.Errorf("failed to create the directory to cache models: %w", err)
	}
	hf, err := huggingface.New("", cacheModels)
	if err != nil {
		return nil, err
	}
	l := &LLM{HF: hf, KnownLLMs: knownLLMs, model: opts.Model, loading: true}
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
		} else {
			// Make sure the server is available.
			var err error
			if llamasrv, isLlamafile, err = getLlama(ctx, cache); err != nil {
				return nil, fmt.Errorf("failed to load llm: %w", err)
			}
			cmd := mangleForLlamafile(isLlamafile, llamasrv, "--version")
			c := exec.CommandContext(ctx, cmd[0], cmd[1:]...)
			d, err := c.CombinedOutput()
			if err != nil {
				return nil, fmt.Errorf("failed to get llm version: %w", err)
			}
			slog.Info("llm", "path", llamasrv, "version", strings.TrimSpace(string(d)))

			// Make sure the model is available.
			if modelFile, err = l.ensureModel(ctx, opts.Model); err != nil {
				return nil, fmt.Errorf("failed to get llm model: %w", err)
			}
		}

		// Create the log file to redirect llamafile's output which is quite verbose.
		port := py.FindFreePort()
		l.url = fmt.Sprintf("http://localhost:%d/v1/chat/completions", port)
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
			common := []string{
				llamasrv, "--model", modelFile, "-ngl", "9999", "--threads", strconv.Itoa(threads), "--port", strconv.Itoa(port),
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
		if opts.Model == "python" {
			svr = "llm.py"
		} else if isLlamafile {
			svr = "llamafile"
		} else {
			svr = "llama-server"
		}
	} else {
		if !py.IsHostPort(opts.Remote) {
			return nil, fmt.Errorf("invalid remote %q; use form 'host:port'", opts.Remote)
		}
		// TODO: Support online paid backends:
		// https://platform.openai.com/docs/api-reference/chat/create
		// https://docs.anthropic.com/en/api/messages-examples
		// https://cloud.google.com/vertex-ai/generative-ai/docs/start/quickstarts/quickstart-multimodal
		l.url = "http://" + opts.Remote + "/v1/chat/completions"
		slog.Info("llm", "state", "loading")
	}

	msgs := []Message{
		{Role: System, Content: "You are an AI assistant. You strictly follow orders. Do not add punctuation. Do not use uppercase letters."},
		{Role: User, Content: "reply with \"ok\""},
	}
	for ctx.Err() == nil {
		if resp, err := l.Prompt(ctx, msgs, 1, 0.1); err == nil {
			// Phi-3 can't follow orders properly.
			if !strings.HasPrefix(strings.ToLower(resp), "ok") {
				_ = l.Close()
				return nil, fmt.Errorf("failed to get initial query from llm server: unexpected response from llm. expected \"ok\", got %q", resp)
			}
			break
		}
		select {
		case err := <-l.done:
			return nil, fmt.Errorf("starting llm server failed: %w", err)
		case <-ctx.Done():
		case <-time.After(100 * time.Millisecond):
		}
	}
	slog.Info("llm", "state", "ready", "model", opts.Model, "using", svr, "url", l.url)
	l.loading = false
	return l, nil
}

func (l *LLM) Close() error {
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
	}
	return err
}

// Prompt prompts the LLM and returns the reply.
//
// Use a non-zero seed to get deterministic output (without strong guarantees).
//
// Use low temperature (<1.0) to get more deterministic and repetitive output.
//
// Use high temperature (>1.0) to get more creative and random text. High
// values can result in nonsensical responses.
//
// It is recommended to use 1.0 by default.
//
// The first message is assumed to be the system prompt. It will be processed
// as described in Options.SystemPrompt.
func (l *LLM) Prompt(ctx context.Context, msgs []Message, seed int, temperature float64) (string, error) {
	start := time.Now()
	lvl := slog.LevelInfo
	if l.loading {
		// Otherwise it storms on startup.
		lvl = slog.LevelDebug
	}
	msgs = l.processMsgs(msgs)
	slog.Log(ctx, lvl, "llm", "msgs", msgs)
	reply, err := l.promptBlocking(ctx, msgs, seed, temperature)
	if err != nil {
		lvl := slog.LevelDebug
		if !l.loading || err == context.Canceled {
			lvl = slog.LevelError
		}
		slog.Log(ctx, lvl, "llm", "msgs", msgs, "error", err, "duration", time.Since(start).Round(time.Millisecond))
		return reply, err
	}
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
// It is recommended to use 1.0 by default.
//
// The first message is assumed to be the system prompt. It will be processed
// as described in Options.SystemPrompt.
func (l *LLM) PromptStreaming(ctx context.Context, msgs []Message, seed int, temperature float64, words chan<- string) error {
	start := time.Now()
	lvl := slog.LevelInfo
	if l.loading {
		// Otherwise it storms on startup.
		lvl = slog.LevelDebug
	}
	msgs = l.processMsgs(msgs)
	slog.Log(ctx, lvl, "llm", "msgs", msgs)
	reply, err := l.promptStreaming(ctx, msgs, seed, temperature, words)
	if err != nil {
		lvl := slog.LevelDebug
		if !l.loading || err == context.Canceled {
			lvl = slog.LevelError
		}
		slog.Log(ctx, lvl, "llm", "reply", reply, "error", err, "duration", time.Since(start).Round(time.Millisecond))
		return err
	}
	slog.Info("llm", "reply", reply, "duration", time.Since(start).Round(time.Millisecond))
	return nil
}

func (l *LLM) promptBlocking(ctx context.Context, msgs []Message, seed int, temperature float64) (string, error) {
	data := openAIChatCompletionRequest{
		Model:       "ignored",
		Messages:    msgs,
		Seed:        seed,
		Temperature: temperature,
	}
	b, err := json.Marshal(data)
	if err != nil {
		return "", fmt.Errorf("internal error: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, "POST", l.url, bytes.NewReader(b))
	if err != nil {
		return "", fmt.Errorf("failed to create HTTP request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to get llama server response: %w", err)
	}
	d := json.NewDecoder(resp.Body)
	d.DisallowUnknownFields()
	msg := openAIChatCompletionsResponse{}
	err = d.Decode(&msg)
	_ = resp.Body.Close()
	if err != nil {
		return "", fmt.Errorf("failed to get llama server response: %w", err)
	}
	if len(msg.Choices) != 1 {
		return "", fmt.Errorf("llama server returned an unexpected number of choices, expected 1, got %d", len(msg.Choices))
	}
	return msg.Choices[0].Message.Content, nil
}

func (l *LLM) promptStreaming(ctx context.Context, msgs []Message, seed int, temperature float64, words chan<- string) (string, error) {
	data := openAIChatCompletionRequest{
		Model:       "ignored",
		Messages:    msgs,
		Stream:      true,
		Seed:        seed,
		Temperature: temperature,
	}
	b, err := json.Marshal(data)
	if err != nil {
		return "", fmt.Errorf("internal error: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, "POST", l.url, bytes.NewReader(b))
	if err != nil {
		return "", fmt.Errorf("failed to get llama server response: %w", err)
	}
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

// ensureModel gets the model if missing.
//
// Currently hard-coded to GGUF files and Hugging Face. Doesn't support split
// files.
func (l *LLM) ensureModel(ctx context.Context, model string) (string, error) {
	// TODO: This is very "meh".
	ext := strings.TrimLeft(strings.ToUpper(filepath.Ext(model)), ".")
	if ext == "" {
		if i := strings.LastIndexByte(model, '-'); i > 0 {
			ext = model[i+1:]
		}
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
		return "", fmt.Errorf("unknown quantization for model %q, did you forget a suffix like 'BF16' or 'Q5_K_M'?", model)
	}

	// Hack: quickly check if the file is there, if so, just return this.
	dst := filepath.Join(l.HF.Cache, model+".gguf")
	_, err := os.Stat(dst)
	if err == nil {
		return dst, nil
	}
	slog.Info("llm", "model", model, "state", "missing")
	var known KnownLLM
	for _, k := range l.KnownLLMs {
		if strings.HasPrefix(model, k.Basename) {
			known = k
			break
		}
	}
	if known.RepoID == "" {
		return "", fmt.Errorf("can't guess model %q huggingface repo", model)
	}
	// Hack: we assume everything is on HuggingFace.
	switch known.PackagingType {
	case "gguf":
		if dst, err = l.HF.EnsureFile(ctx, known.RepoID, model+".gguf", 0o644); err != nil {
			// Get the list of files to help the user.
			parts := strings.SplitN(known.RepoID, "/", 2)
			m := huggingface.Model{ModelRef: huggingface.ModelRef{Author: parts[0], Repo: parts[1]}}
			err = fmt.Errorf("can't find model %q at %s: %w", model, m.URL(), err)
			if err2 := l.HF.GetModelInfo(ctx, &m); err2 != nil {
				return dst, errors.Join(err, err2)
			}
			msg := "Supported quantizations: "
			added := false
			for _, f := range m.Files {
				// TODO: Move this into a common function.
				if !strings.HasPrefix(f, known.Basename) {
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
				msg += strings.TrimSuffix(f[len(known.Basename):], ".gguf")
				added = true
			}
			return dst, fmt.Errorf("%w; %s", err, msg)
		}
	default:
		err = fmt.Errorf("internal error: implement packaging type %s", known.PackagingType)
	}
	return dst, err
}

// processMsgs process the system prompt.
func (l *LLM) processMsgs(msgs []Message) []Message {
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
		"Model": l.model,
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
// checks if one of them is s in the cache directory, otherwise downloads the
// latest version of llamafile from GitHub.
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

	// Time to download.
	// Download llamafile from GitHub. We always want the latest and greatest
	// as it is very actively developed and the model we download likely use an
	// older version.
	url, name, err := getGitHubLatestRelease("Mozilla-Ocho", "llamafile", "application/octet-stream")
	if err != nil {
		return "", false, fmt.Errorf("failed to find latest llamafile release from github: %w", err)
	}
	versioned := filepath.Join(cache, name+execSuffix)
	if err = huggingface.DownloadFile(ctx, url, versioned, "", 0o755); err != nil {
		return "", false, fmt.Errorf("failed to download llamafile from github: %w", err)
	}
	// Copy it as the default executable to use.
	if err = copyFile(llamafile, versioned); err != nil {
		return "", false, fmt.Errorf("failed to copy llamafile in cache: %w", err)
	}
	return llamafile, true, nil
}

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
	_ = resp.Body.Close()
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