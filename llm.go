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
	"syscall"
	"time"

	"github.com/maruel/sillybot/huggingface"
)

// LLMOptions for NewLLM.
type LLMOptions struct {
	// Remote is the host:port of a pre-existing server to use instead of
	// starting our own.
	Remote string
	// Model specifies a model to use. Use "python" to use the python backend.
	Model string
	// Default system prompt to use.
	SystemPrompt string

	_ struct{}
}

// KnownLLM is a known model.
//
// Currently assumes it is hosted on HuggingFace.
type KnownLLM struct {
	// RepoID is the repository in the form <author>/<repo>.
	RepoID string `yaml:"repo"`
	// PackagingType is the file format used in the model. It can be one of
	// "safetensors", "gguf" or "llamafile".
	PackagingType string
	// Basename is the base filename when PackagingType is one of "gguf" or
	// "llamafile".
	Basename string
	// UpstreamID is the upstream repo when the model is based on another one.
	UpstreamID string `yaml:"upstream"`

	_ struct{}
}

// URL returns the canonical URL for this repository.
func (k *KnownLLM) URL() string {
	return "https://huggingface.co/" + k.RepoID
}

func (k *KnownLLM) validate() error {
	if strings.Count(k.RepoID, "/") != 1 {
		return errors.New("invalid repo")
	}
	if strings.Count(k.UpstreamID, "/") != 1 {
		return errors.New("invalid upstream")
	}
	// TODO: more validation.
	return nil
}

// LLM runs a llamafile server and runs queries on it.
//
// While it is expected that the model is an Instruct form, it is not a
// requirement.
type LLM struct {
	HF        *huggingface.Client
	KnownLLMs []KnownLLM

	c       *exec.Cmd
	done    <-chan error
	cancel  func() error
	url     string
	loading bool

	_ struct{}
}

// NewLLM instantiates a llama.cpp or llamafile server, or optionally uses
// python instead.
func NewLLM(ctx context.Context, cache string, opts *LLMOptions, knownLLMs []KnownLLM) (*LLM, error) {
	svr := "remote"
	cacheModels := filepath.Join(cache, "models")
	if err := os.MkdirAll(cacheModels, 0o755); err != nil {
		return nil, fmt.Errorf("failed to create the directory to cache models: %w", err)
	}
	hf, err := huggingface.New("", cacheModels)
	if err != nil {
		return nil, err
	}
	l := &LLM{HF: hf, KnownLLMs: knownLLMs, loading: true}
	cachePy := filepath.Join(cache, "py")
	if opts.Remote == "" {
		llamasrv := ""
		isLlamafile := false
		modelFile := ""
		if opts.Model == "python" {
			if err := os.MkdirAll(cachePy, 0o755); err != nil {
				return nil, fmt.Errorf("failed to create the directory to cache python: %w", err)
			}
			if pyNeedRecreate(cachePy) {
				if err := pyRecreate(ctx, cachePy); err != nil {
					return nil, fmt.Errorf("failed to load llm: %w", err)
				}
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
		port := findFreePort()
		l.url = fmt.Sprintf("http://localhost:%d/v1/chat/completions", port)
		if opts.Model == "python" {
			cmd := []string{filepath.Join(cachePy, "llm.py"), "--port", strconv.Itoa(port)}
			done, cancel, err := runPython(ctx, filepath.Join(cachePy, "venv"), cmd, cachePy, filepath.Join(cachePy, "llm.log"))
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
		if !isHostPort(opts.Remote) {
			return nil, fmt.Errorf("invalid remote %q; use form 'host:port'", opts.Remote)
		}
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
func (l *LLM) Prompt(ctx context.Context, msgs []Message, seed int, temperature float64) (string, error) {
	start := time.Now()
	lvl := slog.LevelInfo
	if l.loading {
		// Otherwise it storms on startup.
		lvl = slog.LevelDebug
	}
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
func (l *LLM) PromptStreaming(ctx context.Context, msgs []Message, seed int, temperature float64, words chan<- string) error {
	start := time.Now()
	lvl := slog.LevelInfo
	if l.loading {
		// Otherwise it storms on startup.
		lvl = slog.LevelDebug
	}
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
// Currently hard-coded to GGUF or llamafile files and Hugging Face. Doesn't
// support split files.
func (l *LLM) ensureModel(ctx context.Context, model string) (string, error) {
	// TODO: This is very "meh".
	switch strings.ToUpper(filepath.Ext(model)) {
	case ".GGUF":
		return "", fmt.Errorf("do not include the .gguf suffix for model %q", model)

	case ".BF16":
		if runtime.GOOS == "darwin" {
			slog.Warn("llm", "message", "As of July 2024, bfloat16 was not fully supported on Apple Silicon system. Remove this warning once this is fixed.")
		}

		// Well known quantizations.
	case ".F16", ".FP16", ".Q8_0", ".Q6_K", ".Q5_K_S", ".Q5_K_M", ".Q5_1", ".Q5_0", ".Q4_K_S", ".Q4_K_M", ".Q4_1", ".Q4_0", ".Q3_K_S", ".Q3_K_M", ".Q3_K_L", ".Q2_K":

	case "":
		return "", fmt.Errorf("you forgot to add a quantization suffix like '.BF16', '.F16', '.Q8_0' or '.Q5_K_M' when specifying model %q", model)

	default:
		return "", fmt.Errorf("unknown quantization for model %q, did you forget a suffix like '.BF16' or '.Q5_K_M'?", model)
	}

	// TODO: It'd be great to enumerate the GGUF files in the repo to help the user.

	// Hack.
	dst := filepath.Join(l.HF.Cache, model+".gguf")
	_, err := os.Stat(dst)
	if err == nil {
		return dst, nil
	}
	slog.Info("llm", "model", model, "state", "missing")
	url := ""
	t := ""
	for _, k := range l.KnownLLMs {
		if strings.HasPrefix(model, k.Basename) {
			url = k.URL()
			t = k.PackagingType
			break
		}
	}
	if url == "" {
		return "", fmt.Errorf("can't guess model %q huggingface repo", model)
	}
	// Hack.
	urlhf := "https://huggingface.co/"
	if !strings.HasPrefix(url, urlhf) {
		return "", fmt.Errorf("can't guess model %q source", model)
	}
	repo := url[len(urlhf):]
	switch t {
	case "gguf":
		if dst, err = l.HF.EnsureFile(ctx, repo, model+".gguf", 0o644); err != nil {
			err = fmt.Errorf("can't find model %q at %s: %w", model, url, err)
		}
	case "llamafile":
		// TODO: Remove support for llamafile.
		if dst, err = l.ensureGGUFFromLlamafile(ctx, repo, model); err != nil {
			err = fmt.Errorf("can't find model %q at %s: %w", model, url, err)
		}
	default:
		err = errors.New("internal error: unknown type")
	}
	return dst, err
}

// ensureGGUFFromLlamafile retrieves a file from an HuggingFace repository if missing.
func (l *LLM) ensureGGUFFromLlamafile(ctx context.Context, repo string, model string) (string, error) {
	gguf := model + ".gguf"
	dstgguf := filepath.Join(l.HF.Cache, gguf)
	if _, err := os.Stat(dstgguf); err == nil {
		return dstgguf, err
	}
	// Get the llamafile first.
	src, err := l.HF.EnsureFile(ctx, repo, model+".llamafile", 0o755)
	if err != nil {
		return dstgguf, err
	}
	z, err := zip.OpenReader(src)
	if err != nil {
		return dstgguf, err
	}
	defer z.Close()
	for _, i := range z.File {
		if i.Name == gguf {
			s, err := i.Open()
			if err != nil {
				return dstgguf, err
			}
			defer s.Close()
			d, err := os.OpenFile(dstgguf, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
			if err != nil {
				return dstgguf, err
			}
			defer d.Close()
			_, err = io.CopyN(d, s, int64(i.UncompressedSize64))
			return dstgguf, err
		}
	}
	return dstgguf, errors.New("gguf not found")
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
