// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package llm runs a LLM locally via llama.cpp, llamafile, or with a python
// server. It takes care of everything, including fetching gguf packed models
// from hugging face.
package llm

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"html/template"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"runtime/trace"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/maruel/genai/genaiapi"
	"github.com/maruel/genai/llamacpp"
	"github.com/maruel/huggingface"
	"github.com/maruel/sillybot/internal"
	"github.com/maruel/sillybot/llm/llamacppsrv"
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
	Model PackedFileRef
	// ContextLength will limit the context length. This is useful with the newer
	// 128K context window models that will require too much memory and quite
	// slow to run. A good value to recommend is 8192 or 32768.
	ContextLength int `yaml:"context_length"`

	_ struct{}
}

// Validate checks for obvious errors in the fields.
func (o *Options) Validate() error {
	// TODO: Remote.
	if o.Model != "" && o.Model != "python" {
		if err := o.Model.Validate(); err != nil {
			return err
		}
	}
	return nil
}

// KnownLLM is a known model.
//
// Currently assumes the model is hosted on HuggingFace.
type KnownLLM struct {
	// Source is the repository in the form "hf:<author>/<repo>/<basename>".
	Source PackedFileRef `yaml:"source"`
	// PackagingType is the file format used in the model. It can be one of
	// "safetensors" or "gguf".
	PackagingType string
	// Upstream is the upstream repo in the form "hf:<author>/<repo>" when the
	// model is based on another one.
	Upstream PackedRepoRef `yaml:"upstream"`
	// PromptEncoding is only used when using llama-server in /completion mode.
	// When not present, llama-server is used in OpenAI compatible API mode.
	PromptEncoding *llamacpp.PromptEncoding `yaml:"prompt_encoding"`

	_ struct{}
}

// Validate checks for obvious errors in the fields.
func (k *KnownLLM) Validate() error {
	if err := k.Source.Validate(); err != nil {
		return fmt.Errorf("invalid source: %w", err)
	}
	if k.PackagingType != "safetensors" && k.PackagingType != "gguf" {
		return fmt.Errorf("invalid packaginetype %q", k.PackagingType)
	}
	if err := k.Upstream.Validate(); err != nil {
		return fmt.Errorf("invalid upstream: %w", err)
	}
	if k.PromptEncoding != nil {
		if err := k.PromptEncoding.Validate(); err != nil {
			return err
		}
	}
	return nil
}

// Session runs a llama.cpp or llamafile server and runs queries on it.
//
// While it is expected that the model is an Instruct form, it is not a
// requirement.
type Session struct {
	HF       *huggingface.Client
	Model    PackedFileRef
	Encoding *llamacpp.PromptEncoding
	baseURL  string
	backend  string
	cp       genaiapi.CompletionProvider

	cache     string
	modelFile string
	c         *exec.Cmd
	done      <-chan error
	cancel    func() error

	_ struct{}
}

// New instantiates a llama.cpp or llamafile server, or optionally uses
// python instead.
func New(ctx context.Context, cache string, opts *Options, knownLLMs []KnownLLM) (*Session, error) {
	if err := opts.Validate(); err != nil {
		return nil, err
	}
	cacheModels := filepath.Join(cache, "models")
	if err := os.MkdirAll(cacheModels, 0o755); err != nil {
		return nil, fmt.Errorf("failed to create the directory to cache models: %w", err)
	}
	hf, err := huggingface.New("")
	if err != nil {
		return nil, err
	}
	l := &Session{HF: hf, Model: opts.Model, cache: cacheModels}
	known := -1
	if opts.Model != "python" {
		for i, k := range knownLLMs {
			if strings.HasPrefix(string(opts.Model), string(k.Source)) {
				known = i
				l.Encoding = k.PromptEncoding
				break
			}
		}
		if known == -1 {
			return nil, fmt.Errorf("unknown LLM model %q, add to knownllms section first", l.Model)
		}
	}

	if opts.Remote == "" {
		llamasrv := ""
		isLlamafile := false
		modelFile := ""
		if opts.Model == "python" {
			l.backend = "python"
		} else {
			// Make sure the server is available.
			if llamasrv, isLlamafile, err = getLlama(ctx, cache); err != nil {
				return nil, fmt.Errorf("failed to load llm: %w", err)
			}
			if l.backend = "llama-server"; isLlamafile {
				l.backend = "llamafile"
			}
			cmd := mangleForLlamafile(isLlamafile, llamasrv, "--version")
			c := exec.CommandContext(ctx, cmd[0], cmd[1:]...)
			c.Dir = cache
			d, err2 := c.CombinedOutput()
			if err2 != nil {
				return nil, fmt.Errorf("failed to get llm version: %w\n%s", err2, string(d))
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
			pysvr, err2 := py.NewServer(ctx, "llm.py", filepath.Join(cache, "py"), filepath.Join(cache, "py_llm.log"), []string{"--port", strconv.Itoa(port)})
			if err2 != nil {
				return nil, err2
			}
			l.done = pysvr.Done
			l.cancel = pysvr.Cmd.Cancel
		} else {
			done := make(chan error)
			l.done = done
			log, err2 := os.OpenFile(filepath.Join(cache, "llm.log"), os.O_WRONLY|os.O_APPEND|os.O_CREATE, 0o644)
			if err2 != nil {
				return nil, fmt.Errorf("failed to create llm server log file: %w", err2)
			}
			defer log.Close()
			// Surprisingly llama-server seems to be hardcoded to 8 threads. Leave 2
			// cores (especially critical when HT) to allow us to get some CPU time.
			// TODO: we should probably nice it a bit.
			threads := runtime.NumCPU() - 2
			if threads == 0 {
				threads = 1
			}
			// TODO: Investigate using -fa.
			// TODO: Doesn't seem to have any effect, need investigation.
			// "--prompt-cache", filepath.Join(cache, "llm-prompt-cache.bin"), "--prompt-cache-all",
			common := []string{
				llamasrv, "--model", modelFile, "--metrics", "-ngl", "9999", "--threads", strconv.Itoa(threads), "--port", strconv.Itoa(port),
			}
			// Limit the context window for now.
			if opts.ContextLength != 0 {
				common = append(common, "--ctx-size", strconv.Itoa(opts.ContextLength))
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
			go l.waitForTerminated(done)
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
	if l.backend == "python" {
		l.cp = &py.CompletionProvider{URL: l.baseURL}
	} else {
		l.cp, err = llamacpp.New(l.baseURL, l.Encoding)
		if err != nil {
			return nil, err
		}
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
		s, ok := er.Sys().(syscall.WaitStatus)
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
	// TODO: Generalize.
	c, err := llamacpp.New(l.baseURL, nil)
	if err != nil {
		return "", err
	}
	return c.GetHealth(ctx)
}

// GetMetrics retrieves the performance statistics from the server.
func (l *Session) GetMetrics(ctx context.Context, m *llamacpp.Metrics) error {
	// TODO: Generalize.
	c, err := llamacpp.New(l.baseURL, nil)
	if err != nil {
		return err
	}
	return c.GetMetrics(ctx, m)
}

// Prompt prompts the LLM and returns the reply.
//
// See PromptStreaming for the arguments values.
//
// The first message is assumed to be the system prompt.
func (l *Session) Prompt(ctx context.Context, msgs []genaiapi.Message, opts any) (string, error) {
	r := trace.StartRegion(ctx, "llm.Prompt")
	defer r.End()
	if len(msgs) == 0 {
		return "", errors.New("input required")
	}
	start := time.Now()
	msgs = l.processMsgs(msgs)
	slog.Info("llm", "num_msgs", len(msgs), "msg", msgs[len(msgs)-1], "type", "blocking")
	msg, err := l.cp.Completion(ctx, msgs, opts)
	if err != nil {
		slog.Error("llm", "msgs", msgs, "error", err, "duration", time.Since(start).Round(time.Millisecond))
		return "", err
	}
	reply := msg.Text
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
func (l *Session) PromptStreaming(ctx context.Context, msgs []genaiapi.Message, opts any, words chan<- string) error {
	r := trace.StartRegion(ctx, "llm.PromptStreaming")
	defer r.End()
	if len(msgs) == 0 {
		return errors.New("input required")
	}
	start := time.Now()
	msgs = l.processMsgs(msgs)
	slog.Info("llm", "num_msgs", len(msgs), "msg", msgs[len(msgs)-1], "type", "streaming")
	err := l.cp.CompletionStream(ctx, msgs, opts, words)
	if err != nil {
		slog.Error("llm", "error", err, "duration", time.Since(start).Round(time.Millisecond))
		return err
	}
	slog.Info("llm", "duration", time.Since(start).Round(time.Millisecond))
	return nil
}

//

func (l *Session) waitForTerminated(done chan<- error) {
	done <- l.c.Wait()
	slog.Info("llm", "state", "terminated")
}

// ensureModel gets the model if missing.
//
// Currently hard-coded to GGUF files and Hugging Face. Doesn't support split
// files.
func (l *Session) ensureModel(ctx context.Context, model PackedFileRef, k KnownLLM) (string, error) {
	// TODO: This is very "meh".
	// Designed to handle special case like Mistral-7B-Instruct-v0.3-Q3_K_M.
	ext := strings.ToUpper(model.Basename())
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
	case "Q8_0":
	case "Q6_K_L", "Q6_K":
	case "Q5_K_L", "Q5_K_M", "Q5_K_S", "Q5_1", "Q5_0":
	case "Q4_K_L", "Q4_K_M", "Q4_K_S", "Q4_K", "Q4_1", "Q4_0":
	case "Q3_K_XL", "Q3_K_L", "Q3_K_M", "Q3_K_S":
	case "Q2_K_L", "Q2_K":
	case "IQ4_NL", "IQ4_XS":
	case "IQ3_M", "IQ3_S", "IQ3_XS", "IQ3_XXS":
	case "IQ2_M", "IQ2_S", "IQ2_XS", "IQ2_XXS":
	case "IQ1_M", "IQ1_S":

	case "":
		return "", fmt.Errorf("you forgot to add a quantization suffix like 'BF16', 'F16', 'Q8_0' or 'Q5_K_M' when specifying model %q", model)

	default:
		return "", fmt.Errorf("unknown quantization %q for model %q, did you forget a suffix like 'BF16' or 'Q5_K_M'?", ext, model)
	}

	// Hack: quickly check if the file is there, if so, just return this.
	dst := filepath.Join(l.cache, model.Basename()+".gguf")
	_, err := os.Stat(dst)
	if err == nil {
		l.modelFile = dst
		return dst, nil
	}
	slog.Info("llm", "model", model, "state", "missing")
	if k.Source.RepoID() == "" {
		return "", fmt.Errorf("can't guess model %q huggingface repo", model)
	}
	// Hack: we assume everything is on HuggingFace.
	ln := filepath.Join(l.cache, model.Basename()+".gguf")
	switch k.PackagingType {
	case "gguf":
		if dst, err = l.HF.EnsureFile(ctx, model.ModelRef(), model.Revision(), model.Basename()+".gguf"); err != nil {
			// Get the list of files to help the user.
			m := huggingface.Model{ModelRef: model.ModelRef()}
			err = fmt.Errorf("can't find model %q at %s: %w", model, m.URL(), err)
			if err2 := l.HF.GetModelInfo(ctx, &m, model.Revision()); err2 != nil {
				return ln, errors.Join(err, err2)
			}
			msg := "Supported quantizations: "
			added := false
			for _, f := range m.Files {
				// TODO: Move this into a common function.
				if !strings.HasPrefix(f, k.Source.Basename()) {
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
				msg += strings.TrimSuffix(f[len(model.Basename()):], ".gguf")
				added = true
			}
			return ln, fmt.Errorf("%w; %s", err, msg)
		}
		// TODO: When I use os.Symlink(), llama-server crashes.
		if err = os.Symlink(dst, ln); err != nil {
			return ln, err
		}
		l.modelFile = ln
		return ln, nil
	default:
		return ln, fmt.Errorf("internal error: implement packaging type %s", k.PackagingType)
	}
}

// processMsgs process the system prompt.
func (l *Session) processMsgs(msgs []genaiapi.Message) []genaiapi.Message {
	if len(msgs) == 0 || msgs[0].Role != genaiapi.System {
		return msgs
	}
	t, err := template.New("").Parse(msgs[0].Text)
	if err != nil {
		slog.Error("llm", "message", "invalid system prompt", "system_prompt", msgs[0].Text, "error", err)
		return msgs
	}

	keys := map[string]string{
		"Now":   time.Now().Format("Monday 2006-01-02T15:04:05 MST"),
		"Model": string(l.Model),
	}
	b := bytes.Buffer{}
	if err = t.Execute(&b, keys); err != nil {
		slog.Error("llm", "message", "invalid system prompt", "system_prompt", msgs[0].Text, "error", err)
		return msgs
	}
	out := make([]genaiapi.Message, len(msgs))
	copy(out, msgs)
	out[0].Text = b.String()
	return out
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
	llamaserver, err := llamacppsrv.DownloadRelease(ctx, cache, 4856)
	return llamaserver, false, err
}
