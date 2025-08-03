// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package llm runs a LLM locally via llama.cpp, llamafile, or with a python
// server. It takes care of everything, including fetching gguf packed models
// from hugging face.
package llm

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"runtime/trace"
	"strconv"
	"strings"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/llamacpp"
	"github.com/maruel/genai/providers/llamacpp/llamacppsrv"
	"github.com/maruel/genai/providers/openaicompatible"
	"github.com/maruel/genaipy"
	"github.com/maruel/huggingface"
	"github.com/maruel/sillybot/internal"
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
	return nil
}

// Session runs a llama.cpp or llamafile server and runs queries on it.
//
// While it is expected that the model is an Instruct form, it is not a
// requirement.
type Session struct {
	HF      *huggingface.Client
	Model   PackedFileRef
	baseURL string
	backend string
	cp      genai.ProviderGen

	cache     string
	modelFile string
	srv       io.Closer
}

// New instantiates a llama.cpp or llamafile server, or optionally uses
// python instead.
func New(ctx context.Context, cache string, opts *Options) (*Session, error) {
	if err := opts.Validate(); err != nil {
		return nil, err
	}

	// Either:
	// - Connecting to a remote server
	// - Starting a local server
	// The server can be either:
	// - llama-server
	// - our custom python backend

	cacheModels := filepath.Join(cache, "models")
	if err := os.MkdirAll(cacheModels, 0o755); err != nil {
		return nil, fmt.Errorf("failed to create the directory to cache models: %w", err)
	}
	hf, err := huggingface.New("")
	if err != nil {
		return nil, err
	}
	l := &Session{HF: hf, Model: opts.Model, cache: cacheModels}

	var done <-chan error
	if opts.Remote == "" {
		// We need to start the server.
		port := internal.FindFreePort(8031)
		l.baseURL = "http://localhost:" + strconv.Itoa(port)

		if opts.Model == "python" {
			l.backend = "python"
			pyDir := filepath.Join(cache, "py")
			if err = os.MkdirAll(pyDir, 0o755); err != nil {
				return nil, err
			}
			srv, err2 := genaipy.NewServer(ctx, "llm.py", pyDir, filepath.Join(cache, "py_llm.log"), []string{"--port", strconv.Itoa(port)})
			if err2 != nil {
				return nil, err2
			}
			l.srv = srv
			done = srv.Done()
		} else {
			llamasrv := ""
			if llamasrv, err = getLlama(ctx, cache); err != nil {
				return nil, fmt.Errorf("failed to load llm: %w", err)
			}
			l.backend = "llama-server"
			// Make sure the model is available.
			modelFile := ""
			if modelFile, err = l.ensureModel(ctx, opts.Model); err != nil {
				return nil, fmt.Errorf("failed to get llm model: %w", err)
			}
			args := []string{"-ngl", "9999"}
			// Limit the context window for performance.
			if opts.ContextLength != 0 {
				args = append(args, "--ctx-size", strconv.Itoa(opts.ContextLength))
			}
			f, err2 := os.Create(filepath.Join(cache, "llm.log"))
			if err2 != nil {
				return nil, err2
			}
			hostPort := fmt.Sprintf("127.0.0.1:%d", port)
			srv, err2 := llamacppsrv.NewServer(ctx, llamasrv, modelFile, f, hostPort, 0, args)
			_ = f.Close()
			if err2 != nil {
				return nil, err2
			}
			l.srv = srv
			done = srv.Done()
		}
	} else {
		if !internal.IsHostPort(opts.Remote) {
			return nil, fmt.Errorf("invalid remote %q; use form 'host:port'", opts.Remote)
		}
		// TODO: Support other backends via genai.
		l.baseURL = "http://" + opts.Remote
		slog.Info("llm", "state", "loading")
		l.backend = "remote"
	}

	if l.backend == "python" {
		l.cp, err = openaicompatible.New(l.baseURL+"/v1/chat/completions", nil, "", nil)
	} else {
		l.cp, err = llamacpp.New(l.baseURL, nil, nil)
	}
	if err != nil {
		return nil, err
	}

	// Do a quick health check. Technically unnecessary when running llama-server
	// locally.
	for ctx.Err() == nil {
		if status, _ := l.GetHealth(ctx); status == "ok" {
			break
		}
		select {
		case err := <-done:
			l.srv = nil
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
	if l.srv != nil {
		return l.srv.Close()
	}
	return nil
}

// GetHealth retrieves the heath of the server.
func (l *Session) GetHealth(ctx context.Context) (string, error) {
	// TODO: Generalize.
	c, err := llamacpp.New(l.baseURL, nil, nil)
	if err != nil {
		return "", err
	}
	return c.GetHealth(ctx)
}

// GetMetrics retrieves the performance statistics from the server.
func (l *Session) GetMetrics(ctx context.Context, m *llamacpp.Metrics) error {
	// TODO: Generalize.
	c, err := llamacpp.New(l.baseURL, nil, nil)
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
func (l *Session) Prompt(ctx context.Context, msgs []genai.Message, opts genai.Options) (string, error) {
	r := trace.StartRegion(ctx, "llm.Prompt")
	defer r.End()
	if len(msgs) == 0 {
		return "", errors.New("input required")
	}
	start := time.Now()
	slog.Info("llm", "num_msgs", len(msgs), "msg", msgs[len(msgs)-1], "type", "blocking")
	result, err := l.cp.GenSync(ctx, msgs, opts)
	if _, ok := err.(*genai.UnsupportedContinuableError); ok {
		err = nil
	}
	if err != nil {
		slog.Error("llm", "msgs", msgs, "error", err, "duration", time.Since(start).Round(time.Millisecond))
		return "", err
	}
	reply := result.AsText()
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
func (l *Session) PromptStreaming(ctx context.Context, msgs []genai.Message, chunks chan<- genai.ContentFragment, opts genai.Options) error {
	r := trace.StartRegion(ctx, "llm.PromptStreaming")
	defer r.End()
	if len(msgs) == 0 {
		return errors.New("input required")
	}
	start := time.Now()
	slog.Info("llm", "num_msgs", len(msgs), "msg", msgs[len(msgs)-1], "type", "streaming")
	result, err := l.cp.GenStream(ctx, msgs, chunks, opts)
	if _, ok := err.(*genai.UnsupportedContinuableError); ok {
		err = nil
	}
	if err != nil {
		slog.Error("llm", "error", err, "duration", time.Since(start).Round(time.Millisecond))
		return err
	}
	slog.Info("llm", "duration", time.Since(start).Round(time.Millisecond), "usage", result)
	return nil
}

//

// ensureModel gets the model if missing.
//
// Currently hard-coded to GGUF files and Hugging Face. Doesn't support split
// files.
func (l *Session) ensureModel(ctx context.Context, model PackedFileRef) (string, error) {
	slog.Info("llm", "model", model, "state", "missing")
	dst, err := getModelPath(model)
	if err != nil {
		return "", err
	}
	if dst != "" {
		// Hack: quickly check if the file is there, if so, just return this.
		dst = filepath.Join(l.cache, dst)
		if _, err = os.Stat(dst); err == nil {
			l.modelFile = dst
			return dst, nil
		}
	}
	// Hack: we assume everything is on HuggingFace and is gguf.
	ln := filepath.Join(l.cache, model.Basename()+".gguf")
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
}

// Tools

func getModelPath(model PackedFileRef) (string, error) {
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
	return model.Basename() + ".gguf", nil
}

// getLlama returns the file path to llama.cpp/llamafile executable.
//
// It first look for llama-server or llamafile if one of them is PATH. Then it
// checks if one of them is s in the cache directory, otherwise downloads an
// hard coded version of llama-server from GitHub.
func getLlama(ctx context.Context, cache string) (string, error) {
	if s, err := exec.LookPath("llama-server"); err == nil {
		return s, nil
	}
	execSuffix := ""
	if runtime.GOOS == "windows" {
		execSuffix = ".exe"
	}
	llamaserver := filepath.Join(cache, "llama-server"+execSuffix)
	if _, err := os.Stat(llamaserver); err == nil {
		return llamaserver, nil
	}
	llamaserver, err := llamacppsrv.DownloadRelease(ctx, cache, llamacppsrv.BuildNumber)
	return llamaserver, err
}
