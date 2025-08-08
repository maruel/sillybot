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
	"strconv"
	"strings"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/llamacpp"
	"github.com/maruel/genai/providers/llamacpp/llamacppsrv"
	"github.com/maruel/genai/providers/openaicompatible"
	"github.com/maruel/genaipy"
	"github.com/maruel/huggingface"
)

// Options for NewLLM.
type Options struct {
	// Backend is the name of the backend to run. It can be "llama-server" or "python" for genaipy.
	Backend string `yaml:"backend"`
	// Model specifies a model to use.
	Model PackedFileRef `yaml:"model"`
	// ContextLength will set the context length when using a locally managed "llamacpp".
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

// Server runs a llamacpp-server or python.
type Server struct {
	HF      *huggingface.Client
	Model   PackedFileRef
	URL     string
	backend string
	cp      genai.ProviderGen

	cache     string
	modelFile string
	srv       io.Closer
}

// New instantiates a llamacpp-server or python.
func New(ctx context.Context, cache string, opts *Options) (*Server, error) {
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
	l := &Server{HF: hf, Model: opts.Model, backend: opts.Backend, cache: cacheModels}
	// We need to start the server.
	switch l.backend {
	case "python":
		if l.Model != "" {
			return nil, fmt.Errorf("backend \"python\" doesn't support explicit model; %q was specified", l.Model)
		}
		pyDir := filepath.Join(cache, "py")
		if err = os.MkdirAll(pyDir, 0o755); err != nil {
			return nil, err
		}
		srv, err2 := genaipy.NewServer(ctx, "llm.py", pyDir, filepath.Join(cache, "py_llm.log"), nil)
		if err2 != nil {
			return nil, err2
		}
		l.URL = srv.URL
		l.srv = srv
		l.cp, err = openaicompatible.New(&genai.OptionsProvider{Remote: l.URL + "/v1/chat/completions"}, nil)
		if err != nil {
			return nil, err
		}
		// Loop until it is available.
		time.Sleep(10 * time.Second)
	case "llama-server":
		llamasrv := ""
		if llamasrv, err = getLlama(ctx, cache); err != nil {
			return nil, fmt.Errorf("failed to load llm: %w", err)
		}
		// Make sure the model is available.
		modelFile := ""
		if modelFile, err = l.ensureModel(ctx, opts.Model); err != nil {
			return nil, fmt.Errorf("failed to get llm model: %w", err)
		}
		args := []string{"-ngl", "9999", "--jinja", "--flash-attn", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"}
		if opts.ContextLength != 0 {
			args = append(args, "--ctx-size", strconv.Itoa(opts.ContextLength))
		}
		f, err2 := os.Create(filepath.Join(cache, "llama-server.log"))
		if err2 != nil {
			return nil, err2
		}
		srv, err2 := llamacppsrv.New(ctx, llamasrv, modelFile, f, "localhost:0", 0, args)
		_ = f.Close()
		if err2 != nil {
			return nil, err2
		}
		l.srv = srv
		l.URL = srv.URL()
		l.cp, err = llamacpp.New(&genai.OptionsProvider{Remote: l.URL}, nil)
		if err != nil {
			return nil, err
		}
	default:
		return nil, fmt.Errorf("unknown backend %q", l.backend)
	}

	slog.Info("llm", "state", "ready", "model", opts.Model, "using", l.backend, "url", l.URL)
	return l, nil
}

func (l *Server) Close() error {
	slog.Info("llm", "state", "terminating")
	if l.srv != nil {
		return l.srv.Close()
	}
	return nil
}

func (l *Server) Client() genai.ProviderGen {
	return l.cp
}

//

// ensureModel gets the model if missing.
//
// Currently hard-coded to GGUF files and Hugging Face. Doesn't support split
// files.
func (l *Server) ensureModel(ctx context.Context, model PackedFileRef) (string, error) {
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

// getLlama returns the file path to llama.cpp executable.
//
// It first look for llama-server in PATH. Then it checks if it is in the cache directory, otherwise downloads
// an hard coded version of llama-server from GitHub.
func getLlama(ctx context.Context, cache string) (string, error) {
	if s, err := exec.LookPath("llama-server"); err == nil {
		return s, nil
	}
	p := filepath.Join(cache, "llama-server")
	if runtime.GOOS == "windows" {
		p += ".exe"
	}
	if _, err := os.Stat(p); err == nil {
		return p, nil
	}
	return llamacppsrv.DownloadRelease(ctx, cache, llamacppsrv.BuildNumber)
}
