// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package huggingface

import (
	"context"
	"flag"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/lmittmann/tint"
	"github.com/mattn/go-colorable"
	"github.com/mattn/go-isatty"
)

func TestPackedFileRef(t *testing.T) {
	p := PackedFileRef("hf:author/repo/HEAD/file")
	if err := p.Validate(); err != nil {
		t.Fatal(err)
	}
	if got := p.RepoID(); got != "author/repo" {
		t.Fatal(got)
	}
	if got := p.Author(); got != "author" {
		t.Fatal(got)
	}
	if got := p.Repo(); got != "repo" {
		t.Fatal(got)
	}
	if got := p.Basename(); got != "file" {
		t.Fatal(got)
	}
	if got := p.RepoURL(); got != "https://huggingface.co/author/repo" {
		t.Fatal(got)
	}
}

func TestGetModelInfo(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/models/microsoft/Phi-3-mini-4k-instruct/revision/HEAD" {
			t.Errorf("unexpected path, got: %s", r.URL.Path)
		}
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(apiRepoPhi3Data))
	}))
	defer server.Close()
	c, err := New("", t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	c.serverBase = server.URL

	got := Model{
		ModelRef: ModelRef{
			Author: "microsoft",
			Repo:   "Phi-3-mini-4k-instruct",
		},
	}
	if err := c.GetModelInfo(context.Background(), &got); err != nil {
		t.Fatal(err)
	}
	want := Model{
		ModelRef: ModelRef{
			Author: "microsoft",
			Repo:   "Phi-3-mini-4k-instruct",
		},
		Files: []string{
			".gitattributes",
			"CODE_OF_CONDUCT.md",
			"LICENSE",
			"NOTICE.md",
			"README.md",
			"SECURITY.md",
			"added_tokens.json",
			"config.json",
			"configuration_phi3.py",
			"generation_config.json",
			"model-00001-of-00002.safetensors",
			"model-00002-of-00002.safetensors",
			"model.safetensors.index.json",
			"modeling_phi3.py",
			"sample_finetune.py",
			"special_tokens_map.json",
			"tokenizer.json",
			"tokenizer.model",
			"tokenizer_config.json",
		},
		Created:    time.Date(2024, 04, 22, 16, 18, 17, 0, time.UTC),
		Modified:   time.Date(2024, 07, 01, 21, 16, 50, 0000, time.UTC),
		TensorType: "BF16",
		NumWeights: 3821079552,
		License:    "mit",
		LicenseURL: "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/resolve/main/LICENSE",
	}
	if diff := cmp.Diff(want, got, cmpopts.IgnoreUnexported(want)); diff != "" {
		t.Fatal(diff)
	}
}

var apiRepoPhi3Data = `
{
		"lastModified": "2024-07-01T21:16:50.000Z",
    "cardData": {
        "license": "mit",
        "license_link": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/resolve/main/LICENSE",
        "language": [
            "en"
        ],
        "inference": {
            "parameters": {
                "temperature": 0
            }
        }
    },
    "siblings": [
        {
            "rfilename": ".gitattributes"
        },
        {
            "rfilename": "CODE_OF_CONDUCT.md"
        },
        {
            "rfilename": "LICENSE"
        },
        {
            "rfilename": "NOTICE.md"
        },
        {
            "rfilename": "README.md"
        },
        {
            "rfilename": "SECURITY.md"
        },
        {
            "rfilename": "added_tokens.json"
        },
        {
            "rfilename": "config.json"
        },
        {
            "rfilename": "configuration_phi3.py"
        },
        {
            "rfilename": "generation_config.json"
        },
        {
            "rfilename": "model-00001-of-00002.safetensors"
        },
        {
            "rfilename": "model-00002-of-00002.safetensors"
        },
        {
            "rfilename": "model.safetensors.index.json"
        },
        {
            "rfilename": "modeling_phi3.py"
        },
        {
            "rfilename": "sample_finetune.py"
        },
        {
            "rfilename": "special_tokens_map.json"
        },
        {
            "rfilename": "tokenizer.json"
        },
        {
            "rfilename": "tokenizer.model"
        },
        {
            "rfilename": "tokenizer_config.json"
        }
    ],
    "createdAt": "2024-04-22T16:18:17.000Z",
    "safetensors": {
        "parameters": {
            "BF16": 3821079552
        },
        "total": 3821079552
    }
}
`

// TestMain sets up the verbose logging.
func TestMain(m *testing.M) {
	flag.Parse()
	l := slog.LevelWarn
	if testing.Verbose() {
		l = slog.LevelDebug
	}
	logger := slog.New(tint.NewHandler(colorable.NewColorable(os.Stderr), &tint.Options{
		Level:      l,
		TimeFormat: time.TimeOnly,
		NoColor:    !isatty.IsTerminal(os.Stderr.Fd()),
	}))
	slog.SetDefault(logger)
	os.Exit(m.Run())
}
