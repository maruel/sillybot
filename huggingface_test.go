// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package sillybot

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
)

func TestList(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/models/microsoft/Phi-3-mini-4k-instruct/revision/main" {
			t.Errorf("unexpected path, got: %s", r.URL.Path)
		}
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(apiRepoPhi3Data))
	}))
	defer server.Close()
	h, err := newHuggingFace("tok", t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	h.serverBase = server.URL
	got, err := h.listRepo(context.Background(), "microsoft/Phi-3-mini-4k-instruct")
	if err != nil {
		t.Fatal(err)
	}
	want := modelInfo{
		files: []string{
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
		created:  time.Date(2024, 04, 22, 16, 18, 17, 0, time.UTC),
		modified: time.Date(2024, 07, 01, 21, 16, 50, 0000, time.UTC),
		tensor:   "BF16",
		size:     3821079552,
	}
	if diff := cmp.Diff(&want, got, cmp.AllowUnexported(want)); diff != "" {
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
