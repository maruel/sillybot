// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package llm

import "testing"

func TestPackedFileRef(t *testing.T) {
	p := PackedFileRef("hf:author/repo/HEAD/dir/file")
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
	if got := p.Basename(); got != "dir/file" {
		t.Fatal(got)
	}
	if got := p.Revision(); got != "HEAD" {
		t.Fatal(got)
	}
	if got := p.RepoURL(); got != "https://huggingface.co/author/repo" {
		t.Fatal(got)
	}
}
