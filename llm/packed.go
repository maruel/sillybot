// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package llm

import (
	"fmt"
	"strings"

	"github.com/maruel/huggingface"
)

// PackedFileRef is a packed reference to a file in an hugging face repository.
//
// The form is "hf:<author>/<repo>/HEAD/<file>"
//
// HEAD is the git commit reference or "revision". HEAD means the default
// branch. It can be replaced with a branch name or a commit hash. The default
// branch used by huggingface_hub official python library is "main".
//
// DEFAULT_REVISION in
// https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/constants.py
type PackedFileRef string

// MakePackedFileRef returns a PackedFileRef
func MakePackedFileRef(author, repo, revision, file string) PackedFileRef {
	return PackedFileRef("hf:" + author + "/" + repo + "/" + revision + "/" + file)
}

// RepoID returns the canonical "<author>/<repo>" for this repository.
func (p PackedFileRef) RepoID() string {
	s := string(p)
	if i := strings.IndexByte(s, '/'); i != -1 {
		if j := strings.IndexByte(s[i+1:], '/'); j != -1 {
			return strings.TrimPrefix(s[:i+j+1], "hf:")
		}
	}
	return ""
}

// Author returns the <author> part of the packed reference.
func (p PackedFileRef) Author() string {
	s := string(p)
	if i := strings.IndexByte(s, '/'); i != -1 {
		return strings.TrimPrefix(s[:i], "hf:")
	}
	return ""
}

// Repo returns the <repo> part of the packed reference.
func (p PackedFileRef) Repo() string {
	s := string(p)
	if i := strings.IndexByte(s, '/'); i != -1 {
		s = s[i+1:]
		if i = strings.IndexByte(s, '/'); i != -1 {
			return s[:i]
		}
	}
	return ""
}

// Revision returns the HEAD part of the packed reference.
func (p PackedFileRef) Revision() string {
	s := string(p)
	if i := strings.IndexByte(s, '/'); i != -1 {
		s = s[i+1:]
		if i = strings.IndexByte(s, '/'); i != -1 {
			s = s[i+1:]
			if i = strings.IndexByte(s, '/'); i != -1 {
				return s[:i]
			}
		}
	}
	return ""
}

// ModelRef returns the ModelRef reference to the repo containing this file.
func (p PackedFileRef) ModelRef() huggingface.ModelRef {
	return huggingface.ModelRef{Author: p.Author(), Repo: p.Repo()}
}

// Basename returns the basename part of this reference.
func (p PackedFileRef) Basename() string {
	s := string(p)
	if i := strings.IndexByte(s, '/'); i != -1 {
		s = s[i+1:]
		if i = strings.IndexByte(s, '/'); i != -1 {
			s = s[i+1:]
			if i = strings.IndexByte(s, '/'); i != -1 {
				return s[i+1:]
			}
		}
	}
	return ""
}

// RepoURL returns the canonical URL for this repository.
func (p PackedFileRef) RepoURL() string {
	return "https://huggingface.co/" + p.RepoID()
}

// Validate checks for obvious errors in the string.
func (p PackedFileRef) Validate() error {
	if !strings.HasPrefix(string(p), "hf:") {
		return fmt.Errorf("invalid file ref %q", p)
	}
	parts := strings.Split(string(p)[4:], "/")
	if len(parts) < 4 {
		return fmt.Errorf("invalid file ref %q", p)
	}
	if len(parts[2]) == 0 {
		return fmt.Errorf("invalid file ref %q", p)
	}
	for _, p := range parts {
		if len(p) < 3 {
			return fmt.Errorf("invalid file ref %q", p)
		}
	}
	return nil
}

// PackedRepoRef is a packed reference to an hugging face repository.
//
// The form is "hf:<author>/<repo>"
type PackedRepoRef string

// RepoID returns the canonical "<author>/<repo>" for this repository.
func (p PackedRepoRef) RepoID() string {
	if !strings.HasPrefix(string(p), "hf:") {
		return ""
	}
	return string(p[3:])
}

// ModelRef converts to a ModelRef reference.
func (p PackedRepoRef) ModelRef() huggingface.ModelRef {
	out := huggingface.ModelRef{}
	if parts := strings.SplitN(p.RepoID(), "/", 2); len(parts) == 2 {
		out.Author = parts[0]
		out.Repo = parts[1]
	}
	return out
}

// RepoURL returns the canonical URL for this repository.
func (p PackedRepoRef) RepoURL() string {
	return "https://huggingface.co/" + strings.TrimPrefix(string(p), "hf:")
}

// Validate checks for obvious errors in the string.
func (p PackedRepoRef) Validate() error {
	if strings.Count(string(p), "/") != 1 {
		return fmt.Errorf("invalid repo %q", p)
	}
	if !strings.HasPrefix(string(p), "hf:") {
		return fmt.Errorf("invalid repo %q", p)
	}
	parts := strings.Split(string(p)[3:], "/")
	if len(parts) != 2 {
		return fmt.Errorf("invalid repo %q", p)
	}
	for _, p := range parts {
		if len(p) < 3 {
			return fmt.Errorf("invalid repo %q", p)
		}
	}
	return nil
}
