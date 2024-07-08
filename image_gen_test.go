// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package sillybot

import (
	"context"
	"path/filepath"
	"testing"
)

func TestImageGen(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test case in short mode")
	}
	cache, err := filepath.Abs("cache")
	if err != nil {
		t.Fatal(err)
	}
	opts := ImageGenOptions{Model: "python"}
	s, err := NewImageGen(context.Background(), cache, &opts)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err2 := s.Close(); err2 != nil {
			t.Error(err2)
		}
	})
	got, err := s.GenImage("cat")
	if err != nil {
		t.Fatal(err)
	}
	if len(got) < 1000 {
		t.Fatal("uh")
	}
}
