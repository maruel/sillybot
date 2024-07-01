// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package main

import (
	"context"
	"path/filepath"
	"testing"
)

func TestSD(t *testing.T) {
	cache, err := filepath.Abs("cache")
	if err != nil {
		t.Fatal(err)
	}
	s, err := newStableDiffusion(context.Background(), cache)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := s.Close(); err != nil {
			t.Error(err)
		}
	})
	got, err := s.genImage("cat")
	if err != nil {
		t.Fatal(err)
	}
	if len(got) < 1000 {
		t.Fatal("uh")
	}
}
