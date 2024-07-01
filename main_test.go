// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package main

import (
	"testing"

	// Fake import so go mod tidy doesn't strip it. It's only used in gen.go
	// which is not built by default.
	_ "github.com/schollz/progressbar/v3"
)

func TestMain(t *testing.T) {
	// TODO.
}
