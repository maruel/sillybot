package main

import (
	// Fake import so go mod tidy doesn't strip it. It's only used in gen.go
	// which is not built by default.
	"testing"

	_ "github.com/schollz/progressbar/v3"
)

func TestMain(t *testing.T) {
	// TODO.
}
