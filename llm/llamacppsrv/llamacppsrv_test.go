// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package llamacppsrv

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/maruel/sillybot/internal"
)

func TestNewServer(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	cache := t.TempDir()
	// It's a bit inefficient to download from github every single time.
	exe, err := DownloadRelease(ctx, cache, 4882)
	if err != nil {
		t.Fatal(err)
	}
	port := internal.FindFreePort(10000)
	modelPath, err := filepath.Abs(filepath.Join("testdata", "dummy.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	srv, err := NewServer(ctx, exe, modelPath, cache, port, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err2 := srv.Close(); err2 != nil {
			t.Error(err2)
		}
	})
	// Do something.
}
