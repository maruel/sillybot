// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package internal

import "testing"

func TestIsHostPort(t *testing.T) {
	if IsHostPort("a:1") {
		t.Fatal()
	}
	if !IsHostPort("aa.bb.ts.net:1") {
		t.Fatal()
	}
}
