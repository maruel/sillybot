// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package sillybot

import (
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
)

func TestMemory(t *testing.T) {
	m := Memory{}
	m.Forget()
	now := time.Now()
	twodaysago := now.Add(-48 * time.Hour)
	c1 := m.Get("user1", "channel1")
	c2 := m.Get("user2", "channel1")
	c3 := m.Get("user1", "channel2")
	c4 := m.Get("user2", "channel2")
	if len(m.conversations) != 4 {
		t.Fatal(len(m.conversations))
	}
	c2.LastUpdate = twodaysago
	c4.LastUpdate = twodaysago
	m.Forget()
	// LRU:
	want := []*Conversation{c3, c1}
	if diff := cmp.Diff(want, m.conversations); diff != "" {
		t.Fatal(diff)
	}
}
