// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package sillybot

import (
	"bytes"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
)

func TestMemory_Forget(t *testing.T) {
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

func TestMemory_Serialize(t *testing.T) {
	m1 := Memory{}
	now := time.Now()
	twodaysago := now.Add(-48 * time.Hour)
	m1.Get("user1", "channel1")
	c2 := m1.Get("user2", "channel1")
	m1.Get("user1", "channel2")
	c4 := m1.Get("user2", "channel2")
	c2.LastUpdate = twodaysago
	c4.LastUpdate = twodaysago

	b := bytes.Buffer{}
	if err := m1.Save(&b); err != nil {
		t.Fatal(err)
	}

	m2 := Memory{}
	if err := m2.Load(&b); err != nil {
		t.Fatal(err)
	}

	if diff := cmp.Diff(m1, m2, cmpopts.IgnoreUnexported(Memory{})); diff != "" {
		t.Fatal(diff)
	}
}
