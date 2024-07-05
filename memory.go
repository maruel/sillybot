// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package sillybot

import (
	"slices"
	"sync"
	"time"
)

// Conversation is a conversation with one user.
type Conversation struct {
	User       string
	Channel    string
	Started    time.Time
	LastUpdate time.Time
	Messages   []Message

	_ struct{}
}

// Memory holds the bot's conversations.
type Memory struct {
	mu            sync.Mutex
	conversations []*Conversation
}

// Get gets a previous conversations or returns a new one if it's a new
// conversation.
func (m *Memory) Get(user, channel string) *Conversation {
	// TODO: Keep a map instead of a silly linear search once we get 100s of
	// conversations, or just sort based on LRUs.
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, c := range m.conversations {
		if c.User == user && c.Channel == channel {
			c.LastUpdate = time.Now()
			return c
		}
	}
	now := time.Now()
	c := &Conversation{User: user, Channel: channel, Started: now, LastUpdate: now}
	m.conversations = append(m.conversations, c)
	return c
}

// Forget forgets old conversations.
func (m *Memory) Forget() {
	m.mu.Lock()
	defer m.mu.Unlock()
	// First sort then cut off. This is so much faster than complex structures
	// like a heap.
	slices.SortFunc(m.conversations, func(a, b *Conversation) int {
		return -1 * a.LastUpdate.Compare(b.LastUpdate)
	})
	cutoff := time.Now().Add(-24 * time.Hour)
	// If we get into 100s of conversations, use a binary search.
	for i, c := range m.conversations {
		if c.LastUpdate.Before(cutoff) {
			m.conversations = m.conversations[:i]
			break
		}
	}
}
