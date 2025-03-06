// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package llm

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"slices"
	"sync"
	"time"

	"github.com/maruel/sillybot/llm/common"
)

// Conversation is a conversation with one user.
type Conversation struct {
	User       string
	Channel    string
	Started    time.Time
	LastUpdate time.Time
	Messages   []common.Message

	_ struct{}
}

// Memory holds the bot's conversations.
type Memory struct {
	mu            sync.Mutex
	conversations []*Conversation
}

// Load loads previous memory.
func (m *Memory) Load(r io.Reader) error {
	d := json.NewDecoder(r)
	d.DisallowUnknownFields()
	s := serializedMemory{}
	if err := d.Decode(&s); err != nil {
		slog.Error("memory", "action", "load", "error", err)
		return err
	}
	m.mu.Lock()
	err := s.to(m)
	l := len(m.conversations)
	m.mu.Unlock()
	if err != nil {
		slog.Error("memory", "action", "load", "error", err)
	}
	if len(s.Conversations) != l {
		return errors.New("internal error")
	}
	slog.Info("memory", "action", "load", "conversations", l)
	m.Forget()
	return nil
}

// Save saves the memory for later reuse.
func (m *Memory) Save(w io.Writer) error {
	m.Forget()
	s := serializedMemory{}
	m.mu.Lock()
	err := s.from(m)
	l := len(m.conversations)
	m.mu.Unlock()
	if err != nil {
		slog.Error("memory", "action", "save", "error", err)
		return err
	}
	e := json.NewEncoder(w)
	if err = e.Encode(s); err != nil {
		slog.Error("memory", "action", "save", "error", err)
		return err
	}
	if len(s.Conversations) != l {
		return errors.New("internal error")
	}
	slog.Info("memory", "action", "save", "conversations", l)
	return nil
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
	// First sort then cut off. This is so much faster than complex structures
	// like a heap.
	slices.SortFunc(m.conversations, func(a, b *Conversation) int {
		return -1 * a.LastUpdate.Compare(b.LastUpdate)
	})
	before := len(m.conversations)
	cutoff := time.Now().Add(-24 * time.Hour)
	// If we get into 100s of conversations, use a binary search.
	for i, c := range m.conversations {
		if c.LastUpdate.Before(cutoff) {
			m.conversations = m.conversations[:i]
			break
		}
	}
	after := len(m.conversations)
	m.mu.Unlock()
	slog.Info("memory", "action", "forget", "before", before, "after", after)
}

//

// serializedMemory is the JSON serialized version of Memory.
//
// It is quite inefficient. Should be fixed later.
type serializedMemory struct {
	Version       int                      `json:"v,omitempty"`
	Conversations []serializedConversation `json:"c,omitempty"`
}

func (s *serializedMemory) from(m *Memory) error {
	s.Version = 1
	s.Conversations = make([]serializedConversation, len(m.conversations))
	for i, c := range m.conversations {
		if err := s.Conversations[i].from(c); err != nil {
			return err
		}
	}
	return nil
}

func (s *serializedMemory) to(m *Memory) error {
	if s.Version != 1 {
		return fmt.Errorf("can't load unknown version %d", s.Version)
	}
	m.conversations = make([]*Conversation, len(s.Conversations))
	for i := range s.Conversations {
		c := &Conversation{}
		if err := s.Conversations[i].to(c); err != nil {
			return err
		}
		m.conversations[i] = c
	}
	return nil
}

type serializedConversation struct {
	User       string              `json:"u,omitempty"`
	Channel    string              `json:"c,omitempty"`
	Started    time.Time           `json:"s,omitempty"`
	LastUpdate time.Time           `json:"l,omitempty"`
	Messages   []serializedMessage `json:"m,omitempty"`
}

func (s *serializedConversation) from(c *Conversation) error {
	s.User = c.User
	s.Channel = c.Channel
	s.Started = c.Started
	s.LastUpdate = c.LastUpdate
	s.Messages = make([]serializedMessage, len(c.Messages))
	for i := range c.Messages {
		if err := s.Messages[i].from(&c.Messages[i]); err != nil {
			return err
		}
	}
	return nil
}

func (s *serializedConversation) to(c *Conversation) error {
	c.User = s.User
	c.Channel = s.Channel
	c.Started = s.Started
	c.LastUpdate = s.LastUpdate
	c.Messages = make([]common.Message, len(s.Messages))
	for i := range s.Messages {
		if err := s.Messages[i].to(&c.Messages[i]); err != nil {
			return err
		}
	}
	return nil
}

type serializedMessage struct {
	Role    int    `json:"r,omitempty"`
	Content string `json:"c,omitempty"`
}

func (s *serializedMessage) from(m *common.Message) error {
	switch m.Role {
	case common.System:
		s.Role = 0
	case common.User:
		s.Role = 1
	case common.Assistant:
		s.Role = 2
	case common.AvailableTools:
		s.Role = 3
	case common.ToolCall:
		s.Role = 4
	case common.ToolCallResult:
		s.Role = 5
	default:
		return fmt.Errorf("unknown role %q", m.Role)
	}
	s.Content = m.Content
	return nil
}

func (s *serializedMessage) to(m *common.Message) error {
	switch s.Role {
	case 0:
		m.Role = common.System
	case 1:
		m.Role = common.User
	case 2:
		m.Role = common.Assistant
	case 3:
		m.Role = common.AvailableTools
	case 4:
		m.Role = common.ToolCall
	case 5:
		m.Role = common.ToolCallResult
	default:
		return fmt.Errorf("unknown role %q", s.Role)
	}
	m.Content = s.Content
	return nil
}
