// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package main

import (
	"bytes"
	"context"
	"fmt"
	"net"
	"strings"
	"time"

	"github.com/bwmarrin/discordgo"
)

func findFreePort() int {
	l, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		panic(err)
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port
}

type bot struct {
	l *llm
	s *stableDiffusion
}

func newBot(ctx context.Context, cache string, dg *discordgo.Session, llm string, sd bool) (*bot, error) {
	start := time.Now()
	logger.Info("bot", "state", "initializing")
	var err error
	b := &bot{}
	if llm != "" {
		if b.l, err = newLLM(ctx, cache, llm); err != nil {
			b.Close()
			logger.Info("bot", "state", "failed", "err", err, "duration", time.Since(start).Round(time.Millisecond))
			return nil, err
		}
	}
	if sd {
		if b.s, err = newStableDiffusion(ctx, cache); err != nil {
			b.Close()
			logger.Info("bot", "state", "failed", "err", err, "duration", time.Since(start).Round(time.Millisecond))
			return nil, err
		}
	}
	_ = dg.AddHandler(b.guildCreate)
	_ = dg.AddHandler(b.messageCreate)
	_ = dg.AddHandler(b.ready)
	dg.Identify.Intents = discordgo.IntentsGuilds | discordgo.IntentsGuildMessages | discordgo.IntentGuildPresences | discordgo.IntentDirectMessages
	logger.Info("bot", "state", "ready", "duration", time.Since(start).Round(time.Millisecond))
	return b, nil
}

func (b *bot) Close() error {
	var err error
	if b.l != nil {
		if err2 := b.l.Close(); err == nil {
			err = err2
		}
	}
	if b.s != nil {
		if err2 := b.s.Close(); err == nil {
			err = err2
		}
	}
	return err
}

// A new message is created on any channel that the authenticated bot has
// access to.
func (b *bot) messageCreate(s *discordgo.Session, m *discordgo.MessageCreate) {
	logger.Debug("discord", "event", "messageCreate", "message", m.Message, "state", s.State)
	// Ignore all messages created by the bot itself. This isn't required in this
	// specific example but it's a good practice.
	botid := s.State.User.ID
	if m.Author.ID == botid {
		return
	}
	content := strings.TrimSpace(strings.TrimPrefix(m.Content, fmt.Sprintf("<@%s>", botid)))
	logger.Info("discord", "event", "messageCreate", "author", m.Author.Username, "message", content)
	var err error
	switch content {
	case "ping":
		_, err = s.ChannelMessageSend(m.ChannelID, "Pong!")
	case "pong":
		_, err = s.ChannelMessageSend(m.ChannelID, "Ping!")
	default:
		if b.l != nil {
			reply := ""
			if reply, err = b.l.prompt(content); err == nil {
				_, err = s.ChannelMessageSend(m.ChannelID, reply)
			} else {
				_, _ = s.ChannelMessageSend(m.ChannelID, err.Error())
			}
		}
		if b.s != nil && err == nil {
			// TODO: insert a stand-in, then replace it.
			var p []byte
			if p, err = b.s.genImage(content); err == nil {
				data := discordgo.MessageSend{
					Files: []*discordgo.File{
						{
							Name:        "prompt.png",
							ContentType: "image/png",
							Reader:      bytes.NewReader(p),
						},
					},
				}
				_, err = s.ChannelMessageSendComplex(m.ChannelID, &data)
			} else {
				_, _ = s.ChannelMessageSend(m.ChannelID, err.Error())
			}
		}
	}
	if err != nil {
		// If an error occurred, we failed to send the message. It may occur either
		// when we do not share a server with the user (highly unlikely as we just
		// received a message) or the user disabled DM in their settings (more
		// likely).
		logger.Error("discord", "message", content, "error", err)
	}
}

// A new guild is joined.
func (b *bot) guildCreate(s *discordgo.Session, event *discordgo.GuildCreate) {
	logger.Debug("discord", "event", "guildCreate", "event", event.Guild)
	logger.Info("discord", "event", "guildCreate", "name", event.Guild.Name)
	if event.Guild.Unavailable {
		return
	}
	for _, channel := range event.Guild.Channels {
		if channel.ID == event.Guild.ID {
			_, _ = s.ChannelMessageSend(channel.ID, "Coucou!")
			return
		}
	}
}

func (b *bot) ready(s *discordgo.Session, r *discordgo.Ready) {
	logger.Debug("discord", "event", "ready", "session", s, "event", r)
	logger.Info("discord", "event", "ready", "user", r.User.String())
}
