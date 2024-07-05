// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package main

import (
	"bytes"
	"context"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/bwmarrin/discordgo"
	"github.com/maruel/sillybot"
)

type discordBot struct {
	ctx          context.Context
	dg           *discordgo.Session
	l            *sillybot.LLM
	ig           *sillybot.ImageGen
	mem          *sillybot.Memory
	systemPrompt string
}

// newDiscordBot opens a websocket connection to Discord and begin listening.
func newDiscordBot(ctx context.Context, token string, verbose bool, l *sillybot.LLM, ig *sillybot.ImageGen, mem *sillybot.Memory) (*discordBot, error) {
	discordgo.Logger = func(msgL, caller int, format string, a ...interface{}) {
		msg := fmt.Sprintf(format, a...)
		switch msgL {
		case discordgo.LogDebug:
			slog.Debug(msg)
		case discordgo.LogInformational:
			slog.Info(msg)
		case discordgo.LogWarning:
			slog.Warn(msg)
		case discordgo.LogError:
			slog.Error(msg)
		}
	}
	dg, err := discordgo.New("Bot " + token)
	if err != nil {
		return nil, err
	}
	if verbose {
		// It's very verbose.
		//dg.LogLevel = discordgo.LogDebug
	}
	if err = dg.Open(); err != nil {
		dg.Close()
		return nil, err
	}
	d := &discordBot{
		ctx:          ctx,
		dg:           dg,
		l:            l,
		ig:           ig,
		mem:          mem,
		systemPrompt: "You are a terse assistant. You reply with short answers. You are often joyful, sometimes humorous, sometimes sarcastic.",
	}
	_ = dg.AddHandler(d.guildCreate)
	_ = dg.AddHandler(d.messageCreate)
	_ = dg.AddHandler(d.ready)
	//dg.Identify.Intents = discordgo.IntentsAllWithoutPrivileged
	dg.Identify.Intents = discordgo.IntentsGuilds | discordgo.IntentsGuildMessages | discordgo.IntentGuildPresences | discordgo.IntentDirectMessages | discordgo.IntentGuildMessageTyping | discordgo.IntentDirectMessageTyping
	slog.Info("discord", "state", "running", "info", "Press CTRL-C to exit.")
	return d, nil
}

func (d *discordBot) Close() error {
	slog.Info("discord", "state", "terminating")
	return d.dg.Close()
}

// A new message is created on any channel that the authenticated bot has
// access to.
func (d *discordBot) messageCreate(s *discordgo.Session, m *discordgo.MessageCreate) {
	slog.Debug("discord", "event", "messageCreate", "message", m.Message, "state", s.State)
	botid := s.State.User.ID
	if m.Author.ID == botid {
		return
	}
	user := fmt.Sprintf("<@%s>", botid)
	if !strings.Contains(m.Content, user) {
		// Ignore if the bot is not explicitly referenced to.
		return
	}
	msg := strings.TrimSpace(strings.ReplaceAll(m.Content, user, ""))
	slog.Info("discord", "event", "messageCreate", "author", m.Author.Username, "message", msg)
	if strings.HasPrefix(msg, "image:") {
		d.handleImage(s, m, strings.TrimSpace(strings.TrimPrefix(msg, "image:")))
		return
	}
	d.handlePrompt(s, m, msg)
}

// handlePrompt uses the LLM to generate a response.
func (d *discordBot) handlePrompt(s *discordgo.Session, m *discordgo.MessageCreate, msg string) {
	if d.l == nil {
		if _, err := s.ChannelMessageSend(m.ChannelID, "LLM is not enabled."); err != nil {
			slog.Error("discord", "event", "failed posting message", "error", err)
		}
		return
	}

	// Immediately signal the user that the bot is preparing a reply.
	if err := s.ChannelTyping(m.ChannelID); err != nil {
		slog.Error("discord", "event", "failed posting 'user typing'", "error", err)
		// Continue anyway.
	}
	c := d.mem.Get(m.Author.ID, m.ChannelID)
	if len(c.Messages) == 0 {
		c.Messages = []sillybot.Message{{Role: sillybot.System, Content: d.systemPrompt}}
	}
	c.Messages = append(c.Messages, sillybot.Message{Role: sillybot.User, Content: msg})
	words := make(chan string, 10)
	wg := sync.WaitGroup{}
	wg.Add(1)
	go func() {
		t := time.NewTicker(3 * time.Second)
		var msg *discordgo.Message = m.Message
		var err error
		text := ""
		pending := ""
		for {
			select {
			case w, ok := <-words:
				if !ok {
					if pending != "" {
						text += pending
						msg, err = s.ChannelMessageSendComplex(m.ChannelID, &discordgo.MessageSend{
							Content:   pending,
							Reference: &discordgo.MessageReference{MessageID: msg.ID, ChannelID: msg.ChannelID, GuildID: msg.GuildID},
						})
						if err != nil {
							slog.Error("discord", "event", "failed posting message", "error", err)
						}
					}
					// Remember our own answer.
					c.Messages = append(c.Messages, sillybot.Message{Role: sillybot.Assistant, Content: text})
					t.Stop()
					wg.Done()
					return
				}
				pending += w
			case <-t.C:
				if pending != "" {
					text += pending
					msg, err = s.ChannelMessageSendComplex(m.ChannelID, &discordgo.MessageSend{
						Content:   pending + " (...continued)",
						Reference: &discordgo.MessageReference{MessageID: msg.ID, ChannelID: msg.ChannelID, GuildID: msg.GuildID},
					})
					pending = ""
					if err != nil {
						slog.Error("discord", "event", "failed posting message", "error", err)
					}
					if err := s.ChannelTyping(m.ChannelID); err != nil {
						slog.Error("discord", "event", "failed posting 'user typing'", "error", err)
						// Continue anyway.
					}
				}
			}
		}
	}()
	err := d.l.PromptStreaming(d.ctx, c.Messages, words)
	close(words)
	wg.Wait()

	if err != nil {
		if _, err = s.ChannelMessageSend(m.ChannelID, "Prompt generation failed: "+err.Error()); err != nil {
			slog.Error("discord", "event", "failed posting message", "error", err)
		}
	}
}

// handleImage generates an image based on the user prompt.
func (d *discordBot) handleImage(s *discordgo.Session, m *discordgo.MessageCreate, msg string) {
	if d.ig == nil {
		if _, err := s.ChannelMessageSend(m.ChannelID, "Image generation is not enabled. Restart with flag \"-ig\""); err != nil {
			slog.Error("discord", "event", "failed posting message", "error", err)
		}
		return
	}

	if err := s.ChannelTyping(m.ChannelID); err != nil {
		slog.Error("discord", "event", "failed posting 'user typing'", "error", err)
		// Continue anyway.
	}

	// TODO: Insert a stand-in, then replace it.
	// TODO: Generate multiple images.
	p, err := d.ig.GenImage(msg)
	if err != nil {
		if _, err := s.ChannelMessageSend(m.ChannelID, "Image generation failed: "+err.Error()); err != nil {
			if err != nil {
				slog.Error("discord", "event", "failed posting message", "error", err)
			}
			return
		}
	}

	data := discordgo.MessageSend{
		Files: []*discordgo.File{
			{
				Name:        "prompt.png",
				ContentType: "image/png",
				Reader:      bytes.NewReader(p),
			},
		},
	}
	if _, err = s.ChannelMessageSendComplex(m.ChannelID, &data); err != nil {
		slog.Error("discord", "event", "failed posting message", "error", err)
	}
}

// A new guild is joined.
func (d *discordBot) guildCreate(s *discordgo.Session, event *discordgo.GuildCreate) {
	slog.Debug("discord", "event", "guildCreate", "event", event.Guild)
	slog.Info("discord", "event", "guildCreate", "name", event.Guild.Name)
	if event.Guild.Unavailable {
		return
	}
	/*
		for _, channel := range event.Guild.Channels {
			if channel.ID == event.Guild.ID {
				_, _ = s.ChannelMessageSend(channel.ID, "Coucou!")
				return
			}
		}
	*/
}

func (d *discordBot) ready(s *discordgo.Session, r *discordgo.Ready) {
	slog.Debug("discord", "event", "ready", "session", s, "event", r)
	slog.Info("discord", "event", "ready", "user", r.User.String())
}
