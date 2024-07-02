// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package main

import (
	"bytes"
	"errors"
	"fmt"
	"net"
	"strings"

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

type discordBot struct {
	dg *discordgo.Session
	l  *llmServer
	s  *stableDiffusionServer
}

// newDiscordBot opens a websocket connection to Discord and begin listening.
func newDiscordBot(token string, verbose bool, l *llmServer, s *stableDiffusionServer) (*discordBot, error) {
	discordgo.Logger = func(msgL, caller int, format string, a ...interface{}) {
		msg := fmt.Sprintf(format, a...)
		switch msgL {
		case discordgo.LogDebug:
			logger.Debug(msg)
		case discordgo.LogInformational:
			logger.Info(msg)
		case discordgo.LogWarning:
			logger.Warn(msg)
		case discordgo.LogError:
			logger.Error(msg)
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
	d := &discordBot{dg: dg, l: l, s: s}
	_ = dg.AddHandler(d.guildCreate)
	_ = dg.AddHandler(d.messageCreate)
	_ = dg.AddHandler(d.ready)
	dg.Identify.Intents = discordgo.IntentsGuilds | discordgo.IntentsGuildMessages | discordgo.IntentGuildPresences | discordgo.IntentDirectMessages
	logger.Info("discord", "state", "running", "info", "Press CTRL-C to exit.")
	return d, nil
}

func (d *discordBot) Close() error {
	logger.Info("discord", "state", "terminating")
	return d.dg.Close()
}

// A new message is created on any channel that the authenticated bot has
// access to.
func (d *discordBot) messageCreate(s *discordgo.Session, m *discordgo.MessageCreate) {
	logger.Debug("discord", "event", "messageCreate", "message", m.Message, "state", s.State)
	// Ignore all messages created by the bot itself. This isn't required in this
	// specific example but it's a good practice.
	botid := s.State.User.ID
	if m.Author.ID == botid {
		return
	}
	user := fmt.Sprintf("<@%s>", botid)
	if !strings.HasPrefix(m.Content, user) {
		// Ignore.
		return
	}
	content := strings.TrimSpace(strings.TrimPrefix(m.Content, user))
	logger.Info("discord", "event", "messageCreate", "author", m.Author.Username, "message", content)
	var err error
	switch content {
	case "ping":
		_, err = s.ChannelMessageSend(m.ChannelID, "Pong!")
	case "pong":
		_, err = s.ChannelMessageSend(m.ChannelID, "Ping!")
	default:
		if strings.HasPrefix(content, "image:") {
			content := strings.TrimSpace(strings.TrimPrefix(content, "image:"))
			if d.s == nil {
				err = errors.New("image generation is not enabled")
			} else {
				// TODO: insert a stand-in, then replace it.
				var p []byte
				if p, err = d.s.genImage(content); err == nil {
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
				}
			}
		} else {
			if d.l == nil {
				err = errors.New("text generation is not enabled")
			} else {
				reply := ""
				if reply, err = d.l.prompt(content); err == nil {
					_, err = s.ChannelMessageSend(m.ChannelID, reply)
				}
			}
		}
		if err != nil {
			_, _ = s.ChannelMessageSend(m.ChannelID, "ERROR: "+err.Error())
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
func (d *discordBot) guildCreate(s *discordgo.Session, event *discordgo.GuildCreate) {
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

func (d *discordBot) ready(s *discordgo.Session, r *discordgo.Ready) {
	logger.Debug("discord", "event", "ready", "session", s, "event", r)
	logger.Info("discord", "event", "ready", "user", r.User.String())
}
