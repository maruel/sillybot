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

// discordBot is the live instance of the bot talking to the Discord API.
//
// Throughout the code, a Discord Server is called a "Guild". See
// https://discord.com/developers/docs/quick-start/overview-of-apps#where-are-apps-installed
type discordBot struct {
	ctx          context.Context
	dg           *discordgo.Session
	l            *sillybot.LLM
	ig           *sillybot.ImageGen
	mem          *sillybot.Memory
	systemPrompt string
	selfRef      string
	chat         chan msgReq
	image        chan msgReq
	wg           sync.WaitGroup
}

// newDiscordBot opens a websocket connection to Discord and begin listening.
func newDiscordBot(ctx context.Context, token string, verbose bool, l *sillybot.LLM, ig *sillybot.ImageGen, mem *sillybot.Memory, systPrmpt string) (*discordBot, error) {
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
		dg.LogLevel = discordgo.LogInformational
		// It's very verbose.
		//dg.LogLevel = discordgo.LogDebug
	}
	// We want to receive as few messages as possible.
	dg.Identify.Intents = discordgo.IntentsGuilds | discordgo.IntentsGuildMessages | discordgo.IntentDirectMessages
	d := &discordBot{
		ctx:          ctx,
		dg:           dg,
		l:            l,
		ig:           ig,
		mem:          mem,
		systemPrompt: systPrmpt,
		chat:         make(chan msgReq, 5),
		image:        make(chan msgReq, 3),
	}
	// The events are listed at
	// https://discord.com/developers/docs/topics/gateway-events#receive-events
	// Note that all messages are called asynchronously.
	_ = dg.AddHandler(d.onReady)
	_ = dg.AddHandler(d.onGuildCreate)
	_ = dg.AddHandler(d.onMessageCreate)
	_ = dg.AddHandler(d.onInteractionCreate)
	d.wg.Add(2)
	go d.chatRoutine()
	go d.imageRoutine()
	if err = dg.Open(); err != nil {
		_ = d.dg.Close()
		return nil, err
	}
	slog.Info("discord", "state", "running", "info", "Press CTRL-C to exit.")
	return d, nil
}

func (d *discordBot) Close() error {
	slog.Info("discord", "state", "terminating")
	// TODO: Send a bye bye before closing.
	// TODO: Set presence to "away". It's already the case for channels but not
	// for direct messages.
	err := d.dg.Close()
	d.chat <- msgReq{}
	d.image <- msgReq{}
	d.wg.Wait()
	return err
}

// Handlers

// onReady is received right after the initial handshake.
//
// It's the very first message. At this point, guilds are not yet available.
// See https://discord.com/developers/docs/topics/gateway-events#ready
func (d *discordBot) onReady(dg *discordgo.Session, r *discordgo.Ready) {
	slog.Debug("discord", "event", "ready", "session", dg, "event", r)
	slog.Info("discord", "event", "ready", "user", r.User.String())

	// TODO: Get list of DMs and tell users "I'm back up!"

	appid := r.Application.ID
	cmd, err := dg.ApplicationCommand(appid, "", "forget")
	if cmd == nil {
		slog.Info("discord", "registering", "forget")
		cmd = &discordgo.ApplicationCommand{
			ID:          "forget",
			Name:        "forget",
			Type:        discordgo.ChatApplicationCommand,
			Description: "Forget all the bot's memory of your conversation here with it.",
		}
		if cmd, err = dg.ApplicationCommandCreate(appid, "", cmd); err != nil {
			slog.Error("discord", "failed to register command: %w", err)
		}
		slog.Info("discord", "message", "registered command", "command", cmd)
	} else {
		slog.Info("discord", "message", "command forget was already registered")
	}
}

// onGuildCreate is received when new guild (server) is joined or becomes
// available right after connecting.
//
// https://discord.com/developers/docs/topics/gateway-events#guild-create
func (d *discordBot) onGuildCreate(dg *discordgo.Session, event *discordgo.GuildCreate) {
	slog.Debug("discord", "event", "guildCreate", "event", event.Guild)
	slog.Info("discord", "event", "guildCreate", "name", event.Guild.Name)
	if event.Guild.Unavailable {
		return
	}
	const welcome = "I'm back up!"
	for _, channel := range event.Guild.Channels {
		if t := channel.Type; t == discordgo.ChannelTypeGuildVoice || t == discordgo.ChannelTypeGuildCategory {
			continue
		}
		// Don't alert again if the last connection was recent, to not spam the
		// channel.
		msgs, err := dg.ChannelMessages(channel.ID, 5, "", "", "")
		if err != nil {
			slog.Error("discord", "error", err)
		}
		skip := false
		for _, msg := range msgs {
			if msg.Author.ID == dg.State.User.ID && msg.Content == welcome {
				slog.Info("discord", "message", "skipping welcome to not spam", "channel", channel.Name)
				skip = true
				break
			}
		}
		if !skip {
			slog.Info("discord", "message", "welcome", "channel", channel.Name)
			_, _ = dg.ChannelMessageSend(channel.ID, welcome)
		}
	}
}

// onMessageCreate is received when new message is created on any channel that
// the authenticated bot has access to.
//
// See https://discord.com/developers/docs/topics/gateway-events#message-create
func (d *discordBot) onMessageCreate(dg *discordgo.Session, m *discordgo.MessageCreate) {
	slog.Debug("discord", "event", "messageCreate", "message", m.Message, "state", dg.State)
	botid := dg.State.User.ID
	if m.Author.ID == botid {
		return
	}
	user := fmt.Sprintf("<@%s>", botid)
	if m.GuildID != "" && !strings.Contains(m.Content, user) {
		// Ignore if the bot is not explicitly referenced to.
		return
	}
	msg := strings.TrimSpace(strings.ReplaceAll(m.Content, user, ""))
	slog.Info("discord", "event", "messageCreate", "author", m.Author.Username, "message", msg)
	imgreq := strings.HasPrefix(msg, "image:")
	if imgreq {
		if d.ig == nil {
			if _, err := dg.ChannelMessageSend(m.ChannelID, "Image generation is not enabled. Restart with flag \"-ig\""); err != nil {
				slog.Error("discord", "event", "failed posting message", "error", err)
			}
			return
		}
		msg = strings.TrimSpace(strings.TrimPrefix(msg, "image:"))
	} else {
		if d.l == nil {
			if _, err := dg.ChannelMessageSend(m.ChannelID, "LLM is not enabled."); err != nil {
				slog.Error("discord", "event", "failed posting message", "error", err)
			}
			return
		}
	}
	// Immediately signal the user that the bot is preparing a reply.
	if err := dg.ChannelTyping(m.ChannelID); err != nil {
		slog.Error("discord", "event", "failed posting 'user typing'", "error", err)
		// Continue anyway.
	}
	req := msgReq{
		msg:       msg,
		authorID:  m.Author.ID,
		channelID: m.ChannelID,
		guildID:   m.GuildID,
		replyToID: m.ID,
	}
	if imgreq {
		select {
		case d.image <- req:
		default:
			_, err := dg.ChannelMessageSendComplex(req.channelID, &discordgo.MessageSend{
				Content:   "Sorry! I have too many pending image requests. Please retry in a moment.",
				Reference: &discordgo.MessageReference{MessageID: req.replyToID, ChannelID: req.channelID, GuildID: req.guildID},
			})
			if err != nil {
				slog.Error("discord", "event", "failed posting message", "error", err)
			}
		}
	} else {
		select {
		case d.chat <- req:
		default:
			_, err := dg.ChannelMessageSendComplex(req.channelID, &discordgo.MessageSend{
				Content:   "Sorry! I have too many pending chat requests. Please retry in a moment.",
				Reference: &discordgo.MessageReference{MessageID: req.replyToID, ChannelID: req.channelID, GuildID: req.guildID},
			})
			if err != nil {
				slog.Error("discord", "event", "failed posting message", "error", err)
			}
		}
	}
}

func (d *discordBot) onInteractionCreate(dg *discordgo.Session, event *discordgo.InteractionCreate) {
	slog.Info("discord", "event", "interactionCreate", "name", event.Data)
	if t := event.Data.Type(); t != discordgo.InteractionApplicationCommand {
		slog.Warn("discord", "message", "surprising interaction", "type", t.String())
		return
	}
	data, ok := event.Data.(discordgo.ApplicationCommandInteractionData)
	if !ok {
		slog.Warn("discord", "message", "invalid type", "type", event.Data)
		return
	}
	if data.Name == "forget" {
		u := event.User
		if event.Member != nil {
			u = event.Member.User
		}
		c := d.mem.Get(u.ID, event.ChannelID)
		c.Messages = c.Messages[:1]
		err := d.dg.InteractionRespond(event.Interaction, &discordgo.InteractionResponse{
			Type: discordgo.InteractionResponseChannelMessageWithSource,
			Data: &discordgo.InteractionResponseData{
				Content: "The memory of our past conversations just got zapped.",
			},
		})
		if err != nil {
			slog.Error("discord", "event", "failed handling interaction", "error", err)
		}
		return
	}
	slog.Warn("discord", "unexpected command", data.Name)
}

// Internal

// chatRoutine serializes the chat requests.
func (d *discordBot) chatRoutine() {
	for req := range d.chat {
		if req.authorID == "" {
			d.wg.Done()
			return
		}
		d.handlePrompt(req)
	}
}

// imageRoutine serializes the chat requests.
func (d *discordBot) imageRoutine() {
	for req := range d.image {
		if req.authorID == "" {
			d.wg.Done()
			return
		}
		d.handleImage(req)
	}
}

// handlePrompt uses the LLM to generate a response.
func (d *discordBot) handlePrompt(req msgReq) {
	c := d.mem.Get(req.authorID, req.channelID)
	if len(c.Messages) == 0 {
		c.Messages = []sillybot.Message{{Role: sillybot.System, Content: d.systemPrompt}}
	}
	c.Messages = append(c.Messages, sillybot.Message{Role: sillybot.User, Content: req.msg})
	words := make(chan string, 10)
	wg := sync.WaitGroup{}
	wg.Add(1)
	go func() {
		t := time.NewTicker(3 * time.Second)
		replyToID := req.replyToID
		text := ""
		pending := ""
		for {
			select {
			case w, ok := <-words:
				if !ok {
					if pending != "" {
						text += pending
						_, err := d.dg.ChannelMessageSendComplex(req.channelID, &discordgo.MessageSend{
							Content:   pending,
							Reference: &discordgo.MessageReference{MessageID: replyToID, ChannelID: req.channelID, GuildID: req.guildID},
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
				// Don't send one word at a time.
				if len(pending) > 30 {
					text += pending
					msg, err := d.dg.ChannelMessageSendComplex(req.channelID, &discordgo.MessageSend{
						Content:   pending + " (...continued)",
						Reference: &discordgo.MessageReference{MessageID: replyToID, ChannelID: req.channelID, GuildID: req.guildID},
					})
					pending = ""
					if err != nil {
						slog.Error("discord", "event", "failed posting message", "error", err)
					} else {
						replyToID = msg.ID
					}
					if err := d.dg.ChannelTyping(req.channelID); err != nil {
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
		if _, err = d.dg.ChannelMessageSend(req.channelID, "Prompt generation failed: "+err.Error()); err != nil {
			slog.Error("discord", "event", "failed posting message", "error", err)
		}
	}
}

// handleImage generates an image based on the user prompt.
func (d *discordBot) handleImage(req msgReq) {
	// TODO: Insert a stand-in, then replace it.
	// TODO: Generate multiple images.
	p, err := d.ig.GenImage(req.msg)
	if err != nil {
		if _, err = d.dg.ChannelMessageSend(req.channelID, "Image generation failed: "+err.Error()); err != nil {
			if err != nil {
				slog.Error("discord", "event", "failed posting message", "error", err)
			}
		}
		return
	}

	data := discordgo.MessageSend{
		Files: []*discordgo.File{
			{
				Name:        "prompt.png",
				ContentType: "image/png",
				Reader:      bytes.NewReader(p),
			},
		},
		Reference: &discordgo.MessageReference{MessageID: req.replyToID, ChannelID: req.channelID, GuildID: req.guildID},
	}
	if _, err = d.dg.ChannelMessageSendComplex(req.channelID, &data); err != nil {
		slog.Error("discord", "event", "failed posting message", "error", err)
	}
}

// msgReq is an incoming message pending to be processed.
type msgReq struct {
	// msg is the message received.
	// See
	// https://discord.com/developers/docs/reference#message-formatting-formats
	// for the formatting of references.
	msg       string
	authorID  string
	channelID string
	guildID   string
	replyToID string
}
