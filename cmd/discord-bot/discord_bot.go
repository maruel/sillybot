// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"image"
	"image/png"
	"log/slog"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/bwmarrin/discordgo"
	"github.com/maruel/sillybot/huggingface"
	"github.com/maruel/sillybot/imagegen"
	"github.com/maruel/sillybot/llm"
	"golang.org/x/sync/errgroup"
)

// discordBot is the live instance of the bot talking to the Discord API.
//
// Throughout the code, a Discord Server is called a "Guild". See
// https://discord.com/developers/docs/quick-start/overview-of-apps#where-are-apps-installed
type discordBot struct {
	ctx          context.Context
	dg           *discordgo.Session
	l            *llm.Session
	mem          *llm.Memory
	knownLLMs    []llm.KnownLLM
	ig           *imagegen.Session
	systemPrompt string
	chat         chan msgReq
	image        chan intReq
	wg           sync.WaitGroup
}

// newDiscordBot opens a websocket connection to Discord and begin listening.
func newDiscordBot(ctx context.Context, token string, verbose bool, l *llm.Session, mem *llm.Memory, knownLLMs []llm.KnownLLM, ig *imagegen.Session, systPrmpt string) (*discordBot, error) {
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
		mem:          mem,
		knownLLMs:    knownLLMs,
		ig:           ig,
		systemPrompt: systPrmpt,
		chat:         make(chan msgReq, 5),
		image:        make(chan intReq, 3),
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
	d.image <- intReq{}
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

	// See https://discord.com/developers/docs/interactions/application-commands
	cmds := []*discordgo.ApplicationCommand{
		// meme_*
		{
			Name:        "meme_auto",
			Type:        discordgo.ChatApplicationCommand,
			Description: "Generate a meme in full automatic mode. Create both the image and labels by leveraging the LLM.",
			Options: []*discordgo.ApplicationCommandOption{
				{
					Type:        discordgo.ApplicationCommandOptionString,
					Name:        "description",
					Description: "Description used to generate both the meme labels and background image. The LLM will enhance both.",
					Required:    true,
				},
				{
					Type:        discordgo.ApplicationCommandOptionInteger,
					Name:        "seed",
					Description: "Seed to use to enable (or disable with 0) deterministic image generation. Defaults to 1",
				},
			},
		},
		{
			Name:        "meme_manual",
			Type:        discordgo.ChatApplicationCommand,
			Description: "Generate a meme in full manual mode. Specify both the image and the labels yourself.",
			Options: []*discordgo.ApplicationCommandOption{
				{
					Type:        discordgo.ApplicationCommandOptionString,
					Name:        "image_prompt",
					Description: "Exact Stable Diffusion style prompt to use to generate the image.",
					Required:    true,
				},
				{
					Type:        discordgo.ApplicationCommandOptionString,
					Name:        "labels_content",
					Description: "Exact text to overlay on the image. Use comma to split lines.",
					Required:    true,
				},
				{
					Type:        discordgo.ApplicationCommandOptionInteger,
					Name:        "seed",
					Description: "Seed to use to enable (or disable with 0) deterministic image generation. Defaults to 1",
				},
			},
		},
		{
			Name:        "meme_labels_auto",
			Type:        discordgo.ChatApplicationCommand,
			Description: "Generate meme labels in automatic mode. Create the text by leveraging the LLM.",
			Options: []*discordgo.ApplicationCommandOption{
				{
					Type:        discordgo.ApplicationCommandOptionString,
					Name:        "description",
					Description: "Description to use to generate the meme labels. The LLM will enhance both.",
					Required:    true,
				},
				{
					Type:        discordgo.ApplicationCommandOptionInteger,
					Name:        "seed",
					Description: "Seed to use to enable (or disable with 0) deterministic image generation. Defaults to 1",
				},
			},
		},

		// image_*
		{
			Name:        "image_auto",
			Type:        discordgo.ChatApplicationCommand,
			Description: "Generate an image in automatic mode. It automatically uses the LLM to enhance the prompt.",
			Options: []*discordgo.ApplicationCommandOption{
				{
					Type:        discordgo.ApplicationCommandOptionString,
					Name:        "description",
					Description: "Description to use to generate the image. The LLM will enhance it.",
					Required:    true,
				},
				{
					Type:        discordgo.ApplicationCommandOptionInteger,
					Name:        "seed",
					Description: "Seed to use to enable (or disable with 0) deterministic image generation. Defaults to 1",
				},
			},
		},
		{
			Name:        "image_manual",
			Type:        discordgo.ChatApplicationCommand,
			Description: "Generate an image in manual mode.",
			Options: []*discordgo.ApplicationCommandOption{
				{
					Type:        discordgo.ApplicationCommandOptionString,
					Name:        "image_prompt",
					Description: "Exact Stable Diffusion style prompt to use to generate the image.",
					Required:    true,
				},
				{
					Type:        discordgo.ApplicationCommandOptionInteger,
					Name:        "seed",
					Description: "Seed to use to enable (or disable with 0) deterministic image generation. Defaults to 1",
				},
			},
		},

		// list_models
		{
			Name:        "list_models",
			Type:        discordgo.ChatApplicationCommand,
			Description: "List models available and the one currently used.",
		},

		// forget
		{
			Name:        "forget",
			Type:        discordgo.ChatApplicationCommand,
			Description: "Forget our past conversation. Optionally overrides the system prompt.",
			Options: []*discordgo.ApplicationCommandOption{
				{
					Type:        discordgo.ApplicationCommandOptionString,
					Name:        "system_prompt",
					Description: "new system prompt to use",
				},
			},
		},
		{
			Name: "forget",
			Type: discordgo.UserApplicationCommand,
		},
	}
	if strings.Contains(dg.State.User.ID, "(dev)") {
		for _, c := range cmds {
			c.Name += "_dev"
		}
	}
	if _, err := dg.ApplicationCommandBulkOverwrite(r.Application.ID, "", cmds); err != nil {
		// TODO: Make this a hard fail.
		slog.Error("discord", "message", "failed to register commands", "error", err)
		return
	}
	slog.Info("discord", "message", "registered commands", "number", len(cmds))
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
	const welcome = "I'm back up! Check out my commands by typing the '/' slash key. I can generate images and memes."
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
	if d.l == nil {
		if _, err := dg.ChannelMessageSend(m.ChannelID, "LLM is not enabled."); err != nil {
			slog.Error("discord", "message", "failed posting message", "error", err)
		}
		return
	}
	// Immediately signal the user that the bot is preparing a reply.
	if err := dg.ChannelTyping(m.ChannelID); err != nil {
		slog.Error("discord", "message", "failed posting 'user typing'", "error", err)
		// Continue anyway.
	}
	req := msgReq{
		msg:       msg,
		authorID:  m.Author.ID,
		channelID: m.ChannelID,
		guildID:   m.GuildID,
		replyToID: m.ID,
	}
	select {
	case d.chat <- req:
	default:
		_, err := dg.ChannelMessageSendComplex(req.channelID, &discordgo.MessageSend{
			Content:   "Sorry! I have too many pending chat requests. Please retry in a moment.",
			Reference: &discordgo.MessageReference{MessageID: req.replyToID, ChannelID: req.channelID, GuildID: req.guildID},
		})
		if err != nil {
			slog.Error("discord", "message", "failed posting message", "error", err)
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
	switch data.Name {
	case "forget":
		d.onForget(event, data)
	case "list_models":
		d.onListModels(event, data)
	case "meme_auto", "meme_manual", "meme_labels_auto", "image_auto", "image_manual":
		d.onImage(event, data)
	default:
		slog.Warn("discord", "unexpected command", data.Name, "data", event.Interaction)
	}
}

func (d *discordBot) onForget(event *discordgo.InteractionCreate, data discordgo.ApplicationCommandInteractionData) {
	u := event.User
	if event.Member != nil {
		u = event.Member.User
	}
	c := d.mem.Get(u.ID, event.ChannelID)
	opts := struct {
		SystemPrompt string `json:"system_prompt"`
	}{SystemPrompt: d.systemPrompt}
	if err := optionsToStruct(data.Options, &opts); err != nil {
		slog.Error("discord", "command", data.Name, "message", "failed decoding command options", "error", err)
		return
	}
	reply := "I don't know you. I can't wait to start our discussion so I can get to know you better!"
	if len(c.Messages) > 1 {
		reply = "The memory of our past conversations just got zapped."
	}
	reply += "\n*System prompt*: " + escapeMarkdown(opts.SystemPrompt)
	c.Messages = []llm.Message{{Role: llm.System, Content: opts.SystemPrompt}}
	if err := d.interactionRespond(event.Interaction, reply); err != nil {
		slog.Error("discord", "command", data.Name, "message", "failed reply", "error", err)
	}
}

func (d *discordBot) onListModels(event *discordgo.InteractionCreate, data discordgo.ApplicationCommandInteractionData) {
	var replies []string
	reply := "Known models:\n"
	for _, k := range d.knownLLMs {
		reply += "- [`" + k.Basename + "`](" + k.URL() + ") "
		parts := strings.SplitN(k.RepoID, "/", 2)
		info := huggingface.Model{ModelRef: huggingface.ModelRef{Author: parts[0], Repo: parts[1]}}
		if err := d.l.HF.GetModelInfo(d.ctx, &info); err != nil {
			reply += " Oh no, we failed to query: " + err.Error()
			slog.Error("discord", "command", data.Name, "error", err)
		} else {
			reply += " Quantizations: "
			added := false
			for _, f := range info.Files {
				// TODO: Move this into a common function.
				if !strings.HasPrefix(f, k.Basename) {
					continue
				}
				if strings.Contains(f, "/") {
					// Skip files in subdirectories for now.
					continue
				}
				if strings.HasPrefix(filepath.Ext(f), ".cat") {
					// TODO: Support split files. For now just hide them. They are large
					// anyway so it's only for power users.
					continue
				}
				if added {
					reply += ", "
				}
				reply += strings.TrimSuffix(f[len(k.Basename):], ".gguf")
				added = true
			}
			if info.Upstream.Author == "" && info.Upstream.Repo == "" {
				// Some forks are not setting up upstream properly. What a shame.
				if parts := strings.SplitN(k.UpstreamID, "/", 2); len(parts) == 2 {
					info.Upstream.Author = parts[0]
					info.Upstream.Repo = parts[1]
				}
			}
			if info.Upstream.Author != "" && info.Upstream.Repo != "" {
				infoUpstream := huggingface.Model{ModelRef: info.Upstream}
				if err = d.l.HF.GetModelInfo(d.ctx, &infoUpstream); err != nil {
					reply += " Oh no, we failed to query: " + err.Error()
					slog.Error("discord", "command", data.Name, "error", err)
				} else {
					if infoUpstream.NumWeights != 0 {
						reply += fmt.Sprintf(" Tensors: %s in %.fB", infoUpstream.TensorType, float64(infoUpstream.NumWeights)*0.000000001)
					}
					if infoUpstream.LicenseURL != "" {
						reply += " License: [" + infoUpstream.License + "](" + infoUpstream.LicenseURL + ")"
					} else {
						reply += " License: " + infoUpstream.License
					}
				}
			}
		}
		reply += "\n"
		if len(reply) > 1000 {
			// Don't hit the 2000 characters limit.
			replies = append(replies, reply)
			reply = ""
		}
	}
	if reply != "" {
		replies = append(replies, reply)
	}
	if err := d.interactionRespond(event.Interaction, replies[0]); err != nil {
		slog.Error("discord", "command", data.Name, "message", "failed reply", "error", err)
	}
	for _, r := range replies[1:] {
		// TODO: use MessageID so it becomes a set of replies.
		if _, err := d.dg.ChannelMessageSend(event.Interaction.ChannelID, r); err != nil {
			slog.Error("discord", "command", data.Name, "message", "failed reply", "error", err)
		}
	}
}

func (d *discordBot) onImage(event *discordgo.InteractionCreate, data discordgo.ApplicationCommandInteractionData) {
	opts := struct {
		// meme_auto, meme_labels_auto, image_auto
		Description string `json:"description"`
		// meme_manual, image_manual
		ImagePrompt string `json:"image_prompt"`
		// meme_manual
		LabelsContent string `json:"labels_content"`
		// meme_auto, meme_manual, image_auto, image_manual
		Seed int `json:"seed"`
	}{Seed: 1}
	if err := optionsToStruct(data.Options, &opts); err != nil {
		slog.Error("discord", "command", data.Name, "message", "failed decoding command options", "error", err)
		return
	}
	if d.ig == nil && strings.HasSuffix(data.Name, "_auto") && data.Name != "meme_labels_auto" {
		if err := d.interactionRespond(event.Interaction, "Image generation is not enabled. Restart with bot.image_gen.model set in config.yml."); err != nil {
			slog.Error("discord", "command", data.Name, "message", "failed reply to enable", "error", err)
		}
		return
	}
	if d.l == nil && strings.HasSuffix(data.Name, "_auto") {
		if err := d.interactionRespond(event.Interaction, "LLM is not enabled. Restart with bot.llm.model set in config.yml."); err != nil {
			slog.Error("discord", "command", data.Name, "message", "failed reply to enable", "error", err)
		}
		return
	}
	if strings.HasSuffix(data.Name, "_auto") {
		if opts.Description = strings.TrimSpace(opts.Description); opts.Description == "" {
			if err := d.interactionRespond(event.Interaction, "Description is required."); err != nil {
				slog.Error("discord", "command", data.Name, "message", "failed reply to enable", "error", err)
			}
			return
		}
	}
	req := intReq{
		description:   opts.Description,
		imagePrompt:   opts.ImagePrompt,
		labelsContent: opts.LabelsContent,
		seed:          opts.Seed,
		cmdName:       data.Name,
		int:           event.Interaction,
	}
	select {
	case d.image <- req:
	default:
		if err := d.interactionRespond(event.Interaction, "Sorry! I have too many pending image requests. Please retry in a moment."); err != nil {
			slog.Error("discord", "command", data.Name, "message", "failed reply rate limit", "error", err)
		}
		return
	}
	r := &discordgo.InteractionResponse{Type: discordgo.InteractionResponseDeferredChannelMessageWithSource}
	if err := d.dg.InteractionRespond(req.int, r); err != nil {
		slog.Error("discord", "command", data.Name, "message", "failed reply update", "error", err)
	}
}

func (d *discordBot) interactionRespond(int *discordgo.Interaction, s string) error {
	r := &discordgo.InteractionResponse{Type: discordgo.InteractionResponseChannelMessageWithSource, Data: &discordgo.InteractionResponseData{Content: s}}
	return d.dg.InteractionRespond(int, r)
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
		if req.int == nil {
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
		c.Messages = []llm.Message{{Role: llm.System, Content: d.systemPrompt}}
	}
	c.Messages = append(c.Messages, llm.Message{Role: llm.User, Content: req.msg})
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
							slog.Error("discord", "message", "failed posting message", "error", err)
						}
					}
					// Remember our own answer.
					c.Messages = append(c.Messages, llm.Message{Role: llm.Assistant, Content: text})
					t.Stop()
					wg.Done()
					return
				}
				pending += w
			case <-t.C:
				i := splitResponse(pending)
				if i == 0 {
					continue
				}
				msg, err := d.dg.ChannelMessageSendComplex(req.channelID, &discordgo.MessageSend{
					Content:   pending[:i],
					Reference: &discordgo.MessageReference{MessageID: replyToID, ChannelID: req.channelID, GuildID: req.guildID},
				})
				text += pending[:i]
				pending = pending[i:]
				if err != nil {
					slog.Error("discord", "message", "failed posting message", "error", err)
				} else {
					replyToID = msg.ID
				}
				if err := d.dg.ChannelTyping(req.channelID); err != nil {
					slog.Error("discord", "message", "failed posting 'user typing'", "error", err)
				}
			}
		}
	}()
	err := d.l.PromptStreaming(d.ctx, c.Messages, 0, 1.0, words)
	close(words)
	wg.Wait()

	if err != nil {
		if _, err = d.dg.ChannelMessageSend(req.channelID, "Prompt generation failed: "+err.Error()); err != nil {
			slog.Error("discord", "message", "failed posting message", "error", err)
		}
	}
}

// splitResponse take pending reply from the LLM and returns the amount of
// bytes to send back. Returns 0 if nothing should be sent.
//
// The basic assumption here is that most LLMs are trained to generate
// markdown, for the better or worst.
//
// If you spot a case where it doesn't work right in the wild, please fix and
// add a test case! Make sure it's 100% test coverage (except the panic).
func splitResponse(t string) int {
	// First priority is 3 backquotes. We must never break that in the middle.
	if backquotes := strings.Count(t, "```"); backquotes == 1 {
		i := strings.Index(t, "```")
		t = t[:i]
	} else if backquotes >= 2 {
		// Cut right after the second.
		start := strings.Index(t, "```") + 3
		return strings.Index(t[start:], "```") + start + 3
	}
	if t == "" {
		return 0
	}

	// If there's a EOL, use it.
	if i := strings.LastIndexByte(t, '\n'); i >= 10 {
		return i + 1
	}

	// Now look for enumerations. The only thing we want to break on for
	// enumerations is '\n'.
	isEnum := strings.HasPrefix(t, "- ") || strings.HasPrefix(t, "* ")
	if !isEnum {
		var err error
		isEnum, err = regexp.MatchString(`^\d+\. .*`, t)
		if err != nil {
			panic(err)
		}
	}
	if isEnum {
		return 0
	}

	if len(t) < 10 {
		// Not enough characters to send, ignore.
		return 0
	}

	// If there's backquotes, e.g. `foo.bar`, they mess up punctuation search. So
	// only start the search after the last backquote.
	start := 0
	if backquotes := strings.Count(t, "`"); (backquotes & 1) == 1 {
		// Impair number of backquotes. Limit ourselves up to the last one.
		t = t[:strings.LastIndexByte(t, '`')]
	}

	// TODO: Highlighting pairs: '*' and '_'
	i := strings.LastIndexAny(t[start:], ".?!")
	if i == -1 {
		return 0
	}
	return start + i + 1
}

// TODO: fine-tune.
const imagePrompt = "You are autoregressive language model that specializes in creating perfect, outstanding prompts for generative art models like Stable Diffusion. Your job is to take user ideas, capture ALL main parts, and turn into amazing prompts. You have to capture everything from the user's prompt and then use your talent to make it amazing. You are a master of art styles, terminology, pop culture, and photography across the globe. Respond only with the new prompt. Exclude article words."

// TODO: fine-tune.
const memeLabelsPrompt = "You are autoregressive language model that specializes in creating perfect, outstanding meme text. Your job is to take user ideas, capture ALL main parts, and turn into amazing meme labels. You have to capture everything from the user's prompt and then use your talent to make it amazing filled with sarcasm. Respond only with the new meme text. Make it as succinct as possible. Use few words. Use exactly one comma. Exclude article words."

// handleImage generates an image based on the user prompt.
func (d *discordBot) handleImage(req intReq) {
	// Generate multiple images when the queue is empty.
	resp := discordgo.WebhookEdit{}
	content := ""
	if req.description != "" {
		content += "*Description*: " + escapeMarkdown(req.description) + "\n"
	}
	if req.imagePrompt != "" {
		content += "*Image prompt*: " + escapeMarkdown(req.imagePrompt) + "\n"
	}
	if req.labelsContent != "" {
		content += "*Labels*: " + escapeMarkdown(req.labelsContent) + "\n"
	}
	seed := req.seed
	for i := 0; i < 4; i++ {
		newreq := req
		newreq.seed = seed
		img, err := d.genImage(&newreq)
		if seed != 0 {
			content += "*Seed*: " + strconv.Itoa(seed) + "\n"
		}
		if newreq.description != req.description {
			content += "*Description*: " + escapeMarkdown(newreq.description) + "\n"
		}
		if newreq.imagePrompt != req.imagePrompt {
			content += "*Image prompt*: " + escapeMarkdown(newreq.imagePrompt) + "\n"
		}
		if newreq.labelsContent != req.labelsContent {
			content += "*Labels*: " + escapeMarkdown(newreq.labelsContent) + "\n"
		}
		if err != nil {
			slog.Error("discord", "imagereq", req, "error", err)
			content += "\n*LLM Eror*: " + escapeMarkdown(err.Error()) + "\n"
		}
		resp.Content = &content
		if len(img) != 0 {
			resp.Files = []*discordgo.File{{Name: "prompt.png", ContentType: "image/png", Reader: bytes.NewReader(img)}}
		}
		if _, err2 := d.dg.InteractionResponseEdit(req.int, &resp); err2 != nil {
			slog.Error("discord", "imagereq", req, "message", "failed posting interaction", "error", err2)
			break
		}
		// If there were an error or there's another request pending, stop.
		if err != nil || len(d.image) != 0 || len(d.chat) != 0 {
			break
		}
		seed++
		if err := d.dg.ChannelTyping(req.int.ChannelID); err != nil {
			slog.Error("discord", "message", "failed posting 'user typing'", "error", err)
			break
		}
	}
}

func (d *discordBot) genImage(req *intReq) ([]byte, error) {
	if req.cmdName == "meme_auto" || req.cmdName == "image_auto" {
		// Image: use the LLM to generate the image prompt based on the description.
		msgs := []llm.Message{{Role: llm.System, Content: imagePrompt}, {Role: llm.User, Content: req.description}}
		reply, err := d.l.Prompt(d.ctx, msgs, req.seed, 1.0)
		if err != nil {
			return nil, fmt.Errorf("failed to enhance image generation prompt: %w", err)
		}
		req.imagePrompt = reply
	}
	var img *image.NRGBA
	eg, ctx := errgroup.WithContext(d.ctx)
	eg.Go(func() error {
		// Labels: use the LLM to generate the labels based on the description.
		if req.cmdName == "meme_auto" || req.cmdName == "meme_labels_auto" {
			msgs := []llm.Message{{Role: llm.System, Content: memeLabelsPrompt}, {Role: llm.User, Content: req.description}}
			meme, err2 := d.l.Prompt(ctx, msgs, req.seed, 1.0)
			if err2 != nil {
				return fmt.Errorf("failed to enhance labels: %w", err2)
			}
			req.labelsContent = strings.Trim(meme, "\",.")
		}
		return nil
	})
	if req.cmdName != "meme_labels_auto" {
		// Generate the image.
		// "meme_labels_auto" is a special case where we don't actually need an image.
		eg.Go(func() error {
			var err2 error
			img, err2 = d.ig.GenImage(ctx, req.imagePrompt, req.seed)
			return err2
		})
	}
	err := eg.Wait()
	w := bytes.Buffer{}
	if img != nil {
		imagegen.DrawLabelsOnImage(img, req.labelsContent)
		if err2 := png.Encode(&w, img); err == nil {
			err = err2
		}
	}
	return w.Bytes(), err
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

// intReq is an interaction request to generate an image.
type intReq struct {
	description   string
	imagePrompt   string
	labelsContent string
	seed          int
	cmdName       string
	// Only there for ID and Token.
	int *discordgo.Interaction
}

func optionsToStruct(opts []*discordgo.ApplicationCommandInteractionDataOption, out interface{}) error {
	// The world's slowest implementation.
	// TODO: Use something faster, e.g. use reflect directly. PR appreciated. ❤
	t := map[string]interface{}{}
	for _, o := range opts {
		t[o.Name] = o.Value
	}
	b, err := json.Marshal(t)
	if err != nil {
		return err
	}
	return json.Unmarshal(b, out)
}

func escapeMarkdown(s string) string {
	const _MARKDOWN_ESCAPE_COMMON = `^>(?:>>)?\s|\[.+\]\(.+\)|^#{1,3}|^\s*-`
	const _MARKDOWN_STOCK_REGEX = `(?P<markdown>[_\\~|\*` + "`" + `]|` + _MARKDOWN_ESCAPE_COMMON + `)`
	re := regexp.MustCompile(_MARKDOWN_STOCK_REGEX)
	return re.ReplaceAllStringFunc(s, func(m string) string { return "\\" + m })
}
