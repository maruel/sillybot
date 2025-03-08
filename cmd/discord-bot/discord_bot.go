// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"image/jpeg"
	"image/png"
	"log/slog"
	"math/big"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/bwmarrin/discordgo"
	"github.com/maruel/genai/genaiapi"
	"github.com/maruel/huggingface"
	"github.com/maruel/sillybot"
	"github.com/maruel/sillybot/imagegen"
	"github.com/maruel/sillybot/llm"
	"github.com/maruel/sillybot/llm/tools"
	"google.golang.org/api/customsearch/v1"
	"google.golang.org/api/option"
)

// Discord returns HTTP 400 BASE_TYPE_MAX_LENGTH if the message is longer than
// that. There's a 4000 limit in some case (embeds?), investigate.
const maxMessage = 2000

// discordBot is the live instance of the bot talking to the Discord API.
//
// Throughout the code, a Discord Server is called a "Guild". See
// https://discord.com/developers/docs/quick-start/overview-of-apps#where-are-apps-installed
type discordBot struct {
	ctx       context.Context
	dg        *discordgo.Session
	l         *llm.Session
	mem       *llm.Memory
	knownLLMs []llm.KnownLLM
	ig        *imagegen.Session
	settings  sillybot.Settings
	memDir    string
	toolsMsg  genaiapi.Message
	chat      chan msgReq
	image     chan intReq
	gcptoken  string
	cxtoken   string
	wg        sync.WaitGroup
}

// newDiscordBot opens a websocket connection to Discord and begin listening.
func newDiscordBot(ctx context.Context, bottoken, gcptoken, cxtoken string, verbose bool, l *llm.Session, mem *llm.Memory, knownLLMs []llm.KnownLLM, ig *imagegen.Session, settings sillybot.Settings, memDir string) (*discordBot, error) {
	toolsMsg := genaiapi.Message{}
	if l.Encoding != nil && strings.Contains(strings.ToLower(string(l.Model)), "mistral") {
		slog.Info("discord", "message", "tools are enabled", "encoding", l.Encoding)
		// HACK: Also an hack.
		availtools := []tools.MistralTool{
			/*
				{
					Type: "function",
					Function: tools.MistralFunction{
						Name:        "web_search",
						Description: "Search the web for information",
						Parameters: &tools.MistralFunctionParams{
							Type: "object",
							Properties: map[string]tools.MistralProperty{
								"query": {
									Type:        "string",
									Description: "Query to use to search on the internet",
								},
							},
							Required: []string{"query"},
						},
					},
				},
			*/
			tools.CalculateMistralTool,
			tools.GetTodayClockTimeMistralTool,
		}
		b, err := json.Marshal(availtools)
		if err != nil {
			return nil, err
		}
		toolsMsg = genaiapi.Message{
			Role:    genaiapi.AvailableTools,
			Content: string(b),
		}
	}

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
	dg, err := discordgo.New("Bot " + bottoken)
	if err != nil {
		return nil, err
	}
	if verbose {
		dg.LogLevel = discordgo.LogInformational
		// It's very verbose.
		// dg.LogLevel = discordgo.LogDebug
	}
	// We want to receive as few messages as possible.
	dg.Identify.Intents = discordgo.IntentsGuilds | discordgo.IntentsGuildMessages | discordgo.IntentDirectMessages
	d := &discordBot{
		ctx:       ctx,
		dg:        dg,
		l:         l,
		mem:       mem,
		knownLLMs: knownLLMs,
		ig:        ig,
		settings:  settings,
		memDir:    memDir,
		toolsMsg:  toolsMsg,
		chat:      make(chan msgReq, 5),
		image:     make(chan intReq, 3),
		gcptoken:  gcptoken,
		cxtoken:   cxtoken,
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
	// slog.Debug("discord", "event", "ready", "session", dg, "event", r)
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
					Description: "Description to use to generate the meme labels. The LLM will enhance it.",
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

		// Various
		{
			Name:        "close_thread",
			Type:        discordgo.ChatApplicationCommand,
			Description: "Close the thread that was created.",
		},
		{
			Name:        "list_models",
			Type:        discordgo.ChatApplicationCommand,
			Description: "List available LLM models and the one currently used.",
		},
		{
			Name:        "metrics",
			Type:        discordgo.ChatApplicationCommand,
			Description: "Displays the current performance metrics.",
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
					Description: "New system prompt to use.",
				},
			},
		},
		{
			Name: "forget",
			Type: discordgo.UserApplicationCommand,
		},
	}
	if strings.Contains(dg.State.User.Username, "(dev)") {
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
	// slog.Debug("discord", "event", "guildCreate", "event", event.Guild)
	slog.Info("discord", "event", "guildCreate", "name", event.Guild.Name)
	if event.Guild.Unavailable {
		return
	}
	// This is too spammy.
	if false {
		const welcome = "I'm back up! 👋 I can do many things!\n" +
			"- Tag me in channels to chat with me. Start a DM to talk alone, then no need to tag me at every messages.\n" +
			"- Check out my commands by typing the '/' slash key:\n" +
			"  * I can generate images and memes 🖼️. Try `/image_auto flowers garden gorgeous realistic`, or `/meme_auto AI overlord` or `/meme_auto flowers garden fun`\n" +
			"  * Get information about me. Try `/list_models`, `/metrics`\n" +
			"  * I sometimes get stuck! Reset my memory 🧠 and optionally change my system prompt with `/forget`\n" +
			"I'm a work in progress! Please submit fixes and improvements at https://github.com/maruel/sillybot !\n" +
			"**Warning**: I have no privacy protection yet. I do not listen unless you tag me directly.\n" +
			"**Important**: Keep it civil otherwise I'll have to be turned down.\n"
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
				//  && msg.Content == welcome
				if msg.Author.ID == dg.State.User.ID {
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
}

// onMessageCreate is received when new message is created on any channel that
// the authenticated bot has access to.
//
// See https://discord.com/developers/docs/topics/gateway-events#message-create
func (d *discordBot) onMessageCreate(dg *discordgo.Session, m *discordgo.MessageCreate) {
	// slog.Debug("discord", "event", "messageCreate", "message", m.Message, "state", dg.State)
	botid := dg.State.User.ID
	if m.Author.ID == botid || m.Pinned {
		return
	}

	// A DM doesn't have a GuildID (server) associated. If it's a DM, it's a
	// message directly for us.
	isDM := m.GuildID == ""
	// If the created the thread, we reply to every messages. In theory it could
	// be a channel but the code doesn't create channels, only threads.
	isThread := false
	if !isDM {
		ch, err := dg.State.Channel(m.ChannelID)
		if err != nil {
			slog.Error("discord", "message", "failed getting channel", "error", err)
			return
		}
		isThread = ch.OwnerID == botid
	}
	user := fmt.Sprintf("<@%s>", botid)
	// Ignore if not DM, threads we created, and not tagged in a public channel.
	if !isDM && !isThread && !strings.Contains(m.Content, user) {
		slog.Debug("discord", "event", "messageCreate", "author", m.Author.Username, "server", m.GuildID, "channel", m.ChannelID, "message", "ignored")
		return
	}
	if d.l == nil {
		if _, err := dg.ChannelMessageSend(m.ChannelID, "LLM is not enabled."); err != nil {
			slog.Error("discord", "message", "failed posting message", "error", err)
		}
		return
	}

	channel := m.ChannelID
	msg := strings.TrimSpace(strings.ReplaceAll(m.Content, user, ""))
	replyToID := m.ID
	if !isDM && !isThread {
		// Create thread.
		title := msg
		if len(title) > 95 {
			title = title[:95] + "..."
		}
		slog.Info("discord", "event", "messageCreate", "message", "created thread", "title", title)
		thread, err := dg.MessageThreadStart(m.ChannelID, m.ID, title, 4320)
		if err != nil {
			slog.Error("discord", "message", "failed starting thread", "error", err)
			return
		}
		channel = thread.ID
		// When creating a thread, there's no initial message to reply to yet.
		replyToID = ""
	}
	slog.Info("discord", "event", "messageCreate", "author", m.Author.Username, "server", m.GuildID, "channel", channel, "isdm", isDM, "isthread", isThread, "message", msg)
	// Immediately signal the user that the bot is preparing a reply.
	if err := dg.ChannelTyping(channel); err != nil {
		slog.Error("discord", "message", "failed posting 'user typing'", "error", err)
		// Continue anyway.
	}
	req := msgReq{
		msg:       msg,
		authorID:  m.Author.ID,
		channelID: channel,
		guildID:   m.GuildID,
		replyToID: replyToID,
	}
	select {
	case d.chat <- req:
	default:
		if _, err := d.channelMessageSendComplex(req.replyToID, req.channelID, req.guildID, "Sorry! I have too many pending chat requests. Please retry in a moment."); err != nil {
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
	data.Name = strings.TrimSuffix(data.Name, "_dev")
	switch data.Name {
	case "close_thread":
		d.onCloseThread(event, data)
	case "forget":
		d.onForget(event, data)
	case "list_models":
		d.onListModels(event, data)
	case "metrics":
		d.onMetrics(event, data)
	case "meme_auto", "meme_manual", "meme_labels_auto", "image_auto", "image_manual":
		d.onImage(event, data)
	default:
		slog.Warn("discord", "unexpected command", data.Name, "data", event.Interaction)
	}
}

func (d *discordBot) onCloseThread(event *discordgo.InteractionCreate, data discordgo.ApplicationCommandInteractionData) {
	r := &discordgo.InteractionResponse{Type: discordgo.InteractionResponseChannelMessageWithSource, Data: &discordgo.InteractionResponseData{Content: "_Archived_."}}
	if err := d.dg.InteractionRespond(event.Interaction, r); err != nil {
		slog.Error("discord", "message", "failed to respond to archive thread", "error", err)
	}
	archived := true
	if _, err := d.dg.ChannelEditComplex(event.ChannelID, &discordgo.ChannelEdit{Archived: &archived}); err != nil {
		slog.Error("discord", "message", "failed to archive thread", "error", err)
	}
}

func (d *discordBot) onForget(event *discordgo.InteractionCreate, data discordgo.ApplicationCommandInteractionData) {
	opts := struct {
		SystemPrompt string `json:"system_prompt"`
	}{SystemPrompt: d.settings.PromptSystem}
	if err := optionsToStruct(data.Options, &opts); err != nil {
		slog.Error("discord", "command", data.Name, "message", "failed decoding command options", "error", err)
		return
	}
	reply := "I don't know you. I can't wait to start our discussion so I can get to know you better!"
	c := d.getMemory(event.ChannelID)
	if len(c.Messages) >= 1 && c.Messages[len(c.Messages)-1].Role != genaiapi.System {
		reply = "The memory of our past conversations just got zapped."
	}
	c.Messages = nil
	c = d.getMemory(event.ChannelID)
	// Either update, remove or add, depending.
	if (opts.SystemPrompt == "") != (d.settings.PromptSystem == "") {
		if opts.SystemPrompt != "" {
			c.Messages = append(c.Messages, genaiapi.Message{Role: genaiapi.System, Content: opts.SystemPrompt})
		} else if len(c.Messages) != 0 {
			c.Messages = c.Messages[:len(c.Messages)-1]
		}
	} else if opts.SystemPrompt != "" {
		c.Messages[len(c.Messages)-1].Content = opts.SystemPrompt
	}

	reply += "\n*System prompt*: " + escapeMarkdown(opts.SystemPrompt)
	if err := d.interactionRespond(event.Interaction, reply); err != nil {
		slog.Error("discord", "command", data.Name, "message", "failed reply", "error", err)
	}
}

func (d *discordBot) onListModels(event *discordgo.InteractionCreate, data discordgo.ApplicationCommandInteractionData) {
	lines := []string{"Known models:"}
	for _, k := range d.knownLLMs {
		line := "- [`" + k.Source.Basename() + "`](" + k.Source.RepoURL() + ") "
		info := huggingface.Model{ModelRef: k.Source.ModelRef()}
		if err := d.l.HF.GetModelInfo(d.ctx, &info, "main"); err != nil {
			line += " Oh no, we failed to query: " + err.Error()
			slog.Error("discord", "command", data.Name, "error", err)
		} else {
			line += " Quantizations: "
			added := false
			for _, f := range info.Files {
				// TODO: Move this into a common function.
				if !strings.HasPrefix(f, k.Source.Basename()) {
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
					line += ", "
				}
				line += strings.TrimSuffix(f[len(k.Source.Basename()):], ".gguf")
				added = true
			}
			if info.Upstream.Author == "" && info.Upstream.Repo == "" {
				// Some forks are not setting up upstream properly. What a shame.
				info.Upstream = k.Upstream.ModelRef()
			}
			if info.Upstream.Author != "" && info.Upstream.Repo != "" {
				infoUpstream := huggingface.Model{ModelRef: info.Upstream}
				if err = d.l.HF.GetModelInfo(d.ctx, &infoUpstream, "main"); err != nil {
					line += " Oh no, we failed to query: " + err.Error()
					slog.Error("discord", "command", data.Name, "error", err)
				} else {
					if infoUpstream.NumWeights != 0 {
						line += fmt.Sprintf(" Tensors: %s in %.fB", infoUpstream.TensorType, float64(infoUpstream.NumWeights)*0.000000001)
					}
					if infoUpstream.LicenseURL != "" {
						line += " License: [" + infoUpstream.License + "](" + infoUpstream.LicenseURL + ")"
					} else {
						line += " License: " + infoUpstream.License
					}
				}
			}
		}
		lines = append(lines, line)
	}
	var toSend []string
	buf := ""
	for _, line := range lines {
		if len(buf)+len(line) >= maxMessage {
			toSend = append(toSend, buf)
			buf = ""
		}
		if buf != "" {
			buf += "\n"
		}
		buf += line
	}
	if buf != "" {
		toSend = append(toSend, buf)
	}
	if err := d.interactionRespond(event.Interaction, toSend[0]); err != nil {
		slog.Error("discord", "command", data.Name, "message", "failed reply", "error", err)
	}
	for _, r := range toSend[1:] {
		// TODO: use MessageID so it becomes a set of replies.
		if _, err := d.dg.ChannelMessageSend(event.Interaction.ChannelID, r); err != nil {
			slog.Error("discord", "command", data.Name, "message", "failed reply", "error", err)
		}
	}
}

func (d *discordBot) onMetrics(event *discordgo.InteractionCreate, data discordgo.ApplicationCommandInteractionData) {
	m := llm.Metrics{}
	if err := d.l.GetMetrics(d.ctx, &m); err != nil {
		if err = d.interactionRespond(event.Interaction, "Internal error: "+err.Error()); err != nil {
			slog.Error("discord", "command", data.Name, "message", "failed reply", "error", err)
		}
		return
	}
	s := fmt.Sprintf(
		"LLM server metrics running %s:\n"+
			"- Prompt: **%4d** tokens; **% 8.2f** tok/s\n"+
			"- Generated: **%4d** tokens; **% 8.2f** tok/s",
		d.l.Model,
		m.Prompt.Count, m.Prompt.Rate(),
		m.Generated.Count, m.Generated.Rate())
	if err := d.interactionRespond(event.Interaction, s); err != nil {
		slog.Error("discord", "command", data.Name, "message", "failed reply", "error", err)
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
		Seed int64 `json:"seed"`
	}{}
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
	// Prewarm the system prompt, clearing previous memory.
	if d.settings.PromptSystem != "" {
		c := d.getMemory("")
		c.Messages = nil
		c = d.getMemory("")
		opts := genaiapi.CompletionOptions{MaxTokens: 100, Temperature: 1.0}
		if _, err := d.l.Prompt(d.ctx, c.Messages, &opts); err != nil {
			slog.Error("discord", "error", err)
		}
	}
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

func (d *discordBot) toolWebSearch(ctx context.Context, query string) (*customsearch.Search, error) {
	// - https://developers.google.com/custom-search/v1/using_rest
	// - https://console.cloud.google.com/apis/credentials
	// - https://console.cloud.google.com/apis/library/customsearch.googleapis.com
	// - https://console.cloud.google.com/apis/api/customsearch.googleapis.com/metrics
	// - http://www.google.com/cse/manage/all
	// - https://developers.google.com/custom-search/v1/introduction
	s, err := customsearch.NewService(ctx, option.WithAPIKey(d.gcptoken))
	if err != nil {
		return nil, err
	}
	// return s.Cse.List().Cx(d.cxtoken).Context(ctx).Q(query).Num(5).Do()
	return s.Cse.List().Cx(d.cxtoken).Q(query).Do()
	// TODO: Doesn't seem to work at the moment. :(
	// Note that the free quota is minimal:
	// - Google: 100 queries per day
	// - Bing: 1000 queries per month, 2.5¢ after
	// Investigate:
	// - https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/overview
	// - https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/quickstarts/rest/go
	// - https://portal.azure.com/#view/Microsoft_Azure_ProjectOxford/CognitiveServicesHub/~/CognitiveSearch
}

func (d *discordBot) getMemory(channelID string) *llm.Conversation {
	// TODO: Send a warning or forget when one of Model, Prompt, Tools changed.
	c := d.mem.Get("", channelID)
	if len(c.Messages) == 0 {
		if d.toolsMsg.Content != "" {
			c.Messages = []genaiapi.Message{d.toolsMsg}
		}
		if d.settings.PromptSystem != "" {
			c.Messages = append(c.Messages, genaiapi.Message{Role: genaiapi.System, Content: d.settings.PromptSystem})
		}
	}
	return c
}

// handlePrompt uses the LLM to generate a response.
func (d *discordBot) handlePrompt(req msgReq) {
	if true {
		d.handlePromptStreaming(req)
	} else {
		d.handlePromptBlocking(req)
	}
}

// handlePromptBlocking asks the LLM to reply back, wait for the whole answer,
// then process it. This function exists for testing.
func (d *discordBot) handlePromptBlocking(req msgReq) {
	c := d.getMemory(req.channelID)
	c.Messages = append(c.Messages, genaiapi.Message{Role: genaiapi.User, Content: req.msg})
	replyToID := req.replyToID
	for {
		// 32768
		opts := genaiapi.CompletionOptions{Temperature: 1.0}
		reply, err := d.l.Prompt(d.ctx, c.Messages, &opts)
		if err != nil {
			if _, err = d.dg.ChannelMessageSend(req.channelID, "Prompt generation failed: "+err.Error()+"\nTry `/forget` to reset the internal state"); err != nil {
				slog.Error("discord", "message", "failed posting message", "error", err)
			}
			return
		}
		// Remember our own answer.
		c.Messages = append(c.Messages, genaiapi.Message{Role: genaiapi.Assistant, Content: reply})
		gotToolCall := false
		for reply != "" {
			if d.l.Encoding != nil && !gotToolCall {
				if called := d.handleMistralToolCall(reply, c); called != "" {
					// TODO: Tell the user a function is being used, not after it was used.
					gotToolCall = true
					if msg, err := d.channelMessageSendComplex(replyToID, req.channelID, req.guildID, "*An instant please, I'm calling tool "+escapeMarkdown(called)+"*"); err != nil {
						slog.Error("discord", "message", "failed posting message", "error", err, "content", "*An instant please, I'm calling tool "+escapeMarkdown(called)+"*")
					} else {
						replyToID = msg.ID
					}
					if err := d.dg.ChannelTyping(req.channelID); err != nil {
						slog.Error("discord", "message", "failed posting 'user typing'", "error", err)
					}
					// We need to do a new loop.
					break
				}
			}
			// Only split the response if it is too large.
			t := reply
			rest := ""
			if len(reply) > maxMessage {
				if t, rest = splitResponse(reply, false); t == "" {
					if t, rest = splitResponse(reply, true); t == "" {
						t = rest[:maxMessage]
						rest = rest[maxMessage:]
					}
				}
			}
			msg, err := d.channelMessageSendComplex(replyToID, req.channelID, req.guildID, t)
			if err != nil {
				slog.Error("discord", "message", "failed posting message", "error", err, "content", t)
			} else {
				replyToID = msg.ID
			}
			reply = rest
		}
		if !gotToolCall {
			return
		}
	}
}

// handlePromptStreaming request a reply from the LLM and streams replies back.
func (d *discordBot) handlePromptStreaming(req msgReq) {
	c := d.getMemory(req.channelID)
	c.Messages = append(c.Messages, genaiapi.Message{Role: genaiapi.User, Content: req.msg})
	wg := sync.WaitGroup{}
	for {
		ctx, cancel := context.WithCancel(d.ctx)
		gotToolCall := false
		// Make it blocking to force a goroutine context switch when a word is
		// received. When it's buffered, there can be significant delay when LLM is
		// running on the CPU.
		words := make(chan string)
		wg.Add(1)
		go func() {
			const rate = 2000 * time.Millisecond
			t := time.NewTicker(rate)
			last := time.Now()
			replyToID := req.replyToID
			text := ""
			pending := ""
			for {
				select {
				case w, ok := <-words:
					// slog.Debug("discord", "w", w, "ok", ok)
					if !ok {
						if d.l.Encoding != nil && !gotToolCall {
							if called := d.handleMistralToolCall(pending, c); called != "" {
								// TODO: Tell the user a function is being used, not after it was used.
								gotToolCall = true
								// No need to wait for additional content.
								// TODO: investigate why it's not taking effect faster.
								cancel()
								if msg, err := d.channelMessageSendComplex(replyToID, req.channelID, req.guildID, "*An instant please, I'm calling tool "+escapeMarkdown(called)+"*"); err != nil {
									slog.Error("discord", "message", "failed posting message", "error", err, "content", "*An instant please, I'm calling tool "+escapeMarkdown(called)+"*")
								} else {
									replyToID = msg.ID
								}
								if err := d.dg.ChannelTyping(req.channelID); err != nil {
									slog.Error("discord", "message", "failed posting 'user typing'", "error", err)
								}
							}
						}
						if !gotToolCall {
							// That's the end, flush all the remaining content.
							if pending != "" {
								text += pending
								// When a model is asked to do a large program, it's frequent
								// that it will buffer the whole response and send it back in
								// one shot. In this case, the content received can be very
								// large.
								// TODO: It could be a function call!! Handle it.
								for len(pending) > maxMessage {
									t, rest := splitResponse(pending, true)
									if t == "" {
										t = rest[:maxMessage]
										rest = rest[maxMessage:]
									}
									msg, err := d.channelMessageSendComplex(replyToID, req.channelID, req.guildID, t)
									if err != nil {
										slog.Error("discord", "message", "failed posting message", "error", err, "content", t)
									} else {
										replyToID = msg.ID
									}
									pending = rest
								}
								if _, err := d.channelMessageSendComplex(replyToID, req.channelID, req.guildID, pending); err != nil {
									slog.Error("discord", "message", "failed posting message", "error", err, "content", pending)
								}
							}
							// Remember our own answer.
							c.Messages = append(c.Messages, genaiapi.Message{Role: genaiapi.Assistant, Content: text})
						}
						t.Stop()
						wg.Done()
						return
					}
					pending += w
				case now := <-t.C:
					// TODO: Handle when the model replies with more than maxMessage per
					// rate.
					// It becomes urgent when it's twice the period.
					if t, rest := splitResponse(pending, now.Sub(last) >= 2*rate); t != "" {
						if d.l.Encoding != nil && !gotToolCall {
							// TODO: function call is when a line, any line, starts with "[".
							// Sometimes the last "]" is not followed by a \n, which breaks json
							// parsing.
							if called := d.handleMistralToolCall(t, c); called != "" {
								// TODO: Tell the user a function is being used, not after it was used.
								gotToolCall = true
								// No need to wait for additional content.
								// TODO: investigate why it's not taking effect faster.
								cancel()
								msg, err := d.channelMessageSendComplex(replyToID, req.channelID, req.guildID, "*An instant please, I'm calling tool "+escapeMarkdown(called)+"*")
								if err != nil {
									slog.Error("discord", "message", "failed posting message", "error", err, "content", "*An instant please, I'm calling tool "+escapeMarkdown(called)+"*")
								} else {
									replyToID = msg.ID
								}
							}
						}
						if !gotToolCall {
							msg, err := d.channelMessageSendComplex(replyToID, req.channelID, req.guildID, t)
							if err != nil {
								slog.Error("discord", "message", "failed posting message", "error", err, "content", t)
							} else {
								replyToID = msg.ID
							}
							last = now
						}
						text += t
						pending = rest
					}
				}
				if err := d.dg.ChannelTyping(req.channelID); err != nil {
					slog.Error("discord", "message", "failed posting 'user typing'", "error", err)
				}
			}
		}()
		// We're chatting, we don't want too much content?
		opts := genaiapi.CompletionOptions{Temperature: 1.0}
		err := d.l.PromptStreaming(ctx, c.Messages, &opts, words)
		close(words)
		wg.Wait()
		cancel()
		if errors.Is(err, context.Canceled) {
			err = nil
		}
		if err != nil {
			if _, err = d.dg.ChannelMessageSend(req.channelID, "Prompt generation failed: "+err.Error()+"\nTry `/forget` to reset the internal state"); err != nil {
				slog.Error("discord", "message", "failed posting message", "error", err)
			}
		}
		if !gotToolCall {
			break
		}
	}
}

func (d *discordBot) channelMessageSendComplex(replyToID, channelID, guildID, content string) (st *discordgo.Message, err error) {
	msgSend := discordgo.MessageSend{Content: content}
	if replyToID != "" {
		msgSend.Reference = &discordgo.MessageReference{MessageID: replyToID, ChannelID: channelID, GuildID: guildID}
	}
	return d.dg.ChannelMessageSendComplex(channelID, &msgSend)
}

// handleMistralToolCall check if the pending string and returns its name if so.
//
// TODO: This shouldn't receive the whole conversation. It should return the
// name before calling so it can alert the user, especially for tools that take
// a long time to run.
func (d *discordBot) handleMistralToolCall(pending string, c *llm.Conversation) string {
	var calls []tools.MistralToolCall
	for _, line := range strings.Split(pending, "\n") {
		if line = strings.TrimSpace(line); line == "" {
			continue
		}
		if err := json.Unmarshal([]byte(line), &calls); err != nil {
			// slog.Debug("discord", "message", "line is not tool call", "line", line)
			continue
		}
		if len(calls) != 1 {
			slog.Warn("discord", "message", "unexpected number of calls", "line", line, "calls", calls)
			continue
		}
		name := calls[0].Name
		result := ""
		// TODO: Use reflect to determine arguments automatically.
		// TODO: Stop hardcoding the function names.
		switch calls[0].Name {
		case "web_search":
			query := calls[0].Arguments["query"]
			if len(calls[0].Arguments) != 1 || query == "" {
				slog.Warn("discord", "message", "not the call we wanted", "line", line, "calls", calls)
				continue
			}
			slog.Info("discord", "tool_call", calls[0].Name, "query", query)
			search, err := d.toolWebSearch(d.ctx, query)
			if err != nil {
				slog.Error("discord", "tool", "web_search", "error", err)
				// Continue as if it wasn't a tool call.
				continue
			}
			for _, l := range search.Items {
				result += "- " + l.Title + ": " + l.Snippet + "\n"
			}
			if result == "" {
				slog.Info("discord", "tool_call", calls[0].Name, "result", result, "error", "no result!")
				result = "No result was found on the internet due to an internal error in discord-bot"
			}
			slog.Info("discord", "tool_call", calls[0].Name, "result", result)
			name += fmt.Sprintf("(query=%q) = %s", query, result)
		case "calculate":
			op := calls[0].Arguments["operation"]
			f := calls[0].Arguments["first_number"]
			s := calls[0].Arguments["second_number"]
			result = tools.Calculate(op, f, s)
			slog.Info("discord", "tool_call", calls[0].Name, "operation", op, "first", f, "second", s, "result", result)
			name += fmt.Sprintf("(operation=%s, first=%s, second=%s) = %s", op, f, s, result)
		case "get_today_date_current_clock_time":
			result = tools.GetTodayClockTime()
			slog.Info("discord", "tool_call", calls[0].Name, "result", result)
			name += fmt.Sprintf("() = %s", result)
		default:
			slog.Warn("discord", "message", "unknown tool", "line", line, "calls", calls)
		}
		// See MistralRequestValidatorV3._validate_tool_message() in
		// https://github.com/mistralai/mistral-common/blob/main/src/mistral_common/protocol/instruct/validator.py
		callid := 0
		for _, c := range c.Messages {
			if c.Role == genaiapi.ToolCall {
				callid++
			}
		}
		res := tools.MistralToolCallResult{Content: result, CallID: fmt.Sprintf("c%08d", callid)}
		b, err := json.Marshal(res)
		if err != nil {
			slog.Error("discord", "tool", "json", "error", err)
			// Continue as if it wasn't a tool call.
			continue
		}
		// We want to ignore the rest of the reply and send a new query.
		// TODO: Inject CallID instead of pending[:i]. We need to determine if
		// Mistral prefers to receive its own content as-is or reformatted?
		c.Messages = append(c.Messages, genaiapi.Message{Role: genaiapi.ToolCall, Content: line})
		c.Messages = append(c.Messages, genaiapi.Message{Role: genaiapi.ToolCallResult, Content: string(b)})
		// TODO: We should probably cancel the context and start over, there's no
		// point in receiving more data.
		return name
	}
	return ""
}

// splitResponse take pending reply from the LLM and returns the amount of
// bytes to send back. Returns 0 if nothing should be sent.
//
// The basic assumption here is that most LLMs are trained to generate
// markdown, for the better or worst.
//
// If you spot a case where it doesn't work right in the wild, please fix and
// add a test case! Make sure it's 100% test coverage (except the panic).
func splitResponse(t string, urgent bool) (string, string) {
	rest := ""
	// First priority is 3 backticks. We must never break that in the middle,
	// unless it's longer than maxMessage.
	if backticks := strings.Count(t, "```"); backticks == 1 {
		// Trim everything before the backticks.
		i := strings.Index(t, "```")
		rest = t[i:]
		t = t[:i]
	} else if backticks >= 2 {
		// Cut right after the second.
		start := strings.Index(t, "```")
		end := strings.Index(t[start+3:], "```") + start + 3 + 3
		if end > maxMessage {
			// Dang we need to slice it. Look for empty lines to split at natural
			// places.
			if i := strings.LastIndex(t[:maxMessage], "\n\n"); i != -1 {
				// Inject a new 3 backticks to reconstruct the escaping.
				suffix := "\n```"
				prefix := "```\n"
				if tend := strings.Index(t[start:], "\n"); tend != -1 {
					// Take the original one as it may contain the highlighting style,
					// like ```python or ```bash.
					prefix = t[start : start+tend]
				}
				return t[:i+1] + suffix, prefix + t[i+1:]
			}
		}
		return t[:end], t[end:]
	}
	if t == "" {
		return t, rest
	}

	minLength := 50
	if urgent {
		minLength = 10
	}

	// If there's an empty line, use it.
	if i := strings.LastIndex(t, "\n\n"); i >= minLength && i < maxMessage {
		return t[:i+1], t[i+1:] + rest
	}
	// If there's a EOL, use it.
	if i := strings.LastIndexByte(t, '\n'); i >= minLength && i < maxMessage {
		return t[:i+1], t[i+1:] + rest
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
	if isEnum && !urgent {
		return "", t + rest
	}

	if len(t) < minLength {
		// Not enough characters to send, ignore.
		return "", t + rest
	}

	// If there's backticks, e.g. `foo.bar`, they mess up punctuation search. So
	// only start the search after the last backticks.
	start := 0
	if backticks := strings.Count(t, "`"); (backticks & 1) == 1 {
		// Impair number of backticks. Limit ourselves up to the last one.
		i := strings.LastIndexByte(t, '`')
		rest = t[i:]
		t = t[:i]
	}

	// TODO: Highlighting pairs: '*' and '_'
	m := punctuation.FindStringIndex(t[start:])
	if m == nil {
		return "", t + rest
	}
	end := start + m[1]
	if end > maxMessage {
		// Arbitrary cut.
		end = maxMessage
	}
	return t[:end], t[end:] + rest
}

// punctuation matches when it's ending the string or when it's followed by a
// whitespace. We don't need to handle \n (LF) since it's already handled
// earlier.
var punctuation = regexp.MustCompile(`[\.\?\!]($| )`)

// handleImage generates images based on the user prompt.
func (d *discordBot) handleImage(req intReq) {
	// Do it in a separate goroutine so we can send updates to the user as it
	// progresses. It provides a much better UX than batching all at once at the
	// end.
	type update struct {
		content string
		img     []byte
		err     error
	}
	ctx, cancel := context.WithCancel(d.ctx)
	defer cancel()
	updates := make(chan update, 10)
	go func() {
		defer close(updates)
		// Generate multiple images when the queue is empty.
		// First repeat what the user provided otherwise it's non-obvious for other
		// users.
		u := update{}
		if req.description != "" {
			u.content += "*Description*: " + escapeMarkdown(req.description) + "\n"
		}
		if req.imagePrompt != "" {
			u.content += "*Image prompt*: " + escapeMarkdown(req.imagePrompt) + "\n"
		}
		if req.labelsContent != "" {
			u.content += "*Labels*: " + escapeMarkdown(req.labelsContent) + "\n"
		}
		updates <- u
		for i := 0; i < 4 && ctx.Err() == nil; i++ {
			// Steps:
			// - Select seed if needed
			// - Generate labels if needed
			// - Generate image description, seeding description + labels if needed
			// - Generate the image if needed (meme_labels_auto is a special case)
			seed := req.seed
			if seed != 0 {
				// Increment by one for each loop.
				seed = seed + int64(i)
			} else {
				// Never pass seed 0, instead select a random seed ourself so the user
				// can still recreate the output. Generate a number between 1 and
				// 65000. It's unclear to me what the upper bound it. Do not use 65535
				// because we loop down below when generating labels.
				i, err := rand.Int(rand.Reader, big.NewInt(65000))
				if err != nil {
					u.err = fmt.Errorf("failed to generate seed: %w", err)
					updates <- u
					return
				}
				// We never want 0.
				seed = i.Int64() + 1
			}
			u.content += "*Image #" + strconv.Itoa(i+1) + "*: *Seed*: " + strconv.FormatInt(seed, 10) + "\n"

			// Labels: use the LLM to generate the labels based on the description.a
			labelsContent := req.labelsContent
			if req.cmdName == "meme_auto" || req.cmdName == "meme_labels_auto" {
				options := [3]string{}
				j := 0
				for ; j < len(options); j++ {
					msgs := []genaiapi.Message{{Role: genaiapi.System, Content: d.settings.PromptLabels}, {Role: genaiapi.User, Content: req.description}}
					// Intentionally limit the number of tokens, otherwise it's Stable
					// Diffusion that is unhappy.
					imgseed := seed + 4*int64(i) + 4*int64(j)
					opts := genaiapi.CompletionOptions{MaxTokens: 70, Seed: imgseed, Temperature: 1.0}
					newLabels, err := d.l.Prompt(ctx, msgs, &opts)
					if err != nil {
						u.err = fmt.Errorf("failed to enhance labels: %w", err)
						updates <- u
						return
					}
					options[j] = strings.Trim(newLabels, "\",.")
					// Is it good enough?
					if m, n := maxCommaLen(options[j]); n <= 3 && m < 30 {
						// Select this one.
						labelsContent = newLabels
						if i != 0 || j != 0 {
							u.content += "*Seed (label)*: " + strconv.FormatInt(imgseed, 10) + "\n"
						}
						break
					}
				}
				if j == len(options) {
					// No great option found, take a guess which is the less bad one.
					// TODO: We loose the seed when we sort.
					slices.SortFunc(options[:], memeLabelHeuristics)
					labelsContent = options[0]
				}
				u.content += "*Labels*: " + escapeMarkdown(labelsContent) + "\n"
				updates <- u
				if req.cmdName == "meme_labels_auto" {
					// "meme_labels_auto" is a special case where we don't actually need an image.
					continue
				}
			}

			imagePrompt := req.imagePrompt
			if req.cmdName == "meme_auto" || req.cmdName == "image_auto" {
				// Image: use the LLM to generate the image prompt based on the description.
				msgs := []genaiapi.Message{
					{Role: genaiapi.System, Content: d.settings.PromptImage},
					{Role: genaiapi.User, Content: "Prompt: " + req.description + "\n" + "Text relevant to the image: " + labelsContent},
				}
				opts := genaiapi.CompletionOptions{MaxTokens: 125, Seed: seed, Temperature: 1.0}
				if imagePrompt, u.err = d.l.Prompt(ctx, msgs, &opts); u.err != nil {
					u.err = fmt.Errorf("failed to enhance image generation prompt: %w", u.err)
					updates <- u
					return
				}
				imagePrompt = strings.TrimSpace(imagePrompt)
				imagePrompt = strings.ReplaceAll(imagePrompt, "\n", " ")
				imagePrompt = strings.ReplaceAll(imagePrompt, "  ", " ")
				if len(u.content)+len(imagePrompt) < maxMessage-100 {
					// We have to skip on these otherwise we hit the 2000 characters limit super fast.
					u.content += "*Image prompt*: " + escapeMarkdown(imagePrompt) + "\n"
					updates <- u
				}
			}

			// Generate the image.
			img, err := d.ig.GenImage(ctx, imagePrompt, seed)
			if err != nil {
				u.err = err
				updates <- u
				return
			}
			w := bytes.Buffer{}
			imagegen.DrawLabelsOnImage(img, labelsContent)
			u.err = jpeg.Encode(&w, img, nil)
			u.img = w.Bytes()
			updates <- u
			u.img = nil
			if u.err != nil {
				return
			}
			// Save it to disk. Don't fail the user in this case, log an error.
			data := map[string]interface{}{
				"channel":      req.int.ChannelID,
				"guild":        req.int.GuildID,
				"description":  req.description,
				"image_prompt": imagePrompt,
				"labels":       labelsContent,
				"seed":         seed,
				"command":      req.cmdName,
				"model":        d.l.Model,
			}
			if req.int.User != nil {
				data["user"] = req.int.User.Username
			}
			if req.int.Member != nil {
				data["user"] = req.int.Member.User.Username
			}
			b, err := json.Marshal(data)
			if err != nil {
				slog.Error("discord", "message", "failed marshaling metadata", "error", err)
			}
			p := filepath.Join(d.memDir, time.Now().Format("2006-01-02-15-04-05.000000"))
			if err2 := os.WriteFile(p+".json", b, 0o644); err2 != nil {
				slog.Error("discord", "message", "failed saving metadata", "error", err2)
				err = err2
			}
			// Create a new buffer.
			w = bytes.Buffer{}
			if err2 := png.Encode(&w, img); err2 != nil {
				slog.Error("discord", "message", "failed encoding png", "error", err2)
				err = err2
			}
			if err2 := os.WriteFile(p+".png", w.Bytes(), 0o644); err2 != nil {
				slog.Error("discord", "message", "failed saving png", "error", err2)
				err = err2
			}
			// If there were an error or there's another request pending, stop.
			if err != nil || len(d.image) != 0 || len(d.chat) != 0 {
				break
			}
		}
	}()

	// Then stream the updates to it feels more interactive. Throttle so discord
	// doesn't block us.
	const period = time.Second
	t := time.NewTicker(period)
	defer t.Stop()
	g := update{}
	hasUpdates := false
	var lastUpdate time.Time
	for {
		ok := false
		skip := false
		select {
		case <-t.C:
			if !hasUpdates {
				skip = true
				if err := d.dg.ChannelTyping(req.int.ChannelID); err != nil {
					slog.Error("discord", "message", "failed posting 'user typing'", "error", err)
					break
				}
			}
		case g, ok = <-updates:
			if !ok {
				return
			}
			hasUpdates = true
			if time.Since(lastUpdate) < period && g.err == nil && g.img == nil {
				// Throttle.
				skip = true
			}
		}
		if skip {
			continue
		}
		if g.err != nil {
			slog.Error("discord", "imagereq", req, "error", g.err)
			g.content += "\n*Error*: " + escapeMarkdown(g.err.Error()) + "\n"
		}
		resp := discordgo.WebhookEdit{Content: &g.content}
		if len(g.img) != 0 {
			resp.Files = []*discordgo.File{{Name: "prompt.jpg", ContentType: "image/jpeg", Reader: bytes.NewReader(g.img)}}
		}
		if _, err := d.dg.InteractionResponseEdit(req.int, &resp); err != nil {
			slog.Error("discord", "imagereq", req, "message", "failed posting interaction", "error", err)
		}
		if g.err != nil {
			return
		}
		hasUpdates = false
	}
}

func maxCommaLen(x string) (int, int) {
	m := 0
	parts := strings.Split(x, ",")
	for _, p := range parts {
		if l := len(p); l > m {
			m = l
		}
	}
	return m, len(parts)
}

// memeLabelHeuristics uses simple heuristics to decide which label is "best".
//
// TODO: Improve heuristics.
func memeLabelHeuristics(a, b string) int {
	ma, na := maxCommaLen(a)
	mb, nb := maxCommaLen(b)
	// We want 2 or 3 items, and then lower max length.
	pa := 0
	if na == 1 {
		pa = 1
	} else if na > 3 {
		pa = na
	}
	pb := 0
	if nb == 1 {
		pb = 1
	} else if nb > 3 {
		pb = nb
	}
	if pa != pb {
		if pa < pb {
			return -1
		}
		return 1
	}
	if ma < mb {
		return -1
	}
	if ma > mb {
		return 1
	}
	return 0
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
	seed          int64
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
