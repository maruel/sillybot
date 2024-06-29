package main

import (
	"context"
	"fmt"
	"strings"

	"github.com/bwmarrin/discordgo"
)

type bot struct {
	l *llm
}

func newBot(ctx context.Context, dg *discordgo.Session) (*bot, error) {
	l, err := newLLM(ctx)
	if err != nil {
		return nil, err
	}
	b := &bot{l: l}
	_ = dg.AddHandler(b.guildCreate)
	_ = dg.AddHandler(b.messageCreate)
	_ = dg.AddHandler(b.ready)
	dg.Identify.Intents = discordgo.IntentsGuilds | discordgo.IntentsGuildMessages | discordgo.IntentGuildPresences | discordgo.IntentDirectMessages
	return b, nil
}

func (b *bot) Close() error {
	if b.l != nil {
		return b.l.Close()
	}
	return nil
}

// A new message is created on any channel that the authenticated bot has
// access to.
func (b *bot) messageCreate(s *discordgo.Session, m *discordgo.MessageCreate) {
	logger.Debug("messageCreate", "event", m.Message, "state", s.State)
	// Ignore all messages created by the bot itself. This isn't required in this
	// specific example but it's a good practice.
	botid := s.State.User.ID
	if m.Author.ID == botid {
		return
	}
	content := strings.TrimSpace(strings.TrimPrefix(m.Content, fmt.Sprintf("<@%s>", botid)))
	logger.Info("messageCreate", "author", m.Author.Username, "message", content)
	var err error
	switch content {
	case "ping":
		_, err = s.ChannelMessageSend(m.ChannelID, "Pong!")
	case "pong":
		_, err = s.ChannelMessageSend(m.ChannelID, "Ping!")
	default:
		resp := ""
		if resp, err = b.l.prompt(content); err != nil {
			logger.Error("prompt failure", "prompt", content, "err", err)
			_, _ = s.ChannelMessageSend(m.ChannelID, "Failed: "+err.Error())
		} else {
			_, err = s.ChannelMessageSend(m.ChannelID, resp)
		}
	}
	if err != nil {
		// If an error occurred, we failed to send the message. It may occur either
		// when we do not share a server with the user (highly unlikely as we just
		// received a message) or the user disabled DM in their settings (more
		// likely).
		fmt.Println("error sending DM message:", err)
	}
}

// A new guild is joined.
func (b *bot) guildCreate(s *discordgo.Session, event *discordgo.GuildCreate) {
	logger.Debug("guildCreate", "event", event.Guild)
	logger.Info("guildCreate", "name", event.Guild.Name)
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
	logger.Debug("ready", "session", s, "event", r)
	logger.Info("ready", "user", r.User.String())
}
