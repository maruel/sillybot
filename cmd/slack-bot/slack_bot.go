// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package main

import (
	"bytes"
	"context"
	"errors"
	"io"
	"log"
	"log/slog"
	"strings"

	"github.com/maruel/sillybot"
	"github.com/slack-go/slack"
	"github.com/slack-go/slack/slackevents"
	"github.com/slack-go/slack/socketmode"
)

type slackBot struct {
	api          *slack.Client
	sc           *socketmode.Client
	l            *sillybot.LLM
	ig           *sillybot.ImageGen
	botID        string
	userID       string
	systemPrompt string
}

type slackLogger struct {
	p string
}

func (s *slackLogger) Output(l int, msg string) error {
	slog.Info(s.p, "msg", msg)
	return nil
}

type ilog interface {
	Output(int, string) error
}

func newSlackBot(apptoken, bottoken string, verbose bool, l *sillybot.LLM, ig *sillybot.ImageGen) (*slackBot, error) {
	if !strings.HasPrefix(apptoken, "xapp-") {
		return nil, errors.New("slack apptoken must have the prefix \"xapp-\"")
	}
	if !strings.HasPrefix(bottoken, "xoxb-") {
		return nil, errors.New("slack bottoken must have the prefix \"xoxb-\"")
	}
	var out ilog = log.New(io.Discard, "", 0)
	if verbose {
		out = &slackLogger{p: "slack/api"}
	}
	api := slack.New(
		bottoken,
		slack.OptionDebug(verbose),
		slack.OptionLog(out),
		slack.OptionAppLevelToken(apptoken),
	)

	if verbose {
		out = &slackLogger{p: "slack/socketmode"}
	}
	sc := socketmode.New(api, socketmode.OptionDebug(verbose), socketmode.OptionLog(out))
	s := &slackBot{
		api:          api,
		sc:           sc,
		l:            l,
		ig:           ig,
		systemPrompt: "You are a terse assistant. You reply with short answers. You are often joyful, sometimes humorous, sometimes sarcastic.",
	}
	return s, nil
}

func (s *slackBot) Run(ctx context.Context) error {
	slog.Info("slack", "state", "running")
	go s.socketEventLoop(ctx)
	return s.sc.RunContext(ctx)
}

func (s *slackBot) socketEventLoop(ctx context.Context) {
	slog.Info("slack", "state", "eventloop")
	for evt := range s.sc.Events {
		// Acknowledge the message even before processing it otherwise Slack will
		// resend the message if we take more [unknown but less than 5] seconds or
		// so to process it.
		if evt.Request != nil {
			slog.Debug("slack", "type", "ack", "envelopeid", evt.Request.EnvelopeID)
			if evt.Request.EnvelopeID != "" {
				var pld interface{}
				if err := s.sc.AckCtx(ctx, evt.Request.EnvelopeID, pld); err != nil {
					slog.Error("slack", "type", "ack", "error", err)
				}
			}
		} else {
			slog.Debug("slack", "type", "no_ack")
		}
		s.handleSocketEvent(ctx, evt)
	}
}

func (s *slackBot) handleSocketEvent(ctx context.Context, evt socketmode.Event) {
	switch evt.Type {
	case "incoming_error":
		// It's generally when the websocket is turned down on shutdown so log as
		// info.
		slog.Info("slack", "event", evt.Type, "payload", evt)
	case "connecting", "connection_error":
		slog.Info("slack", "state", evt.Type)
	case "hello":
		slog.Info("slack", "payload", evt.Type)
		if evt.Request.NumConnections != 1 {
			slog.Warn("slack", "error", "more than one active connection!", "num_connections", evt.Request.NumConnections)
		}
	case "connected":
		a, err := s.api.AuthTest()
		if err != nil {
			slog.Error("slack", "event", evt.Type, "error", err)
			return
		}
		slog.Info("slack", "event", evt.Type, "user", a.User, "userid", a.UserID, "botid", a.BotID)
		s.botID = a.BotID
		s.userID = a.UserID
	case "events_api":
		eventsAPIEvent, ok := evt.Data.(slackevents.EventsAPIEvent)
		if !ok {
			slog.Error("slack", "event", evt.Type, "error", "unknown", "payload", evt)
			return
		}
		switch eventsAPIEvent.Type {
		case "event_callback":
			switch ev := eventsAPIEvent.InnerEvent.Data.(type) {
			case *slackevents.AppMentionEvent:
				slog.Info("slack", "event", evt.Type, "subevent", eventsAPIEvent.Type, "subevent2", eventsAPIEvent.InnerEvent.Type, "user", ev.User, "text", ev.Text, "channel", ev.Channel)
				if ev.User == s.userID {
					// We posted something.
					return
				}
				s.handleAppMention(ctx, ev)
			case *slackevents.MemberJoinedChannelEvent:
				slog.Info("slack", "event", evt.Type, "subevent", eventsAPIEvent.Type, "subevent2", eventsAPIEvent.InnerEvent.Type, "user", ev.User, "channel", ev.Channel)
			case *slackevents.MessageEvent:
				slog.Info("slack", "event", evt.Type, "subevent", eventsAPIEvent.Type, "subevent2", eventsAPIEvent.InnerEvent.Type, "user", ev.User, "channel", ev.Channel, "text", ev.Text)
			default:
				slog.Warn("slack", "event", evt.Type, "subevent", eventsAPIEvent.Type, "subevent2", eventsAPIEvent.InnerEvent.Type, "error", "unknown", "payload", evt)
			}
		default:
			slog.Warn("slack", "event", evt.Type, "subevent", eventsAPIEvent.Type, "payload", evt)
		}
	case "interactive":
		callback, ok := evt.Data.(slack.InteractionCallback)
		if !ok {
			slog.Error("slack", "event", evt.Type, "error", "unknown", "payload", evt)
			return
		}
		switch callback.Type {
		case "block_actions", "dialog_submission", "shortcut", "view_submission":
			slog.Info("slack", "event", evt.Type, "callback", callback.Type)
		default:
			slog.Warn("slack", "event", evt.Type, "error", "unknown", "callback", callback.Type)
		}
	case "slash_commands":
		cmd, ok := evt.Data.(slack.SlashCommand)
		if !ok {
			slog.Error("slack", "event", evt.Type, "error", "unknown", "payload", evt)
			return
		}
		slog.Info("slack", "event", evt.Type, "user", cmd.UserID, "channel", cmd.ChannelID, "command", cmd.Command)
	default:
		slog.Error("slack", "event", evt.Type, "error", "unknown event", "payload", evt)
	}
}

// handleAppMention handles a message where the bot is explicitly mentioned in
// the message.
func (s *slackBot) handleAppMention(ctx context.Context, ev *slackevents.AppMentionEvent) {
	user := "<@" + s.userID + ">"
	msg := strings.TrimSpace(strings.ReplaceAll(ev.Text, user, ""))
	if strings.HasPrefix(msg, "image:") {
		s.handleImage(ctx, ev, strings.TrimSpace(msg[len("image:"):]))
		return
	}
	s.handlePrompt(ctx, ev, msg)
}

// handlePrompt uses the LLM to generate a response.
func (s *slackBot) handlePrompt(ctx context.Context, ev *slackevents.AppMentionEvent, msg string) {
	if s.l == nil {
		if _, _, err := s.sc.PostMessageContext(ctx, ev.Channel, slack.MsgOptionText("LLM is not enabled.", false)); err != nil {
			slog.Error("slack", "event", "failed posting message", "error", err)
		}
		return
	}

	// TODO: Keep log of previous messages.
	msgs := []sillybot.Message{
		{Role: sillybot.System, Content: s.systemPrompt},
		{Role: sillybot.User, Content: msg},
	}
	reply, err := s.l.Prompt(ctx, msgs)
	if err != nil {
		_, _, err = s.sc.PostMessageContext(ctx, ev.Channel, slack.MsgOptionText("Prompt generation failed: "+err.Error(), false))
		if err != nil {
			slog.Error("slack", "event", "failed posting message", "error", err)
		}
		return
	}

	if _, _, err = s.sc.PostMessageContext(ctx, ev.Channel, slack.MsgOptionText(reply, false)); err != nil {
		slog.Error("slack", "event", "failed posting message", "error", err)
	}
}

// handleImage generates an image based on the user prompt.
func (s *slackBot) handleImage(ctx context.Context, ev *slackevents.AppMentionEvent, msg string) {
	if s.ig == nil {
		if _, _, err := s.sc.PostMessageContext(ctx, ev.Channel, slack.MsgOptionText("Image generation is not enabled. Restart with flag \"-ig\"", false)); err != nil {
			slog.Error("slack", "event", "failed posting message", "error", err)
		}
		return
	}
	// Slack doesn't have a way to have a bot give feedback that it is processing
	// the input. https://api.slack.com/events/user_typing is only available in
	// the old deprecated API.
	// Ref:
	// - https://github.com/slackapi/bolt-js/issues/885
	// - https://forums.slackcommunity.com/s/question/0D53a00008OS6wqCAD
	//
	// TODO: Insert a stand-in, then replace it.
	// TODO: Generate multiple images.
	p, err := s.ig.GenImage(msg)
	if err != nil {
		if _, _, err = s.sc.PostMessageContext(ctx, ev.Channel, slack.MsgOptionText("Image generation failed: "+err.Error(), false)); err != nil {
			slog.Error("slack", "event", "failed posting message", "error", err)
		}
		return
	}

	param := slack.UploadFileV2Parameters{
		Title:    "Image",
		Filename: "image.png",
		FileSize: len(p),
		Reader:   bytes.NewReader(p),
		Channel:  ev.Channel,
	}
	if _, err = s.sc.UploadFileV2Context(ctx, param); err != nil {
		slog.Error("slack", "event", "failed posting message", "error", err)
	}
}
