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
		s.handleSocketEvent(ctx, evt)
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
	}
}

func (s *slackBot) handleSocketEvent(ctx context.Context, evt socketmode.Event) {
	switch evt.Type {
	case "incoming_error":
		slog.Error("slack", "event", evt.Type, "payload", evt)
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
				msg := strings.TrimSpace(strings.TrimPrefix(ev.Text, "<@"+s.userID+">"))
				var err error
				if strings.HasPrefix(msg, "image:") {
					if s.ig == nil {
						_, _, err = s.sc.PostMessage(ev.Channel, slack.MsgOptionText("Image generation is not enabled. ", false))
					} else {
						msg = strings.TrimSpace(strings.TrimPrefix(msg, "image:"))
						var p []byte
						if p, err = s.ig.GenImage(msg); err == nil {
							param := slack.UploadFileV2Parameters{
								Title:    "Image",
								Filename: "image.png",
								FileSize: len(p),
								Reader:   bytes.NewReader(p),
								Channel:  ev.Channel,
							}
							_, err = s.sc.UploadFileV2Context(ctx, param)
						}
					}
				} else {
					if s.l == nil {
						_, _, err = s.sc.PostMessage(ev.Channel, slack.MsgOptionText("LLM is not enabled. ", false))
					} else {
						reply := ""
						msgs := []sillybot.Message{
							{Role: sillybot.System, Content: s.systemPrompt},
							{Role: sillybot.User, Content: msg},
						}
						if reply, err = s.l.Prompt(ctx, msgs); err == nil {
							_, _, err = s.sc.PostMessage(ev.Channel, slack.MsgOptionText(reply, false))
						} else {
							_, _, err = s.sc.PostMessage(ev.Channel, slack.MsgOptionText("ERROR: "+err.Error(), false))
						}
					}
				}
				if err != nil {
					slog.Error("slack", "event", "failed posting message", "error", err)
				}
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
