// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package main

import (
	"bytes"
	"context"
	"errors"
	"image/png"
	"io"
	"log"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/sillybot"
	"github.com/maruel/sillybot/imagegen"
	"github.com/maruel/sillybot/llm"
	"github.com/slack-go/slack"
	"github.com/slack-go/slack/slackevents"
	"github.com/slack-go/slack/socketmode"
)

type slackBot struct {
	api      *slack.Client
	sc       *socketmode.Client
	p        genai.ProviderGen
	mem      *llm.Memory
	ig       *imagegen.Session
	settings sillybot.Settings
	chat     chan msgReq
	image    chan *imgReq

	// Filled upon connection in onHello.
	botID  string
	userID string
	appID  string
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

func newSlackBot(apptoken, bottoken string, verbose bool, p genai.ProviderGen, mem *llm.Memory, ig *imagegen.Session, settings sillybot.Settings) (*slackBot, error) {
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
		api:      api,
		sc:       sc,
		p:        p,
		mem:      mem,
		ig:       ig,
		settings: settings,
		chat:     make(chan msgReq, 5),
		image:    make(chan *imgReq, 3),
	}
	return s, nil
}

func (s *slackBot) Run(ctx context.Context) error {
	slog.Info("slack", "state", "running")
	wg := sync.WaitGroup{}
	wg.Add(3)
	go func() {
		s.socketEventLoop(ctx)
		wg.Done()
	}()
	go func() {
		for req := range s.chat {
			if req.userid == "" {
				wg.Done()
				return
			}
			s.handlePrompt(ctx, req)
		}
	}()
	go func() {
		for req := range s.image {
			if req == nil {
				wg.Done()
				return
			}
			s.handleImage(ctx, req)
		}
	}()
	err := s.sc.RunContext(ctx)
	s.chat <- msgReq{}
	s.image <- nil
	wg.Wait()
	return err
}

func (s *slackBot) socketEventLoop(ctx context.Context) {
	slog.Info("slack", "state", "eventloop")
	done := ctx.Done()
	for {
		select {
		case evt := <-s.sc.Events:
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
		case <-done:
			return
		}
	}
}

// handleSocketEvent parses the event and route it to the right handler.
func (s *slackBot) handleSocketEvent(ctx context.Context, evt socketmode.Event) {
	switch evt.Type {
	case "incoming_error":
		// It's generally when the websocket is turned down on shutdown so log as
		// info.
		slog.Info("slack", "event", evt.Type, "payload", evt)
	case "connecting", "connected", "connection_error":
		// These are pseudo-events that are sent by socketmode when the websocket
		// is changing state.
		slog.Info("slack", "state", evt.Type)
	case "hello":
		s.onHello(ctx, evt)
	case "events_api":
		eventsAPIEvent, ok := evt.Data.(slackevents.EventsAPIEvent)
		if !ok {
			slog.Error("slack", "event", evt.Type, "error", "unknown", "payload", evt)
			return
		}
		s.onEventsAPI(ctx, evt, eventsAPIEvent)
	case "interactive":
		callback, ok := evt.Data.(slack.InteractionCallback)
		if !ok {
			slog.Error("slack", "event", evt.Type, "error", "unknown", "payload", evt)
			return
		}
		s.onInteractive(ctx, evt, callback)
	case "slash_commands":
		cmd, ok := evt.Data.(slack.SlashCommand)
		if !ok {
			slog.Error("slack", "event", evt.Type, "error", "unknown", "payload", evt)
			return
		}
		s.onSlashCommand(ctx, evt, cmd)
	default:
		slog.Error("slack", "event", evt.Type, "error", "unknown event", "payload", evt)
	}
}

// onHello handles the very first message sent by the server which is "hello".
func (s *slackBot) onHello(ctx context.Context, evt socketmode.Event) {
	slog.Info("slack", "payload", evt.Type)
	// TODO: This number seems to be all over the place. Maybe we don't do
	// teardown correctly?
	if evt.Request.NumConnections != 1 {
		slog.Warn("slack", "error", "more than one active connection!", "num_connections", evt.Request.NumConnections)
	}
	a, err := s.api.AuthTest()
	if err != nil {
		slog.Error("slack", "event", evt.Type, "error", err)
		return
	}
	slog.Info("slack", "event", evt.Type, "user", a.User, "userid", a.UserID, "botid", a.BotID)
	s.botID = a.BotID
	s.userID = a.UserID
	s.appID = evt.Request.ConnectionInfo.AppID

	// TODO: Have the code here create the slash commands automatically instead
	// of requiring the user to add them manually.
	//
	// An app configuration token can be created per Workspace at
	// https://api.slack.com/reference/manifests#config-tokens
	// The access token has 12h lifetime.
	//
	// m, err := s.api.ExportManifestContext(ctx, "<token>", s.appID)
	// if err != nil {
	// 	slog.Error("slack", "message", "failed to get application manifest", "error", err)
	// } else {
	//  m.Features.SlashCommands = []slack.ManifestSlashCommand{...}
	// 	resp, err := s.api.UpdateManifestContext(ctx, m, "<token>", s.appID)
	// 	slog.Error("slack", "manifest", m)
	// }
}

// onEventsAPI handle an incoming event.
func (s *slackBot) onEventsAPI(ctx context.Context, evt socketmode.Event, eventsAPIEvent slackevents.EventsAPIEvent) {
	if eventsAPIEvent.Type != "event_callback" {
		slog.Warn("slack", "event", evt.Type, "subevent", eventsAPIEvent.Type, "payload", evt)
		return
	}

	// The list is at https://api.slack.com/events?filter=Events
	switch ev := eventsAPIEvent.InnerEvent.Data.(type) {
	case *slackevents.AppHomeOpenedEvent:
		slog.Info("slack", "event", evt.Type, "subevent", eventsAPIEvent.Type, "subevent2", ev.Type, "user", ev.User, "channel", ev.Channel, "tab", ev.Tab)
	case *slackevents.AppMentionEvent:
		// Public message where the app is @ mentioned.
		slog.Info("slack", "event", evt.Type, "subevent", eventsAPIEvent.Type, "subevent2", eventsAPIEvent.InnerEvent.Type, "user", ev.User, "text", ev.Text, "channel", ev.Channel)
		if ev.User == s.userID {
			// We posted something.
			return
		}
		req := msgReq{
			msg:     ev.Text,
			userid:  ev.User,
			channel: ev.Channel,
			ts:      ev.TimeStamp,
		}
		s.onAppMention(ctx, req)
	case *slackevents.MemberJoinedChannelEvent:
		slog.Info("slack", "event", evt.Type, "subevent", eventsAPIEvent.Type, "subevent2", eventsAPIEvent.InnerEvent.Type, "user", ev.User, "channel", ev.Channel)
	case *slackevents.MessageEvent:
		// Private message.
		slog.Info("slack", "event", evt.Type, "subevent", eventsAPIEvent.Type, "subevent2", eventsAPIEvent.InnerEvent.Type, "user", ev.User, "channel", ev.Channel, "text", ev.Text)
		if ev.User == s.userID {
			// We posted something.
			return
		}
		req := msgReq{
			msg:     ev.Text,
			userid:  ev.User,
			channel: ev.Channel,
			// Don't set TS so it's not threaded for direct messages.
			// ts:      ev.TimeStamp,
		}
		s.onAppMention(ctx, req)
	default:
		slog.Warn("slack", "event", evt.Type, "subevent", eventsAPIEvent.Type, "subevent2", eventsAPIEvent.InnerEvent.Type, "error", "unknown", "payload", evt)
	}
}

// onAppMention handles a message where the bot is explicitly mentioned in the
// message.
func (s *slackBot) onAppMention(ctx context.Context, req msgReq) {
	user := "<@" + s.userID + ">"
	req.msg = strings.TrimSpace(strings.ReplaceAll(req.msg, user, ""))
	if s.p == nil {
		if _, _, err := s.sc.PostMessageContext(ctx, req.channel, slack.MsgOptionText("LLM is not enabled.", false), slack.MsgOptionTS(req.ts)); err != nil {
			slog.Error("slack", "message", "failed posting message", "error", err)
		}
		return
	}
	// Slack doesn't have a way to have a bot give feedback that it is processing
	// the input. https://api.slack.com/events/user_typing is only available in
	// the old deprecated API.
	// Ref:
	// - https://github.com/slackapi/bolt-js/issues/885
	// - https://forums.slackcommunity.com/s/question/0D53a00008OS6wqCAD
	select {
	case s.chat <- req:
	default:
		if _, _, err := s.sc.PostMessageContext(ctx, req.channel, slack.MsgOptionText("Sorry! I have too many pending chat requests. Please retry in a moment.", false), slack.MsgOptionTS(req.ts)); err != nil {
			slog.Error("slack", "message", "failed posting message", "error", err)
		}
	}
}

// handlePrompt uses the LLM to generate a response.
func (s *slackBot) handlePrompt(ctx context.Context, req msgReq) {
	c := s.mem.Get(req.userid, req.channel)
	_, ts, err := s.sc.PostMessageContext(ctx, req.channel, slack.MsgOptionText("(generating)", false), slack.MsgOptionTS(req.ts))
	if err != nil {
		slog.Error("slack", "message", "failed posting message", "error", err)
	}
	c.Messages = append(c.Messages, genai.NewTextMessage(req.msg))
	chunks := make(chan genai.ReplyFragment)
	wg := sync.WaitGroup{}
	wg.Add(1)
	go func() {
		pending := ""
		text := ""
		// The API is Tier 3, which means the limit is 50 times per minutes. Limit
		// ourselves to 30/min plus the other messages.
		t := time.NewTicker(2 * time.Second)
		for {
			select {
			case pkt, ok := <-chunks:
				if !ok {
					if pending != "" {
						text += pending
						if _, _, err2 := s.sc.PostMessageContext(ctx, req.channel, slack.MsgOptionUpdate(ts), slack.MsgOptionText(text, false), slack.MsgOptionTS(req.ts)); err2 != nil {
							slog.Error("slack", "message", "failed posting message", "error", err2)
						}
					}
					// Remember LLM's answer.
					t.Stop()
					wg.Done()
					return
				}
				if pkt.TextFragment == "" {
					slog.Error("slack", "pkt", pkt)
				}
				pending += pkt.TextFragment
			case <-t.C:
				// Don't send one word at a time.
				if len(pending) > 30 {
					text += pending
					if _, _, err2 := s.sc.PostMessageContext(ctx, req.channel, slack.MsgOptionUpdate(ts), slack.MsgOptionText(text+" (...generating)", false), slack.MsgOptionTS(req.ts)); err2 != nil {
						slog.Error("slack", "message", "failed posting message", "error", err2)
					}
					pending = ""
				}
			}
		}
	}()
	// We're chatting, we don't want too much content.
	opts := genai.OptionsText{
		MaxTokens:    2000,
		SystemPrompt: s.settings.PromptImage,
	}
	result, err := s.p.GenStream(ctx, c.Messages, chunks, &opts)
	close(chunks)
	wg.Wait()
	if err != nil && !errors.Is(err, context.Canceled) {
		if _, _, err = s.sc.PostMessageContext(ctx, req.channel, slack.MsgOptionUpdate(ts), slack.MsgOptionText("Prompt generation failed: "+err.Error(), false), slack.MsgOptionTS(req.ts)); err != nil {
			slog.Error("slack", "message", "failed posting message", "error", err)
		}
	} else {
		// TODO: Handle non-text reply.
		c.Messages = append(c.Messages, result.Message)
	}
}

// handleImage generates an image based on the user prompt.
func (s *slackBot) handleImage(ctx context.Context, req *imgReq) {
	req.mu.Lock()
	// Dummy command while holding the lock so the linter doesn't complain.
	msg := req.msg
	req.mu.Unlock()
	// Use the LLM to improve the prompt!
	if s.p != nil {
		msgs := genai.Messages{genai.NewTextMessage(req.msg)}

		// Intentionally limit the number of tokens, otherwise it's Stable
		// Diffusion that is unhappy.
		opts := genai.OptionsText{
			MaxTokens:    70,
			SystemPrompt: s.settings.PromptImage,
		}
		if result, err := s.p.GenSync(ctx, msgs, &opts); err != nil {
			slog.Error("discord", "message", "failed to enhance prompt", "error", err)
		} else {
			msg = result.AsText()
		}
	}
	// TODO: Generate multiple images when the queue is empty?
	img, err := s.ig.GenImage(ctx, msg, 1)
	if err != nil {
		_, _, _, err = s.sc.SendMessageContext(
			ctx, req.channel,
			slack.MsgOptionResponseURL(req.responseURL, slack.ResponseTypeInChannel),
			slack.MsgOptionText("Image generation failed: "+err.Error(), false),
		)
		if err != nil {
			slog.Error("slack", "message", "failed posting message", "error", err)
		}
		return
	}
	w := bytes.Buffer{}
	if err = png.Encode(&w, img); err != nil {
		slog.Error("slack", "message", "failed encoding PNG", "error", err)
		return
	}
	// TODO: Figure out how to use a block instead to include the generation request.
	param := slack.UploadFileV2Parameters{
		Title:    req.username + " asked for: " + req.msg,
		Filename: "image.png",
		FileSize: w.Len(),
		Reader:   &w,
		Channel:  req.channel,
	}
	if _, err = s.sc.UploadFileV2Context(ctx, param); err != nil {
		slog.Error("slack", "message", "failed posting message", "error", err)
	}
}

// onSlashCommand handles a "/foo" style command.
//
// They are configured at https://api.slack.com/apps/<appid>/slash-commands
func (s *slackBot) onSlashCommand(ctx context.Context, evt socketmode.Event, cmd slack.SlashCommand) {
	slog.Info("slack", "event", evt.Type, "user", cmd.UserID, "username", cmd.UserName, "channel", cmd.ChannelID, "command", cmd.Command, "text", cmd.Text)
	switch cmd.Command {
	case "/forget":
		c := s.mem.Get(cmd.UserID, cmd.ChannelID)
		// TODO: When in a channel, prefix with the user?
		reply := "I don't know you. I can't wait to start our discussion so I can get to know you better!"
		slog.Info("slack", "user", cmd.UserID, "channel", cmd.ChannelID, "forgetting", len(c.Messages))
		if len(c.Messages) > 1 {
			c.Messages = c.Messages[:1]
			reply = "The memory of our past conversations just got zapped."
		}
		_, _, _, err := s.sc.SendMessageContext(
			ctx, cmd.ChannelID,
			slack.MsgOptionResponseURL(cmd.ResponseURL, slack.ResponseTypeInChannel),
			slack.MsgOptionText(reply, false),
		)
		if err != nil {
			slog.Error("slack", "message", "failed posting message", "error", err)
		}
	case "/image":
		if s.ig == nil {
			str := "Image generation is not enabled. Restart with bot.image_gen.model set in config.yml."
			_, _, _, err := s.sc.SendMessageContext(
				ctx, cmd.ChannelID,
				slack.MsgOptionResponseURL(cmd.ResponseURL, slack.ResponseTypeInChannel),
				slack.MsgOptionText(str, false),
			)
			if err != nil {
				slog.Error("slack", "message", "failed posting message", "error", err)
			}
			return
		}
		// TODO: Create a block with https://upload.wikimedia.org/wikipedia/commons/b/b1/Loading_icon.gif
		req := &imgReq{
			msg:         cmd.Text,
			username:    cmd.UserName,
			channel:     cmd.ChannelID,
			responseURL: cmd.ResponseURL,
		}
		req.mu.Lock()
		select {
		case s.image <- req:
			_, _, _, err := s.sc.SendMessageContext(
				ctx,
				cmd.ChannelID,
				slack.MsgOptionResponseURL(cmd.ResponseURL, slack.ResponseTypeEphemeral),
				slack.MsgOptionText("Generating ...", false),
			)
			if err != nil {
				slog.Error("slack", "message", "failed posting message", "error", err)
			}
			req.mu.Unlock()
		default:
			req.mu.Unlock()
			str := "Sorry! I have too many pending image requests. Please retry in a moment."
			if _, _, err := s.sc.PostMessageContext(ctx, cmd.ChannelID, slack.MsgOptionText(str, false)); err != nil {
				slog.Error("slack", "message", "failed posting message", "error", err)
			}
		}
	default:
		slog.Warn("slack", "message", "unknown slash command", "command", cmd.Command, "text", cmd.Text)
	}
}

// onInteractive handles shortcuts and menus.
//
// They are configured at https://api.slack.com/apps/<appid>/interactive-messages
func (s *slackBot) onInteractive(ctx context.Context, evt socketmode.Event, callback slack.InteractionCallback) {
	switch callback.Type {
	case "shortcut":
		slog.Info("slack", "event", evt.Type, "callback", callback.Type, "name", callback.CallbackID)
	case "block_actions", "dialog_submission", "view_submission":
		slog.Info("slack", "event", evt.Type, "callback", callback.Type, "name", callback.Name)
	default:
		slog.Warn("slack", "event", evt.Type, "error", "unknown", "callback", callback.Type)
	}
}

// msgReq is a chat message request.
type msgReq struct {
	msg     string
	userid  string
	channel string
	ts      string
}

// imgReq is an image request.
type imgReq struct {
	mu          sync.Mutex
	msg         string
	username    string
	channel     string
	responseURL string
}
