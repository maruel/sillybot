// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package main

import (
	"context"
	"encoding/json"
	"log/slog"
	"sync"
	"time"

	"github.com/bluesky-social/indigo/api/bsky"
	"github.com/maruel/sillybot"
	"github.com/maruel/sillybot/imagegen"
	"github.com/maruel/sillybot/llm"
)

// Maximum blue sky message length.
//
// https://github.com/bluesky-social/atproto/blob/main/lexicons/app/bsky/feed/post.json
const maxMessage = 300

// bskyBot is the live instance of the bot talking to the Blue Sky ATproto API.
type bskyBot struct {
	c         *Client
	l         *llm.Session
	knownLLMs []llm.KnownLLM
	ig        *imagegen.Session
	settings  sillybot.Settings
}

func newBskyBot(ctx context.Context, user, pass string, l *llm.Session, knownLLMs []llm.KnownLLM, ig *imagegen.Session, settings sillybot.Settings) (*bskyBot, error) {
	c, err := New(ctx, user, pass)
	if err != nil {
		return nil, err
	}
	slog.Info("bsky", "state", "connected")
	b := &bskyBot{c: c, l: l, knownLLMs: knownLLMs, ig: ig, settings: settings}
	slog.Info("bsky", "state", "running", "info", "Press CTRL-C to exit.")
	return b, nil
}

func (b *bskyBot) Close() error {
	slog.Info("bsky", "state", "terminating")
	return nil
}

func (b *bskyBot) ProcessReplies(ctx context.Context) error {
	/*
		feed, _, err := c.GetTimeline(ctx, "", 5)
		if err != nil {
			return err
		}
		for _, post := range feed {
			b, _ := json.Marshal(post.Post)
			slog.Info("bsky", "context", post.FeedContext, "post", string(b), "reason", post.Reason, "reply", post.Reply)
		}
		if err = c.SearchPosts(ctx, c.client.Auth.Did); err != nil {
			return err
		}
	*/
	wg := sync.WaitGroup{}

	wg.Add(1)
	chProcess := make(chan intReq, 3)
	go func() {
		defer wg.Done()
		for msg := range chProcess {
			if ctx.Err() != nil {
				continue
			}
			b.processImgRequest(ctx, msg)
		}
	}()
	defer close(chProcess)

	wg.Add(1)
	chPost := make(chan *bsky.FeedPost)
	go func() {
		defer wg.Done()
		for p := range chPost {
			// This loop processes every messages appearing on BlueSky. This is a lot. It cannot block.
			if ctx.Err() == nil {
				continue
			}
			if reply := b.processPost(p, chProcess); reply != "" {
				wg.Add(1)
				go func() {
					defer wg.Done()
					d, _ := json.Marshal(p)
					slog.Warn("bsky", "mention", string(d), "msg", reply)
					r := Post{Text: reply}
					if _, _, err2 := b.c.Post(ctx, &r); err2 != nil {
						slog.Error("bsky", "mention", string(d), "overloaded", true, "err", err2)
					}
				}()
			}
		}
	}()
	defer close(chPost)

	return b.c.Listen(ctx, "", chPost)
}

func (b *bskyBot) processPost(p *bsky.FeedPost, ch chan<- intReq) string {
	t, err := time.Parse("2006-01-02T15:04:05.999Z", p.CreatedAt)
	if err != nil {
		slog.Warn("bsky", "bad time", p.CreatedAt)
		return ""
	}
	if d := time.Since(t); d < 0 || d > time.Hour {
		slog.Warn("bsky", "old time", p.CreatedAt)
		return ""
	}
	// If p.Embed contains an image?
	msg := intReq{description: p.Text, seed: 1, replyToCID: "", replyToURI: ""}
	select {
	case ch <- msg:
		return ""
	default:
		// Couldn't send, reply to the user.
		return "Sorry, I am overloaded! Please try again later."
	}
}

func (b *bskyBot) processImgRequest(ctx context.Context, msg intReq) error {
	slog.Info("bsky", "msg", msg)
	r := Post{Text: "Here's an image.", ReplyToCID: msg.replyToCID, ReplyToURI: msg.replyToURI}
	if _, _, err := b.c.Post(ctx, &r); err != nil {
		return err
	}
	return nil
}

// intReq is an interaction request to generate an image.
type intReq struct {
	description string
	replyToCID  string
	replyToURI  string
	seed        int
}
