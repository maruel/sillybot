// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"reflect"
	"strings"
	"time"

	"github.com/bluesky-social/indigo/api/atproto"
	"github.com/bluesky-social/indigo/api/bsky"
	"github.com/bluesky-social/indigo/events"
	"github.com/bluesky-social/indigo/events/schedulers/sequential"
	"github.com/bluesky-social/indigo/lex/util"
	"github.com/bluesky-social/indigo/repo"
	"github.com/bluesky-social/indigo/repomgr"
	"github.com/bluesky-social/indigo/xrpc"
	"github.com/gorilla/websocket"
	"github.com/ipfs/go-cid"
)

// Client is the AT protocol client.
type Client struct {
	client *xrpc.Client
}

// New returns a new blue sky client.
//
// Be conscious of the rate limits at https://docs.bsky.app/docs/advanced-guides/rate-limits
func New(ctx context.Context, handle, pwd string) (*Client, error) {
	c := &Client{client: &xrpc.Client{Host: "https://bsky.social"}}
	// TODO: Can we cache the AccessJwt and RefreshJwt instead of the password? This requires caching the Did too.
	cred := &atproto.ServerCreateSession_Input{Identifier: handle, Password: pwd}
	s, err := atproto.ServerCreateSession(ctx, c.client, cred)
	if err != nil {
		return nil, fmt.Errorf("failed to authenticate: %w", err)
	}
	c.client.Auth = &xrpc.AuthInfo{AccessJwt: s.AccessJwt, RefreshJwt: s.RefreshJwt, Handle: s.Handle, Did: s.Did}
	return c, nil
}

// Post posts a new message to the user's feed and returns the Cid and URI.
func (c *Client) Post(ctx context.Context, p *Post) (string, string, error) {
	d, err := p.build()
	if err != nil {
		return "", "", err
	}
	data := &atproto.RepoCreateRecord_Input{Collection: "app.bsky.feed.post", Repo: c.client.Auth.Did, Record: &util.LexiconTypeDecoder{Val: d}}
	r, err := atproto.RepoCreateRecord(ctx, c.client, data)
	if err != nil {
		return "", "", fmt.Errorf("unable to post, %w", err)
	}
	return r.Cid, r.Uri, nil
}

// GetProfile returns a user profile.
func (c *Client) GetProfile(ctx context.Context, did string) (*bsky.ActorDefs_ProfileViewDetailed, error) {
	return bsky.ActorGetProfile(ctx, c.client, did)
}

// SeachPosts searches for recent posts.
//
// author must be a valid did or an handle with @.
func (c *Client) SearchPosts(ctx context.Context, author string) error {

	cursor := ""
	domain := ""
	lang := ""
	limit := int64(10)
	mentions := ""
	q := "Hello"
	since := ""
	sort := ""
	var tag []string
	until := ""
	url := ""
	// What the heck API. Probably saner to inline.
	// https://docs.bsky.app/docs/api/app-bsky-feed-search-posts
	res, err := bsky.FeedSearchPosts(ctx, c.client, author, cursor, domain, lang, limit, mentions, q, since, sort, tag, until, url)
	if err != nil {
		return err
	}
	slog.Info("bsky", "cursor", res.Cursor, "hits", res.HitsTotal)
	for _, p := range res.Posts {
		// Logging as-is just logs pointers.
		b, _ := json.Marshal(p)
		slog.Info("bsky", "post", string(b))
	}
	return nil
}

// Listen listens to all posts in the feed until the context is canceled.
func (c *Client) Listen(ctx context.Context, cursor string) error {
	u := "wss://bsky.network/xrpc/com.atproto.sync.subscribeRepos"
	if cursor != "" {
		u += "?cursor=" + url.QueryEscape(cursor)
	}
	conn, _, err := websocket.DefaultDialer.Dial(u, http.Header{})
	if err != nil {
		return fmt.Errorf("failed to listed: %w", err)
	}
	defer conn.Close()
	// TODO: We only want events relevant to our own account!
	return events.HandleRepoStream(ctx, conn, sequential.NewScheduler("stream", c.onListenEvent))
}

func (c *Client) onListenEvent(ctx context.Context, xev *events.XRPCStreamEvent) error {
	// https://atproto.com/specs/event-stream
	if xev.RepoCommit != nil {
		evt := xev.RepoCommit
		if evt.TooBig {
			slog.Warn("bsky", "skipping too big events", evt.Seq)
			return nil
		}
		r, err := repo.ReadRepoFromCar(ctx, bytes.NewReader(evt.Blocks))
		if err != nil {
			return fmt.Errorf("reading repo from car (seq: %d, len: %d): %w", evt.Seq, len(evt.Blocks), err)
		}
		for _, op := range evt.Ops {
			switch ek := repomgr.EventKind(op.Action); ek {
			case repomgr.EvtKindCreateRecord, repomgr.EvtKindUpdateRecord:
				rc, rec, err := r.GetRecord(ctx, op.Path)
				if err != nil {
					slog.Error("bsky", "path", op.Path, "cid", *op.Cid, "seq", evt.Seq, "repo", evt.Repo, "err", err)
					continue
				}
				if util.LexLink(rc) != *op.Cid {
					return fmt.Errorf("mismatch in record and op cid: %s != %s", rc, *op.Cid)
				}
				if err := c.onListenRecord(ctx, ek, evt.Seq, op.Path, evt.Repo, &rc, rec); err != nil {
					slog.Error("bsky", "kind", ek, "err", err)
					continue
				}
			case repomgr.EvtKindDeleteRecord:
				// Don't care.
			}
		}
	}
	return nil
}

func (c *Client) onListenRecord(ctx context.Context, op repomgr.EventKind, seq int64, path string, did string, rcid *cid.Cid, rec any) error {
	slog.Debug("bsky", "op", op, "seq", seq, "path", path, "did", did, "rcid", rcid, "rec", rec)
	switch rec.(type) {
	case *bsky.FeedPost:
		// TODO.
	case *bsky.ActorProfile, *bsky.FeedGenerator, *bsky.FeedLike, *bsky.FeedPostgate, *bsky.FeedRepost, *bsky.FeedThreadgate, *bsky.GraphBlock, *bsky.GraphFollow, *bsky.GraphList, *bsky.GraphListblock, *bsky.GraphListitem, *bsky.GraphStarterpack:
	default:
		slog.Error("bsky", "type", reflect.TypeOf(rec).String(), "record", rec)
	}
	return nil
}

//

// FacetType is one of the supported bluesky richtext style.
type FacetType string

const (
	FacetLink    FacetType = "app.bsky.richtext.facet#link"
	FacetMention FacetType = "app.bsky.richtext.facet#mention"
	FacetTag     FacetType = "app.bsky.richtext.facet#tag"
)

// Facet describes a part of a rich post.
//
// See https://docs.bsky.app/docs/advanced-guides/post-richtext#rich-text-facets
type Facet struct {
	Type        FacetType
	ToHighlight string
	Value       string
}

// Link describes a post link, its title, description and thumbnail.
type Link struct {
	Title       string
	URI         url.URL
	Description string
	Thumbnail   *util.LexBlob
}

// ImageRef refers to an uploaded image.
type ImageRef struct {
	Title    string
	URI      url.URL
	Uploaded *util.LexBlob
}

// Post is a skeet to send.
type Post struct {
	Text string

	// Everything else is optional.

	CreatedAt  time.Time
	ReplyToCID string
	ReplyToURI string
	Facets     []Facet
	Link       *Link
	Images     []ImageRef
}

func (p *Post) build() (*bsky.FeedPost, error) {
	if p.Text == "" {
		return nil, errors.New("text is required")
	}
	post := &bsky.FeedPost{Text: p.Text, LexiconTypeID: "app.bsky.feed.post"}
	t := p.CreatedAt
	if t.IsZero() {
		t = time.Now()
	}
	post.CreatedAt = t.Format(time.RFC3339)
	// TODO: Langs = []string{"en-CA"}

	if (p.ReplyToCID == "") != (p.ReplyToURI == "") {
		return nil, errors.New("either both ReplyToCID and ReplyToURI or none must be specified")
	}
	if p.ReplyToCID != "" {
		rep := &atproto.RepoStrongRef{Cid: p.ReplyToCID, Uri: p.ReplyToURI}
		post.Reply = &bsky.FeedPost_ReplyRef{Root: rep, Parent: rep}
	}

	for _, f := range p.Facets {
		var feature *bsky.RichtextFacet_Features_Elem
		switch f.Type {
		case FacetLink:
			feature = &bsky.RichtextFacet_Features_Elem{
				RichtextFacet_Link: &bsky.RichtextFacet_Link{LexiconTypeID: string(f.Type), Uri: f.Value},
			}
		case FacetMention:
			feature = &bsky.RichtextFacet_Features_Elem{
				RichtextFacet_Mention: &bsky.RichtextFacet_Mention{LexiconTypeID: string(f.Type), Did: f.Value},
			}
		case FacetTag:
			feature = &bsky.RichtextFacet_Features_Elem{
				RichtextFacet_Tag: &bsky.RichtextFacet_Tag{LexiconTypeID: string(f.Type), Tag: f.Value},
			}
		default:
			return nil, fmt.Errorf("invalid facet %s", f.Type)
		}
		start := strings.Index(post.Text, f.ToHighlight)
		if start == -1 {
			return nil, fmt.Errorf("unable to find the substring %q", f.ToHighlight)
		}
		post.Facets = append(post.Facets, &bsky.RichtextFacet{
			Index:    &bsky.RichtextFacet_ByteSlice{ByteStart: int64(start), ByteEnd: int64(start + len(f.ToHighlight))},
			Features: []*bsky.RichtextFacet_Features_Elem{feature},
		})
	}
	if p.Link != nil {
		if len(p.Images) != 0 {
			return nil, errors.New("only one of link or images can be set")
		}
		post.Embed = &bsky.FeedPost_Embed{
			EmbedExternal: &bsky.EmbedExternal{
				LexiconTypeID: "app.bsky.embed.external",
				External: &bsky.EmbedExternal_External{
					Title:       p.Link.Title,
					Uri:         p.Link.URI.String(),
					Description: p.Link.Description,
					Thumb:       p.Link.Thumbnail,
				},
			},
		}
	} else if len(p.Images) != 0 {
		post.Embed = &bsky.FeedPost_Embed{
			EmbedImages: &bsky.EmbedImages{
				LexiconTypeID: "app.bsky.embed.images",
				Images:        make([]*bsky.EmbedImages_Image, len(p.Images)),
			},
		}
		for i, img := range p.Images {
			post.Embed.EmbedImages.Images[i] = &bsky.EmbedImages_Image{Alt: img.Title, Image: img.Uploaded}
		}
	}
	return post, nil
}
