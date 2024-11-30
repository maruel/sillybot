// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package main

import (
	"context"
	"errors"
	"fmt"
	"net/url"
	"strings"
	"time"

	"github.com/bluesky-social/indigo/api/atproto"
	"github.com/bluesky-social/indigo/api/bsky"
	"github.com/bluesky-social/indigo/lex/util"
	"github.com/bluesky-social/indigo/xrpc"
)

// Client is the AT protocol client.
type Client struct {
	client *xrpc.Client
}

// New returns a new blue sky client.
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

	CreatedAt time.Time
	Facets    []Facet
	Link      *Link
	Images    []ImageRef
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
