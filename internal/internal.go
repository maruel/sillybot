// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package internal contains various random shared code.
package internal

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"regexp"
)

// General functions I didn't know where to put.

// FindFreePort returns an available TCP port to listen to.
func FindFreePort() int {
	l, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		panic(err)
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port
}

// IsHostPort returns true if the string seems like a valid "host:port" string.
func IsHostPort(s string) bool {
	// Simplified regexp that supports IPv4, IPv6 and hostname and requires a port.
	ipv4 := `\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}`
	ipv6 := `\[[a-fA-F0-9:]+\]`
	hostname := `[a-zA-Z0-9\-\.]{2,}`
	r := `^(?:` + ipv4 + `|` + ipv6 + `|` + hostname + `):\d{1,5}$`
	ok, err := regexp.MatchString(r, s)
	if err != nil {
		panic(err)
	}
	return ok
}

// JSONPos simplifies doing an HTTP POST in JSON.
func JSONPost(ctx context.Context, url string, in, out interface{}) error {
	resp, err := JSONPostRequest(ctx, url, in)
	if err != nil {
		return err
	}
	d := json.NewDecoder(resp.Body)
	d.DisallowUnknownFields()
	err = d.Decode(out)
	_ = resp.Body.Close()
	if err != nil {
		return fmt.Errorf("failed to decode server response: %w", err)
	}
	return nil
}

// JSONPostRequest simplifies doing an HTTP POST in JSON. It initiates
// the requests and returns the response back.
func JSONPostRequest(ctx context.Context, url string, in interface{}) (*http.Response, error) {
	b, err := json.Marshal(in)
	if err != nil {
		return nil, fmt.Errorf("internal error: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	return http.DefaultClient.Do(req)
}

// JSONGet does a HTTP GET and parses the returned JSON.
func JSONGet(ctx context.Context, url string, out interface{}) error {
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	d := json.NewDecoder(resp.Body)
	d.DisallowUnknownFields()
	err = d.Decode(out)
	_ = resp.Body.Close()
	if err != nil {
		return fmt.Errorf("failed to decode server response: %w", err)
	}
	return nil
}
