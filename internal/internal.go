// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package internal contains various random shared code.
package internal

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"regexp"
	"strconv"
)

// General functions I didn't know where to put.

// FindFreePort returns an available TCP port to listen to, first trying
// preferred.
func FindFreePort(preferred ...int) int {
	for _, p := range preferred {
		l, err := net.Listen("tcp", "localhost:"+strconv.Itoa(p))
		if err != nil {
			continue
		}
		defer l.Close()
		return l.Addr().(*net.TCPAddr).Port
	}
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
	var errs []error
	if err != nil {
		errs = append(errs, fmt.Errorf("failed to decode server response: %w", err))
	}
	if resp.StatusCode >= 400 {
		errs = append(errs, &HTTPError{URL: url, StatusCode: resp.StatusCode, Status: resp.Status})
	}
	return errors.Join(errs...)
}

// JSONPostRequest simplifies doing an HTTP POST in JSON. It initiates
// the requests and returns the response back.
func JSONPostRequest(ctx context.Context, url string, in interface{}) (*http.Response, error) {
	b := bytes.Buffer{}
	e := json.NewEncoder(&b)
	// OMG this took me a while to figure this out. This affects token encoding.
	e.SetEscapeHTML(false)
	if err := e.Encode(in); err != nil {
		return nil, fmt.Errorf("internal error: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, "POST", url, &b)
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
	var errs []error
	if err != nil {
		errs = append(errs, fmt.Errorf("failed to decode server response: %w", err))
	}
	if resp.StatusCode >= 400 {
		errs = append(errs, &HTTPError{URL: url, StatusCode: resp.StatusCode, Status: resp.Status})
	}
	return errors.Join(errs...)
}

// HTTPError represents an HTTP request that returned an HTTP error.
type HTTPError struct {
	URL        string
	StatusCode int
	Status     string
}

func (h *HTTPError) Error() string {
	return h.Status
}
