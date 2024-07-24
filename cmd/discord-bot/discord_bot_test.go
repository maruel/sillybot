// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package main

import (
	"strconv"
	"testing"
)

func TestSplitResponse(t *testing.T) {
	data := []struct {
		input string
		want  int
	}{
		{"", 0},
		{"Hi", 0},
		{"Hi!", 0},
		{"Hi fellow kids!", 15},
		{"This is code:\n```", 14},
		{"This is code:\nFoo", 14},
		{"This is code:\n```Foo", 14},
		{"This is code:\n``Foo```", 14},
		{"This is code:\n```Foo```", 23},
		{"This is code:\n```Foo```\nAnd happiness", 23},
		{"This is enumeration:\n1. ", 21},
		{"1. Do stuff.", 0},
		{"1. Do.", 0},
		{"- Do stuff.", 0},
		{"To do what you want, use `os.ReadFile", 0},
		{"To do what you want, use node.js and it's going to be fine", 0},
	}
	for i, line := range data {
		t.Run(strconv.Itoa(i), func(t *testing.T) {
			if got := splitResponse(line.input); line.want != got {
				t.Fatalf("%q: %d != %d\nWant: %q\nGot:  %q", line.input, line.want, got, line.input[:line.want], line.input[:got])
			}
		})
	}
}
