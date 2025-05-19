// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package tools contains structures to generate function calls, tool calling
// from LLMs.
package tools

import (
	"encoding/json"
	"fmt"
	"math"
	"time"

	"github.com/maruel/genai"
)

// Calculate is a tool usable by LLMs.
//
// It executes the arithmetic operation over two float64 numbers in string
// form. The supported operations are "addition", "subtraction",
// "multiplication" and "division".
var CalculateTool = genai.ToolDef{
	Name:        "calculate",
	Description: "Calculate an mathematical arithmetic operation.",
	Callback: func(args *calculateArgs) (string, error) {
		n1, err := args.FirstNumber.Float64()
		if err != nil {
			return "", fmt.Errorf("couldn't understand the first number: %w", err)
		}
		n2, err := args.SecondNumber.Float64()
		if err != nil {
			return "", fmt.Errorf("couldn't understand the second number: %w", err)
		}
		r := 0.
		switch args.Operation {
		case "addition":
			r = n1 + n2
		case "subtraction":
			r = n1 - n2
		case "multiplication":
			r = n1 * n2
		case "division":
			r = n1 / n2
		default:
			return "", fmt.Errorf("unknown operation %q", args.Operation)
		}
		// Do not use %g all the time because it tends to use exponents too quickly
		// and the LLM is super confused about that.
		// Do not use naive %f all the time because the LLM gets confused with
		// decimals.
		if r == math.Trunc(r) {
			return fmt.Sprintf("%.0f", r), nil
		}
		return fmt.Sprintf("%f", r), nil
	},
}

type calculateArgs struct {
	Operation    string `jsonschema:"enum=addition,enum=subtraction,enum=multiplication,enum=division"`
	FirstNumber  json.Number
	SecondNumber json.Number
}

// GetTodayClockTime returns the current time and day in a format that the LLM
// can understand.
var GetTodayClockTime = genai.ToolDef{
	Name:        "get_today_date_current_clock_time",
	Description: "Get the current clock time and today's date.",
	Callback: func(e *empty) (string, error) {
		return time.Now().Format("Monday 2006-01-02 15:04:05"), nil
	},
}

type empty struct{}
