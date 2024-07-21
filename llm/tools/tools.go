// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package tools contains structures to generate function calls, tool calling
// from LLMs.
package tools

import (
	"fmt"
	"math"
	"strconv"
)

// MistralTool is the description of a tool that the mistral models can use.
//
// See
// https://github.com/mistralai/mistral-common/blob/main/src/mistral_common/protocol/instruct/tool_calls.py
// and InstructTokenizerV3 in
// https://github.com/mistralai/mistral-common/blob/main/src/mistral_common/tokens/tokenizers/sentencepiece.py
//
// Sadly the python class and the JSON encoding do not match (!).
//
// The only "real" doc is the example at
// https://github.com/mistralai/mistral-common/blob/main/examples/client.ipynb
type MistralTool struct {
	// Type must be "function". As of 2024-09-20, Mistral doesn't support
	// anything else.
	Type     string
	Function MistralFunction `json:"function,omitempty"`

	_ struct{}
}

// MistralFunction is an available function to call.
type MistralFunction struct {
	Name        string
	Description string
	Parameters  MistralFunctionParams

	_ struct{}
}

// MistralFunctionParams is the list of arguments for an available
// tool.
type MistralFunctionParams struct {
	// Type must be "object".
	Type       string
	Properties map[string]MistralProperty
	// Required is more a "hint".
	Required []string

	_ struct{}
}

// MistralProperty is a single available tool parameter argument.
type MistralProperty struct {
	// Type should be "string" until we find other examples.
	Type string
	// Description should contain a few examples.
	Description string
	// Enum is the list of acceptable values.
	Enum []string `json:"enum,omitempty"`

	_ struct{}
}

// MistralToolCall is returned by the Mistral models when they want to make a
// tool call.
type MistralToolCall struct {
	Name      string
	Arguments map[string]string
	// ID must be exactly 9 characters long.
	//
	// See MistralRequestValidatorV3._validate_tool_message() in
	// https://github.com/mistralai/mistral-common/blob/main/src/mistral_common/protocol/instruct/validator.py
	ID string `json:"id"`

	_ struct{}
}

// MistralToolCallResult is generated when we return the result of a tool call.
type MistralToolCallResult struct {
	// Mistral accepts any JSON result.
	Content interface{}
	// CallID must be exactly 9 characters long.
	//
	// See MistralRequestValidatorV3._validate_tool_call() in
	// https://github.com/mistralai/mistral-common/blob/main/src/mistral_common/protocol/instruct/validator.py
	CallID string `json:"call_id"`

	_ struct{}
}

// Calculate is a tool usable by LLMs.
//
// It executes the arithmetic operation over two float64 numbers in string
// form. The supported operations are "addition", "subtraction",
// "multiplication" and "division".
func Calculate(op, first_number, second_number string) string {
	n1, err := strconv.ParseFloat(first_number, 64)
	if err != nil {
		return fmt.Sprintf("couldn't understand the first number %q", first_number)
	}
	n2, err := strconv.ParseFloat(second_number, 64)
	if err != nil {
		return fmt.Sprintf("couldn't understand the second number %q", second_number)
	}
	r := 0.
	switch op {
	case "addition":
		r = n1 + n2
	case "subtraction":
		r = n1 - n2
	case "multiplication":
		r = n1 * n2
	case "division":
		r = n1 / n2
	default:
		return "unknown operation " + op
	}
	// Do not use %g all the time because it tends to use exponents too quickly
	// and the LLM is super confused about that.
	// Do not use naive %f all the time because the LLM gets confused with
	// decimals.
	if r == math.Trunc(r) {
		return fmt.Sprintf("%.0f", r)
	}
	return fmt.Sprintf("%f", r)
}

// CalculateMistralTool is the structure to use to pass to use Calculate in
// Mistral models.
var CalculateMistralTool = MistralTool{
	Type: "function",
	Function: MistralFunction{
		Name:        "calculate",
		Description: "Calculate an arithmetic operation. Use this tool if you need a calculator to do mathematics calculation.",
		Parameters: MistralFunctionParams{
			Type: "object",
			Properties: map[string]MistralProperty{
				"first_number": MistralProperty{
					Type:        "string",
					Description: "First number in the operation.",
				},
				"second_number": MistralProperty{
					Type:        "string",
					Description: "Second number in the operation.",
				},
				"operation": MistralProperty{
					Type:        "string",
					Description: "Arithmetic operation to do.",
					Enum:        []string{"addition", "subtraction", "multiplication", "division"},
				},
			},
			Required: []string{"first_number", "second_number", "operation"},
		},
	},
}
