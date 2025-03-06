// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package llm

// llamaCPPHealthResponse is documented at
// https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md#api-endpoints
type llamaCPPHealthResponse struct {
	Status          string
	SlotsIdle       int `json:"slots_idle"`
	SlotsProcessing int `json:"slots_processing"`
}

// llamaCPPCompletionRequest is documented at
// https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md#api-endpoints
type llamaCPPCompletionRequest struct {
	SystemPrompt     string      `json:"system_prompt,omitempty"`
	Prompt           string      `json:"prompt"`
	Grammar          string      `json:"grammar,omitempty"`
	JSONSchema       interface{} `json:"json_schema,omitempty"`
	Seed             int64       `json:"seed,omitempty"`
	Temperature      float64     `json:"temperature,omitempty"`
	DynaTempRange    float64     `json:"dynatemp_range,omitempty"`
	DynaTempExponent float64     `json:"dynatemp_exponent,omitempty"`
	CachePrompt      bool        `json:"cache_prompt,omitempty"`
	Stream           bool        `json:"stream"`
	// top_k             float64
	// top_p             float64
	// min_p             float64
	NPredict int64 `json:"n_predict,omitempty"` // Maximum number of tokens to predict
	// n_keep            int64
	// stop              []string
	// tfs_z             float64
	// typical_p         float64
	// repeat_penalty    float64
	// repeat_last_n     int64
	// penalize_nl       bool
	// presence_penalty  float64
	// frequency_penalty float64
	// penalty_prompt    *string
	// mirostat          int32
	// mirostat_tau      float64
	// mirostat_eta      float64
	// ignore_eos   bool
	// logit_bias   []interface{}
	// n_probs      int64
	// min_keep     int64
	// image_data   []byte
	// id_slot      int64
	// samplers     []string
}

// llamaCPPCompletionResponse is documented at
// https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md#result-json
type llamaCPPCompletionResponse struct {
	Content            string      `json:"content"`
	Stop               bool        `json:"stop"`
	GenerationSettings interface{} `json:"generation_settings"`
	Model              string      `json:"model"`
	Prompt             string      `json:"prompt"`
	StoppedEOS         bool        `json:"stopped_eos"`
	StoppedLimit       bool        `json:"stopped_limit"`
	StoppedWord        bool        `json:"stopped_word"`
	StoppingWord       string      `json:"stopping_word"`
	Timings            struct {
		// Undocumented:
		PromptN             int64   `json:"prompt_n"`
		PromptMS            float64 `json:"prompt_ms"`
		PromptPerTokenMS    float64 `json:"prompt_per_token_ms"`
		PromptPerSecond     float64 `json:"prompt_per_second"`
		PredictedN          int64   `json:"predicted_n"`
		PredictedMS         float64 `json:"predicted_ms"`
		PredictedPerTokenMS float64 `json:"predicted_per_token_ms"`
		PredictedPerSecond  float64 `json:"predicted_per_second"`
	}
	TokensCached            int64 `json:"tokens_cached"`
	TokensEvaluated         int64 `json:"tokens_evaluated"`
	Truncated               bool  `json:"truncated"`
	CompletionProbabilities []struct {
		Content string
		Probs   []struct {
			Prob   float64
			TokStr string `json:"tok_str"`
		}
	} `json:"completion_probabilities"`
	// Undocumented:
	HasNewLine      bool  `json:"has_new_line"`
	IDSlot          int64 `json:"id_slot"`
	Index           int64 `json:"index"`
	TokensPredicted int64 `json:"tokens_predicted"`
	Multimodal      bool  `json:"multimodal"`
	// Error case:
	Error errorResponse `json:"error"`
}
