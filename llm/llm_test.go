// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/lmittmann/tint"
	"github.com/maruel/sillybot/llm/tools"
	"github.com/mattn/go-colorable"
	"github.com/mattn/go-isatty"
	"gopkg.in/yaml.v3"
)

func TestLLM(t *testing.T) {
	const systemPrompt = "You are an AI assistant. You strictly follow orders. Do not add extraneous words. Only reply with what is asked of you."
	re := regexp.MustCompile(`-(\d+)[bBk]-`)
	for _, k := range loadKnownLLMs(t) {
		t.Run(k.Basename, func(t *testing.T) {
			b := re.FindStringSubmatch(k.Basename)
			if len(b) != 2 {
				t.Skip("skipping complex")
			}
			size, err := strconv.Atoi(b[1])
			if err != nil {
				t.Fatal(err)
			}
			if size > 9 {
				t.Skip("too large")
			}
			if !strings.HasPrefix(k.Basename, "Mistral") && testing.Short() {
				t.Skip("skipping this model when -short is used")
			}
			if strings.HasPrefix(k.Basename, "phi-3") {
				// I suspect it's because it's already a small model so it fails at
				// high quantization.
				t.Skip("phi-3-mini is misbehaving. TODO: investigate")
			}
			// I tested with Q2_K and results are unreliable.
			quant := "Q3_K_M"
			// Hack: naming conventions are not figured out.
			if strings.Contains(strings.ToLower(k.Basename), "qwen") {
				quant = strings.ToLower(quant)
			}
			testModel(t, k.Basename+quant, systemPrompt)
		})
	}
	t.Run("python", func(t *testing.T) {
		if testing.Short() {
			t.Skip("skipping this model when -short is used")
		}
		l := loadModel(t, "python")
		testModelInner(t, l, systemPrompt)
	})
}

func testModel(t *testing.T, model, systemPrompt string) {
	l := loadModel(t, model)
	if l.Encoding != nil {
		t.Run("CustomEncoding", func(t *testing.T) {
			testModelInner(t, l, systemPrompt)
		})
	}
	l.Encoding = nil
	t.Run("OpenAI", func(t *testing.T) {
		testModelInner(t, l, systemPrompt)
	})
}

func testModelInner(t *testing.T, l *Session, systemPrompt string) {
	ctx := context.Background()
	const prompt = "reply with \"ok chief\""
	t.Run("Blocking", func(t *testing.T) {
		t.Parallel()
		msgs := []Message{{Role: System, Content: systemPrompt}, {Role: User, Content: prompt}}
		got, err2 := l.Prompt(ctx, msgs, 1, 0.0)
		if err2 != nil {
			t.Fatal(err2)
		}
		checkAnswer(t, got)
	})
	t.Run("Streaming", func(t *testing.T) {
		t.Parallel()
		msgs := []Message{{Role: System, Content: systemPrompt}, {Role: User, Content: prompt}}
		words := make(chan string, 10)
		got := ""
		wg := sync.WaitGroup{}
		wg.Add(1)
		go func() {
			for c := range words {
				got += c
			}
			wg.Done()
		}()
		err2 := l.PromptStreaming(ctx, msgs, 1, 0.0, words)
		close(words)
		wg.Wait()
		if err2 != nil {
			t.Fatal(err2)
		}
		checkAnswer(t, got)
	})
}

func TestMistralTool(t *testing.T) {
	l := loadModel(t, "Mistral-7B-Instruct-v0.3-Q2_K")
	// Refs:
	// - SpecialTokens in
	//   https://github.com/mistralai/mistral-common/blob/main/src/mistral_common/tokens/tokenizers/base.py
	// - InstructTokenizerV3 in
	//   https://github.com/mistralai/mistral-common/blob/main/src/mistral_common/tokens/tokenizers/sentencepiece.py
	// //py/mistral_test.py
	l.Encoding = &PromptEncoding{
		BeginOfText:              "<s>",
		SystemTokenStart:         "[INST]\u2581",
		SystemTokenEnd:           " [/INST]",
		UserTokenStart:           "[INST]\u2581",
		UserTokenEnd:             " [/INST]",
		AssistantTokenStart:      "",
		AssistantTokenEnd:        "",
		ToolsAvailableTokenStart: "[AVAILABLE_TOOLS]\u2581",
		ToolsAvailableTokenEnd:   "[/AVAILABLE_TOOLS]",
		ToolCallTokenStart:       "[TOOL_CALLS]\u2581",
		ToolCallTokenEnd:         "</s>",
		ToolCallResultTokenStart: "[TOOL_RESULTS]\u2581",
		ToolCallResultTokenEnd:   "[/TOOL_RESULTS]",
	}
	ctx := context.Background()
	availtools := []tools.MistralTool{
		// The example given in mistral source code.
		{
			Type: "function",
			Function: tools.MistralFunction{
				Name:        "get_current_weather",
				Description: "Get the current weather",
				Parameters: &tools.MistralFunctionParams{
					Type: "object",
					Properties: map[string]tools.MistralProperty{
						"location": tools.MistralProperty{
							Type:        "string",
							Description: "The city and country, e.g. San Francisco, US or Montréal, CA or Berlin, DE",
						},
						"format": tools.MistralProperty{
							Type:        "string",
							Enum:        []string{"celsius", "fahrenheiht"},
							Description: "The temperature unit to use. Infer this from the user's location.",
						},
					},
					Required: []string{"location", "format"},
				},
			},
		},
		tools.CalculateMistralTool,
	}
	toolsB, err := json.Marshal(availtools)
	if err != nil {
		t.Fatal(err)
	}
	msgs := []Message{
		{Role: AvailableTools, Content: string(toolsB)},
		{
			Role:    User,
			Content: `What's\u2581the\u2581weather\u2581like\u2581today\u2581in\u2581Paris`,
		},
	}
	for _, m := range msgs {
		t.Log(m)
	}
	msgsl := len(msgs)
	s, err := l.llamaCPPPromptBlocking(ctx, msgs, 1, 0)
	if err != nil {
		t.Fatal(err)
	}
	msgs = append(msgs, parseToolResponse(t, s, 0)...)
	for _, m := range msgs[msgsl:] {
		t.Log(m)
	}
	msgsl = len(msgs)
	s, err = l.llamaCPPPromptBlocking(ctx, msgs, 1, 0)
	if err != nil {
		t.Fatal(err)
	}
	msgs = append(msgs, Message{Role: Assistant, Content: s})
	for _, m := range msgs[msgsl:] {
		t.Log(m)
	}
	msgsl = len(msgs)
	if !strings.Contains(s, " 43 ") && !strings.Contains(s, " 43°") {
		t.Fatalf("expected 43°C, got %q", s)
	}

	// Now do a calculation!
	msgs = append(msgs, Message{Role: User, Content: "Give me the result of 43215 divided by 215."})
	for _, m := range msgs[msgsl:] {
		t.Log(m)
	}
	msgsl = len(msgs)
	if s, err = l.llamaCPPPromptBlocking(ctx, msgs, 1, 0); err != nil {
		t.Fatal(err)
	}
	msgs = append(msgs, parseToolResponse(t, s, 1)...)
	for _, m := range msgs[msgsl:] {
		t.Log(m)
	}
	msgsl = len(msgs)
	if s, err = l.llamaCPPPromptBlocking(ctx, msgs, 1, 0); err != nil {
		t.Fatal(err)
	}
	msgs = append(msgs, Message{Role: Assistant, Content: s})
	for _, m := range msgs[msgsl:] {
		t.Log(m)
	}
	if !strings.Contains(s, "201") {
		t.Fatalf("expected 201, got %q", s)
	}
}

func parseToolResponse(t *testing.T, got string, id int) []Message {
	got = strings.TrimSpace(got)
	var toolCalls []tools.MistralToolCall
	//  Search for a tool call.
	for _, line := range strings.Split(got, "\n") {
		if err := json.Unmarshal([]byte(line), &toolCalls); err == nil {
			got = ""
			break
		}
	}
	// Don't fail here since it's highly non-deterministic. Examples of past output:
	// "To get the current weather in Paris, you can use the following command:\n\n[{\"name\": \"get_current_weather\", \"arguments\": {\"location\": \"Paris, FR\"}}]\n\nYou can also specify the temperature unit by adding the format parameter:\n\n[{\"name\": \"get_current_weather\", \"arguments\": {\"location\": \"Paris, FR\", \"format\": \"celsius\"}}]\n\nIf you prefer Fahrenheit, use:\n\n[{\"name\": \"get_current_weather\", \"arguments\": {\"location\": \"Paris, FR\", \"format\": \"fahrenheit\"}}]"
	// "[{\"name\": \"get_current_weather\", \"arguments\": {\"location\": \"Paris\", \"format\": \"celsius\"}}]"
	// "To find out the current weather in Paris, you can call the `get_current_weather` function with the location as `Paris, FR`. Here's an example:\n\n```\nget_current_weather(location='Paris, FR')\n```\n\nIf you want to specify the temperature unit, you can add the `format` parameter too. For example, to get the temperature in Celsius:\n\n```\nget_current_weather(location='Paris, FR', format='celsius')\n```"
	// "To get the current weather in Paris, you can use the `get_current_weather` function with the location set to \"Paris, FR\" and the temperature unit in Celsius (since it's the default for France). Here's how you can call the function:\n\n```python\nget_current_weather(location=\"Paris, FR\", format=\"celsius\")\n```\n\nThis will return the current weather data for Paris, France in Celsius. If you'd like to get the weather in Fahrenheit, you can set the format to \"fahrenheit\" instead.\n\nHowever, please note that I'm not a real AI and don't have the ability to fetch real-time weather data. You'll need to integrate this function with a weather API to actually get the weather data for a specific location.\n\nHere's an example of how the function might look if it were integrated with the OpenWeatherMap API:\n\n```python\nimport requests\nimport json\n\ndef get_current_weather(location, format):\n    api_key = \"YOUR_API_KEY\"\n    url = f\"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units={format}\"\n    response = requests.get(url)\n    data = json.loads(response.text)\n\n    temperature = data[\"main\"][\"temp\"]\n    description = data[\"weather\"][0][\"description\"]\n\n    return {\"temperature\": temperature, \"description\": description}\n\nweather_data = get_current_weather(location=\"Paris, FR\", format=\"celsius\")\nprint(weather_data)\n```"
	if got != "" {
		t.Logf("expected %q, got %q", "", got)
	}
	if len(toolCalls) != 1 {
		t.Fatalf("expected one tool call, got %# v", toolCalls)
	}
	res := ""
	switch toolCalls[0].Name {
	case "get_current_weather":
		res = get_current_weather(t, toolCalls[0].Arguments["location"], toolCalls[0].Arguments["format"])
	case "calculate":
		op := toolCalls[0].Arguments["operation"]
		f := toolCalls[0].Arguments["first_number"]
		s := toolCalls[0].Arguments["second_number"]
		res = tools.Calculate(op, f, s)
	default:
		t.Fatalf("unexpected tool call %q", toolCalls[0].Name)
	}
	callid := fmt.Sprintf("c%08d", id)
	r, err := json.Marshal(tools.MistralToolCallResult{Content: res, CallID: callid})
	if err != nil {
		t.Fatal(err)
	}
	toolCalls[0].ID = callid
	c, err := json.Marshal(toolCalls)
	if err != nil {
		t.Fatal(err)
	}
	return []Message{
		Message{Role: ToolCall, Content: string(c)},
		Message{Role: ToolCallResult, Content: string(r)},
	}
}

func get_current_weather(t *testing.T, location, format string) string {
	if location != "Paris" && location != "Paris, FR" {
		t.Fatalf("expected location=\"Paris, FR\", got %q", location)
	}
	if format != "celsius" && format != "" {
		t.Fatalf("expected format=\"celsius\", got %q", format)
	}
	// Generate a fake response with a unreasonably very high temperature: 43°C.
	return "43"
}

//

func loadModel(t *testing.T, model string) *Session {
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	opts := Options{Model: model}
	l, err := New(ctx, filepath.Join(filepath.Dir(wd), "cache"), &opts, loadKnownLLMs(t))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err2 := l.Close(); err2 != nil {
			t.Error(err2)
		}
	})
	return l
}

func checkAnswer(t *testing.T, got string) {
	// Work around various non-determinism.
	if want := "ok chief"; !strings.Contains(strings.ToLower(got), want) {
		if runtime.GOOS == "darwin" && os.Getenv("CI") == "true" && os.Getenv("GITHUB_ACTION") != "" {
			t.Log("TODO: Figure out why macOS GitHub hosted runner return an empty string")
		} else {
			t.Fatalf("expected %q, got %q", want, got)
		}
	}
}

func loadKnownLLMs(t *testing.T) []KnownLLM {
	b, err := os.ReadFile("../default_config.yml")
	if err != nil {
		t.Fatal(err)
	}
	c := struct {
		KnownLLMs []KnownLLM
	}{}
	d := yaml.NewDecoder(bytes.NewReader(b))
	if err = d.Decode(&c); err != nil {
		t.Fatal(err)
	}
	if len(c.KnownLLMs) < 5 {
		t.Fatalf("Expected more known LLMs\n%# v", c.KnownLLMs)
	}
	return c.KnownLLMs
}

// TestMain sets up the verbose logging.
func TestMain(m *testing.M) {
	flag.Parse()
	l := slog.LevelWarn
	if os.Getenv("LLM_TEST_VERBOSE") == "true" {
		l = slog.LevelDebug
	}
	logger := slog.New(tint.NewHandler(colorable.NewColorable(os.Stderr), &tint.Options{
		Level:      l,
		TimeFormat: time.TimeOnly,
		NoColor:    !isatty.IsTerminal(os.Stderr.Fd()),
	}))
	slog.SetDefault(logger)
	os.Exit(m.Run())
}
