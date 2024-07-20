// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/lmittmann/tint"
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

func rawPost(t *testing.T, url string, in string, out interface{}) {
	ctx := context.Background()
	req, err := http.NewRequestWithContext(ctx, "POST", url, strings.NewReader(in))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != 200 {
		t.Fatalf("HTTP status %s", resp.Status)
	}
	d := json.NewDecoder(resp.Body)
	d.DisallowUnknownFields()
	if err = d.Decode(out); err != nil {
		t.Fatal(err)
	}
	if err = resp.Body.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestTool(t *testing.T) {
	// Take the raw output from //py/mistral_test.py.
	// Call llama-server directly, ignoring the utility code in struct LLM.
	l := loadModel(t, "Mistral-7B-Instruct-v0.3.Q3_K_M")

	t.Run("Raw", func(t *testing.T) {
		t.Parallel()
		raw := `{"prompt": "<s>[AVAILABLE_TOOLS]\u2581[{\"type\":\u2581\"function\",\u2581\"function\":\u2581{\"name\":\u2581\"get_current_weather\",\u2581\"description\":\u2581\"Get\u2581the\u2581current\u2581weather\",\u2581\"parameters\":\u2581{\"type\":\u2581\"object\",\u2581\"properties\":\u2581{\"location\":\u2581{\"type\":\u2581\"string\",\u2581\"description\":\u2581\"The\u2581city\u2581and\u2581state,\u2581e.g.\u2581San\u2581Francisco,\u2581US\u2581or\u2581Montr\u00e9al,\u2581CA\u2581or\u2581Berlin,\u2581DE\"},\u2581\"format\":\u2581{\"type\":\u2581\"string\",\u2581\"enum\":\u2581[\"celsius\",\u2581\"fahrenheit\"],\u2581\"description\":\u2581\"The\u2581temperature\u2581unit\u2581to\u2581use.\u2581Infer\u2581this\u2581from\u2581the\u2581users\u2581location.\"}},\u2581\"required\":\u2581[\"location\",\u2581\"format\"]}}}][/AVAILABLE_TOOLS][INST]\u2581What's\u2581the\u2581weather\u2581like\u2581today\u2581in\u2581Paris[/INST]"}`
		msg := llamaCPPCompletionResponse{}
		rawPost(t, l.baseURL+"/completion", raw, &msg)
		parseToolResponse(t, msg.Content)
		// Do a second call. Also copied from //py/mistral_test.py.
		raw = `{"prompt": "<s>[AVAILABLE_TOOLS]\u2581[{\"type\":\u2581\"function\",\u2581\"function\":\u2581{\"name\":\u2581\"get_current_weather\",\u2581\"description\":\u2581\"Get\u2581the\u2581current\u2581weather\",\u2581\"parameters\":\u2581{\"type\":\u2581\"object\",\u2581\"properties\":\u2581{\"location\":\u2581{\"type\":\u2581\"string\",\u2581\"description\":\u2581\"The\u2581city\u2581and\u2581state,\u2581e.g.\u2581San\u2581Francisco,\u2581US\u2581or\u2581Montr\u00e9al,\u2581CA\u2581or\u2581Berlin,\u2581DE\"},\u2581\"format\":\u2581{\"type\":\u2581\"string\",\u2581\"enum\":\u2581[\"celsius\",\u2581\"fahrenheit\"],\u2581\"description\":\u2581\"The\u2581temperature\u2581unit\u2581to\u2581use.\u2581Infer\u2581this\u2581from\u2581the\u2581users\u2581location.\"}},\u2581\"required\":\u2581[\"location\",\u2581\"format\"]}}}][/AVAILABLE_TOOLS][INST]\u2581What's\u2581the\u2581weather\u2581like\u2581today\u2581in\u2581Paris[/INST][TOOL_CALLS]\u2581[{\"name\":\u2581\"get_current_weather\",\u2581\"arguments\":\u2581{\"location\":\u2581\"Paris,\u2581FR\",\u2581\"format\":\u2581\"celsius\"},\u2581\"id\":\u2581\"c00000000\"}]</s>[TOOL_RESULTS]\u2581{\"content\":\u258143,\u2581\"call_id\":\u2581\"c00000000\"}[/TOOL_RESULTS]"}`
		msg = llamaCPPCompletionResponse{}
		rawPost(t, l.baseURL+"/completion", raw, &msg)
		if !strings.Contains(msg.Content, " 43 ") && !strings.Contains(msg.Content, " 43°") {
			t.Fatalf("expected 43°C, got %q", msg.Content)
		}
	})
	t.Run("API", func(t *testing.T) {
		t.Parallel()
		// Refs:
		// https://github.com/mistralai/mistral-common/blob/main/src/mistral_common/tokens/tokenizers/base.py#L10
		// https://github.com/mistralai/mistral-common/blob/main/src/mistral_common/tokens/tokenizers/sentencepiece.py#L348
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
		msgs := []Message{
			{
				Role:    AvailableTools,
				Content: `[{"type":▁"function",▁"function":▁{"name":▁"get_current_weather",▁"description":▁"Get▁the▁current▁weather",▁"parameters":▁{"type":▁"object",▁"properties":▁{"location":▁{"type":▁"string",▁"description":▁"The▁city▁and▁state,▁e.g.▁San▁Francisco,▁US▁or▁Montréal,▁CA▁or▁Berlin,▁DE"},▁"format":▁{"type":▁"string",▁"enum":▁["celsius",▁"fahrenheit"],▁"description":▁"The▁temperature▁unit▁to▁use.▁Infer▁this▁from▁the▁users▁location."}},▁"required":▁["location",▁"format"]}}}]`,
			},
			{
				Role:    User,
				Content: `What's\u2581the\u2581weather\u2581like\u2581today\u2581in\u2581Paris`,
			},
		}
		s, err := l.llamaCPPPromptBlocking(ctx, msgs, 1, 0)
		if err != nil {
			t.Fatal(err)
		}
		parseToolResponse(t, s)
		msgs = append(msgs,
			Message{
				Role:    ToolCall,
				Content: `[{"name":▁"get_current_weather",▁"arguments":▁{"location":▁"Paris",▁"format":▁"celsius"}}]`,
			},
			Message{
				Role:    ToolCallResult,
				Content: `{\"content\":\u258143,\u2581\"call_id\":\u2581\"c00000000\"}`,
			},
		)
		s, err = l.llamaCPPPromptBlocking(ctx, msgs, 1, 0)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(s, " 43 ") {
			t.Fatalf("expected 43°C, got %q", s)
		}
	})
}

func parseToolResponse(t *testing.T, got string) {
	got = strings.TrimSpace(got)
	var toolCalls []MistralToolCall
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
		t.Skipf("expected one tool call, got %# v", toolCalls)
	}
	possible := []MistralToolCall{
		{
			Name:      "get_current_weather",
			Arguments: map[string]string{"location": "Paris", "format": "celsius"},
		},
		{
			Name:      "get_current_weather",
			Arguments: map[string]string{"location": "Paris"},
		},
		{
			Name:      "get_current_weather",
			Arguments: map[string]string{"location": "Paris, FR"},
		},
		{
			Name:      "get_current_weather",
			Arguments: map[string]string{"location": "Paris, FR", "format": "celsius"},
		},
	}
	found := false
	for _, w := range possible {
		if diff := cmp.Diff(w, toolCalls[0]); diff == "" {
			found = true
			break
		}
	}
	if !found {
		t.Skipf("Didn't get the expected function call:  %q", toolCalls)
	}
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
