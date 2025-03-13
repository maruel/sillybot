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
	"github.com/maruel/genai/genaiapi"
	"github.com/maruel/genai/llamacpp"
	"github.com/maruel/sillybot/llm/tools"
	"github.com/mattn/go-colorable"
	"github.com/mattn/go-isatty"
	"gopkg.in/yaml.v3"
)

func estimateModelSize(t *testing.T, name string) int64 {
	name = strings.ToLower(name)
	if strings.HasPrefix(name, "phi-3") {
		// I tweeted about it, let's up next one uses a common naming convention.
		if strings.Contains(name, "-mini-") {
			// https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
			// https://huggingface.co/microsoft/Phi-3-mini-128k-instruct
			return 3800000
		}
		if strings.Contains(name, "-small-") {
			// https://huggingface.co/microsoft/Phi-3-small-8k-instruct
			// https://huggingface.co/microsoft/Phi-3-small-128k-instruct
			return 3800000
		}
		if strings.Contains(name, "-medium-") {
			// https://huggingface.co/microsoft/Phi-3-medium-4k-instruct
			// https://huggingface.co/microsoft/Phi-3-medium-128k-instruct
			return 14000000
		}
		t.Fatalf("couldn't guess phi-3 model size %q", name)
	}
	if strings.Contains(name, "-nemo-") {
		// This is disappointing.
		// https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407
		return 12000000
	}
	// Check if it is a MoE first.
	if b := regexp.MustCompile(`-(\d+)x(\d+)b-`).FindStringSubmatch(name); len(b) == 3 {
		nb, err := strconv.Atoi(b[1])
		if err != nil {
			t.Fatal(err)
		}
		s, err := strconv.Atoi(b[2])
		if err != nil {
			t.Fatal(err)
		}
		return int64(nb) * int64(s) * 1000000
	}
	// Check if it has a "." or "_", e.g. qwen 1.5B
	if b := regexp.MustCompile(`-(\d+[\._]\d+)b-`).FindStringSubmatch(name); len(b) == 2 {
		s, err := strconv.ParseFloat(strings.ReplaceAll(b[1], "_", "."), 64)
		if err != nil {
			t.Fatal(err)
		}
		return int64(s * 1000000)
	}
	if b := regexp.MustCompile(`-(\d+)b-`).FindStringSubmatch(name); len(b) == 2 {
		s, err := strconv.Atoi(b[1])
		if err != nil {
			t.Fatal(err)
		}
		return int64(s) * 1000000
	}
	t.Fatalf("couldn't guess model size for %q", name)
	return -1
}

func TestGuessSize(t *testing.T) {
	// Run to display the estimated size for each model:
	//   go test -v -run TestGuessSize
	m := 0
	for _, k := range loadKnownLLMs(t) {
		if l := len(k.Source.Basename()); l > m {
			m = l
		}
	}
	for _, k := range loadKnownLLMs(t) {
		t.Logf("%-*s: %8d", m, k.Source.Basename(), estimateModelSize(t, k.Source.Basename()))
	}
}

func TestLLM(t *testing.T) {
	// Run with -v to list the model sizes.
	const systemPrompt = "You are an AI assistant. You strictly follow orders. Reply exactly with what is asked of you."
	// Skip on models above 9B by default. We don't gain much by running these,
	// except for full coverage. Comment this out to run the full suite. Only do
	// if you have above 32GiB of RAM. I confirmed this to pass on a M3 Max with
	// 36GiB of RAM except Meta-Llama-3-70B-Instruct swaps like crazy, make sure to
	// stop all apps first. If you never heard your laptop's fan on, you sure
	// will!
	// Warning: enabling this flag will download ~80GiB of data in //cache/models/
	// TODO: We should probably warn the user based on the system's RAM.
	runLarge := false
	if runLarge {
		// This takes a while and the default timeout is 10 minutes. This will not
		// work out so bail out early.
		if _, set := t.Deadline(); set {
			t.Fatal("to run this test case with large models, use -timeout=0")
		}
	}
	var totalSize int64
	for _, k := range loadKnownLLMs(t) {
		// Q2_K is an extremely quantized form. As such the prompt must be very
		// solid. The advantages are:
		// - model files are much smaller to download, enabling use on GitHub
		//   Actions with free quota.
		// - since the models are smaller, they take less memory and they run
		//   faster, which makes unit test is much faster.
		quant := "Q2_K"
		model := strings.ToLower(k.Source.Basename())
		if strings.HasPrefix(model, "meta-llama-3.1-8b") {
			// They didn't upload Q2_L yet.
			quant = "Q3_K_L"
		} else if strings.HasPrefix(model, "llama-3.2-1b") {
			quant = "IQ3_M"
		} else if strings.HasPrefix(model, "mistral") {
			// Use a higher quantization not because it fails on Q2_K but because
			// TestMistralTool requires Q3_K_S and there's no point in downloading
			// two quantization levels.
			quant = "Q3_K_S"
		} else if strings.HasPrefix(model, "mixtral-") {
			// Fails on Q2_K.
			quant = "Q3_K_S"
		} else if strings.HasPrefix(model, "phi-3") {
			if strings.Contains(model, "-128k-") {
				// These models really struggle.
				quant = "Q3_K_M"
			} else {
				// I would have suspected it's because it's already a very small model
				// so it fails at high quantization, but the model fails in Q2_K in
				// both mini and medium!
				quant = "Q3_K_S"
			}
		} else if strings.Contains(model, "qwen") {
			if strings.Contains(model, "0.5b") || strings.Contains(model, "1.5b") {
				// Fails on Q2_K.
				quant = "Q3_K_M"
			}
			// The Alibaba team decided to be wild and lower case the quantization
			// name.
			quant = strings.ToLower(quant)
		}

		t.Run(k.Source.Basename()+quant, func(t *testing.T) {
			if size := estimateModelSize(t, k.Source.Basename()); !runLarge && size > 9000000 {
				t.Skip("skipping large model")
			}
			if testing.Short() && !strings.HasPrefix(model, "qwen2-0_5b-instruct-") {
				t.Skip("skipping this model when -short is used")
			}
			if strings.Contains(model, "-128k-") {
				t.Skip("skipping because -fa has to be enabled first")
			}
			start := time.Now()
			modelFile := testModel(t, k.Source+PackedFileRef(quant), systemPrompt)
			i, err := os.Stat(modelFile)
			if err != nil {
				t.Fatalf("%q: %s", modelFile, err)
			}
			// Note: duration is highly relative to the CPU's temperature and thermal
			// throttling. So the first runs will be super fast and then performance
			// will lower significantly. So take the numbers with a grain of salt.
			t.Logf("model %.1fGiB, took %s", float64(i.Size())*0.000000001, time.Since(start).Round(time.Second/10))
			// This only works because the tests are not run in parallel.
			totalSize += i.Size()
		})
	}
	t.Logf("processed %.1fGiB of model", float64(totalSize)*0.000000001)
	t.Run("python", func(t *testing.T) {
		t.Skip("Need new client implementation")
		if testing.Short() {
			t.Skip("skipping this model when -short is used")
		}
		l := loadModel(t, "python")
		testModelInner(t, l, systemPrompt)
	})
}

func testModel(t *testing.T, model PackedFileRef, systemPrompt string) string {
	l := loadModel(t, model)
	testModelInner(t, l, systemPrompt)
	m := llamacpp.Metrics{}
	if err := l.GetMetrics(context.Background(), &m); err != nil {
		t.Fatal(err)
	}
	t.Logf("prompt:    %4d tokens; % 8.2f tok/s", m.Prompt.Count, m.Prompt.Rate())
	t.Logf("generated: %4d tokens; % 8.2f tok/s", m.Generated.Count, m.Generated.Rate())
	return l.modelFile
}

func testModelInner(t *testing.T, l *Session, systemPrompt string) {
	ctx := context.Background()
	const prompt = "reply with \"ok chief\""
	t.Run("Blocking", func(t *testing.T) {
		t.Parallel()
		msgs := []genaiapi.Message{
			{
				Role: genaiapi.System,
				Type: genaiapi.Text,
				Text: systemPrompt,
			},
			{
				Role: genaiapi.User,
				Type: genaiapi.Text,
				Text: prompt,
			},
		}
		opts := genaiapi.CompletionOptions{MaxTokens: 10, Seed: 1}
		got, err2 := l.Prompt(ctx, msgs, &opts)
		if err2 != nil {
			t.Fatal(err2)
		}
		checkAnswer(t, got)
	})
	t.Run("Streaming", func(t *testing.T) {
		t.Parallel()
		msgs := []genaiapi.Message{
			{
				Role: genaiapi.System,
				Type: genaiapi.Text,
				Text: systemPrompt,
			},
			{
				Role: genaiapi.User,
				Type: genaiapi.Text,
				Text: prompt,
			},
		}
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
		opts := genaiapi.CompletionOptions{MaxTokens: 10, Seed: 1}
		err2 := l.PromptStreaming(ctx, msgs, &opts, words)
		close(words)
		wg.Wait()
		if err2 != nil {
			t.Fatal(err2)
		}
		checkAnswer(t, got)
	})
}

func TestMistralTool(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping this test case when -short is used")
	}
	// Sadly Q2_K is too quantized for this test to pass, so we need to use
	// Q3_K_S which is 3.2GiB.
	l := loadModel(t, "hf:bartowski/Mistral-7B-Instruct-v0.3-GGUF/HEAD/Mistral-7B-Instruct-v0.3-Q3_K_S")
	// Refs:
	// - SpecialTokens in
	//   https://github.com/mistralai/mistral-common/blob/main/src/mistral_common/tokens/tokenizers/base.py
	// - InstructTokenizerV3 in
	//   https://github.com/mistralai/mistral-common/blob/main/src/mistral_common/tokens/tokenizers/sentencepiece.py
	// //py/mistral_test.py
	l.Encoding = &llamacpp.PromptEncoding{
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
				Parameters: tools.MistralFunctionParams{
					Type: "object",
					Properties: map[string]tools.MistralProperty{
						"location": {
							Type:        "string",
							Description: "The city and country, e.g. San Francisco, US or Montréal, CA or Berlin, DE",
						},
						"format": {
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
	msgs := []genaiapi.Message{
		{
			Role: genaiapi.AvailableTools,
			Type: genaiapi.Text,
			Text: string(toolsB),
		},
		{
			Role: genaiapi.User,
			Type: genaiapi.Text,
			Text: `What's\u2581the\u2581weather\u2581like\u2581today\u2581in\u2581Paris`,
		},
	}
	for _, m := range msgs {
		t.Log(m)
	}
	msgsl := len(msgs)
	c, err := llamacpp.New(l.baseURL, l.Encoding)
	if err != nil {
		t.Fatal(err)
	}
	opts := genaiapi.CompletionOptions{MaxTokens: 100, Seed: 1}
	msg, err := c.Completion(ctx, msgs, &opts)
	if err != nil {
		t.Fatal(err)
	}
	s := msg.Text
	msgs = append(msgs, parseToolResponse(t, s, 0)...)
	for _, m := range msgs[msgsl:] {
		t.Log(m)
	}
	msgsl = len(msgs)
	msg, err = c.Completion(ctx, msgs, &opts)
	if err != nil {
		t.Fatal(err)
	}
	s = msg.Text
	msgs = append(msgs, genaiapi.Message{
		Role: genaiapi.Assistant,
		Type: genaiapi.Text,
		Text: s,
	})
	for _, m := range msgs[msgsl:] {
		t.Log(m)
	}
	msgsl = len(msgs)
	if !strings.Contains(s, " 43 ") && !strings.Contains(s, " 43°") {
		t.Fatalf("expected 43°C, got %q", s)
	}

	// Now do a calculation!
	msgs = append(msgs, genaiapi.Message{
		Role: genaiapi.User,
		Type: genaiapi.Text,
		Text: "Give me the result of 43215 divided by 215.",
	})
	for _, m := range msgs[msgsl:] {
		t.Log(m)
	}
	msgsl = len(msgs)
	if msg, err = c.Completion(ctx, msgs, &opts); err != nil {
		t.Fatal(err)
	}
	s = msg.Text
	msgs = append(msgs, parseToolResponse(t, s, 1)...)
	for _, m := range msgs[msgsl:] {
		t.Log(m)
	}
	msgsl = len(msgs)
	if msg, err = c.Completion(ctx, msgs, &opts); err != nil {
		t.Fatal(err)
	}
	s = msg.Text
	msgs = append(msgs, genaiapi.Message{
		Role: genaiapi.Assistant,
		Type: genaiapi.Text,
		Text: s,
	})
	for _, m := range msgs[msgsl:] {
		t.Log(m)
	}
	if !strings.Contains(s, "201") {
		t.Fatalf("expected 201, got %q", s)
	}
}

func parseToolResponse(t *testing.T, got string, id int) []genaiapi.Message {
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
	return []genaiapi.Message{
		{
			Role: genaiapi.ToolCall,
			Type: genaiapi.Text,
			Text: string(c),
		},
		{
			Role: genaiapi.ToolCallResult,
			Type: genaiapi.Text,
			Text: string(r),
		},
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

// loadModel returns the models in ../default_config.yml to ensure they are valid.
func loadModel(t *testing.T, model PackedFileRef) *Session {
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
	processed := strings.ToLower(got)
	// Accept ok, chief.
	processed = strings.Replace(processed, ",", "", 1)
	if want := "ok chief"; !strings.Contains(processed, want) {
		if runtime.GOOS == "darwin" && os.Getenv("CI") == "true" && os.Getenv("GITHUB_ACTION") != "" {
			t.Log("TODO: Figure out why macOS GitHub hosted runner return an empty string")
		} else {
			t.Helper()
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
