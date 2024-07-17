#!/usr/bin/env python3
# Copyright 2024 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

"""Test Mistral tokenization with native tools support.

See
https://github.com/mistralai/mistral-common/blob/main/examples/tokenizer.ipynb
for an example.
"""

import argparse
import json
import logging
import sys
import textwrap

try:
  import requests
  from mistral_common.protocol.instruct import messages
  from mistral_common.protocol.instruct import tool_calls
  from mistral_common.protocol.instruct import request
  from mistral_common.tokens.tokenizers import sentencepiece
  from mistral_common.tokens.tokenizers import mistral
except:
  print("Make sure to setup the environment first", file=sys.stderr)
  print("Run:", file=sys.stderr)
  print("  setup-test.sh", file=sys.stderr)
  print("  source venv-test/bin/activate", file=sys.stderr)
  sys.exit(1)


def create_completion(host, prompt, gbnf_grammar, seed):
  """Calls the /completion API on llama-server.

  See
  https://github.com/ggerganov/llama.cpp/tree/HEAD/examples/server#api-endpoints
  """
  data = {"prompt": prompt}
  print(f"  Request:")
  if seed:
    print(f"    Seed: {seed}")
  if gbnf_grammar:
    data["grammar"] = gbnf_grammar
    print(f"    Grammar:\n{textwrap.indent(gbnf_grammar, '      ')}")
  print(f"    Prompt:\n      {repr(prompt.rstrip())}")
  headers = {"Content-Type": "application/json"}
  # I realized there seem to be sensitivity to the exact JSON encoding used.
  print("Raw Request:")
  print(json.dumps(data))
  result = requests.post(f"http://{host}/completion", headers=headers, json=data).json()
  assert data.get("error") is None, data
  logging.info("Result: %s", result)
  content = result["content"]
  print(f"  Model: {result['model']}")
  # TODO: Handle mixed content.
  try:
    dump = json.dumps(json.loads(content), indent=2)
  except json.decoder.JSONDecodeError:
    dump = content
  print(f"  Result:\n{textwrap.indent(dump, '    ')}")
  return content


def get_current_weather(location: str, format: str) -> str:
  return "43"


def exec_msgs(host, seed, msgs):
  """Tokenize then send the request to the llama-server."""
  tools = [
      tool_calls.Tool(
          function=tool_calls.Function(
              name="get_current_weather",
              description="Get the current weather",
              parameters={
                  "type": "object",
                  "properties": {
                      "location": {
                          "type": "string",
                          "description": "The city and state, e.g. San Francisco, US or Montréal, CA or Berlin, DE",
                      },
                      "format": {
                          "type": "string",
                          "enum": ["celsius", "fahrenheit"],
                          "description": "The temperature unit to use. Infer this from the users location.",
                      },
                  },
                  "required": ["location", "format"],
              },
          )
      )
  ]
  tokenizer_v3 = mistral.MistralTokenizer.v3()
  tokenized = tokenizer_v3.encode_chat_completion(request.ChatCompletionRequest(tools=tools, messages=msgs))
  logging.info("Tokens: %s", tokenized.tokens)
  logging.info("Text: %s", tokenized.text)
  return create_completion(host, tokenized.text, "", seed)


def run(host, seed):
  exitcode = 0
  msgs = [
      messages.UserMessage(content="What's the weather like today in Paris"),
  ]
  ret = exec_msgs(host, seed, msgs)
  print(f"Got: {repr(ret)}")
  retdata = json.loads(ret)
  want = [
    {
      "name": "get_current_weather",
      "arguments": {
        "location": "Paris, FR",
        "format": "celsius"
      }
    }
  ]
  if retdata != want:
    print("Surprising return value")
    exitcode = 1

  # Call the functions requested.
  for i, callreq in enumerate(retdata):
    callid = "c%08d" % i
    name = callreq["name"]
    arguments = callreq["arguments"]
    call = tool_calls.ToolCall(id=callid, function=tool_calls.FunctionCall(name=name, arguments=arguments))
    msgs.append(messages.AssistantMessage(content=None, tool_calls=[call]))
    callret = getattr(sys.modules[__name__], name)(**arguments)
    msgs.append(messages.ToolMessage(tool_call_id=callid, name=name, content=callret))

  # Follow up with the model.
  ret = exec_msgs(host, seed, msgs)
  print(f"Got: {repr(ret)}")
  if " 43 " not in ret:
    exitcode = 1
  return exitcode


def main():
  parser = argparse.ArgumentParser(description=sys.modules[__name__].__doc__)
  parser.add_argument("--host", default="localhost:8080", help="llama.cpp server")
  parser.add_argument("--seed", default=1, help="seed value", type=int)
  parser.add_argument("-v", "--verbose", action="store_true", help="enables logging")
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO if args.verbose else logging.ERROR)
  try:
    return run(args.host, args.seed)
  except requests.exceptions.ConnectionError as e:
    print("\nDid you forget to pass --host?", file=sys.stderr)
    print(f"Error: {e}", file=sys.stderr)


if __name__ == "__main__":
  sys.exit(main())
