#!/usr/bin/env python3
# Copyright 2024 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

"""Runs a LLM."""

import argparse
import base64
import datetime
import http.server
import io
import json
import logging
import os
import signal
import sys
import time

import transformers
import huggingface_hub
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE in ("cuda", "mps") else torch.float32


def get_generator(seed):
  """Returns a deterministic random number generator."""
  if DEVICE in ("cuda", "mps"):
    return torch.Generator(DEVICE).manual_seed(seed)
  return torch.Generator().manual_seed(seed)


def load_phi_3_mini():
  """Returns Phi-3 medium."""
  # https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
  model_id = "microsoft/Phi-3-mini-4k-instruct"
  return transformers.pipeline(
      "text-generation",
      model=transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=DEVICE,
        torch_dtype="auto",
        attn_implementation="eager",
        trust_remote_code=True),
      tokenizer=transformers.AutoTokenizer.from_pretrained(model_id))


def load_phi_3_medium():
  """Returns Phi-3 medium."""
  # https://huggingface.co/microsoft/Phi-3-medium-128k-instruct
  model_id = "microsoft/Phi-3-medium-128k-instruct"
  # attn_implementation="eager" ?
  return transformers.pipeline(
      "text-generation",
      model=transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=DEVICE,
        torch_dtype="auto",
        attn_implementation="eager",
        trust_remote_code=True),
      tokenizer=transformers.AutoTokenizer.from_pretrained(model_id))


class Handler(http.server.BaseHTTPRequestHandler):
  _pipe = None

  def do_POST(self):
    try:
      logging.info("Got request %s", self.path)
      if self.path != "/v1/chat/completions":
        self.send_error(404, "unknown path")
        return
      start = time.time()
      content_length = int(self.headers['Content-Length'])
      post_data = self.rfile.read(content_length)
      data = json.loads(post_data)
      # TODO: Structured format and verifications.
      stream = data.get("stream", False)
      prompt = data["messages"]
      output = self._pipe(
          prompt,
          max_new_tokens=500,
          return_full_text=False,
          temperature=0.0, # ??
          do_sample=False,
          )[0]['generated_text']
      if stream:
        resp = {
          "choices": [
            {
              "delta": {
                "content": output,
              },
              "finish_reason": "stop",
            },
          ],
        }
        raw = b'data: ' + json.dumps(resp).encode("ascii")
      else:
        resp = {
          "choices": [
            {
              "finish_reason": "stop",
              "message": {
                "role": "assistant",
                "content": output,
              },
            },
          ],
        }
        raw = json.dumps(resp).encode("ascii")

      self.send_response(200)
      self.send_header("Content-Type", "application/json")
      self.end_headers()
      self.wfile.write(raw)
      logging.info(f"Generated text for {prompt} -> {output} in {time.time()-start:.1f}s")
    except:
      self.send_error(500)
      raise

def main():
  parser = argparse.ArgumentParser(description=sys.modules[__name__].__doc__)
  parser.add_argument("--token",
                      help="Create a read token at htps://huggingface.co/settings/tokens")
  parser.add_argument("--port", default=8032, type=int)
  args = parser.parse_args()
  logging.basicConfig(level=logging.DEBUG)

  if args.token:
    huggingface_hub.login(token=args.token)
  if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
  Handler._pipe = load_phi_3_mini()
  httpd = http.server.HTTPServer(("localhost", args.port), Handler)
  logging.info(f"Started server on port {args.port}")

  def handle(signum, frame):
    # It tried various way to quick cleanly but it's just a pain.
    # TODO: Figure out how to do it the right way.
    logging.info("Got signal")
    #httpd.server_close()
    #logging.info("Did server_close")
    sys.exit(0)
  signal.signal(signal.SIGINT, handle)
  signal.signal(signal.SIGTERM, handle)

  httpd.serve_forever()
  logging.info("quitting (this never runs)")
  return 0


if __name__ == "__main__":
  sys.exit(main())
