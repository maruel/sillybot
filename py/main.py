#!/usr/bin/env python3
# Copyright 2024 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

"""Runs Stable Diffusion 3 Medium."""

import argparse
import base64
import datetime
import http.server
import io
import json
import logging
import os
import sys
import time

import diffusers
import huggingface_hub
import torch


def load():
  pipe = diffusers.StableDiffusion3Pipeline.from_pretrained(
      "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
  if sys.platform == "darwin":
    pipe = pipe.to("mps")
  else:
    pipe = pipe.to("cuda")
  return pipe


class Handler(http.server.BaseHTTPRequestHandler):
  _pipe = None
  _neg = "out of frame, lowers, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face"
  # , disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"

  def do_POST(self):
    print("Got request")
    start = time.time()
    content_length = int(self.headers['Content-Length'])
    post_data = self.rfile.read(content_length)
    data = json.loads(post_data)
    prompt = data["message"]
    img = self._pipe(
        prompt=prompt,
        negative_prompt=self._neg,
        num_inference_steps=28,
        guidance_scale=7.0,
    ).images[0]
    d = io.BytesIO()
    img.save(d, format="png")
    resp = {
        "image": base64.b64encode(d.getvalue()).decode(),
    }
    self.send_response(200)
    self.send_header("Content-Type", "application/json")
    self.end_headers()
    self.wfile.write(json.dumps(resp).encode("ascii"))
    print(f"Generated image for {prompt} in {time.time()-start:.1f}s")
    img.save(datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S") + ".png")


def main():
  parser = argparse.ArgumentParser(description=sys.modules[__name__].__doc__)
  parser.add_argument("--token",
                      help="Create a read token at htps://huggingface.co/settings/tokens")
  parser.add_argument("--port", default=8032, type=int)
  args = parser.parse_args()
  if args.token:
    huggingface_hub.login(token=args.token)
  Handler._pipe = load()
  httpd = http.server.HTTPServer(("", args.port), Handler)
  print(f"Started server on port {args.port}")
  sys.stdout.flush()
  try:
    httpd.serve_forever()
  except KeyboardInterrupt:
    pass
  httpd.server_close()
  print("quitting")
  return 0


if __name__ == "__main__":
  sys.exit(main())
