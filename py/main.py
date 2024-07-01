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
import signal
import sys
import time

import diffusers
import huggingface_hub
import torch


def load():
  if False:
    pipe = diffusers.StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
  else:
    pipe = diffusers.DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", variant="fp16")
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
    pipe.scheduler = diffusers.LCMScheduler.from_config(pipe.scheduler.config)
  if sys.platform == "darwin":
    pipe = pipe.to("mps", dtype=torch.float16)
  else:
    fuck()
    pipe = pipe.to("cuda", dtype=torch.float16)
  return pipe


class Handler(http.server.BaseHTTPRequestHandler):
  _pipe = None
  _neg = "out of frame, lowers, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face"
  # , disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"

  def do_POST(self):
    logging.info("Got request")
    start = time.time()
    content_length = int(self.headers['Content-Length'])
    post_data = self.rfile.read(content_length)
    data = json.loads(post_data)
    # TODO: Structured format and verifications.
    prompt = data["message"]
    steps = data["steps"]
    img = self._pipe(
        prompt=prompt,
        negative_prompt=self._neg,
        num_inference_steps=steps,
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
    logging.info(f"Generated image for {prompt} in {time.time()-start:.1f}s")
    img.save(datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S") + ".png")


def main():
  parser = argparse.ArgumentParser(description=sys.modules[__name__].__doc__)
  parser.add_argument("--token",
                      help="Create a read token at htps://huggingface.co/settings/tokens")
  parser.add_argument("--port", default=8032, type=int)
  args = parser.parse_args()
  logging.basicConfig(level=logging.DEBUG)

  if args.token:
    huggingface_hub.login(token=args.token)
  Handler._pipe = load()
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
