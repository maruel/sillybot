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

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE in ("cuda", "mps") else torch.float32


def get_generator(seed):
  """Returns a deterministic random number generator."""
  if DEVICE in ("cuda", "mps"):
    return torch.Generator(DEVICE).manual_seed(seed)
  return torch.Generator().manual_seed(seed)


def load_sd3():
  """Returns Stable Diffusion 3 Medium. Requires authentication to Hugging
  Face."""
  return diffusers.StableDiffusion3Pipeline.from_pretrained(
      "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)


def load_sdxl_lcm_lora():
  """Loads Stable Diffusion 1.0 XL with LCM LoRa."""
  pipe = diffusers.DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", variant="fp16")
  pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
  pipe.scheduler = diffusers.LCMScheduler.from_config(pipe.scheduler.config)
  return pipe


def load_segmind_ssd_1b_lcm_lora():
  """Returns Segmind SSD 1B with LCM LoRa."""
  pipe = diffusers.DiffusionPipeline.from_pretrained("segmind/SSD-1B")
  pipe.load_lora_weights("latent-consistency/lcm-lora-ssd-1b")
  pipe.scheduler = diffusers.LCMScheduler.from_config(pipe.scheduler.config)
  return pipe


def load_segmind_moe():
  import segmoe
  return segmoe.SegMoEPipeline("segmind/SegMoE-2x1-v0", device=DEVICE)


class Handler(http.server.BaseHTTPRequestHandler):
  _pipe = None
  #_neg = "out of frame, lowers, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face"
  # , disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"
  #_neg = "bad quality, worse quality"

  # See https://huggingface.co/segmind/SSD-1B
  # 1:1 Square:
  #_width = 1024
  #_height = 1024
  # 9:7:
  #_width = 1152
  #_height = 896
  # 19:13:
  _width = 1216
  _height = 832
  # 7:4 (which is close to 16:9):
  #_width = 1344
  #_height = 768

  def do_GET(self):
    try:
      if self.path == "/health":
        self.on_health()
      else:
        self.send_error(404)
    except Exception as e:
      self.send_error(500)
      print(str(e), file=sys.stderr)
      sys.exit(1)

  def do_POST(self):
    try:
      logging.info("Got request %s", self.path)
      if self.path == "/api/generate":
        self.on_generate()
      elif self.path == "/api/quit":
        self.on_quit()
      else:
        self.send_error(404)
    except Exception as e:
      self.send_error(500)
      print(str(e), file=sys.stderr)
      sys.exit(1)

  def reply_json(self, data):
    self.send_response(200)
    self.send_header("Content-Type", "application/json")
    self.end_headers()
    self.wfile.write(json.dumps(data).encode("ascii"))

  def on_health(self):
    self.reply_json({"status": "ok"})

  def on_quit(self):
    self.reply_json({"quitting": True})
    self.server.server_close()

  def on_generate(self):
    start = time.time()
    content_length = int(self.headers['Content-Length'])
    post_data = self.rfile.read(content_length)
    data = json.loads(post_data)
    # TODO: Structured format and verifications.
    prompt = data["message"]
    # Use 8 for Segmind + LCM Lora, 25 to 40 otherwise.
    steps = data["steps"]
    seed = data["seed"]
    img = self.gen_image(prompt, steps, seed)
    d = io.BytesIO()
    img.save(d, format="png")
    resp = {
        "image": base64.b64encode(d.getvalue()).decode(),
    }
    self.reply_json(resp)
    name = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S") + ".png"
    logging.info(f"Generated image for {prompt} in {time.time()-start:.1f}s; saving as {name}")
    img.save(name)

  @classmethod
  def gen_image(cls, prompt, steps, seed):
    img = cls._pipe(
        prompt=prompt,
        # Ned is not used when guidance_scale is 1.0.
        #negative_prompt=cls._neg,
        num_inference_steps=steps,
        generator=get_generator(seed),
        # Use 1.0 when using Segmind + LCM LoRA, 9.0 for Segmind raw, 7.0 for SD3.
        guidance_scale=1.0,
        width=cls._width,
        height=cls._height,
    ).images[0]
    return img


def main():
  parser = argparse.ArgumentParser(description=sys.modules[__name__].__doc__)
  parser.add_argument("--token",
                      help="Token to fetch from Hugging Face. Create a read token at htps://huggingface.co/settings/tokens")
  parser.add_argument("--host", default="localhost",
                      help="Host to listen to. Use 0.0.0.0 to listen on all IPs")
  parser.add_argument("--port", default=8032, type=int)
  parser.add_argument("--prompt", help="Run once and exit")
  args = parser.parse_args()
  logging.basicConfig(level=logging.DEBUG)

  if args.token:
    # Needed to retrieve SD3.
    huggingface_hub.login(token=args.token)
  if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
  Handler._pipe = load_segmind_ssd_1b_lcm_lora().to(DEVICE, dtype=DTYPE)
  #Handler._pipe = load_segmind_moe()
  logging.info("Model loaded using %s", DEVICE)

  if args.prompt:
    start = time.time()
    logging.info(f"Generating image for {args.prompt}")
    img = Handler.gen_image(args.prompt, 25, 1)
    name = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S") + ".png"
    logging.info(f"Generated image for {args.prompt} in {time.time()-start:.1f}s; saving as {name}")
    img.save(name)
    return 0

  httpd = http.server.HTTPServer((args.host, args.port), Handler)
  logging.info(f"Started server on port {args.host}:{args.port}")

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
