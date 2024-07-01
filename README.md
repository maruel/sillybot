# discord bot

A simple discord bot written in Go that natively serves LLM and Stable Diffusion
(in python for now).

- Works on linux, macOS and Windows.
- LLM: Tested to work with
  [Meta-Llama3-8B-instruct](https://huggingface.co/Mozilla/Meta-Llama-3-8B-Instruct-llamafile)
  at various quantization levels and [Gemma-2-27B
  instruct](https://huggingface.co/jartine/gemma-2-27b-it-llamafile) in Q6_K.
- Image: Tested to work with [Segmind SSD-1B](https://huggingface.co/segmind/SSD-1B) coupled with [LCM Lora](https://huggingface.co/latent-consistency/lcm-lora-ssd-1b) which is super fast, and [Stable
  Diffusion 3
  Medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium) which is
  too slow on a M3 Max.
    - Segmind SSD-1B renders under 5s on a MacBook Pro M3 Max, under 50s on an Intel i7-13700 on Ubuntu and under 80s on a i9-13900H on Windows 11.

## Setup

1. Discord App:
    - User settings, Advanced, Enable "Developer Mode"
    - User settings, My Account, Enable SECURITY KEYS (or another MFA).
2. https://discord.com/developers/applications
    - "New Team"
    - "New Application"
    - Add name, description, picture. You can generate a free picture with
      https://meta.ai.
    - (seems to be optional) Setup web server, add TERMS OF SERVICE URL and PRIVACY POLICY URL.
    - Bot, Privileged Gateway Intents, Enable: MESSAGE CONTENT INTENT; SERVER MEMBERS INTENT PRESENCE INTENT
    - Installation, Guild Install:
        - Scopes: applications.commands, bot
        - Permissions: Connect, Send Messages
