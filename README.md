# Silly Bot

A simple Discord and Slack bot written in Go that natively serves LLM and Stable
Diffusion to chat, generate images, and memes!

⚠️**Warning**⚠️ : *This is a work in progress. There is no privacy control. Please
file an [issue](https://github.com/maruel/sillybot/issues/new) for feature
requests or bugs found.*

I made a talk about it: [youtu.be/uGkrjsMl-ok](https://youtu.be/uGkrjsMl-ok)!
[![sillybot](https://github.com/user-attachments/assets/cff11181-b28b-440a-91d9-42d7a669bbde)](https://youtu.be/uGkrjsMl-ok)


## Usage

Type in chat:

`/meme_auto sillybot meme generator awesome`

Receive:

![Meme Lord in Training](https://raw.githubusercontent.com/wiki/maruel/sillybot/meme_lord.png)

Talk with it and use its commands as described at:

- For Discord, see [cmd/discord-bot/README.md](cmd/discord-bot#usage).
- For Slack, see [cmd/slack-bot/README.md](cmd/slack-bot#usage).


## Features

- Generates memes in full automatic mode; it generates both labels and image
  description.
- Generates memes in manual mode for more precision.
- Chat interface with resettable system prompt.
- Uses WebSocket so no need to setup a web server! 🎉
- Works on Ubuntu (linux), macOS and Windows! 🪟
- Supported backends:
    - Uses [llama.cpp](https://github.com/ggerganov/llama.cpp).
    - Optionally uses pytorch python backend for experiments.
- Runs as a chat bot LLM instructs. Tested to work with:
    - [Gemma-2-9B instruct](https://huggingface.co/google/gemma-2-9b-it) in
      [quantized form](https://huggingface.co/bartowski/gemma-2-9b-it-GGUF)
      with 8K context window.
    - [Gemma-2-27B instruct](https://huggingface.co/google/gemma-2-27b-it) in
      [quantized form](https://huggingface.co/bartowski/gemma-2-27b-it-GGUF)
      with 8K context window.
    - [Meta-Llama-3.1-8B-instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
      in [quantized
      form](https://huggingface.co/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF)
      with 128K context window.
    - [Meta-Llama-3-70B-instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct)
      in [quantized
      form](https://huggingface.co/bartowski/Meta-Llama-3-70B-Instruct-GGUF)
      with 8K context window.
    - [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
      in [quantized
      form](https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF)
      with 32K (!) context window.
    - [Mistral-Nemo-Instruct-2407](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407)
      in [quantized
      form](https://huggingface.co/bartowski/Mistral-Nemo-Instruct-2407-GGUF)
      with 128K (!) context window.
    - [Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) "Phi-3.1"
      in [quantized
      form](https://huggingface.co/bartowski/Phi-3.1-mini-4k-instruct-GGUF)
      with 4K context window.
    - [Phi-3-mini-128k-instruct](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct) "Phi-3.1"
      in [quantized
      form](https://huggingface.co/bartowski/Phi-3.1-mini-4128-instruct-GGUF)
      with 128K (!!) context window. Requires a ton of RAM.
    - [Phi-3-medium-4k-instruct](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct)
      in [quantized
      form](https://huggingface.co/bartowski/Phi-3-medium-4k-instruct-GGUF)
      with 4K context window.
    - [Phi-3-medium-128k-instruct](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct)
      in [quantized
      form](https://huggingface.co/bartowski/Phi-3-medium-128k-instruct-GGUF)
      with 128K (!!) context window. Requires a ton of RAM.
    - [Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) in
      [quantized form](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF)
      with 32K (!) context window. Extremely small model, super fast. Perfect to
      run in unit tests.
    - [Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct) in
      [quantized form](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct-GGUF)
      with 32K (!) context window. Super small model, super fast.
    - [Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) in
      [quantized form](https://huggingface.co/Qwen/Qwen2-7B-Instruct-GGUF)
      with 32K (!) context window.
- Runs as an image generator. Tested to work with:
    - [Segmind SSD-1B](https://huggingface.co/segmind/SSD-1B) coupled with [LCM
      Lora](https://huggingface.co/latent-consistency/lcm-lora-ssd-1b) is quite
      fast. Segmind SSD-1B renders under 10s on a MacBook Pro M3 Max, under
      100s on an Intel i7-13700 on Ubuntu and under 160s on a i9-13900H on Windows
      11. (Depends on the number of steps selected)
    - [Stable Diffusion 3
      Medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium)
      which is too slow on a M3 Max IMO.


## Hardware requirement

- **LLM**: any <4 years old computer, really. A GPU is not required. If you are
  unsure, start with Qwen2 0.5B by using `model: "qwen2-0_5b-instruct-q5_k_m"`
  in the `llm:` section of `config.yml`. This requires 1GiB of RAM. Go up with
  larger models from there.
    - I recommend to chat with the bot, use the `/metrics` command and iterate
      to use increasingly larger models as long as you can keep >=200tok/s for
      prompt parsing and >=20tok/s for generation.
    - When using models with large context window (>8K) like Llama 3.1 and
      Mistral Nemo, you may want to explicitly limit the context window to a
      smaller size like 16384, 32768 or 65536 so it reduces memory requirements.
      Counter-intuitively, reducing the context window too much will slow down
      generation.
- **Image Generation**: a nVidia GPU with 6GiB of video memory (VRAM) (12GiB is
  better) or a MacBook Pro. While it works on CPU, expect a minute or two to
  generate each image.

**Pro-tip**: You can use 2 computers: one running the LLM and one the image
generation! Start the llm/imagegen server manually then use the `remote:` option
in `config.yml`.


## Installation

Both function essentially the same but the Application configuration on the
server (Discord or Slack) is different:

- For Discord, see
  [cmd/discord-bot/README.md](cmd/discord-bot#app-configuration).
- For Slack, see
  [cmd/slack-bot/README.md](cmd/slack-bot#app-configuration).


## Dev

[![Go Reference](https://pkg.go.dev/badge/github.com/maruel/sillybot/.svg)](https://pkg.go.dev/github.com/maruel/sillybot/)
[![codecov](https://codecov.io/gh/maruel/sillybot/graph/badge.svg?token=33RREVZMMP)](https://codecov.io/gh/maruel/sillybot)


## Acknowledgements

This project greatly benefit from
[llama.cpp](https://github.com/ggerganov/llama.cpp) by [Georgi
Gerganov](https://github.com/ggerganov), previous versions leveraged
[llamafile](https://github.com/Mozilla-Ocho/llamafile) by [Justine
Tunney](https://github.com/jart), all open source contributors and all the
companies providing open-weights models.


## Author

`sillybot` was created with ❤️️ and passion by [Marc-Antoine
Ruel](https://github.com/maruel).
