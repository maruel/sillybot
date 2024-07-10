# Silly Bot

A simple Discord and Slack bot written in Go that natively serves LLM and Stable
Diffusion (in python for now).

`/meme_auto sillybot meme generator awesome`

![Meme Lord in Training](https://raw.githubusercontent.com/wiki/maruel/sillybot/meme_lord.png)

- Uses WebSocket so no need to setup a web server! 🎉
- Keeps memory of past conversations.
- Works on Ubuntu (linux), macOS and Windows! 🪟
- Supported backends:
    - Uses [llama.cpp](https://github.com/ggerganov/llama.cpp)'s `llama-server`
      if in `PATH`
    - Defaults to [llamafile](https://github.com/Mozilla-Ocho/llamafile)
      otherwise.
    - Optionally uses pytorch python backend for experiments.
- Runs as a chat bot LLM instructs. Tested to work with:
    - [Gemma-2-9B instruct](https://huggingface.co/google/gemma-2-9b-it) in
      [quantized form](https://huggingface.co/jartine/gemma-2-9b-it-llamafile)
      with 8K context window.
    - [Gemma-2-27B instruct](https://huggingface.co/google/gemma-2-27b-it) in
      [quantized form](https://huggingface.co/jartine/gemma-2-27b-it-llamafile)
      with 8K context window.
    - [Meta-Llama3-8B-instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
      in [quantized
      form](https://huggingface.co/Mozilla/Meta-Llama-3-8B-Instruct-llamafile)
      with 8K context window.
    - [Meta-Llama3-70B-instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct)
      in [quantized
      form](https://huggingface.co/Mozilla/Meta-Llama-3-70B-Instruct-llamafile)
      with 8K context window.
    - [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
      in [quantized
      form](https://huggingface.co/MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF)
      with 32K (!) context window.
    - [Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
      in [quantized
      form](https://huggingface.co/Mozilla/Mixtral-8x7B-Instruct-v0.1-llamafile)
      with 32K (!) context window.
    - [Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
      in [quantized
      form](https://huggingface.co/Mozilla/Phi-3-mini-4k-instruct-llamafile)
      (currently only supported when using llama.cpp llama-server, waiting
      for llamafile to be fixed upstream)
      with 4K context window.
    - [Phi-3-medium-128k-instruct](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct)
      in [quantized
      form](https://huggingface.co/Mozilla/Phi-3-medium-128k-instruct-llamafile)
      (currently only supported when using llama.cpp llama-server, waiting
      for llamafile to be fixed upstream)
      with 128K (!!) context window. Requires a ton of RAM.
- Runs as an image generator. Tested to work with:
    - [Segmind SSD-1B](https://huggingface.co/segmind/SSD-1B) coupled with [LCM
      Lora](https://huggingface.co/latent-consistency/lcm-lora-ssd-1b) is super
      fast. Segmind SSD-1B renders under 10s on a MacBook Pro M3 Max, under
      100s on an Intel i7-13700 on Ubuntu and under 160s on a i9-13900H on Windows
      11. (Depends on the number of steps selected)
    - [Stable Diffusion 3
      Medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium)
      which is too slow on a M3 Max IMO.

[![Go Reference](https://pkg.go.dev/badge/github.com/maruel/sillybot/.svg)](https://pkg.go.dev/github.com/maruel/sillybot/)


## Installation

Both function essentially the same but the Application configuration on the
server (Discord or Slack) is different:

- For Discord, see [cmd/discord-bot/README.md](cmd/discord-bot/README.md).
- For Slack, see [cmd/slack-bot/README.md](cmd/slack-bot/README.md).


## Acknowledgements

This project greatly benefit from
[llamafile](https://github.com/Mozilla-Ocho/llamafile) by [Justine
Tunney](https://github.com/jart),
[llama.cpp](https://github.com/ggerganov/llama.cpp) by [Georgi
Gerganov](https://github.com/ggerganov), all open source contributors and all
the companies providing open-weights models.


## Author

`sillybot` was created with ❤️️ and passion by [Marc-Antoine
Ruel](https://github.com/maruel).
