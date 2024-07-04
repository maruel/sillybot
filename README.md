# Silly Bot

A simple Discord and Slack bot written in Go that natively serves LLM and Stable
Diffusion (in python for now).

- Uses WebSocket so no need to setup a web server! 🎉
- Works on Ubuntu (linux), macOS and Windows! 🪟
- Runs as a chat bot LLM instructs via [Mozilla's excellent
  llamafile](https://github.com/Mozilla-Ocho/llamafile): Tested to work with:
    - [Meta-Llama3-8B-instruct](https://huggingface.co/Mozilla/Meta-Llama-3-8B-Instruct-llamafile)
      at various quantization levels
    - [Gemma-2-27B
      instruct](https://huggingface.co/jartine/gemma-2-27b-it-llamafile) in
      Q6_K.
    - [Phi-3-mini-4k-instruct](https://huggingface.co/Mozilla/Phi-3-mini-4k-instruct-llamafile)
      and
      [Phi-3-medium-128k-instruct](https://huggingface.co/Mozilla/Phi-3-medium-128k-instruct-llamafile)
      are currently broken, probably a misconfiguration. Help is appreciated! 🙋
- Runs as an image generator. Tested to work wwith:
    - [Segmind SSD-1B](https://huggingface.co/segmind/SSD-1B) coupled with [LCM
      Lora](https://huggingface.co/latent-consistency/lcm-lora-ssd-1b) which is
      super fast. Segmind SSD-1B renders under 5s on a MacBook Pro M3 Max, under
      50s on an Intel i7-13700 on Ubuntu and under 80s on a i9-13900H on Windows
      11.
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

This projects wouldn't have been as easy to make if it weren't for
[llamafile](https://github.com/Mozilla-Ocho/llamafile) by [Justine
Tunney](https://github.com/jart) and all the companies providing open-weights
models.


## Author

`sillybot` was created with ❤️️ and passion by [Marc-Antoine
Ruel](https://github.com/maruel).
