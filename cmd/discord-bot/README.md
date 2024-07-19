# discord-bot

discord-bot integrates seamlessly into Discord to run either a LLM instruct or an
image generation, or both!


## Usage

Chat with it!


### List of commands

- `/meme_auto <description> <seed>`: Generate a meme in full automatic mode.
  Create both the image and labels by leveraging the LLM.
    - `<description>`: Description used to generate both the meme labels and
      background image. The LLM will enhance both.
    - `<seed>`: Seed to use to enable (or disable with 0) deterministic image
      generation. Defaults to 1"
- `/meme_manual <image_prompt> <labels_content> <seed>`: Generate a meme in full
  manual mode. Specify both the image and the labels yourself.
    - `<image_prompt>`: Exact Stable Diffusion style prompt to use to generate the image.
    - `<labels_content>`: Exact text to overlay on the image. Use comma to split lines.
    - `<seed>`: Seed to use to enable (or disable with 0) deterministic image
      generation. Defaults to 1
- `/meme_labels_auto <description> <seed>`: Generate meme labels in automatic
  mode. Create the text by leveraging the LLM.
    - `<description>`: Description to use to generate the meme labels. The LLM will enhance
      it.
    - `<seed>`: Seed to use to enable (or disable with 0) deterministic image
      generation. Defaults to 1
- `/image_auto <description> <seed>`: Generate an image in automatic mode. It
  automatically uses the LLM to enhance the prompt.
    - `<description>`: Description to use to generate the image. The LLM will
      enhance it.
    - `<seed>`: Seed to use to enable (or disable with 0) deterministic image
      generation. Defaults to 1
- `/image_manual <image_prompt> <seed>`: Generate an image in manual mode.
    - `<image_prompt>`: Exact Stable Diffusion style prompt to use to generate
      the image.
    - `<seed>`: Seed to use to enable (or disable with 0) deterministic image
      generation. Defaults to 1
- `/list_models`: List available LLM models and the one currently used.
- `/forget <system_prompt>`: Forget our past conversation. Optionally
  overrides the system prompt. Use this to iterate quickly on new system
  prompts. You can use it without argument to revert to the standard system
  prompt configured in `config.yml`.
    - `<system_prompt>`: New system prompt to use.

Find the list in [`discord_bot.go`](discord_bot.go) by searching for
`ApplicationCommand`.


## App Configuration

Do this first.

1. In your Discord client App:
    - User settings, Advanced, Enable "Developer Mode"
    - User settings, My Account, Enable SECURITY KEYS (or another MFA).
2. Configure the Discord Application at https://discord.com/developers/applications
    - "New Team"
    - "New Application"
    - Add name, description, picture. You can generate a free picture with
      https://meta.ai.
    - Setup web server: (seems to be optional)
        - Add TERMS OF SERVICE URL and PRIVACY POLICY URL.
    - Bot
        - Privileged Gateway Intents:
            - Enable: MESSAGE CONTENT INTENT; SERVER MEMBERS INTENT PRESENCE INTENT
      - Token
          - Click "Reset Token"
          - Click "Copy" and save it as `token_discord.txt`
    - Installation, Guild Install:
        - Scopes: applications.commands, bot
            - Permissions: Connect, Send Messages
        - Install Link
            - Copy the link and share it to friend.


## Enable web search tool

Tool (function) calling is only supported on select LLM models.

You need a Google Cloud project API key and a search key.

- Get a GCP API key at https://cloud.google.com/docs/authentication/api-keys
- Get a Programmable Search Engine at
  https://programmablesearchengine.google.com/controlpanel/create


## Local installation and running

Install the [Go toolchain](https://go.dev/dl/)

```
go install github.com/maruel/sillybot/cmd/discord-bot@latest
discord-bot -ig -token <bot-token>
```

Warning: it takes a while (several minutes, more if you don't have a 1Gbps
network) to download the models on first run, be patient.


### Optional

- Install `llama-server` from [llama.cpp](https://github.com/ggerganov/llama.cpp).
- If you have nvidia hardware, install [CUDA](https://developer.nvidia.com/cuda-downloads).
- On macOS, use [`mac_vram.sh`](/mac_vram.sh) to increase the amount of VRAM
  usable for the model.
- You can run either or both the LLM and Image generation servers on separate
  machines, especially if your computer is not powerful enough to run both
  simultaneously. See the `remote` configuration in config.yml.
