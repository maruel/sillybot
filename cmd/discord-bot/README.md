# discord-bot

discord-bot integrates seamlessly into Discord to run either a LLM instruct or an
image generation, or both!


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
