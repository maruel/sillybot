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
    - (seems to be optional) Setup web server, add TERMS OF SERVICE URL and PRIVACY POLICY URL.
    - Bot, Privileged Gateway Intents, Enable: MESSAGE CONTENT INTENT; SERVER MEMBERS INTENT PRESENCE INTENT
    - Installation, Guild Install:
        - Scopes: applications.commands, bot
        - Permissions: Connect, Send Messages


## Local installation and running

Install the [Go toolchain](https://go.dev/dl/)

```
go install github.com/maruel/sillybot/cmd/discord-bot@latest
discord-bot -ig -token <bot-token>
```

Warning: it takes a while (several minutes, more if you don't have a 1Gbps
network) to download the models on first run, be patient.


### Optional

Install `llama-server` from [llama.cpp](https://github.com/ggerganov/llama.cpp).
