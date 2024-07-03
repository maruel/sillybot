# Silly Bot

A simple Discord and Slack bot written in Go that natively serves LLM and Stable
Diffusion (in python for now).

- Uses WebSocket so no need to setup a web server!
- Works on Ubuntu (linux), macOS and Windows!
- LLM: Tested to work with
  [Meta-Llama3-8B-instruct](https://huggingface.co/Mozilla/Meta-Llama-3-8B-Instruct-llamafile)
  at various quantization levels and [Gemma-2-27B
  instruct](https://huggingface.co/jartine/gemma-2-27b-it-llamafile) in Q6_K.
- Image: Tested to work with [Segmind SSD-1B](https://huggingface.co/segmind/SSD-1B) coupled with [LCM Lora](https://huggingface.co/latent-consistency/lcm-lora-ssd-1b) which is super fast, and [Stable
  Diffusion 3
  Medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium) which is
  too slow on a M3 Max.
    - Segmind SSD-1B renders under 5s on a MacBook Pro M3 Max, under 50s on an Intel i7-13700 on Ubuntu and under 80s on a i9-13900H on Windows 11.

## Discord

1. In the Discord App:
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

## Slack

Configure the Slack App at https://api.slack.com/apps:

1. Socket Mode
    - Enable Socket Mode.
2. OAuth & Permissions
    - Scopes, Bot Token Scopes, Add an OAuth Scope. _This is about what the bot
      has access to. The more it has access to, the more 'damage' it can do._
        - app_mentions:read
        - channels:history
        - channels:join
        - channels:read
        - channels:write.topic
        - chat:write
        - chat:write.customize
        - files:write
        - groups:history
        - groups:read
        - groups:write
        - groups:write.topic
        - im:history
        - im:read
        - im:write
        - metadata.message:read
        - mpim:history
        - mpim:read
        - mpim:write
        - mpim:write.history
        - reactions:read
        - team:read
        - users:read
        - users:write
3. App Home
    - Show Tabs, Messages Tab, "Allow users to send Slash commands and messages
      from the messages tab"
4. Event Subscriptions
    - Enable Events
    - Subscript to bot events. _This is which events the bot receives. This more
      it receives, the more spam it gets._
        - app_mention
        - member_joined_channel
        - message.channels
        - message.mpim
5. Install App
    - Install to a workspace you are admin of

Read more at https://api.slack.com/apis/socket-mode. In particular, you cannot
add an app using Socket Mode to the public Slack App Directory.
