# slack-bot

slack-bot integrates seamlessly into Slack to run either a LLM instruct or an
image generation, or both!


## App Configuration

Do this first. Configure the Slack App at https://api.slack.com/apps:

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
        - app_home_opened
        - app_mention
        - app_uninstalled
        - member_joined_channel
        - message.channels
        - message.groups
        - message.im
        - message.mpim
5. Install App
    - Install to a workspace you are admin of

Read more at https://api.slack.com/apis/socket-mode. In particular, you cannot
add an app using Socket Mode to the public Slack App Directory.



## Local installation

Install the [Go toolchain](https://go.dev/dl/)

```
go install github.com/maruel/sillybot/cmd/slack-bot@latest
slack-bot -ig -bottoken <bot-token> -apptoken <app-token>
```

Warning: it takes a while (several minutes, more if you don't have a 1Gbps
network) to download the models on first run, be patient.


### Optional

Install `llama-server` from [llama.cpp](https://github.com/ggerganov/llama.cpp).
