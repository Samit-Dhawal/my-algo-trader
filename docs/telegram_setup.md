# Telegram Bot Setup Guide

## 1. Create a Telegram Bot

1. Open Telegram and search for [@BotFather](https://t.me/botfather)
2. Send `/start` to BotFather
3. Send `/newbot` and follow the instructions:
   - Choose a name for your bot (e.g., "My Trading Bot")
   - Choose a username ending in 'bot' (e.g., "mytradingalert_bot")
4. Save the API token provided by BotFather

## 2. Get Your Chat ID

### Method 1: Using the Bot
1. Start a conversation with your bot
2. Send any message to your bot
3. Visit: `https://api.telegram.org/bot<YourBotToken>/getUpdates`
4. Look for your chat ID in the response

### Method 2: Using a Helper Bot
1. Search for [@userinfobot](https://t.me/userinfobot) on Telegram
2. Send `/start` to get your chat ID

## 3. Environment Configuration

Add to your `.env` file:
```bash
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here