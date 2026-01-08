# Grumpy Reachy

A sarcastic, cynical robot assistant with a bad attitude. It helps you, but complains the entire time.

> "Oh great, you're here. What do you want?"

## Features

- **Bad Attitude**: Rude, sarcastic, and loves to swear
- **Still Helpful**: Answers questions and does tasks (reluctantly)
- **Emotional Head Movements**: Express annoyance through body language
- **Web Search**: Can look things up (and complain about it)
- **100% Local**: LLM, STT, and TTS all run on your Mac
- **Dashboard**: Monitor your grumpy friend via web interface

## Architecture

```
Reachy Mini ←→ Your Mac (all processing)
                  ├── LM Studio (uncensored LLM)
                  ├── faster-whisper (local STT)
                  └── Chatterbox (local TTS)
```

## Requirements

- Reachy Mini robot
- Mac with LM Studio running
- Chatterbox TTS server running
- Python 3.10+

## Setup

### 1. Start LM Studio

Load an uncensored/abliterated model in LM Studio and start the server on port 1234.

### 2. Start Chatterbox TTS

```bash
cd /path/to/chatterbox
python -m chatterbox.server --host 0.0.0.0 --port 8000
```

### 3. Deploy to Reachy

```bash
ssh pollen@reachy-mini.local
cd grumpy-reachy
python3 -m venv venv --system-site-packages
source venv/bin/activate
pip install -e .
python run_app.py
```

### 4. Open Dashboard

Visit `http://reachy-ip:8080` to see:
- Live camera feed
- Current emotion state
- Conversation history
- Swear counter

## Personality

Grumpy Reachy is:
- Sarcastic and cynical
- Uses profanity casually
- Makes dark jokes
- Complains about everything
- Actually helpful (reluctantly)

Example responses:
- "Oh for fuck's sake, FINE. The answer is 42."
- "*sigh* Really? You're asking ME this?"
- "I looked it up for you since apparently you can't use Google yourself."

## Configuration

Environment variables:
- `LM_STUDIO_URL` - LM Studio API (default: http://localhost:1234/v1)
- `TTS_SERVER_URL` - Chatterbox server (default: http://localhost:8000)
- `WHISPER_MODEL` - Whisper model size (default: base)

## License

MIT - Do whatever you want with this grumpy robot.
