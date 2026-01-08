"""
Text-to-Speech for Grumpy Reachy
================================
Uses Chatterbox TTS server (local, supports voice cloning)
"""

import httpx
import base64
from typing import Optional
from pathlib import Path


class ChatterboxTTS:
    """Chatterbox TTS client - local server with voice cloning support."""

    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url.rstrip("/")
        self.voice_sample_path: Optional[str] = None

    async def synthesize(self, text: str) -> Optional[bytes]:
        """Convert text to speech audio bytes."""
        if not text.strip():
            return None

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                # Build request
                data = {
                    "text": text,
                    "exaggeration": 0.7,  # More expressive for grumpy delivery
                    "cfg_weight": 0.5,
                }

                # Add voice sample for cloning if set
                if self.voice_sample_path and Path(self.voice_sample_path).exists():
                    with open(self.voice_sample_path, "rb") as f:
                        audio_bytes = f.read()
                    data["audio_prompt_base64"] = base64.b64encode(audio_bytes).decode()

                response = await client.post(
                    f"{self.server_url}/synthesize",
                    json=data
                )
                response.raise_for_status()

                result = response.json()
                audio_base64 = result.get("audio_base64")

                if audio_base64:
                    return base64.b64decode(audio_base64)
                return None

        except httpx.TimeoutException:
            print("[TTS] Timeout - speech generation took too long")
            return None
        except Exception as e:
            print(f"[TTS] Error: {e}")
            return None

    def set_voice_sample(self, path: str):
        """Set a voice sample for cloning."""
        if Path(path).exists():
            self.voice_sample_path = path
            print(f"[TTS] Voice sample set: {path}")
        else:
            print(f"[TTS] Voice sample not found: {path}")

    async def test_connection(self) -> bool:
        """Test if Chatterbox server is running."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.server_url}/health")
                return response.status_code == 200
        except:
            return False


async def play_audio_on_reachy(reachy, audio_bytes: bytes, sample_rate: int = 24000):
    """Play audio through Reachy's speakers."""
    if audio_bytes is None:
        return

    try:
        # Reachy's audio playback
        if hasattr(reachy, 'audio') and hasattr(reachy.audio, 'play'):
            await reachy.audio.play(audio_bytes, sample_rate)
        else:
            # Fallback: try media.speaker
            if hasattr(reachy, 'media') and hasattr(reachy.media, 'speaker'):
                reachy.media.speaker.play(audio_bytes, sample_rate)
            else:
                print("[TTS] No audio output available on Reachy")
    except Exception as e:
        print(f"[TTS] Playback error: {e}")
