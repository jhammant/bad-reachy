"""
Text-to-Speech for Bad Reachy
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
        self.voice_id: Optional[str] = None

    async def synthesize(self, text: str) -> Optional[bytes]:
        """Convert text to speech audio bytes."""
        if not text.strip():
            return None

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                # Build request for /speak endpoint
                data = {
                    "text": text,
                    "temperature": 0.8,
                }

                # Add voice_id if set
                if self.voice_id:
                    data["voice_id"] = self.voice_id

                response = await client.post(
                    f"{self.server_url}/speak",
                    json=data
                )
                response.raise_for_status()

                result = response.json()
                audio_url = result.get("audio_url")

                if audio_url:
                    # Fetch the audio file
                    full_url = f"{self.server_url}{audio_url}" if audio_url.startswith("/") else audio_url
                    audio_response = await client.get(full_url)
                    audio_response.raise_for_status()
                    return audio_response.content
                return None

        except httpx.TimeoutException:
            print("[TTS] Timeout - speech generation took too long")
            return None
        except Exception as e:
            print(f"[TTS] Error: {e}")
            return None

    def set_voice_sample(self, path: str):
        """Set a voice sample for cloning (kept for compatibility)."""
        if Path(path).exists():
            self.voice_sample_path = path
            print(f"[TTS] Voice sample set: {path}")
        else:
            print(f"[TTS] Voice sample not found: {path}")

    def set_voice_id(self, voice_id: str):
        """Set the voice ID to use for TTS."""
        self.voice_id = voice_id
        print(f"[TTS] Voice ID set: {voice_id}")

    async def test_connection(self) -> bool:
        """Test if Chatterbox server is running."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.server_url}/health")
                return response.status_code == 200
        except:
            return False


async def play_audio_on_reachy(reachy, audio_bytes: bytes, sample_rate: int = 24000):
    """Play audio through Reachy's speakers or system audio."""
    if audio_bytes is None:
        return

    try:
        played = False

        # Try Reachy's audio playback first
        if reachy:
            if hasattr(reachy, 'audio') and hasattr(reachy.audio, 'play'):
                await reachy.audio.play(audio_bytes, sample_rate)
                played = True
            elif hasattr(reachy, 'media') and hasattr(reachy.media, 'speaker'):
                reachy.media.speaker.play(audio_bytes, sample_rate)
                played = True

        # Fallback to sounddevice
        if not played:
            try:
                import sounddevice as sd
                import numpy as np
                from scipy import signal
                from scipy.io import wavfile
                import io

                # TTS returns WAV file - use scipy which handles float32 WAVs
                wav_buffer = io.BytesIO(audio_bytes)
                try:
                    file_sample_rate, audio_data = wavfile.read(wav_buffer)
                    print(f"[TTS] WAV: {audio_data.dtype}, {file_sample_rate}Hz, {len(audio_data)} samples")

                    # Convert to float32 if needed
                    if audio_data.dtype == np.float32:
                        audio_np = audio_data
                    elif audio_data.dtype == np.int16:
                        audio_np = audio_data.astype(np.float32) / 32768.0
                    elif audio_data.dtype == np.int32:
                        audio_np = audio_data.astype(np.float32) / 2147483648.0
                    else:
                        audio_np = audio_data.astype(np.float32)

                    # Use sample rate from WAV file
                    sample_rate = file_sample_rate
                except Exception as e:
                    print(f"[TTS] WAV parse failed: {e}")
                    return

                # Reachy Mini Audio only supports 16000 Hz - resample if needed
                device_rate = 16000
                if sample_rate != device_rate:
                    num_samples = int(len(audio_np) * device_rate / sample_rate)
                    audio_np = signal.resample(audio_np, num_samples)
                    print(f"[TTS] Resampled from {sample_rate}Hz to {device_rate}Hz ({len(audio_np)} samples)")

                # Normalize to prevent clipping
                max_val = np.max(np.abs(audio_np))
                if max_val > 1.0:
                    audio_np = audio_np / max_val
                    print(f"[TTS] Normalized audio (was {max_val:.2f})")

                print(f"[TTS] Playing via sounddevice ({len(audio_np)} samples at {device_rate}Hz)")
                sd.play(audio_np, device_rate)
                sd.wait()
            except ImportError as e:
                if 'scipy' in str(e):
                    print("[TTS] scipy not installed - cannot resample audio. Install with: pip install scipy")
                else:
                    print("[TTS] No audio output available (sounddevice not installed)")
            except Exception as e:
                print(f"[TTS] sounddevice playback error: {e}")

    except Exception as e:
        print(f"[TTS] Playback error: {e}")
