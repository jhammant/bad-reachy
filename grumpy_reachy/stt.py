"""
Local Speech-to-Text for Grumpy Reachy
======================================
Uses faster-whisper for local transcription on Mac
"""

import numpy as np
from typing import Optional
import wave
import io


class LocalWhisperSTT:
    """Local speech-to-text using faster-whisper."""

    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the whisper model."""
        try:
            from faster_whisper import WhisperModel

            # Use CPU on Mac for compatibility, or try MPS
            print(f"[STT] Loading faster-whisper model: {self.model_size}")
            self.model = WhisperModel(
                self.model_size,
                device="cpu",  # cpu is most compatible
                compute_type="int8"  # Faster on CPU
            )
            print("[STT] Model loaded successfully")
        except ImportError:
            print("[STT] ERROR: faster-whisper not installed!")
            print("[STT] Install with: pip install faster-whisper")
            self.model = None
        except Exception as e:
            print(f"[STT] Error loading model: {e}")
            self.model = None

    async def transcribe(self, audio_data: bytes, sample_rate: int = 16000) -> Optional[str]:
        """Transcribe audio bytes to text."""
        if self.model is None:
            print("[STT] Model not loaded!")
            return None

        try:
            # Convert bytes to numpy array
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            # Transcribe
            segments, info = self.model.transcribe(
                audio_np,
                beam_size=5,
                language="en",
                vad_filter=True,  # Filter out non-speech
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=400,
                )
            )

            # Combine segments
            text = " ".join([segment.text for segment in segments]).strip()

            if text:
                print(f"[STT] Transcribed: {text}")

            return text if text else None

        except Exception as e:
            print(f"[STT] Transcription error: {e}")
            return None

    def is_ready(self) -> bool:
        """Check if STT is ready."""
        return self.model is not None


class GroqWhisperSTT:
    """Fallback: Groq's Whisper API if local doesn't work."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        import httpx
        self.client = httpx.AsyncClient(timeout=30.0)

    async def transcribe(self, audio_data: bytes, sample_rate: int = 16000) -> Optional[str]:
        """Transcribe using Groq's Whisper."""
        if not self.api_key:
            return None

        try:
            # Create WAV in memory
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data)
            wav_buffer.seek(0)

            response = await self.client.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={"file": ("audio.wav", wav_buffer, "audio/wav")},
                data={"model": "whisper-large-v3-turbo", "language": "en"}
            )
            response.raise_for_status()
            return response.json().get("text", "").strip() or None
        except Exception as e:
            print(f"[STT-Groq] Error: {e}")
            return None
