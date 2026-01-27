"""
Fast STT Options for Bad Reachy
================================
Groq Whisper API (very fast) + VAD for smart recording.
"""

import asyncio
import numpy as np
import wave
import io
from typing import Optional, Tuple
import os


class GroqWhisperSTT:
    """Groq's Whisper API - extremely fast cloud STT."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self._client = None

    async def transcribe(self, audio_data: bytes, sample_rate: int = 16000) -> Optional[str]:
        """Transcribe audio bytes to text."""
        if not self.api_key:
            print("[STT] No GROQ_API_KEY set")
            return None

        try:
            import httpx

            # Create WAV in memory
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data)
            wav_buffer.seek(0)

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    files={"file": ("audio.wav", wav_buffer, "audio/wav")},
                    data={
                        "model": "whisper-large-v3-turbo",
                        "language": "en",
                        "response_format": "text"
                    }
                )
                response.raise_for_status()
                text = response.text.strip()

                if text:
                    print(f"[STT-Groq] Transcribed: {text}")
                return text if text else None

        except Exception as e:
            print(f"[STT-Groq] Error: {e}")
            return None

    def is_ready(self) -> bool:
        return bool(self.api_key)


class VADRecorder:
    """Voice Activity Detection for smart recording."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.vad = None
        self._init_vad()

    def _init_vad(self):
        """Initialize VAD - try silero first, fall back to webrtcvad."""
        try:
            # Try silero VAD (most accurate)
            import torch
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=True
            )
            self.vad = ('silero', model, utils)
            print("[VAD] Using Silero VAD")
            return
        except Exception as e:
            print(f"[VAD] Silero not available: {e}")

        try:
            # Try webrtcvad
            import webrtcvad
            self.vad = ('webrtc', webrtcvad.Vad(1))  # Aggressiveness 0-3 (1=more sensitive)
            print("[VAD] Using WebRTC VAD")
            return
        except ImportError:
            print("[VAD] webrtcvad not installed. Install with: pip install webrtcvad")

        print("[VAD] No VAD available - using energy-based detection")
        self.vad = ('energy', None)

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Check if audio chunk contains speech."""
        if self.vad is None:
            return True

        vad_type, vad_model = self.vad[0], self.vad[1]

        try:
            if vad_type == 'silero':
                model = vad_model
                # Silero expects float32 tensor
                import torch
                audio_tensor = torch.from_numpy(audio_chunk.astype(np.float32))
                if audio_tensor.max() > 1.0:
                    audio_tensor = audio_tensor / 32768.0
                speech_prob = model(audio_tensor, self.sample_rate).item()
                return speech_prob > 0.5

            elif vad_type == 'webrtc':
                # WebRTC VAD expects 16-bit PCM, 10/20/30ms frames
                if audio_chunk.dtype == np.float32:
                    audio_int16 = (audio_chunk * 32767).astype(np.int16)
                else:
                    audio_int16 = audio_chunk.astype(np.int16)

                # Process in 30ms chunks
                frame_size = int(self.sample_rate * 0.03)
                for i in range(0, len(audio_int16) - frame_size, frame_size):
                    frame = audio_int16[i:i+frame_size]
                    if vad_model.is_speech(frame.tobytes(), self.sample_rate):
                        return True
                return False

            else:  # energy-based
                # Simple energy threshold - higher to ignore echo
                if audio_chunk.dtype == np.float32:
                    energy = np.sqrt(np.mean(audio_chunk ** 2))
                    return energy > 0.03  # Higher threshold
                else:
                    energy = np.sqrt(np.mean((audio_chunk / 32768.0) ** 2))
                    return energy > 0.03  # Higher threshold

        except Exception as e:
            print(f"[VAD] Error: {e}")
            return True  # Assume speech on error

    async def record_until_silence(
        self,
        max_duration: float = 5.0,
        silence_duration: float = 0.5,
        min_speech_duration: float = 0.2
    ) -> Tuple[bytes, float]:
        """
        Record audio until silence is detected after speech.

        Returns (audio_bytes, duration)
        """
        try:
            import sounddevice as sd
        except ImportError:
            print("[VAD] sounddevice not installed")
            return b'', 0.0

        chunk_duration = 0.03  # 30ms chunks for FASTER detection
        chunk_samples = int(self.sample_rate * chunk_duration)

        audio_chunks = []
        speech_started = False
        silence_counter = 0
        speech_counter = 0
        total_duration = 0.0

        silence_chunks_needed = int(silence_duration / chunk_duration)
        min_speech_chunks = int(min_speech_duration / chunk_duration)

        print(f"[VAD] Listening (max {max_duration}s, silence threshold {silence_duration}s)...")

        try:
            # Start recording
            stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='int16',
                blocksize=chunk_samples
            )
            stream.start()

            while total_duration < max_duration:
                # Read chunk
                chunk, overflowed = stream.read(chunk_samples)
                chunk = chunk.flatten()
                audio_chunks.append(chunk)
                total_duration += chunk_duration

                # Check for speech
                has_speech = self.is_speech(chunk)

                if has_speech:
                    speech_counter += 1
                    silence_counter = 0
                    if speech_counter >= min_speech_chunks:
                        if not speech_started:
                            print("[VAD] Speech detected")
                        speech_started = True
                else:
                    if speech_started:
                        silence_counter += 1
                        if silence_counter >= silence_chunks_needed:
                            print(f"[VAD] Silence detected after {total_duration:.1f}s")
                            break

            stream.stop()
            stream.close()

            # Skip if no speech was detected (saves API calls)
            if not speech_started:
                print("[VAD] No speech detected, skipping transcription")
                return b'', 0.0

            # Combine chunks
            if audio_chunks:
                audio_np = np.concatenate(audio_chunks)
                audio_bytes = audio_np.astype(np.int16).tobytes()
                return audio_bytes, total_duration

        except Exception as e:
            print(f"[VAD] Recording error: {e}")
            import traceback
            traceback.print_exc()

        return b'', 0.0


class FastSTT:
    """Combined fast STT with VAD support."""

    def __init__(self, use_groq: bool = True, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.vad = VADRecorder(sample_rate)

        # Try Groq first, fall back to local whisper
        if use_groq and os.getenv("GROQ_API_KEY"):
            self.stt = GroqWhisperSTT()
            self.stt_type = "groq"
            print("[STT] Using Groq Whisper API")
        else:
            # Fall back to local whisper
            from .stt import LocalWhisperSTT
            self.stt = LocalWhisperSTT("base")
            self.stt_type = "local"
            print("[STT] Using local Whisper")

    async def listen_and_transcribe(
        self,
        max_duration: float = 8.0,
        silence_duration: float = 0.8
    ) -> Optional[str]:
        """Listen with VAD and transcribe."""
        # Record with VAD
        audio_bytes, duration = await self.vad.record_until_silence(
            max_duration=max_duration,
            silence_duration=silence_duration
        )

        if not audio_bytes or duration < 0.3:
            return None

        print(f"[STT] Transcribing {duration:.1f}s of audio...")

        # Transcribe
        text = await self.stt.transcribe(audio_bytes, self.sample_rate)
        return text

    def is_ready(self) -> bool:
        return self.stt.is_ready() if hasattr(self.stt, 'is_ready') else True
