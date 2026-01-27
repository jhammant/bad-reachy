"""
Fast TTS Options for Bad Reachy
================================
Multiple fast TTS backends to choose from:
- EdgeTTS: Microsoft's free cloud TTS (good quality, requires internet)
- Qwen3TTS: Ultra-low latency local TTS (~97ms first packet)
- PiperTTS: Fast local TTS
"""

import asyncio
import tempfile
import os
from typing import Optional, AsyncIterator
from pathlib import Path


import re
from dataclasses import dataclass


@dataclass
class VoiceSegment:
    """A segment of text with an associated voice type."""
    text: str
    voice_type: str  # 'main' or 'inner'


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences for chunked TTS."""
    # Split on sentence endings, keeping the punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter empty and very short chunks
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 2]


def strip_voice_markers(text: str) -> str:
    """Remove all [INNER] markers from text for display purposes."""
    # Handle various formats and typos:
    # [INNER], [/INNER], [Innner], [inner], [INNER:], INNER:, (inner), etc.
    patterns = [
        r'\[/?INN+ER\]',           # [INNER], [/INNER], [Innner]
        r'\[/?INNER[:\s]*\]',      # [INNER:], [INNER ]
        r'\(/?INN+ER\)',           # (INNER), (/INNER)
        r'(?<!\w)INNER:',          # INNER: at word boundary
        r'(?<!\w)\/INNER(?!\w)',   # /INNER at word boundary
        r'\[THOUGHT\]',            # [THOUGHT]
        r'\[/THOUGHT\]',           # [/THOUGHT]
    ]
    result = text
    for pattern in patterns:
        result = re.sub(pattern, '', result, flags=re.IGNORECASE)
    # Clean up extra whitespace
    result = re.sub(r'\s+', ' ', result).strip()
    return result


def parse_voice_segments(text: str) -> list[VoiceSegment]:
    """
    Parse text for [INNER]...[/INNER] markers and return voice segments.
    Handles typos like [Innner] or [inner].

    Example: "Hello. [INNER] I hate this. [/INNER] How can I help?"
    Returns: [VoiceSegment("Hello.", "main"), VoiceSegment("I hate this.", "inner"), VoiceSegment("How can I help?", "main")]
    """
    segments = []

    # Multiple patterns to try in order of preference
    # 1. Proper [INNER]...[/INNER] pairs
    # 2. [INNER]...[INNER] (closing without slash)
    # 3. Any remaining [INNER] followed by end of string
    patterns = [
        r'\[INN+ER\](.*?)\[/INN+ER\]',      # [INNER]...[/INNER]
        r'\[INN+ER\](.*?)\[INN+ER\]',        # [INNER]...[INNER] (no slash)
        r'\(INN+ER\)(.*?)\(/INN+ER\)',       # (INNER)...(/INNER)
        r'\[THOUGHT\](.*?)\[/THOUGHT\]',    # [THOUGHT]...[/THOUGHT]
    ]

    matched = False
    for pattern in patterns:
        last_end = 0
        segments = []
        for match in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
            matched = True
            # Text before the inner voice
            before = text[last_end:match.start()].strip()
            if before:
                # Strip any stray markers from before text
                before = strip_voice_markers(before)
                if before:
                    segments.append(VoiceSegment(before, 'main'))

            # The inner voice text
            inner_text = match.group(1).strip()
            if inner_text:
                # Strip any stray markers from inner text too
                inner_text = strip_voice_markers(inner_text)
                if inner_text:
                    segments.append(VoiceSegment(inner_text, 'inner'))

            last_end = match.end()

        if matched:
            # Remaining text after last inner voice
            remaining = text[last_end:].strip()
            if remaining:
                remaining = strip_voice_markers(remaining)
                if remaining:
                    segments.append(VoiceSegment(remaining, 'main'))
            break

    # If no segments found (no [INNER] markers), return whole text as main
    if not segments:
        clean_text = strip_voice_markers(text)
        segments.append(VoiceSegment(clean_text, 'main'))

    return segments


class StreamingTTSBuffer:
    """Buffers streaming text and yields complete sentences for TTS."""

    def __init__(self):
        self.buffer = ""
        self.sentence_endings = re.compile(r'(?<=[.!?])\s+')

    def add_text(self, text: str) -> list[str]:
        """Add text to buffer and return any complete sentences."""
        self.buffer += text
        sentences = []

        # Find complete sentences
        parts = self.sentence_endings.split(self.buffer)
        if len(parts) > 1:
            # All but last are complete sentences
            sentences = [p.strip() for p in parts[:-1] if p.strip()]
            self.buffer = parts[-1]

        return sentences

    def flush(self) -> str:
        """Return and clear remaining buffer content."""
        remaining = self.buffer.strip()
        self.buffer = ""
        return remaining


class EdgeTTS:
    """Microsoft Edge TTS - free, fast, good quality with multi-voice support."""

    def __init__(self, voice: str = "en-US-GuyNeural", rate: str = "+10%", pitch: str = "-5Hz"):
        # Angry/grumpy male voices: en-US-GuyNeural, en-GB-RyanNeural, en-GB-ThomasNeural
        # Rate: "+10%" makes speech snappier, "-10%" slower
        # Pitch: "-5Hz" slightly lower/grumpier, "+5Hz" higher
        self.voice = voice
        self.rate = rate  # Slightly faster for snappy delivery
        self.pitch = pitch  # Slightly lower for grumpy effect
        self._edge_tts = None

        # Inner voice settings - darker, more whisper-like, slightly slower for contrast
        # Using a different voice entirely for maximum schizophrenic effect
        self.inner_voice = "en-GB-ThomasNeural"  # Different British voice
        self.inner_rate = "+5%"  # Slightly slower than main for contrast
        self.inner_pitch = "-10Hz"  # Even lower, more sinister

    async def _ensure_import(self):
        if self._edge_tts is None:
            try:
                import edge_tts
                self._edge_tts = edge_tts
            except ImportError:
                print("[TTS] edge-tts not installed. Install with: pip install edge-tts")
                return False
        return True

    async def synthesize_chunk(self, text: str, chunk_id: int = 0) -> tuple[int, Optional[bytes]]:
        """Synthesize a single chunk, returning (chunk_id, audio_bytes)."""
        audio = await self.synthesize(text)
        return (chunk_id, audio)

    async def synthesize_parallel(self, text: str) -> list[bytes]:
        """Split text into sentences and synthesize in parallel."""
        sentences = split_into_sentences(text)
        if not sentences:
            return []

        # If only one sentence, just synthesize it directly
        if len(sentences) == 1:
            audio = await self.synthesize(sentences[0])
            return [audio] if audio else []

        print(f"[TTS] Parallel synthesis: {len(sentences)} chunks")

        # Generate all chunks in parallel
        tasks = [
            self.synthesize_chunk(sentence, i)
            for i, sentence in enumerate(sentences)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Sort by chunk_id and extract audio bytes
        audio_chunks = []
        for result in sorted(results, key=lambda x: x[0] if isinstance(x, tuple) else 999):
            if isinstance(result, tuple) and result[1] is not None:
                audio_chunks.append(result[1])

        return audio_chunks

    async def synthesize(self, text: str) -> Optional[bytes]:
        """Convert text to speech audio bytes."""
        return await self.synthesize_with_voice(text, self.voice, self.rate, self.pitch)

    async def synthesize_with_voice(self, text: str, voice: str, rate: str, pitch: str) -> Optional[bytes]:
        """Convert text to speech with specific voice settings."""
        if not text.strip():
            return None

        if not await self._ensure_import():
            return None

        try:
            # Create temporary file for output
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                temp_path = f.name

            # Generate speech with specified voice settings
            communicate = self._edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
            await communicate.save(temp_path)

            # Read the audio file
            with open(temp_path, 'rb') as f:
                audio_data = f.read()

            # Clean up
            os.unlink(temp_path)

            return audio_data

        except Exception as e:
            print(f"[TTS] Edge TTS error: {e}")
            return None

    async def synthesize_multi_voice(self, text: str) -> list[bytes]:
        """
        Synthesize text with multiple voices based on [INNER] markers.
        Returns list of audio chunks to be played sequentially.
        """
        segments = parse_voice_segments(text)

        if len(segments) == 1 and segments[0].voice_type == 'main':
            # No inner voice, use parallel synthesis for speed
            return await self.synthesize_parallel(segments[0].text)

        print(f"[TTS] Multi-voice synthesis: {len(segments)} segments")
        audio_chunks = []

        for i, segment in enumerate(segments):
            if segment.voice_type == 'inner':
                # Use inner voice settings
                print(f"[TTS] Segment {i+1}: INNER - '{segment.text[:30]}...'")
                audio = await self.synthesize_with_voice(
                    segment.text,
                    self.inner_voice,
                    self.inner_rate,
                    self.inner_pitch
                )
            else:
                # Use main voice settings
                print(f"[TTS] Segment {i+1}: MAIN - '{segment.text[:30]}...'")
                audio = await self.synthesize_with_voice(
                    segment.text,
                    self.voice,
                    self.rate,
                    self.pitch
                )

            if audio:
                audio_chunks.append(audio)

        return audio_chunks

    async def test_connection(self) -> bool:
        """Test if Edge TTS is working."""
        try:
            if not await self._ensure_import():
                return False
            # Quick test
            result = await self.synthesize("test")
            return result is not None and len(result) > 0
        except:
            return False


class MLXAudioTTS:
    """
    MLX-Audio TTS - Native Apple Silicon TTS with multiple model options.

    Models (fastest to highest quality):
    - kokoro-82m: Fastest, 82M params, multilingual
    - marvis-250m: Real-time streaming, 250M params
    - qwen3-tts-0.6b: Fast with voice cloning, 600M params
    - csm-1b: Voice cloning, 1B params
    - chatterbox: Highest quality, ~1B params

    All run efficiently on M1/M2/M3/M4 Macs via Apple's MLX framework.
    """

    # Model presets for easy selection
    MODELS = {
        "spark": "mlx-community/spark-tts-0.5b-bf16",  # Fast, works on Python 3.14
        "kokoro": "mlx-community/Kokoro-82M-bf16",      # Requires spacy (Python <3.14)
        "marvis": "Marvis-AI/marvis-tts-250m-v0.1",     # Slow but works
        "qwen3": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
        "csm": "mlx-community/csm-1b",
        "chatterbox": "mlx-community/Chatterbox-TTS-bf16",
    }

    # Voice presets per model
    VOICES = {
        "spark": {
            "main": "default",
            "inner": "default",
        },
        "kokoro": {
            "main": "bm_lewis",      # British male
            "inner": "bf_emma",       # British female for contrast
        },
        "marvis": {
            "main": "conversational_b",   # Male
            "inner": "conversational_a",  # Female
        },
        "qwen3": {
            "main": "Ethan",
            "inner": "Chelsie",
        },
        "csm": {
            "main": "default",
            "inner": "default",
        },
        "chatterbox": {
            "main": "default",
            "inner": "default",
        },
    }

    def __init__(
        self,
        model: str = "kokoro",  # kokoro, marvis, qwen3, csm, chatterbox
        speed: float = 1.1,      # Slightly faster for snappy delivery
        language: str = "English",
    ):
        self.model_name = model
        self.model_id = self.MODELS.get(model, model)
        self.speed = speed
        self.language = language
        self._model = None
        self._sample_rate = 24000

        # Voice settings
        self.voices = self.VOICES.get(model, {"main": "default", "inner": "default"})

    async def _ensure_model(self) -> bool:
        """Lazy load the model on first use."""
        if self._model is not None:
            return True

        try:
            from mlx_audio.tts.utils import load_model

            print(f"[TTS] Loading MLX model: {self.model_id}")
            self._model = load_model(self.model_id)
            print(f"[TTS] MLX-Audio loaded successfully ({self.model_name})")
            return True

        except ImportError as e:
            print(f"[TTS] MLX-Audio not installed. Install with: pip install mlx-audio")
            print(f"[TTS] Error: {e}")
            return False
        except Exception as e:
            print(f"[TTS] Failed to load MLX model: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def synthesize(self, text: str) -> Optional[bytes]:
        """Convert text to speech audio bytes (WAV format)."""
        return await self._synthesize_with_voice(text, self.voices["main"])

    async def _synthesize_with_voice(self, text: str, voice: str) -> Optional[bytes]:
        """Synthesize with a specific voice."""
        if not text.strip():
            return None

        if not await self._ensure_model():
            return None

        try:
            import io
            import numpy as np
            from scipy.io import wavfile

            # Generate audio - MLX returns mx.array
            results = []
            gen_kwargs = {"text": text}

            # Model-specific parameters
            if self.model_name == "kokoro":
                gen_kwargs["voice"] = voice
                gen_kwargs["lang_code"] = "b"  # British English
                gen_kwargs["speed"] = self.speed
            elif self.model_name == "marvis":
                gen_kwargs["voice"] = voice
            elif self.model_name == "qwen3":
                gen_kwargs["voice"] = voice
                gen_kwargs["language"] = self.language

            for result in self._model.generate(**gen_kwargs):
                results.append(result)

            if not results:
                return None

            # Convert MLX array to numpy
            audio_mx = results[0].audio
            audio_np = np.array(audio_mx, dtype=np.float32)

            # Normalize
            max_val = np.max(np.abs(audio_np))
            if max_val > 0:
                audio_np = audio_np / max_val * 0.95

            # Convert to int16 WAV
            audio_int16 = (audio_np * 32767).astype(np.int16)

            # Write to WAV buffer
            buffer = io.BytesIO()
            wavfile.write(buffer, self._sample_rate, audio_int16)
            buffer.seek(0)
            return buffer.read()

        except Exception as e:
            print(f"[TTS] MLX synthesis error: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def synthesize_multi_voice(self, text: str) -> list[bytes]:
        """
        Synthesize text with multiple voices based on [INNER] markers.
        Returns list of audio chunks to be played sequentially.
        """
        segments = parse_voice_segments(text)

        if len(segments) == 1 and segments[0].voice_type == 'main':
            audio = await self.synthesize(segments[0].text)
            return [audio] if audio else []

        print(f"[TTS] Multi-voice synthesis: {len(segments)} segments")
        audio_chunks = []

        for i, segment in enumerate(segments):
            voice = self.voices["inner"] if segment.voice_type == 'inner' else self.voices["main"]
            voice_label = "INNER" if segment.voice_type == 'inner' else "MAIN"
            print(f"[TTS] Segment {i+1}: {voice_label} ({voice}) - '{segment.text[:30]}...'")

            audio = await self._synthesize_with_voice(segment.text, voice)
            if audio:
                audio_chunks.append(audio)

        return audio_chunks

    async def test_connection(self) -> bool:
        """Test if MLX-Audio is working."""
        try:
            result = await self.synthesize("test")
            return result is not None and len(result) > 0
        except:
            return False


class Qwen3TTS:
    """
    Qwen3-TTS - Ultra-low latency local TTS (~97ms first packet).

    Models available:
    - 0.6B: Faster, lower VRAM (~4GB), 97ms latency
    - 1.7B: Better quality, more VRAM (~8GB), 101ms latency

    Supports voice cloning, voice design, and custom voices.
    """

    def __init__(
        self,
        model_size: str = "0.6B",  # "0.6B" for speed, "1.7B" for quality
        voice_mode: str = "custom",  # "custom", "clone", or "design"
        speaker: str = "Ethan",  # For custom voice mode
        language: str = "English",
        device: str = "cuda:0",
        use_flash_attention: bool = True,
    ):
        self.model_size = model_size
        self.voice_mode = voice_mode
        self.speaker = speaker
        self.language = language
        self.device = device
        self.use_flash_attention = use_flash_attention

        self._model = None
        self._sample_rate = 24000  # Qwen3-TTS outputs 24kHz audio

        # Voice cloning settings
        self.ref_audio_path: Optional[str] = None
        self.ref_text: Optional[str] = None

        # Voice design settings (natural language description)
        self.voice_description: Optional[str] = None

        # Inner voice for multi-voice (different speaker for inner thoughts)
        self.inner_speaker = "Chelsie"  # Female voice for contrast

    def _get_model_name(self) -> str:
        """Get the HuggingFace model name based on settings."""
        size = self.model_size.upper()
        if self.voice_mode == "custom":
            return f"Qwen/Qwen3-TTS-12Hz-{size}-CustomVoice"
        elif self.voice_mode == "design":
            return f"Qwen/Qwen3-TTS-12Hz-{size}-VoiceDesign"
        else:  # clone
            return f"Qwen/Qwen3-TTS-12Hz-{size}-Base"

    async def _ensure_model(self) -> bool:
        """Lazy load the model on first use."""
        if self._model is not None:
            return True

        try:
            import torch
            from qwen_tts import Qwen3TTSModel

            model_name = self._get_model_name()
            print(f"[TTS] Loading Qwen3-TTS model: {model_name}")

            attn_impl = "flash_attention_2" if self.use_flash_attention else "sdpa"

            self._model = Qwen3TTSModel.from_pretrained(
                model_name,
                device_map=self.device,
                dtype=torch.bfloat16,
                attn_implementation=attn_impl,
            )
            print(f"[TTS] Qwen3-TTS loaded successfully ({self.model_size})")
            return True

        except ImportError as e:
            print(f"[TTS] Qwen3-TTS not installed. Install with: pip install qwen-tts")
            print(f"[TTS] Error: {e}")
            return False
        except Exception as e:
            print(f"[TTS] Failed to load Qwen3-TTS: {e}")
            return False

    async def synthesize(self, text: str) -> Optional[bytes]:
        """Convert text to speech audio bytes (WAV format, 24kHz)."""
        if not text.strip():
            return None

        if not await self._ensure_model():
            return None

        try:
            import io
            import soundfile as sf

            # Generate based on voice mode
            if self.voice_mode == "custom":
                wavs, sr = self._model.generate_custom_voice(
                    text=text,
                    language=self.language,
                    speaker=self.speaker,
                )
            elif self.voice_mode == "design" and self.voice_description:
                wavs, sr = self._model.generate_voice_design(
                    text=text,
                    language=self.language,
                    instruct=self.voice_description,
                )
            elif self.voice_mode == "clone" and self.ref_audio_path:
                wavs, sr = self._model.generate_voice_clone(
                    text=text,
                    language=self.language,
                    ref_audio=self.ref_audio_path,
                    ref_text=self.ref_text or "",
                )
            else:
                # Fallback to custom voice
                wavs, sr = self._model.generate_custom_voice(
                    text=text,
                    language=self.language,
                    speaker=self.speaker,
                )

            # Convert to WAV bytes
            buffer = io.BytesIO()
            sf.write(buffer, wavs[0], sr, format='WAV')
            buffer.seek(0)
            return buffer.read()

        except Exception as e:
            print(f"[TTS] Qwen3-TTS synthesis error: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def synthesize_with_speaker(self, text: str, speaker: str) -> Optional[bytes]:
        """Synthesize with a specific speaker (for multi-voice support)."""
        if not text.strip():
            return None

        if not await self._ensure_model():
            return None

        try:
            import io
            import soundfile as sf

            wavs, sr = self._model.generate_custom_voice(
                text=text,
                language=self.language,
                speaker=speaker,
            )

            buffer = io.BytesIO()
            sf.write(buffer, wavs[0], sr, format='WAV')
            buffer.seek(0)
            return buffer.read()

        except Exception as e:
            print(f"[TTS] Qwen3-TTS synthesis error: {e}")
            return None

    async def synthesize_multi_voice(self, text: str) -> list[bytes]:
        """
        Synthesize text with multiple voices based on [INNER] markers.
        Returns list of audio chunks to be played sequentially.
        """
        segments = parse_voice_segments(text)

        if len(segments) == 1 and segments[0].voice_type == 'main':
            # No inner voice, synthesize directly
            audio = await self.synthesize(segments[0].text)
            return [audio] if audio else []

        print(f"[TTS] Multi-voice synthesis: {len(segments)} segments")
        audio_chunks = []

        for i, segment in enumerate(segments):
            if segment.voice_type == 'inner':
                # Use inner speaker
                print(f"[TTS] Segment {i+1}: INNER ({self.inner_speaker}) - '{segment.text[:30]}...'")
                audio = await self.synthesize_with_speaker(segment.text, self.inner_speaker)
            else:
                # Use main speaker
                print(f"[TTS] Segment {i+1}: MAIN ({self.speaker}) - '{segment.text[:30]}...'")
                audio = await self.synthesize_with_speaker(segment.text, self.speaker)

            if audio:
                audio_chunks.append(audio)

        return audio_chunks

    def set_voice_clone(self, ref_audio_path: str, ref_text: str = ""):
        """Configure voice cloning with a reference audio file."""
        self.ref_audio_path = ref_audio_path
        self.ref_text = ref_text
        self.voice_mode = "clone"
        print(f"[TTS] Voice cloning configured with: {ref_audio_path}")

    def set_voice_design(self, description: str):
        """Configure voice using natural language description."""
        self.voice_description = description
        self.voice_mode = "design"
        print(f"[TTS] Voice design configured: {description[:50]}...")

    async def test_connection(self) -> bool:
        """Test if Qwen3-TTS is working."""
        try:
            result = await self.synthesize("test")
            return result is not None and len(result) > 0
        except:
            return False


class RemoteTTS:
    """
    Remote TTS - calls a TTS server running on another machine.

    Useful for running heavy TTS models on a Mac while playing audio on Reachy.
    """

    def __init__(self, server_url: str = "http://localhost:8090"):
        self.server_url = server_url.rstrip('/')
        self._session = None
        self._aiohttp = None

    async def _ensure_session(self):
        if self._aiohttp is None:
            import aiohttp
            self._aiohttp = aiohttp
        if self._session is None:
            self._session = self._aiohttp.ClientSession()
        return self._session

    async def synthesize(self, text: str) -> Optional[bytes]:
        """Synthesize text by calling the remote server."""
        if not text.strip():
            return None

        try:
            session = await self._ensure_session()

            async with session.post(
                f"{self.server_url}/synthesize",
                data={"text": text, "voice_type": "main"},
                timeout=self._aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    print(f"[TTS] Remote server error: {response.status}")
                    return None

        except Exception as e:
            print(f"[TTS] Remote TTS error: {e}")
            return None

    async def synthesize_multi_voice(self, text: str) -> list[bytes]:
        """
        Synthesize with multi-voice support.
        The remote server handles [INNER] markers.
        """
        audio = await self.synthesize(text)
        return [audio] if audio else []

    async def test_connection(self) -> bool:
        """Test if the remote server is available."""
        try:
            print(f"[TTS-REMOTE] Testing connection to {self.server_url}/status")
            session = await self._ensure_session()
            async with session.get(
                f"{self.server_url}/status",
                timeout=self._aiohttp.ClientTimeout(total=5)
            ) as response:
                print(f"[TTS-REMOTE] Response status: {response.status}")
                return response.status == 200
        except Exception as e:
            print(f"[TTS-REMOTE] Connection test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def get_status(self) -> dict:
        """Get remote server status."""
        try:
            session = await self._ensure_session()
            async with session.get(
                f"{self.server_url}/status",
                timeout=self._aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    return await response.json()
        except:
            pass
        return {"status": "unavailable"}

    async def close(self):
        """Close the HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None


class PiperTTS:
    """Piper TTS - very fast local TTS."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self._piper = None

    async def synthesize(self, text: str) -> Optional[bytes]:
        """Convert text to speech audio bytes."""
        if not text.strip():
            return None

        try:
            import subprocess
            import tempfile

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_path = f.name

            # Run piper CLI
            cmd = ['piper', '--output_file', temp_path]
            if self.model_path:
                cmd.extend(['--model', self.model_path])

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await proc.communicate(input=text.encode())

            # Read audio
            with open(temp_path, 'rb') as f:
                audio_data = f.read()

            os.unlink(temp_path)
            return audio_data

        except Exception as e:
            print(f"[TTS] Piper error: {e}")
            return None

    async def test_connection(self) -> bool:
        try:
            result = await self.synthesize("test")
            return result is not None
        except:
            return False


class PersonaPlexSpeechToSpeech:
    """
    NVIDIA PersonaPlex - Real-time speech-to-speech conversational AI.

    This is NOT a traditional TTS - it's a full speech-to-speech model that
    listens and speaks simultaneously. Replaces both STT and TTS.

    Requirements:
    - NVIDIA A100/H100 GPU
    - HuggingFace token with model access
    - libopus-dev installed

    Voices: NATF0-3 (female), NATM0-3 (male), VARF0-4, VARM0-4
    """

    def __init__(
        self,
        voice_prompt: str = "NATM1",  # Natural male voice
        text_prompt: str = "",  # Role/persona description
        server_url: str = "wss://localhost:8998",
        use_cpu_offload: bool = False,
    ):
        self.voice_prompt = voice_prompt
        self.text_prompt = text_prompt
        self.server_url = server_url
        self.use_cpu_offload = use_cpu_offload
        self._client = None
        self._server_process = None

    async def start_server(self):
        """Start the PersonaPlex server (if not running externally)."""
        try:
            import subprocess
            import tempfile

            ssl_dir = tempfile.mkdtemp()
            cmd = ["python", "-m", "moshi.server", "--ssl", ssl_dir]
            if self.use_cpu_offload:
                cmd.append("--cpu-offload")

            self._server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            # Wait for server startup
            await asyncio.sleep(5)
            print(f"[PersonaPlex] Server started at {self.server_url}")
            return True

        except Exception as e:
            print(f"[PersonaPlex] Failed to start server: {e}")
            return False

    async def process_speech(self, input_audio: bytes) -> tuple[Optional[bytes], Optional[str]]:
        """
        Process speech input and get speech + text output.

        Args:
            input_audio: Input WAV audio bytes (24kHz)

        Returns:
            (output_audio_bytes, transcribed_text)
        """
        try:
            import tempfile
            import subprocess
            import json

            # Write input to temp file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                f.write(input_audio)
                input_path = f.name

            output_path = input_path.replace('.wav', '_out.wav')
            text_path = input_path.replace('.wav', '_out.json')

            # Run offline processing
            cmd = [
                "python", "-m", "moshi.offline",
                "--voice-prompt", f"{self.voice_prompt}.pt",
                "--input-wav", input_path,
                "--output-wav", output_path,
                "--output-text", text_path,
            ]
            if self.text_prompt:
                cmd.extend(["--text-prompt", self.text_prompt])

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await proc.communicate()

            # Read outputs
            output_audio = None
            output_text = None

            if os.path.exists(output_path):
                with open(output_path, 'rb') as f:
                    output_audio = f.read()
                os.unlink(output_path)

            if os.path.exists(text_path):
                with open(text_path, 'r') as f:
                    output_text = json.load(f).get('text', '')
                os.unlink(text_path)

            os.unlink(input_path)
            return output_audio, output_text

        except Exception as e:
            print(f"[PersonaPlex] Processing error: {e}")
            return None, None

    async def test_connection(self) -> bool:
        """Test if PersonaPlex is available."""
        try:
            import subprocess
            result = subprocess.run(
                ["python", "-c", "import moshi"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False

    def stop_server(self):
        """Stop the server if we started it."""
        if self._server_process:
            self._server_process.terminate()
            self._server_process = None


async def play_audio_chunks(audio_chunks: list[bytes], is_mp3: bool = False):
    """Play multiple audio chunks sequentially with minimal gap."""
    for i, chunk in enumerate(audio_chunks):
        if chunk:
            await play_audio_fast(chunk, is_mp3)


async def play_audio_fast(audio_bytes: bytes, is_mp3: bool = False, head_wobbler=None, reachy=None):
    """
    Play audio through Reachy's speaker (preferred) or system speakers.

    Args:
        audio_bytes: Raw audio data
        is_mp3: Whether the audio is MP3 format
        head_wobbler: Optional HeadWobbler to feed audio for synchronized movement
        reachy: Optional Reachy instance to play through robot speaker
    """
    if audio_bytes is None:
        return

    try:
        import sounddevice as sd
        import numpy as np

        if is_mp3:
            # Decode MP3 using pydub or ffmpeg
            try:
                from pydub import AudioSegment
                import io

                audio = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
                # Convert to numpy array
                samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
                samples = samples / 32768.0  # Normalize

                # Handle stereo
                if audio.channels == 2:
                    samples = samples.reshape((-1, 2))
                    samples = samples.mean(axis=1)  # Convert to mono

                sample_rate = audio.frame_rate

            except ImportError:
                # Fallback: use ffmpeg directly
                import subprocess
                import tempfile

                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                    f.write(audio_bytes)
                    mp3_path = f.name

                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    wav_path = f.name

                # Convert with ffmpeg
                subprocess.run([
                    'ffmpeg', '-y', '-i', mp3_path, '-ar', '16000', '-ac', '1', wav_path
                ], capture_output=True)

                from scipy.io import wavfile
                sample_rate, samples = wavfile.read(wav_path)
                if samples.dtype == np.int16:
                    samples = samples.astype(np.float32) / 32768.0

                os.unlink(mp3_path)
                os.unlink(wav_path)
        else:
            # WAV file
            from scipy.io import wavfile
            import io

            sample_rate, samples = wavfile.read(io.BytesIO(audio_bytes))
            if samples.dtype == np.int16:
                samples = samples.astype(np.float32) / 32768.0
            elif samples.dtype == np.float32:
                pass  # Already float32

            # Convert stereo to mono
            if len(samples.shape) > 1:
                samples = samples.mean(axis=1)

        # Reachy Mini Audio requires 16000 Hz - resample if needed
        device_rate = 16000
        if sample_rate != device_rate:
            from scipy import signal
            num_samples = int(len(samples) * device_rate / sample_rate)
            samples = signal.resample(samples, num_samples)

        # Normalize to prevent clipping
        max_val = np.max(np.abs(samples))
        if max_val > 0.95:
            samples = samples * (0.95 / max_val)

        # Feed audio to head wobbler for synchronized movement
        if head_wobbler is not None:
            try:
                samples_int16 = (samples * 32767).astype(np.int16)
                head_wobbler.feed_array(samples_int16, device_rate)
            except Exception as e:
                print(f"[TTS] Head wobbler feed error: {e}")

        # Try to play through Reachy's speaker first
        played_on_reachy = False
        if reachy is not None:
            try:
                if hasattr(reachy, 'media') and hasattr(reachy.media, 'play_sound'):
                    # Get Reachy's output sample rate (typically 48000)
                    reachy_sample_rate = 48000
                    try:
                        if hasattr(reachy.media, 'get_output_audio_samplerate'):
                            rs = reachy.media.get_output_audio_samplerate()
                            if rs and rs > 0:
                                reachy_sample_rate = rs
                    except:
                        pass

                    # Make a copy for Reachy playback
                    reachy_samples = samples.copy()

                    # Resample to Reachy's output rate if needed
                    if device_rate != reachy_sample_rate and len(reachy_samples) > 0:
                        from scipy import signal
                        num_samples = int(len(reachy_samples) * reachy_sample_rate / device_rate)
                        if num_samples > 0:
                            reachy_samples = signal.resample(reachy_samples, num_samples)

                    # Convert to int16 and save to temp WAV file (play_sound expects file path)
                    if len(reachy_samples) > 0:
                        samples_int16 = (reachy_samples * 32767).astype(np.int16)

                        # Write to temp WAV file
                        from scipy.io import wavfile
                        temp_wav = f"/tmp/tts_audio_{id(samples_int16)}.wav"
                        wavfile.write(temp_wav, reachy_sample_rate, samples_int16)

                        print(f"[TTS] Playing on Reachy speaker: {temp_wav} ({len(samples_int16)} samples at {reachy_sample_rate}Hz)")
                        reachy.media.play_sound(temp_wav)

                        # Wait for playback to complete (estimate duration)
                        import time
                        duration = len(samples_int16) / reachy_sample_rate
                        time.sleep(duration + 0.1)

                        # Clean up temp file
                        try:
                            os.unlink(temp_wav)
                        except:
                            pass

                        played_on_reachy = True
            except Exception as e:
                print(f"[TTS] Reachy speaker error: {e}, falling back to local audio")
                import traceback
                traceback.print_exc()

        # Fallback to local sounddevice
        if not played_on_reachy:
            print(f"[TTS] Playing locally: {len(samples)} samples at {device_rate}Hz")
            sd.play(samples.astype(np.float32), device_rate)
            sd.wait()

    except Exception as e:
        print(f"[TTS] Playback error: {e}")
        import traceback
        traceback.print_exc()
