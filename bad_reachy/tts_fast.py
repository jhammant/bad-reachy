"""
Fast TTS Options for Bad Reachy
================================
Multiple fast TTS backends to choose from.
"""

import asyncio
import tempfile
import os
from typing import Optional
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
    # Handle various typos/formats: [INNER], [Innner], [inner], etc.
    pattern = r'\[/?INN*ER\]'
    return re.sub(pattern, '', text, flags=re.IGNORECASE).strip()


def parse_voice_segments(text: str) -> list[VoiceSegment]:
    """
    Parse text for [INNER]...[/INNER] markers and return voice segments.
    Handles typos like [Innner] or [inner].

    Example: "Hello. [INNER] I hate this. [/INNER] How can I help?"
    Returns: [VoiceSegment("Hello.", "main"), VoiceSegment("I hate this.", "inner"), VoiceSegment("How can I help?", "main")]
    """
    segments = []

    # Pattern to match [INNER]...[/INNER] blocks - flexible for typos (INN*ER)
    pattern = r'\[INN*ER\](.*?)\[/INN*ER\]'

    last_end = 0
    for match in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
        # Text before the inner voice
        before = text[last_end:match.start()].strip()
        if before:
            segments.append(VoiceSegment(before, 'main'))

        # The inner voice text
        inner_text = match.group(1).strip()
        if inner_text:
            segments.append(VoiceSegment(inner_text, 'inner'))

        last_end = match.end()

    # Remaining text after last inner voice
    remaining = text[last_end:].strip()
    if remaining:
        segments.append(VoiceSegment(remaining, 'main'))

    # If no segments found (no [INNER] markers), return whole text as main
    if not segments:
        segments.append(VoiceSegment(text.strip(), 'main'))

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


async def play_audio_chunks(audio_chunks: list[bytes], is_mp3: bool = False):
    """Play multiple audio chunks sequentially with minimal gap."""
    for i, chunk in enumerate(audio_chunks):
        if chunk:
            await play_audio_fast(chunk, is_mp3)


async def play_audio_fast(audio_bytes: bytes, is_mp3: bool = False, head_wobbler=None):
    """
    Play audio through system speakers - handles MP3 and WAV.

    Args:
        audio_bytes: Raw audio data
        is_mp3: Whether the audio is MP3 format
        head_wobbler: Optional HeadWobbler to feed audio for synchronized movement
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

        # Resample to 16000 Hz for Reachy Mini Audio
        device_rate = 16000
        if sample_rate != device_rate:
            from scipy import signal
            num_samples = int(len(samples) * device_rate / sample_rate)
            samples = signal.resample(samples, num_samples)

        # Normalize
        max_val = np.max(np.abs(samples))
        if max_val > 1.0:
            samples = samples / max_val

        # Feed audio to head wobbler for synchronized movement
        if head_wobbler is not None:
            try:
                # Convert to int16 for wobbler and feed in chunks
                samples_int16 = (samples * 32767).astype(np.int16)
                # Feed entire buffer - wobbler will process in chunks
                head_wobbler.feed_array(samples_int16, device_rate)
            except Exception as e:
                print(f"[TTS] Head wobbler feed error: {e}")

        print(f"[TTS] Playing {len(samples)} samples at {device_rate}Hz")
        sd.play(samples.astype(np.float32), device_rate)
        sd.wait()

    except Exception as e:
        print(f"[TTS] Playback error: {e}")
        import traceback
        traceback.print_exc()
