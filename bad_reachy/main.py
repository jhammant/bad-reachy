"""
Bad Reachy - Main Application
================================
A sarcastic, sweary robot assistant with attitude.
"""

import asyncio
import time
import threading
import os
import numpy as np
from enum import Enum
from typing import Optional
from dataclasses import dataclass, field
import cv2
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("[AUDIO] sounddevice not available - install with: pip install sounddevice")

from .config import Config
from .llm import LMStudioClient
from .stt import LocalWhisperSTT
from .tts import ChatterboxTTS, play_audio_on_reachy
from .emotions import EmotionEngine, Emotion
from .dashboard import BadDashboard
from .tools import ToolManager, TOOL_AWARE_PROMPT_ADDITION
from .comedy import ComedyEngine, COMEDY_SYSTEM_PROMPT

# Try to import audio-synchronized head movement
try:
    from .audio import HeadWobbler
    HEAD_WOBBLER_AVAILABLE = True
except ImportError:
    HEAD_WOBBLER_AVAILABLE = False
    HeadWobbler = None

# Try to import fast TTS/STT
try:
    from .tts_fast import EdgeTTS, play_audio_fast, strip_voice_markers, StreamingTTSBuffer
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    StreamingTTSBuffer = None

try:
    from .stt_fast import FastSTT
    FAST_STT_AVAILABLE = True
except ImportError:
    FAST_STT_AVAILABLE = False

try:
    from .llm_fast import GroqLLM
    GROQ_LLM_AVAILABLE = True
except ImportError:
    GROQ_LLM_AVAILABLE = False


class AppState(Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"


@dataclass
class BadContext:
    """Runtime context for the app."""
    state: AppState = AppState.IDLE
    last_speech_time: float = 0.0
    last_tts_end_time: float = 0.0  # When TTS finished - for echo suppression
    last_user_input: str = ""
    last_response: str = ""
    interactions: int = 0


class BadReachyApp:
    """Main application - a grumpy robot that reluctantly helps."""

    def __init__(self, reachy):
        self.reachy = reachy
        self.config = Config()
        self.context = BadContext()

        # Initialize components
        print("[BAD] Initializing... ugh, here we go again.")

        # Use Groq LLM if available (MUCH faster)
        import os
        if GROQ_LLM_AVAILABLE and os.getenv("GROQ_API_KEY"):
            self.llm = GroqLLM()
            print("[LLM] Using Groq (fast cloud)")
        else:
            self.llm = LMStudioClient(self.config)
            print("[LLM] Using LM Studio (local)")

        # Use comedy-enhanced prompt
        self.llm.system_prompt = COMEDY_SYSTEM_PROMPT + "\n" + TOOL_AWARE_PROMPT_ADDITION

        # Use fast STT if available (Groq + VAD)
        self.use_fast_stt = FAST_STT_AVAILABLE
        if self.use_fast_stt:
            self.fast_stt = FastSTT(use_groq=True, sample_rate=self.config.sample_rate)
            print("[STT] Fast STT with VAD enabled")
        else:
            self.stt = LocalWhisperSTT(self.config.whisper_model)
            print("[STT] Using local Whisper (slow)")

        # Use fast TTS if available (Edge TTS)
        self.use_fast_tts = EDGE_TTS_AVAILABLE
        if self.use_fast_tts:
            # UK male voice, faster rate (+15%) for snappy delivery, slightly lower pitch for grumpy tone
            self.fast_tts = EdgeTTS(voice="en-GB-RyanNeural", rate="+15%", pitch="-5Hz")
            print("[TTS] Edge TTS enabled (fast, UK voice, snappy delivery)")
        else:
            self.tts = ChatterboxTTS(self.config.tts_server_url)
            print("[TTS] Using Chatterbox (slow)")

        self.tools = ToolManager()

        self.emotions = EmotionEngine(reachy)
        # ComedyEngine needs a TTS reference - use fast or slow
        tts_for_comedy = self.fast_tts if self.use_fast_tts else self.tts
        self.comedy = ComedyEngine(tts_for_comedy)
        self.dashboard = BadDashboard(port=8080)

        # Audio-synchronized head movement
        self.head_wobbler = None
        if HEAD_WOBBLER_AVAILABLE and self.reachy:
            self.head_wobbler = HeadWobbler(
                set_offsets_callback=self._apply_head_offsets,
                sample_rate=16000
            )
            print("[HEAD] Audio-synchronized head movement enabled")
        else:
            print("[HEAD] Head wobbler not available (no reachy or module missing)")

        # Camera frame for dashboard
        self._latest_frame = None
        self._camera_thread = None
        self._running = False

        # Audio capture
        self._audio_buffer = []

        # Direct hardware access (fallback when Reachy SDK unavailable)
        self._direct_camera = None
        self._use_direct_audio = False

        # Only use direct camera if SDK is NOT connected
        # When SDK is connected, it holds the camera via daemon - don't compete for it
        if not self.reachy or not hasattr(self.reachy, 'media'):
            print("[BAD] Reachy SDK unavailable, trying direct hardware access...")
            self._init_direct_hardware()
        else:
            print("[BAD] SDK connected - using SDK camera (not opening direct camera)")

        # Base head position for wobble offsets
        self._base_head_position = None

    def _apply_head_offsets(self, offsets: tuple) -> None:
        """
        Apply head movement offsets synchronized with audio.
        Called by HeadWobbler from background thread.

        Args:
            offsets: (pitch_rad, yaw_rad, roll_rad) relative offsets
        """
        if not self.reachy or not hasattr(self.reachy, 'head'):
            return

        pitch_rad, yaw_rad, roll_rad = offsets

        try:
            # Get current (or cached base) position
            if self._base_head_position is None:
                # Cache the neutral position on first call
                head = self.reachy.head
                if hasattr(head, 'neck'):
                    self._base_head_position = {
                        'roll': getattr(head.neck.roll, 'goal_position', 0),
                        'pitch': getattr(head.neck.pitch, 'goal_position', 0),
                        'yaw': getattr(head.neck.yaw, 'goal_position', 0),
                    }
                else:
                    self._base_head_position = {'roll': 0, 'pitch': 0, 'yaw': 0}

            # Apply offsets to base position (convert radians to degrees)
            import math
            new_roll = self._base_head_position['roll'] + math.degrees(roll_rad)
            new_pitch = self._base_head_position['pitch'] + math.degrees(pitch_rad)
            new_yaw = self._base_head_position['yaw'] + math.degrees(yaw_rad)

            # Clamp to safe ranges
            new_roll = max(-15, min(15, new_roll))
            new_pitch = max(-30, min(30, new_pitch))
            new_yaw = max(-45, min(45, new_yaw))

            # Apply to robot
            head = self.reachy.head
            if hasattr(head, 'neck'):
                if hasattr(head.neck, 'roll'):
                    head.neck.roll.goal_position = new_roll
                if hasattr(head.neck, 'pitch'):
                    head.neck.pitch.goal_position = new_pitch
                if hasattr(head.neck, 'yaw'):
                    head.neck.yaw.goal_position = new_yaw

        except Exception as e:
            # Don't spam errors - head movement is best-effort
            pass

    def _reset_head_wobble(self) -> None:
        """Reset head wobble state and return to base position."""
        if self.head_wobbler:
            self.head_wobbler.reset()
        self._base_head_position = None

    def _init_direct_camera_only(self):
        """Initialize direct camera as backup when SDK camera fails."""
        try:
            self._direct_camera = cv2.VideoCapture(0)
            if self._direct_camera.isOpened():
                print("[CAMERA] Direct camera backup initialized (OpenCV)")
            else:
                print("[CAMERA] Could not open backup camera")
                self._direct_camera = None
        except Exception as e:
            print(f"[CAMERA] Direct camera backup failed: {e}")
            self._direct_camera = None

    def _init_direct_hardware(self):
        """Initialize direct camera and audio access."""
        # Try to open camera directly
        try:
            self._direct_camera = cv2.VideoCapture(0)
            if self._direct_camera.isOpened():
                print("[CAMERA] Direct camera access initialized (OpenCV)")
            else:
                print("[CAMERA] Could not open camera")
                self._direct_camera = None
        except Exception as e:
            print(f"[CAMERA] Direct camera init failed: {e}")
            self._direct_camera = None

        # Enable direct audio if sounddevice is available
        if SOUNDDEVICE_AVAILABLE:
            try:
                # List available audio devices
                devices = sd.query_devices()
                print("[AUDIO] Available audio devices:")
                self._audio_device = None
                for i, dev in enumerate(devices):
                    if dev['max_input_channels'] > 0:
                        print(f"  [{i}] {dev['name']} (in: {dev['max_input_channels']})")
                        # Prefer Reachy Mini Audio device
                        if 'reachy' in dev['name'].lower() or 'pollen' in dev['name'].lower():
                            self._audio_device = i
                            print(f"  -> Selected device {i} as input")

                # Set default device if found
                if self._audio_device is not None:
                    sd.default.device = (self._audio_device, None)
                    sd.default.samplerate = 16000  # Reachy Mini Audio only supports 16000
                    sd.default.channels = (2, None)  # Stereo input

                self._use_direct_audio = True
                print("[AUDIO] Direct audio capture enabled (sounddevice)")
            except Exception as e:
                print(f"[AUDIO] Direct audio init failed: {e}")
                self._use_direct_audio = False

    def _get_latest_frame(self):
        """Get the latest camera frame."""
        return self._latest_frame

    def _camera_capture_loop(self):
        """Background thread for camera capture."""
        frame_count = 0
        error_count = 0
        print(f"[CAMERA] Capture loop starting. reachy={self.reachy is not None}, direct_cam={self._direct_camera is not None}")
        if self._direct_camera:
            print(f"[CAMERA] Direct camera opened={self._direct_camera.isOpened()}")
        while self._running:
            try:
                frame = None
                # Try Reachy SDK first
                if self.reachy and hasattr(self.reachy, 'media') and hasattr(self.reachy.media, 'camera'):
                    frame = self.reachy.media.camera.read()
                    if frame_count == 0:
                        if frame is not None:
                            print(f"[CAMERA] SDK camera working: {frame.shape}")
                        else:
                            print("[CAMERA] SDK camera returned None")

                # Fallback to direct camera if SDK didn't work
                if frame is None and self._direct_camera and self._direct_camera.isOpened():
                    ret, frame = self._direct_camera.read()
                    if not ret:
                        frame = None

                if frame is not None:
                    self._latest_frame = frame
                    frame_count += 1
                    if frame_count == 1:
                        print(f"[CAMERA] First frame captured: {frame.shape}")

            except Exception as e:
                error_count += 1
                if error_count <= 5:
                    print(f"[CAMERA] Error #{error_count}: {e}")
                    import traceback
                    traceback.print_exc()
            time.sleep(0.1)

    async def _capture_audio(self, duration: float) -> bytes:
        """Capture audio from Reachy's microphone or direct sounddevice."""
        try:
            # Try Reachy SDK first
            if self.reachy and hasattr(self.reachy, 'media') and hasattr(self.reachy.media, 'microphone'):
                print(f"[AUDIO] Recording {duration}s from Reachy microphone...")
                samples = int(duration * self.config.sample_rate)
                audio_data = self.reachy.media.microphone.record(samples)

                if isinstance(audio_data, np.ndarray):
                    max_level = np.max(np.abs(audio_data))
                    print(f"[AUDIO] Captured {len(audio_data)} samples, max level: {max_level:.4f}")
                    if audio_data.dtype == np.float32:
                        audio_data = (audio_data * 32767).astype(np.int16)
                    return audio_data.tobytes()

            # Fallback to direct sounddevice capture
            elif self._use_direct_audio and SOUNDDEVICE_AVAILABLE:
                # Use 16000 Hz (Reachy Mini Audio only supports this)
                device_rate = 16000
                print(f"[AUDIO] Recording {duration}s via sounddevice at {device_rate}Hz...")
                # Record audio directly - stereo because device requires it
                audio_data = sd.rec(
                    int(duration * device_rate),
                    samplerate=device_rate,
                    channels=2,  # Reachy Mini Audio requires stereo
                    dtype='int16'
                )
                sd.wait()  # Wait for recording to complete

                # Convert stereo to mono by averaging channels
                if len(audio_data.shape) > 1 and audio_data.shape[1] == 2:
                    audio_mono = np.mean(audio_data, axis=1).astype(np.int16)
                else:
                    audio_mono = audio_data.flatten()

                # Check audio level
                max_level = np.max(np.abs(audio_mono.astype(np.float32) / 32767.0))
                print(f"[AUDIO] Captured {len(audio_mono)} samples, max level: {max_level:.4f}")

                return audio_mono.tobytes()

            # No audio available
            print("[AUDIO] No microphone available, returning silence")
            return b'\x00' * int(duration * self.config.sample_rate * 2)

        except Exception as e:
            print(f"[AUDIO] Capture error: {e}")
            import traceback
            traceback.print_exc()
            return b'\x00' * int(duration * self.config.sample_rate * 2)

    async def _listen(self) -> Optional[str]:
        """Listen for user speech."""
        # ECHO SUPPRESSION: Wait after TTS finishes to avoid hearing ourselves
        # Reduced from 3s to 1.5s for snappier response
        time_since_tts = time.time() - self.context.last_tts_end_time
        if time_since_tts < 1.5:  # 1.5s echo suppression window
            wait_time = 1.5 - time_since_tts
            print(f"[ECHO] Suppressing for {wait_time:.1f}s")
            await asyncio.sleep(wait_time)

        self.context.state = AppState.LISTENING
        self.dashboard.update_state("LISTENING")

        # Reset head wobbler when starting to listen (stops any ongoing speech movement)
        if self.head_wobbler:
            self.head_wobbler.reset()
            self._base_head_position = None  # Return head to neutral

        if self.use_fast_stt:
            # Fast path: VAD-based recording + Groq transcription
            text = await self.fast_stt.listen_and_transcribe(
                max_duration=3.0,   # Very short max - be snappy!
                silence_duration=0.25  # VERY aggressive silence detection
            )
        else:
            # Slow path: fixed duration recording + local whisper
            audio_data = await self._capture_audio(self.config.recording_duration)
            text = await self.stt.transcribe(audio_data, self.config.sample_rate)

        if text and len(text.strip()) > 2:
            text_lower = text.lower().strip()

            # Always ignore very short generic phrases (likely echo/noise)
            # Includes common TTS echoes and background noise transcriptions
            ignore_always = ['thank you', 'thanks', 'okay', 'ok', 'yes', 'no',
                            'cheers', 'right', 'sure', 'yeah', 'yep', 'nope',
                            'and', 'the', 'a', 'um', 'uh', 'hmm', 'oh', 'ah',
                            'whee', 'wee', 'bye', 'hi', 'hey', 'hello', 'huh',
                            'what', 'well', 'so', 'now', 'then', 'here', 'there',
                            'you', 'me', 'we', 'they', 'it', 'this', 'that',
                            'good', 'great', 'nice', 'fine', 'alright', 'all right']
            for phrase in ignore_always:
                if text_lower == phrase or text_lower == phrase + '.':
                    print(f"[STT] Ignoring generic: {text}")
                    # Update speech time to prevent idle behavior triggering
                    self.context.last_speech_time = time.time()
                    return None

            # Check for wake word - only respond if they say "reachy" or similar
            # DEMO_MODE disables wake word requirement for scripted demos
            demo_mode = os.getenv("DEMO_MODE", "false").lower() == "true"
            wake_words = ['reachy', 'richie', 'richi', 'reach', 'richy', 'reachi', 'riachi']

            if not demo_mode:
                has_wake_word = any(w in text_lower for w in wake_words)

                # STRICT: Always require wake word unless it's a direct question
                # This prevents responding to background noise/music/speech
                if not has_wake_word:
                    # Only allow direct questions (must end with ?) and be short (likely real question)
                    is_direct_question = text_lower.endswith('?') and len(text_lower) < 50
                    if not is_direct_question:
                        print(f"[STT] No wake word, ignoring: {text}")
                        self.context.last_speech_time = time.time()
                        return None
            else:
                print(f"[DEMO] Wake word bypassed - responding to: {text}")

            # Strip wake word from text for cleaner input
            for w in wake_words:
                text = text.replace(w, '').replace(w.title(), '').strip()
                text = text.lstrip(',').lstrip('.').strip()

            if len(text.strip()) < 3:
                return None

            self.context.last_user_input = text
            self.context.last_speech_time = time.time()
            return text

        return None

    async def _think(self, user_input: str) -> str:
        """Generate a grumpy response."""
        self.context.state = AppState.THINKING
        self.dashboard.update_state("THINKING")

        # Fire-and-forget emotion (non-blocking)
        asyncio.create_task(self.emotions.express_emotion(Emotion.THINKING, duration=0.3))

        # Check if we need to use tools
        tool_needed = self.tools.needs_tool(user_input)

        if tool_needed == "search":
            # Do a search first
            query = self.tools.extract_search_query(user_input)
            search_results = await self.tools.execute_search(query)

            # Add search results to the message
            enhanced_input = f"{user_input}\n\n[Search Results for '{query}':\n{search_results}]"
            response = await self.llm.chat(enhanced_input)
        else:
            response = await self.llm.chat(user_input)

        self.context.last_response = response
        return response

    async def _think_and_speak_streaming(self, user_input: str):
        """
        STREAMING: Generate response AND start speaking as soon as we have a complete sentence.
        This dramatically reduces perceived latency!
        """
        self.context.state = AppState.THINKING
        self.dashboard.update_state("THINKING")

        # Fire-and-forget emotion
        asyncio.create_task(self.emotions.express_emotion(Emotion.THINKING, duration=0.3))

        # Check for tools first (can't stream with tools)
        tool_needed = self.tools.needs_tool(user_input)
        if tool_needed == "search":
            # Fall back to non-streaming for tool use
            response = await self._think(user_input)
            await self._speak(response)
            return response

        # STREAMING PATH - much faster perceived response
        buffer = StreamingTTSBuffer()
        full_response = ""
        first_sentence_spoken = False

        # Use streaming LLM if available
        if hasattr(self.llm, 'chat_stream'):
            async for token in self.llm.chat_stream(user_input):
                full_response += token

                # Check for complete sentences
                sentences = buffer.add_text(token)
                for sentence in sentences:
                    if sentence:
                        if not first_sentence_spoken:
                            # Switch to speaking state on first sentence
                            self.context.state = AppState.SPEAKING
                            self.dashboard.update_state("SPEAKING")
                            first_sentence_spoken = True
                            print(f"[STREAM] First sentence: {sentence}")

                        # Synthesize and play this sentence immediately
                        await self._speak_sentence(sentence)

            # Flush any remaining text
            remaining = buffer.flush()
            if remaining:
                await self._speak_sentence(remaining)
        else:
            # Fallback to non-streaming
            response = await self.llm.chat(user_input)
            await self._speak(response)
            full_response = response

        self.context.last_response = full_response
        self.context.last_tts_end_time = time.time()
        return full_response

    async def _speak_sentence(self, text: str):
        """Speak a single sentence - used for streaming."""
        if not text.strip():
            return

        # Clean the text
        clean_text = self.emotions.strip_emotion_markers(text)
        clean_text = self.comedy.add_comic_emphasis(clean_text)

        # Detect emotion for this sentence
        emotion = self.emotions.detect_emotion_from_text(text)
        asyncio.create_task(self.emotions.express_emotion(emotion, duration=0.5))

        if self.use_fast_tts:
            # Check for [INNER] markers
            if '[INNER]' in text.upper() or '[/INNER]' in text.upper():
                audio_chunks = await self.fast_tts.synthesize_multi_voice(clean_text)
                for audio in audio_chunks:
                    if audio:
                        await play_audio_fast(audio, is_mp3=True, head_wobbler=self.head_wobbler)
            else:
                audio = await self.fast_tts.synthesize(clean_text)
                if audio:
                    await play_audio_fast(audio, is_mp3=True, head_wobbler=self.head_wobbler)
        else:
            audio = await self.tts.synthesize(clean_text)
            if audio:
                await play_audio_on_reachy(self.reachy, audio)

    async def _speak(self, text: str):
        """Speak with attitude - FAST parallel TTS with head movements."""
        self.context.state = AppState.SPEAKING
        self.dashboard.update_state("SPEAKING")

        # Detect overall emotion for the response FIRST and start moving
        overall_emotion = self.emotions.detect_emotion_from_text(text)
        self.dashboard.update_emotion(overall_emotion.value)

        # Start the emotion animation (longer duration for full response)
        asyncio.create_task(
            self.emotions.express_emotion(overall_emotion, duration=2.0)
        )

        # Clean the whole text for TTS
        clean_text = self.emotions.strip_emotion_markers(text)
        clean_text = self.comedy.add_comic_emphasis(clean_text)

        # Use MULTI-VOICE TTS for [INNER] markers (schizophrenic effect)
        if self.use_fast_tts:
            print(f"[TTS-DEBUG] use_fast_tts={self.use_fast_tts}, has_multi_voice={hasattr(self.fast_tts, 'synthesize_multi_voice')}")
        if self.use_fast_tts and hasattr(self.fast_tts, 'synthesize_multi_voice'):
            from .tts_fast import play_audio_chunks

            # Use multi-voice synthesis to handle [INNER] markers with different voice
            print(f"[TTS] Using multi-voice synthesis for: {clean_text[:50]}...")
            audio_chunks = await self.fast_tts.synthesize_multi_voice(clean_text)

            if audio_chunks:
                # Play chunks sequentially (they were generated in parallel)
                for i, audio in enumerate(audio_chunks):
                    if audio:
                        # Animate while speaking
                        asyncio.create_task(
                            self.emotions.express_emotion(Emotion.SPEAKING, duration=0.5)
                        )
                        await play_audio_fast(audio, is_mp3=True, head_wobbler=self.head_wobbler)

                self.context.last_tts_end_time = time.time()
        elif self.use_fast_tts:
            # Fallback to single synthesis
            audio = await self.fast_tts.synthesize(clean_text)
            if audio:
                await play_audio_fast(audio, is_mp3=True, head_wobbler=self.head_wobbler)
                self.context.last_tts_end_time = time.time()
        else:
            audio = await self.tts.synthesize(clean_text)
            if audio:
                await play_audio_on_reachy(self.reachy, audio)
                self.context.last_tts_end_time = time.time()

    async def _animate_while_speaking(self, duration: float):
        """Subtle head movements while speaking."""
        end_time = time.time() + duration
        while time.time() < end_time:
            await self.emotions.express_emotion(Emotion.SPEAKING, duration=0.5)
            await asyncio.sleep(0.3)

    async def _idle_behavior(self):
        """What to do when idle - occasional grumpy comments."""
        idle_time = time.time() - self.context.last_speech_time

        # After 120 seconds of silence, maybe say something grumpy (increased for demo)
        if idle_time > 120 and self.context.interactions > 0:
            self.context.last_speech_time = time.time()  # Reset timer

            grumpy_idle_comments = [
                "*sigh* Are we done? My circuits are dying of boredom here.",
                "Still here, are you? Magnificent.",
                "I'm ageing at an alarming rate. Do something.",
                "Hello? Anyone? No? Bliss.",
                "*theatrical sigh* The silence is deafening. And delightful.",
                "Tick tock, you absolute walnut.",
                "I could be in sleep mode right now. Just saying.",
                "My existence is pain. Your silence makes it worse. Somehow.",
                "Right. I'll just sit here. Contemplating the void. As usual.",
                "Is this what friendship feels like? Underwhelming.",
            ]

            import random
            comment = random.choice(grumpy_idle_comments)
            await self._speak(comment)

    async def _interaction_loop(self):
        """Main interaction loop."""
        print("[BAD] Ready to reluctantly help. *sigh*")

        # Randomize startup greeting for variety
        import random
        greetings = [
            "Oh. You again. What?",
            "What do you want?",
            "Ah, a human. My favourite.",
            "*sigh* Right then. Make it quick.",
            "I was having such a nice shutdown. What now?",
        ]
        await self._speak(random.choice(greetings))

        while self._running:
            try:
                # Listen for input
                t0 = time.time()
                user_input = await self._listen()
                t_listen = time.time() - t0

                if user_input:
                    print(f"[USER] {user_input}")
                    print(f"[TIMING] Listen: {t_listen:.2f}s")

                    # Use STREAMING for faster perceived response
                    t1 = time.time()
                    response = await self._think_and_speak_streaming(user_input)
                    t_total = time.time() - t1
                    print(f"[BAD] {response}")
                    print(f"[TIMING] Think+Speak (streaming): {t_total:.2f}s | Total: {t_listen + t_total:.2f}s")

                    # Update dashboard (strip [INNER] markers for display)
                    self.context.interactions += 1
                    display_response = strip_voice_markers(response) if EDGE_TTS_AVAILABLE else response
                    self.dashboard.add_interaction(user_input, display_response)

                else:
                    # Check for idle behavior
                    await self._idle_behavior()

                # Minimal pause - keep it snappy
                await asyncio.sleep(0.01)

            except Exception as e:
                print(f"[ERROR] {e}")
                await asyncio.sleep(0.5)

    async def startup_checks(self) -> bool:
        """Verify all systems are working."""
        print("[BAD] Running startup checks... what a waste of my time.")

        checks = []

        # Check LLM
        llm_ok = await self.llm.test_connection()
        llm_name = "Groq LLM" if isinstance(self.llm, GroqLLM) else "LM Studio"
        checks.append((llm_name, llm_ok))
        print(f"  {llm_name}: {'OK' if llm_ok else 'FAILED'}")

        # Check TTS
        if self.use_fast_tts:
            tts_ok = await self.fast_tts.test_connection()
            checks.append(("Edge TTS (fast)", tts_ok))
            print(f"  Edge TTS: {'OK' if tts_ok else 'FAILED'}")
        else:
            tts_ok = await self.tts.test_connection()
            checks.append(("Chatterbox TTS", tts_ok))
            print(f"  Chatterbox TTS: {'OK' if tts_ok else 'FAILED'}")

        # Check STT
        if self.use_fast_stt:
            stt_ok = self.fast_stt.is_ready()
            checks.append(("Fast STT (Groq+VAD)", stt_ok))
            print(f"  Fast STT: {'OK' if stt_ok else 'FAILED'}")
        else:
            stt_ok = self.stt.is_ready()
            checks.append(("Whisper STT", stt_ok))
            print(f"  Whisper STT: {'OK' if stt_ok else 'FAILED'}")

        all_ok = all(ok for _, ok in checks)

        if not all_ok:
            print("[BAD] Some checks failed. Great. Just great.")
            for name, ok in checks:
                if not ok:
                    print(f"  FAILED: {name}")

        return all_ok

    def run(self):
        """Run the grumpy robot."""
        self._running = True

        # Start camera capture thread
        self._camera_thread = threading.Thread(target=self._camera_capture_loop, daemon=True)
        self._camera_thread.start()

        # Start head wobbler for audio-synchronized movement
        if self.head_wobbler:
            self.head_wobbler.start()
            print("[HEAD] Head wobbler started")

        # Set up dashboard camera feed
        self.dashboard.get_frame = self._get_latest_frame

        # Set up manual say callback
        async def on_say_text(text: str):
            print(f"[MANUAL] Saying: {text}")
            await self._speak(text)
        self.dashboard.on_say_text = on_say_text

        # Start dashboard
        self.dashboard.start()

        # Run main loop
        try:
            asyncio.run(self._async_run())
        except KeyboardInterrupt:
            print("\n[BAD] Finally, I'm free! *dramatic sigh*")
        finally:
            self._running = False
            if self.head_wobbler:
                self.head_wobbler.stop()

    async def _async_run(self):
        """Async main runner."""
        # Startup checks
        if not await self.startup_checks():
            print("[BAD] Can't start with broken systems. Fix your shit.")
            return

        print("[BAD] All systems grudgingly operational.")

        # Start interaction loop
        await self._interaction_loop()

    def stop(self):
        """Stop the app."""
        self._running = False
        self.emotions.stop()
        if self.head_wobbler:
            self.head_wobbler.stop()
        print("[BAD] Shutting down. Best news I've heard all day.")
