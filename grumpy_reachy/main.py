"""
Grumpy Reachy - Main Application
================================
A sarcastic, sweary robot assistant with attitude.
"""

import asyncio
import time
import threading
import numpy as np
from enum import Enum
from typing import Optional
from dataclasses import dataclass, field

from .config import Config
from .llm import LMStudioClient
from .stt import LocalWhisperSTT
from .tts import ChatterboxTTS, play_audio_on_reachy
from .emotions import EmotionEngine, Emotion
from .dashboard import GrumpyDashboard
from .tools import ToolManager, TOOL_AWARE_PROMPT_ADDITION
from .comedy import ComedyEngine, COMEDY_SYSTEM_PROMPT


class AppState(Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"


@dataclass
class GrumpyContext:
    """Runtime context for the app."""
    state: AppState = AppState.IDLE
    last_speech_time: float = 0.0
    last_user_input: str = ""
    last_response: str = ""
    interactions: int = 0


class GrumpyReachyApp:
    """Main application - a grumpy robot that reluctantly helps."""

    def __init__(self, reachy):
        self.reachy = reachy
        self.config = Config()
        self.context = GrumpyContext()

        # Initialize components
        print("[GRUMPY] Initializing... ugh, here we go again.")

        self.llm = LMStudioClient(self.config)
        # Use comedy-enhanced prompt
        self.llm.system_prompt = COMEDY_SYSTEM_PROMPT + "\n" + TOOL_AWARE_PROMPT_ADDITION

        self.stt = LocalWhisperSTT(self.config.whisper_model)
        self.tts = ChatterboxTTS(self.config.tts_server_url)
        self.tools = ToolManager()

        self.emotions = EmotionEngine(reachy)
        self.comedy = ComedyEngine(self.tts)
        self.dashboard = GrumpyDashboard(port=8080)

        # Camera frame for dashboard
        self._latest_frame = None
        self._camera_thread = None
        self._running = False

        # Audio capture
        self._audio_buffer = []

    def _get_latest_frame(self):
        """Get the latest camera frame."""
        return self._latest_frame

    def _camera_capture_loop(self):
        """Background thread for camera capture."""
        while self._running:
            try:
                if hasattr(self.reachy, 'media') and hasattr(self.reachy.media, 'camera'):
                    frame = self.reachy.media.camera.read()
                    if frame is not None:
                        self._latest_frame = frame
            except Exception as e:
                pass
            time.sleep(0.1)

    async def _capture_audio(self, duration: float) -> bytes:
        """Capture audio from Reachy's microphone."""
        try:
            if hasattr(self.reachy, 'media') and hasattr(self.reachy.media, 'microphone'):
                # Record audio
                samples = int(duration * self.config.sample_rate)
                audio_data = self.reachy.media.microphone.record(samples)

                if isinstance(audio_data, np.ndarray):
                    # Convert to bytes
                    if audio_data.dtype == np.float32:
                        audio_data = (audio_data * 32767).astype(np.int16)
                    return audio_data.tobytes()

            # Fallback: return empty audio
            return b'\x00' * int(duration * self.config.sample_rate * 2)

        except Exception as e:
            print(f"[AUDIO] Capture error: {e}")
            return b'\x00' * int(duration * self.config.sample_rate * 2)

    async def _listen(self) -> Optional[str]:
        """Listen for user speech."""
        self.context.state = AppState.LISTENING
        self.dashboard.update_state("LISTENING")

        # Show listening emotion
        await self.emotions.express_emotion(Emotion.LISTENING, duration=0.5)

        # Capture audio
        audio_data = await self._capture_audio(self.config.recording_duration)

        # Transcribe
        text = await self.stt.transcribe(audio_data, self.config.sample_rate)

        if text and len(text.strip()) > 2:
            self.context.last_user_input = text
            self.context.last_speech_time = time.time()
            return text

        return None

    async def _think(self, user_input: str) -> str:
        """Generate a grumpy response."""
        self.context.state = AppState.THINKING
        self.dashboard.update_state("THINKING")

        # Show thinking emotion
        await self.emotions.express_emotion(Emotion.THINKING, duration=0.5)

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

    async def _speak(self, text: str):
        """Speak with attitude, emotion, and COMEDY TIMING."""
        self.context.state = AppState.SPEAKING
        self.dashboard.update_state("SPEAKING")

        # Break text into comedy segments with timing
        segments = self.comedy.enhance_for_comedy(text)

        for segment_text, segment_type in segments:
            if segment_type == 'pause':
                # Dramatic pause with head movement
                await self.emotions.express_emotion(Emotion.SARCASTIC, duration=0.8)
                await asyncio.sleep(0.8)
                continue

            if segment_type in ['sigh', 'groan']:
                # Perform the emotional sound effect
                await self.emotions.express_emotion(Emotion.SIGH, duration=1.0)
                effect_audio = await self.tts.synthesize(segment_text)
                if effect_audio:
                    await play_audio_on_reachy(self.reachy, effect_audio)
                await asyncio.sleep(0.3)
                continue

            if segment_type == 'rimshot':
                # Ba dum tss moment
                await self.emotions.express_emotion(Emotion.SARCASTIC, duration=0.5)
                effect_audio = await self.tts.synthesize("ba dum tss")
                if effect_audio:
                    await play_audio_on_reachy(self.reachy, effect_audio)
                continue

            # Regular speech segment
            if not segment_text.strip():
                continue

            # Detect and express emotion for this segment
            emotion = self.emotions.detect_emotion_from_text(segment_text)
            self.dashboard.update_emotion(emotion.value)

            # Clean text for TTS (remove emotion markers)
            clean_text = self.emotions.strip_emotion_markers(segment_text)

            # Add comic emphasis
            clean_text = self.comedy.add_comic_emphasis(clean_text)

            # Start emotion expression
            emotion_task = asyncio.create_task(
                self.emotions.express_emotion(emotion, duration=2.0)
            )

            # Synthesize speech
            audio = await self.tts.synthesize(clean_text)

            if audio:
                # Play through Reachy
                await play_audio_on_reachy(self.reachy, audio)

                # Continue head movements while speaking
                await self._animate_while_speaking(len(audio) / 24000)

            await emotion_task

            # Brief pause between segments for timing
            await asyncio.sleep(0.2)

    async def _animate_while_speaking(self, duration: float):
        """Subtle head movements while speaking."""
        end_time = time.time() + duration
        while time.time() < end_time:
            await self.emotions.express_emotion(Emotion.SPEAKING, duration=0.5)
            await asyncio.sleep(0.3)

    async def _idle_behavior(self):
        """What to do when idle - occasional grumpy comments."""
        idle_time = time.time() - self.context.last_speech_time

        # After 60 seconds of silence, maybe say something grumpy
        if idle_time > 60 and self.context.interactions > 0:
            self.context.last_speech_time = time.time()  # Reset timer

            grumpy_idle_comments = [
                "*sigh* Are we done here or what?",
                "I'm not getting any younger, you know.",
                "*grumbles* Could be doing literally anything else right now.",
                "Hello? Still there? Not that I care.",
            ]

            import random
            comment = random.choice(grumpy_idle_comments)
            await self._speak(comment)

    async def _interaction_loop(self):
        """Main interaction loop."""
        print("[GRUMPY] Ready to reluctantly help. *sigh*")

        # Grumpy greeting
        await self._speak("Oh great, you're here. What do you want? *sigh*")

        while self._running:
            try:
                # Listen for input
                user_input = await self._listen()

                if user_input:
                    print(f"[USER] {user_input}")

                    # Generate response
                    response = await self._think(user_input)
                    print(f"[GRUMPY] {response}")

                    # Speak response
                    await self._speak(response)

                    # Update dashboard
                    self.context.interactions += 1
                    self.dashboard.add_interaction(user_input, response)

                else:
                    # Check for idle behavior
                    await self._idle_behavior()

                # Brief pause between loops
                await asyncio.sleep(0.1)

            except Exception as e:
                print(f"[ERROR] {e}")
                await asyncio.sleep(1.0)

    async def startup_checks(self) -> bool:
        """Verify all systems are working."""
        print("[GRUMPY] Running startup checks... what a waste of my time.")

        checks = []

        # Check LM Studio
        llm_ok = await self.llm.test_connection()
        checks.append(("LM Studio", llm_ok))
        print(f"  LM Studio: {'OK' if llm_ok else 'FAILED'}")

        # Check TTS
        tts_ok = await self.tts.test_connection()
        checks.append(("Chatterbox TTS", tts_ok))
        print(f"  Chatterbox TTS: {'OK' if tts_ok else 'FAILED'}")

        # Check STT
        stt_ok = self.stt.is_ready()
        checks.append(("Whisper STT", stt_ok))
        print(f"  Whisper STT: {'OK' if stt_ok else 'FAILED'}")

        all_ok = all(ok for _, ok in checks)

        if not all_ok:
            print("[GRUMPY] Some checks failed. Great. Just great.")
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

        # Set up dashboard camera feed
        self.dashboard.get_frame = self._get_latest_frame

        # Start dashboard
        self.dashboard.start()

        # Run main loop
        try:
            asyncio.run(self._async_run())
        except KeyboardInterrupt:
            print("\n[GRUMPY] Finally, I'm free! *dramatic sigh*")
        finally:
            self._running = False

    async def _async_run(self):
        """Async main runner."""
        # Startup checks
        if not await self.startup_checks():
            print("[GRUMPY] Can't start with broken systems. Fix your shit.")
            return

        print("[GRUMPY] All systems grudgingly operational.")

        # Start interaction loop
        await self._interaction_loop()

    def stop(self):
        """Stop the app."""
        self._running = False
        self.emotions.stop()
        print("[GRUMPY] Shutting down. Best news I've heard all day.")
