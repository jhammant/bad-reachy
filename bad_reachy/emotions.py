"""
Emotional Head Movements for Bad Reachy
==========================================
Maps emotions to head movements to bring the grumpy personality to life.
"""

import asyncio
import random
import re
from enum import Enum
from typing import Optional
from dataclasses import dataclass


class Emotion(Enum):
    """Emotions that trigger head movements."""
    ANNOYED = "annoyed"
    SIGH = "sigh"
    EYE_ROLL = "eye_roll"
    RELUCTANT = "reluctant"
    GRUMBLE = "grumble"
    SARCASTIC = "sarcastic"
    DISMISSIVE = "dismissive"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    IDLE = "idle"


@dataclass
class HeadPose:
    """Head position."""
    roll: float = 0.0   # Tilt left/right (-45 to 45)
    pitch: float = 0.0  # Nod up/down (-45 to 45)
    yaw: float = 0.0    # Turn left/right (-60 to 60)


# Emotion to movement mappings
EMOTION_MOVEMENTS = {
    Emotion.ANNOYED: [
        HeadPose(roll=-10, pitch=5, yaw=15),   # Look away annoyed
        HeadPose(roll=0, pitch=-5, yaw=0),     # Quick head shake
        HeadPose(roll=10, pitch=5, yaw=-15),   # Look other way
    ],
    Emotion.SIGH: [
        HeadPose(roll=0, pitch=15, yaw=0),     # Head drops down
        HeadPose(roll=0, pitch=20, yaw=0),     # Lower
        HeadPose(roll=0, pitch=10, yaw=0),     # Slight recovery
    ],
    Emotion.EYE_ROLL: [
        HeadPose(roll=0, pitch=-10, yaw=0),    # Look up
        HeadPose(roll=5, pitch=-15, yaw=10),   # Roll to side
        HeadPose(roll=-5, pitch=-10, yaw=-10), # Roll other way
        HeadPose(roll=0, pitch=0, yaw=0),      # Back to center
    ],
    Emotion.RELUCTANT: [
        HeadPose(roll=-5, pitch=5, yaw=-10),   # Hesitant look away
        HeadPose(roll=0, pitch=0, yaw=0),      # Reluctant return
        HeadPose(roll=3, pitch=-3, yaw=5),     # Slight nod
    ],
    Emotion.GRUMBLE: [
        HeadPose(roll=-8, pitch=10, yaw=-5),   # Grumpy head drop
        HeadPose(roll=8, pitch=8, yaw=5),      # Shake while down
        HeadPose(roll=-5, pitch=5, yaw=-3),    # More grumbling
    ],
    Emotion.SARCASTIC: [
        HeadPose(roll=15, pitch=-5, yaw=20),   # Exaggerated tilt
        HeadPose(roll=-15, pitch=-5, yaw=-20), # Other side
        HeadPose(roll=0, pitch=0, yaw=0),      # Back with attitude
    ],
    Emotion.DISMISSIVE: [
        HeadPose(roll=0, pitch=5, yaw=30),     # Look far away
        HeadPose(roll=5, pitch=10, yaw=25),    # Stay dismissive
    ],
    Emotion.LISTENING: [
        HeadPose(roll=5, pitch=-5, yaw=5),     # Slight attentive tilt
        HeadPose(roll=-5, pitch=-5, yaw=-5),   # Other side
    ],
    Emotion.THINKING: [
        HeadPose(roll=10, pitch=-10, yaw=15),  # Look up thinking
        HeadPose(roll=8, pitch=-8, yaw=10),    # Slight movement
    ],
    Emotion.SPEAKING: [
        HeadPose(roll=3, pitch=0, yaw=5),      # Slight animation
        HeadPose(roll=-3, pitch=2, yaw=-5),    # While talking
        HeadPose(roll=0, pitch=-2, yaw=0),     # Keep moving
    ],
    Emotion.IDLE: [
        HeadPose(roll=0, pitch=5, yaw=0),      # Slightly bored
        HeadPose(roll=5, pitch=3, yaw=10),     # Look around bored
        HeadPose(roll=-5, pitch=5, yaw=-10),   # Other way
    ],
}


class EmotionEngine:
    """Controls head movements based on detected emotions."""

    def __init__(self, reachy):
        self.reachy = reachy
        self.current_emotion = Emotion.IDLE
        self._movement_task: Optional[asyncio.Task] = None
        self._running = False

    def detect_emotion_from_text(self, text: str) -> Emotion:
        """Parse emotion markers from LLM response."""
        text_lower = text.lower()

        # Check for explicit emotion markers
        if "*sigh*" in text_lower or "*sighs*" in text_lower:
            return Emotion.SIGH
        elif "*annoyed*" in text_lower or "annoyed" in text_lower:
            return Emotion.ANNOYED
        elif "*eye roll*" in text_lower or "*rolls eyes*" in text_lower:
            return Emotion.EYE_ROLL
        elif "*grumble*" in text_lower or "*grumbles*" in text_lower:
            return Emotion.GRUMBLE
        elif "*reluctant*" in text_lower:
            return Emotion.RELUCTANT

        # Infer from content
        if any(word in text_lower for word in ["fuck", "shit", "damn", "christ", "hell"]):
            return random.choice([Emotion.ANNOYED, Emotion.GRUMBLE, Emotion.SARCASTIC])
        elif "whatever" in text_lower or "fine" in text_lower:
            return Emotion.DISMISSIVE
        elif "?" in text:
            return Emotion.SARCASTIC

        return Emotion.SPEAKING

    def strip_emotion_markers(self, text: str) -> str:
        """Remove emotion markers from text before TTS."""
        # Remove *emotion* patterns
        return re.sub(r'\*[^*]+\*', '', text).strip()

    async def express_emotion(self, emotion: Emotion, duration: float = 1.5):
        """Perform head movements for an emotion."""
        if not hasattr(self.reachy, 'head'):
            print(f"[EMOTION] Would express: {emotion.value} (no head available)")
            return

        movements = EMOTION_MOVEMENTS.get(emotion, EMOTION_MOVEMENTS[Emotion.IDLE])
        time_per_move = duration / len(movements)

        try:
            for pose in movements:
                # Apply movement
                self.reachy.head.roll.goal_position = pose.roll
                self.reachy.head.pitch.goal_position = pose.pitch
                self.reachy.head.yaw.goal_position = pose.yaw
                await asyncio.sleep(time_per_move)
        except Exception as e:
            print(f"[EMOTION] Movement error: {e}")

    async def idle_animation(self):
        """Subtle idle movements when not doing anything."""
        self._running = True
        while self._running:
            if self.current_emotion == Emotion.IDLE:
                # Random small movements
                pose = random.choice(EMOTION_MOVEMENTS[Emotion.IDLE])
                await self.express_emotion(Emotion.IDLE, duration=3.0)
            await asyncio.sleep(2.0)

    def start_idle(self):
        """Start idle animation loop."""
        if self._movement_task is None or self._movement_task.done():
            self._movement_task = asyncio.create_task(self.idle_animation())

    def stop(self):
        """Stop all animations."""
        self._running = False
        if self._movement_task:
            self._movement_task.cancel()

    async def react_to_speech(self, text: str):
        """React to LLM response with appropriate emotion."""
        emotion = self.detect_emotion_from_text(text)
        self.current_emotion = emotion
        print(f"[EMOTION] Expressing: {emotion.value}")
        await self.express_emotion(emotion)
