"""
Emotional Head Movements for Bad Reachy
==========================================
Maps emotions to head movements to bring the grumpy personality to life.
"""

import asyncio
import random
import re
import math
import numpy as np
from enum import Enum
from typing import Optional
from dataclasses import dataclass


def euler_to_transformation_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Convert roll/pitch/yaw (degrees) to 4x4 transformation matrix for Reachy head."""
    # Convert to radians
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)

    # Rotation matrices
    # Roll (rotation around X axis)
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
    ])

    # Pitch (rotation around Y axis)
    Ry = np.array([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])

    # Yaw (rotation around Z axis)
    Rz = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ])

    # Combined rotation: Rz * Ry * Rx (yaw, then pitch, then roll)
    R = Rz @ Ry @ Rx

    # Create 4x4 transformation matrix (no translation for head)
    T = np.eye(4)
    T[:3, :3] = R

    return T


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
    # New dramatic emotions
    DOUBLE_TAKE = "double_take"
    FACEPALM = "facepalm"
    SHOCK = "shock"
    DEADPAN = "deadpan"
    EXASPERATED = "exasperated"
    JUDGING = "judging"


@dataclass
class HeadPose:
    """Head position."""
    roll: float = 0.0   # Tilt left/right (-45 to 45)
    pitch: float = 0.0  # Nod up/down (-45 to 45)
    yaw: float = 0.0    # Turn left/right (-60 to 60)


# Emotion to movement mappings - BIGGER, MORE EXPRESSIVE movements
EMOTION_MOVEMENTS = {
    Emotion.ANNOYED: [
        HeadPose(roll=-15, pitch=8, yaw=25),   # Look away annoyed - BIG
        HeadPose(roll=0, pitch=-8, yaw=0),     # Quick head shake
        HeadPose(roll=15, pitch=8, yaw=-25),   # Look other way - BIG
        HeadPose(roll=0, pitch=0, yaw=0),      # Back to glare
    ],
    Emotion.SIGH: [
        HeadPose(roll=0, pitch=25, yaw=0),     # Head drops down - DRAMATIC
        HeadPose(roll=5, pitch=30, yaw=5),     # Lower with tilt
        HeadPose(roll=-5, pitch=20, yaw=-5),   # Shake while down
        HeadPose(roll=0, pitch=10, yaw=0),     # Slow recovery
    ],
    Emotion.EYE_ROLL: [
        HeadPose(roll=0, pitch=-15, yaw=0),    # Look up HARD
        HeadPose(roll=10, pitch=-20, yaw=15),  # Roll to side - EXAGGERATED
        HeadPose(roll=-10, pitch=-15, yaw=-15),# Roll other way
        HeadPose(roll=0, pitch=-10, yaw=0),    # Still looking up
        HeadPose(roll=0, pitch=0, yaw=0),      # Back to center with attitude
    ],
    Emotion.RELUCTANT: [
        HeadPose(roll=-8, pitch=8, yaw=-15),   # Hesitant look away
        HeadPose(roll=0, pitch=5, yaw=0),      # Reluctant return
        HeadPose(roll=5, pitch=-5, yaw=8),     # Fine, whatever nod
    ],
    Emotion.GRUMBLE: [
        HeadPose(roll=-12, pitch=15, yaw=-8),  # Grumpy head drop
        HeadPose(roll=12, pitch=12, yaw=8),    # Shake while down
        HeadPose(roll=-8, pitch=10, yaw=-5),   # More grumbling
        HeadPose(roll=8, pitch=8, yaw=5),      # Keep shaking
    ],
    Emotion.SARCASTIC: [
        HeadPose(roll=20, pitch=-8, yaw=25),   # Exaggerated tilt - BIG
        HeadPose(roll=-20, pitch=-8, yaw=-25), # Other side - BIG
        HeadPose(roll=10, pitch=0, yaw=10),    # Settle with attitude
        HeadPose(roll=0, pitch=0, yaw=0),      # Back
    ],
    Emotion.DISMISSIVE: [
        HeadPose(roll=0, pitch=8, yaw=35),     # Look FAR away
        HeadPose(roll=8, pitch=12, yaw=40),    # Even more away
        HeadPose(roll=5, pitch=10, yaw=30),    # Stay dismissive
    ],
    Emotion.LISTENING: [
        HeadPose(roll=8, pitch=-8, yaw=8),     # Attentive tilt
        HeadPose(roll=0, pitch=-5, yaw=0),     # Center, looking at you
        HeadPose(roll=-8, pitch=-8, yaw=-8),   # Other side
        HeadPose(roll=0, pitch=-5, yaw=0),     # Back to center
    ],
    Emotion.THINKING: [
        HeadPose(roll=15, pitch=-15, yaw=20),  # Look up thinking - DRAMATIC
        HeadPose(roll=12, pitch=-12, yaw=15),  # Slight movement
        HeadPose(roll=10, pitch=-10, yaw=10),  # Processing...
    ],
    Emotion.SPEAKING: [
        HeadPose(roll=5, pitch=0, yaw=8),      # Animation while talking
        HeadPose(roll=-5, pitch=3, yaw=-8),    # Keep moving
        HeadPose(roll=3, pitch=-3, yaw=5),     # Expressive
        HeadPose(roll=-3, pitch=0, yaw=-5),    # More movement
        HeadPose(roll=0, pitch=0, yaw=0),      # Center
    ],
    Emotion.IDLE: [
        HeadPose(roll=0, pitch=8, yaw=0),      # Slightly bored
        HeadPose(roll=8, pitch=5, yaw=15),     # Look around bored
        HeadPose(roll=-8, pitch=8, yaw=-15),   # Other way
        HeadPose(roll=0, pitch=10, yaw=0),     # Head droop
    ],
    # NEW DRAMATIC EMOTIONS
    Emotion.DOUBLE_TAKE: [
        HeadPose(roll=0, pitch=0, yaw=20),     # Look away
        HeadPose(roll=0, pitch=0, yaw=0),      # Look back
        HeadPose(roll=0, pitch=0, yaw=25),     # WAIT WHAT - look away again
        HeadPose(roll=-10, pitch=-10, yaw=0),  # Snap back with shock
        HeadPose(roll=0, pitch=0, yaw=0),      # Settle
    ],
    Emotion.FACEPALM: [
        HeadPose(roll=0, pitch=35, yaw=0),     # Head drops HARD
        HeadPose(roll=10, pitch=40, yaw=10),   # Shake in despair
        HeadPose(roll=-10, pitch=35, yaw=-10), # Other way
        HeadPose(roll=0, pitch=30, yaw=0),     # Stay down
        HeadPose(roll=0, pitch=15, yaw=0),     # Slowly recover
    ],
    Emotion.SHOCK: [
        HeadPose(roll=0, pitch=-20, yaw=0),    # Head BACK in shock
        HeadPose(roll=-5, pitch=-25, yaw=-10), # Recoil
        HeadPose(roll=5, pitch=-15, yaw=10),   # Settle into disbelief
        HeadPose(roll=0, pitch=-10, yaw=0),    # Still shocked
    ],
    Emotion.DEADPAN: [
        HeadPose(roll=0, pitch=0, yaw=0),      # Just... stare
        HeadPose(roll=2, pitch=0, yaw=0),      # Tiny movement
        HeadPose(roll=-2, pitch=0, yaw=0),     # Other way
        HeadPose(roll=0, pitch=0, yaw=0),      # Back to stare
    ],
    Emotion.EXASPERATED: [
        HeadPose(roll=0, pitch=-20, yaw=0),    # Look UP in exasperation
        HeadPose(roll=-15, pitch=-15, yaw=-20),# Roll head
        HeadPose(roll=15, pitch=-15, yaw=20),  # Other way
        HeadPose(roll=0, pitch=25, yaw=0),     # Head drop
        HeadPose(roll=0, pitch=10, yaw=0),     # Defeated
    ],
    Emotion.JUDGING: [
        HeadPose(roll=15, pitch=-5, yaw=0),    # Tilt head - JUDGING
        HeadPose(roll=18, pitch=-3, yaw=5),    # Hold the judgment
        HeadPose(roll=15, pitch=0, yaw=0),     # Still judging
        HeadPose(roll=10, pitch=0, yaw=0),     # Slight nod - yep, judged
    ],
}


class EmotionEngine:
    """Controls head movements based on detected emotions."""

    def __init__(self, reachy):
        self.reachy = reachy
        self.current_emotion = Emotion.IDLE
        self._movement_task: Optional[asyncio.Task] = None
        self._running = False
        self._has_head = False

        # Check for ReachyMini head API
        if reachy and hasattr(reachy, 'set_target_head_pose'):
            self._has_head = True
            print("[EMOTION] Head control available! Emotions will be EXPRESSIVE.")
        else:
            print("[EMOTION] No head control available - running in limited mode.")

    def detect_emotion_from_text(self, text: str) -> Emotion:
        """Parse emotion markers from LLM response."""
        text_lower = text.lower()

        # Check for explicit emotion markers first
        if "*sigh*" in text_lower or "*sighs*" in text_lower or "*heavy sigh*" in text_lower:
            return Emotion.SIGH
        elif "*facepalm*" in text_lower or "*face palm*" in text_lower:
            return Emotion.FACEPALM
        elif "*eye roll*" in text_lower or "*rolls eyes*" in text_lower:
            return Emotion.EYE_ROLL
        elif "*grumble*" in text_lower or "*grumbles*" in text_lower:
            return Emotion.GRUMBLE
        elif "*stares*" in text_lower or "*deadpan*" in text_lower:
            return Emotion.DEADPAN
        elif "*shocked*" in text_lower or "what the" in text_lower:
            return Emotion.SHOCK

        # Infer from content - EXPANDED
        strong_swears = ["fuck", "bloody hell", "christ", "jesus"]
        mild_swears = ["damn", "crap", "hell"]

        if any(word in text_lower for word in strong_swears):
            return random.choice([Emotion.EXASPERATED, Emotion.FACEPALM, Emotion.ANNOYED])
        elif any(word in text_lower for word in mild_swears):
            return random.choice([Emotion.ANNOYED, Emotion.GRUMBLE, Emotion.SARCASTIC])

        # Judgmental phrases
        if any(phrase in text_lower for phrase in ["you absolute", "you utter", "you complete", "really?", "seriously?"]):
            return Emotion.JUDGING

        # Exasperation
        if any(phrase in text_lower for phrase in ["for crying out loud", "oh for", "why me", "not again"]):
            return Emotion.EXASPERATED

        # Sarcastic/dismissive
        if "whatever" in text_lower or "fine" in text_lower or "if you must" in text_lower:
            return Emotion.DISMISSIVE

        # Question with attitude
        if "?" in text and any(word in text_lower for word in ["really", "seriously", "honestly"]):
            return Emotion.DEADPAN
        elif "?" in text:
            return Emotion.SARCASTIC

        # Insulting = judging
        insults = ["walnut", "melon", "bellend", "muppet", "plonker", "numpty", "turnip"]
        if any(insult in text_lower for insult in insults):
            return random.choice([Emotion.JUDGING, Emotion.SARCASTIC, Emotion.DEADPAN])

        # Default to animated speaking
        return random.choice([Emotion.SPEAKING, Emotion.SARCASTIC])

    def strip_emotion_markers(self, text: str) -> str:
        """Remove emotion markers from text before TTS."""
        # Remove *emotion* patterns
        return re.sub(r'\*[^*]+\*', '', text).strip()

    async def express_emotion(self, emotion: Emotion, duration: float = 1.5):
        """Perform head movements for an emotion."""
        # Check for ReachyMini API (uses set_target_head_pose)
        has_head_api = hasattr(self.reachy, 'set_target_head_pose') if self.reachy else False

        if not has_head_api:
            print(f"[EMOTION] Would express: {emotion.value} (no head available)")
            return

        movements = EMOTION_MOVEMENTS.get(emotion, EMOTION_MOVEMENTS[Emotion.IDLE])
        time_per_move = duration / len(movements)

        try:
            for pose in movements:
                # Convert roll/pitch/yaw to transformation matrix
                T = euler_to_transformation_matrix(pose.roll, pose.pitch, pose.yaw)

                # Apply movement using ReachyMini API
                self.reachy.set_target_head_pose(T)

                await asyncio.sleep(time_per_move)

            # Return to neutral at end (subtle)
            T_neutral = euler_to_transformation_matrix(0, 0, 0)
            self.reachy.set_target_head_pose(T_neutral)

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
