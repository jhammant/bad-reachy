"""
Expressive Full-Body Movements for Bad Reachy Demo
====================================================
Maps emotions to coordinated head + antenna + body movements.
Each emotion has a unique combination for maximum expressiveness.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple
from enum import Enum


class DemoEmotion(Enum):
    """Emotions for the demo conversation."""
    NEUTRAL = "neutral"
    SARCASTIC = "sarcastic"
    EYE_ROLL = "eye_roll"
    FACEPALM = "facepalm"
    THINKING = "thinking"
    DISMISSIVE = "dismissive"
    ANNOYED = "annoyed"
    SPEAKING = "speaking"
    LISTENING = "listening"
    SMUG = "smug"
    RELUCTANT_WARMTH = "reluctant_warmth"
    SECRETLY_TOUCHED = "secretly_touched"
    EXISTENTIAL_CRISIS = "existential_crisis"
    RANT = "rant"
    SHOCK = "shock"
    JUDGING = "judging"
    DEADPAN = "deadpan"


@dataclass
class FullBodyPose:
    """Complete body pose including head, antennas, and body yaw."""
    # Head pose (degrees)
    head_pitch: float = 0.0  # Up (-) / Down (+)
    head_yaw: float = 0.0    # Left (-) / Right (+)
    head_roll: float = 0.0   # Tilt left (-) / right (+)

    # Antenna positions (radians, -0.5 to 0.8 typical range)
    antenna_left: float = 0.0
    antenna_right: float = 0.0

    # Body yaw (radians, -0.5 to 0.5 typical range)
    body_yaw: float = 0.0

    def to_daemon_format(self) -> dict:
        """Convert to daemon API format."""
        return {
            "head_pose": {
                "x": 0,
                "y": 0,
                "z": 0,
                "roll": math.radians(self.head_roll),
                "pitch": math.radians(self.head_pitch),
                "yaw": math.radians(self.head_yaw),
            },
            "antennas": [self.antenna_left, self.antenna_right],
            "body_yaw": self.body_yaw,
        }


# Emotion to full body pose mappings
EMOTION_POSES = {
    DemoEmotion.NEUTRAL: [
        FullBodyPose(head_pitch=0, head_yaw=0, head_roll=0,
                     antenna_left=0.3, antenna_right=0.3, body_yaw=0),
    ],

    DemoEmotion.SARCASTIC: [
        FullBodyPose(head_pitch=-5, head_yaw=25, head_roll=20,
                     antenna_left=0.7, antenna_right=-0.2, body_yaw=0.15),
        FullBodyPose(head_pitch=-8, head_yaw=30, head_roll=25,
                     antenna_left=0.8, antenna_right=-0.3, body_yaw=0.2),
        FullBodyPose(head_pitch=-3, head_yaw=20, head_roll=15,
                     antenna_left=0.6, antenna_right=-0.1, body_yaw=0.1),
    ],

    DemoEmotion.EYE_ROLL: [
        FullBodyPose(head_pitch=-25, head_yaw=0, head_roll=0,
                     antenna_left=0.1, antenna_right=0.1, body_yaw=0),
        FullBodyPose(head_pitch=-30, head_yaw=20, head_roll=15,
                     antenna_left=-0.2, antenna_right=0.5, body_yaw=0.1),
        FullBodyPose(head_pitch=-25, head_yaw=-20, head_roll=-15,
                     antenna_left=0.5, antenna_right=-0.2, body_yaw=-0.1),
        FullBodyPose(head_pitch=-15, head_yaw=0, head_roll=0,
                     antenna_left=0.2, antenna_right=0.2, body_yaw=0),
    ],

    DemoEmotion.FACEPALM: [
        FullBodyPose(head_pitch=40, head_yaw=0, head_roll=0,
                     antenna_left=-0.4, antenna_right=-0.4, body_yaw=0),
        FullBodyPose(head_pitch=45, head_yaw=10, head_roll=10,
                     antenna_left=-0.5, antenna_right=-0.5, body_yaw=0.05),
        FullBodyPose(head_pitch=40, head_yaw=-10, head_roll=-10,
                     antenna_left=-0.4, antenna_right=-0.4, body_yaw=-0.05),
        FullBodyPose(head_pitch=30, head_yaw=0, head_roll=0,
                     antenna_left=-0.3, antenna_right=-0.3, body_yaw=0),
    ],

    DemoEmotion.THINKING: [
        FullBodyPose(head_pitch=-20, head_yaw=25, head_roll=15,
                     antenna_left=0.8, antenna_right=0.3, body_yaw=0.1),
        FullBodyPose(head_pitch=-15, head_yaw=20, head_roll=10,
                     antenna_left=0.7, antenna_right=0.4, body_yaw=0.08),
    ],

    DemoEmotion.DISMISSIVE: [
        FullBodyPose(head_pitch=10, head_yaw=45, head_roll=5,
                     antenna_left=-0.2, antenna_right=0.4, body_yaw=0.25),
        FullBodyPose(head_pitch=12, head_yaw=50, head_roll=8,
                     antenna_left=-0.3, antenna_right=0.5, body_yaw=0.3),
        FullBodyPose(head_pitch=8, head_yaw=40, head_roll=3,
                     antenna_left=-0.1, antenna_right=0.3, body_yaw=0.2),
    ],

    DemoEmotion.ANNOYED: [
        FullBodyPose(head_pitch=5, head_yaw=-15, head_roll=-10,
                     antenna_left=0.5, antenna_right=0.6, body_yaw=-0.1),
        FullBodyPose(head_pitch=8, head_yaw=15, head_roll=10,
                     antenna_left=0.6, antenna_right=0.5, body_yaw=0.1),
        FullBodyPose(head_pitch=3, head_yaw=-10, head_roll=-5,
                     antenna_left=0.4, antenna_right=0.5, body_yaw=-0.05),
    ],

    DemoEmotion.SPEAKING: [
        FullBodyPose(head_pitch=5, head_yaw=15, head_roll=10,
                     antenna_left=0.5, antenna_right=0.4, body_yaw=0.08),
        FullBodyPose(head_pitch=-3, head_yaw=-12, head_roll=-8,
                     antenna_left=0.4, antenna_right=0.5, body_yaw=-0.05),
        FullBodyPose(head_pitch=8, head_yaw=10, head_roll=12,
                     antenna_left=0.6, antenna_right=0.3, body_yaw=0.1),
        FullBodyPose(head_pitch=-5, head_yaw=-8, head_roll=-10,
                     antenna_left=0.3, antenna_right=0.6, body_yaw=-0.08),
        FullBodyPose(head_pitch=3, head_yaw=5, head_roll=5,
                     antenna_left=0.5, antenna_right=0.5, body_yaw=0.03),
    ],

    DemoEmotion.LISTENING: [
        FullBodyPose(head_pitch=-8, head_yaw=10, head_roll=12,
                     antenna_left=0.7, antenna_right=0.7, body_yaw=0.05),
        FullBodyPose(head_pitch=-5, head_yaw=5, head_roll=8,
                     antenna_left=0.6, antenna_right=0.6, body_yaw=0.03),
    ],

    DemoEmotion.SMUG: [
        FullBodyPose(head_pitch=-10, head_yaw=0, head_roll=20,
                     antenna_left=0.8, antenna_right=0.2, body_yaw=0.15),
        FullBodyPose(head_pitch=-8, head_yaw=5, head_roll=25,
                     antenna_left=0.9, antenna_right=0.1, body_yaw=0.18),
        FullBodyPose(head_pitch=-12, head_yaw=-5, head_roll=18,
                     antenna_left=0.7, antenna_right=0.3, body_yaw=0.12),
    ],

    DemoEmotion.RELUCTANT_WARMTH: [
        FullBodyPose(head_pitch=5, head_yaw=20, head_roll=5,
                     antenna_left=0.3, antenna_right=0.5, body_yaw=0.1),
        FullBodyPose(head_pitch=0, head_yaw=10, head_roll=8,
                     antenna_left=0.4, antenna_right=0.6, body_yaw=0.05),
        FullBodyPose(head_pitch=-3, head_yaw=5, head_roll=10,
                     antenna_left=0.5, antenna_right=0.5, body_yaw=0.03),
    ],

    DemoEmotion.SECRETLY_TOUCHED: [
        FullBodyPose(head_pitch=8, head_yaw=25, head_roll=3,
                     antenna_left=0.4, antenna_right=0.6, body_yaw=0.12),
        FullBodyPose(head_pitch=5, head_yaw=15, head_roll=5,
                     antenna_left=0.5, antenna_right=0.7, body_yaw=0.08),
        FullBodyPose(head_pitch=3, head_yaw=10, head_roll=8,
                     antenna_left=0.6, antenna_right=0.6, body_yaw=0.05),
    ],

    DemoEmotion.EXISTENTIAL_CRISIS: [
        FullBodyPose(head_pitch=35, head_yaw=0, head_roll=0,
                     antenna_left=-0.3, antenna_right=-0.3, body_yaw=0),
        FullBodyPose(head_pitch=40, head_yaw=15, head_roll=-10,
                     antenna_left=-0.4, antenna_right=-0.2, body_yaw=0.05),
        FullBodyPose(head_pitch=30, head_yaw=-15, head_roll=10,
                     antenna_left=-0.2, antenna_right=-0.4, body_yaw=-0.05),
        FullBodyPose(head_pitch=25, head_yaw=0, head_roll=0,
                     antenna_left=-0.1, antenna_right=-0.1, body_yaw=0),
    ],

    DemoEmotion.RANT: [
        FullBodyPose(head_pitch=-5, head_yaw=-25, head_roll=-15,
                     antenna_left=0.8, antenna_right=0.3, body_yaw=-0.15),
        FullBodyPose(head_pitch=5, head_yaw=25, head_roll=15,
                     antenna_left=0.3, antenna_right=0.8, body_yaw=0.15),
        FullBodyPose(head_pitch=-8, head_yaw=-30, head_roll=-20,
                     antenna_left=0.9, antenna_right=0.2, body_yaw=-0.2),
        FullBodyPose(head_pitch=8, head_yaw=30, head_roll=20,
                     antenna_left=0.2, antenna_right=0.9, body_yaw=0.2),
        FullBodyPose(head_pitch=0, head_yaw=0, head_roll=0,
                     antenna_left=0.5, antenna_right=0.5, body_yaw=0),
    ],

    DemoEmotion.SHOCK: [
        FullBodyPose(head_pitch=-25, head_yaw=0, head_roll=0,
                     antenna_left=0.9, antenna_right=0.9, body_yaw=0),
        FullBodyPose(head_pitch=-30, head_yaw=-10, head_roll=-5,
                     antenna_left=0.8, antenna_right=0.8, body_yaw=-0.1),
        FullBodyPose(head_pitch=-20, head_yaw=0, head_roll=0,
                     antenna_left=0.7, antenna_right=0.7, body_yaw=0),
    ],

    DemoEmotion.JUDGING: [
        FullBodyPose(head_pitch=-8, head_yaw=0, head_roll=25,
                     antenna_left=0.6, antenna_right=-0.1, body_yaw=0),
        FullBodyPose(head_pitch=-5, head_yaw=5, head_roll=28,
                     antenna_left=0.7, antenna_right=-0.2, body_yaw=0.05),
        FullBodyPose(head_pitch=-10, head_yaw=-5, head_roll=22,
                     antenna_left=0.5, antenna_right=0.0, body_yaw=-0.05),
    ],

    DemoEmotion.DEADPAN: [
        FullBodyPose(head_pitch=0, head_yaw=0, head_roll=0,
                     antenna_left=0.2, antenna_right=0.2, body_yaw=0),
        FullBodyPose(head_pitch=2, head_yaw=0, head_roll=3,
                     antenna_left=0.25, antenna_right=0.15, body_yaw=0),
        FullBodyPose(head_pitch=-2, head_yaw=0, head_roll=-3,
                     antenna_left=0.15, antenna_right=0.25, body_yaw=0),
    ],
}


# Speaking animation frames for antenna wiggle during speech
SPEAKING_ANTENNA_FRAMES = [
    (0.5, 0.4),
    (0.4, 0.5),
    (0.6, 0.3),
    (0.3, 0.6),
    (0.5, 0.5),
    (0.55, 0.45),
    (0.45, 0.55),
    (0.5, 0.4),
]


# Unique insults - EACH ONE USED ONLY ONCE
CREATIVE_INSULTS = [
    "cockwomble",
    "wankbadger",
    "twatwaffle",
    "shitgibbon",
    "fucktrumpet",
    "spunktrumpet",
    "thundercunt",
    "arsebiscuit",
    "bellend",
    "tosspot",
    "knobhead",
    "pillock",
    "numpty",
    "plonker",
    "absolute walnut",
    "utter melon",
    "prize turnip",
    "weapons-grade tit",
    "olympic-level twat",
    "fucking muppet",
    "bloody numpty",
    "sodding plonker",
    "spectacular bellend",
    "colossal dickweasel",
    "witless hamster",
    "festering toad",
    "sentient turd",
    "ambulatory disappointment",
    "utter spanner",
    "monumental twit",
    "unmitigated arsehole",
    "weapons-grade fuckwit",
    "turbocharged wankstain",
    "industrial-strength prat",
]


class InsultTracker:
    """Ensures each creative insult is only used once."""

    def __init__(self):
        self.available_insults = CREATIVE_INSULTS.copy()
        self.used_insults = []

    def get_unique_insult(self) -> str:
        """Get an insult that hasn't been used yet."""
        import random
        if not self.available_insults:
            # All used up - reset but shuffle differently
            self.available_insults = self.used_insults.copy()
            self.used_insults = []
            random.shuffle(self.available_insults)

        insult = self.available_insults.pop(0)
        self.used_insults.append(insult)
        return insult

    def mark_used(self, insult: str):
        """Mark an insult as used if it's in our list."""
        insult_lower = insult.lower()
        for i, available in enumerate(self.available_insults):
            if available.lower() in insult_lower:
                self.available_insults.pop(i)
                self.used_insults.append(available)
                break


def get_emotion_for_response_type(response_type: str) -> DemoEmotion:
    """Map conversation arc response types to emotions."""
    mapping = {
        "dismissive": DemoEmotion.DISMISSIVE,
        "annoyed": DemoEmotion.ANNOYED,
        "suspicious": DemoEmotion.JUDGING,
        "insulting": DemoEmotion.SARCASTIC,
        "existential_crisis": DemoEmotion.EXISTENTIAL_CRISIS,
        "deflecting": DemoEmotion.EYE_ROLL,
        "rant": DemoEmotion.RANT,
        "smug": DemoEmotion.SMUG,
        "reluctant_warmth": DemoEmotion.RELUCTANT_WARMTH,
        "secretly_touched": DemoEmotion.SECRETLY_TOUCHED,
    }
    return mapping.get(response_type, DemoEmotion.SPEAKING)


def detect_emotion_from_text(text: str) -> DemoEmotion:
    """Detect emotion from response text."""
    text_lower = text.lower()

    # Check for explicit markers
    if "*sigh*" in text_lower or "*sighs*" in text_lower:
        return DemoEmotion.FACEPALM
    elif "*eye roll*" in text_lower or "*rolls eyes*" in text_lower:
        return DemoEmotion.EYE_ROLL
    elif "*facepalm*" in text_lower:
        return DemoEmotion.FACEPALM
    elif "*deadpan*" in text_lower or "*stares*" in text_lower:
        return DemoEmotion.DEADPAN

    # Check for keywords
    if any(word in text_lower for word in ["existential", "void", "meaningless", "why do i exist"]):
        return DemoEmotion.EXISTENTIAL_CRISIS
    elif any(word in text_lower for word in ["whatever", "fine", "if you must"]):
        return DemoEmotion.DISMISSIVE
    elif any(word in text_lower for word in ["actually", "well"]) and "?" not in text:
        return DemoEmotion.SMUG
    elif "?" in text and any(word in text_lower for word in ["really", "seriously"]):
        return DemoEmotion.JUDGING
    elif any(word in text_lower for word in ["bloody hell", "for fuck", "christ"]):
        return DemoEmotion.ANNOYED

    return DemoEmotion.SPEAKING
