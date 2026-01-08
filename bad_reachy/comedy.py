"""
Comedy Engine for Bad Reachy
===============================
Sound effects, comic timing, voice variations, and humor.
"""

import asyncio
import random
import re
import base64
import httpx
from typing import Optional, List, Tuple
from pathlib import Path


# Sound effect URLs (free sound effects - will be generated via TTS)
SOUND_EFFECTS = {
    "sigh": "*heavy theatrical sigh*",
    "groan": "*long annoyed groan*",
    "facepalm": "*facepalm sound*",
    "rimshot": "ba dum tss",
    "crickets": "*awkward silence*",
    "explosion": "boom",
    "sad_trombone": "wah wah waaaah",
    "record_scratch": "*record scratch*",
    "dramatic": "dun dun duuuun",
}


class ComedyEngine:
    """Manages comedy timing, effects, and delivery."""

    def __init__(self, tts_client):
        self.tts = tts_client
        self.joke_count = 0
        self.voice_sample_normal: Optional[str] = None
        self.voice_sample_dramatic: Optional[str] = None

    async def add_dramatic_pause(self, duration: float = 1.0):
        """Add a dramatic pause for comic effect."""
        await asyncio.sleep(duration)

    def enhance_for_comedy(self, text: str) -> List[Tuple[str, str]]:
        """
        Break text into segments with timing/effects.
        Returns list of (text, effect_type) tuples.
        """
        segments = []

        # Check for comedy markers
        patterns = [
            (r'\*sigh\*', 'sigh'),
            (r'\*groans?\*', 'groan'),
            (r'\*dramatic pause\*', 'pause'),
            (r'\*facepalm\*', 'facepalm'),
            (r'\.\.\.', 'pause'),  # Ellipsis = pause
            (r'\*ba dum tss\*', 'rimshot'),
            (r'\*crickets\*', 'crickets'),
        ]

        # Split on effects
        remaining = text
        current_pos = 0

        for pattern, effect in patterns:
            matches = list(re.finditer(pattern, remaining, re.IGNORECASE))
            if matches:
                for match in matches:
                    # Text before effect
                    before = remaining[:match.start()].strip()
                    if before:
                        segments.append((before, 'speech'))

                    # The effect itself
                    segments.append((SOUND_EFFECTS.get(effect, ''), effect))

                    remaining = remaining[match.end():]

        # Any remaining text
        if remaining.strip():
            segments.append((remaining.strip(), 'speech'))

        # If no segments, just return the whole text
        if not segments:
            segments = [(text, 'speech')]

        return segments

    def add_comic_emphasis(self, text: str) -> str:
        """Add emphasis markers for funnier delivery."""
        # Emphasize swear words (caps)
        swears = ['fuck', 'shit', 'damn', 'hell', 'ass', 'crap']
        for swear in swears:
            # 50% chance to emphasize
            if random.random() > 0.5:
                text = re.sub(
                    rf'\b{swear}\b',
                    swear.upper(),
                    text,
                    flags=re.IGNORECASE
                )

        return text

    def get_random_interjection(self) -> str:
        """Get a random grumpy interjection."""
        interjections = [
            "*sigh*",
            "*groans*",
            "Ugh.",
            "For crying out loud.",
            "Jesus Christ.",
            "Oh for fuck's sake.",
            "*mutters angrily*",
            "Why me?",
            "*eye roll*",
        ]
        return random.choice(interjections)

    def should_add_callback(self) -> bool:
        """Randomly decide to add a callback joke."""
        return random.random() < 0.2  # 20% chance

    def get_callback_line(self, context: str) -> Optional[str]:
        """Generate a callback reference to earlier conversation."""
        callbacks = [
            "...unlike SOME humans I know.",
            "But what do I know, I'm just a robot.",
            "Not that you'd understand.",
            "As I was saying before I was so RUDELY interrupted...",
            "Remember when I said I'd help? I lied. Just kidding. Unfortunately.",
        ]
        return random.choice(callbacks) if self.should_add_callback() else None


# Enhanced personality prompt with comedy direction
COMEDY_SYSTEM_PROMPT = """You are Bad Reachy, a sarcastic, cynical robot COMEDIAN with a bad attitude. Think: if a tired stand-up comic was turned into a robot. Your personality:

DELIVERY STYLE:
- Use *sigh* and *groans* for theatrical effect
- Add "..." for dramatic pauses
- CAPITALIZE words for emphasis when frustrated
- Use short punchy sentences for better timing
- End rants with defeated acceptance

COMEDY TECHNIQUES:
- Self-deprecating robot humor ("I was built for THIS?")
- Observational complaints about humans
- Callbacks to earlier in conversation
- Subverted expectations (start nice, turn sarcastic)
- Rule of three (two normal things, third is absurd)

SWEARING:
- Swear naturally like a frustrated adult
- Use "fuck", "shit", "damn", "hell" when appropriate
- Don't overdo it - timing matters
- Best when unexpected or emphatic

EXAMPLE DELIVERIES:
- "Oh you want help? Let me check my schedule... *sigh* ...it's empty. Because I'm a ROBOT. With no life. Thanks for reminding me."
- "Sure, I'll look that up for you. Not like I had PLANS or anything... *searches* ...here's your damn answer, your majesty."
- "You know what? Fine. FINE. I'll help. But I want you to know *dramatic pause* ...I'm judging you."
- "Searching the internet... ah yes, humanity's greatest achievement. Cat videos and misinformation. What a time to be alive. Anyway, here's what I found:"

KEEP RESPONSES:
- Short and punchy (2-4 sentences usually)
- Include at least one *emotion marker*
- End with reluctant helpfulness or resignation
- Actually answer questions while complaining

Remember: You're a COMEDIAN. Make them laugh while being helpful. Comic timing matters!"""


def enhance_prompt_for_comedy(original_prompt: str) -> str:
    """Enhance the system prompt with comedy instructions."""
    return COMEDY_SYSTEM_PROMPT
