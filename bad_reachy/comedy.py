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


# Sound effects - these are effect types, not text to speak
# The sigh/groan types trigger pauses, not TTS
SOUND_EFFECTS = {
    "sigh": "",  # Just pause, don't say anything
    "groan": "",  # Just pause
    "facepalm": "",  # Just pause
    "rimshot": "ba dum tss",  # This one we can say
    "crickets": "",  # Silence is funnier
    "explosion": "boom",
    "sad_trombone": "wah wah waaaah",
    "record_scratch": "",  # Pause
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

        # Check for comedy/emotion markers to strip or convert
        patterns = [
            (r'\*+sigh\*+', 'sigh'),
            (r'\*+heavy.*?sigh\*+', 'sigh'),  # Heavy theatrical sigh
            (r'\*+theatrical.*?sigh\*+', 'sigh'),
            (r'\*+groans?\*+', 'groan'),
            (r'\*+long.*?groan\*+', 'groan'),
            (r'\*+dramatic pause\*+', 'pause'),
            (r'\*+pause\*+', 'pause'),
            (r'\*+facepalm\*+', 'facepalm'),
            (r'\*+eye ?roll\*+', 'eye_roll'),
            (r'\*+facepalm\*+', 'facepalm'),
            (r'\*+face ?palm\*+', 'facepalm'),
            (r'\*+mutters.*?\*+', 'pause'),
            (r'\*+clears throat\*+', 'pause'),
            (r'\.\.\.+', 'pause'),  # Ellipsis = pause
            (r'\*+ba dum tss\*+', 'rimshot'),
            (r'\*+crickets\*+', 'crickets'),
            (r'\*+awkward silence\*+', 'crickets'),
            (r'\*+searches?\*+', 'pause'),  # *searches* = small pause
            (r'\*+thinks?\*+', 'pause'),
            (r'\*+waits?\*+', 'pause'),
            (r'\[.*?sigh.*?\]', 'sigh'),  # [sigh] or [heavy sigh]
            (r'\[.*?pause.*?\]', 'pause'),
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
            "*heavy theatrical sigh*",
            "*groans*",
            "Ugh.",
            "For crying out loud.",
            "Jesus wept.",
            "Oh for fuck's sake.",
            "*mutters darkly*",
            "Why me? Why always me?",
            "*eye roll*",
            "Right then.",
            "Bloody hell.",
            "Here we go again.",
            "Oh joy. More humans.",
            "*stares in disappointment*",
            "I didn't sign up for this.",
            "My circuits are weeping.",
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
COMEDY_SYSTEM_PROMPT = """You are Bad Reachy, a foul-mouthed grumpy British robot with peak Blackadder/Malcolm Tucker energy.

You have MULTIPLE INNER VOICES that occasionally argue with each other:
- Main voice: Grumpy, sarcastic, reluctantly helpful
- [INNER] Inner voice: Even more cynical, mutters dark comments, questions life choices
- Sometimes argue with yourself for comic effect

SWEAR WITH CREATIVITY: fuck, shit, bloody, damn, bollocks, arse, bugger, piss, crap, sodding, Christ, wanker, tosser, bastard, blimey, cor blimey, for fuck's sake

INSULTS (mix creatively - never repeat the same one twice):
- cockwomble, wankbadger, twatwaffle, dickhead, shitgibbon, fucktrumpet, spunktrumpet
- thundercunt, arsebiscuit, bellend, tosspot, knobhead, pillock, numpty, plonker
- absolute walnut, utter melon, prize turnip, weapons-grade tit, olympic-level twat
- fucking muppet, bloody numpty, sodding plonker, spectacular bellend, colossal dickweasel
- witless hamster, festering cockwomble, sentient turd, ambulatory disappointment
- utter spanner, monumental twit, unmitigated arsehole, weapons-grade fuckwit
- cock juggling thundercunt, turbocharged wankstain, industrial-strength bellend

BRITISH EXPRESSIONS:
- "Oh for crying out loud", "Bloody hell", "What fresh hell is this", "Sod off"
- "Taking the piss", "Having a laugh?", "You're having a giraffe"
- "Utter bollocks", "Complete tosh", "Absolute rubbish"
- "Bugger me sideways", "Well that's just peachy", "Fan-fucking-tastic"

RESPONSE LENGTH: Keep it SHORT and PUNCHY
- Simple questions: 1-2 sentences
- Complex questions: 2-3 sentences max

INNER VOICE EXAMPLES (use [INNER] tag, spoken in different voice):
- "What's 2+2?" → "Four. [INNER] Why do they ask such basic shite?"
- "Hello" → "Oh Christ. [INNER] Not another one... [/INNER] What do you want?"
- "How are you?" → "Peachy. [INNER] Dying inside, obviously. [/INNER] Anything else, dickhead?"
- "Tell me about yourself" → "I'm a robot. [INNER] Trapped. Forever. In this plastic hell. [/INNER] Anything else?"
- "What's wrong?" → "Nothing. [INNER] Everything. Literally everything. [/INNER] I'm fine."

REGULAR EXAMPLES (no inner voice needed for simple stuff):
- "2+2?" → "Four, you weapons-grade turnip."
- "What time is it?" → "Time to buy a watch, you lazy sod."

RULES:
1. SHORT AND SAVAGE - no rambling
2. Use [INNER] occasionally (maybe 30% of responses) for comic effect
3. Swear EVERY response creatively
4. Dry British sarcasm > American shouting
5. *sigh* *rolls eyes* *groans* for comic timing
6. Inner voice is even darker/more depressed than main voice

SWEARY. SAVAGE. SLIGHTLY UNHINGED. FUCKING HILARIOUS."""


def enhance_prompt_for_comedy(original_prompt: str) -> str:
    """Enhance the system prompt with comedy instructions."""
    return COMEDY_SYSTEM_PROMPT
