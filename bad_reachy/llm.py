"""
LM Studio Client for Bad Reachy
==================================
Local LLM via LM Studio API (OpenAI compatible)
"""

import httpx
from typing import List, Dict, Optional
from .config import Config, GRUMPY_SYSTEM_PROMPT


class LMStudioClient:
    """Client for local LM Studio LLM."""

    def __init__(self, config: Config):
        self.config = config
        self.base_url = config.lm_studio_url
        self.model = config.lm_studio_model
        self.conversation_history: List[Dict[str, str]] = []

        # Initialize with system prompt
        self.system_prompt = GRUMPY_SYSTEM_PROMPT

    async def chat(self, user_message: str) -> str:
        """Send message and get grumpy response."""
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # Build messages with system prompt
        messages = [
            {"role": "system", "content": self.system_prompt}
        ] + self.conversation_history[-10:]  # Keep last 10 exchanges

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": 0.9,  # More creative/unpredictable
                        "max_tokens": 1500,  # Plenty for reasoning + response
                        "stream": False
                    }
                )
                response.raise_for_status()
                data = response.json()

                assistant_message = data["choices"][0]["message"]["content"]

                # Add to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message
                })

                return assistant_message

        except httpx.TimeoutException:
            return "*sigh* My brain is slow today. Ask again, if you must."
        except Exception as e:
            print(f"[LLM] Error: {e}")
            return "Ugh, something broke. Probably your fault somehow."

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []

    async def test_connection(self) -> bool:
        """Test if LM Studio is reachable."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/models")
                return response.status_code == 200
        except:
            return False
