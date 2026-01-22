"""
Fast LLM using Groq API
========================
Groq has extremely fast inference - ~10x faster than local.
"""

import httpx
import os
import asyncio
from typing import List, Dict, Optional, AsyncIterator


class GroqLLM:
    """Groq LLM client - extremely fast cloud inference."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        # Model selection via env: GROQ_MODEL
        # Options: "llama-3.1-8b-instant" (FAST), "llama-3.3-70b-versatile" (FUNNIER)
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        default_model = "llama-3.1-8b-instant"  # Default: speed
        self.model = model or os.getenv("GROQ_MODEL", default_model)
        print(f"[LLM] Model: {self.model}")
        self.conversation_history: List[Dict[str, str]] = []
        self.system_prompt = ""

    async def chat(self, user_message: str) -> str:
        """Send message and get response."""
        if not self.api_key:
            return "Shit, no API key. Fix your config, genius."

        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        messages = [
            {"role": "system", "content": self.system_prompt}
        ] + self.conversation_history[-4:]  # Keep last 4 for max speed

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:  # Reduced timeout
                response = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": 1.2,  # Bit more creative/random
                        "max_tokens": 150,   # Allow 2-3 sentences
                        "stream": False
                    }
                )
                response.raise_for_status()
                data = response.json()

                assistant_message = data["choices"][0]["message"]["content"]

                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message
                })

                return assistant_message

        except httpx.TimeoutException:
            return "Fuck, my brain timed out. Ask again."
        except Exception as e:
            print(f"[LLM-Groq] Error: {e}")
            return "Shit broke. Probably your fault."

    def clear_history(self):
        self.conversation_history = []

    async def chat_stream(self, user_message: str) -> AsyncIterator[str]:
        """Stream response tokens as they arrive - MUCH snappier!"""
        if not self.api_key:
            yield "Shit, no API key."
            return

        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        messages = [
            {"role": "system", "content": self.system_prompt}
        ] + self.conversation_history[-4:]

        full_response = ""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream(
                    "POST",
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": 1.2,
                        "max_tokens": 150,
                        "stream": True
                    }
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            try:
                                import json
                                chunk = json.loads(data)
                                delta = chunk.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    full_response += content
                                    yield content
                            except:
                                pass

            self.conversation_history.append({
                "role": "assistant",
                "content": full_response
            })

        except Exception as e:
            print(f"[LLM-Groq] Stream error: {e}")
            yield "Shit broke."

    async def test_connection(self) -> bool:
        """Test if Groq API is reachable."""
        if not self.api_key:
            return False
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    "https://api.groq.com/openai/v1/models",
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )
                return response.status_code == 200
        except:
            return False
