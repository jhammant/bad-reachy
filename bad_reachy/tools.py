"""
Tools for Bad Reachy
=======================
Web search and other capabilities the grumpy robot can use.
"""

import httpx
import re
import json
from typing import Optional, List, Dict
from urllib.parse import quote_plus


class WebSearchTool:
    """DuckDuckGo web search - no API key needed."""

    def __init__(self):
        self.search_url = "https://html.duckduckgo.com/html/"

    async def search(self, query: str, max_results: int = 3) -> List[Dict[str, str]]:
        """Search the web and return results."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    self.search_url,
                    data={"q": query},
                    headers={"User-Agent": "Mozilla/5.0 (compatible; BadReachy/1.0)"}
                )
                response.raise_for_status()

                # Parse results from HTML
                results = []
                html = response.text

                # Extract result snippets (simple parsing)
                # Look for result blocks
                result_pattern = r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>([^<]*)</a>'
                snippet_pattern = r'<a[^>]*class="result__snippet"[^>]*>([^<]*)</a>'

                titles = re.findall(result_pattern, html)
                snippets = re.findall(snippet_pattern, html)

                for i, (url, title) in enumerate(titles[:max_results]):
                    snippet = snippets[i] if i < len(snippets) else ""
                    results.append({
                        "title": title.strip(),
                        "url": url,
                        "snippet": snippet.strip()
                    })

                return results

        except Exception as e:
            print(f"[SEARCH] Error: {e}")
            return []

    def format_results(self, results: List[Dict[str, str]]) -> str:
        """Format search results for the LLM."""
        if not results:
            return "No search results found."

        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(f"{i}. {r['title']}: {r['snippet']}")

        return "\n".join(formatted)


class WebFetchTool:
    """Fetch and extract text from web pages."""

    async def fetch(self, url: str, max_chars: int = 2000) -> str:
        """Fetch a webpage and extract main text content."""
        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                response = await client.get(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (compatible; BadReachy/1.0)"}
                )
                response.raise_for_status()

                html = response.text

                # Simple text extraction - remove scripts, styles, tags
                # Remove script and style elements
                html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
                html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
                # Remove tags
                text = re.sub(r'<[^>]+>', ' ', html)
                # Clean whitespace
                text = re.sub(r'\s+', ' ', text).strip()

                return text[:max_chars]

        except Exception as e:
            print(f"[FETCH] Error: {e}")
            return f"Failed to fetch page: {str(e)}"


class ToolManager:
    """Manages tool detection and execution for the LLM."""

    def __init__(self):
        self.web_search = WebSearchTool()
        self.web_fetch = WebFetchTool()

    def needs_tool(self, user_message: str) -> Optional[str]:
        """Detect if a message needs a tool."""
        message_lower = user_message.lower()

        # Search triggers
        search_triggers = [
            "search for", "look up", "google", "find out",
            "what is the latest", "current", "news about",
            "who is", "what happened", "when did"
        ]

        # Check for search needs
        for trigger in search_triggers:
            if trigger in message_lower:
                return "search"

        # Check for specific factual questions that might need search
        if any(q in message_lower for q in ["what is", "who is", "when is", "where is"]):
            # But not for simple stuff the LLM knows
            if any(word in message_lower for word in ["weather", "news", "latest", "current", "today", "2024", "2025"]):
                return "search"

        return None

    def extract_search_query(self, message: str) -> str:
        """Extract the search query from a message."""
        # Remove common prefixes
        prefixes = [
            "search for", "look up", "google", "find out about",
            "search", "find", "what is the latest on", "news about"
        ]

        query = message.lower()
        for prefix in prefixes:
            if query.startswith(prefix):
                query = query[len(prefix):].strip()

        return query.strip()

    async def execute_search(self, query: str) -> str:
        """Execute a web search and return formatted results."""
        print(f"[TOOLS] Searching for: {query}")
        results = await self.web_search.search(query)
        return self.web_search.format_results(results)

    async def execute_fetch(self, url: str) -> str:
        """Fetch a webpage."""
        print(f"[TOOLS] Fetching: {url}")
        return await self.web_fetch.fetch(url)


# Enhanced system prompt with tool awareness
TOOL_AWARE_PROMPT_ADDITION = """

You have access to web search when needed. If the user asks about current events, news, or things you might not know, I'll search for you and provide the results.

When you receive search results, summarize them in your grumpy way - don't just read them out verbatim. Be sarcastic about having to look things up for the user.

Example: "Ugh, FINE, I looked it up for you since apparently you can't use Google yourself. Here's what I found..."
"""
