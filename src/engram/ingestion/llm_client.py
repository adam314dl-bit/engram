"""LLM client for OpenAI-compatible endpoints (Ollama, vLLM, etc.)."""

import asyncio
import json
import logging
import re
from typing import Any

import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

from engram.config import settings

logger = logging.getLogger(__name__)

# Thinking tag patterns for models like Kimi, GLM, DeepSeek, etc.
THINKING_PATTERNS = [
    re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE),
    re.compile(r'<thinking>.*?</thinking>', re.DOTALL | re.IGNORECASE),
    re.compile(r'<thought>.*?</thought>', re.DOTALL | re.IGNORECASE),
    re.compile(r'<reason>.*?</reason>', re.DOTALL | re.IGNORECASE),
    re.compile(r'<reasoning>.*?</reasoning>', re.DOTALL | re.IGNORECASE),
    re.compile(r'<reflection>.*?</reflection>', re.DOTALL | re.IGNORECASE),
    re.compile(r'<inner_monologue>.*?</inner_monologue>', re.DOTALL | re.IGNORECASE),
    # Orphan closing tags (without opening tag)
    re.compile(r'</think>', re.IGNORECASE),
    re.compile(r'</thinking>', re.IGNORECASE),
    re.compile(r'</thought>', re.IGNORECASE),
    re.compile(r'</reason>', re.IGNORECASE),
    re.compile(r'</reasoning>', re.IGNORECASE),
    re.compile(r'</reflection>', re.IGNORECASE),
]


def strip_thinking_tags(text: str) -> str:
    """Remove thinking/reasoning tags from model output."""
    result = text
    for pattern in THINKING_PATTERNS:
        result = pattern.sub('', result)
    # Clean up extra whitespace left after removal
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result.strip()


class LLMClient:
    """Async-wrapped client for OpenAI-compatible LLM APIs using requests."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        max_concurrent: int | None = None,
    ) -> None:
        self.base_url = (base_url or settings.llm_base_url).rstrip("/")
        self.model = model or settings.llm_model
        self.api_key = api_key or settings.llm_api_key
        self.timeout = timeout or settings.llm_timeout
        self.max_concurrent = max_concurrent or settings.llm_max_concurrent

        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        self._session: requests.Session | None = None

    def _get_session(self) -> requests.Session:
        """Get or create requests session with connection pooling."""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            })
            # Increase connection pool size for parallel requests
            adapter = HTTPAdapter(
                pool_connections=self.max_concurrent,
                pool_maxsize=self.max_concurrent * 2,
                max_retries=Retry(total=2, backoff_factor=0.5),
            )
            self._session.mount("http://", adapter)
            self._session.mount("https://", adapter)
        return self._session

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session is not None:
            self._session.close()
            self._session = None

    def _sync_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> str:
        """Synchronous chat request (runs in thread)."""
        session = self._get_session()

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }

        response = session.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        # Handle various response formats
        choices = data.get("choices", [])
        if not choices:
            logger.error(f"LLM returned empty choices: {data}")
            raise ValueError("LLM returned empty choices")

        message = choices[0].get("message", {})
        content = message.get("content")

        # Kimi K2 thinking model uses "reasoning" field
        if content is None:
            content = message.get("reasoning")

        # Some APIs use "text" instead of "content"
        if content is None:
            content = choices[0].get("text")

        if content is None:
            logger.error(f"LLM returned None content. Full response: {data}")
            raise ValueError(f"LLM returned None content: {data}")

        # Strip thinking tags from models like Kimi, DeepSeek, etc.
        return strip_thinking_tags(content)

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> str:
        """Generate a completion from the LLM."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return await self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> str:
        """Send chat messages and get response."""
        async with self._semaphore:
            try:
                # Run sync request in thread pool to not block event loop
                response = await asyncio.to_thread(
                    self._sync_chat,
                    messages,
                    temperature,
                    max_tokens,
                    **kwargs,
                )
                return response

            except requests.HTTPError as e:
                logger.error(f"LLM API error: {e.response.status_code} - {e.response.text}")
                raise
            except Exception as e:
                logger.error(f"LLM request failed: {e}")
                raise

    async def generate_json(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,  # Lower for structured output
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> dict[str, Any] | list[Any]:
        """Generate and parse JSON response."""
        response = await self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        # Clean response - remove markdown code blocks if present
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            logger.debug(f"Raw response: {response}")
            # Try to extract JSON object from response
            json_match = re.search(r"\{[\s\S]*\}", cleaned)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            # Try to extract JSON array from response
            array_match = re.search(r"\[[\s\S]*\]", cleaned)
            if array_match:
                try:
                    return json.loads(array_match.group())
                except json.JSONDecodeError:
                    pass
            logger.warning(f"Could not parse LLM response as JSON: {response[:200]}")
            raise ValueError(f"Could not parse LLM response as JSON: {response[:200]}")


# Global client instance
_llm_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Get or create the global LLM client."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


async def close_llm_client() -> None:
    """Close the global LLM client."""
    global _llm_client
    if _llm_client is not None:
        await _llm_client.close()
        _llm_client = None
