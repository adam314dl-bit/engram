"""LLM client for OpenAI-compatible endpoints (Ollama, vLLM, etc.).

FIXED version - simplified thinking handling. The --reasoning-parser kimi_k2 flag
handles separation automatically at vLLM level. This client just needs to:
1. Accept content from normal field
2. Fall back to reasoning field if content is empty
3. Strip any remaining thinking tags via ThinkingStripper
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional

import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

from engram.config import settings
from engram.preprocessing.thinking_stripper import ThinkingStripper, OutputParser

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Structured LLM response."""
    content: str                      # Cleaned content (thinking removed)
    raw_content: str                  # Original content
    thinking: Optional[str] = None    # Extracted thinking (if any)
    usage: Optional[dict] = None      # Token usage
    finish_reason: Optional[str] = None


def strip_thinking_tags(text: str) -> str:
    """Remove thinking/reasoning tags from model output.

    Uses ThinkingStripper for comprehensive cleaning.
    """
    return ThinkingStripper.strip(text, aggressive=False)


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
        return_response: bool = False,
        aggressive_strip: bool = False,
        **kwargs: Any,
    ) -> str | LLMResponse:
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
        raw_content = message.get("content")

        # Kimi K2 thinking model uses "reasoning" or "reasoning_content" field
        reasoning_content = message.get("reasoning_content") or message.get("reasoning")

        # Some APIs use "text" instead of "content"
        if raw_content is None:
            raw_content = choices[0].get("text")

        # FALLBACK: If content is empty but reasoning has data
        # This happens when model puts answer in thinking tags due to anti-leakage prompts
        # The --reasoning-parser kimi_k2 flag handles separation, so we can use reasoning directly
        if raw_content is None and reasoning_content:
            logger.warning("Content empty, using reasoning field as content")
            # Use reasoning as content - ThinkingStripper will clean it
            raw_content = reasoning_content
            reasoning_content = None  # Don't double-process

        if raw_content is None:
            logger.error(f"LLM returned None content. Full response: {data}")
            raise ValueError(f"LLM returned None content: {data}")

        # Extract thinking and clean content
        thinking, content = ThinkingStripper.extract_thinking_and_content(raw_content)
        content = ThinkingStripper.strip(content, aggressive=aggressive_strip)

        # Prefer vLLM-extracted reasoning if available
        if reasoning_content and not thinking:
            thinking = reasoning_content

        if return_response:
            return LLMResponse(
                content=content,
                raw_content=raw_content,
                thinking=thinking,
                usage=data.get("usage"),
                finish_reason=choices[0].get("finish_reason"),
            )

        return content

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 8192,
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
        max_tokens: int = 8192,
        return_response: bool = False,
        aggressive_strip: bool = False,
        **kwargs: Any,
    ) -> str | LLMResponse:
        """Send chat messages and get response."""
        async with self._semaphore:
            try:
                # Run sync request in thread pool to not block event loop
                response = await asyncio.to_thread(
                    self._sync_chat,
                    messages,
                    temperature,
                    max_tokens,
                    return_response,
                    aggressive_strip,
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
        max_tokens: int = 8192,
        fallback: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any] | list[Any] | Any:
        """Generate and parse JSON response using robust OutputParser."""
        response = await self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        result = OutputParser.parse_json(response, fallback=fallback)

        if result is None and fallback is None:
            logger.warning(f"Could not parse LLM response as JSON: {response[:200]}")
            raise ValueError(f"Could not parse LLM response as JSON: {response[:200]}")

        return result

    async def generate_list(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 8192,
        separator: str = "auto",
        **kwargs: Any,
    ) -> list[str]:
        """Generate and parse list response."""
        response = await self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        return OutputParser.parse_list(response, separator=separator)


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
