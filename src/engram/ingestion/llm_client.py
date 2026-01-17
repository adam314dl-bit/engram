"""LLM client for OpenAI-compatible endpoints (Ollama, vLLM, etc.)."""

import asyncio
import json
import logging
from typing import Any

import httpx

from engram.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Async client for OpenAI-compatible LLM APIs."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        max_concurrent: int | None = None,
    ) -> None:
        self.base_url = (base_url or settings.llm_base_url).rstrip("/")
        self.model = model or settings.llm_model
        self.timeout = timeout or settings.llm_timeout
        self.max_concurrent = max_concurrent or settings.llm_max_concurrent

        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

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
            client = await self._get_client()

            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs,
            }

            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]

            except httpx.HTTPStatusError as e:
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
            # Try to extract JSON from response
            import re
            json_match = re.search(r"\{[\s\S]*\}", cleaned)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
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
