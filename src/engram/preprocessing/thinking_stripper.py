"""
Thinking/reasoning content stripper for Kimi K2 Thinking model.

Handles:
- <think>...</think> tags
- Incomplete thinking tags (model stopped mid-thought)
- System prompt leakage
- Reasoning phrases in Russian and English
"""

import re
from typing import Optional, Tuple, Any
import json
import logging

logger = logging.getLogger(__name__)


class ThinkingStripper:
    """
    Strips thinking/reasoning content from LLM output.
    Designed for Kimi K2 Thinking model with Russian content.
    """

    # Primary thinking tag patterns
    THINKING_PATTERNS = [
        # Standard tags (greedy to handle nested)
        re.compile(r'<think>[\s\S]*?</think>', re.DOTALL),
        re.compile(r'<thinking>[\s\S]*?</thinking>', re.DOTALL),

        # Kimi-specific tokens
        re.compile(r'<\|im_thinking\|>[\s\S]*?<\|im_end\|>', re.DOTALL),

        # Incomplete tags (model stopped mid-thought)
        re.compile(r'<think>[\s\S]*$', re.DOTALL),
        re.compile(r'<thinking>[\s\S]*$', re.DOTALL),
        re.compile(r'<\|im_thinking\|>[\s\S]*$', re.DOTALL),
    ]

    # System prompt leak patterns
    SYSTEM_LEAK_PATTERNS = [
        re.compile(r'You are Kimi[\s\S]*?(?=\n\n|\Z)', re.DOTALL),
        re.compile(r'<\|im_system\|>[\s\S]*?<\|im_end\|>', re.DOTALL),
        re.compile(r'<\|im_start\|>system[\s\S]*?<\|im_end\|>', re.DOTALL),
    ]

    # Reasoning phrases (Russian + English) - for aggressive mode
    REASONING_PHRASES_RU = [
        re.compile(r'Дай(?:те)? (?:мне )?подумать[\s\S]*?(?=\n\n|\Z)', re.DOTALL | re.IGNORECASE),
        re.compile(r'Мне нужно (?:по)?думать[\s\S]*?(?=\n\n|\Z)', re.DOTALL | re.IGNORECASE),
        re.compile(r'Давай(?:те)? разберём[\s\S]*?(?=\n\n|\Z)', re.DOTALL | re.IGNORECASE),
        re.compile(r'Сначала (?:я )?(?:должен|нужно)[\s\S]*?(?=\n\n|\Z)', re.DOTALL | re.IGNORECASE),
        re.compile(r'Рассмотрим[\s\S]*?(?=\n\n|\Z)', re.DOTALL | re.IGNORECASE),
        re.compile(r'Анализируя[\s\S]*?(?=\n\n|\Z)', re.DOTALL | re.IGNORECASE),
    ]

    REASONING_PHRASES_EN = [
        re.compile(r'Let me think[\s\S]*?(?=\n\n|\Z)', re.DOTALL | re.IGNORECASE),
        re.compile(r'I need to think[\s\S]*?(?=\n\n|\Z)', re.DOTALL | re.IGNORECASE),
        re.compile(r"Let's break (?:this )?down[\s\S]*?(?=\n\n|\Z)", re.DOTALL | re.IGNORECASE),
        re.compile(r'First,? I (?:need to|should|must)[\s\S]*?(?=\n\n|\Z)', re.DOTALL | re.IGNORECASE),
    ]

    @classmethod
    def strip(
        cls,
        text: str,
        aggressive: bool = False,
        preserve_structure: bool = True
    ) -> str:
        """
        Strip thinking content from text.

        Args:
            text: Raw LLM output
            aggressive: Also strip reasoning-like phrases
            preserve_structure: Try to preserve paragraph structure

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        result = text

        # 1. Strip thinking tags (always)
        for pattern in cls.THINKING_PATTERNS:
            result = pattern.sub('', result)

        # 2. Strip system prompt leaks (always)
        for pattern in cls.SYSTEM_LEAK_PATTERNS:
            result = pattern.sub('', result)

        # 3. Aggressive mode: strip reasoning phrases
        if aggressive:
            for pattern in cls.REASONING_PHRASES_RU + cls.REASONING_PHRASES_EN:
                result = pattern.sub('', result)

        # 4. Clean up whitespace
        if preserve_structure:
            # Keep paragraph breaks but normalize
            result = re.sub(r'\n{3,}', '\n\n', result)
            result = re.sub(r'[ \t]+', ' ', result)
        else:
            # Collapse all whitespace
            result = re.sub(r'\s+', ' ', result)

        return result.strip()

    @classmethod
    def extract_thinking_and_content(cls, text: str) -> Tuple[Optional[str], str]:
        """
        Separate thinking from content (useful for logging/debugging).

        Returns:
            Tuple of (thinking_content, main_content)
        """
        # Try to extract thinking
        think_match = re.search(r'<think>([\s\S]*?)</think>', text, re.DOTALL)

        if think_match:
            thinking = think_match.group(1).strip()
            content = cls.strip(text)
            return thinking, content

        return None, cls.strip(text)

    @classmethod
    def has_thinking_leak(cls, text: str) -> bool:
        """Check if text contains thinking content."""
        if not text:
            return False

        # Check for tags
        for pattern in cls.THINKING_PATTERNS[:3]:  # Only complete tags
            if pattern.search(text):
                return True

        return False


class OutputParser:
    """
    Robust output parser with thinking removal and type coercion.
    Handles Russian content and various output formats.
    """

    @staticmethod
    def parse_json(
        raw_output: str,
        fallback: Any = None,
        strip_thinking: bool = True
    ) -> Any:
        """
        Parse JSON from LLM output.

        Handles:
        - Thinking tags before/around JSON
        - Markdown code blocks
        - Partial JSON recovery
        """
        if not raw_output:
            return fallback

        text = raw_output

        # Step 1: Strip thinking
        if strip_thinking:
            text = ThinkingStripper.strip(text, aggressive=True)

        if not text:
            return fallback

        # Step 2: Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Step 3: Extract from markdown code block
        code_patterns = [
            re.compile(r'```json\s*([\s\S]*?)\s*```', re.DOTALL),
            re.compile(r'```\s*([\s\S]*?)\s*```', re.DOTALL),
        ]

        for pattern in code_patterns:
            match = pattern.search(text)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue

        # Step 4: Find JSON object or array
        json_patterns = [
            re.compile(r'(\{[\s\S]*\})', re.DOTALL),
            re.compile(r'(\[[\s\S]*\])', re.DOTALL),
        ]

        for pattern in json_patterns:
            match = pattern.search(text)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue

        logger.warning(f"Failed to parse JSON from output: {text[:200]}...")
        return fallback

    @staticmethod
    def parse_list(
        raw_output: str,
        separator: str = "auto",
        strip_thinking: bool = True
    ) -> list:
        """
        Parse list from LLM output.

        Args:
            raw_output: Raw LLM output
            separator: "auto", "newline", "comma", or custom
            strip_thinking: Whether to strip thinking tags
        """
        if not raw_output:
            return []

        text = raw_output

        if strip_thinking:
            text = ThinkingStripper.strip(text, aggressive=True)

        if not text:
            return []

        # Auto-detect separator
        if separator == "auto":
            if '\n' in text and text.count('\n') >= text.count(','):
                separator = "newline"
            else:
                separator = "comma"

        # Split
        if separator == "newline":
            items = text.split('\n')
        elif separator == "comma":
            items = text.split(',')
        else:
            items = text.split(separator)

        # Clean items
        cleaned = []
        for item in items:
            # Remove bullet points, numbers, dashes
            item = re.sub(r'^[\s\-\*\•\d\.]+', '', item)
            item = item.strip()
            if item:
                cleaned.append(item)

        return cleaned

    @staticmethod
    def parse_key_value(
        raw_output: str,
        strip_thinking: bool = True
    ) -> dict:
        """
        Parse key-value pairs from LLM output.

        Handles formats like:
        - key: value
        - key = value
        - **key**: value
        """
        if not raw_output:
            return {}

        text = raw_output

        if strip_thinking:
            text = ThinkingStripper.strip(text, aggressive=True)

        result = {}

        # Pattern for key-value pairs
        patterns = [
            re.compile(r'\*\*([^*]+)\*\*\s*[:=]\s*(.+?)(?=\n|$)', re.MULTILINE),
            re.compile(r'^([^:\n]+)\s*:\s*(.+?)$', re.MULTILINE),
            re.compile(r'^([^=\n]+)\s*=\s*(.+?)$', re.MULTILINE),
        ]

        for pattern in patterns:
            for match in pattern.finditer(text):
                key = match.group(1).strip()
                value = match.group(2).strip()
                if key and value:
                    result[key] = value

        return result

    @staticmethod
    def extract_between_markers(
        text: str,
        start_marker: str,
        end_marker: str,
        strip_thinking: bool = True
    ) -> Optional[str]:
        """Extract content between custom markers."""
        if strip_thinking:
            text = ThinkingStripper.strip(text)

        pattern = re.compile(
            re.escape(start_marker) + r'\s*([\s\S]*?)\s*' + re.escape(end_marker),
            re.DOTALL
        )

        match = pattern.search(text)
        return match.group(1) if match else None
