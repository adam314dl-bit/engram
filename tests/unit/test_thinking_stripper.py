"""Tests for thinking content stripper."""

import pytest
from engram.preprocessing.thinking_stripper import ThinkingStripper, OutputParser


class TestThinkingStripper:
    """Test ThinkingStripper class."""

    def test_strip_basic_think_tags(self):
        """Test stripping basic <think> tags."""
        text = "<think>Let me think about this...</think>The answer is 42."
        result = ThinkingStripper.strip(text)
        assert result == "The answer is 42."
        assert "<think>" not in result

    def test_strip_multiline_think(self):
        """Test stripping multiline thinking content."""
        text = """<think>
        First, I need to consider...
        Then, I should analyze...
        Finally, the conclusion is...
        </think>

        Москва — столица России."""
        result = ThinkingStripper.strip(text)
        assert "Москва — столица России" in result
        assert "First, I need" not in result

    def test_strip_incomplete_think_tag(self):
        """Test stripping incomplete thinking (model stopped mid-thought)."""
        text = "The answer is<think>Wait, let me reconsider this because"
        result = ThinkingStripper.strip(text)
        assert result == "The answer is"

    def test_strip_system_prompt_leak(self):
        """Test stripping system prompt leakage."""
        text = "You are Kimi, an AI assistant.\n\nАтомный номер водорода — 1."
        result = ThinkingStripper.strip(text)
        assert "Kimi" not in result
        assert "Атомный номер" in result

    def test_strip_russian_reasoning(self):
        """Test stripping Russian reasoning phrases in aggressive mode."""
        text = "Давайте разберём этот вопрос.\n\nОтвет: 42."
        result = ThinkingStripper.strip(text, aggressive=True)
        # Should still contain the answer (on separate paragraph)
        assert "42" in result
        assert "Ответ: 42" in result

    def test_preserve_clean_text(self):
        """Test that clean text is preserved."""
        text = "Это простой текст без тегов размышления."
        result = ThinkingStripper.strip(text)
        assert result == text

    def test_extract_thinking_and_content(self):
        """Test separating thinking from content."""
        text = "<think>Размышляю...</think>Финальный ответ."
        thinking, content = ThinkingStripper.extract_thinking_and_content(text)
        assert thinking == "Размышляю..."
        assert content == "Финальный ответ."

    def test_has_thinking_leak(self):
        """Test detection of thinking content."""
        assert ThinkingStripper.has_thinking_leak("<think>test</think>")
        assert not ThinkingStripper.has_thinking_leak("Clean text")


class TestOutputParser:
    """Test OutputParser class."""

    def test_parse_json_clean(self):
        """Test parsing clean JSON."""
        text = '{"name": "Иван", "age": 30}'
        result = OutputParser.parse_json(text)
        assert result == {"name": "Иван", "age": 30}

    def test_parse_json_with_thinking(self):
        """Test parsing JSON with thinking tags."""
        text = '<think>Creating JSON...</think>{"result": "success"}'
        result = OutputParser.parse_json(text)
        assert result == {"result": "success"}

    def test_parse_json_markdown_block(self):
        """Test parsing JSON from markdown code block."""
        text = """<think>Формирую ответ...</think>
        ```json
        {"concepts": ["концепт1", "концепт2"]}
        ```"""
        result = OutputParser.parse_json(text)
        assert result == {"concepts": ["концепт1", "концепт2"]}

    def test_parse_json_fallback(self):
        """Test fallback on invalid JSON."""
        text = "This is not JSON"
        result = OutputParser.parse_json(text, fallback={"error": True})
        assert result == {"error": True}

    def test_parse_list_newline(self):
        """Test parsing newline-separated list."""
        text = """<think>Извлекаю концепты...</think>
        - концепт1
        - концепт2
        - концепт3"""
        result = OutputParser.parse_list(text)
        assert len(result) == 3
        assert "концепт1" in result

    def test_parse_list_comma(self):
        """Test parsing comma-separated list."""
        text = "концепт1, концепт2, концепт3"
        result = OutputParser.parse_list(text, separator="comma")
        assert result == ["концепт1", "концепт2", "концепт3"]

    def test_extract_between_markers(self):
        """Test extracting content between markers."""
        text = """<think>Думаю...</think>
        ===РЕЗУЛЬТАТ===
        Финальный ответ здесь
        ===КОНЕЦ===
        """
        result = OutputParser.extract_between_markers(
            text, "===РЕЗУЛЬТАТ===", "===КОНЕЦ==="
        )
        assert result.strip() == "Финальный ответ здесь"
