"""Test set evaluation for incomplete/lazy human reference answers.

v4.2 Evaluation System - Designed for regression testing with partial annotations.

Key insight:
- Old evaluation: "Is Engram answer == Human answer?" (fails for incomplete references)
- New evaluation: "Is Human answer ⊆ Engram answer?" (subset/contained relationship)

Metrics:
- source_match: Does Engram use same source as human URL? (URL comparison)
- key_info_match: Does Engram CONTAIN key info from human answer? (LLM judge)
- relevance: Does Engram answer the question? (LLM judge)
- no_contradiction: No conflicts with human answer? (LLM judge)
- overall: Weighted combination by answer type

Usage:
    uv run python -m engram.evaluation.evaluator test_data.csv

Input CSV format (url column is optional):
    question,answer
    "Кто лид фронтенда?","Или"

    # Or with URL for source matching:
    question,answer,url
    "Кто лид фронтенда?","Или","https://confluence.company.com/..."
"""

import argparse
import csv
import json
import logging
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests

from engram.config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class EvaluationResult:
    """Result for a single question evaluation."""

    question: str
    human_answer: str
    human_url: str
    engram_answer: str
    engram_sources: list[str]
    answer_type: str  # url_only, short, full, empty

    # Individual metrics (0.0-1.0)
    source_match: float
    key_info_match: float
    relevance: float
    no_contradiction: float
    overall: float

    # Debug info
    reasoning: str = ""
    error: str = ""
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for CSV/JSON export."""
        return {
            "question": self.question,
            "human_answer": self.human_answer,
            "human_url": self.human_url,
            "engram_answer": self.engram_answer,
            "engram_sources": ";".join(self.engram_sources),
            "answer_type": self.answer_type,
            "source_match": self.source_match,
            "key_info_match": self.key_info_match,
            "relevance": self.relevance,
            "no_contradiction": self.no_contradiction,
            "overall": self.overall,
            "reasoning": self.reasoning,
            "error": self.error,
            "latency_ms": self.latency_ms,
        }


@dataclass
class EvaluationSummary:
    """Aggregate statistics for evaluation run."""

    total: int = 0
    successful: int = 0
    failed: int = 0

    # Average scores
    avg_source_match: float = 0.0
    avg_key_info_match: float = 0.0
    avg_relevance: float = 0.0
    avg_no_contradiction: float = 0.0
    avg_overall: float = 0.0

    # Scores by answer type
    by_type: dict[str, dict[str, float]] = field(default_factory=dict)

    # Timing
    total_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "total": self.total,
            "successful": self.successful,
            "failed": self.failed,
            "avg_source_match": round(self.avg_source_match, 3),
            "avg_key_info_match": round(self.avg_key_info_match, 3),
            "avg_relevance": round(self.avg_relevance, 3),
            "avg_no_contradiction": round(self.avg_no_contradiction, 3),
            "avg_overall": round(self.avg_overall, 3),
            "by_type": self.by_type,
            "total_latency_ms": round(self.total_latency_ms, 1),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
        }


# =============================================================================
# HTTP Clients
# =============================================================================


class ThreadSafeSession:
    """Thread-local requests.Session for parallel HTTP requests."""

    def __init__(self, timeout: float = 120.0) -> None:
        self._local = threading.local()
        self._timeout = timeout

    def get_session(self) -> requests.Session:
        """Get thread-local session."""
        if not hasattr(self._local, "session"):
            self._local.session = requests.Session()
            self._local.session.headers.update(
                {"Content-Type": "application/json"}
            )
        return self._local.session

    def post(
        self,
        url: str,
        json_data: dict,
        timeout: float | None = None,
    ) -> requests.Response:
        """POST request with thread-local session."""
        session = self.get_session()
        return session.post(
            url,
            json=json_data,
            timeout=timeout or self._timeout,
        )


class EngramClient:
    """Client for querying Engram API with source extraction."""

    def __init__(
        self,
        base_url: str = "http://localhost:7777/v1",
        model: str = "engram",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._http = ThreadSafeSession(timeout=timeout)

    def check_availability(self) -> tuple[bool, str]:
        """
        Check if Engram API is available.

        Returns:
            Tuple of (is_available, error_message)
        """
        # Try health endpoint first
        health_url = self.base_url.replace("/v1", "") + "/health"

        try:
            session = self._http.get_session()
            response = session.get(health_url, timeout=10)
            if response.status_code == 200:
                return True, ""
            return False, f"Engram health check returned status {response.status_code}"
        except requests.exceptions.ConnectionError:
            return False, f"Cannot connect to Engram at {self.base_url}"
        except requests.exceptions.Timeout:
            return False, f"Engram at {self.base_url} timed out"
        except Exception as e:
            return False, f"Engram check failed: {e}"

    def query(self, question: str) -> tuple[str, list[str]]:
        """
        Query Engram and extract answer and sources.

        Args:
            question: User question

        Returns:
            Tuple of (answer_text, list_of_source_urls)
        """
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": question}],
        }

        try:
            response = self._http.post(url, payload)
            response.raise_for_status()
            data = response.json()

            # Extract answer
            answer = ""
            if "choices" in data and data["choices"]:
                message = data["choices"][0].get("message", {})
                answer = message.get("content", "")

            # Extract sources
            sources = self._extract_sources(data, answer)

            return answer, sources

        except requests.RequestException as e:
            logger.error(f"Engram request failed: {e}")
            raise

    def _extract_sources(self, data: dict, answer: str) -> list[str]:
        """
        Extract source URLs from API response.

        Checks:
        1. Structured `sources_used` field (list of {title, url} objects)
        2. URLs in answer text (fallback)
        """
        sources: list[str] = []

        # 1. From structured sources_used field
        if "sources_used" in data:
            for src in data["sources_used"]:
                if isinstance(src, dict) and src.get("url"):
                    url = src["url"].strip().rstrip(".,;:")
                    if url:
                        sources.append(url)

        # 2. Fallback: Extract URLs from answer text
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\])(\']+'
        answer_urls = re.findall(url_pattern, answer)
        for url in answer_urls:
            url = url.strip().rstrip(".,;:")
            if url and url not in sources:
                sources.append(url)

        # Deduplicate while preserving order
        seen = set()
        unique_sources = []
        for s in sources:
            if s not in seen:
                seen.add(s)
                unique_sources.append(s)

        return unique_sources


class JudgeLLM:
    """LLM judge for semantic evaluation of answers."""

    # Weights by answer type
    WEIGHTS = {
        "url_only": {
            "source_match": 0.50,
            "key_info_match": 0.10,
            "relevance": 0.30,
            "no_contradiction": 0.10,
        },
        "short": {
            "source_match": 0.20,
            "key_info_match": 0.40,
            "relevance": 0.25,
            "no_contradiction": 0.15,
        },
        "short_no_url": {  # Short answer without URL - no source matching
            "source_match": 0.00,
            "key_info_match": 0.50,
            "relevance": 0.30,
            "no_contradiction": 0.20,
        },
        "full": {
            "source_match": 0.10,
            "key_info_match": 0.40,
            "relevance": 0.30,
            "no_contradiction": 0.20,
        },
        "full_no_url": {  # Full answer without URL - no source matching
            "source_match": 0.00,
            "key_info_match": 0.45,
            "relevance": 0.35,
            "no_contradiction": 0.20,
        },
        "empty": {
            "source_match": 0.00,
            "key_info_match": 0.00,
            "relevance": 0.70,
            "no_contradiction": 0.30,
        },
    }

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        # Default to LLM settings from .env
        base_url = base_url or settings.llm_base_url
        model = model or settings.llm_model
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._http = ThreadSafeSession(timeout=timeout)

    def evaluate(
        self,
        question: str,
        human_answer: str,
        engram_answer: str,
        answer_type: str,
    ) -> dict[str, float]:
        """
        Evaluate Engram answer against human reference.

        Args:
            question: Original question
            human_answer: Human reference answer
            engram_answer: Engram's response
            answer_type: Classification of human answer

        Returns:
            Dict with key_info_match, relevance, no_contradiction, reasoning
        """
        # For empty human answers, only check relevance
        if answer_type == "empty":
            return self._evaluate_relevance_only(question, engram_answer)

        prompt = self._build_prompt(question, human_answer, engram_answer)

        try:
            response = self._call_llm(prompt)
            return self._parse_response(response)
        except Exception as e:
            logger.warning(f"Judge LLM failed: {e}")
            return self._default_scores(str(e))

    def _evaluate_relevance_only(
        self, question: str, engram_answer: str
    ) -> dict[str, float]:
        """Evaluate only relevance when no human answer provided."""
        prompt = f"""Оцени, отвечает ли ответ на вопрос (0.0-1.0).

Вопрос: {question}
Ответ: {engram_answer}

ВАЖНО: Ответь ТОЛЬКО одной строкой:
RESULT|0.0|<relevance>|1.0|<причина>

Пример: RESULT|0.0|0.9|1.0|по теме

НЕ ПИШИ НИЧЕГО КРОМЕ СТРОКИ RESULT|..."""

        try:
            response = self._call_llm(prompt)
            return self._parse_response(response)
        except Exception as e:
            logger.warning(f"Judge LLM relevance-only failed: {e}")
            return self._default_scores(str(e))

    def _build_prompt(
        self,
        question: str,
        human_answer: str,
        engram_answer: str,
    ) -> str:
        """Build evaluation prompt for LLM judge."""
        return f"""Ты - судья, оценивающий качество ответа RAG-системы.

ВАЖНО: Проверяй ВХОЖДЕНИЕ, а не РАВЕНСТВО. Ответ Engram может содержать БОЛЬШЕ информации - это ХОРОШО.

Вопрос: {question}

Эталонный ответ (может быть неполным):
{human_answer}

Ответ системы Engram:
{engram_answer}

Оцени по трём критериям (каждый от 0.0 до 1.0):

1. key_info_match: Содержит ли ответ Engram ВСЮ ключевую информацию из эталона?
   - 1.0 = вся информация из эталона присутствует (плюс дополнительная - ОК)
   - 0.5 = часть информации присутствует
   - 0.0 = ключевая информация отсутствует

2. relevance: Отвечает ли ответ на вопрос?
   - 1.0 = полностью отвечает
   - 0.5 = частично отвечает
   - 0.0 = не отвечает

3. no_contradiction: Нет ли противоречий с эталонным ответом?
   - 1.0 = нет противоречий
   - 0.5 = незначительные расхождения
   - 0.0 = прямые противоречия

ВАЖНО: Ответь ТОЛЬКО одной строкой без объяснений:
RESULT|<key_info_match>|<relevance>|<no_contradiction>|<причина>

Пример ответа: RESULT|0.9|1.0|1.0|информация присутствует

НЕ ПИШИ НИЧЕГО КРОМЕ СТРОКИ RESULT|..."""

    def check_availability(self) -> tuple[bool, str]:
        """
        Check if Judge LLM is available and returns valid responses.

        Returns:
            Tuple of (is_available, error_message)
        """
        test_prompt = "Reply with exactly: OK"

        try:
            response = self._call_llm(test_prompt)
            if response is None:
                return False, (
                    f"Judge LLM at {self.base_url} returned None content. "
                    f"Model '{self.model}' may not support chat completions format."
                )
            return True, ""
        except requests.exceptions.ConnectionError:
            return False, f"Cannot connect to Judge LLM at {self.base_url}"
        except requests.exceptions.Timeout:
            return False, f"Judge LLM at {self.base_url} timed out"
        except requests.exceptions.HTTPError as e:
            return False, f"Judge LLM HTTP error: {e}"
        except Exception as e:
            return False, f"Judge LLM check failed: {e}"

    def _call_llm(self, prompt: str) -> str | None:
        """Call LLM API and return response content."""
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 4096,  # Enough for thinking + response
        }

        response = self._http.post(url, payload)
        response.raise_for_status()
        data = response.json()

        if "choices" in data and data["choices"]:
            content = data["choices"][0].get("message", {}).get("content")
            # Handle None content explicitly
            if content is None:
                return None
            return content

        return ""

    def _parse_response(self, content: str | None) -> dict[str, float]:
        """Parse pipe-delimited response from judge."""
        if content is None:
            return self._default_scores("LLM returned None content")

        # Strip <think>...</think> blocks from thinking models
        # Also handle unclosed <think> tags (when output is truncated)
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        content = re.sub(r"<think>.*", "", content, flags=re.DOTALL)  # unclosed
        content = content.strip()

        logger.debug(f"Judge LLM response (after strip): {content[:300]}")

        for line in content.strip().split("\n"):
            line = line.strip()
            # Handle both "RESULT|" and variations like "RESULT |" or "Result|"
            if line.upper().startswith("RESULT"):
                # Remove "RESULT" prefix and clean up
                rest = line[6:].lstrip(" |:")
                parts = [""] + rest.split("|")  # Add empty first element to match expected format
                if len(parts) >= 4:
                    try:
                        result = {
                            "key_info_match": float(parts[1].strip()),
                            "relevance": float(parts[2].strip()),
                            "no_contradiction": float(parts[3].strip()),
                            "reasoning": parts[4].strip() if len(parts) > 4 else "",
                        }
                        logger.debug(f"Parsed scores: {result}")
                        return result
                    except ValueError:
                        pass

        # Fallback: try to extract scores from unstructured text
        logger.warning(f"Could not find RESULT| line, using fallback parser. Response: {content[:200]}")
        return self._fallback_parse(content)

    def _fallback_parse(self, content: str) -> dict[str, float]:
        """Attempt to extract scores from unstructured LLM response."""
        scores = {
            "key_info_match": 0.5,
            "relevance": 0.5,
            "no_contradiction": 0.5,
            "reasoning": "Parsed from unstructured response",
        }

        # Try to find three consecutive numbers (0.X format) that look like scores
        # This handles cases like "0.9|1.0|1.0" or "0.9, 1.0, 1.0"
        number_pattern = r"(0(?:\.\d+)?|1(?:\.0)?)"
        numbers = re.findall(number_pattern, content)

        # Filter to likely score values (0.0-1.0)
        score_numbers = []
        for n in numbers:
            try:
                val = float(n)
                if 0.0 <= val <= 1.0:
                    score_numbers.append(val)
            except ValueError:
                pass

        # If we found 3+ numbers that look like scores, use them
        if len(score_numbers) >= 3:
            scores["key_info_match"] = score_numbers[0]
            scores["relevance"] = score_numbers[1]
            scores["no_contradiction"] = score_numbers[2]
            scores["reasoning"] = "Extracted from numbers in response"
            logger.debug(f"Fallback extracted scores: {scores}")
            return scores

        # Try named patterns like "key_info_match: 0.8" or "relevance = 0.9"
        patterns = {
            "key_info_match": r"key_info[_\s]?match[:\s=]+([0-9.]+)",
            "relevance": r"relevan\w*[:\s=]+([0-9.]+)",
            "no_contradiction": r"no[_\s]?contradiction[:\s=]+([0-9.]+)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                try:
                    scores[key] = min(1.0, max(0.0, float(match.group(1))))
                except ValueError:
                    pass

        logger.debug(f"Fallback final scores: {scores}")
        return scores

    def _default_scores(self, error_msg: str) -> dict[str, float]:
        """Return neutral scores on error."""
        return {
            "key_info_match": 0.5,
            "relevance": 0.5,
            "no_contradiction": 0.5,
            "reasoning": f"Error: {error_msg}",
        }

    def compute_overall(
        self,
        answer_type: str,
        source_match: float,
        key_info_match: float,
        relevance: float,
        no_contradiction: float,
    ) -> float:
        """Compute weighted overall score based on answer type."""
        weights = self.WEIGHTS.get(answer_type, self.WEIGHTS["full"])

        overall = (
            weights["source_match"] * source_match
            + weights["key_info_match"] * key_info_match
            + weights["relevance"] * relevance
            + weights["no_contradiction"] * no_contradiction
        )

        return min(1.0, max(0.0, overall))


# =============================================================================
# Source Matching
# =============================================================================


def classify_answer(human_answer: str, human_url: str) -> str:
    """
    Classify human answer type.

    Returns:
        - empty: no answer and no URL
        - url_only: only URL provided, no text
        - short: <50 chars or <5 words (with URL)
        - short_no_url: <50 chars or <5 words (without URL)
        - full: complete answer (with URL)
        - full_no_url: complete answer (without URL)
    """
    answer = human_answer.strip() if human_answer else ""
    url = human_url.strip() if human_url else ""

    if not answer and not url:
        return "empty"

    if not answer and url:
        return "url_only"

    words = answer.split()
    is_short = len(answer) < 50 or len(words) < 5
    has_url = bool(url)

    if is_short:
        return "short" if has_url else "short_no_url"

    return "full" if has_url else "full_no_url"


def check_source_match(human_url: str, engram_sources: list[str]) -> float:
    """
    Check if Engram sources match human URL.

    Returns:
        - 1.0 = exact URL match
        - 1.0 = exact path match (different domain)
        - 0.95 = same pageId query param
        - 0.9 = same page ID in last path segment
        - 0.8 = path containment
        - 0.0 = no match
    """
    if not human_url or not engram_sources:
        return 0.0

    human_url = human_url.strip().rstrip("/")
    human_parsed = urlparse(human_url)
    human_path = human_parsed.path.rstrip("/")
    human_query = parse_qs(human_parsed.query)

    # Extract pageId from human URL
    human_page_id = None
    if "pageId" in human_query:
        human_page_id = human_query["pageId"][0]
    elif human_path:
        # Try to get last path segment as page ID
        segments = [s for s in human_path.split("/") if s]
        if segments:
            human_page_id = segments[-1]

    best_score = 0.0

    for source in engram_sources:
        source = source.strip().rstrip("/")
        source_parsed = urlparse(source)
        source_path = source_parsed.path.rstrip("/")
        source_query = parse_qs(source_parsed.query)

        # 1. Exact URL match
        if human_url == source:
            return 1.0

        # 2. Exact path match (different domain)
        if human_path and source_path and human_path == source_path:
            best_score = max(best_score, 1.0)
            continue

        # 3. Same pageId query param
        if human_page_id and "pageId" in source_query:
            if source_query["pageId"][0] == human_page_id:
                best_score = max(best_score, 0.95)
                continue

        # 4. Same page ID in last path segment
        if human_page_id and source_path:
            source_segments = [s for s in source_path.split("/") if s]
            if source_segments and source_segments[-1] == human_page_id:
                best_score = max(best_score, 0.9)
                continue

        # 5. Path containment
        if human_path and source_path:
            if human_path in source_path or source_path in human_path:
                best_score = max(best_score, 0.8)

    return best_score


# =============================================================================
# Main Evaluator
# =============================================================================


class EngramEvaluator:
    """
    Orchestrates parallel evaluation of test questions.

    Queries Engram API, uses LLM judge for semantic evaluation,
    and computes aggregate metrics.
    """

    def __init__(
        self,
        engram_url: str = "http://localhost:7777/v1",
        engram_model: str = "engram",
        judge_url: str | None = None,
        judge_model: str | None = None,
        workers: int = 4,
        skip_url_only: bool = False,
    ) -> None:
        self.engram = EngramClient(engram_url, engram_model)
        self.judge = JudgeLLM(judge_url, judge_model)
        self.workers = workers
        self.skip_url_only = skip_url_only

    def evaluate_csv(
        self,
        csv_path: str | Path,
        output_path: str | Path | None = None,
        limit: int | None = None,
    ) -> tuple[EvaluationSummary, list[EvaluationResult]]:
        """
        Evaluate all questions from CSV file.

        Args:
            csv_path: Path to input CSV (question,answer,url)
            output_path: Optional path for results CSV
            limit: Max number of questions to evaluate (for testing)

        Returns:
            Tuple of (EvaluationSummary, list of EvaluationResult)
        """
        csv_path = Path(csv_path)

        # Load questions
        questions = self._load_csv(csv_path)
        logger.info(f"Loaded {len(questions)} questions from {csv_path}")

        # Filter out url_only questions if requested
        if self.skip_url_only:
            original_count = len(questions)
            questions = [
                (q, a, u) for q, a, u in questions
                if classify_answer(a, u) != "url_only"
            ]
            skipped = original_count - len(questions)
            if skipped > 0:
                logger.info(f"Skipped {skipped} url_only questions")

        # Apply limit if specified
        if limit is not None and limit > 0:
            questions = questions[:limit]
            logger.info(f"Limited to {len(questions)} questions")

        # Evaluate in parallel
        results = self._evaluate_parallel(questions)

        # Compute summary
        summary = self._compute_summary(results)

        # Save results
        if output_path is None:
            output_path = csv_path.with_suffix("").name + "_results.csv"
        output_path = Path(output_path)

        self._save_results(results, output_path)
        self._save_summary(summary, output_path.with_suffix(".summary.json"))

        return summary, results

    def _load_csv(
        self, csv_path: Path
    ) -> list[tuple[str, str, str]]:
        """Load questions from CSV file.

        Supports:
        - Comma or semicolon delimiters (auto-detected)
        - Required columns: question, answer
        - Optional column: url (for source matching)
        - English columns: question, answer, url
        - Russian columns: Вопрос, Правильный ответ, Ссылка на правильный ответ
        - UTF-8 with or without BOM
        """
        questions = []

        # Read with utf-8-sig to handle BOM automatically
        with open(csv_path, encoding="utf-8-sig") as f:
            # Read first line to detect delimiter
            first_line = f.readline()
            f.seek(0)

            # Auto-detect delimiter
            delimiter = ";" if ";" in first_line else ","
            logger.debug(f"Detected delimiter: '{delimiter}'")
            logger.debug(f"First line: {first_line[:100]!r}")

            reader = csv.DictReader(f, delimiter=delimiter)

            # Log detected columns
            if reader.fieldnames:
                logger.debug(f"Columns: {reader.fieldnames}")

            # Column name mappings (Russian -> English)
            column_maps = {
                "question": ["question", "Вопрос"],
                "answer": ["answer", "Правильный ответ"],
                "url": ["url", "Ссылка на правильный ответ"],
            }

            for row in reader:
                # Find question column
                question = ""
                for col in column_maps["question"]:
                    if col in row:
                        question = (row[col] or "").strip()
                        break

                # Find answer column
                answer = ""
                for col in column_maps["answer"]:
                    if col in row:
                        answer = (row[col] or "").strip()
                        break

                # Find URL column
                url = ""
                for col in column_maps["url"]:
                    if col in row:
                        url = (row[col] or "").strip()
                        break

                if question:
                    questions.append((question, answer, url))

        return questions

    def _evaluate_parallel(
        self,
        questions: list[tuple[str, str, str]],
    ) -> list[EvaluationResult]:
        """Evaluate questions in parallel."""
        results: list[EvaluationResult] = []

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {
                executor.submit(
                    self._evaluate_single, question, answer, url, idx
                ): idx
                for idx, (question, answer, url) in enumerate(questions)
            }

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(
                        f"[{len(results)}/{len(questions)}] "
                        f"Q{idx}: overall={result.overall:.2f} "
                        f"({result.answer_type})"
                    )
                except Exception as e:
                    logger.error(f"Q{idx} failed: {e}")
                    # Create error result
                    question, answer, url = questions[idx]
                    results.append(
                        EvaluationResult(
                            question=question,
                            human_answer=answer,
                            human_url=url,
                            engram_answer="",
                            engram_sources=[],
                            answer_type=classify_answer(answer, url),
                            source_match=0.0,
                            key_info_match=0.0,
                            relevance=0.0,
                            no_contradiction=0.0,
                            overall=0.0,
                            error=str(e),
                        )
                    )

        # Sort by original order
        return results

    def _evaluate_single(
        self,
        question: str,
        human_answer: str,
        human_url: str,
        idx: int,
    ) -> EvaluationResult:
        """Evaluate a single question."""
        start_time = time.time()

        # Classify answer type
        answer_type = classify_answer(human_answer, human_url)

        # Query Engram
        try:
            engram_answer, engram_sources = self.engram.query(question)
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return EvaluationResult(
                question=question,
                human_answer=human_answer,
                human_url=human_url,
                engram_answer="",
                engram_sources=[],
                answer_type=answer_type,
                source_match=0.0,
                key_info_match=0.0,
                relevance=0.0,
                no_contradiction=0.0,
                overall=0.0,
                error=f"Engram query failed: {e}",
                latency_ms=latency_ms,
            )

        # Check source match
        source_match = check_source_match(human_url, engram_sources)

        # LLM judge evaluation
        judge_result = self.judge.evaluate(
            question=question,
            human_answer=human_answer,
            engram_answer=engram_answer,
            answer_type=answer_type,
        )

        key_info_match = judge_result["key_info_match"]
        relevance = judge_result["relevance"]
        no_contradiction = judge_result["no_contradiction"]
        reasoning = judge_result.get("reasoning", "")

        # Compute overall score
        overall = self.judge.compute_overall(
            answer_type=answer_type,
            source_match=source_match,
            key_info_match=key_info_match,
            relevance=relevance,
            no_contradiction=no_contradiction,
        )

        latency_ms = (time.time() - start_time) * 1000

        return EvaluationResult(
            question=question,
            human_answer=human_answer,
            human_url=human_url,
            engram_answer=engram_answer,
            engram_sources=engram_sources,
            answer_type=answer_type,
            source_match=source_match,
            key_info_match=key_info_match,
            relevance=relevance,
            no_contradiction=no_contradiction,
            overall=overall,
            reasoning=reasoning,
            latency_ms=latency_ms,
        )

    def _compute_summary(
        self, results: list[EvaluationResult]
    ) -> EvaluationSummary:
        """Compute aggregate statistics."""
        summary = EvaluationSummary()
        summary.total = len(results)

        # Count successes/failures
        successful_results = [r for r in results if not r.error]
        summary.successful = len(successful_results)
        summary.failed = summary.total - summary.successful

        if not successful_results:
            return summary

        # Compute averages
        summary.avg_source_match = sum(
            r.source_match for r in successful_results
        ) / len(successful_results)
        summary.avg_key_info_match = sum(
            r.key_info_match for r in successful_results
        ) / len(successful_results)
        summary.avg_relevance = sum(
            r.relevance for r in successful_results
        ) / len(successful_results)
        summary.avg_no_contradiction = sum(
            r.no_contradiction for r in successful_results
        ) / len(successful_results)
        summary.avg_overall = sum(
            r.overall for r in successful_results
        ) / len(successful_results)

        # Compute by type
        by_type: dict[str, list[EvaluationResult]] = {}
        for r in successful_results:
            if r.answer_type not in by_type:
                by_type[r.answer_type] = []
            by_type[r.answer_type].append(r)

        for answer_type, type_results in by_type.items():
            n = len(type_results)
            summary.by_type[answer_type] = {
                "count": n,
                "avg_source_match": round(
                    sum(r.source_match for r in type_results) / n, 3
                ),
                "avg_key_info_match": round(
                    sum(r.key_info_match for r in type_results) / n, 3
                ),
                "avg_relevance": round(
                    sum(r.relevance for r in type_results) / n, 3
                ),
                "avg_no_contradiction": round(
                    sum(r.no_contradiction for r in type_results) / n, 3
                ),
                "avg_overall": round(
                    sum(r.overall for r in type_results) / n, 3
                ),
            }

        # Timing
        summary.total_latency_ms = sum(r.latency_ms for r in results)
        summary.avg_latency_ms = summary.total_latency_ms / len(results)

        return summary

    def _save_results(
        self, results: list[EvaluationResult], output_path: Path
    ) -> None:
        """Save detailed results to CSV."""
        if not results:
            return

        fieldnames = list(results[0].to_dict().keys())

        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result.to_dict())

        logger.info(f"Saved results to {output_path}")

    def _save_summary(
        self, summary: EvaluationSummary, output_path: Path
    ) -> None:
        """Save summary to JSON."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, ensure_ascii=False, indent=2)

        logger.info(f"Saved summary to {output_path}")


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate Engram against human reference answers"
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to input CSV (question,answer) or (question,answer,url)",
    )
    parser.add_argument(
        "--engram-url",
        type=str,
        default="http://localhost:7777/v1",
        help="Engram API base URL",
    )
    parser.add_argument(
        "--engram-model",
        type=str,
        default="engram",
        help="Engram model name",
    )
    parser.add_argument(
        "--judge-url",
        type=str,
        default=None,
        help=f"Judge LLM API base URL (default: LLM_BASE_URL from .env = {settings.llm_base_url})",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help=f"Judge LLM model name (default: LLM_MODEL from .env = {settings.llm_model})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: {input}_results.csv)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--skip-url-only",
        action="store_true",
        help="Skip questions that only have URL (no text answer)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions to evaluate (for testing)",
    )
    parser.add_argument(
        "--show-below",
        type=float,
        default=None,
        metavar="THRESHOLD",
        help="Show questions with overall score below threshold (e.g., 0.5)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Create evaluator
    evaluator = EngramEvaluator(
        engram_url=args.engram_url,
        engram_model=args.engram_model,
        judge_url=args.judge_url,
        judge_model=args.judge_model,
        workers=args.workers,
        skip_url_only=args.skip_url_only,
    )

    # Check availability before starting
    print("Checking service availability...")

    # Check Engram
    engram_ok, engram_err = evaluator.engram.check_availability()
    if not engram_ok:
        print(f"\n[ERROR] Engram API not available:")
        print(f"  URL: {args.engram_url}")
        print(f"  Error: {engram_err}")
        print("\nMake sure Engram API is running:")
        print("  uv run python -m engram.api.main")
        sys.exit(1)

    print(f"  [OK] Engram API at {args.engram_url}")

    # Check Judge LLM
    judge_url = args.judge_url or settings.llm_base_url
    judge_model = args.judge_model or settings.llm_model
    judge_ok, judge_err = evaluator.judge.check_availability()
    if not judge_ok:
        print(f"\n[ERROR] Judge LLM not available:")
        print(f"  URL: {judge_url}")
        print(f"  Model: {judge_model}")
        print(f"  Error: {judge_err}")
        print("\nTroubleshooting:")
        print("  1. Check if LLM server is running")
        print("  2. Verify model name is correct")
        print("  3. Try a different model with --judge-model")
        print("  4. Try Ollama locally: --judge-url http://localhost:11434/v1 --judge-model qwen3:8b")
        sys.exit(1)

    print(f"  [OK] Judge LLM at {judge_url} (model: {judge_model})")
    print()

    # Run evaluation
    summary, results = evaluator.evaluate_csv(args.csv_path, args.output, args.limit)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total:     {summary.total}")
    print(f"Success:   {summary.successful}")
    print(f"Failed:    {summary.failed}")
    print()
    print(f"Avg Source Match:      {summary.avg_source_match:.3f}")
    print(f"Avg Key Info Match:    {summary.avg_key_info_match:.3f}")
    print(f"Avg Relevance:         {summary.avg_relevance:.3f}")
    print(f"Avg No Contradiction:  {summary.avg_no_contradiction:.3f}")
    print(f"Avg Overall:           {summary.avg_overall:.3f}")
    print()
    print(f"Total Latency:         {summary.total_latency_ms:.0f} ms")
    print(f"Avg Latency:           {summary.avg_latency_ms:.0f} ms")

    if summary.by_type:
        print()
        print("By Answer Type:")
        for answer_type, stats in summary.by_type.items():
            print(f"  {answer_type}: {stats['count']} items, "
                  f"overall={stats['avg_overall']:.3f}")

    # Show questions below threshold
    if args.show_below is not None:
        below_threshold = [
            r for r in results
            if r.overall < args.show_below and not r.error
        ]
        if below_threshold:
            print()
            print("=" * 60)
            print(f"QUESTIONS BELOW {args.show_below} ({len(below_threshold)} items)")
            print("=" * 60)
            # Sort by overall score ascending (worst first)
            below_threshold.sort(key=lambda r: r.overall)
            for i, r in enumerate(below_threshold, 1):
                print(f"\n[{i}] Overall: {r.overall:.3f} | Type: {r.answer_type}")
                print(f"    Source: {r.source_match:.2f} | Key: {r.key_info_match:.2f} | "
                      f"Rel: {r.relevance:.2f} | NoCont: {r.no_contradiction:.2f}")
                print(f"    Q: {r.question[:100]}{'...' if len(r.question) > 100 else ''}")
                if r.human_answer:
                    print(f"    Human: {r.human_answer[:80]}{'...' if len(r.human_answer) > 80 else ''}")
                if r.human_url:
                    print(f"    URL: {r.human_url}")
                if r.reasoning:
                    print(f"    Reason: {r.reasoning[:100]}{'...' if len(r.reasoning) > 100 else ''}")
        else:
            print(f"\nNo questions below {args.show_below}")


if __name__ == "__main__":
    main()
