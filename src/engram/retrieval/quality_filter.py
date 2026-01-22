"""Quality filtering for retrieval results.

Provides chunk quality scoring and source-based weighting to filter
low-quality results before indexing and at retrieval time.
"""

import re
from dataclasses import dataclass

from engram.preprocessing.russian import RUSSIAN_STOPWORDS


@dataclass
class QualityScore:
    """Quality metrics for a text chunk."""

    overall: float  # Combined score 0-1
    length_score: float  # Score based on word count
    uniqueness_score: float  # Score based on unique words ratio
    stopword_score: float  # Score based on non-stopword ratio
    info_density: float  # Information density (unique content words / total)


# Source type weights for retrieval scoring
# Higher weight = more trusted source
SOURCE_WEIGHTS: dict[str, float] = {
    "document": 1.0,  # Original document text
    "table": 0.95,  # Table content (structured data)
    "list": 0.9,  # List items with context
    "fact": 0.85,  # Extracted facts
    "summary": 0.8,  # LLM-generated summaries
    "enriched": 0.75,  # LLM-enriched content
    "default": 0.7,  # Unknown source type
}

# Minimum thresholds
MIN_CHUNK_WORDS = 20
QUALITY_THRESHOLD = 0.4


def get_source_weight(source_type: str | None) -> float:
    """
    Get weight for a memory source type.

    Args:
        source_type: Type of memory source (document, table, list, etc.)

    Returns:
        Weight multiplier (0-1)
    """
    if source_type is None:
        return SOURCE_WEIGHTS["default"]
    return SOURCE_WEIGHTS.get(source_type.lower(), SOURCE_WEIGHTS["default"])


def calculate_chunk_quality(
    text: str,
    min_words: int = MIN_CHUNK_WORDS,
    stopwords: set[str] | None = None,
) -> QualityScore:
    """
    Calculate quality score for a text chunk.

    Quality metrics:
    - Length: Penalize very short or very long chunks
    - Uniqueness: Ratio of unique words (penalizes repetitive content)
    - Stopword ratio: Penalize high stopword density
    - Info density: Unique content words / total words

    Args:
        text: Text chunk to score
        min_words: Minimum words for full length score
        stopwords: Set of stopwords to filter (defaults to Russian)

    Returns:
        QualityScore with all metrics
    """
    if stopwords is None:
        stopwords = RUSSIAN_STOPWORDS

    # Tokenize
    words = re.findall(r"\w+", text.lower())
    word_count = len(words)

    # Edge case: empty or very short text
    if word_count < 5:
        return QualityScore(
            overall=0.0,
            length_score=0.0,
            uniqueness_score=0.0,
            stopword_score=0.0,
            info_density=0.0,
        )

    # Length score: sigmoid around min_words
    # Score = 1 for chunks >= min_words, drops for shorter chunks
    if word_count >= min_words:
        length_score = 1.0
    else:
        length_score = word_count / min_words

    # Penalize excessively long chunks (> 500 words)
    if word_count > 500:
        length_score *= 0.9  # Slight penalty for very long chunks

    # Uniqueness score: ratio of unique words
    unique_words = set(words)
    uniqueness_score = len(unique_words) / word_count

    # Boost uniqueness for short texts (naturally lower unique ratio)
    if word_count < 50:
        uniqueness_score = min(1.0, uniqueness_score * 1.2)

    # Stopword score: ratio of non-stopwords
    content_words = [w for w in words if w not in stopwords]
    stopword_score = len(content_words) / word_count if word_count > 0 else 0.0

    # Info density: unique content words / total
    unique_content_words = set(content_words)
    info_density = len(unique_content_words) / word_count if word_count > 0 else 0.0

    # Combined score with weights
    # Length: 20%, Uniqueness: 30%, Stopword ratio: 25%, Info density: 25%
    overall = (
        length_score * 0.20
        + uniqueness_score * 0.30
        + stopword_score * 0.25
        + info_density * 0.25
    )

    return QualityScore(
        overall=overall,
        length_score=length_score,
        uniqueness_score=uniqueness_score,
        stopword_score=stopword_score,
        info_density=info_density,
    )


def filter_low_quality_chunks(
    texts: list[str],
    threshold: float = QUALITY_THRESHOLD,
    min_words: int = MIN_CHUNK_WORDS,
) -> list[tuple[str, QualityScore]]:
    """
    Filter out low-quality chunks before indexing.

    Args:
        texts: List of text chunks to filter
        threshold: Minimum quality score (0-1)
        min_words: Minimum word count

    Returns:
        List of (text, score) tuples that passed the filter
    """
    results: list[tuple[str, QualityScore]] = []

    for text in texts:
        score = calculate_chunk_quality(text, min_words=min_words)
        if score.overall >= threshold:
            results.append((text, score))

    return results


def apply_source_weight(
    score: float,
    source_type: str | None,
    source_weight_factor: float = 0.2,
) -> float:
    """
    Apply source-based weight adjustment to a retrieval score.

    Args:
        score: Original retrieval score
        source_type: Type of source (document, table, etc.)
        source_weight_factor: How much source weight affects final score (0-1)

    Returns:
        Adjusted score
    """
    source_weight = get_source_weight(source_type)

    # Blend original score with source weight
    # score_weight_factor=0.2 means 80% original, 20% source influence
    adjusted = score * (1 - source_weight_factor) + score * source_weight * source_weight_factor

    return adjusted


def is_quality_chunk(
    text: str,
    threshold: float = QUALITY_THRESHOLD,
    min_words: int = MIN_CHUNK_WORDS,
) -> bool:
    """
    Quick check if a chunk passes quality threshold.

    Args:
        text: Text to check
        threshold: Minimum quality score
        min_words: Minimum word count

    Returns:
        True if chunk is high enough quality
    """
    score = calculate_chunk_quality(text, min_words=min_words)
    return score.overall >= threshold
