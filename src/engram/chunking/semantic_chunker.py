"""Semantic chunker using BGE-M3 tokenizer for accurate token counting.

v5: Creates semantic chunks for vector retrieval with:
- BGE-M3 tokenizer for accurate token counting
- Russian-aware sentence splitting
- Embedding-based similarity for break point detection
- Target 512-1024 tokens per chunk
"""

import logging
import re
import threading
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from engram.config import settings

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A semantic chunk of text."""

    text: str
    token_count: int
    start_char: int
    end_char: int
    chunk_index: int
    metadata: dict = field(default_factory=dict)


# Russian sentence-ending patterns
SENTENCE_END_PATTERN = re.compile(
    r'(?<=[.!?])\s+(?=[A-ZА-ЯЁ])|'  # After .!? followed by capital letter
    r'(?<=[.!?])\s*$|'  # After .!? at end
    r'(?<=[.!?])\s+(?=\d)|'  # After .!? followed by digit (numbered list)
    r'(?<=\n)\s*(?=[-•*])'  # Before bullet points
)

# Abbreviations that shouldn't end sentences (Russian)
RUSSIAN_ABBREVS = {
    'т.е.', 'т.к.', 'и.т.д.', 'и.т.п.', 'т.н.', 'т.д.', 'т.п.',
    'г.', 'гг.', 'в.', 'вв.', 'н.э.', 'до н.э.',
    'см.', 'ср.', 'напр.', 'прим.', 'др.', 'пр.',
    'ок.', 'обл.', 'р.', 'руб.', 'коп.',
    'тыс.', 'млн.', 'млрд.', 'трлн.',
    'им.', 'ул.', 'пер.', 'пл.', 'д.', 'кв.', 'стр.',
}


@lru_cache(maxsize=1)
def get_bge_tokenizer():
    """Get BGE-M3 tokenizer (cached singleton)."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            settings.bge_model_name,
            trust_remote_code=True,
        )
        logger.info(f"Loaded BGE tokenizer: {settings.bge_model_name}")
        return tokenizer
    except Exception as e:
        logger.warning(f"Failed to load BGE tokenizer: {e}, falling back to simple tokenizer")
        return None


class SemanticChunker:
    """
    Semantic text chunker optimized for vector retrieval.

    Features:
    - Uses BGE-M3 tokenizer for accurate token counting
    - Russian-aware sentence splitting
    - Embedding similarity for detecting semantic breaks
    - Target chunk size: 512-1024 tokens
    - Overlap support for context preservation
    """

    def __init__(
        self,
        min_tokens: int | None = None,
        max_tokens: int | None = None,
        similarity_threshold: float | None = None,
        overlap_tokens: int = 50,
    ) -> None:
        """
        Initialize semantic chunker.

        Args:
            min_tokens: Minimum tokens per chunk (default from settings)
            max_tokens: Maximum tokens per chunk (default from settings)
            similarity_threshold: Threshold for semantic break detection (default from settings)
            overlap_tokens: Tokens to overlap between chunks for context
        """
        self.min_tokens = min_tokens or settings.chunk_min_tokens
        self.max_tokens = max_tokens or settings.chunk_max_tokens
        self.similarity_threshold = similarity_threshold or settings.chunk_similarity_threshold
        self.overlap_tokens = overlap_tokens

        self._tokenizer = None
        self._tokenizer_lock = threading.Lock()

    def _get_tokenizer(self):
        """Get tokenizer (lazy load)."""
        if self._tokenizer is None:
            with self._tokenizer_lock:
                if self._tokenizer is None:
                    self._tokenizer = get_bge_tokenizer()
        return self._tokenizer

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using BGE tokenizer."""
        tokenizer = self._get_tokenizer()
        if tokenizer:
            return len(tokenizer.encode(text, add_special_tokens=False))
        # Fallback: approximate with word count * 1.3 for Russian
        return int(len(text.split()) * 1.3)

    def _split_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences with Russian language awareness.

        Handles:
        - Standard punctuation (.!?)
        - Russian abbreviations
        - Bulleted/numbered lists
        """
        # Protect abbreviations by replacing dots with placeholders
        protected = text
        for abbrev in RUSSIAN_ABBREVS:
            protected = protected.replace(abbrev, abbrev.replace('.', '\x00'))

        # Split on sentence boundaries
        sentences = []
        last_end = 0

        for match in SENTENCE_END_PATTERN.finditer(protected):
            end_pos = match.start()
            if end_pos > last_end:
                sentence = protected[last_end:end_pos + 1].strip()
                if sentence:
                    # Restore dots in abbreviations
                    sentence = sentence.replace('\x00', '.')
                    sentences.append(sentence)
            last_end = match.end()

        # Add remaining text
        if last_end < len(protected):
            remaining = protected[last_end:].strip()
            if remaining:
                remaining = remaining.replace('\x00', '.')
                sentences.append(remaining)

        # If no splits found, split by newlines
        if len(sentences) <= 1:
            lines = text.split('\n')
            sentences = [line.strip() for line in lines if line.strip()]

        return sentences if sentences else [text]

    def _compute_sentence_embeddings(
        self,
        sentences: list[str],
    ) -> np.ndarray | None:
        """
        Compute embeddings for sentences using BGE-M3.

        Returns None if embedding service unavailable.
        """
        if not sentences:
            return None

        try:
            from engram.embeddings.bge_service import get_bge_embedding_service
            service = get_bge_embedding_service()
            embeddings = service.embed_batch_sync(sentences)
            return np.array(embeddings)
        except Exception as e:
            logger.debug(f"Could not compute embeddings for chunking: {e}")
            return None

    def _find_semantic_breaks(
        self,
        sentences: list[str],
        embeddings: np.ndarray | None,
    ) -> list[int]:
        """
        Find semantic break points based on embedding similarity drops.

        Returns list of sentence indices where breaks should occur.
        """
        if embeddings is None or len(sentences) <= 1:
            return []

        # Compute cosine similarities between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i + 1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1]) + 1e-10
            )
            similarities.append(sim)

        # Find break points where similarity drops below threshold
        # or is significantly lower than neighbors
        breaks = []
        mean_sim = np.mean(similarities) if similarities else 0.5

        for i, sim in enumerate(similarities):
            # Break if similarity is below threshold
            if sim < self.similarity_threshold:
                breaks.append(i + 1)  # Break after sentence i
            # Also break if similarity drops significantly from local average
            elif i > 0 and i < len(similarities) - 1:
                local_avg = (similarities[i - 1] + similarities[i + 1]) / 2
                if sim < local_avg * 0.7:  # 30% drop from local average
                    breaks.append(i + 1)

        return breaks

    def chunk_document(
        self,
        text: str,
        doc_metadata: dict | None = None,
    ) -> list[Chunk]:
        """
        Split document into semantic chunks.

        Args:
            text: Document text to chunk
            doc_metadata: Optional metadata to include in chunks

        Returns:
            List of Chunk objects
        """
        if not text.strip():
            return []

        doc_metadata = doc_metadata or {}

        # Split into sentences
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        # Compute embeddings for semantic break detection
        embeddings = self._compute_sentence_embeddings(sentences)

        # Find semantic break points
        semantic_breaks = set(self._find_semantic_breaks(sentences, embeddings))

        # Build chunks respecting token limits and semantic breaks
        chunks = []
        current_sentences: list[str] = []
        current_tokens = 0
        current_start = 0
        char_pos = 0

        for i, sentence in enumerate(sentences):
            sentence_tokens = self.count_tokens(sentence)

            # Check if we should start a new chunk
            should_break = False

            # Force break if adding sentence would exceed max
            if current_tokens + sentence_tokens > self.max_tokens and current_sentences:
                should_break = True
            # Break at semantic boundaries if we have enough tokens
            elif i in semantic_breaks and current_tokens >= self.min_tokens:
                should_break = True

            if should_break:
                # Save current chunk
                chunk_text = ' '.join(current_sentences)
                chunks.append(Chunk(
                    text=chunk_text,
                    token_count=current_tokens,
                    start_char=current_start,
                    end_char=char_pos,
                    chunk_index=len(chunks),
                    metadata=dict(doc_metadata),
                ))

                # Start new chunk (with overlap if configured)
                if self.overlap_tokens > 0 and current_sentences:
                    # Include last sentence(s) for overlap
                    overlap_sentences = []
                    overlap_tokens = 0
                    for sent in reversed(current_sentences):
                        sent_tokens = self.count_tokens(sent)
                        if overlap_tokens + sent_tokens <= self.overlap_tokens:
                            overlap_sentences.insert(0, sent)
                            overlap_tokens += sent_tokens
                        else:
                            break
                    current_sentences = overlap_sentences
                    current_tokens = overlap_tokens
                else:
                    current_sentences = []
                    current_tokens = 0
                current_start = char_pos

            # Add sentence to current chunk
            current_sentences.append(sentence)
            current_tokens += sentence_tokens
            char_pos += len(sentence) + 1  # +1 for space/newline

        # Save final chunk
        if current_sentences:
            chunk_text = ' '.join(current_sentences)
            chunks.append(Chunk(
                text=chunk_text,
                token_count=current_tokens,
                start_char=current_start,
                end_char=len(text),
                chunk_index=len(chunks),
                metadata=dict(doc_metadata),
            ))

        # Post-process: merge small chunks
        chunks = self._merge_small_chunks(chunks)

        logger.debug(
            f"Created {len(chunks)} chunks from {len(sentences)} sentences "
            f"(avg {sum(c.token_count for c in chunks) / len(chunks):.0f} tokens)"
        )

        return chunks

    def _merge_small_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Merge chunks that are too small."""
        if len(chunks) <= 1:
            return chunks

        merged = []
        current = chunks[0]

        for next_chunk in chunks[1:]:
            combined_tokens = current.token_count + next_chunk.token_count

            # Merge if current is too small and combined won't exceed max
            if current.token_count < self.min_tokens and combined_tokens <= self.max_tokens:
                current = Chunk(
                    text=current.text + ' ' + next_chunk.text,
                    token_count=combined_tokens,
                    start_char=current.start_char,
                    end_char=next_chunk.end_char,
                    chunk_index=current.chunk_index,
                    metadata=current.metadata,
                )
            else:
                merged.append(current)
                current = Chunk(
                    text=next_chunk.text,
                    token_count=next_chunk.token_count,
                    start_char=next_chunk.start_char,
                    end_char=next_chunk.end_char,
                    chunk_index=len(merged),
                    metadata=next_chunk.metadata,
                )

        merged.append(current)

        # Update chunk indices
        for i, chunk in enumerate(merged):
            chunk.chunk_index = i

        return merged

    def _split_large_chunk(self, chunk: Chunk) -> list[Chunk]:
        """Split a chunk that's too large into smaller pieces."""
        if chunk.token_count <= self.max_tokens:
            return [chunk]

        # Split by sentences and create sub-chunks
        sentences = self._split_sentences(chunk.text)
        sub_chunks = []
        current_text = []
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = self.count_tokens(sentence)

            if current_tokens + sent_tokens > self.max_tokens and current_text:
                sub_chunks.append(Chunk(
                    text=' '.join(current_text),
                    token_count=current_tokens,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    chunk_index=0,
                    metadata=chunk.metadata,
                ))
                current_text = []
                current_tokens = 0

            current_text.append(sentence)
            current_tokens += sent_tokens

        if current_text:
            sub_chunks.append(Chunk(
                text=' '.join(current_text),
                token_count=current_tokens,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                chunk_index=0,
                metadata=chunk.metadata,
            ))

        return sub_chunks
