"""Unit tests for document parser."""

from pathlib import Path

import pytest

from engram.ingestion.parser import (
    DocumentParser,
    chunk_content,
    clean_markdown,
    compute_hash,
    detect_doc_type,
    extract_title_from_markdown,
    parse_content,
)


class TestDocumentParser:
    """Tests for document parser functions."""

    def test_detect_doc_type_markdown(self) -> None:
        """Test markdown file detection."""
        assert detect_doc_type(Path("test.md")) == "markdown"
        assert detect_doc_type(Path("test.markdown")) == "markdown"

    def test_detect_doc_type_text(self) -> None:
        """Test text file detection."""
        assert detect_doc_type(Path("test.txt")) == "text"

    def test_detect_doc_type_html(self) -> None:
        """Test HTML file detection."""
        assert detect_doc_type(Path("test.html")) == "html"
        assert detect_doc_type(Path("test.htm")) == "html"

    def test_detect_doc_type_unknown(self) -> None:
        """Test unknown file type defaults to text."""
        assert detect_doc_type(Path("test.xyz")) == "text"

    def test_extract_title_h1(self) -> None:
        """Test title extraction from h1."""
        content = "# My Title\n\nSome content"
        assert extract_title_from_markdown(content) == "My Title"

    def test_extract_title_frontmatter(self) -> None:
        """Test title extraction from frontmatter."""
        content = "---\ntitle: Front Title\n---\n\n# Another"
        # H1 takes precedence when both exist
        assert extract_title_from_markdown(content) == "Another"

    def test_extract_title_fallback(self) -> None:
        """Test title fallback to first line."""
        content = "Just some text\nMore text"
        assert extract_title_from_markdown(content) == "Just some text"

    def test_clean_markdown_removes_frontmatter(self) -> None:
        """Test frontmatter removal."""
        content = "---\ntitle: Test\ndate: 2024-01-01\n---\n\nActual content"
        cleaned = clean_markdown(content)
        assert "---" not in cleaned
        assert "Actual content" in cleaned

    def test_clean_markdown_normalizes_whitespace(self) -> None:
        """Test whitespace normalization."""
        content = "First\n\n\n\n\nSecond"
        cleaned = clean_markdown(content)
        assert "\n\n\n" not in cleaned

    def test_compute_hash(self) -> None:
        """Test content hash computation."""
        content = "Test content"
        hash1 = compute_hash(content)
        hash2 = compute_hash(content)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex

        # Different content = different hash
        hash3 = compute_hash("Different")
        assert hash1 != hash3

    def test_chunk_content_small(self) -> None:
        """Test chunking small content."""
        content = "Short content"
        chunks = chunk_content(content, chunk_size=100)
        assert len(chunks) == 1
        assert chunks[0][0] == content

    def test_chunk_content_large(self) -> None:
        """Test chunking large content."""
        # Create content with multiple paragraphs
        paragraphs = ["Paragraph " + str(i) + " " * 100 for i in range(10)]
        content = "\n\n".join(paragraphs)

        chunks = chunk_content(content, chunk_size=300, overlap=50)
        assert len(chunks) > 1

        # Verify offsets make sense
        for text, start, end in chunks:
            assert len(text) > 0
            assert start >= 0

    def test_parse_content(self) -> None:
        """Test parsing raw content."""
        content = "# Test Doc\n\nSome content here"
        doc = parse_content(content)

        assert doc.title == "Test Doc"
        assert "content here" in doc.content
        assert doc.doc_type == "markdown"
        assert doc.source_hash is not None


class TestDocumentParserClass:
    """Tests for DocumentParser class."""

    def test_parser_init(self) -> None:
        """Test parser initialization."""
        parser = DocumentParser(chunk_size=1000, overlap=100)
        assert parser.chunk_size == 1000
        assert parser.overlap == 100

    def test_parser_parse_content(self) -> None:
        """Test parser content parsing."""
        parser = DocumentParser()
        doc = parser.parse_content("# Title\n\nBody", title="Custom Title")
        assert doc.title == "Custom Title"

    def test_parser_chunk_document(self) -> None:
        """Test parser document chunking."""
        parser = DocumentParser(chunk_size=100)
        doc = parser.parse_content("# Test\n\n" + "Word " * 100)
        chunks = parser.chunk_document(doc)

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.document_id == doc.id
            assert chunk.content

    @pytest.mark.asyncio
    async def test_parser_parse_file(self, mock_docs_path: Path) -> None:
        """Test parsing a file."""
        parser = DocumentParser()
        doc_path = mock_docs_path / "docker_basics.md"

        doc = await parser.parse_file(doc_path)

        assert doc.title == "Docker Basics"
        assert "контейнер" in doc.content.lower()
        assert doc.source_path is not None
