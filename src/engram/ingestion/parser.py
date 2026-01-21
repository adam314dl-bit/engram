"""Document parser for various file formats."""

import hashlib
import re
import uuid
from pathlib import Path

import aiofiles

from engram.models import Document, DocumentChunk, DocumentType


def generate_id() -> str:
    """Generate a unique ID."""
    return str(uuid.uuid4())


def compute_hash(content: str) -> str:
    """Compute SHA256 hash of content."""
    return hashlib.sha256(content.encode()).hexdigest()


def detect_doc_type(path: Path) -> DocumentType:
    """Detect document type from file extension."""
    suffix = path.suffix.lower()
    type_map: dict[str, DocumentType] = {
        ".md": "markdown",
        ".markdown": "markdown",
        ".txt": "text",
        ".html": "html",
        ".htm": "html",
        ".pdf": "pdf",
    }
    return type_map.get(suffix, "text")


def extract_confluence_metadata(content: str) -> tuple[str | None, str | None]:
    """
    Extract title and URL from Confluence export format.

    Format:
        Навигация: ...
        ID страницы: 123456
        Заголовок страницы: Page Title
        URL страницы: https://confluence.example.com/...

    Returns:
        (title, url) tuple, either can be None if not found
    """
    title = None
    url = None

    # Extract title from "Заголовок страницы:"
    match = re.search(r"^Заголовок страницы:\s*(.+)$", content, re.MULTILINE)
    if match:
        title = match.group(1).strip()

    # Extract URL from "URL страницы:"
    match = re.search(r"^URL страницы:\s*(https?://\S+)", content, re.MULTILINE)
    if match:
        url = match.group(1).strip()

    return title, url


def extract_title_from_markdown(content: str) -> str:
    """Extract title from markdown content (first h1 or filename)."""
    # First try Confluence format
    title, _ = extract_confluence_metadata(content)
    if title:
        return title

    # Look for # heading
    match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    # Look for --- title block
    match = re.search(r"^title:\s*[\"']?(.+?)[\"']?\s*$", content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    # Return first non-empty line
    for line in content.split("\n"):
        line = line.strip()
        if line and not line.startswith("---"):
            return line[:100]
    return "Untitled"


def clean_markdown(content: str) -> str:
    """Clean markdown content for processing."""
    # Remove frontmatter
    content = re.sub(r"^---\n.*?\n---\n", "", content, flags=re.DOTALL)
    # Normalize whitespace
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content.strip()


def chunk_content(
    content: str,
    chunk_size: int = 2000,
    overlap: int = 200,
) -> list[tuple[str, int, int]]:
    """
    Split content into overlapping chunks.

    Returns list of (chunk_text, start_offset, end_offset).
    Tries to split on paragraph boundaries.
    """
    if len(content) <= chunk_size:
        return [(content, 0, len(content))]

    chunks: list[tuple[str, int, int]] = []
    paragraphs = re.split(r"\n\n+", content)

    current_chunk = ""
    current_start = 0
    current_pos = 0

    for para in paragraphs:
        para_with_sep = para + "\n\n"

        # If adding this paragraph exceeds chunk size
        if len(current_chunk) + len(para_with_sep) > chunk_size:
            if current_chunk:
                # Save current chunk
                chunks.append((
                    current_chunk.strip(),
                    current_start,
                    current_pos,
                ))
                # Start new chunk with overlap
                overlap_start = max(0, len(current_chunk) - overlap)
                current_chunk = current_chunk[overlap_start:] + para_with_sep
                current_start = current_pos - (len(current_chunk) - len(para_with_sep))
            else:
                # Single paragraph too large, force split
                current_chunk = para_with_sep
        else:
            current_chunk += para_with_sep

        current_pos += len(para_with_sep)

    # Don't forget last chunk
    if current_chunk.strip():
        chunks.append((
            current_chunk.strip(),
            current_start,
            current_pos,
        ))

    return chunks


async def parse_file(path: Path) -> Document:
    """Parse a file into a Document."""
    async with aiofiles.open(path, encoding="utf-8") as f:
        content = await f.read()

    doc_type = detect_doc_type(path)

    # Try to extract Confluence metadata (title and URL)
    confluence_title, confluence_url = extract_confluence_metadata(content)

    if doc_type == "markdown":
        title = extract_title_from_markdown(content)
        content = clean_markdown(content)
    else:
        # For txt files, prefer Confluence title, fallback to filename
        title = confluence_title or path.stem

    # Use Confluence URL if available, otherwise use local file path
    source_path = confluence_url or str(path.absolute())

    return Document(
        id=generate_id(),
        title=title,
        content=content,
        doc_type=doc_type,
        source_path=source_path,
        source_hash=compute_hash(content),
    )


def parse_content(content: str, title: str | None = None) -> Document:
    """Parse raw content into a Document."""
    if title is None:
        title = extract_title_from_markdown(content)

    cleaned = clean_markdown(content)

    return Document(
        id=generate_id(),
        title=title,
        content=cleaned,
        doc_type="markdown",
        source_hash=compute_hash(content),
    )


def create_chunks(document: Document, chunk_size: int = 2000) -> list[DocumentChunk]:
    """Create chunks from a document."""
    raw_chunks = chunk_content(document.content, chunk_size=chunk_size)

    return [
        DocumentChunk(
            id=generate_id(),
            document_id=document.id,
            content=text,
            chunk_index=idx,
            start_offset=start,
            end_offset=end,
        )
        for idx, (text, start, end) in enumerate(raw_chunks)
    ]


class DocumentParser:
    """Parser for loading and chunking documents."""

    def __init__(self, chunk_size: int = 2000, overlap: int = 200) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    async def parse_file(self, path: Path | str) -> Document:
        """Parse a file into a Document."""
        if isinstance(path, str):
            path = Path(path)
        return await parse_file(path)

    def parse_content(self, content: str, title: str | None = None) -> Document:
        """Parse raw content into a Document."""
        return parse_content(content, title)

    def chunk_document(self, document: Document) -> list[DocumentChunk]:
        """Split document into chunks."""
        return create_chunks(document, chunk_size=self.chunk_size)

    async def parse_directory(
        self,
        directory: Path | str,
        extensions: list[str] | None = None,
    ) -> list[Document]:
        """Parse all matching files in a directory."""
        if isinstance(directory, str):
            directory = Path(directory)

        extensions = extensions or [".md", ".txt", ".markdown"]
        documents: list[Document] = []

        for ext in extensions:
            for file_path in directory.rglob(f"*{ext}"):
                try:
                    doc = await self.parse_file(file_path)
                    documents.append(doc)
                except Exception as e:
                    # Log but continue on errors
                    print(f"Error parsing {file_path}: {e}")

        return documents
