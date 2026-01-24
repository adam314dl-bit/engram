"""Content context extraction for tables and lists.

Shared module for extracting Confluence hierarchy and document context
to provide rich context for LLM enrichment.

v4.1: Used by both table_enricher and list_enricher.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class ContentType(Enum):
    """Type of content being enriched."""
    TABLE = "table"
    LIST = "list"


@dataclass
class ContentContext:
    """
    Rich context for a table or list, including Confluence hierarchy.

    Used to provide context to LLM enrichment prompts.
    """

    # Confluence hierarchy (from page metadata)
    space_name: str | None = None
    parent_pages: list[str] = field(default_factory=list)  # ["Команды", "Frontend"]
    page_title: str | None = None
    page_url: str | None = None

    # Document structure context
    headings_above: list[str] = field(default_factory=list)  # ["## Команда", "### Участники"]
    text_before: str = ""  # 1-2 paragraphs before the content
    text_after: str = ""  # 1 paragraph after (optional)

    # Content position
    start_line: int = 0
    end_line: int = 0

    def to_prompt_dict(self) -> dict[str, str]:
        """
        Format context for prompt templates.

        Returns dict with keys suitable for prompt formatting.
        """
        parts = {}

        # Build hierarchy string
        hierarchy_parts = []
        if self.space_name:
            hierarchy_parts.append(f"Пространство: {self.space_name}")
        if self.parent_pages:
            hierarchy_parts.append(f"Путь: {' > '.join(self.parent_pages)}")
        if self.page_title:
            hierarchy_parts.append(f"Страница: {self.page_title}")

        parts["hierarchy"] = "\n".join(hierarchy_parts) if hierarchy_parts else ""

        # Headings context
        if self.headings_above:
            parts["headings"] = "\n".join(self.headings_above)
        else:
            parts["headings"] = ""

        # Text context
        parts["text_before"] = self.text_before.strip() if self.text_before else ""
        parts["text_after"] = self.text_after.strip() if self.text_after else ""

        return parts

    def get_best_context_hint(self) -> str:
        """
        Get the most specific context hint available.

        Priority:
        1. Heading directly above content (most specific)
        2. Page title (if meaningful, not generic)
        3. Last parent page
        4. Space name (fallback)

        Returns:
            Single-line context hint
        """
        # Try heading above
        if self.headings_above:
            # Get the most specific (last) heading
            last_heading = self.headings_above[-1]
            # Strip markdown heading markers
            cleaned = re.sub(r"^#+\s*", "", last_heading).strip()
            if cleaned and len(cleaned) > 2:
                return cleaned

        # Try page title (skip generic ones)
        generic_titles = {"команда", "документация", "описание", "информация", "главная", "home", "index"}
        if self.page_title:
            title_lower = self.page_title.lower()
            if title_lower not in generic_titles and len(self.page_title) > 2:
                return self.page_title

        # Try last parent page
        if self.parent_pages:
            return self.parent_pages[-1]

        # Fallback to space name
        if self.space_name:
            return self.space_name

        return ""

    def format_for_prompt(self, content_type: ContentType = ContentType.TABLE) -> str:
        """
        Format full context as a string for LLM prompt.

        Args:
            content_type: Type of content (TABLE or LIST)

        Returns:
            Formatted context string for prompt
        """
        lines = []

        # Add hierarchy
        if self.space_name:
            lines.append(f"Пространство Confluence: {self.space_name}")

        if self.parent_pages:
            lines.append(f"Навигация: {' → '.join(self.parent_pages)}")

        if self.page_title:
            lines.append(f"Страница: {self.page_title}")

        # Add headings
        if self.headings_above:
            lines.append("Заголовки над контентом:")
            for h in self.headings_above:
                lines.append(f"  {h}")

        # Add surrounding text
        if self.text_before:
            lines.append(f"Текст перед {'таблицей' if content_type == ContentType.TABLE else 'списком'}:")
            lines.append(self.text_before.strip())

        return "\n".join(lines) if lines else "Контекст отсутствует."


@dataclass
class PageMetadata:
    """
    Metadata extracted from Confluence page.

    Passed from pipeline to context extractor.
    """
    space_name: str | None = None
    parent_pages: list[str] = field(default_factory=list)
    title: str | None = None
    url: str | None = None


class ContentContextExtractor:
    """
    Extract rich context for tables and lists from markdown.

    Analyzes document structure to provide context for enrichment.
    """

    def __init__(
        self,
        context_paragraphs_before: int = 2,
        context_paragraphs_after: int = 1,
        max_context_chars: int = 500,
    ) -> None:
        """
        Initialize context extractor.

        Args:
            context_paragraphs_before: Number of paragraphs to include before content
            context_paragraphs_after: Number of paragraphs to include after content
            max_context_chars: Maximum characters for surrounding text
        """
        self.context_paragraphs_before = context_paragraphs_before
        self.context_paragraphs_after = context_paragraphs_after
        self.max_context_chars = max_context_chars

    def extract_context(
        self,
        markdown: str,
        start_line: int,
        end_line: int,
        page_metadata: PageMetadata | None = None,
    ) -> ContentContext:
        """
        Extract context for content at the given line range.

        Args:
            markdown: Full document markdown
            start_line: First line of content (0-indexed)
            end_line: Last line of content (0-indexed)
            page_metadata: Optional page metadata from Confluence

        Returns:
            ContentContext with all available context
        """
        lines = markdown.split("\n")

        # Initialize context
        ctx = ContentContext(
            start_line=start_line,
            end_line=end_line,
        )

        # Add page metadata if available
        if page_metadata:
            ctx.space_name = page_metadata.space_name
            ctx.parent_pages = page_metadata.parent_pages.copy()
            ctx.page_title = page_metadata.title
            ctx.page_url = page_metadata.url

        # Extract headings above
        ctx.headings_above = self._find_headings_above(lines, start_line)

        # Extract text before
        ctx.text_before = self._extract_text_before(lines, start_line)

        # Extract text after
        ctx.text_after = self._extract_text_after(lines, end_line)

        return ctx

    def _find_headings_above(self, lines: list[str], content_start: int) -> list[str]:
        """
        Find markdown headings above the content.

        Returns headings in order from highest level to most specific.
        """
        headings: list[str] = []
        heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$")

        # Track heading levels to maintain hierarchy
        level_stack: list[tuple[int, str]] = []

        for i in range(content_start):
            line = lines[i].strip()
            match = heading_pattern.match(line)
            if match:
                level = len(match.group(1))
                heading_text = line

                # Pop headings of same or lower level (deeper nesting)
                while level_stack and level_stack[-1][0] >= level:
                    level_stack.pop()

                level_stack.append((level, heading_text))

        # Return headings in hierarchy order
        headings = [h for _, h in level_stack]

        return headings

    def _extract_text_before(self, lines: list[str], content_start: int) -> str:
        """Extract text paragraphs before the content."""
        if content_start <= 0:
            return ""

        # Collect non-empty, non-heading lines before content
        text_lines: list[str] = []
        heading_pattern = re.compile(r"^#{1,6}\s+")

        for i in range(content_start - 1, -1, -1):
            line = lines[i].strip()

            # Skip empty lines
            if not line:
                if text_lines:
                    # Found a paragraph break
                    break
                continue

            # Skip headings
            if heading_pattern.match(line):
                break

            # Skip table lines
            if line.startswith("|"):
                continue

            # Skip list markers at start of paragraph
            if line.startswith(("-", "*", "•")) and not text_lines:
                continue

            text_lines.insert(0, line)

            # Limit context size
            if len(" ".join(text_lines)) > self.max_context_chars:
                break

        return " ".join(text_lines)

    def _extract_text_after(self, lines: list[str], content_end: int) -> str:
        """Extract text paragraph after the content."""
        if content_end >= len(lines) - 1:
            return ""

        # Collect non-empty, non-heading lines after content
        text_lines: list[str] = []
        heading_pattern = re.compile(r"^#{1,6}\s+")

        for i in range(content_end + 1, len(lines)):
            line = lines[i].strip()

            # Skip empty lines at start
            if not line:
                if text_lines:
                    break
                continue

            # Stop at headings
            if heading_pattern.match(line):
                break

            # Stop at new table or list
            if line.startswith("|") or line.startswith(("-", "*", "•")):
                break

            text_lines.append(line)

            # Limit to one paragraph
            if len(" ".join(text_lines)) > self.max_context_chars // 2:
                break

        return " ".join(text_lines)


def extract_navigation_path(content: str) -> list[str]:
    """
    Extract navigation path from Confluence export.

    Looks for "Навигация:" line and parses the breadcrumb path.

    Args:
        content: Document content

    Returns:
        List of navigation items, e.g., ["Главная", "Команды", "Frontend"]
    """
    match = re.search(r"^Навигация:\s*(.+)$", content, re.MULTILINE)
    if not match:
        return []

    nav_text = match.group(1).strip()

    # Common separators: " > ", " / ", " → "
    for sep in [" > ", " → ", " / "]:
        if sep in nav_text:
            return [part.strip() for part in nav_text.split(sep) if part.strip()]

    # If no separator found, return as single item
    return [nav_text] if nav_text else []


def extract_page_metadata_from_content(content: str) -> PageMetadata:
    """
    Extract page metadata from Confluence export format.

    Parses:
    - Навигация: Parent > Child > Page
    - Заголовок страницы: Page Title
    - URL страницы: https://...

    Args:
        content: Document content

    Returns:
        PageMetadata with extracted values
    """
    metadata = PageMetadata()

    # Extract navigation path
    nav_path = extract_navigation_path(content)
    if nav_path:
        # First item is usually space name
        if len(nav_path) > 0:
            metadata.space_name = nav_path[0]
        # Middle items are parent pages
        if len(nav_path) > 1:
            metadata.parent_pages = nav_path[1:]

    # Extract title
    match = re.search(r"^Заголовок страницы:\s*(.+)$", content, re.MULTILINE)
    if match:
        metadata.title = match.group(1).strip()

    # Extract URL
    match = re.search(r"^URL страницы:\s*(https?://\S+)", content, re.MULTILINE)
    if match:
        metadata.url = match.group(1).strip()

    return metadata
