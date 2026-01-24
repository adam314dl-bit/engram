"""List parser for detecting and parsing markdown lists.

v4.1: Simple regex-based detection that returns list positions for context extraction.
Works with list_enricher.py for LLM-based enrichment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class ListType(Enum):
    """Type of list detected."""
    BULLET = "bullet"  # - item, * item, • item
    NUMBERED = "numbered"  # 1. item, 1) item
    CHECKLIST = "checklist"  # - [ ] item, - [x] item
    DEFINITION = "definition"  # term: definition, term — definition


@dataclass
class ParsedList:
    """
    Represents a parsed markdown list.

    Mirrors ParsedTable structure for consistency.
    """

    items: list[str]  # List items (text only, markers stripped)
    raw_markdown: str  # Full list markdown
    list_type: ListType = ListType.BULLET
    start_line: int = 0
    end_line: int = 0
    title: str | None = None  # Heading above list if found
    context: str | None = None  # Text before the list

    @property
    def item_count(self) -> int:
        """Number of items in the list."""
        return len(self.items)

    def to_markdown(self) -> str:
        """Return raw markdown representation."""
        return self.raw_markdown

    def to_text(self) -> str:
        """
        Convert list to plain text representation.

        Useful for embedding search.
        """
        parts = []
        if self.title:
            parts.append(self.title)
        parts.extend(self.items)
        return "\n".join(parts)


@dataclass
class ListBoundary:
    """Represents the location of a list in text."""

    start_line: int
    end_line: int
    list_type: ListType
    title: str | None = None
    context_lines: list[str] = field(default_factory=list)


# Patterns for list detection
BULLET_PATTERN = re.compile(r"^(\s*)[-*•●◦]\s+(.+)$")
NUMBERED_PATTERN = re.compile(r"^(\s*)(\d+)[.)]\s+(.+)$")
CHECKLIST_PATTERN = re.compile(r"^(\s*)[-*]\s*\[([ xX])\]\s+(.+)$")
DEFINITION_PATTERN = re.compile(r"^(.+?)\s*[—–:]\s+(.{15,})$")  # At least 15 chars in definition


class ListParser:
    """
    Parse and detect lists from markdown text.

    Detects:
    - Bullet lists (-, *, •)
    - Numbered lists (1., 1))
    - Checklists (- [ ], - [x])
    - Definition lists (term: definition, term — definition)
    """

    def __init__(self, min_items: int = 2) -> None:
        """
        Initialize list parser.

        Args:
            min_items: Minimum items to consider a valid list
        """
        self.min_items = min_items

    def detect_lists(self, markdown: str) -> list[tuple[int, int, ListType]]:
        """
        Detect list boundaries in markdown text.

        Args:
            markdown: Markdown text to scan

        Returns:
            List of (start_line, end_line, list_type) tuples
        """
        lines = markdown.split("\n")
        boundaries: list[tuple[int, int, ListType]] = []

        i = 0
        while i < len(lines):
            line = lines[i]

            # Try each list type
            list_type = self._detect_line_type(line)
            if list_type:
                start = i
                end = self._find_list_end(lines, i, list_type)

                # Check minimum items
                item_count = end - start + 1
                if item_count >= self.min_items:
                    boundaries.append((start, end, list_type))
                    i = end + 1
                    continue

            i += 1

        return boundaries

    def _detect_line_type(self, line: str) -> ListType | None:
        """Detect if a line starts a list item."""
        stripped = line.strip()
        if not stripped:
            return None

        if CHECKLIST_PATTERN.match(stripped):
            return ListType.CHECKLIST
        if NUMBERED_PATTERN.match(stripped):
            return ListType.NUMBERED
        if BULLET_PATTERN.match(stripped):
            return ListType.BULLET
        if DEFINITION_PATTERN.match(stripped):
            return ListType.DEFINITION

        return None

    def _find_list_end(self, lines: list[str], start: int, list_type: ListType) -> int:
        """Find the end of a list starting at the given line."""
        end = start
        consecutive_empty = 0
        expected_num = None

        if list_type == ListType.NUMBERED:
            match = NUMBERED_PATTERN.match(lines[start].strip())
            if match:
                expected_num = int(match.group(2))

        for i in range(start, len(lines)):
            line = lines[i]
            stripped = line.strip()

            # Handle empty lines
            if not stripped:
                consecutive_empty += 1
                if consecutive_empty > 1:
                    # Two empty lines end the list
                    break
                continue

            consecutive_empty = 0

            # Check if line matches expected list type
            detected = self._detect_line_type(stripped)

            if list_type == ListType.NUMBERED and detected == ListType.NUMBERED:
                match = NUMBERED_PATTERN.match(stripped)
                if match:
                    num = int(match.group(2))
                    # Allow some gaps in numbering
                    if expected_num is None or num <= expected_num + 2:
                        end = i
                        expected_num = num + 1
                        continue
                break

            if detected == list_type:
                end = i
            elif detected is not None or stripped.startswith("#"):
                # Different list type or heading ends the list
                break
            elif line.startswith("    ") or line.startswith("\t"):
                # Indented continuation
                end = i
            else:
                # Non-matching line ends the list
                break

        return end

    def parse_list(
        self,
        markdown: str,
        start_line: int,
        end_line: int,
        list_type: ListType,
    ) -> ParsedList:
        """
        Parse a list given its boundaries.

        Args:
            markdown: Full document markdown
            start_line: First line of list (0-indexed)
            end_line: Last line of list (0-indexed)
            list_type: Type of list

        Returns:
            ParsedList with extracted items
        """
        lines = markdown.split("\n")
        list_lines = lines[start_line:end_line + 1]
        raw_markdown = "\n".join(list_lines)

        items = self._extract_items(list_lines, list_type)

        # Look for title (heading above)
        title = None
        for i in range(start_line - 1, max(0, start_line - 5), -1):
            prev_line = lines[i].strip()
            if prev_line.startswith("#"):
                title = prev_line.lstrip("#").strip()
                break
            if prev_line:
                break

        # Get context (text before list)
        context_parts = []
        for i in range(start_line - 1, max(0, start_line - 3), -1):
            prev_line = lines[i].strip()
            if not prev_line:
                break
            if prev_line.startswith("#"):
                break
            if not prev_line.startswith(("-", "*", "•", "|")):
                context_parts.insert(0, prev_line)

        context = " ".join(context_parts) if context_parts else None

        return ParsedList(
            items=items,
            raw_markdown=raw_markdown,
            list_type=list_type,
            start_line=start_line,
            end_line=end_line,
            title=title,
            context=context,
        )

    def _extract_items(self, lines: list[str], list_type: ListType) -> list[str]:
        """Extract item text from list lines."""
        items: list[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            item_text = self._extract_item_text(stripped, list_type)
            if item_text:
                items.append(item_text)

        return items

    def _extract_item_text(self, line: str, list_type: ListType) -> str | None:
        """Extract the text content from a list item line."""
        if list_type == ListType.CHECKLIST:
            match = CHECKLIST_PATTERN.match(line)
            if match:
                return match.group(3).strip()

        if list_type == ListType.NUMBERED:
            match = NUMBERED_PATTERN.match(line)
            if match:
                return match.group(3).strip()

        if list_type == ListType.BULLET:
            match = BULLET_PATTERN.match(line)
            if match:
                return match.group(2).strip()

        if list_type == ListType.DEFINITION:
            match = DEFINITION_PATTERN.match(line)
            if match:
                term = match.group(1).strip()
                definition = match.group(2).strip()
                return f"{term}: {definition}"

        return None


def extract_lists(markdown: str, min_items: int = 2) -> list[ParsedList]:
    """
    Extract all lists from markdown text.

    Args:
        markdown: Markdown text containing lists
        min_items: Minimum items to consider a valid list

    Returns:
        List of ParsedList objects
    """
    parser = ListParser(min_items=min_items)
    boundaries = parser.detect_lists(markdown)

    lists: list[ParsedList] = []
    for start_line, end_line, list_type in boundaries:
        parsed = parser.parse_list(markdown, start_line, end_line, list_type)
        lists.append(parsed)

    return lists


def remove_lists_from_text(markdown: str, placeholder: str = "[СПИСОК]") -> str:
    """
    Remove lists from markdown text, replacing with placeholder.

    Args:
        markdown: Markdown text containing lists
        placeholder: Text to insert where lists were removed

    Returns:
        Text with lists replaced by placeholder
    """
    parser = ListParser()
    boundaries = parser.detect_lists(markdown)

    if not boundaries:
        return markdown

    lines = markdown.split("\n")
    result_lines: list[str] = []
    last_end = 0

    for start, end, _ in sorted(boundaries, key=lambda x: x[0]):
        # Add lines before this list
        result_lines.extend(lines[last_end:start])

        # Add placeholder
        result_lines.append(placeholder)
        result_lines.append("")  # Blank line after placeholder

        last_end = end + 1

    # Add remaining lines after last list
    result_lines.extend(lines[last_end:])

    # Clean up multiple blank lines
    text = "\n".join(result_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
