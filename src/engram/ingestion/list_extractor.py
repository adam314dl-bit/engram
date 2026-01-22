"""Structure-aware list extraction for improved retrieval.

Extracts definition lists, procedures, and bullet lists from documents
while preserving context (section headers, surrounding text).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ListItem:
    """A single item in a list."""

    term: str  # The term or key (for definition lists) or item text
    definition: str | None = None  # Definition text (for definition lists)
    index: int = 0  # Position in list (for numbered lists)
    sublists: list[ListItem] = field(default_factory=list)  # Nested items


@dataclass
class ExtractedList:
    """An extracted list with its context."""

    type: str  # "definition", "procedure", "bullet"
    items: list[ListItem]
    parent_context: str  # Section header + intro paragraph
    source_section: str | None = None  # Section header if found
    raw_text: str = ""  # Original list text


# Russian patterns for list detection
DEFINITION_PATTERNS = [
    # Em-dash pattern: "Термин — определение"
    r"^(.+?)\s*[—–-]\s+(.+)$",
    # Colon pattern: "Термин: определение"
    r"^(.+?):\s+(.{20,})$",
    # Arrow pattern: "Термин → определение"
    r"^(.+?)\s*[→=>]\s+(.+)$",
]

PROCEDURE_PATTERNS = [
    # Numbered steps: "1. Сделать что-то" or "1) Сделать"
    r"^(\d+)[.)]\s+(.+)$",
    # Step keywords: "Шаг 1: действие"
    r"^[Шш]аг\s+(\d+)[.:]\s*(.+)$",
]

BULLET_PATTERNS = [
    # Standard bullets: "- item" or "* item" or "• item"
    r"^[-*•●◦]\s+(.+)$",
    # Russian-style: "— item"
    r"^[—–]\s+(.+)$",
]


class ListExtractor:
    """Extract and structure lists from document text."""

    def __init__(self, min_items: int = 2, max_context_chars: int = 500) -> None:
        """
        Initialize list extractor.

        Args:
            min_items: Minimum items to consider a valid list
            max_context_chars: Maximum characters for context preservation
        """
        self.min_items = min_items
        self.max_context_chars = max_context_chars

    def extract_lists(self, text: str) -> list[ExtractedList]:
        """
        Extract all lists from text.

        Args:
            text: Document text

        Returns:
            List of ExtractedList objects
        """
        results: list[ExtractedList] = []

        # Split into sections by headers
        sections = self._split_by_sections(text)

        for section_header, section_text in sections:
            # Try to extract different list types
            definition_lists = self._extract_definition_lists(
                section_text, section_header
            )
            results.extend(definition_lists)

            procedure_lists = self._extract_procedure_lists(
                section_text, section_header
            )
            results.extend(procedure_lists)

            bullet_lists = self._extract_bullet_lists(
                section_text, section_header
            )
            results.extend(bullet_lists)

        return results

    def _split_by_sections(self, text: str) -> list[tuple[str | None, str]]:
        """
        Split text into sections by markdown headers.

        Returns:
            List of (header, section_text) tuples
        """
        # Pattern for markdown headers
        header_pattern = r"^(#{1,6})\s+(.+)$"

        lines = text.split("\n")
        sections: list[tuple[str | None, str]] = []
        current_header: str | None = None
        current_lines: list[str] = []

        for line in lines:
            match = re.match(header_pattern, line.strip())
            if match:
                # Save previous section
                if current_lines:
                    sections.append((current_header, "\n".join(current_lines)))
                # Start new section
                current_header = match.group(2)
                current_lines = []
            else:
                current_lines.append(line)

        # Save last section
        if current_lines:
            sections.append((current_header, "\n".join(current_lines)))

        return sections if sections else [(None, text)]

    def _get_context(self, text: str, list_start: int) -> str:
        """
        Extract context before a list.

        Args:
            text: Full section text
            list_start: Character position where list starts

        Returns:
            Context string (intro paragraph before list)
        """
        if list_start <= 0:
            return ""

        # Get text before list
        before = text[:list_start].strip()

        # Find the last paragraph
        paragraphs = before.split("\n\n")
        if paragraphs:
            context = paragraphs[-1].strip()
            # Truncate if too long
            if len(context) > self.max_context_chars:
                context = context[-self.max_context_chars:]
            return context

        return ""

    def _extract_definition_lists(
        self, text: str, section_header: str | None
    ) -> list[ExtractedList]:
        """Extract definition-style lists (term — definition)."""
        results: list[ExtractedList] = []
        lines = text.split("\n")
        current_items: list[ListItem] = []
        list_start = 0
        raw_lines: list[str] = []

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                # Empty line might end a list
                if len(current_items) >= self.min_items:
                    context = self._get_context(text, list_start)
                    parent_context = f"{section_header}: {context}" if section_header else context
                    results.append(ExtractedList(
                        type="definition",
                        items=current_items,
                        parent_context=parent_context.strip(),
                        source_section=section_header,
                        raw_text="\n".join(raw_lines),
                    ))
                current_items = []
                raw_lines = []
                continue

            # Try definition patterns
            matched = False
            for pattern in DEFINITION_PATTERNS:
                match = re.match(pattern, line)
                if match:
                    if not current_items:
                        # Record start position
                        list_start = text.find(line)
                    term = match.group(1).strip()
                    definition = match.group(2).strip()
                    # Filter out false positives (very short terms or definitions)
                    if len(term) > 2 and len(definition) > 10:
                        current_items.append(ListItem(
                            term=term,
                            definition=definition,
                            index=len(current_items),
                        ))
                        raw_lines.append(line)
                        matched = True
                        break

            # If line doesn't match but we have items, might be end of list
            if not matched and current_items:
                if len(current_items) >= self.min_items:
                    context = self._get_context(text, list_start)
                    parent_context = f"{section_header}: {context}" if section_header else context
                    results.append(ExtractedList(
                        type="definition",
                        items=current_items,
                        parent_context=parent_context.strip(),
                        source_section=section_header,
                        raw_text="\n".join(raw_lines),
                    ))
                current_items = []
                raw_lines = []

        # Handle end of text
        if len(current_items) >= self.min_items:
            context = self._get_context(text, list_start)
            parent_context = f"{section_header}: {context}" if section_header else context
            results.append(ExtractedList(
                type="definition",
                items=current_items,
                parent_context=parent_context.strip(),
                source_section=section_header,
                raw_text="\n".join(raw_lines),
            ))

        return results

    def _extract_procedure_lists(
        self, text: str, section_header: str | None
    ) -> list[ExtractedList]:
        """Extract numbered procedure lists (1. Do this, 2. Do that)."""
        results: list[ExtractedList] = []
        lines = text.split("\n")
        current_items: list[ListItem] = []
        expected_num = 1
        list_start = 0
        raw_lines: list[str] = []

        for line in lines:
            line = line.strip()
            if not line:
                if len(current_items) >= self.min_items:
                    context = self._get_context(text, list_start)
                    parent_context = f"{section_header}: {context}" if section_header else context
                    results.append(ExtractedList(
                        type="procedure",
                        items=current_items,
                        parent_context=parent_context.strip(),
                        source_section=section_header,
                        raw_text="\n".join(raw_lines),
                    ))
                current_items = []
                expected_num = 1
                raw_lines = []
                continue

            # Try procedure patterns
            for pattern in PROCEDURE_PATTERNS:
                match = re.match(pattern, line)
                if match:
                    num = int(match.group(1))
                    content = match.group(2).strip()

                    # Check for sequential numbering (allow some gaps)
                    if num == expected_num or num == expected_num + 1 or not current_items:
                        if not current_items:
                            list_start = text.find(line)
                        current_items.append(ListItem(
                            term=content,
                            index=num,
                        ))
                        raw_lines.append(line)
                        expected_num = num + 1
                    break

        # Handle end of text
        if len(current_items) >= self.min_items:
            context = self._get_context(text, list_start)
            parent_context = f"{section_header}: {context}" if section_header else context
            results.append(ExtractedList(
                type="procedure",
                items=current_items,
                parent_context=parent_context.strip(),
                source_section=section_header,
                raw_text="\n".join(raw_lines),
            ))

        return results

    def _extract_bullet_lists(
        self, text: str, section_header: str | None
    ) -> list[ExtractedList]:
        """Extract bullet lists (- item, * item)."""
        results: list[ExtractedList] = []
        lines = text.split("\n")
        current_items: list[ListItem] = []
        list_start = 0
        raw_lines: list[str] = []

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                if len(current_items) >= self.min_items:
                    context = self._get_context(text, list_start)
                    parent_context = f"{section_header}: {context}" if section_header else context
                    results.append(ExtractedList(
                        type="bullet",
                        items=current_items,
                        parent_context=parent_context.strip(),
                        source_section=section_header,
                        raw_text="\n".join(raw_lines),
                    ))
                current_items = []
                raw_lines = []
                continue

            # Try bullet patterns
            matched = False
            for pattern in BULLET_PATTERNS:
                match = re.match(pattern, line_stripped)
                if match:
                    if not current_items:
                        list_start = text.find(line_stripped)
                    content = match.group(1).strip()
                    if len(content) > 5:  # Filter very short items
                        current_items.append(ListItem(
                            term=content,
                            index=len(current_items),
                        ))
                        raw_lines.append(line_stripped)
                        matched = True
                        break

            # Non-matching line might end the list
            if not matched and current_items:
                if len(current_items) >= self.min_items:
                    context = self._get_context(text, list_start)
                    parent_context = f"{section_header}: {context}" if section_header else context
                    results.append(ExtractedList(
                        type="bullet",
                        items=current_items,
                        parent_context=parent_context.strip(),
                        source_section=section_header,
                        raw_text="\n".join(raw_lines),
                    ))
                current_items = []
                raw_lines = []

        # Handle end of text
        if len(current_items) >= self.min_items:
            context = self._get_context(text, list_start)
            parent_context = f"{section_header}: {context}" if section_header else context
            results.append(ExtractedList(
                type="bullet",
                items=current_items,
                parent_context=parent_context.strip(),
                source_section=section_header,
                raw_text="\n".join(raw_lines),
            ))

        return results

    def to_grouped_chunks(
        self,
        extracted_list: ExtractedList,
        items_per_chunk: int = 4,
        complex_item_standalone: bool = True,
    ) -> list[str]:
        """
        Convert extracted list to grouped chunks for indexing.

        Groups 3-5 simple items together, keeps complex items standalone.

        Args:
            extracted_list: List to convert
            items_per_chunk: Target items per chunk
            complex_item_standalone: Whether complex items get their own chunk

        Returns:
            List of chunk strings with context prepended
        """
        chunks: list[str] = []
        context_prefix = extracted_list.parent_context

        if extracted_list.type == "definition":
            # Group definition items
            current_group: list[str] = []
            for item in extracted_list.items:
                item_text = f"{item.term}: {item.definition}"

                # Complex item check: long definition or has multiple sentences
                is_complex = (
                    item.definition and
                    (len(item.definition) > 200 or item.definition.count(".") > 2)
                )

                if is_complex and complex_item_standalone:
                    # Save current group
                    if current_group:
                        chunk = f"{context_prefix}\n\n" + "\n".join(current_group)
                        chunks.append(chunk.strip())
                        current_group = []
                    # Add complex item as standalone
                    chunks.append(f"{context_prefix}\n\n{item_text}".strip())
                else:
                    current_group.append(item_text)
                    if len(current_group) >= items_per_chunk:
                        chunk = f"{context_prefix}\n\n" + "\n".join(current_group)
                        chunks.append(chunk.strip())
                        current_group = []

            # Remaining items
            if current_group:
                chunk = f"{context_prefix}\n\n" + "\n".join(current_group)
                chunks.append(chunk.strip())

        elif extracted_list.type == "procedure":
            # Procedures: keep steps together for coherence
            steps = [f"{item.index}. {item.term}" for item in extracted_list.items]
            # Group by items_per_chunk
            for i in range(0, len(steps), items_per_chunk):
                group = steps[i:i + items_per_chunk]
                chunk = f"{context_prefix}\n\n" + "\n".join(group)
                chunks.append(chunk.strip())

        else:  # bullet list
            # Group bullet items
            items = [f"- {item.term}" for item in extracted_list.items]
            for i in range(0, len(items), items_per_chunk):
                group = items[i:i + items_per_chunk]
                chunk = f"{context_prefix}\n\n" + "\n".join(group)
                chunks.append(chunk.strip())

        return chunks


def extract_lists_from_text(text: str) -> list[ExtractedList]:
    """
    Convenience function to extract lists from text.

    Args:
        text: Document text

    Returns:
        List of ExtractedList objects
    """
    extractor = ListExtractor()
    return extractor.extract_lists(text)


def lists_to_chunks(
    extracted_lists: list[ExtractedList],
    items_per_chunk: int = 4,
) -> list[str]:
    """
    Convert extracted lists to chunks for indexing.

    Args:
        extracted_lists: Lists to convert
        items_per_chunk: Items per chunk

    Returns:
        List of chunk strings
    """
    extractor = ListExtractor()
    chunks: list[str] = []
    for lst in extracted_lists:
        chunks.extend(extractor.to_grouped_chunks(lst, items_per_chunk))
    return chunks
