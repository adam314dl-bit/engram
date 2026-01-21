"""Table parser for extracting and parsing markdown tables."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from engram.preprocessing.normalizer import normalize_table_cell


@dataclass
class ParsedTable:
    """Represents a parsed markdown table."""

    headers: list[str]
    rows: list[list[str]]
    title: str | None = None
    context: str | None = None  # Text before the table
    start_line: int = 0
    end_line: int = 0
    raw_text: str = ""

    @property
    def is_comparison_table(self) -> bool:
        """
        Check if this is a comparison/feature matrix table.

        Comparison tables typically have:
        - Multiple columns with product/option names
        - Rows with features
        - да/нет or +/- values
        """
        if len(self.headers) < 2:
            return False

        # Count boolean-like cells
        boolean_cells = 0
        total_cells = 0

        for row in self.rows:
            for cell in row[1:]:  # Skip first column (usually feature name)
                total_cells += 1
                normalized = normalize_table_cell(cell)
                if normalized in ("да", "нет", ""):
                    boolean_cells += 1

        # If more than 50% of data cells are boolean, it's likely a comparison table
        if total_cells > 0 and boolean_cells / total_cells > 0.5:
            return True

        return False

    @property
    def column_count(self) -> int:
        """Number of columns in the table."""
        return len(self.headers)

    @property
    def row_count(self) -> int:
        """Number of data rows (excluding header)."""
        return len(self.rows)

    def to_markdown(self) -> str:
        """Convert table back to markdown format."""
        if not self.headers:
            return ""

        lines = []

        # Title if present
        if self.title:
            lines.append(f"### {self.title}")
            lines.append("")

        # Header row
        header_line = "| " + " | ".join(self.headers) + " |"
        lines.append(header_line)

        # Separator row
        separator = "| " + " | ".join(["---"] * len(self.headers)) + " |"
        lines.append(separator)

        # Data rows
        for row in self.rows:
            # Pad row if needed
            padded_row = row + [""] * (len(self.headers) - len(row))
            row_line = "| " + " | ".join(padded_row) + " |"
            lines.append(row_line)

        return "\n".join(lines)

    def to_row_texts(self) -> list[str]:
        """
        Convert table to row-oriented text descriptions.

        Each row becomes a sentence like:
        "Feature X: Product A - да, Product B - нет"

        Returns:
            List of row descriptions
        """
        if not self.headers or not self.rows:
            return []

        texts = []
        for row in self.rows:
            if not row:
                continue

            # First cell is usually the row label/feature
            feature = row[0] if row else ""
            values = []

            for i, cell in enumerate(row[1:], 1):
                if i < len(self.headers):
                    header = self.headers[i]
                    if cell:
                        values.append(f"{header}: {cell}")

            if feature and values:
                text = f"{feature} — " + ", ".join(values)
                texts.append(text)
            elif feature:
                texts.append(feature)

        return texts

    def to_column_texts(self) -> list[str]:
        """
        Convert table to column-oriented text descriptions.

        Each column becomes a summary like:
        "Product A: Feature 1 - да, Feature 2 - нет"

        Useful for comparison tables.

        Returns:
            List of column descriptions
        """
        if not self.headers or not self.rows:
            return []

        texts = []

        # Skip first column (usually feature names)
        for col_idx in range(1, len(self.headers)):
            header = self.headers[col_idx]
            features = []

            for row in self.rows:
                if col_idx < len(row):
                    feature_name = row[0] if row else ""
                    value = row[col_idx]
                    if feature_name and value:
                        features.append(f"{feature_name}: {value}")

            if features:
                text = f"{header}: " + ", ".join(features)
                texts.append(text)

        return texts

    def to_searchable_text(self) -> str:
        """
        Convert table to searchable text representation.

        Combines table metadata and content into a single text block
        suitable for embedding and search.

        Returns:
            Searchable text representation
        """
        parts = []

        if self.title:
            parts.append(f"Таблица: {self.title}")

        if self.context:
            parts.append(f"Контекст: {self.context}")

        # Add headers
        if self.headers:
            parts.append("Столбцы: " + ", ".join(self.headers))

        # Add row descriptions
        row_texts = self.to_row_texts()
        if row_texts:
            parts.append("Содержимое:")
            parts.extend(row_texts)

        return "\n".join(parts)


@dataclass
class TableBoundary:
    """Represents the location of a table in text."""

    start_line: int
    end_line: int
    title: str | None = None
    context_lines: list[str] = field(default_factory=list)


def detect_tables(markdown: str) -> list[TableBoundary]:
    """
    Detect table boundaries in markdown text.

    Args:
        markdown: Markdown text to scan

    Returns:
        List of TableBoundary objects indicating table locations
    """
    lines = markdown.split("\n")
    boundaries: list[TableBoundary] = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Look for table start (line starting with |)
        if line.startswith("|") and "|" in line[1:]:
            start_line = i

            # Look back for title (heading before table)
            title = None
            context_lines = []
            for j in range(i - 1, max(0, i - 5), -1):
                prev_line = lines[j].strip()
                if prev_line.startswith("#"):
                    title = prev_line.lstrip("#").strip()
                    break
                elif prev_line and not prev_line.startswith("|"):
                    context_lines.insert(0, prev_line)
                elif not prev_line:
                    # Empty line - stop looking for context
                    break

            # Find table end
            end_line = i
            while end_line < len(lines):
                current = lines[end_line].strip()
                if current.startswith("|"):
                    end_line += 1
                else:
                    break

            boundaries.append(TableBoundary(
                start_line=start_line,
                end_line=end_line - 1,  # Last table line
                title=title,
                context_lines=context_lines,
            ))

            i = end_line
        else:
            i += 1

    return boundaries


def parse_table(text: str) -> ParsedTable | None:
    """
    Parse a markdown table into structured format.

    Args:
        text: Markdown table text (just the table, no surrounding content)

    Returns:
        ParsedTable object or None if parsing fails
    """
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]

    if len(lines) < 2:
        return None

    # Parse header row
    header_line = lines[0]
    if not header_line.startswith("|"):
        return None

    headers = [cell.strip() for cell in header_line.split("|")]
    # Remove empty first and last elements from split
    headers = [h for h in headers if h or headers.index(h) not in (0, len(headers) - 1)]
    headers = [h.strip() for h in headers if h.strip()]

    if not headers:
        return None

    # Skip separator row (|---|---|)
    data_start = 1
    if len(lines) > 1:
        second_line = lines[1]
        if re.match(r"^\|[\s\-:|]+\|$", second_line):
            data_start = 2

    # Parse data rows
    rows: list[list[str]] = []
    for line in lines[data_start:]:
        if not line.startswith("|"):
            continue

        cells = line.split("|")
        # Clean up cells
        row = []
        for cell in cells:
            cleaned = cell.strip()
            if cleaned or len(row) < len(headers):
                row.append(cleaned)

        # Remove empty first and last cells from split
        if row and not row[0]:
            row = row[1:]
        if row and not row[-1]:
            row = row[:-1]

        if row:
            rows.append(row)

    if not rows:
        return None

    return ParsedTable(
        headers=headers,
        rows=rows,
        raw_text=text,
    )


def extract_tables(markdown: str) -> list[ParsedTable]:
    """
    Extract all tables from markdown text.

    Args:
        markdown: Markdown text containing tables

    Returns:
        List of ParsedTable objects with context
    """
    boundaries = detect_tables(markdown)
    lines = markdown.split("\n")
    tables: list[ParsedTable] = []

    for boundary in boundaries:
        # Extract table text
        table_lines = lines[boundary.start_line:boundary.end_line + 1]
        table_text = "\n".join(table_lines)

        parsed = parse_table(table_text)
        if parsed:
            parsed.title = boundary.title
            parsed.context = " ".join(boundary.context_lines) if boundary.context_lines else None
            parsed.start_line = boundary.start_line
            parsed.end_line = boundary.end_line
            tables.append(parsed)

    return tables


def remove_tables_from_text(markdown: str, placeholder: str = "[ТАБЛИЦА]") -> str:
    """
    Remove tables from markdown text, replacing with placeholder.

    Args:
        markdown: Markdown text containing tables
        placeholder: Text to insert where tables were removed

    Returns:
        Text with tables replaced by placeholder
    """
    boundaries = detect_tables(markdown)

    if not boundaries:
        return markdown

    lines = markdown.split("\n")
    result_lines: list[str] = []
    last_end = 0

    for boundary in boundaries:
        # Add lines before this table
        result_lines.extend(lines[last_end:boundary.start_line])

        # Add placeholder
        result_lines.append(placeholder)
        result_lines.append("")  # Blank line after placeholder

        last_end = boundary.end_line + 1

    # Add remaining lines after last table
    result_lines.extend(lines[last_end:])

    # Clean up multiple blank lines
    text = "\n".join(result_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
