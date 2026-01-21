"""Symbol and content normalizer for preprocessing."""

import re
import unicodedata

# Symbol mappings for table cells and general content
CHECKMARK_SYMBOLS = frozenset([
    "✓", "✔", "☑", "✅", "⬛", "▣", "☒",  # Checkmarks and checked boxes
    "yes", "Yes", "YES", "да", "Да", "ДА",
])

CROSS_SYMBOLS = frozenset([
    "✗", "✘", "☐", "❌", "⬜", "☓", "✕",  # Crosses and empty boxes
    "no", "No", "NO", "нет", "Нет", "НЕТ",
])

# Bullet point symbols to normalize
BULLET_SYMBOLS = frozenset([
    "•", "●", "◦", "◆", "◇", "▪", "▫", "▸", "▹", "►", "▻",
    "‣", "⁃", "⁌", "⁍", "∙", "⋅", "⊙", "⊚", "○", "◌",
])


def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode text to NFC form.

    NFC (Canonical Decomposition, followed by Canonical Composition)
    ensures consistent representation of characters.

    Args:
        text: Text to normalize

    Returns:
        NFC-normalized text
    """
    return unicodedata.normalize("NFC", text)


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace: collapse multiple spaces, normalize line endings.

    Args:
        text: Text to normalize

    Returns:
        Text with normalized whitespace
    """
    # Normalize line endings to \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Replace tabs with spaces (except in table structures)
    # We'll preserve structure by not collapsing whitespace aggressively

    # Collapse multiple spaces (but preserve leading whitespace for indentation)
    lines = text.split("\n")
    result_lines = []
    for line in lines:
        # Preserve leading whitespace
        leading = len(line) - len(line.lstrip())
        leading_ws = line[:leading]
        rest = line[leading:]
        # Collapse multiple spaces in the rest
        rest = re.sub(r"  +", " ", rest)
        result_lines.append(leading_ws + rest)

    text = "\n".join(result_lines)

    # Remove trailing whitespace from lines
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    # Collapse multiple blank lines into two
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


def normalize_table_cell(cell: str) -> str:
    """
    Normalize a single table cell value.

    Converts symbols like +/- to да/нет for boolean cells,
    normalizes checkmarks and crosses.

    Args:
        cell: Cell content to normalize

    Returns:
        Normalized cell content
    """
    cell = cell.strip()

    # Single character boolean indicators
    if cell in ("+", "＋"):  # Plus signs
        return "да"
    if cell in ("-", "−", "–", "—", "―"):  # Various dashes/minuses
        return "нет"

    # Checkmark symbols
    if cell in CHECKMARK_SYMBOLS:
        return "да"

    # Cross symbols
    if cell in CROSS_SYMBOLS:
        return "нет"

    # Empty cell indicators
    if cell in ("", "—", "–", "-", "N/A", "n/a", "NA", "na", "Н/Д", "н/д"):
        return ""

    return cell


def normalize_bullet(char: str) -> str:
    """
    Normalize bullet point character.

    Args:
        char: Bullet character

    Returns:
        Normalized bullet (always "-")
    """
    if char in BULLET_SYMBOLS:
        return "-"
    return char


def normalize_bullets_in_text(text: str) -> str:
    """
    Normalize all bullet points in text to "-".

    Args:
        text: Text containing bullet points

    Returns:
        Text with normalized bullets
    """
    # Create pattern for all bullet symbols
    bullet_pattern = "[" + re.escape("".join(BULLET_SYMBOLS)) + "]"

    # Replace bullets at the start of lines (possibly with leading whitespace)
    text = re.sub(
        rf"^(\s*){bullet_pattern}(\s)",
        r"\1-\2",
        text,
        flags=re.MULTILINE,
    )

    return text


def normalize_table_cells_in_text(text: str) -> str:
    """
    Normalize table cell values in markdown tables.

    Finds markdown tables and normalizes +/- symbols to да/нет.

    Args:
        text: Text containing markdown tables

    Returns:
        Text with normalized table cells
    """
    lines = text.split("\n")
    result_lines = []
    in_table = False

    for line in lines:
        stripped = line.strip()

        # Detect table row (starts with |)
        if stripped.startswith("|"):
            in_table = True

            # Skip separator rows (|---|---|)
            if re.match(r"^\|[\s\-:|]+\|$", stripped):
                result_lines.append(line)
                continue

            # Process table cells
            cells = stripped.split("|")
            normalized_cells = []
            for cell in cells:
                if cell.strip():  # Non-empty cell
                    normalized = normalize_table_cell(cell.strip())
                    # Preserve spacing
                    normalized_cells.append(f" {normalized} ")
                else:
                    normalized_cells.append(cell)

            result_lines.append("|".join(normalized_cells))
        else:
            # If we were in a table and now we're not, table ended
            if in_table and stripped:
                in_table = False
            result_lines.append(line)

    return "\n".join(result_lines)


def normalize_full(text: str) -> str:
    """
    Apply all normalizations to text.

    Order of operations:
    1. Unicode normalization (NFC)
    2. Whitespace normalization
    3. Bullet point normalization
    4. Table cell normalization

    Args:
        text: Text to normalize

    Returns:
        Fully normalized text
    """
    text = normalize_unicode(text)
    text = normalize_whitespace(text)
    text = normalize_bullets_in_text(text)
    text = normalize_table_cells_in_text(text)
    return text
