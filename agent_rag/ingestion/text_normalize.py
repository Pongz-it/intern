"""Comprehensive text cleaning utilities for document processing and search.

This module provides unified text cleaning functionality:
- Unicode normalization (Kangxi radicals, full-width characters)
- Markdown formatting cleanup
- Special character handling for various document formats
- Chinese text detection and processing
"""

import re
import unicodedata
from typing import Optional


CJK_RADICALS_RANGE = (0x2F00, 0x2FDF)
FULLWIDTH_RANGE = (0xFF00, 0xFFEF)
CJK_UNIFIED_IDEOGRAPHS = (0x4E00, 0x9FFF)
CJK_UNIFIED_IDEOGRAPHS_EXT_A = (0x3400, 0x4DBF)
CJK_SYMBOLS_PUNCTUATION = (0x3000, 0x303F)
KANGXI_RADICALS = (0x2E80, 0x2EFF)
BOPOMOFO = (0x3100, 0x312F)
CJK_EXT_B = (0x31A0, 0x31BF)


def normalize_text(text: str) -> str:
    """Normalize Unicode compatibility characters.

    Handles:
    - Kangxi radicals (康熙部首) from PDF extraction
    - CJK compatibility ideographs
    - Full-width to half-width characters

    Args:
        text: Input text that may contain compatibility characters

    Returns:
        Normalized text with standard Unicode characters
    """
    if not text:
        return text

    result = []
    for char in text:
        cp = ord(char)

        if CJK_RADICALS_RANGE[0] <= cp <= CJK_RADICALS_RANGE[1]:
            result.append(unicodedata.normalize('NFKC', char))
        elif FULLWIDTH_RANGE[0] <= cp <= FULLWIDTH_RANGE[1]:
            result.append(chr(cp - 0xFEE0))
        else:
            result.append(char)

    return ''.join(result)


normalize_unicode = normalize_text


def normalize_chunk_content(content: str) -> str:
    """Normalize chunk content for consistent indexing and search.

    Args:
        content: Raw content from document parsing

    Returns:
        Normalized content ready for indexing
    """
    return clean_for_indexing(content)


def clean_special_chars(text: str) -> str:
    """Clean various special characters from documents.

    Handles common special characters from different document formats:
    - Control characters
    - Zero-width characters
    - Various whitespace variants
    - Common encoding artifacts

    Args:
        text: Input text with potential special characters

    Returns:
        Cleaned text with special characters removed
    """
    if not text:
        return text

    result = []

    for char in text:
        cp = ord(char)

        if cp < 0x20 and char not in '\n\t\r':
            continue

        if 0x7F <= cp < 0xA0:
            continue

        if 0x200B <= cp <= 0x200F:
            continue

        if 0x202A <= cp <= 0x202E:
            continue

        if 0x2060 <= cp <= 0x206F:
            continue

        if 0xFE00 <= cp <= 0xFE0F:
            continue

        if cp in (0x00A0, 0x1680, 0x180E):
            result.append(' ')
            continue

        if 0x3000 <= cp <= 0x3007:
            if cp == 0x3000:
                result.append(' ')
                continue

        result.append(char)

    cleaned = ''.join(result)

    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

    return cleaned


def clean_line_markers(text: str) -> str:
    """Clean common line markers and numbering from documents.

    Args:
        text: Input text with potential line markers

    Returns:
        Cleaned text
    """
    if not text:
        return text

    lines = []
    for line in text.split('\n'):
        cleaned = re.sub(r'^[\s]*[·•▪▫‣⁃○◦▪️]+\s*', '', line)
        cleaned = re.sub(r'^[\s]*[\d]+[.)]\s*', '', cleaned)
        cleaned = re.sub(r'^[\s]*[a-zA-Z][.)]\s*', '', cleaned)
        lines.append(cleaned)

    return '\n'.join(lines)


def clean_markdown(text: str, max_lines: Optional[int] = None) -> str:
    """Clean markdown formatting from text for better readability.

    Args:
        text: Text with potential markdown formatting
        max_lines: Optional limit on number of lines

    Returns:
        Cleaned text with markdown converted to plain text
    """
    if not text:
        return text

    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        cleaned_line = _clean_markdown_line(line)
        if cleaned_line.strip():
            cleaned_lines.append(cleaned_line)
        elif cleaned_lines:
            cleaned_lines.append('')

    if max_lines and len(cleaned_lines) > max_lines:
        cleaned_lines = cleaned_lines[:max_lines]
        if cleaned_lines and cleaned_lines[-1]:
            cleaned_lines[-1] = cleaned_lines[-1].rstrip() + '...'

    return '\n'.join(cleaned_lines).strip()


def _clean_markdown_line(line: str) -> str:
    """Clean a single line of markdown formatting."""
    for pattern, prefix in [
        (r'^###\s+', ''),
        (r'^##\s+', ''),
        (r'^#\s+', ''),
    ]:
        line = re.sub(pattern, prefix, line)

    for pattern, replacement in [
        (r'\*\*([^*]+?)\*\*', r'\1'),
        (r'__([^_]+?)__', r'\1'),
    ]:
        line = re.sub(pattern, replacement, line)

    for pattern, replacement in [
        (r'\*([^*]+?)\*', r'\1'),
        (r'_([^_]+?)_', r'\1'),
    ]:
        line = re.sub(pattern, replacement, line)

    line = re.sub(r'\[([^\]]+?)\]\([^\)]+?\)', r'\1', line)

    line = re.sub(r'^[\s]*[-*+]\s+', '• ', line)

    line = re.sub(r'^[ \t]*\d+\.\s+', lambda m: f"{m.group()}", line)

    line = re.sub(r'`{3,}[a-z]*\n?', '', line)
    line = re.sub(r'`([^`]+?)`', r'\1', line)

    line = re.sub(r'^>\s+', '', line)

    line = re.sub(r'^\|[\s|-]*\|?\s*$', '', line)

    return line


def clean_table(text: str) -> str:
    """Convert table formatting to readable plain text.

    Args:
        text: Table content with potential markdown/HTML formatting

    Returns:
        Cleaned table text
    """
    if not text:
        return text

    lines = text.strip().split('\n')
    if len(lines) < 2:
        return text

    processed_lines = []
    for line in lines:
        if line.strip().startswith('|') and line.strip().endswith('|'):
            line = line.strip()[1:-1]
        elif line.strip().startswith('|'):
            line = line.strip()[1:]
        elif line.strip().endswith('|'):
            line = line.strip()[:-1]
        else:
            processed_lines.append(line)
            continue

        cells = [cell.strip() for cell in line.split('|')]
        processed_lines.append(' | '.join(cells))

    return '\n'.join(processed_lines)


def clean_for_display(
    text: str,
    max_lines: Optional[int] = None,
    max_length: Optional[int] = None,
) -> str:
    """Comprehensive text cleaning for display purposes.

    Combines all cleaning operations in a sensible order.

    Args:
        text: Raw text from document
        max_lines: Optional limit on number of lines
        max_length: Optional maximum length of returned text

    Returns:
        Fully cleaned text ready for display
    """
    if not text:
        return ""

    cleaned = normalize_text(text)
    cleaned = clean_special_chars(cleaned)
    cleaned = clean_markdown(cleaned)
    cleaned = clean_line_markers(cleaned)

    if max_lines:
        lines = cleaned.split('\n')
        if len(lines) > max_lines:
            cleaned = '\n'.join(lines[:max_lines])
            if not cleaned.endswith('...'):
                cleaned = cleaned.rstrip() + '...'

    if max_length and len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rsplit(' ', 1)[0] + '...'

    return cleaned.strip()


def clean_for_indexing(text: str) -> str:
    """Clean text for indexing purposes.

    Less aggressive than clean_for_display - preserves structure
    for searchability while normalizing characters.

    Args:
        text: Raw text from document

    Returns:
        Text ready for indexing
    """
    if not text:
        return text

    cleaned = normalize_text(text)
    cleaned = clean_special_chars(cleaned)

    return cleaned.strip()


def is_chinese_text(text: str, threshold: float = 0.3) -> bool:
    """Check if text contains significant Chinese characters.

    Args:
        text: Text to check
        threshold: Minimum ratio of Chinese characters (default 30%)

    Returns:
        True if text is primarily Chinese
    """
    if not text or not text.strip():
        return False

    chinese_count = sum(1 for char in text if _is_cjk_char(char))
    total_chars = len(text.strip())

    return (chinese_count / total_chars) > threshold if total_chars > 0 else False


def _is_cjk_char(char: str) -> bool:
    """Check if a character is a CJK character."""
    cp = ord(char)

    return (
        CJK_UNIFIED_IDEOGRAPHS[0] <= cp <= CJK_UNIFIED_IDEOGRAPHS[1] or
        CJK_UNIFIED_IDEOGRAPHS_EXT_A[0] <= cp <= CJK_UNIFIED_IDEOGRAPHS_EXT_A[1] or
        CJK_SYMBOLS_PUNCTUATION[0] <= cp <= CJK_SYMBOLS_PUNCTUATION[1] or
        KANGXI_RADICALS[0] <= cp <= KANGXI_RADICALS[1] or
        BOPOMOFO[0] <= cp <= BOPOMOFO[1] or
        CJK_EXT_B[0] <= cp <= CJK_EXT_B[1]
    )


def extract_text_from_html(text: str) -> str:
    """Extract plain text from HTML-like content.

    Args:
        text: Text potentially containing HTML tags

    Returns:
        Text with HTML tags removed
    """
    if not text:
        return text

    result = re.sub(r'<[^>]+>', '', text)
    result = re.sub(r'&nbsp;', ' ', result)
    result = re.sub(r'&amp;', '&', result)
    result = re.sub(r'&lt;', '<', result)
    result = re.sub(r'&gt;', '>', result)
    result = re.sub(r'&quot;', '"', result)
    result = re.sub(r'&#\d+;', '', result)

    return result.strip()


def truncate_text(text: str, max_length: int, suffix: str = '...') -> str:
    """Smart text truncation that respects word boundaries.

    Args:
        text: Text to truncate
        max_length: Maximum length of returned text
        suffix: Suffix to append when truncating

    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text

    if max_length <= len(suffix):
        return text[:max_length]

    truncated = text[:max_length - len(suffix)]
    truncated = truncated.rsplit(' ', 1)[0]

    return truncated + suffix


__all__ = [
    'normalize_text',
    'normalize_unicode',
    'normalize_chunk_content',
    'clean_special_chars',
    'clean_line_markers',
    'clean_markdown',
    'clean_table',
    'clean_for_display',
    'clean_for_indexing',
    'is_chinese_text',
    'extract_text_from_html',
    'truncate_text',
]
