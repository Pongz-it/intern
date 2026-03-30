"""Text normalization utilities for parsed documents."""

import re
import unicodedata
from typing import Optional


def normalize_text(text: str) -> str:
    """
    Normalize text content according to design spec.

    Normalization rules:
    - Strip control characters except \\n and \\t
    - Collapse repeated whitespace
    - Collapse excessive blank lines (max 2 consecutive)
    - Normalize Unicode to NFC form
    - Remove BOM if present

    Args:
        text: Raw text content

    Returns:
        Normalized text
    """
    if not text:
        return ""

    # Remove BOM if present
    if text.startswith("\ufeff"):
        text = text[1:]

    # Normalize Unicode to NFC form (canonical composition)
    text = unicodedata.normalize("NFC", text)

    # Strip control characters except \\n and \\t
    text = strip_control_characters(text)

    # Normalize line endings to \\n
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse repeated spaces (but preserve single spaces)
    text = re.sub(r" {2,}", " ", text)

    # Collapse repeated tabs
    text = re.sub(r"\t{2,}", "\t", text)

    # Collapse excessive blank lines (max 2 consecutive newlines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip trailing whitespace from each line
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Strip leading/trailing whitespace from entire text
    text = text.strip()

    return text


def strip_control_characters(text: str, keep: Optional[str] = None) -> str:
    """
    Remove control characters from text.

    Default behavior: Keep \\n and \\t only.

    Args:
        text: Input text
        keep: Characters to keep (default: "\\n\\t")

    Returns:
        Text with control characters removed
    """
    if keep is None:
        keep = "\n\t"

    # Build set of characters to keep
    keep_chars = set(keep)

    # Filter out control characters
    result = []
    for char in text:
        # Keep if not a control character, or if in keep list
        if not unicodedata.category(char).startswith("C") or char in keep_chars:
            result.append(char)

    return "".join(result)


def collapse_whitespace(text: str, preserve_newlines: bool = True) -> str:
    """
    Collapse consecutive whitespace characters.

    Args:
        text: Input text
        preserve_newlines: If True, preserve \\n; if False, collapse all whitespace

    Returns:
        Text with collapsed whitespace
    """
    if preserve_newlines:
        # Collapse spaces and tabs, but keep newlines
        text = re.sub(r"[ \t]+", " ", text)
        # Collapse excessive newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
    else:
        # Collapse all whitespace to single space
        text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_title_from_text(text: str, max_length: int = 200) -> Optional[str]:
    """
    Extract potential title from text.

    Heuristics:
    - First non-empty line
    - Heading markers (# in markdown, etc.)
    - ALL CAPS lines
    - Short lines before paragraph breaks

    Args:
        text: Document text
        max_length: Maximum title length

    Returns:
        Extracted title, or None
    """
    if not text:
        return None

    lines = text.strip().split("\n")

    for line in lines[:10]:  # Check first 10 lines only
        line = line.strip()

        if not line:
            continue

        # Check for markdown heading
        if line.startswith("#"):
            title = line.lstrip("#").strip()
            if title:
                return title[:max_length]

        # Check for ALL CAPS (potential title)
        if len(line) <= max_length and line.isupper() and len(line) > 3:
            return line[:max_length]

        # Check for short first line followed by blank
        if len(line) <= max_length and len(line) > 3:
            # If next line is blank or significantly longer, this might be title
            if len(lines) > 1 and (not lines[1].strip() or len(lines[1]) > len(line) * 2):
                return line[:max_length]

            # Otherwise, return first line as fallback
            return line[:max_length]

    return None


def clean_metadata_value(value: str, max_length: int = 1000) -> str:
    """
    Clean and normalize metadata field value.

    Args:
        value: Raw metadata value
        max_length: Maximum value length

    Returns:
        Cleaned metadata value
    """
    if not value:
        return ""

    # Strip whitespace
    value = value.strip()

    # Remove control characters
    value = strip_control_characters(value, keep=" ")

    # Collapse whitespace
    value = collapse_whitespace(value, preserve_newlines=False)

    # Truncate if too long
    if len(value) > max_length:
        value = value[: max_length - 3] + "..."

    return value


def extract_links_from_text(text: str) -> list[str]:
    """
    Extract URLs from text.

    Finds:
    - http:// and https:// URLs
    - Markdown-style links [text](url)
    - Plain URLs in text

    Args:
        text: Input text

    Returns:
        List of extracted URLs
    """
    links = []

    # Extract markdown links [text](url)
    markdown_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    markdown_links = re.findall(markdown_pattern, text)
    links.extend([url for _, url in markdown_links])

    # Extract plain URLs
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    plain_urls = re.findall(url_pattern, text)
    links.extend(plain_urls)

    # Deduplicate while preserving order
    seen = set()
    unique_links = []
    for link in links:
        if link not in seen:
            seen.add(link)
            unique_links.append(link)

    return unique_links


def truncate_text(
    text: str,
    max_length: int,
    suffix: str = "...",
    preserve_words: bool = True,
) -> str:
    """
    Truncate text to maximum length.

    Args:
        text: Input text
        max_length: Maximum length (including suffix)
        suffix: Suffix to add when truncated (default: "...")
        preserve_words: If True, truncate at word boundary

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    # Account for suffix length
    target_length = max_length - len(suffix)

    if preserve_words:
        # Find last space before target length
        truncated = text[:target_length]
        last_space = truncated.rfind(" ")
        if last_space > 0:
            truncated = truncated[:last_space]
    else:
        truncated = text[:target_length]

    return truncated + suffix


def remove_duplicate_lines(text: str, case_sensitive: bool = True) -> str:
    """
    Remove duplicate consecutive lines from text.

    Args:
        text: Input text
        case_sensitive: If False, ignore case when detecting duplicates

    Returns:
        Text with duplicate lines removed
    """
    lines = text.split("\n")
    result = []
    prev_line = None

    for line in lines:
        compare_line = line if case_sensitive else line.lower()
        compare_prev = prev_line if case_sensitive else (prev_line.lower() if prev_line else None)

        if compare_line != compare_prev:
            result.append(line)

        prev_line = line

    return "\n".join(result)


def extract_sentences(text: str, max_sentences: Optional[int] = None) -> list[str]:
    """
    Split text into sentences.

    Simple sentence splitting heuristic (not perfect for all languages).

    Args:
        text: Input text
        max_sentences: Maximum number of sentences to return (optional)

    Returns:
        List of sentences
    """
    # Simple sentence boundary detection
    # Matches: . ! ? followed by space and capital letter, or end of string
    sentence_pattern = r'[.!?]+(?:\s+(?=[A-Z])|$)'

    sentences = re.split(sentence_pattern, text)

    # Clean up sentences
    sentences = [s.strip() for s in sentences if s.strip()]

    if max_sentences:
        sentences = sentences[:max_sentences]

    return sentences


def count_words(text: str) -> int:
    """
    Count words in text.

    Simple whitespace-based word counting.

    Args:
        text: Input text

    Returns:
        Word count
    """
    if not text:
        return 0

    # Split on whitespace and count non-empty tokens
    words = text.split()
    return len(words)


def create_excerpt(
    text: str,
    max_length: int = 200,
    prefer_sentences: bool = True,
) -> str:
    """
    Create short excerpt from text.

    Args:
        text: Input text
        max_length: Maximum excerpt length
        prefer_sentences: If True, try to end at sentence boundary

    Returns:
        Excerpt text
    """
    if len(text) <= max_length:
        return text

    if prefer_sentences:
        # Extract first few sentences that fit
        sentences = extract_sentences(text)
        excerpt = ""

        for sentence in sentences:
            if len(excerpt) + len(sentence) + 1 <= max_length:
                if excerpt:
                    excerpt += " "
                excerpt += sentence
            else:
                break

        if excerpt:
            return excerpt

    # Fallback to word-boundary truncation
    return truncate_text(text, max_length, preserve_words=True)
