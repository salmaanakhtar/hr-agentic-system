"""
Policy Parser Tool for Phase 7 Policy Compliance Agent.

Uses pypdf for text extraction and a sliding-window paragraph chunker:
- Splits text on double-newline boundaries
- Merges short paragraphs (<20 words) into the following paragraph
- Accumulates paragraphs into chunks up to max_words
- Overlaps consecutive chunks by `overlap` paragraphs to preserve context
- Called via asyncio.to_thread from async code (synchronous)
"""

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def _extract_text(file_path: str) -> str:
    """Extract and return raw text from a PDF file."""
    from pypdf import PdfReader

    reader = PdfReader(file_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def _split_paragraphs(text: str) -> list[str]:
    """
    Split text on double-newline boundaries, normalise whitespace within
    each paragraph, and discard empty results.
    """
    raw = re.split(r"\n{2,}", text)
    paragraphs = []
    for para in raw:
        cleaned = re.sub(r"\s+", " ", para).strip()
        if cleaned:
            paragraphs.append(cleaned)
    return paragraphs


def _merge_short(paragraphs: list[str], min_words: int = 20) -> list[str]:
    """
    Merge any paragraph shorter than min_words into the following paragraph.
    If the short paragraph is the last one it is merged into the preceding one.
    """
    if not paragraphs:
        return paragraphs

    merged: list[str] = []
    carry = ""

    for para in paragraphs:
        combined = (carry + " " + para).strip() if carry else para
        word_count = len(combined.split())

        if word_count < min_words:
            # Still too short — keep carrying forward
            carry = combined
        else:
            merged.append(combined)
            carry = ""

    # Flush any leftover short tail into the last chunk
    if carry:
        if merged:
            merged[-1] = (merged[-1] + " " + carry).strip()
        else:
            merged.append(carry)

    return merged


def _build_chunks(paragraphs: list[str], max_words: int, overlap: int) -> list[str]:
    """
    Slide a window over `paragraphs`, accumulating up to max_words per chunk.
    Consecutive chunks share `overlap` paragraphs at their boundary so that
    context is not lost at split points.

    A single paragraph that already exceeds max_words is emitted as its own
    chunk (hard-split by word count as a fallback).
    """
    if not paragraphs:
        return []

    chunks: list[str] = []
    window: list[str] = []   # paragraphs in the current chunk
    window_words = 0

    def flush(win: list[str]) -> str:
        return " ".join(win)

    def word_count(s: str) -> int:
        return len(s.split())

    i = 0
    while i < len(paragraphs):
        para = paragraphs[i]
        para_words = word_count(para)

        if para_words > max_words:
            # Oversized paragraph — emit any accumulated window first, then
            # hard-split the paragraph by word count.
            if window:
                chunks.append(flush(window))
                window = window[-overlap:] if overlap else []
                window_words = sum(word_count(p) for p in window)

            words = para.split()
            start = 0
            while start < len(words):
                chunk_words = words[start : start + max_words]
                chunks.append(" ".join(chunk_words))
                start += max_words
            i += 1
            continue

        if window_words + para_words > max_words and window:
            # Current window is full — flush it and start a new window with
            # the overlap tail of the previous one.
            chunks.append(flush(window))
            window = window[-overlap:] if overlap else []
            window_words = sum(word_count(p) for p in window)

        window.append(para)
        window_words += para_words
        i += 1

    if window:
        chunks.append(flush(window))

    return chunks


def parse_and_chunk_policy(
    file_path: str,
    max_words: int = 400,
    overlap: int = 1,
) -> list[str]:
    """
    Main entry point. Extracts text from a PDF policy document at file_path,
    splits it into semantically coherent chunks of at most max_words words
    with overlap paragraphs of context carried across chunk boundaries.

    Returns a list of text chunks ready for embedding.

    This function is synchronous and must be called via asyncio.to_thread
    from async code.
    """
    path = Path(file_path)
    if not path.exists():
        logger.warning(f"Policy file not found: {file_path}")
        return []

    try:
        raw_text = _extract_text(str(path))
    except Exception as exc:
        logger.error(f"PDF text extraction failed for {file_path}: {exc}")
        return []

    if not raw_text:
        logger.warning(f"No text extracted from policy document: {file_path}")
        return []

    paragraphs = _split_paragraphs(raw_text)
    paragraphs = _merge_short(paragraphs, min_words=20)
    chunks = _build_chunks(paragraphs, max_words=max_words, overlap=overlap)

    logger.info(
        f"Policy parsed - paragraphs: {len(paragraphs)}, "
        f"chunks: {len(chunks)}, file: {path.name}"
    )

    return chunks
