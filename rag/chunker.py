"""Simple sliding-window text chunker."""

from __future__ import annotations

from typing import List


def chunk_text(text: str, size: int = 800, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks of ~`size` characters.

    Tries to break on paragraph or sentence boundaries when possible to keep
    chunks semantically coherent.

    Args:
        text: Full document text.
        size: Target chunk size in characters.
        overlap: Number of characters to overlap between consecutive chunks.

    Returns:
        List of non-empty chunk strings.
    """
    if size <= 0:
        raise ValueError("size must be > 0")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be in [0, size)")

    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + size, n)

        if end < n:
            window = text[start:end]
            for sep in ("\n\n", "\n", ". ", " "):
                idx = window.rfind(sep)
                if idx != -1 and idx >= int(size * 0.5):
                    end = start + idx + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break
        start = max(end - overlap, start + 1)

    return chunks
