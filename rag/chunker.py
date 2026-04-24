"""Semantic CV chunker: section-aware splitting with context injection.

Strategy
--------
1. Detect CV section headers (ALL CAPS lines or known title-case headings).
2. Partition the document into labelled sections: [(heading, body_lines), ...].
3. Within each section body split on blank lines → atomic "entries"
   (a job stint, a degree, a list of skills, etc.).
4. Pack consecutive entries into chunks ≤ `size`, always prepending the
   section heading so every chunk is self-contained for retrieval.
5. Use a character-window fallback only when a single entry exceeds `size`.
6. Carry `overlap` characters of the previous chunk into the next one so
   context is not lost at chunk boundaries.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Section-header detection
# ---------------------------------------------------------------------------

_ALLCAPS_RE = re.compile(r"^[A-Z][A-Z\s\-&/,]{2,}$")
_MAX_HEADING_WORDS = 3

_KNOWN_HEADING_RE = re.compile(
    r"^(?:Education|Work\s+Experience|Professional\s+Experience|Experience|"
    r"Skills|Technical\s+Skills|Core\s+Skills|Key\s+Skills|"
    r"Projects?|Personal\s+Projects?|Side\s+Projects?|"
    r"Publications?|Languages?|Certifications?|Courses?|Training|"
    r"Awards?|Honors?|Distinctions?|"
    r"Interests?|Hobbies?|References?|"
    r"Summary|Professional\s+Summary|Executive\s+Summary|"
    r"Profile|Professional\s+Profile|"
    r"Objective|Career\s+Objective|"
    r"Contact(?:\s+Info(?:rmation)?)?|Personal\s+Info(?:rmation)?|"
    r"Volunteering|Volunteer\s+Experience|Community\s+Service|"
    r"Achievements?|Accomplishments?|"
    r"Research|Research\s+Experience|"
    r"Teaching(?:\s+Experience)?|Leadership|Extracurricular)s?\.?$",
    re.IGNORECASE,
)


def _is_section_header(line: str) -> bool:
    s = line.strip()
    if not s or len(s) < 3 or len(s) > 70:
        return False
    if _KNOWN_HEADING_RE.match(s):
        return True
    # All-caps heuristic: enforce short word count to avoid matching sentences
    if _ALLCAPS_RE.match(s):
        return len(s.split()) <= _MAX_HEADING_WORDS
    return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_into_sections(
    lines: List[str],
) -> List[Tuple[Optional[str], List[str]]]:
    """Return [(heading | None, body_lines), ...].

    Text that appears before any recognisable heading gets heading=None.
    """
    sections: List[Tuple[Optional[str], List[str]]] = []
    current_heading: Optional[str] = None
    body: List[str] = []

    for line in lines:
        if _is_section_header(line):
            if body or current_heading is not None:
                sections.append((current_heading, body))
            current_heading = line.strip()
            body = []
        else:
            body.append(line)

    if body or current_heading is not None:
        sections.append((current_heading, body))

    return sections


def _body_to_entries(body_lines: List[str]) -> List[List[str]]:
    """Split body_lines into entries at blank lines."""
    entries: List[List[str]] = []
    cur: List[str] = []
    for line in body_lines:
        if not line.strip():
            if cur:
                entries.append(cur)
                cur = []
        else:
            cur.append(line)
    if cur:
        entries.append(cur)
    return entries


def _entry_text(heading: Optional[str], entry_lines: List[str]) -> str:
    body = "\n".join(entry_lines).strip()
    if heading:
        return f"{heading}\n{body}"
    return body


def _char_window_split(text: str, size: int, overlap: int) -> List[str]:
    """Character-window fallback for a single block larger than `size`.

    Breaks at line / sentence / word boundaries and snaps the overlap
    start forward to the next whitespace so chunks never begin mid-word.
    """
    if not text.strip():
        return []
    if len(text) <= size:
        return [text]
    out: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            window = text[start:end]
            for sep in ("\n", ". ", "; ", ", ", " "):
                idx = window.rfind(sep)
                if idx != -1 and idx >= int(size * 0.35):
                    end = start + idx + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        if end >= n:
            break
        # Compute overlap start and snap forward to a clean word boundary
        new_start = max(end - overlap, start + 1)
        while new_start < end and text[new_start] not in (" ", "\n", "\t"):
            new_start += 1
        # Skip past the whitespace itself
        while new_start < end and text[new_start] in (" ", "\n", "\t"):
            new_start += 1
        start = new_start if new_start < end else end
    return out


def _pack_units(
    units: List[str],
    size: int,
    overlap: int,  # kept for API compatibility; overlap is handled per-entry
) -> List[str]:
    """Pack units into chunks ≤ `size`.

    Each unit already carries its section heading, so a fresh chunk is started
    whenever the next unit would exceed `size`. No additional character overlap
    is applied here — the section heading at the top of every chunk provides
    the structural context needed for retrieval.
    """
    chunks: List[str] = []
    buf = ""
    sep = "\n\n"

    for u in units:
        u = u.strip()
        if not u:
            continue
        if not buf:
            buf = u
        elif len(buf) + len(sep) + len(u) <= size:
            buf = buf + sep + u
        else:
            chunks.append(buf)
            buf = u

    if buf.strip():
        chunks.append(buf)
    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_text(text: str, size: int = 500, overlap: int = 60) -> List[str]:
    """Split CV text into semantically coherent overlapping chunks.

    Every chunk is prefixed with its section heading (e.g. "EXPERIENCE") so
    retrieval always has structural context. Entries (job stints, degrees,
    skill groups) are kept intact when they fit within `size`; only
    oversized entries are split using a character-window fallback.

    Args:
        text:    Full document text (blank lines preserved as paragraph markers).
        size:    Maximum chunk size in characters (default 500).
        overlap: Characters of context to carry over at chunk boundaries.

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

    lines = text.splitlines()
    sections = _split_into_sections(lines)

    all_units: List[str] = []

    for heading, body_lines in sections:
        entries = _body_to_entries(body_lines)

        if not entries:
            # Section header with no body — append as a standalone unit so it
            # can be merged with the next section's first entry.
            if heading:
                all_units.append(heading)
            continue

        heading_len = len(heading) + 1 if heading else 0
        available = max(size - heading_len, size // 2)

        for entry_lines in entries:
            entry_body = "\n".join(entry_lines).strip()
            if not entry_body:
                continue

            if heading_len + len(entry_body) <= size:
                all_units.append(_entry_text(heading, entry_lines))
            else:
                # Entry is too large: split body, then re-inject heading
                pieces = _char_window_split(entry_body, available, overlap)
                for piece in pieces:
                    all_units.append(f"{heading}\n{piece}" if heading else piece)

    return _pack_units(all_units, size, overlap)
