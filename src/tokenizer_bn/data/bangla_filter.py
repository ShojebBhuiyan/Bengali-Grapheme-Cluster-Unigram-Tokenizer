"""Bengali text filtering and normalization utilities."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

# Bengali Unicode block
_BENGALI_RE = re.compile(r"[\u0980-\u09FF]")
# Letters and marks in Bengali script (exclude digits/punctuation within block)
_BENGALI_LETTER_RE = re.compile(r"[\u0980-\u09FF\u09E6-\u09EF]")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str, nfc: bool = True) -> str:
    """Strip, collapse whitespace, and optionally NFC-normalize."""
    text = _WHITESPACE_RE.sub(" ", text.strip())
    if nfc:
        text = unicodedata.normalize("NFC", text)
    return text


def count_bengali_chars(text: str) -> int:
    return len(_BENGALI_RE.findall(text))


def count_letters(text: str) -> int:
    return sum(1 for ch in text if ch.isalpha() or _BENGALI_LETTER_RE.match(ch))


def bangla_char_ratio(text: str) -> float:
    """Fraction of non-whitespace characters that are Bengali script."""
    stripped = text.replace(" ", "").replace("\t", "")
    if not stripped:
        return 0.0
    return count_bengali_chars(stripped) / len(stripped)


def is_bangla_text(text: str, min_ratio: float = 0.5) -> bool:
    """Return True if text has sufficient Bengali script content."""
    if not text or not text.strip():
        return False
    if count_bengali_chars(text) < 2:
        return False
    return bangla_char_ratio(text) >= min_ratio


def detect_bangla_column(columns: Iterable[str], sample_rows: list[dict]) -> str | None:
    """Auto-detect the column richest in Bengali script from sample rows."""
    best_col: str | None = None
    best_score = 0.0
    for col in columns:
        scores = [bangla_char_ratio(str(row.get(col, ""))) for row in sample_rows]
        avg = sum(scores) / len(scores) if scores else 0.0
        if avg > best_score:
            best_score = avg
            best_col = col
    return best_col if best_score >= 0.3 else None


def detect_english_column(columns: Iterable[str], sample_rows: list[dict], exclude: str | None = None) -> str | None:
    """Auto-detect the Latin-script column (for parallel pairs)."""
    latin_re = re.compile(r"[A-Za-z]")
    best_col: str | None = None
    best_score = 0.0
    for col in columns:
        if col == exclude:
            continue
        scores = []
        for row in sample_rows:
            text = str(row.get(col, ""))
            stripped = text.replace(" ", "")
            if not stripped:
                scores.append(0.0)
                continue
            scores.append(len(latin_re.findall(stripped)) / len(stripped))
        avg = sum(scores) / len(scores) if scores else 0.0
        if avg > best_score:
            best_score = avg
            best_col = col
    return best_col if best_score >= 0.5 else None


def filter_bangla_line(text: str, min_ratio: float = 0.5, normalize_nfc: bool = True) -> str | None:
    """Normalize and return text if it passes Bengali filter, else None."""
    normalized = normalize_text(text, nfc=normalize_nfc)
    if is_bangla_text(normalized, min_ratio=min_ratio):
        return normalized
    return None
