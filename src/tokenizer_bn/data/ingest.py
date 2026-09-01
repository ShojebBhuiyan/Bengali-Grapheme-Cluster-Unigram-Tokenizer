"""Streaming data ingestion from raw dataset files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator


def stream_txt_lines(path: Path, encoding: str = "utf-8") -> Iterator[str]:
    """Yield non-empty lines from a plain text file."""
    with open(path, encoding=encoding, errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield line


def estimate_file_bytes(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0
