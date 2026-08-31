"""Streaming data ingestion from raw dataset files."""

from __future__ import annotations

from pathlib import Path
from typing import Generator, Iterator

import pandas as pd
import pyarrow.parquet as pq

from tokenizer_bn.data.bangla_filter import detect_bangla_column, detect_english_column


def stream_txt_lines(path: Path, encoding: str = "utf-8") -> Iterator[str]:
    """Yield non-empty lines from a plain text file."""
    with open(path, encoding=encoding, errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield line


def stream_tatoeba_tsv(path: Path, encoding: str = "utf-8") -> Generator[tuple[str, str], None, None]:
    """Yield (english, bengali) pairs from Tatoeba-style TSV: en \\t bn \\t attribution."""
    with open(path, encoding=encoding, errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                yield parts[0].strip(), parts[1].strip()


def stream_parquet_pairs(path: Path) -> Generator[tuple[str, str, str], None, None]:
    """Yield (bengali, english, source_tag) from a parquet file with auto-detected columns."""
    pf = pq.ParquetFile(path)
    schema_cols = [field.name for field in pf.schema_arrow]
    # Read a small batch to detect columns
    batch = next(pf.iter_batches(batch_size=100))
    sample_rows = batch.to_pydict()
    sample = [{c: sample_rows[c][i] for c in schema_cols} for i in range(min(100, batch.num_rows))]

    bn_col = detect_bangla_column(schema_cols, sample)
    en_col = detect_english_column(schema_cols, sample, exclude=bn_col)

    # Fallback to known column names
    if bn_col is None:
        for candidate in ("bn", "bengali", "bangla", "Bengali"):
            if candidate in schema_cols:
                bn_col = candidate
                break
    if en_col is None:
        for candidate in ("en", "english", "English"):
            if candidate in schema_cols:
                en_col = candidate
                break

    if bn_col is None:
        return

    for batch in pf.iter_batches(batch_size=10_000):
        data = batch.to_pydict()
        n = batch.num_rows
        for i in range(n):
            bn_text = str(data[bn_col][i] or "").strip()
            en_text = str(data[en_col][i] or "").strip() if en_col else ""
            if bn_text:
                yield bn_text, en_text, path.stem


def stream_csv_bangla(path: Path, encoding: str = "utf-8") -> Generator[str, None, None]:
    """Yield Bengali text from a CSV, auto-detecting the Bengali column."""
    # Read header + sample
    sample_df = pd.read_csv(path, nrows=50, encoding=encoding)
    bn_col = detect_bangla_column(sample_df.columns, sample_df.to_dict("records"))

    if bn_col is None:
        for candidate in ("Bengali", "bengali", "bn", "bangla"):
            if candidate in sample_df.columns:
                bn_col = candidate
                break

    if bn_col is None:
        return

    for chunk in pd.read_csv(path, usecols=[bn_col], chunksize=5000, encoding=encoding):
        for text in chunk[bn_col].dropna().astype(str):
            text = text.strip()
            if text:
                yield text


def estimate_file_bytes(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0
