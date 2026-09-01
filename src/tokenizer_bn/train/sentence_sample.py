"""Resolve SentencePiece training subsample size for large corpora."""

from __future__ import annotations

from pathlib import Path

from tokenizer_bn.data.ingest import stream_txt_lines


def count_corpus_lines(corpus_path: Path) -> int:
    """Count non-empty lines in a corpus file (streaming, O(1) memory)."""
    return sum(1 for _ in stream_txt_lines(corpus_path))


def resolve_sp_input_sentence_size(
    input_sentence_size: int,
    unlimited_soft_cap: int,
    corpus_path: Path,
) -> tuple[int | None, int | None, str]:
    """Pick a SentencePiece subsample size that avoids loading huge corpora into RAM.

    SentencePiece reservoir-samples ``input_sentence_size`` sentences from the
    full input stream (with ``shuffle_input_sentence=True``), so a bounded value
    gives representative full-corpus coverage at fixed memory. That is the
    standard, memory-safe way to train on very large corpora.

    Returns:
        sp_size: value for SentencePiece ``input_sentence_size`` (``None`` = omit;
            SP then trains on every sentence — only safe for modest corpora).
        remapped_limit: lines to write into the grapheme training corpus
            (``None`` = stream the full corpus).
        log_message: human-readable explanation for logs.
    """
    line_count = count_corpus_lines(corpus_path)

    if input_sentence_size > 0:
        n = min(input_sentence_size, line_count)
        return n, n, f"training subsample: {n:,} of {line_count:,} lines"

    # Unlimited (0 or negative): honour the full corpus unless a soft cap protects RAM.
    if unlimited_soft_cap > 0 and line_count > unlimited_soft_cap:
        return (
            unlimited_soft_cap,
            unlimited_soft_cap,
            (
                f"input_sentence_size=0 (unlimited) but corpus has {line_count:,} lines; "
                f"reservoir-sampling {unlimited_soft_cap:,} sentences from the full stream "
                f"(training.unlimited_sentence_soft_cap) to avoid OOM"
            ),
        )

    return None, None, f"training on all {line_count:,} sentences (unlimited)"
