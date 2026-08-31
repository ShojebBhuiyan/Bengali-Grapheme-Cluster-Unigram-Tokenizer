"""Full-corpus tokenizer metrics (streamed, memory-efficient)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from tokenizer_bn.data.ingest import stream_txt_lines
from tokenizer_bn.eval.gpu_metrics import batch_token_lengths
from tokenizer_bn.eval.metrics import MetricResult, TokenizerLike


@dataclass
class CorpusTotals:
    total_tokens: int = 0
    total_chars: int = 0
    total_words: int = 0
    num_lines: int = 0


def _accumulate_batch(tokenizer: TokenizerLike, batch: list[str], totals: CorpusTotals) -> None:
    token_lengths = batch_token_lengths(tokenizer, batch)
    totals.total_tokens += sum(token_lengths)
    totals.total_chars += sum(len(text.replace(" ", "")) for text in batch)
    totals.total_words += sum(len(text.split()) for text in batch)
    totals.num_lines += len(batch)


def compute_corpus_totals(
    tokenizer: TokenizerLike,
    corpus_path: Path,
    batch_size: int = 512,
    max_lines: int = 0,
) -> CorpusTotals:
    """Stream the corpus and accumulate token/char/word counts."""
    totals = CorpusTotals()
    batch: list[str] = []
    line_iter = stream_txt_lines(corpus_path)
    if max_lines <= 0:
        progress = tqdm(line_iter, desc=f"corpus:{tokenizer.name}", unit="lines")
    else:
        progress = tqdm(line_iter, desc=f"corpus:{tokenizer.name}", unit="lines", total=max_lines)

    for i, line in enumerate(progress):
        if max_lines > 0 and i >= max_lines:
            break
        batch.append(line)
        if len(batch) >= batch_size:
            _accumulate_batch(tokenizer, batch, totals)
            batch = []

    if batch:
        _accumulate_batch(tokenizer, batch, totals)

    return totals


def corpus_token_count(totals: CorpusTotals) -> MetricResult:
    """Total tokens over the full corpus (lower is better)."""
    return MetricResult("corpus_token_count", float(totals.total_tokens), totals.num_lines)


def compression_ratio(totals: CorpusTotals) -> MetricResult:
    """Characters per token over the full corpus (higher is better)."""
    value = totals.total_chars / totals.total_tokens if totals.total_tokens else 0.0
    return MetricResult("compression_ratio", value, totals.num_lines)


def tokens_per_bengali_word(totals: CorpusTotals) -> MetricResult:
    """Average tokens per Bengali word over the full corpus (lower is better)."""
    value = totals.total_tokens / totals.total_words if totals.total_words else 0.0
    return MetricResult("tokens_per_bengali_word", value, totals.num_lines)


def compute_corpus_metrics(
    tokenizer: TokenizerLike,
    corpus_path: Path,
    batch_size: int = 512,
    max_lines: int = 0,
) -> dict[str, MetricResult]:
    """Compute full-corpus metrics for a tokenizer."""
    totals = compute_corpus_totals(tokenizer, corpus_path, batch_size=batch_size, max_lines=max_lines)
    return {
        "corpus_token_count": corpus_token_count(totals),
        "compression_ratio": compression_ratio(totals),
        "tokens_per_bengali_word": tokens_per_bengali_word(totals),
    }
