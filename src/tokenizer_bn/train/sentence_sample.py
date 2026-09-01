"""Resolve how SentencePiece training should subsample a large corpus."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from tokenizer_bn.data.ingest import stream_txt_lines


class TrainingMode(str, Enum):
    """How to feed sentences into SentencePiece training."""

    # One pass: optional single subsample (or full corpus when small enough).
    SINGLE = "single"
    # Sequential shards over the full corpus; each shard seeds the next from the
    # previous model vocabulary (see train/sharded_training.py).
    SHARDED = "sharded"


@dataclass(frozen=True)
class TrainingPlan:
    mode: TrainingMode
    line_count: int
    # SINGLE mode: SP input_sentence_size (None = omit / use all lines in input file).
    sp_size: int | None = None
    remapped_limit: int | None = None
    # SHARDED mode: number of round-robin shards and target lines per shard.
    num_shards: int = 1
    shard_sentences: int = 0
    log_message: str = ""


def count_corpus_lines(corpus_path: Path) -> int:
    """Count non-empty lines in a corpus file (streaming, O(1) memory)."""
    return sum(1 for _ in stream_txt_lines(corpus_path))


def compute_num_shards(line_count: int, shard_sentences: int) -> int:
    """Number of shards needed to cover ``line_count`` at ``shard_sentences`` per shard."""
    if line_count <= 0 or shard_sentences <= 0:
        return 1
    return max(1, (line_count + shard_sentences - 1) // shard_sentences)


def resolve_training_plan(
    input_sentence_size: int,
    unlimited_soft_cap: int,
    shard_sentences: int,
    corpus_path: Path,
) -> TrainingPlan:
    """Choose single-subsample vs sharded full-corpus training.

    When ``input_sentence_size`` is unlimited (<= 0) and ``shard_sentences`` > 0,
    large corpora use **sharded training**: the corpus is split into disjoint
    round-robin shards, each trained in sequence with the previous shard's
    vocabulary as a seed. This exposes the model to every line without loading
    the full corpus into RAM at once.

    Otherwise falls back to a single subsample (explicit size or soft cap).
    """
    line_count = count_corpus_lines(corpus_path)

    if input_sentence_size > 0:
        n = min(input_sentence_size, line_count)
        return TrainingPlan(
            mode=TrainingMode.SINGLE,
            line_count=line_count,
            sp_size=n,
            remapped_limit=n,
            log_message=f"single-pass subsample: {n:,} of {line_count:,} lines",
        )

    # Unlimited: prefer sharded full-corpus training when configured.
    if shard_sentences > 0 and line_count > shard_sentences:
        num_shards = compute_num_shards(line_count, shard_sentences)
        return TrainingPlan(
            mode=TrainingMode.SHARDED,
            line_count=line_count,
            num_shards=num_shards,
            shard_sentences=shard_sentences,
            log_message=(
                f"sharded full-corpus training: {line_count:,} lines in {num_shards} shards "
                f"(~{shard_sentences:,} lines/shard, vocab carried between shards)"
            ),
        )

    if unlimited_soft_cap > 0 and line_count > unlimited_soft_cap:
        return TrainingPlan(
            mode=TrainingMode.SINGLE,
            line_count=line_count,
            sp_size=unlimited_soft_cap,
            remapped_limit=unlimited_soft_cap,
            log_message=(
                f"single-pass subsample: input_sentence_size=0 (unlimited) but corpus has "
                f"{line_count:,} lines; using {unlimited_soft_cap:,} lines "
                f"(training.unlimited_sentence_soft_cap). "
                f"Set training.shard_sentences to cover the full corpus via sharded training."
            ),
        )

    return TrainingPlan(
        mode=TrainingMode.SINGLE,
        line_count=line_count,
        sp_size=None,
        remapped_limit=None,
        log_message=f"single-pass on all {line_count:,} sentences (unlimited)",
    )


def resolve_sp_input_sentence_size(
    input_sentence_size: int,
    unlimited_soft_cap: int,
    corpus_path: Path,
) -> tuple[int | None, int | None, str]:
    """Backward-compatible wrapper around :func:`resolve_training_plan`."""
    plan = resolve_training_plan(
        input_sentence_size,
        unlimited_soft_cap,
        shard_sentences=0,
        corpus_path=corpus_path,
    )
    return plan.sp_size, plan.remapped_limit, plan.log_message
