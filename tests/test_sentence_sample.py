"""Tests for SentencePiece training subsample resolution."""

from pathlib import Path

from tokenizer_bn.train.sentence_sample import (
    TrainingMode,
    compute_num_shards,
    resolve_sp_input_sentence_size,
    resolve_training_plan,
)
from tokenizer_bn.train.sharded_training import write_round_robin_shard


def test_resolve_explicit_subsample(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("line\n" * 20, encoding="utf-8")
    plan = resolve_training_plan(10, 5_000_000, 5_000_000, corpus)
    assert plan.mode == TrainingMode.SINGLE
    assert plan.sp_size == 10
    assert plan.remapped_limit == 10


def test_resolve_unlimited_soft_cap(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("line\n" * 100, encoding="utf-8")
    plan = resolve_training_plan(0, 50, 0, corpus)
    assert plan.mode == TrainingMode.SINGLE
    assert plan.sp_size == 50
    assert plan.remapped_limit == 50


def test_resolve_unlimited_small_corpus(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("line\n" * 10, encoding="utf-8")
    plan = resolve_training_plan(0, 50, 5_000_000, corpus)
    assert plan.mode == TrainingMode.SINGLE
    assert plan.sp_size is None
    assert plan.remapped_limit is None


def test_resolve_sharded_full_corpus(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("line\n" * 100, encoding="utf-8")
    plan = resolve_training_plan(0, 50, 30, corpus)
    assert plan.mode == TrainingMode.SHARDED
    assert plan.num_shards == 4
    assert plan.shard_sentences == 30


def test_compute_num_shards():
    assert compute_num_shards(37_599_994, 5_000_000) == 8


def test_round_robin_shard_partition(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(f"line{i}" for i in range(10)), encoding="utf-8")
    out = tmp_path / "shard0.txt"
    n = write_round_robin_shard(corpus, out, shard_idx=0, num_shards=3)
    assert n == 4
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert lines == ["line0", "line3", "line6", "line9"]


def test_backward_compatible_wrapper(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("line\n" * 10, encoding="utf-8")
    sp_size, remapped_limit, _ = resolve_sp_input_sentence_size(5, 50, corpus)
    assert sp_size == 5
    assert remapped_limit == 5
