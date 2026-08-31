"""Tests for akshara segmentation and grapheme remapping."""

from pathlib import Path

from tokenizer_bn.segmentation.akshara import count_aksharas, segment_aksharas, validate_segmentation
from tokenizer_bn.segmentation.remap import build_grapheme_seed_file


def test_segment_basic_bengali():
    text = "বাংলা"
    clusters = segment_aksharas(text)
    assert len(clusters) >= 1
    assert "".join(clusters) == text


def test_segment_conjunct():
    text = "স্কুল"
    clusters = segment_aksharas(text)
    assert len(clusters) >= 1
    assert "".join(clusters) == text


def test_count_aksharas():
    text = "আমি ভালো আছি"
    assert count_aksharas(text) > 0


def test_build_grapheme_seed(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("বাংলা ভাষা\nআমি ভালো আছি\n", encoding="utf-8")
    seed = tmp_path / "seed.txt"
    n = build_grapheme_seed_file(corpus, seed)
    assert n > 0
    assert seed.exists()


def test_validate_segmentation():
    samples = ["বাংলা", "আমি ভালো আছি", "স্কুলে যাই"]
    result = validate_segmentation(samples)
    assert result["num_samples"] == 3
    assert result["total_aksharas"] > 0
