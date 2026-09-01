"""Tests for akshara segmentation and grapheme remapping."""

from pathlib import Path

from tokenizer_bn.segmentation.akshara import count_aksharas, segment_aksharas, validate_segmentation
from tokenizer_bn.segmentation.remap import (
    build_akshara_map,
    build_inverse_map,
    prepare_grapheme_corpus,
    remap_text,
    unmap_text,
)


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


def test_build_akshara_map(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("বাংলা ভাষা\nআমি ভালো আছি\n", encoding="utf-8")
    mapping = build_akshara_map(corpus)
    assert len(mapping) > 0
    # Space is never mapped; every value is a unique single character.
    assert " " not in mapping
    assert len(set(mapping.values())) == len(mapping)
    assert all(len(v) == 1 for v in mapping.values())


def test_remap_roundtrip(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("বাংলা ভাষা\nআমি ভালো আছি\n", encoding="utf-8")
    mapping = build_akshara_map(corpus)
    inverse = build_inverse_map(mapping)

    text = "আমি বাংলা"
    remapped = remap_text(text, mapping)
    # Each akshara becomes one codepoint; spaces are preserved.
    assert " " in remapped
    assert unmap_text(remapped, inverse) == text


def test_prepare_grapheme_corpus_caches(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("বাংলা ভাষা\nআমি ভালো আছি\n", encoding="utf-8")
    map_path = tmp_path / "akshara_map.json"
    remapped_path = tmp_path / "remapped.txt"
    meta_path = tmp_path / "remapped.meta.json"

    mapping, n_symbols = prepare_grapheme_corpus(
        corpus, remapped_path, map_path, meta_path, max_training_lines=None, seed=42
    )
    assert n_symbols == len(mapping) > 0
    assert map_path.exists()
    assert remapped_path.exists()
    assert meta_path.exists()

    # Second call reuses cached artifacts and returns an identical mapping.
    mapping2, n_symbols2 = prepare_grapheme_corpus(
        corpus, remapped_path, map_path, meta_path, max_training_lines=None, seed=42
    )
    assert mapping2 == mapping
    assert n_symbols2 == n_symbols


def test_remapped_training_corpus_subsample(tmp_path):
    from tokenizer_bn.segmentation.remap import build_remapped_training_corpus

    corpus = tmp_path / "corpus.txt"
    lines = [f"বাংলা লাইন {i}" for i in range(100)]
    corpus.write_text("\n".join(lines), encoding="utf-8")
    mapping = build_akshara_map(corpus)
    out = tmp_path / "sub.txt"
    n = build_remapped_training_corpus(corpus, out, mapping, max_lines=10, seed=42)
    assert n == 10
    assert len(out.read_text(encoding="utf-8").strip().splitlines()) == 10


def test_validate_segmentation():
    samples = ["বাংলা", "আমি ভালো আছি", "স্কুলে যাই"]
    result = validate_segmentation(samples)
    assert result["num_samples"] == 3
    assert result["total_aksharas"] > 0
