"""Tests for akshara segmentation and grapheme remapping."""

from tokenizer_bn.segmentation.akshara import count_aksharas, segment_aksharas, validate_segmentation
from tokenizer_bn.segmentation.remap import GraphemeRemapper


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


def test_remapper_roundtrip():
  remapper = GraphemeRemapper()
  original = "বাংলা ভাষা"
  encoded = remapper.encode_line(original)
  decoded = remapper.decode_line(encoded)
  assert decoded == original


def test_validate_segmentation():
  samples = ["বাংলা", "আমি ভালো আছি", "স্কুলে যাই"]
  result = validate_segmentation(samples)
  assert result["num_samples"] == 3
  assert result["total_aksharas"] > 0
