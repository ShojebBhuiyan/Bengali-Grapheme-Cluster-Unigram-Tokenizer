"""Tests for SentencePiece training subsample resolution."""

from tokenizer_bn.train.sentence_sample import count_corpus_lines, resolve_sp_input_sentence_size


def test_resolve_explicit_subsample(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("line\n" * 20, encoding="utf-8")
    sp_size, remapped_limit, _ = resolve_sp_input_sentence_size(10, 5_000_000, corpus)
    assert sp_size == 10
    assert remapped_limit == 10


def test_resolve_explicit_subsample_capped_to_corpus(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("line\n" * 5, encoding="utf-8")
    sp_size, remapped_limit, _ = resolve_sp_input_sentence_size(100, 5_000_000, corpus)
    assert sp_size == 5
    assert remapped_limit == 5


def test_resolve_unlimited_soft_cap(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("line\n" * 100, encoding="utf-8")
    sp_size, remapped_limit, msg = resolve_sp_input_sentence_size(0, 50, corpus)
    assert sp_size == 50
    assert remapped_limit == 50
    assert "reservoir" in msg.lower()


def test_resolve_unlimited_small_corpus(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("line\n" * 10, encoding="utf-8")
    sp_size, remapped_limit, _ = resolve_sp_input_sentence_size(0, 50, corpus)
    assert sp_size is None
    assert remapped_limit is None


def test_count_corpus_lines_skips_blank(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("a\n\nb\n   \nc\n", encoding="utf-8")
    assert count_corpus_lines(corpus) == 3
