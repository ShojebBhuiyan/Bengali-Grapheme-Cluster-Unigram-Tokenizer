"""Grapheme seed preparation for SentencePiece training."""

from __future__ import annotations

from pathlib import Path

from tokenizer_bn.data.ingest import stream_txt_lines
from tokenizer_bn.segmentation.akshara import segment_aksharas

_SPACE_MARKER = "<SP>"


def grapheme_tokenize_line(text: str) -> str:
    """Convert Bengali text to space-separated grapheme clusters."""
    tokens = []
    for cluster in segment_aksharas(text):
        tokens.append(_SPACE_MARKER if cluster == " " else cluster)
    return " ".join(tokens)


def build_grapheme_spaced_corpus(input_path: Path, output_path: Path) -> int:
    """Write grapheme-space-separated corpus for BPE training."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(input_path, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if line:
                fout.write(grapheme_tokenize_line(line) + "\n")
                count += 1
    return count


def build_grapheme_seed_file(corpus_path: Path, seed_path: Path, max_lines: int = 200_000) -> int:
    """Extract unique grapheme clusters from corpus and write as SP seed sentencepieces."""
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    graphemes: set[str] = set()
    for i, line in enumerate(stream_txt_lines(corpus_path)):
        if i >= max_lines:
            break
        graphemes.update(segment_aksharas(line))
    graphemes.discard("")
    graphemes.discard(" ")
    with open(seed_path, "w", encoding="utf-8") as fh:
        for g in sorted(graphemes):
            # SentencePiece seed format: <piece>\t<frequency>
            fh.write(f"{g}\t1\n")
    return len(graphemes)
