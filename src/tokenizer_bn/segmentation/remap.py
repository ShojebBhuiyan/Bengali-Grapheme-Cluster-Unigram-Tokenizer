"""Grapheme-cluster (akshara) remapping for true grapheme-initialized tokenizers.

The 2x2 ablation compares the *initialization unit* (grapheme vs byte). To make
"grapheme" a real base alphabet, we bijectively remap each distinct akshara
(extended grapheme cluster) to a single Private-Use-Area codepoint before
training SentencePiece. Because every akshara becomes exactly one character:

* the base symbol set SentencePiece sees is the akshara inventory, and
* every learned piece (unigram or BPE merge) is a sequence of *whole* aksharas,
  so pieces never split inside a cluster.

This lets both grapheme variants grow to the target vocab (they are no longer
capped by an atomic seed set), which is what makes RQ1 (grapheme vs byte) a
controlled comparison. At inference the tokenizer remaps input the same way and
inverts the mapping for decoding.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from tokenizer_bn.data.ingest import stream_txt_lines
from tokenizer_bn.segmentation.akshara import segment_aksharas

# Supplementary Private-Use Area (Plane 15): U+F0000..U+FFFFD => 65534 slots.
# Far larger than the Bengali akshara inventory (a few thousand clusters), and
# disjoint from any character that appears in normal text.
_PUA_BASE = 0xF0000
_PUA_LIMIT = 0xFFFFD
_MAX_SYMBOLS = _PUA_LIMIT - _PUA_BASE  # reserve one slot for OOV
# All aksharas that do not fit in the mapping collapse to this single symbol.
OOV_CHAR = chr(_PUA_LIMIT)
# Real spaces are preserved so SentencePiece keeps normal word-boundary handling.
_SPACE = " "


def build_akshara_map(corpus_path: Path, max_symbols: int = _MAX_SYMBOLS) -> dict[str, str]:
    """Scan the full corpus and map each distinct akshara to a unique PUA char.

    Aksharas are ordered by descending frequency so the most common clusters get
    the lowest codepoints (purely for determinism/readability). Returns a mapping
    ``{akshara: pua_char}``. Spaces are handled separately and never mapped.
    """
    counter: Counter[str] = Counter()
    for line in stream_txt_lines(corpus_path):
        for cluster in segment_aksharas(line):
            if cluster != _SPACE:
                counter[cluster] += 1

    cap = min(max_symbols, _MAX_SYMBOLS)
    mapping: dict[str, str] = {}
    for idx, (cluster, _freq) in enumerate(counter.most_common(cap)):
        mapping[cluster] = chr(_PUA_BASE + idx)
    return mapping


def remap_text(text: str, mapping: dict[str, str]) -> str:
    """Remap a string to PUA codepoints (one per akshara); spaces are preserved."""
    out: list[str] = []
    for cluster in segment_aksharas(text):
        if cluster == _SPACE:
            out.append(_SPACE)
        else:
            out.append(mapping.get(cluster, OOV_CHAR))
    return "".join(out)


def build_remapped_corpus(corpus_path: Path, output_path: Path, mapping: dict[str, str]) -> int:
    """Write an akshara-remapped copy of the corpus for SentencePiece training."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(corpus_path, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            fout.write(remap_text(line, mapping) + "\n")
            count += 1
    return count


def save_akshara_map(mapping: dict[str, str], path: Path) -> None:
    """Persist the akshara->PUA mapping as JSON (codepoints stored as ints)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {cluster: ord(pua) for cluster, pua in mapping.items()}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(serializable, fh, ensure_ascii=False)


def load_akshara_map(path: Path) -> dict[str, str]:
    """Load an akshara->PUA mapping saved by :func:`save_akshara_map`."""
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return {cluster: chr(int(codepoint)) for cluster, codepoint in raw.items()}


def build_inverse_map(mapping: dict[str, str]) -> dict[str, str]:
    """Return the PUA-char -> akshara inverse of ``mapping``."""
    return {pua: cluster for cluster, pua in mapping.items()}


def unmap_text(text: str, inverse: dict[str, str]) -> str:
    """Invert :func:`remap_text`, turning PUA codepoints back into aksharas."""
    return "".join(inverse.get(ch, "" if ch == OOV_CHAR else ch) for ch in text)


def prepare_grapheme_corpus(
    corpus_path: Path,
    remapped_corpus_path: Path,
    map_path: Path,
    max_symbols: int = _MAX_SYMBOLS,
) -> tuple[dict[str, str], int]:
    """Build (and cache) the shared akshara map + remapped corpus.

    Reuses existing artifacts when both are present so the two grapheme variants
    do not each re-scan the full corpus. Returns ``(mapping, num_symbols)``.
    """
    if map_path.exists() and remapped_corpus_path.exists():
        mapping = load_akshara_map(map_path)
        return mapping, len(mapping)

    mapping = build_akshara_map(corpus_path, max_symbols=max_symbols)
    save_akshara_map(mapping, map_path)
    build_remapped_corpus(corpus_path, remapped_corpus_path, mapping)
    return mapping, len(mapping)
