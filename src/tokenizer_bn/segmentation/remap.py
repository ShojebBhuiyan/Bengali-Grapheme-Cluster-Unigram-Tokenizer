"""Grapheme-to-synthetic-identifier remapping for SentencePiece training."""

from __future__ import annotations

import json
from pathlib import Path

from tokenizer_bn.segmentation.akshara import segment_aksharas

# SentencePiece user-defined symbols prefix
_SYNTHETIC_PREFIX = "\u2581"  # ▁ (SP default space marker, reused as prefix)
_SPECIAL_CHARS = {"\n": "<NL>", "\t": "<TAB>", " ": "<SP>"}


class GraphemeRemapper:
    """Map aksharas to synthetic identifiers for SentencePiece input."""

    def __init__(self) -> None:
        self.grapheme_to_id: dict[str, str] = {}
        self.id_to_grapheme: dict[str, str] = {}
        self._counter = 0

    def _next_id(self, grapheme: str) -> str:
        if grapheme in self.grapheme_to_id:
            return self.grapheme_to_id[grapheme]
        synthetic = f"<G{self._counter:06d}>"
        self._counter += 1
        self.grapheme_to_id[grapheme] = synthetic
        self.id_to_grapheme[synthetic] = grapheme
        return synthetic

    def encode_line(self, text: str) -> str:
        """Convert a line of Bengali text to space-separated synthetic IDs."""
        tokens = []
        for cluster in segment_aksharas(text):
            if cluster in _SPECIAL_CHARS:
                tokens.append(_SPECIAL_CHARS[cluster])
            elif cluster == " ":
                tokens.append(_SPECIAL_CHARS[" "])
            else:
                tokens.append(self._next_id(cluster))
        return " ".join(tokens)

    def decode_line(self, synthetic_line: str) -> str:
        """Convert synthetic IDs back to Bengali text."""
        parts = synthetic_line.split()
        result = []
        for part in parts:
            if part in self.id_to_grapheme:
                result.append(self.id_to_grapheme[part])
            elif part in _SPECIAL_CHARS.values():
                inv = {v: k for k, v in _SPECIAL_CHARS.items()}
                result.append(inv.get(part, part))
            else:
                result.append(part)
        return "".join(result)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "grapheme_to_id": self.grapheme_to_id,
            "id_to_grapheme": self.id_to_grapheme,
            "counter": self._counter,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "GraphemeRemapper":
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        remapper = cls()
        remapper.grapheme_to_id = data["grapheme_to_id"]
        remapper.id_to_grapheme = data["id_to_grapheme"]
        remapper._counter = data["counter"]
        return remapper


def remap_corpus(input_path: Path, output_path: Path, vocab_path: Path | None = None) -> GraphemeRemapper:
    """Remap an entire corpus file to synthetic grapheme IDs."""
    remapper = GraphemeRemapper()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(input_path, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if line:
                fout.write(remapper.encode_line(line) + "\n")
    if vocab_path:
        remapper.save(vocab_path)
    return remapper
