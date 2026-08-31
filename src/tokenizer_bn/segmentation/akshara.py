"""Extended grapheme-cluster (akshara) segmentation for Bengali."""

from __future__ import annotations

import regex as re

# UAX #29 extended grapheme cluster boundary
_GRAPHEME_PATTERN = re.compile(r"\X", re.UNICODE)


def segment_aksharas(text: str) -> list[str]:
    """Split text into extended grapheme clusters (aksharas)."""
    if not text:
        return []
    return _GRAPHEME_PATTERN.findall(text)


def count_aksharas(text: str) -> int:
    return len(segment_aksharas(text))


def akshara_fertility(text: str) -> float:
    """Characters per akshara (should be ~1.0 for well-formed clusters)."""
    clusters = segment_aksharas(text)
    if not clusters:
        return 0.0
    return len(text) / len(clusters)


def validate_segmentation(samples: list[str]) -> dict:
    """Run basic validation checks on akshara segmentation."""
    results = {
        "num_samples": len(samples),
        "total_aksharas": 0,
        "avg_aksharas_per_sample": 0.0,
        "max_akshara_len": 0,
        "samples_with_empty": 0,
    }
    if not samples:
        return results

    akshara_counts = []
    max_len = 0
    for text in samples:
        clusters = segment_aksharas(text)
        akshara_counts.append(len(clusters))
        results["total_aksharas"] += len(clusters)
        for c in clusters:
            max_len = max(max_len, len(c))
        if not clusters:
            results["samples_with_empty"] += 1

    results["avg_aksharas_per_sample"] = sum(akshara_counts) / len(akshara_counts)
    results["max_akshara_len"] = max_len
    return results
