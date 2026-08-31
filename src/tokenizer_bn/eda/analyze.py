"""Exploratory data analysis for the processed Bangla corpus."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tokenizer_bn.checkpoint import CheckpointManager
from tokenizer_bn.config import Config, ensure_dirs, load_config
from tokenizer_bn.data.bangla_filter import bangla_char_ratio, count_bengali_chars
from tokenizer_bn.data.ingest import stream_txt_lines
from tokenizer_bn.logging_utils import get_logger
from tokenizer_bn.segmentation.akshara import count_aksharas, segment_aksharas

STEP = "eda"
sns.set_theme(style="whitegrid", palette="muted")


def _sample_corpus_lines(corpus_path: Path, n: int, seed: int = 42) -> list[str]:
    """Reservoir-sample lines from the corpus without loading it all."""
    rng = random.Random(seed)
    reservoir: list[str] = []
    for i, line in enumerate(stream_txt_lines(corpus_path)):
        if i < n:
            reservoir.append(line)
        else:
            j = rng.randint(0, i)
            if j < n:
                reservoir[j] = line
    return reservoir


def run_eda(config: Config | None = None, ckpt: CheckpointManager | None = None) -> dict:
    """Run EDA on processed corpus and save stats + charts."""
    cfg = config or load_config()
    logger = get_logger(STEP, cfg)
    ensure_dirs(cfg)
    out_dir = cfg.paths.results_dir / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = cfg.corpus_path
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}. Run build-corpus first.")

    logger.info("Sampling %d lines from %s", cfg.eda.sample_lines, corpus_path)
    lines = _sample_corpus_lines(corpus_path, cfg.eda.sample_lines, seed=cfg.training.seed)
    logger.info("Sampled %d lines", len(lines))

    # Compute per-line stats
    char_lens = [len(line) for line in lines]
    akshara_lens = [count_aksharas(line) for line in lines]
    word_lens = [len(line.split()) for line in lines]
    bangla_ratios = [bangla_char_ratio(line) for line in lines]

    # Akshara frequency
    akshara_counter: Counter = Counter()
    for line in lines:
        akshara_counter.update(segment_aksharas(line))

    stats = {
        "num_lines_sampled": len(lines),
        "char_len_mean": sum(char_lens) / len(char_lens) if char_lens else 0,
        "char_len_median": sorted(char_lens)[len(char_lens) // 2] if char_lens else 0,
        "akshara_len_mean": sum(akshara_lens) / len(akshara_lens) if akshara_lens else 0,
        "word_len_mean": sum(word_lens) / len(word_lens) if word_lens else 0,
        "bangla_ratio_mean": sum(bangla_ratios) / len(bangla_ratios) if bangla_ratios else 0,
        "unique_aksharas": len(akshara_counter),
        "top_aksharas": akshara_counter.most_common(cfg.eda.top_n_aksharas),
    }

    with open(out_dir / "eda_stats.json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)

    # --- Charts ---
    _plot_length_distribution(char_lens, "Character Length", out_dir / "char_length_dist.png")
    _plot_length_distribution(akshara_lens, "Akshara Count", out_dir / "akshara_count_dist.png")
    _plot_length_distribution(word_lens, "Word Count", out_dir / "word_count_dist.png")
    _plot_bangla_ratio(bangla_ratios, out_dir / "bangla_ratio_dist.png")
    _plot_top_aksharas(akshara_counter, cfg.eda.top_n_aksharas, out_dir / "top_aksharas.png")

    # Per-source contribution from manifest
    manifest_path = cfg.manifest_path
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        _plot_source_contribution(manifest, out_dir / "source_contribution.png")

    logger.info("EDA complete. Results in %s", out_dir)
    return stats


def _plot_length_distribution(lengths: list[int], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(lengths, bins=50, color="steelblue", edgecolor="white", alpha=0.85)
    ax.set_xlabel(title)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of {title}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_bangla_ratio(ratios: list[float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(ratios, bins=30, color="seagreen", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Bengali Script Ratio")
    ax.set_ylabel("Frequency")
    ax.set_title("Bengali Script Purity Distribution")
    ax.axvline(0.5, color="red", linestyle="--", label="min_ratio=0.5")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_top_aksharas(counter: Counter, top_n: int, path: Path) -> None:
    top = counter.most_common(top_n)
    if not top:
        return
    labels, counts = zip(*top)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(range(len(labels)), counts, color="coral", alpha=0.85)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontfamily="Nirmala UI")
    ax.invert_yaxis()
    ax.set_xlabel("Frequency")
    ax.set_title(f"Top {top_n} Aksharas")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_source_contribution(manifest: dict, path: Path) -> None:
    sources = manifest.get("sources", {})
    if not sources:
        return
    names = list(sources.keys())
    kept = [sources[n]["lines_kept"] for n in names]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(names, kept, color="mediumpurple", alpha=0.85)
    ax.set_xlabel("Lines Kept")
    ax.set_title("Per-Source Corpus Contribution")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
