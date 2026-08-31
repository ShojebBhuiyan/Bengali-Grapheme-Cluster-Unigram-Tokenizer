"""Plotting utilities for evaluation results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")

METRIC_LABELS = {
    "fertility": ("Average Tokens per Word (sample)", "lower"),
    "chars_per_token": ("Characters per Token (sample)", "higher"),
    "parity_bn_en": ("Parity BN/EN (sample)", "lower"),
    "strr": ("Single Token Retention Rate", "higher"),
    "corpus_token_count": ("Corpus Token Count (full corpus)", "lower"),
    "compression_ratio": ("Compression Ratio — Chars per Token (full corpus)", "higher"),
    "tokens_per_bengali_word": ("Avg Tokens per Bengali Word (full corpus)", "lower"),
}

CORPUS_METRICS = ("corpus_token_count", "compression_ratio", "tokens_per_bengali_word")


def _direction_note(direction: str) -> str:
    return "lower is better" if direction == "lower" else "higher is better"


def plot_metric_bars(
    results_df: pd.DataFrame,
    metric: str,
    path: Path,
    title: str | None = None,
    color: str | None = None,
) -> None:
    subset = results_df[results_df["metric"] == metric].copy()
    if subset.empty:
        return

    label, direction = METRIC_LABELS.get(metric, (metric.replace("_", " ").title(), ""))
    fig, ax = plt.subplots(figsize=(12, 6))
    palette = color or ("coral" if direction == "lower" else "seagreen")
    sns.barplot(data=subset, x="tokenizer", y="value", ax=ax, color=palette, hue="tokenizer", legend=False)
    ax.set_title(title or f"{label}\n({_direction_note(direction)})" if direction else label)
    ax.set_ylabel("Value")
    ax.set_xlabel("Tokenizer")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_all_metrics(results_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for metric in results_df["metric"].unique():
        plot_metric_bars(results_df, metric, out_dir / f"{metric}.png")


def plot_corpus_metrics_dashboard(corpus_df: pd.DataFrame, path: Path) -> None:
    """Three-panel dashboard for full-corpus metrics."""
    if corpus_df.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = {"lower": "coral", "higher": "seagreen"}

    for ax, metric in zip(axes, CORPUS_METRICS):
        subset = corpus_df[corpus_df["metric"] == metric]
        if subset.empty:
            ax.set_visible(False)
            continue
        label, direction = METRIC_LABELS[metric]
        sns.barplot(
            data=subset,
            x="tokenizer",
            y="value",
            ax=ax,
            color=colors[direction],
            hue="tokenizer",
            legend=False,
        )
        ax.set_title(f"{label}\n({_direction_note(direction)})", fontsize=11)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
        if metric == "corpus_token_count":
            ax.ticklabel_format(style="plain", axis="y")
            # Show millions on y-axis for readability
            ymax = subset["value"].max()
            if ymax > 1_000_000:
                ax.set_ylabel("Total tokens (millions)")
                for bar in ax.patches:
                    h = bar.get_height()
                    ax.annotate(
                        f"{h/1e6:.1f}M",
                        (bar.get_x() + bar.get_width() / 2, h),
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
            else:
                ax.set_ylabel("Total tokens")
        else:
            ax.set_ylabel("Value")

    fig.suptitle("Full Corpus Tokenizer Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_eval_metrics_dashboard(results_df: pd.DataFrame, path: Path) -> None:
    """Dashboard for sample-based evaluation metrics."""
    sample_metrics = [m for m in results_df["metric"].unique() if m not in CORPUS_METRICS]
    if not sample_metrics:
        return

    n = len(sample_metrics)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    colors = {"lower": "coral", "higher": "seagreen"}
    for ax, metric in zip(axes, sample_metrics):
        subset = results_df[results_df["metric"] == metric]
        label, direction = METRIC_LABELS.get(metric, (metric, "lower"))
        sns.barplot(
            data=subset,
            x="tokenizer",
            y="value",
            ax=ax,
            color=colors.get(direction, "steelblue"),
            hue="tokenizer",
            legend=False,
        )
        ax.set_title(f"{label}\n({_direction_note(direction)})", fontsize=10)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle("Evaluation Metrics (sampled corpus)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_metric_heatmap(results_df: pd.DataFrame, path: Path) -> None:
    """Normalized heatmap comparing tokenizers across all metrics."""
    if results_df.empty:
        return

    pivot = results_df.pivot(index="tokenizer", columns="metric", values="value")
    # Normalize each column to 0-1 for visual comparison (invert lower-is-better metrics)
    normalized = pivot.copy()
    for col in pivot.columns:
        col_min, col_max = pivot[col].min(), pivot[col].max()
        if col_max == col_min:
            normalized[col] = 0.5
            continue
        scaled = (pivot[col] - col_min) / (col_max - col_min)
        _, direction = METRIC_LABELS.get(col, ("", "lower"))
        normalized[col] = 1 - scaled if direction == "lower" else scaled

    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot) * 0.6)))
    sns.heatmap(
        normalized,
        annot=pivot.round(3),
        fmt="",
        cmap="RdYlGn",
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "Normalized score (1 = best)"},
    )
    ax.set_title("Tokenizer Comparison Heatmap\n(green = better per metric direction)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_parity_comparison(results_df: pd.DataFrame, path: Path) -> None:
    parity = results_df[results_df["metric"] == "parity_bn_en"]
    if parity.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=parity, x="tokenizer", y="value", ax=ax, color="tomato", hue="tokenizer", legend=False)
    ax.axhline(1.0, color="green", linestyle="--", label="parity=1.0")
    ax.set_title("Parity Relative to English (lower is better)")
    ax.set_ylabel("BN tokens / EN tokens")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_rq_comparisons(stats_df: pd.DataFrame, path: Path) -> None:
    """Bar chart of paired test p-values for RQ comparisons."""
    if stats_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=stats_df, x="comparison", y="p_value", hue="metric", ax=ax)
    ax.axhline(0.05, color="red", linestyle="--", label="α=0.05")
    ax.set_title("Paired Statistical Tests (RQ1/RQ2/RQ3)")
    ax.set_ylabel("p-value")
    ax.legend()
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
