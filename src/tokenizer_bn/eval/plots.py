"""Plotting utilities for evaluation results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")


def plot_metric_bars(results_df: pd.DataFrame, metric: str, path: Path, title: str | None = None) -> None:
  fig, ax = plt.subplots(figsize=(12, 6))
  subset = results_df[results_df["metric"] == metric]
  sns.barplot(data=subset, x="tokenizer", y="value", ax=ax, palette="Set2")
  ax.set_title(title or metric.replace("_", " ").title())
  ax.set_ylabel(metric)
  ax.tick_params(axis="x", rotation=45)
  fig.tight_layout()
  fig.savefig(path, dpi=150)
  plt.close(fig)


def plot_all_metrics(results_df: pd.DataFrame, out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  for metric in results_df["metric"].unique():
    plot_metric_bars(results_df, metric, out_dir / f"{metric}.png")


def plot_parity_comparison(results_df: pd.DataFrame, path: Path) -> None:
  parity = results_df[results_df["metric"] == "parity_bn_en"]
  if parity.empty:
    return
  fig, ax = plt.subplots(figsize=(10, 6))
  sns.barplot(data=parity, x="tokenizer", y="value", ax=ax, color="tomato")
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
