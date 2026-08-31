"""Evaluation harness: run metrics, statistical tests, and produce charts."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
from scipy import stats

from tokenizer_bn.checkpoint import CheckpointManager
from tokenizer_bn.config import Config, ensure_dirs, load_config
from tokenizer_bn.data.ingest import stream_txt_lines
from tokenizer_bn.device import log_device_info, resolve_device, torch_available
from tokenizer_bn.eval.corpus_metrics import compute_corpus_metrics
from tokenizer_bn.eval.gpu_metrics import compute_all_metrics_gpu, paired_fertility_diffs_gpu
from tokenizer_bn.eval.metrics import compute_all_metrics
from tokenizer_bn.eval.plots import (
    plot_all_metrics,
    plot_corpus_metrics_dashboard,
    plot_eval_metrics_dashboard,
    plot_metric_heatmap,
    plot_parity_comparison,
    plot_rq_comparisons,
)
from tokenizer_bn.logging_utils import get_logger
from tokenizer_bn.tok.wrapper import TiktokenTokenizer, load_all_tokenizers

STEP = "evaluate"


def _sample_lines(path: Path, n: int, seed: int = 42) -> list[str]:
  rng = random.Random(seed)
  reservoir: list[str] = []
  for i, line in enumerate(stream_txt_lines(path)):
    if i < n:
      reservoir.append(line)
    else:
      j = rng.randint(0, i)
      if j < n:
        reservoir[j] = line
  return reservoir


def _load_parallel_pairs(config: Config, max_pairs: int = 5000) -> list[tuple[str, str]]:
  path = config.parallel_path
  if not path.exists():
    return []
  df = pd.read_parquet(path)
  pairs = list(zip(df["en"].astype(str), df["bn"].astype(str)))
  if len(pairs) > max_pairs:
    rng = random.Random(config.training.seed)
    pairs = rng.sample(pairs, max_pairs)
  return pairs


def _build_word_list(texts: list[str], size: int) -> list[str]:
  words: set[str] = set()
  for text in texts:
    for w in text.split():
      if len(w) >= 2:
        words.add(w)
      if len(words) >= size:
        break
    if len(words) >= size:
      break
  return list(words)[:size]


def _paired_test(diffs: list[float]) -> dict:
  if len(diffs) < 5:
    return {"test": "insufficient_data", "p_value": 1.0, "statistic": 0.0}
  # Shapiro-Wilk for normality
  _, normality_p = stats.shapiro(diffs[:min(5000, len(diffs))])
  if normality_p > 0.05:
    stat, p = stats.ttest_1samp(diffs, 0.0)
    test_name = "paired_t_test"
  else:
    stat, p = stats.wilcoxon(diffs)
    test_name = "wilcoxon"
  return {"test": test_name, "p_value": float(p), "statistic": float(stat)}


def _use_gpu_metrics(config: Config) -> bool:
  return config.device.use_gpu and torch_available() and resolve_device(config.device.device) != "cpu"


def run_evaluation(config: Config | None = None, ckpt: CheckpointManager | None = None) -> dict:
  cfg = config or load_config()
  ensure_dirs(cfg)
  log = get_logger(STEP, cfg)

  resolved_device = log_device_info(log, cfg.device.device)
  use_gpu = _use_gpu_metrics(cfg)
  if use_gpu:
    log.info("Using GPU-accelerated metric computation on %s", resolved_device)
  else:
    log.info("Using CPU metric computation")

  corpus_path = cfg.corpus_path
  if not corpus_path.exists():
    raise FileNotFoundError(f"Corpus not found: {corpus_path}")

  log.info("Loading tokenizers...")
  tokenizers = load_all_tokenizers(cfg)
  if not tokenizers:
    raise RuntimeError("No tokenizers available for evaluation")

  log.info("Sampling %d eval lines", cfg.evaluation.eval_sample_lines)
  eval_texts = _sample_lines(corpus_path, cfg.evaluation.eval_sample_lines, seed=cfg.training.seed)
  parallel_pairs = _load_parallel_pairs(cfg)
  word_list = _build_word_list(eval_texts, cfg.evaluation.strr_word_list_size)

  # English baseline for parity
  try:
    en_baseline = TiktokenTokenizer()
  except Exception:
    en_baseline = None

  rows = []
  corpus_rows = []
  per_tokenizer_metrics: dict[str, dict] = {}

  baseline_for_parity = en_baseline if en_baseline else tokenizers[0]
  for tok in tokenizers:
    log.info("Evaluating: %s", tok.name)
    if use_gpu:
      metrics = compute_all_metrics_gpu(
        tok,
        eval_texts,
        device=resolved_device,
        batch_size=cfg.device.batch_size,
        parallel_pairs=parallel_pairs if parallel_pairs else None,
        word_list=word_list,
        baseline_tokenizer=baseline_for_parity,
      )
    else:
      metrics = compute_all_metrics(
        tok,
        eval_texts,
        parallel_pairs=parallel_pairs if parallel_pairs else None,
        word_list=word_list,
        baseline_tokenizer=baseline_for_parity,
      )
    per_tokenizer_metrics[tok.name] = metrics
    for metric_name, result in metrics.items():
      rows.append({
        "tokenizer": tok.name,
        "metric": metric_name,
        "value": result.value,
        "n_samples": result.n_samples,
        "scope": "sample",
      })

  # Full-corpus metrics
  log.info("Computing full-corpus metrics (max_lines=%s)", cfg.evaluation.corpus_eval_max_lines or "all")
  for tok in tokenizers:
    log.info("Full corpus eval: %s", tok.name)
    corpus_metrics = compute_corpus_metrics(
      tok,
      corpus_path,
      batch_size=cfg.evaluation.corpus_eval_batch_size,
      max_lines=cfg.evaluation.corpus_eval_max_lines,
    )
    for metric_name, result in corpus_metrics.items():
      corpus_rows.append({
        "tokenizer": tok.name,
        "metric": metric_name,
        "value": result.value,
        "n_samples": result.n_samples,
        "scope": "full_corpus",
      })
      rows.append({
        "tokenizer": tok.name,
        "metric": metric_name,
        "value": result.value,
        "n_samples": result.n_samples,
        "scope": "full_corpus",
      })

  results_df = pd.DataFrame(rows)
  corpus_df = pd.DataFrame(corpus_rows)
  tables_dir = cfg.paths.results_dir / "tables"
  figures_dir = cfg.paths.results_dir / "figures"
  tables_dir.mkdir(parents=True, exist_ok=True)
  figures_dir.mkdir(parents=True, exist_ok=True)

  results_df.to_csv(tables_dir / "eval_metrics.csv", index=False)
  corpus_df.to_csv(tables_dir / "corpus_metrics.csv", index=False)
  log.info("Saved metrics table to %s", tables_dir / "eval_metrics.csv")
  log.info("Saved corpus metrics to %s", tables_dir / "corpus_metrics.csv")

  # Statistical tests for RQ1/RQ2/RQ3
  stats_rows = _run_rq_tests(tokenizers, eval_texts, cfg, use_gpu=use_gpu, device=resolved_device)
  stats_df = pd.DataFrame(stats_rows)
  if not stats_df.empty:
    stats_df.to_csv(tables_dir / "rq_statistical_tests.csv", index=False)

  # Plots
  plot_all_metrics(results_df, figures_dir)
  plot_corpus_metrics_dashboard(corpus_df, figures_dir / "corpus_metrics_dashboard.png")
  plot_eval_metrics_dashboard(results_df, figures_dir / "eval_metrics_dashboard.png")
  plot_metric_heatmap(results_df, figures_dir / "metrics_heatmap.png")
  plot_parity_comparison(results_df, figures_dir / "parity_comparison.png")
  if not stats_df.empty:
    plot_rq_comparisons(stats_df, figures_dir / "rq_statistical_tests.png")

  summary = {
    "num_tokenizers": len(tokenizers),
    "num_eval_texts": len(eval_texts),
    "num_parallel_pairs": len(parallel_pairs),
    "corpus_eval_max_lines": cfg.evaluation.corpus_eval_max_lines,
    "device": resolved_device,
    "gpu_metrics": use_gpu,
    "metrics": rows,
    "corpus_metrics": corpus_rows,
    "statistical_tests": stats_rows,
  }
  with open(tables_dir / "eval_summary.json", "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=2, ensure_ascii=False)

  log.info("Evaluation complete")
  return summary


def _run_rq_tests(
  tokenizers,
  eval_texts: list[str],
  config: Config,
  use_gpu: bool = False,
  device: str = "cpu",
) -> list[dict]:
  """Paired comparisons for RQ1 (init unit), RQ2 (model type), RQ3 (best vs baselines)."""
  tok_map = {t.name: t for t in tokenizers}
  stats_rows = []

  comparisons = [
    ("RQ1_grapheme_vs_byte_unigram", "grapheme_unigram", "byte_unigram"),
    ("RQ1_grapheme_vs_byte_bpe", "grapheme_bpe", "byte_bpe"),
    ("RQ2_unigram_vs_bpe_grapheme", "grapheme_unigram", "grapheme_bpe"),
    ("RQ2_unigram_vs_bpe_byte", "byte_unigram", "byte_bpe"),
    ("RQ3_grapheme_unigram_vs_gpt4", "grapheme_unigram", "gpt4_tiktoken"),
  ]

  for label, name_a, name_b in comparisons:
    if name_a not in tok_map or name_b not in tok_map:
      continue
    tok_a, tok_b = tok_map[name_a], tok_map[name_b]
    if use_gpu:
      diffs = paired_fertility_diffs_gpu(
        tok_a, tok_b, eval_texts, device=device, batch_size=config.device.batch_size
      )
    else:
      fert_a = [len(tok_a.tokenize(t)) / max(len(t.split()), 1) for t in eval_texts]
      fert_b = [len(tok_b.tokenize(t)) / max(len(t.split()), 1) for t in eval_texts]
      diffs = [a - b for a, b in zip(fert_a, fert_b)]
    test_result = _paired_test(diffs)
    stats_rows.append({
      "comparison": label,
      "metric": "fertility",
      "tokenizer_a": name_a,
      "tokenizer_b": name_b,
      "mean_diff": sum(diffs) / len(diffs),
      **test_result,
    })

  return stats_rows
