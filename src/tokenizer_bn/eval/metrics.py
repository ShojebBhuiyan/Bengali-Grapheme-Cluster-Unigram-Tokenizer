"""Tokenizer evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class TokenizerLike(Protocol):
  name: str

  def encode(self, text: str) -> list[int]: ...

  def tokenize(self, text: str) -> list[str]: ...


@dataclass
class MetricResult:
  name: str
  value: float
  n_samples: int


def fertility(tokenizer: TokenizerLike, texts: list[str]) -> MetricResult:
  """Average tokens per whitespace-delimited word."""
  total_words = 0
  total_tokens = 0
  for text in texts:
    words = text.split()
    if not words:
      continue
    tokens = tokenizer.tokenize(text)
    total_words += len(words)
    total_tokens += len(tokens)
  value = total_tokens / total_words if total_words else 0.0
  return MetricResult("fertility", value, len(texts))


def chars_per_token(tokenizer: TokenizerLike, texts: list[str]) -> MetricResult:
  """Average characters per token (higher = more efficient)."""
  total_chars = 0
  total_tokens = 0
  for text in texts:
    tokens = tokenizer.tokenize(text)
    if not tokens:
      continue
    total_chars += len(text.replace(" ", ""))
    total_tokens += len(tokens)
  value = total_chars / total_tokens if total_tokens else 0.0
  return MetricResult("chars_per_token", value, len(texts))


def parity_relative_to_english(
  tokenizer: TokenizerLike,
  baseline_tokenizer: TokenizerLike,
  parallel_pairs: list[tuple[str, str]],
) -> MetricResult:
  """Ratio of Bengali tokens to English tokens (lower = better parity)."""
  ratios = []
  for en_text, bn_text in parallel_pairs:
    en_tokens = len(tokenizer.tokenize(en_text))
    bn_tokens = len(tokenizer.tokenize(bn_text))
    if en_tokens > 0:
      ratios.append(bn_tokens / en_tokens)
  value = sum(ratios) / len(ratios) if ratios else 0.0
  return MetricResult("parity_bn_en", value, len(ratios))


def single_token_retention_rate(
  tokenizer: TokenizerLike,
  word_list: list[str],
) -> MetricResult:
  """Fraction of words encoded as a single token (STRR)."""
  retained = 0
  for word in word_list:
    tokens = tokenizer.tokenize(word)
    if len(tokens) == 1:
      retained += 1
  value = retained / len(word_list) if word_list else 0.0
  return MetricResult("strr", value, len(word_list))


def compute_all_metrics(
  tokenizer: TokenizerLike,
  eval_texts: list[str],
  parallel_pairs: list[tuple[str, str]] | None = None,
  word_list: list[str] | None = None,
  baseline_tokenizer: TokenizerLike | None = None,
) -> dict[str, MetricResult]:
  """Compute all metrics for a tokenizer."""
  results = {
    "fertility": fertility(tokenizer, eval_texts),
    "chars_per_token": chars_per_token(tokenizer, eval_texts),
  }
  if parallel_pairs and baseline_tokenizer:
    results["parity_bn_en"] = parity_relative_to_english(
      tokenizer, baseline_tokenizer, parallel_pairs
    )
  if word_list:
    results["strr"] = single_token_retention_rate(tokenizer, word_list)
  return results
