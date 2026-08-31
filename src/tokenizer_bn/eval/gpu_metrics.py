"""GPU-accelerated metric computation for tokenizer evaluation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from tokenizer_bn.device import torch_available
from tokenizer_bn.eval.metrics import MetricResult, TokenizerLike

if TYPE_CHECKING:
    import torch


def _require_torch():
    if not torch_available():
        raise RuntimeError("PyTorch is required for GPU metrics. Install with: pip install torch")
    import torch

    return torch


def batch_token_lengths(tokenizer: TokenizerLike, texts: list[str], batch_size: int = 256) -> list[int]:
    """Compute per-text token counts, using HF batch encoding when available."""
    if hasattr(tokenizer, "batch_token_lengths"):
        lengths: list[int] = []
        for start in range(0, len(texts), batch_size):
            lengths.extend(tokenizer.batch_token_lengths(texts[start : start + batch_size]))
        return lengths

    if len(texts) < batch_size:
        return [len(tokenizer.tokenize(text)) for text in texts]

    with ThreadPoolExecutor() as pool:
        return list(pool.map(lambda text: len(tokenizer.tokenize(text)), texts))


def _tensor_mean_ratio(numerators: list[int], denominators: list[int], device: str) -> float:
    torch = _require_torch()
    num = torch.tensor(numerators, dtype=torch.float32, device=device)
    den = torch.tensor(denominators, dtype=torch.float32, device=device)
    mask = den > 0
    if not mask.any():
        return 0.0
    return (num[mask].sum() / den[mask].sum()).item()


def fertility_gpu(
    tokenizer: TokenizerLike,
    texts: list[str],
    device: str,
    batch_size: int = 256,
) -> MetricResult:
    token_lengths = batch_token_lengths(tokenizer, texts, batch_size=batch_size)
    word_lengths = [len(text.split()) for text in texts]
    value = _tensor_mean_ratio(token_lengths, word_lengths, device)
    return MetricResult("fertility", value, len(texts))


def chars_per_token_gpu(
    tokenizer: TokenizerLike,
    texts: list[str],
    device: str,
    batch_size: int = 256,
) -> MetricResult:
    torch = _require_torch()
    token_lengths = batch_token_lengths(tokenizer, texts, batch_size=batch_size)
    char_lengths = [len(text.replace(" ", "")) for text in texts]
    num = torch.tensor(char_lengths, dtype=torch.float32, device=device)
    den = torch.tensor(token_lengths, dtype=torch.float32, device=device)
    mask = den > 0
    if not mask.any():
        return MetricResult("chars_per_token", 0.0, len(texts))
    value = (num[mask].sum() / den[mask].sum()).item()
    return MetricResult("chars_per_token", value, len(texts))


def parity_relative_to_english_gpu(
    tokenizer: TokenizerLike,
    parallel_pairs: list[tuple[str, str]],
    device: str,
    batch_size: int = 256,
) -> MetricResult:
    torch = _require_torch()
    en_texts = [en for en, _ in parallel_pairs]
    bn_texts = [bn for _, bn in parallel_pairs]
    en_lengths = batch_token_lengths(tokenizer, en_texts, batch_size=batch_size)
    bn_lengths = batch_token_lengths(tokenizer, bn_texts, batch_size=batch_size)
    en_t = torch.tensor(en_lengths, dtype=torch.float32, device=device)
    bn_t = torch.tensor(bn_lengths, dtype=torch.float32, device=device)
    mask = en_t > 0
    if not mask.any():
        return MetricResult("parity_bn_en", 0.0, 0)
    value = (bn_t[mask] / en_t[mask]).mean().item()
    return MetricResult("parity_bn_en", value, int(mask.sum().item()))


def single_token_retention_rate_gpu(
    tokenizer: TokenizerLike,
    word_list: list[str],
    device: str,
    batch_size: int = 256,
) -> MetricResult:
    torch = _require_torch()
    token_lengths = batch_token_lengths(tokenizer, word_list, batch_size=batch_size)
    lengths = torch.tensor(token_lengths, dtype=torch.float32, device=device)
    value = (lengths == 1).float().mean().item()
    return MetricResult("strr", value, len(word_list))


def compute_all_metrics_gpu(
    tokenizer: TokenizerLike,
    eval_texts: list[str],
    device: str,
    batch_size: int = 256,
    parallel_pairs: list[tuple[str, str]] | None = None,
    word_list: list[str] | None = None,
    baseline_tokenizer: TokenizerLike | None = None,
) -> dict[str, MetricResult]:
    """Compute all metrics with GPU-accelerated aggregation."""
    results = {
        "fertility": fertility_gpu(tokenizer, eval_texts, device, batch_size),
        "chars_per_token": chars_per_token_gpu(tokenizer, eval_texts, device, batch_size),
    }
    if parallel_pairs and baseline_tokenizer:
        results["parity_bn_en"] = parity_relative_to_english_gpu(
            tokenizer, parallel_pairs, device, batch_size
        )
    if word_list:
        results["strr"] = single_token_retention_rate_gpu(
            tokenizer, word_list, device, batch_size
        )
    return results


def paired_fertility_diffs_gpu(
    tok_a: TokenizerLike,
    tok_b: TokenizerLike,
    eval_texts: list[str],
    device: str,
    batch_size: int = 256,
) -> list[float]:
    """Compute paired fertility differences on GPU for statistical tests."""
    torch = _require_torch()
    lengths_a = batch_token_lengths(tok_a, eval_texts, batch_size=batch_size)
    lengths_b = batch_token_lengths(tok_b, eval_texts, batch_size=batch_size)
    words = torch.tensor([max(len(text.split()), 1) for text in eval_texts], dtype=torch.float32, device=device)
    fert_a = torch.tensor(lengths_a, dtype=torch.float32, device=device) / words
    fert_b = torch.tensor(lengths_b, dtype=torch.float32, device=device) / words
    return (fert_a - fert_b).tolist()
