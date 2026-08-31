"""Unified tokenizer wrapper for internal variants and external baselines."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import sentencepiece as spm

from tokenizer_bn.config import Config, load_config
from tokenizer_bn.train.train_variants import InitUnit, ModelType, variant_name

logger = logging.getLogger(__name__)


class SentencePieceTokenizer:
    """Wrapper around a trained SentencePiece model."""

    def __init__(self, name: str, model_path: Path):
        self.name = name
        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(str(model_path))

    def encode(self, text: str) -> list[int]:
        return self._sp.EncodeAsIds(text)

    def tokenize(self, text: str) -> list[str]:
        return self._sp.EncodeAsPieces(text)

    def decode(self, ids: list[int]) -> str:
        return self._sp.DecodeIds(ids)

    def batch_token_lengths(self, texts: list[str]) -> list[int]:
        return [len(self._sp.EncodeAsIds(text)) for text in texts]


class TiktokenTokenizer:
    """GPT-4 / tiktoken baseline."""

    def __init__(self, name: str = "gpt4_tiktoken", encoding_name: str = "cl100k_base"):
        import tiktoken
        self.name = name
        self._enc = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text)

    def tokenize(self, text: str) -> list[str]:
        return [self._enc.decode([t]) for t in self._enc.encode(text)]

    def batch_token_lengths(self, texts: list[str]) -> list[int]:
        return [len(self._enc.encode(text)) for text in texts]


class HFTokenizer:
    """HuggingFace AutoTokenizer baseline."""

    def __init__(self, name: str, model_id: str):
        from transformers import AutoTokenizer
        self.name = name
        self._tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def tokenize(self, text: str) -> list[str]:
        return self._tok.tokenize(text)

    def batch_token_lengths(self, texts: list[str]) -> list[int]:
        encoded = self._tok(texts, add_special_tokens=False, return_attention_mask=True)
        return [sum(mask) for mask in encoded["attention_mask"]]


def load_internal_variant(config: Config, init: InitUnit, model: ModelType) -> Optional[SentencePieceTokenizer]:
    name = variant_name(init, model)
    model_path = config.paths.models_dir / name / f"{name}.model"
    if not model_path.exists():
        logger.warning("Internal variant not found: %s", model_path)
        return None
    return SentencePieceTokenizer(name, model_path)


def load_all_tokenizers(config: Config | None = None) -> list:
    """Load all available internal variants and external baselines."""
    cfg = config or load_config()
    tokenizers = []

    for init in InitUnit:
        for model in ModelType:
            tok = load_internal_variant(cfg, init, model)
            if tok:
                tokenizers.append(tok)

    _try_load_baseline(tokenizers, lambda: TiktokenTokenizer())

    baseline_specs = [
        ("llama3", "meta-llama/Llama-3.2-1B"),
        ("banglabert", "sagorsarker/bangla-bert-base"),
    ]
    for name, model_id in baseline_specs:
        _try_load_baseline(tokenizers, lambda n=name, m=model_id: HFTokenizer(n, m))

    return tokenizers


def _try_load_baseline(tokenizers: list, factory) -> None:
    try:
        tok = factory()
        tokenizers.append(tok)
        logger.info("Loaded baseline: %s", tok.name)
    except Exception as exc:
        logger.warning("Could not load baseline: %s", exc)
