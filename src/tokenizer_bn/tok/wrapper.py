"""Unified tokenizer wrapper for internal variants and external baselines."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import sentencepiece as spm

from tokenizer_bn.config import Config, load_config
from tokenizer_bn.segmentation.remap import GraphemeRemapper
from tokenizer_bn.train.train_variants import InitUnit, ModelType, variant_name

logger = logging.getLogger(__name__)


class SentencePieceTokenizer:
  """Wrapper around a trained SentencePiece model."""

  def __init__(self, name: str, model_path: Path, remapper: GraphemeRemapper | None = None):
    self.name = name
    self._sp = spm.SentencePieceProcessor()
    self._sp.Load(str(model_path))
    self._remapper = remapper

  def encode(self, text: str) -> list[int]:
    if self._remapper:
      text = self._remapper.encode_line(text)
    return self._sp.EncodeAsIds(text)

  def tokenize(self, text: str) -> list[str]:
    if self._remapper:
      text = self._remapper.encode_line(text)
    return self._sp.EncodeAsPieces(text)

  def decode(self, ids: list[int]) -> str:
    text = self._sp.DecodeIds(ids)
    if self._remapper:
      text = self._remapper.decode_line(text)
    return text


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


def load_internal_variant(config: Config, init: InitUnit, model: ModelType) -> Optional[SentencePieceTokenizer]:
  name = variant_name(init, model)
  model_path = config.paths.models_dir / name / f"{name}.model"
  if not model_path.exists():
    logger.warning("Internal variant not found: %s", model_path)
    return None
  remapper = None
  if init == InitUnit.GRAPHEME:
    vocab_path = config.paths.models_dir / name / "grapheme_vocab.json"
    if vocab_path.exists():
      remapper = GraphemeRemapper.load(vocab_path)
  return SentencePieceTokenizer(name, model_path, remapper=remapper)


def load_all_tokenizers(config: Config | None = None) -> list:
  """Load all available internal variants and external baselines."""
  cfg = config or load_config()
  tokenizers = []

  # Internal 2x2 variants
  for init in InitUnit:
    for model in ModelType:
      tok = load_internal_variant(cfg, init, model)
      if tok:
        tokenizers.append(tok)

  # External baselines with graceful fallback
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
