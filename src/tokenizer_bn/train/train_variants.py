"""Train 2x2 ablation tokenizer variants via SentencePiece."""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path

import sentencepiece as spm

from tokenizer_bn.checkpoint import CheckpointManager
from tokenizer_bn.config import Config, ensure_dirs, load_config
from tokenizer_bn.logging_utils import get_logger
from tokenizer_bn.segmentation.remap import build_grapheme_seed_file, build_grapheme_spaced_corpus

STEP = "train"
logger = get_logger(STEP)


class InitUnit(str, Enum):
    GRAPHEME = "grapheme"
    BYTE = "byte"


class ModelType(str, Enum):
    UNIGRAM = "unigram"
    BPE = "bpe"


VARIANTS = [
    (InitUnit.GRAPHEME, ModelType.UNIGRAM),
    (InitUnit.GRAPHEME, ModelType.BPE),
    (InitUnit.BYTE, ModelType.UNIGRAM),
    (InitUnit.BYTE, ModelType.BPE),
]


def variant_name(init: InitUnit, model: ModelType) -> str:
    return f"{init.value}_{model.value}"


def _cleanup_model_prefix(model_prefix: str) -> None:
    for ext in (".model", ".vocab"):
        path = Path(f"{model_prefix}{ext}")
        if path.exists():
            path.unlink()


def _train_sentencepiece(sp_kwargs: dict, model_prefix: str, log) -> int:
    """Train SentencePiece with adaptive vocab sizing."""
    target_vocab = sp_kwargs["vocab_size"]
    min_vocab = 128 if sp_kwargs.get("byte_fallback") else 64
    attempt_vocab = target_vocab

    while attempt_vocab >= min_vocab:
        sp_kwargs["vocab_size"] = attempt_vocab
        if "seed_sentencepiece_size" in sp_kwargs:
            sp_kwargs["seed_sentencepiece_size"] = attempt_vocab

        _cleanup_model_prefix(model_prefix)
        try:
            spm.SentencePieceTrainer.Train(**sp_kwargs)
            if Path(f"{model_prefix}.model").exists():
                if attempt_vocab != target_vocab:
                    log.warning("Trained with vocab_size=%d (target was %d)", attempt_vocab, target_vocab)
                return attempt_vocab
        except RuntimeError as exc:
            msg = str(exc)
            if "Vocabulary size too high" in msg:
                match = re.search(r"<= (\d+)", msg)
                if match:
                    attempt_vocab = int(match.group(1))
                    log.warning("Corpus supports max vocab_size=%d, retrying", attempt_vocab)
                    continue
                attempt_vocab //= 2
                log.warning("vocab_size too high, retrying with %d", attempt_vocab)
                continue
            if "smaller than required_chars" in msg:
                match = re.search(r"(\d+) vs (\d+)", msg)
                if match:
                    attempt_vocab = int(match.group(2)) + 4
                    log.warning("vocab_size too small, increasing to %d", attempt_vocab)
                    continue
            raise

    raise RuntimeError(f"Could not train model at {model_prefix} with vocab >= {min_vocab}")


def train_all_variants(config: Config | None = None, ckpt: CheckpointManager | None = None) -> dict:
    """Train all four ablation variants."""
    cfg = config or load_config()
    ensure_dirs(cfg)
    log = get_logger(STEP, cfg)
    checkpoint = ckpt or CheckpointManager(cfg)

    corpus_path = cfg.corpus_path
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    results = {}
    for init, model in VARIANTS:
        name = variant_name(init, model)
        shard_key = f"variant_{name}"
        if checkpoint.is_shard_done(STEP, shard_key):
            log.info("Skipping variant %s (checkpoint done)", name)
            results[name] = {"status": "skipped"}
            continue

        log.info("Training variant: %s", name)
        model_dir = cfg.paths.models_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)
        model_prefix = str(model_dir / name)

        train_input = corpus_path
        seed_file = None
        sp_kwargs_extra: dict = {}

        if init == InitUnit.GRAPHEME:
            if model == ModelType.UNIGRAM:
                seed_path = model_dir / "grapheme_seed.txt"
                if not seed_path.exists():
                    log.info("Building grapheme seed file for %s", name)
                    n_seeds = build_grapheme_seed_file(corpus_path, seed_path)
                    log.info("Grapheme seed file: %d unique clusters", n_seeds)
                seed_file = str(seed_path)
            else:
                spaced_path = model_dir / "corpus_grapheme_spaced.txt"
                if not spaced_path.exists():
                    log.info("Building grapheme-spaced corpus for %s", name)
                    build_grapheme_spaced_corpus(corpus_path, spaced_path)
                train_input = spaced_path
                sp_kwargs_extra = dict(
                    split_by_whitespace=True,
                    split_by_unicode_script=False,
                )

        sp_model_type = "unigram" if model == ModelType.UNIGRAM else "bpe"
        sp_kwargs = dict(
            input=str(train_input),
            model_prefix=model_prefix,
            model_type=sp_model_type,
            vocab_size=cfg.training.vocab_size,
            character_coverage=cfg.training.character_coverage,
            max_sentence_length=cfg.training.max_sentence_length,
            input_sentence_size=cfg.training.input_sentence_size,
            shuffle_input_sentence=True,
            num_threads=cfg.training.num_threads,
            train_extremely_large_corpus=True,
            **sp_kwargs_extra,
        )

        if seed_file:
            sp_kwargs["seed_sentencepieces_file"] = seed_file
            sp_kwargs["seed_sentencepiece_size"] = cfg.training.vocab_size

        if init == InitUnit.BYTE:
            sp_kwargs["byte_fallback"] = True

        log.info("SentencePiece training: %s", {k: v for k, v in sp_kwargs.items() if k != "input"})
        effective_vocab = _train_sentencepiece(sp_kwargs, model_prefix, log)

        checkpoint.mark_shard_done(STEP, shard_key)
        results[name] = {
            "status": "done",
            "model_prefix": model_prefix,
            "seed_file": seed_file,
            "effective_vocab_size": effective_vocab,
        }
        log.info("Variant %s training complete (vocab=%d)", name, effective_vocab)

    return results
