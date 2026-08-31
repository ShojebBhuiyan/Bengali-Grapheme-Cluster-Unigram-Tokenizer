"""Train 2x2 ablation tokenizer variants via SentencePiece."""

from __future__ import annotations

import shutil
from enum import Enum
from pathlib import Path

import sentencepiece as spm

from tokenizer_bn.checkpoint import CheckpointManager
from tokenizer_bn.config import Config, ensure_dirs, load_config
from tokenizer_bn.logging_utils import get_logger
from tokenizer_bn.segmentation.remap import remap_corpus

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
        vocab_map_path = None

        if init == InitUnit.GRAPHEME:
            remapped_path = model_dir / "corpus_remapped.txt"
            vocab_map_path = model_dir / "grapheme_vocab.json"
            if not remapped_path.exists():
                log.info("Remapping corpus to grapheme synthetic IDs for %s", name)
                remap_corpus(corpus_path, remapped_path, vocab_map_path)
            train_input = remapped_path

        sp_model_type = "unigram" if model == ModelType.UNIGRAM else "bpe"
        sp_kwargs = dict(
            input=str(train_input),
            model_prefix=model_prefix,
            model_type=sp_model_type,
            vocab_size=cfg.training.vocab_size,
            character_coverage=cfg.training.character_coverage,
            max_sentence_length=cfg.training.max_sentence_length,
            seed_sentencepiece_size=cfg.training.vocab_size,
            input_sentence_size=cfg.training.input_sentence_size,
            shuffle_input_sentence=True,
            num_threads=cfg.training.num_threads,
            train_extremely_large_corpus=True,
        )

        if init == InitUnit.BYTE:
            sp_kwargs["byte_fallback"] = True

        log.info("SentencePiece training: %s", sp_kwargs)
        spm.SentencePieceTrainer.Train(**sp_kwargs)

        checkpoint.mark_shard_done(STEP, shard_key)
        results[name] = {
            "status": "done",
            "model_prefix": model_prefix,
            "vocab_map": str(vocab_map_path) if vocab_map_path else None,
        }
        log.info("Variant %s training complete", name)

    return results
