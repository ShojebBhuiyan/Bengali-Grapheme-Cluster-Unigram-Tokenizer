"""Train 2x2 ablation tokenizer variants via SentencePiece."""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path

import sentencepiece as spm

from tokenizer_bn.checkpoint import CheckpointManager
from tokenizer_bn.config import Config, ensure_dirs, load_config
from tokenizer_bn.logging_utils import get_logger
from tokenizer_bn.segmentation.remap import ensure_akshara_map, prepare_grapheme_corpus, remap_text
from tokenizer_bn.train.sentence_sample import TrainingMode, resolve_training_plan
from tokenizer_bn.train.sharded_training import train_sharded_sentencepiece

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


def _assert_vocab_ok(name: str, effective: int, target: int, min_ratio: float, log) -> None:
    """Fail loudly if a variant collapsed far below its target vocabulary."""
    if min_ratio <= 0:
        return
    floor = int(target * min_ratio)
    if effective < floor:
        raise RuntimeError(
            f"Variant '{name}' trained to vocab_size={effective}, below the required "
            f"floor {floor} ({min_ratio:.0%} of target {target}). The corpus likely "
            f"cannot support this vocabulary, or the initialization collapsed. "
            f"Lower training.vocab_size, provide more data, or adjust "
            f"training.min_vocab_ratio."
        )
    if effective < target:
        log.warning(
            "Variant %s reached vocab=%d (< target %d) but is within the %.0f%% floor",
            name,
            effective,
            target,
            min_ratio * 100,
        )


def train_all_variants(config: Config | None = None, ckpt: CheckpointManager | None = None) -> dict:
    """Train all four ablation variants."""
    cfg = config or load_config()
    ensure_dirs(cfg)
    log = get_logger(STEP, cfg)
    checkpoint = ckpt or CheckpointManager(cfg)

    corpus_path = cfg.corpus_path
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    plan = resolve_training_plan(
        cfg.training.input_sentence_size,
        cfg.training.unlimited_sentence_soft_cap,
        cfg.training.shard_sentences,
        corpus_path,
    )
    log.info("Training plan: %s", plan.log_message)

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
        line_transform = None
        char_coverage = cfg.training.character_coverage

        if init == InitUnit.GRAPHEME:
            log.info("Building akshara map for %s", name)
            mapping, n_symbols = ensure_akshara_map(corpus_path, cfg.grapheme_map_path)
            log.info("Grapheme akshara alphabet: %d symbols", n_symbols)
            line_transform = lambda text, m=mapping: remap_text(text, m)
            char_coverage = 1.0

            if plan.mode == TrainingMode.SINGLE:
                log.info("Preparing subsampled remapped training corpus for %s", name)
                prepare_grapheme_corpus(
                    corpus_path,
                    cfg.grapheme_corpus_path,
                    cfg.grapheme_map_path,
                    cfg.grapheme_corpus_meta_path,
                    max_training_lines=plan.remapped_limit,
                    seed=cfg.training.seed,
                )
                train_input = cfg.grapheme_corpus_path
            else:
                train_input = corpus_path

        sp_model_type = "unigram" if model == ModelType.UNIGRAM else "bpe"
        sp_kwargs = dict(
            input=str(train_input),
            model_prefix=model_prefix,
            model_type=sp_model_type,
            vocab_size=cfg.training.vocab_size,
            character_coverage=char_coverage,
            max_sentence_length=cfg.training.max_sentence_length,
            shuffle_input_sentence=True,
            num_threads=cfg.training.num_threads,
            train_extremely_large_corpus=True,
        )

        if plan.mode == TrainingMode.SINGLE and plan.sp_size is not None and plan.sp_size > 0:
            sp_kwargs["input_sentence_size"] = plan.sp_size

        if init == InitUnit.BYTE:
            sp_kwargs["byte_fallback"] = True

        log.info("SentencePiece training: %s", {k: v for k, v in sp_kwargs.items() if k != "input"})

        if plan.mode == TrainingMode.SHARDED:
            effective_vocab = train_sharded_sentencepiece(
                sp_kwargs,
                model_prefix,
                train_input,
                plan.num_shards,
                _train_sentencepiece,
                log,
                checkpoint=checkpoint,
                step=STEP,
                variant_key=shard_key,
                line_transform=line_transform,
            )
        else:
            effective_vocab = _train_sentencepiece(sp_kwargs, model_prefix, log)

        _assert_vocab_ok(name, effective_vocab, cfg.training.vocab_size, cfg.training.min_vocab_ratio, log)

        checkpoint.mark_shard_done(STEP, shard_key)
        results[name] = {
            "status": "done",
            "model_prefix": model_prefix,
            "effective_vocab_size": effective_vocab,
            "training_mode": plan.mode.value,
            "num_shards": plan.num_shards if plan.mode == TrainingMode.SHARDED else 1,
        }
        log.info("Variant %s training complete (vocab=%d)", name, effective_vocab)

    return results
