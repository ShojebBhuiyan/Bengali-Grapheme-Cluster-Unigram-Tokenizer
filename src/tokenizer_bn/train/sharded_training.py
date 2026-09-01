"""Sequential sharded SentencePiece training with vocabulary carry-over."""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path

import sentencepiece as spm

from tokenizer_bn.checkpoint import CheckpointManager
from tokenizer_bn.data.ingest import stream_txt_lines

LineTransform = Callable[[str], str]


def write_round_robin_shard(
    corpus_path: Path,
    output_path: Path,
    shard_idx: int,
    num_shards: int,
    line_transform: LineTransform | None = None,
) -> int:
    """Write one round-robin shard of ``corpus_path`` to ``output_path``.

    Line *i* goes to shard ``i % num_shards``, which spreads content across the
    corpus without materialising all shards on disk at once.
    """
    count = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        for line_idx, line in enumerate(stream_txt_lines(corpus_path)):
            if line_idx % num_shards != shard_idx:
                continue
            out = line_transform(line) if line_transform else line
            fout.write(out + "\n")
            count += 1
    return count


def train_sharded_sentencepiece(
    sp_kwargs_base: dict,
    model_prefix: str,
    corpus_path: Path,
    num_shards: int,
    train_fn,
    log,
    checkpoint: CheckpointManager | None = None,
    step: str = "train",
    variant_key: str = "",
    line_transform: LineTransform | None = None,
) -> int:
    """Train SentencePiece sequentially on round-robin shards.

    Each shard after the first is initialised from the previous shard's
    ``.vocab`` file (``seed_sentencepieces_file``), so piece frequencies and
    merges accumulate across the full corpus without loading it all at once.

    ``train_fn`` is the project's adaptive trainer (``_train_sentencepiece``).
    """
    if num_shards <= 1:
        raise ValueError("train_sharded_sentencepiece requires num_shards > 1")

    effective_vocab = 0
    seed_vocab: Path | None = None
    model_dir = Path(model_prefix).parent

    for shard_idx in range(num_shards):
        shard_ckpt = f"{variant_key}_shard_{shard_idx}"
        if checkpoint and checkpoint.is_shard_done(step, shard_ckpt):
            log.info("Skipping shard %d/%d (checkpoint done)", shard_idx + 1, num_shards)
            seed_vocab = Path(f"{model_prefix}.vocab")
            if seed_vocab.exists():
                proc = spm.SentencePieceProcessor()
                proc.Load(f"{model_prefix}.model")
                effective_vocab = proc.GetPieceSize()
            continue

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".txt",
            prefix=f"shard_{shard_idx}_",
            dir=model_dir,
            delete=False,
        ) as tmp:
            shard_path = Path(tmp.name)

        try:
            n_lines = write_round_robin_shard(
                corpus_path,
                shard_path,
                shard_idx,
                num_shards,
                line_transform=line_transform,
            )
            if n_lines == 0:
                log.warning("Shard %d/%d is empty, skipping", shard_idx + 1, num_shards)
                if checkpoint:
                    checkpoint.mark_shard_done(step, shard_ckpt)
                continue

            log.info(
                "Shard %d/%d: training on %s lines (seeding from %s)",
                shard_idx + 1,
                num_shards,
                f"{n_lines:,}",
                seed_vocab.name if seed_vocab else "scratch",
            )

            sp_kwargs = dict(sp_kwargs_base)
            sp_kwargs["input"] = str(shard_path)
            # Each shard file already fits in memory; train on every line in it.
            sp_kwargs.pop("input_sentence_size", None)

            seed_copy_path: Path | None = None
            if seed_vocab is not None and seed_vocab.exists():
                # Copy the seed vocab so _train_sentencepiece's model-prefix cleanup
                # does not delete it before SentencePiece reads it.
                seed_copy_path = model_dir / f".seed_shard_{shard_idx}.txt"
                seed_copy_path.write_text(seed_vocab.read_text(encoding="utf-8"), encoding="utf-8")
                sp_kwargs["seed_sentencepieces_file"] = str(seed_copy_path)
                sp_kwargs["seed_sentencepiece_size"] = sp_kwargs["vocab_size"]

            try:
                effective_vocab = train_fn(sp_kwargs, model_prefix, log)
            finally:
                if seed_copy_path is not None:
                    seed_copy_path.unlink(missing_ok=True)

            seed_vocab = Path(f"{model_prefix}.vocab")

            if checkpoint:
                checkpoint.mark_shard_done(step, shard_ckpt)
        finally:
            if shard_path.exists():
                shard_path.unlink()

    return effective_vocab
