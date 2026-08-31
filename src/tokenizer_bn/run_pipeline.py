"""CLI entry point for the tokenizer research pipeline."""

from __future__ import annotations

import argparse
import sys

from tokenizer_bn.checkpoint import CheckpointManager
from tokenizer_bn.config import ensure_dirs, load_config
from tokenizer_bn.logging_utils import get_logger


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bengali grapheme-cluster unigram tokenizer pipeline",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (default: configs/default.yaml)",
    )
    parser.add_argument(
        "command",
        choices=["build-corpus", "eda", "train", "evaluate", "all"],
        help="Pipeline step to run",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run step even if checkpoint marks it done",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)
    ensure_dirs(config)
    logger = get_logger("pipeline", config)
    ckpt = CheckpointManager(config)

    command = args.command
    logger.info("Starting command: %s", command)

    from tokenizer_bn.device import device_info, log_device_info
    log_device_info(logger, config.device.device)
    if command in ("evaluate", "eda", "all"):
        info = device_info(config.device.device)
        if config.device.use_gpu and info["resolved_device"] == "cpu" and info["torch_available"]:
            logger.info("GPU requested but only CPU backend available for this step")
        elif not info["torch_available"] and config.device.use_gpu:
            logger.info("GPU metrics disabled: install PyTorch with `pip install torch`")

    if command in ("build-corpus", "all"):
        if args.force:
            ckpt.reset_step("build-corpus")
        if not ckpt.is_step_done("build-corpus"):
            from tokenizer_bn.data.build_corpus import build_corpus

            ckpt.mark_step_started("build-corpus")
            build_corpus(config, ckpt)
            ckpt.mark_step_done("build-corpus")
        else:
            logger.info("Skipping build-corpus (already done)")

    if command in ("eda", "all"):
        if args.force:
            ckpt.reset_step("eda")
        if not ckpt.is_step_done("eda"):
            from tokenizer_bn.eda.analyze import run_eda

            ckpt.mark_step_started("eda")
            run_eda(config, ckpt)
            ckpt.mark_step_done("eda")
        else:
            logger.info("Skipping eda (already done)")

    if command in ("train", "all"):
        if args.force:
            ckpt.reset_step("train")
        if not ckpt.is_step_done("train"):
            from tokenizer_bn.train.train_variants import train_all_variants

            ckpt.mark_step_started("train")
            train_all_variants(config, ckpt)
            ckpt.mark_step_done("train")
        else:
            logger.info("Skipping train (already done)")

    if command in ("evaluate", "all"):
        if args.force:
            ckpt.reset_step("evaluate")
        if not ckpt.is_step_done("evaluate"):
            from tokenizer_bn.eval.harness import run_evaluation

            ckpt.mark_step_started("evaluate")
            run_evaluation(config, ckpt)
            ckpt.mark_step_done("evaluate")
        else:
            logger.info("Skipping evaluate (already done)")

    logger.info("Command finished: %s", command)
    return 0


if __name__ == "__main__":
    sys.exit(main())
