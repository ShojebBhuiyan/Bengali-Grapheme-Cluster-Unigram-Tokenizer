"""Build the processed Bangla-only corpus and parallel evaluation set."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from tokenizer_bn.checkpoint import CheckpointManager
from tokenizer_bn.config import Config, ensure_dirs, load_config
from tokenizer_bn.data.bangla_filter import filter_bangla_line
from tokenizer_bn.data.ingest import (
    estimate_file_bytes,
    stream_csv_bangla,
    stream_parquet_pairs,
    stream_tatoeba_tsv,
    stream_txt_lines,
)
from tokenizer_bn.logging_utils import get_logger

STEP = "build-corpus"


@dataclass
class SourceStats:
    name: str
    lines_read: int = 0
    lines_kept: int = 0
    bytes_written: int = 0
    parallel_pairs: int = 0


@dataclass
class CorpusBuilder:
    config: Config
    ckpt: CheckpointManager
    logger: object = field(default=None, repr=False)
    seen_hashes: set[str] = field(default_factory=set)
    stats: dict[str, SourceStats] = field(default_factory=dict)
    parallel_rows: list[dict] = field(default_factory=list)
    total_bytes: int = 0
    budget_bytes: int = 0

    def __post_init__(self) -> None:
        if self.logger is None:
            self.logger = get_logger(STEP, self.config)
        self.budget_bytes = self.config.corpus.corpus_sample_bytes
        ensure_dirs(self.config)

    def _dedup_key(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _should_keep(self, text: str) -> bool:
        if not self.config.corpus.dedup:
            return True
        key = self._dedup_key(text)
        if key in self.seen_hashes:
            return False
        self.seen_hashes.add(key)
        return True

    def _budget_remaining(self) -> int:
        return max(0, self.budget_bytes - self.total_bytes)

    def _write_line(self, fh, text: str, source: str) -> bool:
        if self._budget_remaining() <= 0:
            return False
        line_bytes = len(text.encode("utf-8")) + 1
        if line_bytes > self._budget_remaining():
            return False
        fh.write(text + "\n")
        self.total_bytes += line_bytes
        self.stats[source].lines_kept += 1
        self.stats[source].bytes_written += line_bytes
        return True

    def _process_source(self, source_name: str, corpus_fh) -> None:
        if self.ckpt.is_shard_done(STEP, source_name):
            self.logger.info("Skipping source %s (checkpoint done)", source_name)
            return

        self.stats[source_name] = SourceStats(name=source_name)
        self.logger.info("Processing source: %s", source_name)
        ds = self.config.paths.datasets_dir

        if source_name == "bangla/bn.txt":
            self._process_bn_txt(ds / "bangla" / "bn.txt", corpus_fh, source_name)
        elif source_name == "bangla-english/ben.txt":
            self._process_ben_tsv(ds / "bangla-english" / "ben.txt", corpus_fh, source_name)
        elif source_name.startswith("syncorpus/"):
            parquet_path = ds / source_name
            self._process_parquet(parquet_path, corpus_fh, source_name)
        elif source_name == "transliterated/bengali_transliterated.csv":
            self._process_transliterated(ds / "transliterated" / "bengali_transliterated.csv", corpus_fh, source_name)
        else:
            self.logger.warning("Unknown source: %s", source_name)

        self.ckpt.mark_shard_done(STEP, source_name)
        self.logger.info(
            "Finished %s: read=%d kept=%d bytes=%d parallel=%d",
            source_name,
            self.stats[source_name].lines_read,
            self.stats[source_name].lines_kept,
            self.stats[source_name].bytes_written,
            self.stats[source_name].parallel_pairs,
        )

    def _process_bn_txt(self, path: Path, corpus_fh, source: str) -> None:
        if not path.exists():
            self.logger.warning("File not found: %s", path)
            return
        for line in tqdm(stream_txt_lines(path), desc=source, unit="lines"):
            self.stats[source].lines_read += 1
            if self._budget_remaining() <= 0:
                break
            filtered = filter_bangla_line(
                line,
                min_ratio=self.config.corpus.min_bangla_ratio,
                normalize_nfc=self.config.corpus.normalize_nfc,
            )
            if filtered and self._should_keep(filtered):
                self._write_line(corpus_fh, filtered, source)

    def _process_ben_tsv(self, path: Path, corpus_fh, source: str) -> None:
        if not path.exists():
            return
        for en_text, bn_text in tqdm(stream_tatoeba_tsv(path), desc=source, unit="pairs"):
            self.stats[source].lines_read += 1
            filtered = filter_bangla_line(
                bn_text,
                min_ratio=self.config.corpus.min_bangla_ratio,
                normalize_nfc=self.config.corpus.normalize_nfc,
            )
            if filtered and self._should_keep(filtered):
                self._write_line(corpus_fh, filtered, source)
            if en_text and filtered:
                self.parallel_rows.append({"en": en_text, "bn": filtered, "source": source})
                self.stats[source].parallel_pairs += 1

    def _process_parquet(self, path: Path, corpus_fh, source: str) -> None:
        if not path.exists():
            return
        for bn_text, en_text, _ in tqdm(stream_parquet_pairs(path), desc=source, unit="rows"):
            self.stats[source].lines_read += 1
            if self._budget_remaining() <= 0:
                break
            filtered = filter_bangla_line(
                bn_text,
                min_ratio=self.config.corpus.min_bangla_ratio,
                normalize_nfc=self.config.corpus.normalize_nfc,
            )
            if filtered and self._should_keep(filtered):
                self._write_line(corpus_fh, filtered, source)
            if en_text and filtered:
                self.parallel_rows.append({"en": en_text, "bn": filtered, "source": source})
                self.stats[source].parallel_pairs += 1

    def _process_transliterated(self, path: Path, corpus_fh, source: str) -> None:
        if not path.exists():
            return
        for text in tqdm(stream_csv_bangla(path), desc=source, unit="rows"):
            self.stats[source].lines_read += 1
            if self._budget_remaining() <= 0:
                break
            filtered = filter_bangla_line(
                text,
                min_ratio=self.config.corpus.min_bangla_ratio,
                normalize_nfc=self.config.corpus.normalize_nfc,
            )
            if filtered and self._should_keep(filtered):
                self._write_line(corpus_fh, filtered, source)

    def build(self) -> dict:
        """Run corpus build and return manifest dict."""
        sources = self._discover_sources()
        self.logger.info("Discovered %d sources, budget=%d bytes", len(sources), self.budget_bytes)

        corpus_path = self.config.corpus_path
        corpus_path.parent.mkdir(parents=True, exist_ok=True)

        with open(corpus_path, "w", encoding="utf-8") as corpus_fh:
            for source in sources:
                self._process_source(source, corpus_fh)

        # Write parallel eval set
        parallel_path = self.config.parallel_path
        if self.parallel_rows:
            pd.DataFrame(self.parallel_rows).to_parquet(parallel_path, index=False)
            self.logger.info("Wrote %d parallel pairs to %s", len(self.parallel_rows), parallel_path)

        manifest = self._build_manifest(sources)
        with open(self.config.manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False)

        self.logger.info("Corpus build complete: %d bytes, manifest at %s", self.total_bytes, self.config.manifest_path)
        return manifest

    def _discover_sources(self) -> list[str]:
        ds = self.config.paths.datasets_dir
        sources: list[str] = []

        # Small sources first (fully included up to budget)
        ben = ds / "bangla-english" / "ben.txt"
        if ben.exists():
            sources.append("bangla-english/ben.txt")

        syncorpus_dir = ds / "syncorpus"
        if syncorpus_dir.exists():
            for p in sorted(syncorpus_dir.glob("*.parquet")):
                sources.append(f"syncorpus/{p.name}")

        translit = ds / "transliterated" / "bengali_transliterated.csv"
        if translit.exists():
            sources.append("transliterated/bengali_transliterated.csv")

        # Large bn.txt last (fills remaining budget)
        bn = ds / "bangla" / "bn.txt"
        if bn.exists():
            sources.append("bangla/bn.txt")

        return sources

    def _build_manifest(self, sources: list[str]) -> dict:
        return {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "corpus_sample_bytes_budget": self.budget_bytes,
            "corpus_bytes_written": self.total_bytes,
            "corpus_path": str(self.config.corpus_path),
            "parallel_path": str(self.config.parallel_path),
            "parallel_pairs": len(self.parallel_rows),
            "dedup": self.config.corpus.dedup,
            "min_bangla_ratio": self.config.corpus.min_bangla_ratio,
            "sources": {
                name: {
                    "lines_read": s.lines_read,
                    "lines_kept": s.lines_kept,
                    "bytes_written": s.bytes_written,
                    "parallel_pairs": s.parallel_pairs,
                    "file_bytes": estimate_file_bytes(self.config.paths.datasets_dir / name),
                }
                for name, s in self.stats.items()
            },
            "source_order": sources,
        }


def build_corpus(config: Config | None = None, ckpt: CheckpointManager | None = None) -> dict:
    """Entry point: build processed Bangla corpus and parallel eval set."""
    cfg = config or load_config()
    checkpoint = ckpt or CheckpointManager(cfg)
    builder = CorpusBuilder(config=cfg, ckpt=checkpoint)
    return builder.build()
