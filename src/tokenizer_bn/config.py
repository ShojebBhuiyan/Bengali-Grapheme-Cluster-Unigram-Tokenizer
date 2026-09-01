"""Configuration loading for the tokenizer pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


@dataclass
class PathsConfig:
    datasets_dir: Path
    processed_dir: Path
    models_dir: Path
    logs_dir: Path
    checkpoints_dir: Path
    results_dir: Path


@dataclass
class CorpusConfig:
    corpus_sample_bytes: int = 314_572_800
    min_bangla_ratio: float = 0.5
    dedup: bool = True
    normalize_nfc: bool = True
    shard_size_lines: int = 50_000


@dataclass
class TrainingConfig:
    vocab_size: int = 8_000
    max_sentence_length: int = 4192
    character_coverage: float = 0.9995
    input_sentence_size: int = 300_000  # 0 (or negative) = use every sentence (unlimited)
    seed: int = 42
    num_threads: int = 4
    # When input_sentence_size is unlimited (<=0) and the corpus exceeds this
    # line count, subsample to this many sentences to avoid SentencePiece OOM.
    # Set 0 to disable the cap (may OOM on multi-GB corpora).
    unlimited_sentence_soft_cap: int = 5_000_000
    # When input_sentence_size is unlimited (<=0) and the corpus exceeds this
    # line count, train sequentially on round-robin shards of this size, carrying
    # vocabulary forward between shards (covers the full corpus without OOM).
    # Set 0 to disable sharded training (falls back to unlimited_sentence_soft_cap).
    shard_sentences: int = 5_000_000
    # Fail training if a variant's vocab collapses below this fraction of the
    # target vocab_size (set 0 to disable the guard).
    min_vocab_ratio: float = 0.9


@dataclass
class EvaluationConfig:
    eval_sample_lines: int = 5000
    strr_word_list_size: int = 1000
    corpus_eval_max_lines: int = 0  # 0 = evaluate full processed corpus
    corpus_eval_batch_size: int = 512


@dataclass
class EDAConfig:
    sample_lines: int = 50_000
    top_n_aksharas: int = 30


@dataclass
class DeviceConfig:
    use_gpu: bool = True
    device: str = "auto"  # auto, cpu, cuda, mps
    batch_size: int = 256


@dataclass
class Config:
    paths: PathsConfig
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    eda: EDAConfig = field(default_factory=EDAConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    project_root: Path = PROJECT_ROOT

    @property
    def corpus_path(self) -> Path:
        return self.paths.processed_dir / "corpus_bn.txt"

    @property
    def parallel_path(self) -> Path:
        return self.paths.processed_dir / "parallel_bn_en.parquet"

    @property
    def manifest_path(self) -> Path:
        return self.paths.processed_dir / "manifest.json"

    @property
    def grapheme_map_path(self) -> Path:
        """Shared akshara->PUA mapping used by the grapheme variants."""
        return self.paths.processed_dir / "akshara_map.json"

    @property
    def grapheme_corpus_path(self) -> Path:
        """Subsampled akshara-remapped corpus used to train grapheme variants."""
        return self.paths.processed_dir / "corpus_grapheme_remapped_train.txt"

    @property
    def grapheme_corpus_meta_path(self) -> Path:
        """Metadata for the grapheme training corpus (subsample size, seed)."""
        return self.paths.processed_dir / "corpus_grapheme_remapped_train.meta.json"


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path


def load_config(config_path: str | Path | None = None) -> Config:
    """Load configuration from YAML, resolving relative paths against project root."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path, encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    paths_raw = raw.get("paths", {})
    paths = PathsConfig(
        datasets_dir=_resolve_path(PROJECT_ROOT, paths_raw.get("datasets_dir", "datasets")),
        processed_dir=_resolve_path(PROJECT_ROOT, paths_raw.get("processed_dir", "data/processed")),
        models_dir=_resolve_path(PROJECT_ROOT, paths_raw.get("models_dir", "models")),
        logs_dir=_resolve_path(PROJECT_ROOT, paths_raw.get("logs_dir", "logs")),
        checkpoints_dir=_resolve_path(PROJECT_ROOT, paths_raw.get("checkpoints_dir", "checkpoints")),
        results_dir=_resolve_path(PROJECT_ROOT, paths_raw.get("results_dir", "results")),
    )

    corpus = CorpusConfig(**raw.get("corpus", {}))
    training = TrainingConfig(**raw.get("training", {}))
    evaluation = EvaluationConfig(**raw.get("evaluation", {}))
    eda = EDAConfig(**raw.get("eda", {}))
    device = DeviceConfig(**raw.get("device", {}))

    return Config(
        paths=paths,
        corpus=corpus,
        training=training,
        evaluation=evaluation,
        eda=eda,
        device=device,
    )


def ensure_dirs(config: Config) -> None:
    """Create all output directories if they do not exist."""
    for directory in (
        config.paths.processed_dir,
        config.paths.models_dir,
        config.paths.logs_dir,
        config.paths.checkpoints_dir,
        config.paths.results_dir,
        config.paths.results_dir / "eda",
        config.paths.results_dir / "figures",
        config.paths.results_dir / "tables",
    ):
        directory.mkdir(parents=True, exist_ok=True)
