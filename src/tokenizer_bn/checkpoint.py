"""Checkpoint management for resumable pipeline steps."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tokenizer_bn.config import Config, ensure_dirs, load_config


class CheckpointManager:
    """Track pipeline step completion and per-shard progress."""

    def __init__(self, config: Config | None = None, state_file: str = "pipeline_state.json"):
        self.config = config or load_config()
        ensure_dirs(self.config)
        self.state_path = self.config.paths.checkpoints_dir / state_file
        self._state = self._load()

    def _load(self) -> dict[str, Any]:
        if self.state_path.exists():
            with open(self.state_path, encoding="utf-8") as fh:
                return json.load(fh)
        return {"steps": {}, "shards": {}}

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as fh:
            json.dump(self._state, fh, indent=2, ensure_ascii=False)

    def is_step_done(self, step: str) -> bool:
        return self._state.get("steps", {}).get(step, {}).get("status") == "done"

    def mark_step_done(self, step: str, metadata: dict[str, Any] | None = None) -> None:
        self._state.setdefault("steps", {})[step] = {
            "status": "done",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }
        self._save()

    def mark_step_started(self, step: str) -> None:
        self._state.setdefault("steps", {})[step] = {
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save()

    def is_shard_done(self, step: str, shard_id: str) -> bool:
        return self._state.get("shards", {}).get(step, {}).get(shard_id) == "done"

    def mark_shard_done(self, step: str, shard_id: str) -> None:
        self._state.setdefault("shards", {}).setdefault(step, {})[shard_id] = "done"
        self._save()

    def reset_step(self, step: str) -> None:
        self._state.get("steps", {}).pop(step, None)
        self._state.get("shards", {}).pop(step, None)
        self._save()

    def get_step_metadata(self, step: str) -> dict[str, Any]:
        return self._state.get("steps", {}).get(step, {}).get("metadata", {})
