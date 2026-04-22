from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class ArtifactStore:
    _REPLACE_RETRY_ATTEMPTS = 12
    _REPLACE_RETRY_BASE_DELAY = 0.05

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def create_run(self, config: dict[str, Any]) -> tuple[str, Path]:
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S%fZ")
        run_id = f"run-{timestamp}"
        run_dir = self.root / run_id
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        self.write_json(
            run_dir / "summary.json",
            {
                "runId": run_id,
                "status": "running",
                "paused": False,
                "startedAt": datetime.now(tz=UTC).isoformat(),
                "updatedAt": datetime.now(tz=UTC).isoformat(),
                "currentEpisode": 0,
                "checkpoints": [],
                "config": config,
            },
        )
        self.write_json(run_dir / "metrics.json", {"runId": run_id, "series": []})
        self.write_json(run_dir / "checkpoints" / "index.json", {"runId": run_id, "checkpoints": []})
        return run_id, run_dir

    def latest_run_dir(self) -> Path | None:
        runs = sorted((path for path in self.root.iterdir() if path.is_dir() and path.name.startswith("run-")), reverse=True)
        return runs[0] if runs else None

    def latest_run_id(self) -> str | None:
        latest = self.latest_run_dir()
        return latest.name if latest else None

    def write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

        last_error: OSError | None = None
        try:
            for attempt in range(1, self._REPLACE_RETRY_ATTEMPTS + 1):
                try:
                    os.replace(temp_path, path)
                    last_error = None
                    break
                except OSError as exc:
                    if not self._is_retryable_replace_error(exc) or attempt == self._REPLACE_RETRY_ATTEMPTS:
                        raise
                    last_error = exc
                    time.sleep(self._REPLACE_RETRY_BASE_DELAY * attempt)
            if last_error is not None:
                raise last_error
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

    def read_json(self, path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))

    def read_run_summary(self, run_id: str) -> dict[str, Any]:
        return self.read_json(self.root / run_id / "summary.json")

    def update_run_summary(self, run_id: str, updater: dict[str, Any]) -> dict[str, Any]:
        summary_path = self.root / run_id / "summary.json"
        summary = self.read_json(summary_path)
        summary.update(updater)
        summary["updatedAt"] = datetime.now(tz=UTC).isoformat()
        self.write_json(summary_path, summary)
        return summary

    def write_metrics(self, run_id: str, series: list[dict[str, Any]]) -> None:
        self.write_json(self.root / run_id / "metrics.json", {"runId": run_id, "series": series})

    def write_checkpoint(self, run_id: str, checkpoint_id: str, checkpoint_payload: dict[str, Any]) -> None:
        checkpoint_dir = self.root / run_id / "checkpoints" / checkpoint_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.write_json(checkpoint_dir / "summary.json", checkpoint_payload["summary"])
        self.write_json(checkpoint_dir / "episode.json", checkpoint_payload["episode"])
        self.write_json(checkpoint_dir / "policy.json", checkpoint_payload["policy"])
        self.write_json(checkpoint_dir / "embedding.json", checkpoint_payload["embedding"])
        if "regime_analysis" in checkpoint_payload:
            self.write_json(checkpoint_dir / "regime_analysis.json", checkpoint_payload["regime_analysis"])
        if "expert_analysis" in checkpoint_payload:
            self.write_json(checkpoint_dir / "expert_analysis.json", checkpoint_payload["expert_analysis"])
        if "resume_state" in checkpoint_payload:
            self.write_json(checkpoint_dir / "resume_state.json", checkpoint_payload["resume_state"])
        if "stats" in checkpoint_payload:
            self.write_json(checkpoint_dir / "stats.json", checkpoint_payload["stats"])
        if "explainability" in checkpoint_payload:
            self.write_json(checkpoint_dir / "explainability.json", checkpoint_payload["explainability"])
        if "repro" in checkpoint_payload:
            self.write_json(checkpoint_dir / "repro.json", checkpoint_payload["repro"])
        if "data_fit" in checkpoint_payload:
            self.write_json(checkpoint_dir / "data_fit.json", checkpoint_payload["data_fit"])

    def write_checkpoint_index(self, run_id: str, summaries: list[dict[str, Any]]) -> None:
        self.write_json(self.root / run_id / "checkpoints" / "index.json", {"runId": run_id, "checkpoints": summaries})

    def checkpoint_episode(self, run_id: str, checkpoint_id: str) -> dict[str, Any]:
        return self.read_json(self.root / run_id / "checkpoints" / checkpoint_id / "episode.json")

    def checkpoint_policy(self, run_id: str, checkpoint_id: str) -> dict[str, Any]:
        return self.read_json(self.root / run_id / "checkpoints" / checkpoint_id / "policy.json")

    def checkpoint_embedding(self, run_id: str, checkpoint_id: str) -> dict[str, Any]:
        return self.read_json(self.root / run_id / "checkpoints" / checkpoint_id / "embedding.json")

    def checkpoint_regime_analysis(self, run_id: str, checkpoint_id: str) -> dict[str, Any] | None:
        path = self.root / run_id / "checkpoints" / checkpoint_id / "regime_analysis.json"
        if not path.exists():
            return None
        return self.read_json(path)

    def checkpoint_expert_analysis(self, run_id: str, checkpoint_id: str) -> dict[str, Any] | None:
        path = self.root / run_id / "checkpoints" / checkpoint_id / "expert_analysis.json"
        if not path.exists():
            return None
        return self.read_json(path)

    def checkpoint_resume_state(self, run_id: str, checkpoint_id: str) -> dict[str, Any] | None:
        path = self.root / run_id / "checkpoints" / checkpoint_id / "resume_state.json"
        if not path.exists():
            return None
        return self.read_json(path)

    def checkpoint_stats(self, run_id: str, checkpoint_id: str) -> dict[str, Any] | None:
        path = self.root / run_id / "checkpoints" / checkpoint_id / "stats.json"
        if not path.exists():
            return None
        return self.read_json(path)

    def checkpoint_explainability(self, run_id: str, checkpoint_id: str) -> dict[str, Any] | None:
        path = self.root / run_id / "checkpoints" / checkpoint_id / "explainability.json"
        if not path.exists():
            return None
        return self.read_json(path)

    def checkpoint_data_fit(self, run_id: str, checkpoint_id: str) -> dict[str, Any] | None:
        path = self.root / run_id / "checkpoints" / checkpoint_id / "data_fit.json"
        if not path.exists():
            return None
        return self.read_json(path)

    def model_weights_dir(self, run_id: str, checkpoint_id: str) -> Path:
        """Return the directory for model weight files within a checkpoint."""
        return self.root / run_id / "checkpoints" / checkpoint_id / "weights"

    def has_model_weights(self, run_id: str, checkpoint_id: str) -> bool:
        """Check whether model weights exist for a given checkpoint."""
        d = self.model_weights_dir(run_id, checkpoint_id)
        return (d / "online.pt").exists()

    def write_run_file(self, run_id: str, name: str, payload: Any) -> Path:
        path = self.root / run_id / name
        self.write_json(path, payload)
        return path

    def copy_checkpoint(self, source_run_id: str, checkpoint_id: str, target_run_id: str) -> None:
        source = self.root / source_run_id / "checkpoints" / checkpoint_id
        target = self.root / target_run_id / "checkpoints" / checkpoint_id
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)

    @staticmethod
    def _is_retryable_replace_error(exc: OSError) -> bool:
        return exc.winerror in {5, 32} if getattr(exc, "winerror", None) is not None else False
