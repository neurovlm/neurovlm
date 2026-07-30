"""Standard run-directory and lifecycle management."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any

from .config import RunConfig
from .provenance import (
    environment_provenance,
    fingerprint_references,
    git_provenance,
    sha256_value,
)
from .serialization import atomic_write_json


ARTIFACT_MANIFEST_VERSION = 1
RUN_DIRECTORIES = (
    "config",
    "provenance",
    "checkpoints",
    "metrics",
    "plots",
    "generated_maps",
    "logs",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class RunArtifacts:
    """Paths and atomic writers for one standardized run directory."""

    def __init__(self, config: RunConfig):
        self.config = config
        self.root = config.run_dir

    def path(self, relative: str | Path) -> Path:
        candidate = self.root / relative
        try:
            candidate.resolve().relative_to(self.root.resolve())
        except ValueError as error:
            raise ValueError(f"Artifact path must stay inside the run directory: {relative}") from error
        return candidate

    def relative(self, path: str | Path) -> str:
        """Return a portable POSIX path relative to this run."""

        path = Path(path)
        if not path.is_absolute():
            path = self.root / path
        try:
            return path.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError as error:
            raise ValueError(f"Artifact path is outside run directory: {path}") from error

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def status_path(self) -> Path:
        return self.root / "status.json"

    def initialize(self, *, repo: str | Path = ".") -> "RunArtifacts":
        """Create the golden tree and write config/provenance atomically."""

        self.root.mkdir(parents=True, exist_ok=True)
        for directory in RUN_DIRECTORIES:
            self.path(directory).mkdir(parents=True, exist_ok=True)
        expected_manifest = self._manifest()
        if self.manifest_path.exists():
            existing = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if (
                existing.get("run_id") != self.config.run_id
                or existing.get("model_spec") != expected_manifest["model_spec"]
                or existing.get("config_sha256") != expected_manifest["config_sha256"]
            ):
                raise ValueError(
                    "Existing run manifest is incompatible with the requested run configuration"
                )
        atomic_write_json(self.path("config/requested.json"), self.config.requested_dict())
        atomic_write_json(self.path("config/effective.json"), self.config.effective_dict())
        atomic_write_json(self.path("provenance/environment.json"), environment_provenance())
        atomic_write_json(self.path("provenance/git.json"), git_provenance(repo))
        atomic_write_json(
            self.path("provenance/data.json"), fingerprint_references(self.config.data)
        )
        atomic_write_json(
            self.path("provenance/resources.json"),
            fingerprint_references(self.config.resources),
        )
        atomic_write_json(
            self.path("provenance/initialization.json"),
            fingerprint_references(self.config.initialization),
        )
        if not self.manifest_path.exists():
            atomic_write_json(self.manifest_path, expected_manifest)
        return self

    def _manifest(self) -> dict[str, Any]:
        paths = {
            "status": "status.json",
            "requested_config": "config/requested.json",
            "effective_config": "config/effective.json",
            "environment": "provenance/environment.json",
            "git": "provenance/git.json",
            "data": "provenance/data.json",
            "resources": "provenance/resources.json",
            "initialization": "provenance/initialization.json",
            "checkpoint_manifest": "checkpoints/checkpoint_manifest.json",
            "metric_history": "metrics/history.csv",
            "metric_summary": "metrics/summary.csv",
            "metric_curves": "metrics/curves.csv",
        }
        return {
            "version": ARTIFACT_MANIFEST_VERSION,
            "run_id": self.config.run_id,
            "model_spec": self.config.model_spec_dict(),
            "config_sha256": sha256_value(self.config.effective_dict()),
            "primary_metric": self.config.primary_metric,
            "metric_direction": self.config.metric_direction.value,
            "artifacts": paths,
            "directories": {name: name for name in RUN_DIRECTORIES},
        }

    def read_status(self) -> dict[str, Any] | None:
        if not self.status_path.exists():
            return None
        return json.loads(self.status_path.read_text(encoding="utf-8"))

    def set_status(self, state: str, **details: Any) -> dict[str, Any]:
        """Atomically update lifecycle state while retaining original start."""

        if state not in {"running", "completed", "failed"}:
            raise ValueError("state must be running, completed, or failed")
        previous = self.read_status() or {}
        now = _utc_now()
        status = {
            "version": 1,
            "run_id": self.config.run_id,
            "state": state,
            "started_at": previous.get("started_at", now),
            "updated_at": now,
            "resume_count": int(previous.get("resume_count", -1)) + 1
            if state == "running"
            else int(previous.get("resume_count", 0)),
        }
        if state in {"completed", "failed"}:
            status["finished_at"] = now
        status.update(details)
        atomic_write_json(self.status_path, status)
        return status


class RunContext:
    """Context manager that guarantees running/completed/failed status."""

    def __init__(self, config: RunConfig, *, repo: str | Path = "."):
        self.artifacts = RunArtifacts(config)
        self.repo = repo

    def __enter__(self) -> RunArtifacts:
        self.artifacts.initialize(repo=self.repo)
        self.artifacts.set_status("running")
        return self.artifacts

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if exc_value is None:
            self.artifacts.set_status("completed")
        else:
            self.artifacts.set_status(
                "failed",
                error={"type": exc_type.__name__ if exc_type else None, "message": str(exc_value)},
            )
        return False


__all__ = ["ARTIFACT_MANIFEST_VERSION", "RUN_DIRECTORIES", "RunArtifacts", "RunContext"]
