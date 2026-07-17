"""Shared, task-agnostic training and inference pipeline infrastructure."""

from .artifacts import RunArtifacts, RunContext
from .checkpoints import CheckpointManager, ResumeState
from .config import MetricDirection, RunConfig
from .metrics import METRIC_COLUMNS, MetricRecorder, curve_rows, metric_row, summary_rows
from .provenance import (
    environment_provenance,
    fingerprint_path,
    fingerprint_references,
    git_provenance,
    sha256_file,
    sha256_value,
)
from .serialization import (
    append_csv_union,
    atomic_write_csv,
    atomic_write_json,
    json_safe,
    union_fieldnames,
)

__all__ = [
    "METRIC_COLUMNS",
    "CheckpointManager",
    "MetricDirection",
    "MetricRecorder",
    "ResumeState",
    "RunArtifacts",
    "RunConfig",
    "RunContext",
    "append_csv_union",
    "atomic_write_csv",
    "atomic_write_json",
    "curve_rows",
    "environment_provenance",
    "fingerprint_path",
    "fingerprint_references",
    "git_provenance",
    "json_safe",
    "metric_row",
    "sha256_file",
    "sha256_value",
    "summary_rows",
    "union_fieldnames",
]
