"""Offline tests for shared pipeline infrastructure."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pytest
import torch

from neurovlm.pipelines import (
    METRIC_COLUMNS,
    CheckpointManager,
    MetricDirection,
    MetricRecorder,
    RunArtifacts,
    RunConfig,
    RunContext,
    append_csv_union,
    atomic_write_csv,
    atomic_write_json,
    fingerprint_path,
    json_safe,
    sha256_file,
)


def _config(tmp_path: Path, *, direction: str = "min", run_id: str = "run-001") -> RunConfig:
    return RunConfig.resolve(
        task="autoencoder",
        output_root=tmp_path,
        run_id=run_id,
        primary_metric="val_loss",
        metric_direction=direction,
        data={"split": "hf://published/split.jsonl"},
        resources={"checkpoint": "hf://published/model.pt"},
        initialization={"source": "released"},
    )


def test_config_resolves_model_spec_without_loading_and_records_defaults(tmp_path: Path) -> None:
    config = RunConfig.resolve(
        family="cnn",
        task="contrastive",
        domain="nilearn",
        output_root=tmp_path,
        run_id="cnn-run",
        requested={"batch_size": 7},
        effective={"batch_size": 4},
    )
    assert config.model_spec.canonical_name == "cnn:contrastive:nilearn:mixed_baseline"
    assert config.requested["variant"] is None
    assert config.effective["variant"] == "mixed_baseline"
    assert config.requested["batch_size"] == 7
    assert config.effective["batch_size"] == 4
    with pytest.raises(ValueError, match="domain is required"):
        RunConfig.resolve(family="cnn", task="contrastive", output_root=tmp_path)
    with pytest.raises(ValueError, match="metric_direction"):
        RunConfig.resolve(task="autoencoder", metric_direction="sideways")


def test_golden_run_tree_manifest_and_idempotent_resume(tmp_path: Path) -> None:
    config = _config(tmp_path)
    artifacts = RunArtifacts(config).initialize(repo=tmp_path)
    expected = {
        "config",
        "provenance",
        "checkpoints",
        "metrics",
        "plots",
        "generated_maps",
        "logs",
    }
    assert expected.issubset({path.name for path in config.run_dir.iterdir() if path.is_dir()})
    manifest = json.loads(artifacts.manifest_path.read_text())
    all_paths = [*manifest["artifacts"].values(), *manifest["directories"].values()]
    assert all(not Path(path).is_absolute() for path in all_paths)
    assert manifest["model_spec"]["canonical_name"] == "mlp:autoencoder:default"
    assert (config.run_dir / "config" / "requested.json").exists()
    assert (config.run_dir / "provenance" / "initialization.json").exists()

    artifacts.set_status("running")
    artifacts.initialize(repo=tmp_path).set_status("running")
    assert artifacts.read_status()["resume_count"] == 1

    incompatible = RunConfig.resolve(
        task="text_to_brain",
        variant="mse",
        output_root=tmp_path,
        run_id=config.run_id,
    )
    with pytest.raises(ValueError, match="incompatible"):
        RunArtifacts(incompatible).initialize(repo=tmp_path)
    changed_seed = RunConfig.resolve(
        task="autoencoder", output_root=tmp_path, run_id=config.run_id, seed=7
    )
    with pytest.raises(ValueError, match="incompatible"):
        RunArtifacts(changed_seed).initialize(repo=tmp_path)


def test_run_context_marks_completion_and_failure(tmp_path: Path) -> None:
    complete = _config(tmp_path, run_id="complete")
    with RunContext(complete, repo=tmp_path):
        assert json.loads((complete.run_dir / "status.json").read_text())["state"] == "running"
    assert json.loads((complete.run_dir / "status.json").read_text())["state"] == "completed"

    failed = _config(tmp_path, run_id="failed")
    with pytest.raises(RuntimeError, match="boom"):
        with RunContext(failed, repo=tmp_path):
            raise RuntimeError("boom")
    status = json.loads((failed.run_dir / "status.json").read_text())
    assert status["state"] == "failed"
    assert status["error"] == {"message": "boom", "type": "RuntimeError"}


def test_atomic_json_replacement_and_json_safety(tmp_path: Path) -> None:
    class Choice(str, Enum):
        A = "a"

    @dataclass
    class Example:
        path: Path
        choice: Choice

    path = tmp_path / "value.json"
    atomic_write_json(path, {"old": 1})
    atomic_write_json(
        path,
        {
            "example": Example(Path("relative"), Choice.A),
            "tensor": torch.tensor([1.0, float("nan")]),
            "array": np.array([float("inf"), 2.0]),
        },
    )
    value = json.loads(path.read_text())
    assert value == {
        "array": [None, 2.0],
        "example": {"choice": "a", "path": "relative"},
        "tensor": [1.0, None],
    }
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(TypeError):
        json_safe(object())


def test_union_csv_fields_and_atomic_logical_append(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    atomic_write_csv(path, [{"a": 1}, {"b": 2, "a": 3}])
    append_csv_union(path, [{"c": 4}])
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    assert reader.fieldnames == ["a", "b", "c"]
    assert rows[-1] == {"a": "", "b": "", "c": "4"}


def test_metric_recorder_writes_canonical_history_summary_and_curves(tmp_path: Path) -> None:
    config = _config(tmp_path)
    recorder = MetricRecorder(config)
    recorder.record(split="train", metric="loss", value=2.0, epoch=0, step=1, n=8)
    recorder.record(split="val", metric="val_loss", value=1.5, epoch=0, step=2, n=4)
    recorder.record(split="val", metric="val_loss", value=1.0, epoch=1, step=4, n=4)
    outputs = recorder.flush()
    with outputs["history"].open(newline="") as stream:
        reader = csv.DictReader(stream)
        history = list(reader)
    assert tuple(reader.fieldnames or ()) == METRIC_COLUMNS
    assert history[-1]["metric"] == "val_loss"
    summary = json.loads(outputs["summary_json"].read_text())
    val = next(row for row in summary if row["metric"] == "val_loss")
    assert val["best"] == 1.0
    assert val["last_epoch"] == 1
    assert len(list(csv.DictReader(outputs["curves"].open()))) == 3

    resumed = MetricRecorder(config)
    resumed.record(split="val", metric="val_loss", value=0.5, epoch=2)
    resumed.flush()
    assert len(list(csv.DictReader(outputs["history"].open()))) == 4


@pytest.mark.parametrize(
    ("direction", "values", "expected"),
    [("min", [3.0, 1.0, 2.0], 1.0), ("max", [1.0, 3.0, 2.0], 3.0)],
)
def test_checkpoint_best_direction_last_alias_hash_and_resume(
    tmp_path: Path, direction: str, values: list[float], expected: float
) -> None:
    config = _config(tmp_path, direction=direction)
    RunArtifacts(config).initialize(repo=tmp_path)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    manager = CheckpointManager(config, expected_architecture={"width": 2})
    for epoch, value in enumerate(values):
        manager.save_best(
            model,
            value,
            epoch=epoch,
            step=epoch + 10,
            optimizer=optimizer,
            architecture={"width": 2},
            aliases=("best_val_loss.pt",),
        )
    last = manager.save_last(
        model,
        epoch=9,
        step=99,
        optimizer=optimizer,
        architecture={"width": 2},
    )
    manifest = json.loads(manager.manifest_path.read_text())
    assert manifest["best_value"] == expected
    assert manifest["aliases"]["checkpoints/best_val_loss.pt"]["target"] == "checkpoints/best.pt"
    assert all(not Path(row["path"]).is_absolute() for row in manifest["checkpoints"])
    last_record = next(row for row in manifest["checkpoints"] if row["role"] == "last")
    assert last_record["sha256"] == sha256_file(last)

    restored = torch.nn.Linear(2, 1)
    state = manager.load_resume(model=restored, optimizer=optimizer)
    assert state.epoch == 9
    assert state.step == 99
    for expected_parameter, actual_parameter in zip(model.parameters(), restored.parameters()):
        assert torch.equal(expected_parameter, actual_parameter)


def test_checkpoint_validation_and_integrity_fail_closed(tmp_path: Path) -> None:
    config = _config(tmp_path)
    RunArtifacts(config).initialize(repo=tmp_path)
    model = torch.nn.Linear(2, 1)
    manager = CheckpointManager(config, expected_architecture={"width": 3})
    last = manager.save_last(model, architecture={"width": 2})
    with pytest.raises(ValueError, match="architecture mismatch"):
        manager.load_resume(last)

    manager = CheckpointManager(config)
    last.write_bytes(last.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        manager.load_resume(last)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_checkpoint_best_rejects_nonfinite_primary_metric(
    tmp_path: Path, value: float
) -> None:
    config = _config(tmp_path)
    manager = CheckpointManager(config)
    with pytest.raises(ValueError, match="must be finite"):
        manager.save_best(torch.nn.Linear(2, 1), value)
    assert not (config.run_dir / "checkpoints" / "best.pt").exists()


def test_fingerprint_path_is_content_sensitive_and_relative_within_tree(tmp_path: Path) -> None:
    folder = tmp_path / "data"
    folder.mkdir()
    file = folder / "x.txt"
    file.write_text("first")
    first = fingerprint_path(folder)
    file.write_text("second")
    second = fingerprint_path(folder)
    assert first["sha256"] != second["sha256"]
    assert second["files"][0]["path"] == "x.txt"
