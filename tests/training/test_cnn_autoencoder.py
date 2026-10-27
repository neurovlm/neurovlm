from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from neurovlm.evaluation import reconstruction_metrics
from neurovlm.training import (
    AutoencoderTrainConfig,
    autoencoder_from_checkpoint,
    build_autoencoder,
    train_autoencoder,
)


class _TinyVolumes(Dataset):
    def __init__(self, split: str, sources: tuple[str, ...] = ("pubmed", "nilearn")):
        self.items = []
        for index, source in enumerate(sources):
            generator = torch.Generator().manual_seed(100 + index)
            volume = torch.rand((1, 4, 4, 4), generator=generator)
            self.items.append(
                {
                    "volume": volume,
                    "map_id": f"{split}-{index}",
                    "metadata": {"source": source},
                    "positive_texts": [],
                }
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class _TinyProvider:
    def __init__(self):
        self.train = _TinyVolumes("train", ("pubmed", "nilearn", "neurovault"))
        self.val = _TinyVolumes("val")
        self.test = _TinyVolumes("test")


def _config(tmp_path: Path, **overrides) -> AutoencoderTrainConfig:
    values = {
        "output_root": tmp_path,
        "run_id": "tiny-ae",
        "epochs": 1,
        "batch_size": 2,
        "eval_batch_size": 2,
        "device": "cpu",
        "amp": False,
        "early_stopping_patience": None,
        "target_shape": (4, 4, 4),
        "base_channels": 2,
        "num_blocks": 1,
        "latent_dim": 4,
        "preset": "custom",
    }
    values.update(overrides)
    return AutoencoderTrainConfig(**values)


def test_spatial_metrics_preserve_overlap_alias_and_report_true_recall() -> None:
    target = torch.arange(100.0).reshape(1, 1, 4, 5, 5) / 99.0
    perfect = reconstruction_metrics(target, target)
    assert perfect["mse"] == pytest.approx(0.0)
    assert perfect["spatial_corr"] == pytest.approx(1.0)
    for k in (1, 5, 10):
        assert perfect[f"top{k}_dice"] == pytest.approx(1.0)
        assert perfect[f"top{k}_overlap"] == perfect[f"top{k}_dice"]
        assert perfect[f"top{k}_target_recall"] == pytest.approx(1.0)

    raw = target.clone()
    raw.flatten()[0] = -2
    raw.flatten()[1] = 3
    diagnostics = reconstruction_metrics(raw, target)
    assert diagnostics["top5_overlap"] == diagnostics["top5_dice"]
    assert diagnostics["raw_pred_fraction_below_zero"] > 0
    assert diagnostics["raw_pred_fraction_above_one"] > 0


def test_typed_config_defaults_and_variant_selection(tmp_path: Path) -> None:
    mixed = _config(tmp_path)
    assert mixed.primary_metric == "val_loss"
    assert mixed.architecture()["base_channels"] == 2
    fine = _config(tmp_path, variant="finetuned", domain="pubmed")
    assert fine.primary_metric == "val_top5_dice"
    assert fine.metric_direction.value == "max"
    with pytest.raises(ValueError, match="requires a domain"):
        _config(tmp_path, variant="finetuned")
    with pytest.raises(ValueError, match="domain-independent"):
        _config(tmp_path, domain="pubmed")
    with pytest.raises(ValueError, match="at most one"):
        _config(tmp_path, from_run="a", init_checkpoint="b")
    with pytest.raises(ValueError, match="use preset='custom'"):
        AutoencoderTrainConfig(base_channels=2)


def test_tiny_offline_training_artifacts_checkpoint_reload_and_determinism(tmp_path: Path) -> None:
    first = train_autoencoder(_config(tmp_path / "first"), provider=_TinyProvider())
    second = train_autoencoder(_config(tmp_path / "second"), provider=_TinyProvider())

    assert first.best_checkpoint.is_file()
    assert first.last_checkpoint.is_file()
    assert first.test_metrics["reconstruction_mse"] >= 0
    assert (first.run_dir / "metrics" / "history.csv").is_file()
    assert (first.run_dir / "metrics" / "test_summary.csv").is_file()
    assert (first.run_dir / "metrics" / "by_source.csv").is_file()
    effective = json.loads((first.run_dir / "config" / "effective.json").read_text())
    assert effective["primary_metric"] == "val_loss"
    assert effective["values"]["loss"] == "raw_mse"
    manifest = json.loads((first.run_dir / "manifest.json").read_text())
    assert manifest["metric_direction"] == "min"

    reloaded = autoencoder_from_checkpoint(first.best_checkpoint)
    sample = torch.rand(1, 1, 4, 4, 4)
    with torch.no_grad():
        assert torch.allclose(first.model(sample), reloaded(sample))
    assert first.best_metric == pytest.approx(second.best_metric, abs=1e-7)
    for left, right in zip(first.model.parameters(), second.model.parameters()):
        assert torch.equal(left, right)

    bad = torch.load(first.best_checkpoint, weights_only=True)
    bad["architecture"]["base_channels"] = 3
    bad_path = tmp_path / "bad.pt"
    torch.save(bad, bad_path)
    with pytest.raises(ValueError, match="recorded architecture"):
        autoencoder_from_checkpoint(bad_path)


def test_resume_keeps_metric_history(monkeypatch, tmp_path: Path) -> None:
    import neurovlm.training.autoencoder as module

    original = module._train_epoch
    calls = 0

    def interrupt_after_first(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return original(*args, **kwargs)

    config = _config(tmp_path, epochs=2)
    monkeypatch.setattr(module, "_train_epoch", interrupt_after_first)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train_autoencoder(config, provider=_TinyProvider())
    history_path = tmp_path / "tiny-ae" / "metrics" / "history.csv"
    before = list(csv.DictReader(history_path.open()))
    assert {row["epoch"] for row in before} == {"1"}

    monkeypatch.setattr(module, "_train_epoch", original)
    resumed_config = _config(tmp_path, epochs=2, resume="last.pt")
    result = train_autoencoder(resumed_config, provider=_TinyProvider())
    after = list(csv.DictReader(history_path.open()))
    assert result.epochs_completed == 2
    assert {row["epoch"] for row in after if row["epoch"]} == {"1", "2"}
    assert len(after) > len(before)


def test_finetuned_defaults_to_fresh_trainable_released_mixed(monkeypatch, tmp_path: Path) -> None:
    import neurovlm.models.base as models

    released = build_autoencoder(_config(tmp_path))
    for parameter in released.parameters():
        parameter.requires_grad_(False)
    calls = []

    def fake_load_model(**kwargs):
        calls.append(kwargs)
        return released

    monkeypatch.setattr(models, "load_model", fake_load_model)
    result = train_autoencoder(
        _config(tmp_path, variant="finetuned", domain="pubmed"),
        provider=_TinyProvider(),
    )
    assert calls == [{"family": "cnn", "task": "autoencoder", "variant": "mixed_baseline"}]
    assert all(parameter.requires_grad for parameter in result.model.parameters())
    assert all(not parameter.requires_grad for parameter in released.parameters())
    effective = json.loads((result.run_dir / "config" / "effective.json").read_text())
    assert effective["primary_metric"] == "val_top5_dice"
    assert effective["metric_direction"] == "max"
    assert effective["values"]["initialization_source"] == "released_mixed"
