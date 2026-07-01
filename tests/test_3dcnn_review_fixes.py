from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import pytest
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
THREEDCNN = REPO_ROOT / "experiments" / "3dcnn"
if str(THREEDCNN) not in sys.path:
    sys.path.insert(0, str(THREEDCNN))
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from atlas_free_cnn.evaluation.stage1_checkpoint_evaluation import create_stage1b_selection
from atlas_free_cnn.training.checkpointing import CheckpointManager, metric_higher_is_better
from atlas_free_cnn.training.train_ale_cnn import trainability_report


class _TinyStage3Trainer:
    def __init__(self) -> None:
        self.brain_encoder = nn.Linear(2, 2)
        self.text_proj = nn.Linear(2, 2)
        self.optimizer = torch.optim.SGD(
            [
                {"params": self.brain_encoder.parameters(), "lr": 0.1, "weight_decay": 0.0},
                {"params": self.text_proj.parameters(), "lr": 0.01, "weight_decay": 0.0},
            ]
        )


def _stage3_args(tmp_path: Path, *, strict: bool) -> argparse.Namespace:
    return argparse.Namespace(
        encoder_init="random",
        text_proj_init="pretrained_infonce",
        out_dim=384,
        run_dir=str(tmp_path),
        strict_controlled_recipe=strict,
    )


def test_stage3_strict_controlled_recipe_fails_invalid_recipe(tmp_path: Path) -> None:
    trainer = _TinyStage3Trainer()
    for param in trainer.text_proj.parameters():
        param.requires_grad_(False)

    with pytest.raises(RuntimeError, match="controlled-recipe verification failed"):
        trainability_report(trainer, _stage3_args(tmp_path, strict=True))

    report = json.loads((tmp_path / "stage3_trainability_report.json").read_text())
    assert report["strict_controlled_recipe"] is True
    assert "encoder_init is not autoencoder_pretrained" in report["failures"]
    assert "text projection is frozen" in report["failures"]


def test_stage3_non_strict_controlled_recipe_warns_invalid_recipe(tmp_path: Path) -> None:
    trainer = _TinyStage3Trainer()
    for param in trainer.text_proj.parameters():
        param.requires_grad_(False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = trainability_report(trainer, _stage3_args(tmp_path, strict=False))

    assert report["status"] == "failed"
    assert report["strict_controlled_recipe"] is False
    assert any("controlled-recipe verification failed" in str(item.message) for item in caught)


def test_notebook_6a_stage3_launches_with_strict_controlled_recipe() -> None:
    nb = json.loads((THREEDCNN / "6 multi source stage3 stage4.ipynb").read_text())
    source = "\n".join("".join(cell.get("source", "")) for cell in nb["cells"])
    assert '"--strict-controlled-recipe"' in source


def test_notebook_guide_references_existing_notebooks() -> None:
    guide = (THREEDCNN / "NOTEBOOK_GUIDE.md").read_text()
    notebooks = re.findall(r"`([^`]+\.ipynb)`", guide)
    assert notebooks
    missing = [name for name in notebooks if not (THREEDCNN / name).exists()]
    assert missing == []


def test_autoencoder_metric_direction_policy() -> None:
    assert metric_higher_is_better("top5_dice") is True
    assert metric_higher_is_better("spatial_corr") is True
    assert metric_higher_is_better("foreground_mse") is False
    with pytest.raises(ValueError, match="Unknown checkpoint/selection metric direction"):
        metric_higher_is_better("mystery_metric")


def test_checkpoint_manager_uses_metric_direction_for_best_selection(tmp_path: Path) -> None:
    ckpt = CheckpointManager(
        tmp_path,
        maximize={
            "top5_dice": metric_higher_is_better("top5_dice"),
            "foreground_mse": metric_higher_is_better("foreground_mse"),
        },
        require_explicit_direction=True,
    )
    payload = {"model": {"w": torch.ones(1)}}

    assert ckpt.maybe_save_best("top5_dice", 0.2, payload, epoch=1)
    assert ckpt.maybe_save_best("top5_dice", 0.8, payload, epoch=2)
    assert not ckpt.maybe_save_best("top5_dice", 0.1, payload, epoch=3)
    assert ckpt.best["top5_dice"] == pytest.approx(0.8)

    assert ckpt.maybe_save_best("foreground_mse", 0.5, payload, epoch=1)
    assert ckpt.maybe_save_best("foreground_mse", 0.2, payload, epoch=2)
    assert not ckpt.maybe_save_best("foreground_mse", 0.9, payload, epoch=3)
    assert ckpt.best["foreground_mse"] == pytest.approx(0.2)


def _stage1b_row(checkpoint: str, *, top5: float, spatial: float, fg: float) -> dict[str, object]:
    return {
        "variant": "mixed_to_pubmed",
        "stage": "stage1b",
        "test_domain": "pubmed",
        "eval_scope": "primary",
        "alias_status": "canonical",
        "load_status": "loaded",
        "checkpoint_name": checkpoint,
        "checkpoint_path": f"/tmp/{checkpoint}",
        "checkpoint_epoch": 1,
        "top5_dice": top5,
        "spatial_corr": spatial,
        "foreground_mse": fg,
        "reconstruction_mse": fg * 2,
    }


def test_stage1b_checkpoint_selection_uses_correct_metric_directions(tmp_path: Path) -> None:
    rows = [
        _stage1b_row("best_val_loss.pt", top5=0.20, spatial=0.90, fg=0.01),
        _stage1b_row("best_top5_dice.pt", top5=0.80, spatial=0.40, fg=0.20),
        _stage1b_row("last.pt", top5=0.50, spatial=0.50, fg=0.001),
    ]

    selected = create_stage1b_selection(rows, tmp_path)["pubmed"]

    assert selected[0]["checkpoint_name"] == "best_top5_dice.pt"


def test_stage4_checkpoint_manifest_last_epoch_and_optional_semantic(tmp_path: Path) -> None:
    ckpt = CheckpointManager(
        tmp_path,
        maximize={"val_spatial_corr": True, "val_top5_dice": True},
        require_explicit_direction=True,
    )
    payload = {"text_projector": {"w": torch.ones(1)}}
    ckpt.save_last(payload, epoch=4)
    ckpt.maybe_save_best("val_spatial_corr", 0.5, payload, epoch=3)
    ckpt.maybe_save_best("val_top5_dice", 0.6, payload, epoch=4)

    assert sorted(path.name for path in tmp_path.glob("*.pt")) == [
        "best_val_spatial_corr.pt",
        "best_val_top5_dice.pt",
        "last.pt",
    ]
    manifest = json.loads((tmp_path / "checkpoint_manifest.json").read_text())
    last_row = next(row for row in manifest["checkpoints"] if row["checkpoint_name"] == "last.pt")
    assert last_row["epoch"] == 4

    ckpt.maximize["val_generation_normalized_auc"] = True
    ckpt.maybe_save_best("val_generation_normalized_auc", 0.7, payload, epoch=4)
    assert (tmp_path / "best_val_generation_normalized_auc.pt").exists()
