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


def test_current_stage3_launches_with_strict_controlled_recipe() -> None:
    nb = json.loads((THREEDCNN / "4 multi source stage3 stage4.ipynb").read_text())
    source = "\n".join("".join(cell.get("source", "")) for cell in nb["cells"])
    assert '"--strict-controlled-recipe"' in source


def test_technical_guide_references_existing_notebooks() -> None:
    guide = (THREEDCNN / "3DCNN_TECHNICAL_GUIDE.md").read_text()
    notebooks = re.findall(r"`([^`]+\.ipynb)`", guide)
    assert notebooks
    missing = [name for name in notebooks if not (THREEDCNN / name).exists()]
    assert missing == []


def test_technical_guide_explains_notebook_2_global_task() -> None:
    guide = (THREEDCNN / "3DCNN_TECHNICAL_GUIDE.md").read_text()
    assert "`2 resnet48 multi scale attention.ipynb`" in guide
    assert "global PubMed brain-text retrieval task" in guide
    assert "Independent; not consumed by notebooks 1, 3, or 4" in guide


def test_ae_comparison_defines_figure_before_export() -> None:
    notebook = json.loads((THREEDCNN / "model_comparison" / "ae_reconstruction_comparison.ipynb").read_text())
    source = "\n".join("".join(cell.get("source", "")) for cell in notebook["cells"])
    assert "fig_metrics, axes = plt.subplots" in source
    assert source.index("fig_metrics, axes = plt.subplots") < source.index(
        'figures={"ae_reconstruction_metrics": fig_metrics}'
    )


def test_plotting_palette_is_self_contained() -> None:
    source = (THREEDCNN / "model_comparison" / "plotting_utils.py").read_text()
    assert "references/palette.md" not in source
    assert "FAMILY_COLOR" in source


def test_cnn_tutorial_uses_only_packaged_model_api() -> None:
    notebook = json.loads((REPO_ROOT / "docs" / "tutorials" / "06_atlas_free_cnn.ipynb").read_text())
    source = "\n".join("".join(cell.get("source", "")) for cell in notebook["cells"])
    code = "\n".join(
        "".join(cell.get("source", "")) for cell in notebook["cells"] if cell.get("cell_type") == "code"
    )
    assert 'load_model("autoencoder_cnn")' in source
    assert 'load_model("contrastive_cnn_pubmed")' in source
    assert "experiments/" not in code
    assert "sys.path" not in code


def test_core_3dcnn_modules_do_not_import_notebook_helpers() -> None:
    core_paths = [
        THREEDCNN / "atlas_free_cnn" / "pipeline_outputs.py",
        THREEDCNN / "atlas_free_cnn" / "stage1_selection_integration.py",
        *sorted((THREEDCNN / "atlas_free_cnn" / "training").glob("*.py")),
    ]

    offenders = []
    for path in core_paths:
        source = path.read_text()
        if "atlas_free_cnn.notebook_utils" in source or "from atlas_free_cnn import notebook_utils" in source:
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert offenders == []


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
