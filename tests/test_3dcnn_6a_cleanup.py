"""Smoke tests for the atlas-free 3D CNN 6a cleanup path."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
THREEDCNN = REPO_ROOT / "experiments" / "3dcnn"
if str(THREEDCNN) not in sys.path:
    sys.path.insert(0, str(THREEDCNN))
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from atlas_free_cnn.evaluation.stage4_semantic import evaluate_generation_semantic_loader
from atlas_free_cnn.notebook_utils import (
    CORRECTED_STAGE4_CHECKPOINT,
    CORRECTED_STAGE4_DIRNAME,
    LOCKED_STAGE1_CHECKPOINT_NAMES,
    NORMALIZED_STAGE3_CHECKPOINT,
    NORMALIZED_STAGE3_DIRNAME,
    STAGE4_PRIMARY_SPATIAL_CHECKPOINT,
    STAGE4_SPATIAL_CORR_CHECKPOINT,
    STAGE4_SEMANTIC_CHECKPOINT,
    corrected_stage4_dirname_for_text_embedding_convention,
    locked_stage1_checkpoint_selection,
    resolve_text_embedding_cache,
    run_subprocess_streaming,
    select_six_downstream_runs,
    six_branch_specs,
    stage3_dirname_for_text_embedding_convention,
    stage_output_dir,
    text_embedding_metadata_fields,
    text_embedding_convention_dir_suffix,
    validate_normalized_specter_cache,
)
from atlas_free_cnn.pipeline_outputs import write_status_report
from atlas_free_cnn.training.model_wrappers import build_generative_text_to_ae_latent


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_run_subprocess_streaming_forwards_stdout_and_stderr(capsys) -> None:
    completed = run_subprocess_streaming(
        [
            sys.executable,
            "-u",
            "-c",
            (
                "import os, sys\n"
                "print('streaming smoke stdout', flush=True)\n"
                "print('streaming smoke stderr', file=sys.stderr, flush=True)\n"
                "print('PYTHONUNBUFFERED=' + os.environ.get('PYTHONUNBUFFERED', ''), flush=True)\n"
            ),
        ],
        label="pytest-stream",
    )

    output = capsys.readouterr().out
    assert completed.returncode == 0
    assert "streaming smoke stdout" in output
    assert "streaming smoke stderr" in output
    assert "PYTHONUNBUFFERED=1" in output


def test_run_subprocess_streaming_failure_mentions_command_and_return_code() -> None:
    try:
        run_subprocess_streaming([sys.executable, "-u", "-c", "import sys; sys.exit(7)"], label="pytest-fail")
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("run_subprocess_streaming should fail on nonzero return code")

    assert "pytest-fail" in message
    assert "return code 7" in message
    assert "-c" in message


def test_fixed_six_branch_specs_have_expected_run_list() -> None:
    expected = [
        "mixed_stage1a_on_pubmed",
        "mixed_to_pubmed_stage1b_on_pubmed",
        "mixed_stage1a_on_nilearn",
        "mixed_to_nilearn_stage1b_on_nilearn",
        "mixed_stage1a_on_neurovault",
        "mixed_to_neurovault_stage1b_on_neurovault",
    ]
    specs = six_branch_specs()
    assert [spec["run"] for spec in specs] == expected
    assert [spec["run"] for spec in select_six_downstream_runs(list(reversed(specs)))] == expected


def test_status_requires_all_six_branches(tmp_path: Path) -> None:
    for spec in six_branch_specs():
        branch_dir = tmp_path / spec["domain_dir"] / spec["branch"]
        stage3_dir = branch_dir / NORMALIZED_STAGE3_DIRNAME
        stage4_dir = branch_dir / CORRECTED_STAGE4_DIRNAME
        _write_json(stage3_dir / "NORMALIZED_STAGE3_COMPLETE.json", {"status": "complete"})
        _write_json(stage3_dir / "eval_results.json", {"paper_recall_curve_auc": 0.7})
        _write_json(stage4_dir / "training_stop.json", {"stop_reason": "smoke"})
        _write_json(stage4_dir / "generation_eval_metrics.json", [{"source": "all"}])

    statuses = write_status_report(
        tmp_path,
        {"stage3_normalized_specter": True, "corrected_stage4_normalized_specter": True},
    )
    by_stage = {row["stage"]: row for row in statuses}
    assert by_stage["stage3"]["completed_runs"] == 6
    assert by_stage["stage4"]["completed_runs"] == 6


def test_notebook_defaults_use_fixed_six_runs_and_required_stage1b() -> None:
    nb6a = json.loads((REPO_ROOT / "experiments/3dcnn/5 multi source stage3 stage4.ipynb").read_text())
    nb5 = json.loads((REPO_ROOT / "experiments/3dcnn/4 multi source autoencoder ablation.ipynb").read_text())
    source6a = "\n".join("".join(cell.get("source", "")) for cell in nb6a["cells"])
    source5 = "\n".join("".join(cell.get("source", "")) for cell in nb5["cells"])

    assert "AE_BRANCH_MODE" not in source6a
    assert "six_branch_specs()" in source6a
    assert "select_six_downstream_runs" in source6a
    assert "AE_BRANCH_REQUIRED_SELECTION_KEYS" in source6a
    assert "required_selection_keys=tuple(AE_BRANCH_REQUIRED_SELECTION_KEYS)" in source6a
    assert "RUN_SUBPROCESS_STREAMING_SMOKE_TEST" in source6a
    assert "run_subprocess_streaming(cmd, cwd=REPO_DIR" in source6a
    assert "run_subprocess_streaming(cmd, env=env, cwd=REPO_DIR" in source6a
    assert "--strict-controlled-recipe" in source6a
    assert "RUN_STAGE1A_MIXED_PRETRAINING = True" in source5
    assert "RUN_STAGE1B_FINETUNING" not in source5
    assert "if results:" in source5


def test_6a_status_detects_normalized_stage3_and_corrected_stage4_without_legacy_dirs(tmp_path: Path) -> None:
    run_dir = tmp_path / "6a_results_smoke"
    for spec in six_branch_specs():
        branch_dir = run_dir / spec["domain_dir"] / spec["branch"]
        stage3_dir = branch_dir / NORMALIZED_STAGE3_DIRNAME
        stage4_dir = branch_dir / CORRECTED_STAGE4_DIRNAME
        _write_json(stage3_dir / "NORMALIZED_STAGE3_COMPLETE.json", {"status": "complete"})
        _write_json(stage3_dir / "eval_results.json", {"paper_recall_curve_auc": 0.7})
        (stage3_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (stage3_dir / "checkpoints" / NORMALIZED_STAGE3_CHECKPOINT).write_bytes(b"stage3")
        _write_json(stage4_dir / "training_stop.json", {"stop_reason": "max_epochs"})
        _write_json(stage4_dir / "generation_eval_metrics.json", [{"source": "all", "generation_mean_normalized_auc": 0.6}])
        (stage4_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (stage4_dir / "checkpoints" / CORRECTED_STAGE4_CHECKPOINT).write_bytes(b"stage4")

    statuses = write_status_report(
        run_dir,
        {"stage3_normalized_specter": True, "corrected_stage4_normalized_specter": True, "stage5": False},
    )
    by_stage = {row["stage"]: row for row in statuses}
    assert by_stage["stage3"]["status"] == "ran successfully"
    assert by_stage["stage3"]["completed_runs"] == 6
    assert by_stage["stage4"]["status"] == "ran successfully"
    assert by_stage["stage4"]["completed_runs"] == 6
    assert not list(run_dir.glob("[0-9][0-9]_*/*/stage3"))
    assert not list(run_dir.glob("[0-9][0-9]_*/*/stage4"))


def test_6a_status_allows_exported_metrics_without_checkpoint_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "6a_results_export"
    for spec in six_branch_specs():
        branch_dir = run_dir / spec["domain_dir"] / spec["branch"]
        stage3_dir = branch_dir / NORMALIZED_STAGE3_DIRNAME
        stage4_dir = branch_dir / CORRECTED_STAGE4_DIRNAME
        _write_json(stage3_dir / "NORMALIZED_STAGE3_COMPLETE.json", {"status": "complete"})
        _write_json(stage3_dir / "eval_results.json", {"paper_recall_curve_auc": 0.7})
        _write_json(stage4_dir / "training_stop.json", {"stop_reason": "early_stopping"})
        _write_json(stage4_dir / "generation_eval_metrics.json", [{"source": "all"}])

    statuses = write_status_report(
        run_dir,
        {"stage3_normalized_specter": True, "corrected_stage4_normalized_specter": True},
    )
    by_stage = {row["stage"]: row for row in statuses}
    assert by_stage["stage3"]["status"] == "ran successfully"
    assert by_stage["stage4"]["status"] == "ran successfully"
    assert by_stage["stage4"]["runs"][0]["training_completed_on_drive"] is True
    assert by_stage["stage4"]["runs"][0]["metrics_exported"] is True
    assert by_stage["stage4"]["runs"][0]["checkpoints_in_export_zip"] is False


class TinyEncoder(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.flatten(1)
        return flat[:, :384]


class TinyAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = TinyEncoder()
        self.decoder = nn.Linear(384, 384)


def test_corrected_stage4_projector_is_fresh_and_targets_ae_latent() -> None:
    autoencoder = TinyAutoencoder()
    for param in autoencoder.parameters():
        param.requires_grad_(False)
    projector = build_generative_text_to_ae_latent(device="cpu", in_dim=768, hidden_dim=512, latent_dim=384)
    stage3_projection_state = {"aligner.0.weight": torch.randn(512, 768), "aligner.3.weight": torch.randn(384, 512)}
    compatible = {
        key: value
        for key, value in stage3_projection_state.items()
        if key in projector.state_dict() and tuple(value.shape) == tuple(projector.state_dict()[key].shape)
    }
    text = torch.randn(2, 768)
    brain = torch.randn(2, 1, 8, 8, 6)
    predicted_latent = projector(text)
    true_latent = autoencoder.encoder(brain)
    assert compatible == {}
    assert not any(param.requires_grad for param in autoencoder.encoder.parameters())
    assert not any(param.requires_grad for param in autoencoder.decoder.parameters())
    assert predicted_latent.shape == (2, 384)
    assert true_latent.shape == (2, 384)


def test_normalized_cache_validation_requires_768_unit_vectors_and_text_ids(tmp_path: Path) -> None:
    embeddings = torch.nn.functional.normalize(torch.randn(4, 768), dim=1)
    cache_path = tmp_path / "specter2_stage3_stage4_emptycentered_unitnorm.pt"
    torch.save({"embeddings": embeddings, "text_ids": ["t0", "t1", "t2", "t3"], "metadata": {}}, cache_path)
    audit = validate_normalized_specter_cache(cache_path, required_text_ids={"t0", "t3"})
    assert audit["stats"]["dim"] == 768
    assert abs(audit["stats"]["norm_mean"] - 1.0) < 1e-6
    assert audit["stats"]["required_text_ids_present"] is True


def test_text_embedding_resolver_is_normalized_only(tmp_path: Path) -> None:
    spec = resolve_text_embedding_cache(local_cache_dir=tmp_path, env_override=False)

    assert spec["cache_name"] == "specter2_stage3_stage4_emptycentered_unitnorm.pt"
    assert spec["hf_path"] == "text_embeddings/specter2_stage3_stage4_emptycentered_unitnorm.pt"
    assert spec["metadata_hf_path"] == "text_embeddings/specter2_stage3_stage4_emptycentered_unitnorm_metadata.json"
    assert spec["preprocessing"] == "empty_string_centered_l2_unit_normalized"
    assert spec["expect_unit_norm"] is True



def test_normalized_stage3_stage4_folder_names_and_status_smoke(tmp_path: Path) -> None:
    assert text_embedding_convention_dir_suffix("normalized_specter2") == "normalized_specter"
    assert stage3_dirname_for_text_embedding_convention("normalized_specter2") == NORMALIZED_STAGE3_DIRNAME
    assert corrected_stage4_dirname_for_text_embedding_convention("normalized_specter2") == CORRECTED_STAGE4_DIRNAME
    for spec in six_branch_specs():
        stage3_dir = stage_output_dir(tmp_path, spec["domain"], spec["branch"], "stage3")
        stage4_dir = stage_output_dir(tmp_path, spec["domain"], spec["branch"], "stage4")
        _write_json(stage3_dir / "eval_results.json", {"paper_recall_curve_auc": 0.7})
        (stage3_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (stage3_dir / "checkpoints" / NORMALIZED_STAGE3_CHECKPOINT).write_bytes(b"stage3")
        _write_json(stage4_dir / "training_stop.json", {"stop_reason": "smoke"})
        _write_json(stage4_dir / "generation_eval_metrics.json", [{"source": "all"}])
        (stage4_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (stage4_dir / "checkpoints" / CORRECTED_STAGE4_CHECKPOINT).write_bytes(b"stage4")

    statuses = write_status_report(
        tmp_path,
        {"stage3_normalized_specter": True, "corrected_stage4_normalized_specter": True},
    )
    by_stage = {row["stage"]: row for row in statuses}
    assert by_stage["stage3"]["completed_runs"] == 6
    assert by_stage["stage4"]["completed_runs"] == 6

class FakeDecoder(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z[:, :8].reshape(z.shape[0], 1, 2, 2, 2)


class FakeAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = FakeDecoder()


class FakeProjector(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x[:, :8], torch.zeros(x.shape[0], 376)], dim=1)


class FakeBrainEncoder(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(1)[:, :4]


class FakeTextProjection(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :4]


def test_stage4_semantic_evaluator_returns_raw_clamped_and_duplicate_aware_auc() -> None:
    cache = {
        "alpha": torch.tensor([1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.2, 0.1] + [0.0] * 760),
        "beta": torch.tensor([0.0, 1.0, 0.0, 0.0, 0.3, 0.4, 0.2, 0.1] + [0.0] * 760),
    }
    loader = [
        {
            "texts": ["alpha", "beta"],
            "text_entries": [{"text_id": "a", "text": "same"}, {"text_id": "b", "text": "same"}],
            "metadata": [{"publication_id": "p1"}, {"publication_id": "p1"}],
            "map_id": ["m1", "m2"],
        }
    ]
    metrics = evaluate_generation_semantic_loader(
        FakeAutoencoder(),
        FakeProjector(),
        FakeBrainEncoder(),
        FakeTextProjection(),
        loader,
        cache,
        torch.device("cpu"),
        prefix="generation",
    )
    assert "generation_raw_strict_map_mean_normalized_auc" in metrics
    assert "generation_clamped_strict_map_mean_normalized_auc" in metrics
    assert "generation_clamped_same_text_group_mean_normalized_auc" in metrics
    assert "generation_clamped_publication_group_mean_normalized_auc" in metrics
    assert "generation_mean_normalized_auc" in metrics


def test_5b_locked_checkpoint_selection_names_are_authoritative(tmp_path: Path) -> None:
    registry = {}
    for variant in ["mixed_baseline_raw_mse", "mixed_to_pubmed", "mixed_to_nilearn", "mixed_to_neurovault"]:
        run_dir = tmp_path / variant
        (run_dir / "checkpoints").mkdir(parents=True)
        for name in set(LOCKED_STAGE1_CHECKPOINT_NAMES.values()) | {"empirical_other.pt"}:
            (run_dir / "checkpoints" / name).write_bytes(f"{variant}:{name}".encode())
        registry[variant] = {"run_dir": str(run_dir)}
    locked = locked_stage1_checkpoint_selection(registry)
    assert {key: row["checkpoint_name"] for key, row in locked.items()} == LOCKED_STAGE1_CHECKPOINT_NAMES
    empirical = {"mixed_to_nilearn_stage1b": {"checkpoint_name": "empirical_other.pt"}}
    assert empirical["mixed_to_nilearn_stage1b"]["checkpoint_name"] != locked["mixed_to_nilearn_stage1b"]["checkpoint_name"]
    assert locked["mixed_to_nilearn_stage1b"]["checkpoint_name"] == "best_val_loss.pt"


# ---------------------------------------------------------------------------
# Checkpoint filename policy tests
# ---------------------------------------------------------------------------

def test_stage3_canonical_checkpoint_name_is_best_val_normalized_recall_auc() -> None:
    assert NORMALIZED_STAGE3_CHECKPOINT == "best_val_normalized_recall_auc.pt"


def test_stage3_trainer_produces_canonical_checkpoint(tmp_path: Path) -> None:
    """Simulate the save_best / save_last checkpoint names produced by ALETrainer."""
    import types
    import argparse
    from atlas_free_cnn.training.train_ale_cnn import ALETrainer

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    args = argparse.Namespace(
        checkpoint_dir=str(ckpt_dir),
        temperature=0.07,
        freeze_text_proj=False,
        amp=False,
        grad_accum_steps=1,
        pin_memory=False,
        monitor_metric="paper_recall_curve_auc",
        early_stopping_patience=None,
        val_interval=1,
    )
    device = torch.device("cpu")
    brain_encoder = nn.Linear(4, 4)
    text_proj = nn.Linear(4, 4)
    trainer = ALETrainer.__new__(ALETrainer)
    trainer.brain_encoder = brain_encoder
    trainer.text_proj = text_proj
    trainer.args = args
    trainer.device = device
    trainer.loss_fn = None
    trainer.optimizer = torch.optim.SGD(brain_encoder.parameters(), lr=0.01)
    trainer.scheduler = None
    trainer.grad_accum_steps = 1
    trainer.use_amp = False
    trainer.amp_dtype = torch.float32
    trainer.scaler = torch.cuda.amp.GradScaler(enabled=False)
    trainer.best_state = {
        "brain_encoder": brain_encoder.state_dict(),
        "text_proj": text_proj.state_dict(),
        "epoch": 1,
        "metrics": {},
        "config": {},
    }
    trainer.best_score = 0.5
    trainer._printed_contrastive_sanity = False
    trainer.preflight_peak_vram_mb = 0.0
    trainer.history = {"train_loss": [], "epoch_time_sec": [], "peak_vram_mb": [], "lr_brain": [], "lr_proj": [], "val_epoch": []}
    trainer.timing_profile = {}

    trainer.save_best()
    trainer.save_last(epoch=1, bad_checks=0)

    assert (ckpt_dir / "best_val_normalized_recall_auc.pt").exists(), "canonical Stage 3 best checkpoint must exist"
    assert (ckpt_dir / "last.pt").exists(), "canonical Stage 3 last checkpoint must exist"

    assert not (ckpt_dir / "best_ale_cnn.pt").exists()
    assert not (ckpt_dir / "last_ale_cnn.pt").exists()


def test_stage4_canonical_checkpoint_names() -> None:
    assert STAGE4_PRIMARY_SPATIAL_CHECKPOINT == "best_val_top5_dice.pt"
    assert STAGE4_SEMANTIC_CHECKPOINT == "best_val_generation_normalized_auc.pt"
    assert CORRECTED_STAGE4_CHECKPOINT == "best_val_generation_normalized_auc.pt"
    assert STAGE4_SPATIAL_CORR_CHECKPOINT == "best_val_spatial_corr.pt"


def test_stage4_trainer_produces_only_canonical_checkpoints(tmp_path: Path) -> None:
    """Verify the Stage 4 CheckpointManager only saves canonical files by default."""
    from atlas_free_cnn.training.checkpointing import CheckpointManager
    ckpt = CheckpointManager(
        tmp_path,
        maximize={"val_spatial_corr": True, "val_top5_dice": True, "val_generation_normalized_auc": True},
        require_explicit_direction=True,
    )
    payload = {"dummy": torch.zeros(1)}
    ckpt.save_last(payload, epoch=3)
    ckpt.maybe_save_best("val_spatial_corr", 0.5, payload, epoch=1)
    ckpt.maybe_save_best("val_top5_dice", 0.6, payload, epoch=2)
    ckpt.maybe_save_best("val_generation_normalized_auc", 0.7, payload, epoch=3)

    assert (tmp_path / "last.pt").exists()
    assert (tmp_path / "best_val_top5_dice.pt").exists()
    assert (tmp_path / "best_val_generation_normalized_auc.pt").exists()
    assert (tmp_path / "best_val_spatial_corr.pt").exists()
    assert (tmp_path / "checkpoint_manifest.json").exists()
    manifest = json.loads((tmp_path / "checkpoint_manifest.json").read_text())
    last_row = next(row for row in manifest["checkpoints"] if row["checkpoint_name"] == "last.pt")
    assert last_row["epoch"] == 3

    legacy_names = [
        "best_val_loss.pt",
        "best_val_latent_mse.pt",
        "best_val_reconstruction_mse.pt",
        "best_generation_top5_dice.pt",
        "best_generation_spatial_correlation.pt",
    ]
    for name in legacy_names:
        assert not (tmp_path / name).exists(), f"legacy checkpoint {name} must not exist by default"






def test_stage4_primary_is_spatial_not_semantic(tmp_path: Path) -> None:
    """Verify the spatial checkpoint (best_val_top5_dice.pt) is the primary Stage 4 file."""
    assert STAGE4_PRIMARY_SPATIAL_CHECKPOINT == "best_val_top5_dice.pt"
    assert STAGE4_PRIMARY_SPATIAL_CHECKPOINT != CORRECTED_STAGE4_CHECKPOINT


def test_generation_auc_val_interval_defaults_to_5() -> None:
    import inspect, ast
    from atlas_free_cnn.training import train_text_to_brain
    src = inspect.getsource(train_text_to_brain)
    assert 'cfg.setdefault("generation_auc_val_interval", 5)' in src, (
        "generation_auc_val_interval must default to 5"
    )






def test_stage4_primary_checkpoint_metric_defaults_to_val_top5_dice() -> None:
    """primary_checkpoint_metric must default to val_top5_dice (spatial primary), not val_generation_normalized_auc."""
    import inspect
    from atlas_free_cnn.training import train_text_to_brain
    src = inspect.getsource(train_text_to_brain)
    assert 'cfg.setdefault("primary_checkpoint_metric", _checkpoint_metric_from_name(str(cfg["stage4_primary_checkpoint"])))' in src, (
        "primary_checkpoint_metric must default to val_top5_dice (spatial primary)"
    )
    assert 'cfg.setdefault("primary_checkpoint_metric", "val_generation_normalized_auc")' not in src, (
        "Semantic primary checkpoint must not be the default anymore"
    )
    assert 'ckpt.save_last(payload, epoch=epoch)' in src


def test_stage3_checkpoint_path_returns_canonical_name() -> None:
    """stage_checkpoint_path() must return NORMALIZED_STAGE3_CHECKPOINT for stage3, not best_ale_cnn.pt."""
    from atlas_free_cnn.notebook_utils import stage_checkpoint_path, NORMALIZED_STAGE3_CHECKPOINT
    from pathlib import Path

    path = stage_checkpoint_path("/some/run", "pubmed", "baseline_mixed_stage1a", "stage3")
    assert path.name == NORMALIZED_STAGE3_CHECKPOINT, (
        f"stage_checkpoint_path() for stage3 must return {NORMALIZED_STAGE3_CHECKPOINT}, got {path.name}"
    )
    assert path.name != "best_ale_cnn.pt", "stage_checkpoint_path() must not return legacy best_ale_cnn.pt"


def test_timing_profile_has_named_summary_fields() -> None:
    """timing_profile.json must include the explicitly named scalar timing summary fields."""
    import inspect
    from atlas_free_cnn.training import train_text_to_brain
    src = inspect.getsource(train_text_to_brain)
    for field in [
        "train_epoch_time_sec",
        "val_metric_time_sec",
        "generation_auc_eval_time_sec",
        "checkpoint_save_time_sec",
        "branch_total_time_sec",
    ]:
        assert f'timing_profile["{field}"]' in src, f"timing_profile must include field {field!r}"
