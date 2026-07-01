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
    AE_BRANCH_MODES,
    CORRECTED_STAGE4_CHECKPOINT,
    CORRECTED_STAGE4_DIRNAME,
    CORRECTED_LEGACY_STAGE4_DIRNAME,
    LEGACY_STAGE3_DIRNAME,
    LOCKED_STAGE1_CHECKPOINT_NAMES,
    NORMALIZED_STAGE3_CHECKPOINT,
    NORMALIZED_STAGE3_DIRNAME,
    STAGE4_PRIMARY_SPATIAL_CHECKPOINT,
    STAGE4_SPATIAL_CORR_CHECKPOINT,
    STAGE4_SEMANTIC_CHECKPOINT,
    ae_branch_specs,
    corrected_stage4_dirname_for_text_embedding_convention,
    locked_stage1_checkpoint_selection,
    resolve_text_embedding_cache,
    run_subprocess_streaming,
    selected_downstream_runs_for_ae_branch_mode,
    six_branch_specs,
    stage3_dirname_for_text_embedding_convention,
    stage_output_dir,
    text_embedding_metadata_fields,
    text_embedding_convention_dir_suffix,
    validate_legacy_specter_cache,
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


def test_ae_branch_mode_specs_have_expected_run_lists() -> None:
    expected = {
        "mixed_only": [
            "mixed_stage1a_on_pubmed",
            "mixed_stage1a_on_nilearn",
            "mixed_stage1a_on_neurovault",
        ],
        "mixed_and_specialized": [
            "mixed_stage1a_on_pubmed",
            "mixed_to_pubmed_stage1b_on_pubmed",
            "mixed_stage1a_on_nilearn",
            "mixed_to_nilearn_stage1b_on_nilearn",
            "mixed_stage1a_on_neurovault",
            "mixed_to_neurovault_stage1b_on_neurovault",
        ],
        "specialized_only": [
            "mixed_to_pubmed_stage1b_on_pubmed",
            "mixed_to_nilearn_stage1b_on_nilearn",
            "mixed_to_neurovault_stage1b_on_neurovault",
        ],
    }

    assert set(AE_BRANCH_MODES) == set(expected)
    for mode, run_names in expected.items():
        specs = ae_branch_specs(mode)
        assert [spec["run"] for spec in specs] == run_names
        assert [spec["run"] for spec in selected_downstream_runs_for_ae_branch_mode(six_branch_specs(), mode)] == run_names


def test_6a_status_uses_selected_ae_branch_mode_counts(tmp_path: Path) -> None:
    for mode in AE_BRANCH_MODES:
        run_dir = tmp_path / mode
        for spec in ae_branch_specs(mode):
            branch_dir = run_dir / spec["domain_dir"] / spec["branch"]
            stage3_dir = branch_dir / NORMALIZED_STAGE3_DIRNAME
            stage4_dir = branch_dir / CORRECTED_STAGE4_DIRNAME
            _write_json(stage3_dir / "NORMALIZED_STAGE3_COMPLETE.json", {"status": "complete"})
            _write_json(stage3_dir / "eval_results.json", {"paper_recall_curve_auc": 0.7})
            _write_json(stage4_dir / "training_stop.json", {"stop_reason": "smoke"})
            _write_json(stage4_dir / "generation_eval_metrics.json", [{"source": "all"}])

        statuses = write_status_report(
            run_dir,
            {"stage3_normalized_specter": True, "corrected_stage4_normalized_specter": True},
            layout="normalized_specter",
            ae_branch_mode=mode,
        )
        by_stage = {row["stage"]: row for row in statuses}
        expected_count = len(ae_branch_specs(mode))
        assert by_stage["stage3"]["ae_branch_mode"] == mode
        assert by_stage["stage4"]["ae_branch_mode"] == mode
        assert by_stage["stage3"]["completed_runs"] == expected_count
        assert by_stage["stage4"]["completed_runs"] == expected_count


def test_notebook_defaults_keep_downstream_branch_and_stage1b_controls_explicit() -> None:
    nb6a = json.loads((REPO_ROOT / "experiments/3dcnn/6 multi source stage3 stage4.ipynb").read_text())
    nb5 = json.loads((REPO_ROOT / "experiments/3dcnn/5 multi source autoencoder ablation.ipynb").read_text())
    source6a = "\n".join("".join(cell.get("source", "")) for cell in nb6a["cells"])
    source5 = "\n".join("".join(cell.get("source", "")) for cell in nb5["cells"])

    assert 'AE_BRANCH_MODE = "mixed_only"' in source6a
    assert "ae_branch_mode=AE_BRANCH_MODE" in source6a
    assert "AE_BRANCH_REQUIRED_SELECTION_KEYS" in source6a
    assert "required_selection_keys=tuple(AE_BRANCH_REQUIRED_SELECTION_KEYS)" in source6a
    assert "RUN_SUBPROCESS_STREAMING_SMOKE_TEST" in source6a
    assert "run_subprocess_streaming(cmd, cwd=REPO_DIR" in source6a
    assert "run_subprocess_streaming(cmd, env=env, cwd=REPO_DIR" in source6a
    assert "--strict-controlled-recipe" in source6a
    assert "RUN_STAGE1A_MIXED_PRETRAINING = True" in source5
    assert "RUN_STAGE1B_FINETUNING = False" in source5
    assert "if RUN_STAGE1B_FINETUNING and results:" in source5


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
        layout="normalized_specter",
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
        layout="normalized_specter",
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


def test_text_embedding_resolver_records_normalized_and_legacy_conventions(tmp_path: Path) -> None:
    normalized = resolve_text_embedding_cache("normalized_specter2", local_cache_dir=tmp_path, env_override=False)
    legacy = resolve_text_embedding_cache("legacy_specter2", local_cache_dir=tmp_path, env_override=False)

    assert normalized["cache_name"] == "specter2_stage3_stage4_emptycentered_unitnorm.pt"
    assert normalized["hf_path"] == "text_embeddings/specter2_stage3_stage4_emptycentered_unitnorm.pt"
    assert normalized["metadata_hf_path"] == "text_embeddings/specter2_stage3_stage4_emptycentered_unitnorm_metadata.json"
    assert normalized["preprocessing"] == "empty_string_centered_l2_unit_normalized"
    assert normalized["expect_unit_norm"] is True

    assert legacy["cache_name"] == "specter_text_cache.pt"
    assert legacy["hf_path"] == "text_embeddings/specter_text_cache.pt"
    assert legacy["preprocessing"] == "legacy_existing_cache_convention"
    assert legacy["expect_unit_norm"] is False


def test_legacy_cache_validation_does_not_require_unit_norm_and_records_checksum(tmp_path: Path) -> None:
    cache_path = tmp_path / "specter_text_cache.pt"
    torch.save({"alpha": torch.ones(768), "beta": torch.arange(768).float()}, cache_path)

    audit = validate_legacy_specter_cache(cache_path, required_texts={"alpha", "beta"})
    spec = resolve_text_embedding_cache("legacy_specter2", local_cache_dir=tmp_path, env_override=False)
    metadata = text_embedding_metadata_fields(spec, audit)

    assert audit["stats"]["dim"] == 768
    assert audit["stats"]["required_texts_present"] is True
    assert audit["stats"]["sha256"]
    assert metadata["text_embedding_cache_checksum"] == audit["stats"]["sha256"]
    assert metadata["text_embedding_cache_path"] == spec["local_cache_path"]
    assert metadata["expect_unit_norm"] is False


def test_convention_aware_stage3_stage4_folder_names_and_status_smoke(tmp_path: Path) -> None:
    cases = [
        ("normalized_specter2", NORMALIZED_STAGE3_DIRNAME, CORRECTED_STAGE4_DIRNAME),
        ("legacy_specter2", LEGACY_STAGE3_DIRNAME, CORRECTED_LEGACY_STAGE4_DIRNAME),
    ]
    for convention, expected_stage3_name, expected_stage4_name in cases:
        run_dir = tmp_path / convention
        suffix = text_embedding_convention_dir_suffix(convention)
        assert stage3_dirname_for_text_embedding_convention(convention) == expected_stage3_name
        assert corrected_stage4_dirname_for_text_embedding_convention(convention) == expected_stage4_name

        for spec in six_branch_specs():
            stage3_dir = stage_output_dir(
                run_dir,
                spec["domain"],
                spec["branch"],
                "stage3",
                text_embedding_convention=convention,
            )
            stage4_dir = stage_output_dir(
                run_dir,
                spec["domain"],
                spec["branch"],
                "stage4",
                text_embedding_convention=convention,
            )
            assert stage3_dir.name == expected_stage3_name
            assert stage4_dir.name == expected_stage4_name
            _write_json(stage3_dir / "eval_results.json", {"paper_recall_curve_auc": 0.7})
            (stage3_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            (stage3_dir / "checkpoints" / NORMALIZED_STAGE3_CHECKPOINT).write_bytes(b"stage3")
            _write_json(stage4_dir / "training_stop.json", {"stop_reason": "smoke"})
            _write_json(stage4_dir / "generation_eval_metrics.json", [{"source": "all"}])
            (stage4_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            (stage4_dir / "checkpoints" / CORRECTED_STAGE4_CHECKPOINT).write_bytes(b"stage4")

        statuses = write_status_report(
            run_dir,
            {f"stage3_{suffix}": True, f"corrected_stage4_{suffix}": True},
            text_embedding_convention=convention,
        )
        by_stage = {row["stage"]: row for row in statuses}
        assert by_stage["stage3"]["status"] == "ran successfully"
        assert by_stage["stage4"]["status"] == "ran successfully"
        assert by_stage["stage3"]["completed_runs"] == 6
        assert by_stage["stage4"]["completed_runs"] == 6

        if convention == "legacy_specter2":
            misleading_dirs = [path for path in run_dir.glob("**/*normalized*") if path.is_dir()]
            assert misleading_dirs == []


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
    from atlas_free_cnn.training.train_ale_cnn import ALETrainer, SAVE_LEGACY_STAGE3_CHECKPOINT_ALIASES

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

    if not SAVE_LEGACY_STAGE3_CHECKPOINT_ALIASES:
        assert not (ckpt_dir / "best_ale_cnn.pt").exists(), "legacy alias best_ale_cnn.pt must not exist by default"
        assert not (ckpt_dir / "last_ale_cnn.pt").exists(), "legacy alias last_ale_cnn.pt must not exist by default"


def test_stage4_canonical_checkpoint_names() -> None:
    assert STAGE4_PRIMARY_SPATIAL_CHECKPOINT == "best_val_top5_dice.pt"
    assert STAGE4_SEMANTIC_CHECKPOINT == "best_val_generation_normalized_auc.pt"
    assert CORRECTED_STAGE4_CHECKPOINT == "best_val_generation_normalized_auc.pt"
    assert STAGE4_SPATIAL_CORR_CHECKPOINT == "best_val_spatial_corr.pt"


def test_stage4_trainer_produces_only_canonical_checkpoints(tmp_path: Path) -> None:
    """Verify the Stage 4 CheckpointManager only saves canonical files by default."""
    from atlas_free_cnn.training.checkpointing import CheckpointManager
    from atlas_free_cnn.training.train_text_to_brain import SAVE_LEGACY_CHECKPOINT_ALIASES

    assert not SAVE_LEGACY_CHECKPOINT_ALIASES, "SAVE_LEGACY_CHECKPOINT_ALIASES must default to False"

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


def test_stage3_detection_prioritises_canonical_over_legacy_aliases(tmp_path: Path) -> None:
    """detect_stage_status prefers best_val_normalized_recall_auc.pt and does not require old aliases."""
    from atlas_free_cnn.pipeline_outputs import detect_stage_status

    stage3_dir = tmp_path / "stage3_normalized_specter"
    ckpt_dir = stage3_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)

    (stage3_dir / "NORMALIZED_STAGE3_COMPLETE.json").write_text('{"status":"complete"}')
    (stage3_dir / "eval_results.json").write_text('{"paper_recall_curve_auc": 0.72}')
    (ckpt_dir / NORMALIZED_STAGE3_CHECKPOINT).write_bytes(b"canonical")

    status = detect_stage_status("stage3", requested=True, stage_dir=stage3_dir)
    assert status["status"] == "ran successfully", f"expected 'ran successfully', got {status}"
    assert status["checkpoints_in_export_zip"] is True

    no_ckpt_dir = tmp_path / "stage3_no_ckpt" / "checkpoints"
    no_ckpt_dir.mkdir(parents=True)
    stage3_no_ckpt = tmp_path / "stage3_no_ckpt"
    (stage3_no_ckpt / "NORMALIZED_STAGE3_COMPLETE.json").write_text('{"status":"complete"}')
    (stage3_no_ckpt / "eval_results.json").write_text('{"paper_recall_curve_auc": 0.70}')

    status2 = detect_stage_status("stage3", requested=True, stage_dir=stage3_no_ckpt)
    assert status2["status"] == "ran successfully", "marker + metrics alone should be sufficient without checkpoint"
    assert status2["checkpoints_in_export_zip"] is False


def test_stage3_detection_accepts_legacy_alias_as_fallback(tmp_path: Path) -> None:
    """Runs with only best_ale_cnn.pt (pre-rename) should still be detected as completed."""
    from atlas_free_cnn.pipeline_outputs import detect_stage_status

    stage3_dir = tmp_path / "stage3_normalized_specter"
    ckpt_dir = stage3_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)

    (ckpt_dir / "best_ale_cnn.pt").write_bytes(b"legacy")
    (stage3_dir / "eval_results.json").write_text('{"paper_recall_curve_auc": 0.68}')

    status = detect_stage_status("stage3", requested=True, stage_dir=stage3_dir)
    assert status["status"] == "ran successfully", "legacy best_ale_cnn.pt should still be accepted as fallback"
    assert status["checkpoints_in_export_zip"] is True


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


def test_convention_aware_stage3_checkpoint_names_do_not_mix() -> None:
    """normalized mode dirs must not contain legacy names and vice versa."""
    from atlas_free_cnn.notebook_utils import (
        NORMALIZED_STAGE3_DIRNAME,
        LEGACY_STAGE3_DIRNAME,
        CORRECTED_STAGE4_DIRNAME,
        CORRECTED_LEGACY_STAGE4_DIRNAME,
    )
    assert "normalized" in NORMALIZED_STAGE3_DIRNAME
    assert "legacy" in LEGACY_STAGE3_DIRNAME
    assert "normalized" in CORRECTED_STAGE4_DIRNAME
    assert "legacy" in CORRECTED_LEGACY_STAGE4_DIRNAME
    assert "normalized" not in LEGACY_STAGE3_DIRNAME
    assert "normalized" not in CORRECTED_LEGACY_STAGE4_DIRNAME


def test_mixed_only_ae_branch_mode_expects_exactly_3_branches() -> None:
    specs = ae_branch_specs("mixed_only")
    assert len(specs) == 3
    run_names = [s["run"] for s in specs]
    assert "mixed_stage1a_on_pubmed" in run_names
    assert "mixed_stage1a_on_nilearn" in run_names
    assert "mixed_stage1a_on_neurovault" in run_names


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
