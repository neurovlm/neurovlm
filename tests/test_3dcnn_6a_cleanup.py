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
    locked_stage1_checkpoint_selection,
    resolve_text_embedding_cache,
    six_branch_specs,
    text_embedding_metadata_fields,
    validate_legacy_specter_cache,
    validate_normalized_specter_cache,
)
from atlas_free_cnn.pipeline_outputs import write_status_report
from atlas_free_cnn.training.model_wrappers import build_generative_text_to_ae_latent


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


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
    assert metadata["expect_unit_norm"] is False


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
