from __future__ import annotations

import json

import pytest
import torch

from neurovlm.experiments.stage4_semantic_bridge import (
    BRIDGE_ARCHITECTURES,
    BRIDGE_PATHS,
    BridgeCheckpointManager,
    BridgeLossConfig,
    bridge_architecture_record,
    bridge_latent_metrics,
    build_bridge_model,
    compute_bridge_loss,
    fixed_derangement,
    semantic_alignment_metrics,
    validate_semantic_embeddings,
)


@pytest.mark.parametrize("architecture", BRIDGE_ARCHITECTURES)
@pytest.mark.parametrize("path", BRIDGE_PATHS)
def test_all_semantic_bridge_paths_emit_raw_384d_latents(
    path: str,
    architecture: str,
) -> None:
    model = build_bridge_model(path, architecture=architecture)
    raw_text = torch.randn(3, 768)
    text_semantic = torch.nn.functional.normalize(torch.randn(3, 384), dim=1)
    brain_semantic = torch.nn.functional.normalize(torch.randn(3, 384), dim=1)
    if path == "direct_baseline":
        output = model(raw_text)
    elif path == "stage3_text_bridge":
        output = model(text_semantic)
    elif path == "stage3_brain_bridge_oracle":
        output = model(brain_semantic)
    elif path == "shared_bridge_dual_supervision":
        output = model(text_semantic, brain_semantic)
    else:
        output = model(raw_text, text_semantic)
    outputs = output if isinstance(output, tuple) else (output,)
    assert all(value.shape == (3, 384) for value in outputs)
    record = bridge_architecture_record(path, architecture, model)
    assert record["final_output_transform"] is None
    assert record["decoder_input_convention"] == "raw_stage1_ae_latent"


def test_primary_loss_is_exactly_raw_plus_decoded_mse() -> None:
    prediction = torch.tensor([[1.0, 2.0]])
    target = torch.zeros_like(prediction)
    prediction_volume = torch.tensor([[[1.0, 0.0]]])
    target_volume = torch.zeros_like(prediction_volume)
    loss = compute_bridge_loss(
        prediction,
        target,
        prediction_volume,
        target_volume,
        training_latent_mean=torch.zeros(2),
        training_latent_std=torch.tensor([2.0, 4.0]),
        config=BridgeLossConfig(
            variant="primary_raw_decoded",
            latent_weight=1.0,
            decoded_weight=1.0,
            cosine_weight=99.0,
            norm_weight=99.0,
        ),
    )
    assert torch.allclose(
        loss.total,
        loss.raw_latent_mse + loss.decoded_volume_mse,
    )


def test_secondary_loss_axes_are_explicit() -> None:
    configs = [
        BridgeLossConfig(variant="standardized_decoded"),
        BridgeLossConfig(variant="standardized_cosine_decoded"),
        BridgeLossConfig(variant="standardized_cosine_norm_decoded"),
    ]
    assert all(
        config.effective_dict()["architecture_loss_axis"]
        == "separate_loss_sensitivity"
        for config in configs
    )


def test_fixed_derangement_is_reproducible_and_has_no_self_pairs() -> None:
    first = fixed_derangement(31, 8)
    second = fixed_derangement(31, 8)
    assert torch.equal(first, second)
    assert torch.equal(first.sort().values, torch.arange(31))
    assert not bool((first == torch.arange(31)).any())


def test_semantic_validation_and_matched_shuffled_alignment() -> None:
    text = torch.eye(4)
    text = torch.nn.functional.pad(text, (0, 380))
    brain = text.clone()
    validation = validate_semantic_embeddings(text, label="text")
    alignment = semantic_alignment_metrics(
        text,
        brain,
        shuffled_indices=torch.tensor([1, 2, 3, 0]),
    )
    assert validation["dimension"] == 384
    assert alignment["stage3_text_brain_matched_cosine"] == pytest.approx(1.0)
    assert alignment["stage3_text_brain_shuffled_cosine"] == pytest.approx(0.0)


def test_bridge_latent_metrics_include_requested_variance_and_r_squared() -> None:
    target = torch.randn(12, 5)
    metrics, rows = bridge_latent_metrics(
        target,
        target.clone(),
        training_mean=target.mean(0),
        training_std=target.std(0, unbiased=False),
        nearest_reference=target,
        distance_device="cpu",
    )
    assert metrics["raw_latent_mse"] == pytest.approx(0.0)
    assert metrics["standardized_latent_mse"] == pytest.approx(0.0)
    assert metrics["global_explained_variance"] == pytest.approx(1.0)
    assert metrics["mean_per_dimension_r_squared"] == pytest.approx(1.0)
    assert metrics["nearest_real_latent_distance"] == pytest.approx(0.0)
    assert len(rows) == 5
    assert all(row["r_squared"] == pytest.approx(1.0) for row in rows)


def test_checkpoint_manager_saves_all_objectives_and_rejects_new_binding(
    tmp_path,
) -> None:
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    architecture = {"name": "tiny"}
    config = {"seed": 4}
    binding = {"stage1": "ae", "stage3": "semantic", "split": "train"}
    manager = BridgeCheckpointManager(
        tmp_path / "run",
        binding=binding,
        effective_config=config,
        architecture=architecture,
    )
    saved = manager.save_epoch(
        model,
        optimizer,
        epoch=1,
        metrics={
            "top5_dice": 0.1,
            "spatial_corr": 0.2,
            "global_explained_variance": 0.3,
            "semantic_normalized_auc": 0.4,
        },
    )
    assert set(saved) == {
        "top5_dice",
        "spatial_correlation",
        "latent_explained_variance",
        "semantic_normalized_auc",
        "last",
    }
    payload = manager.load("last", model=model, optimizer=optimizer)
    assert payload["binding"] == binding
    manifest = json.loads(manager.manifest_path.read_text())
    assert set(manifest["checkpoints"]) == set(saved)

    mismatched = BridgeCheckpointManager(
        tmp_path / "run",
        binding={**binding, "split": "different"},
        effective_config=config,
        architecture=architecture,
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        mismatched.load("last", model=model)
