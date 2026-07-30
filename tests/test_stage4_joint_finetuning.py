from __future__ import annotations

import copy

import pytest
import torch

from neurovlm.ale_cnn import ALE3DCNNAutoEncoder
from neurovlm.cnn import GenerativeTextToAELatent
from neurovlm.experiments.stage4_joint_finetuning import (
    JOINT_FINETUNING_VARIANTS,
    JointLossWeights,
    ae_retention_decision,
    assert_frozen_parameters_unchanged,
    checkpoint_binding,
    compute_joint_loss,
    configure_trainable_variant,
    optimizer_group_settings,
    parameter_snapshot,
    validate_checkpoint_binding,
)
from neurovlm.pipelines import sha256_state_dict


def _models():
    ae = ALE3DCNNAutoEncoder(
        output_shape=(4, 4, 4),
        base_channels=2,
        num_blocks=1,
        latent_dim=4,
        dropout=0.0,
    )
    projector = GenerativeTextToAELatent(in_dim=6, hidden_dim=8, latent_dim=4)
    return ae, projector


@pytest.mark.parametrize("variant", JOINT_FINETUNING_VARIANTS)
def test_variant_freezes_every_unselected_parameter_and_groups_are_disjoint(
    variant: str,
) -> None:
    ae, projector = _models()
    groups = configure_trainable_variant(ae, projector, variant)
    names = [name for values in groups.values() for name, _ in values]
    assert len(names) == len(set(names))
    assert groups["projector"]
    assert all(parameter.requires_grad for _, parameter in groups["projector"])
    assert all(
        parameter.requires_grad == (name in names)
        for name, parameter in ae.named_parameters()
    )
    if variant == "projector_only_baseline":
        assert not groups["decoder"]
        assert not groups["encoder_head"]


def test_replay_path_trains_selected_decoder_but_preserves_frozen_state() -> None:
    torch.manual_seed(3)
    adapted, projector = _models()
    original = copy.deepcopy(adapted).eval()
    for parameter in original.parameters():
        parameter.requires_grad_(False)
    original_identity = sha256_state_dict(original)
    groups = configure_trainable_variant(
        adapted, projector, "projector_plus_decoder_output"
    )
    settings, _ = optimizer_group_settings(groups)
    optimizer = torch.optim.AdamW(settings)
    initial_state = {
        name: value.detach().cpu().clone()
        for name, value in adapted.state_dict().items()
    }
    initial_parameters = parameter_snapshot(groups)
    target = torch.rand(3, 1, 4, 4, 4)
    text = torch.rand(3, 6)
    result = compute_joint_loss(
        projector,
        adapted,
        original,
        text,
        target,
        weights=JointLossWeights(
            generation_latent=0.0,
            generation_image=0.0,
            replay=1.0,
            distill=0.0,
        ),
        initial_parameters=initial_parameters,
    )
    assert result.replay_volume.shape == target.shape
    assert result.components["AE_reconstruction_replay"].requires_grad
    result.total.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for values in groups.values()
        for _, parameter in values
    )
    optimizer.step()
    unfrozen_ae = [
        name for name, parameter in adapted.named_parameters() if parameter.requires_grad
    ]
    assert_frozen_parameters_unchanged(adapted, initial_state, unfrozen_ae)
    assert any(
        not torch.equal(adapted.state_dict()[name].cpu(), initial_state[name])
        for name in unfrozen_ae
    )
    assert sha256_state_dict(original) == original_identity


def test_checkpoint_binding_rejects_identity_changes() -> None:
    ae, projector = _models()
    original = copy.deepcopy(ae)
    groups = configure_trainable_variant(
        ae, projector, "projector_plus_decoder_output"
    )
    _, settings = optimizer_group_settings(groups)
    common = {
        "original_ae_identity": {"sha256": "released-ae"},
        "starting_ae": original,
        "adapted_ae": ae,
        "projector": projector,
        "text_cache_identity": {"sha256": "text"},
        "split_fingerprints": {"train": "train", "val": "val", "test": "test"},
        "unfrozen_parameter_names": [
            name for values in groups.values() for name, _ in values
        ],
        "loss_weights": {"replay": 4.0},
        "optimizer_groups": settings,
    }
    recorded = checkpoint_binding(**common)
    expected = checkpoint_binding(**common)
    validate_checkpoint_binding(recorded, expected)
    changed = {**expected, "text_cache_identity": {"sha256": "different"}}
    with pytest.raises(ValueError, match="text_cache_identity"):
        validate_checkpoint_binding(recorded, changed)


def test_ae_retention_rule_is_direction_aware_and_reports_all_thresholds() -> None:
    original = {"top5_dice": 0.50, "spatial_corr": 0.40, "mse": 0.10}
    two_percent = {"top5_dice": 0.49, "spatial_corr": 0.392, "mse": 0.102}
    result = ae_retention_decision(original, two_percent)
    assert result["top5_dice_degradation_percent"] == pytest.approx(2.0)
    assert not result["satisfies_1_percent"]
    assert result["satisfies_2_percent"]
    assert result["satisfies_5_percent"]
    assert result["safe"]
    rejected = ae_retention_decision(
        original,
        {"top5_dice": 0.47, "spatial_corr": 0.38, "mse": 0.11},
    )
    assert rejected["action"] == "reject_and_stop"
