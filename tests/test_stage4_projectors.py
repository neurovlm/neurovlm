from __future__ import annotations

import pytest
import torch

from neurovlm.experiments.stage4_projectors import (
    PROJECTOR_NAMES,
    ProjectorBuildConfig,
    architecture_record,
    build_scheduler,
    build_stage4_projector,
    gradient_diagnostics,
    pareto_front,
    projector_checkpoint_metadata,
    projector_definition,
    validate_projector_checkpoint_metadata,
)


@pytest.mark.parametrize("name", PROJECTOR_NAMES)
def test_every_experimental_projector_emits_unconstrained_raw_384d_latent(
    name: str,
) -> None:
    torch.manual_seed(4)
    config = ProjectorBuildConfig(name=name, dropout=0.1)
    projector = build_stage4_projector(name, dropout=0.1)
    output = projector(torch.randn(3, 768))
    assert output.shape == (3, 384)
    assert projector_definition(config)["final_output_transform"] is None
    assert isinstance(list(projector.modules())[-1], torch.nn.Linear)


def test_residual_2048_supports_one_or_two_blocks() -> None:
    one = build_stage4_projector("residual_2048", residual_2048_blocks=1)
    two = build_stage4_projector("residual_2048", residual_2048_blocks=2)
    assert len(one.blocks) == 1
    assert len(two.blocks) == 2
    assert sum(p.numel() for p in two.parameters()) > sum(
        p.numel() for p in one.parameters()
    )


def test_architecture_record_has_parameter_and_activation_memory() -> None:
    config = ProjectorBuildConfig(name="deep_mlp", dropout=0.0)
    projector = build_stage4_projector(config.name)
    record = architecture_record(
        config,
        projector,
        batch_size=64,
        dtype=torch.bfloat16,
    )
    assert record["parameter_count"] == sum(p.numel() for p in projector.parameters())
    assert record["activation_memory"]["forward_bytes"] > 0
    assert record["activation_memory"]["training_saved_bytes_estimate"] == (
        3 * record["activation_memory"]["forward_bytes"]
    )


def test_checkpoint_metadata_rejects_architecture_binding_config_and_state_changes() -> None:
    config = ProjectorBuildConfig(name="wider_mlp", dropout=0.0)
    binding = {"ae_sha256": "ae", "split_sha256": "split"}
    effective = {"learning_rate": 3e-4, "weight_decay": 0.0}
    projector = build_stage4_projector(config.name)
    metadata = projector_checkpoint_metadata(
        config,
        projector,
        binding=binding,
        effective_config=effective,
    )
    validate_projector_checkpoint_metadata(
        metadata,
        config,
        binding=binding,
        effective_config=effective,
        module=projector,
    )
    with pytest.raises(ValueError, match="binding_sha256"):
        validate_projector_checkpoint_metadata(
            metadata,
            config,
            binding={**binding, "split_sha256": "different"},
            effective_config=effective,
        )
    with torch.no_grad():
        next(projector.parameters()).add_(1)
    with pytest.raises(ValueError, match="projector_state_sha256"):
        validate_projector_checkpoint_metadata(
            metadata,
            config,
            binding=binding,
            effective_config=effective,
            module=projector,
        )


def test_gradient_diagnostics_include_zero_percent_by_layer() -> None:
    projector = build_stage4_projector("retained_mlp")
    projector(torch.ones(2, 768)).sum().backward()
    summary, rows = gradient_diagnostics(projector)
    assert summary["total_gradient_norm"] > 0
    assert rows
    assert all("zero_gradient_percent" in row for row in rows)


def test_scheduler_factory_covers_all_requested_policies() -> None:
    for name in (
        "constant",
        "warmup_cosine",
        "reduce_on_plateau",
        "cosine_restarts",
    ):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = build_scheduler(optimizer, name, epochs=20)
        assert (scheduler is None) == (name == "constant")


def test_pareto_front_preserves_tradeoffs_and_rejects_nonfinite_rows() -> None:
    rows = [
        {"dice": 0.8, "mse": 0.4},
        {"dice": 0.7, "mse": 0.3},
        {"dice": 0.7, "mse": 0.5},
        {"dice": float("nan"), "mse": 0.1},
    ]
    assert pareto_front(rows, {"dice": "max", "mse": "min"}) == [
        True,
        True,
        False,
        False,
    ]
