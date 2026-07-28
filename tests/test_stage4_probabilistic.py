from __future__ import annotations

import copy

import pytest
import torch

from neurovlm.experiments.stage4_probabilistic import (
    ConditionalLatentVAE,
    ConditionalVAEConfig,
    LatentStandardization,
    checkpoint_payload,
    kl_per_dimension,
    load_checkpoint,
    validate_provenance,
)


def _provenance() -> dict:
    return {
        "autoencoder": {"sha256": "ae", "encoder_state_sha256": "encoder"},
        "text_cache": {"sha256": "text", "normalization": "l2"},
        "splits": {
            "train": {"ordered_rows_sha256": "train"},
            "val": {"ordered_rows_sha256": "val"},
            "test": {"ordered_rows_sha256": "test"},
        },
        "branch": {"branch": "mixed_to_pubmed", "domain": "pubmed"},
    }


@pytest.mark.parametrize("u_dim", [32, 64, 128])
def test_conditional_vae_shapes_and_exact_standardization_inverse(u_dim: int) -> None:
    model = ConditionalLatentVAE(ConditionalVAEConfig(u_dim=u_dim))
    condition = torch.randn(3, 768)
    target = torch.randn(3, 384)
    output = model(condition, target)
    assert output["standardized_prediction"].shape == (3, 384)
    assert output["mu"].shape == (3, u_dim)
    assert output["logvar"].shape == (3, u_dim)
    assert model.sample_prior(condition, k=8, seed=7).shape == (3, 8, 384)

    raw = torch.randn(10, 384)
    standardization = LatentStandardization.fit(raw)
    torch.testing.assert_close(
        standardization.inverse(standardization.transform(raw)),
        raw,
        rtol=1e-5,
        atol=1e-6,
    )


def test_reparameterization_is_seeded_and_has_expected_zero_variance_limit() -> None:
    mu = torch.randn(4, 32)
    logvar = torch.full_like(mu, -40.0)
    first = ConditionalLatentVAE.reparameterize(
        mu, logvar, generator=torch.Generator().manual_seed(9)
    )
    second = ConditionalLatentVAE.reparameterize(
        mu, logvar, generator=torch.Generator().manual_seed(9)
    )
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first, mu, atol=1e-7, rtol=0)


def test_kl_matches_closed_form() -> None:
    mu = torch.zeros(2, 5)
    logvar = torch.zeros_like(mu)
    torch.testing.assert_close(kl_per_dimension(mu, logvar), torch.zeros_like(mu))

    mu = torch.ones(2, 5)
    expected = torch.full_like(mu, 0.5)
    torch.testing.assert_close(kl_per_dimension(mu, logvar), expected)


def test_prior_sampling_is_deterministic_under_fixed_seed() -> None:
    model = ConditionalLatentVAE(ConditionalVAEConfig(u_dim=32)).eval()
    condition = torch.randn(2, 768)
    first = model.sample_prior(condition, k=16, seed=123)
    second = model.sample_prior(condition, k=16, seed=123)
    third = model.sample_prior(condition, k=16, seed=124)
    torch.testing.assert_close(first, second)
    assert not torch.equal(first, third)


def test_checkpoint_reload_and_provenance_validation(tmp_path) -> None:
    model = ConditionalLatentVAE(ConditionalVAEConfig(u_dim=64))
    standardization = LatentStandardization.fit(torch.randn(20, 384))
    provenance = _provenance()
    path = tmp_path / "checkpoint.pt"
    torch.save(
        checkpoint_payload(
            model,
            standardization=standardization,
            provenance=provenance,
            epoch=3,
            global_step=17,
            metrics={"val_top5_dice": 0.2},
        ),
        path,
    )
    reloaded, reloaded_standardization, payload = load_checkpoint(
        path, expected_provenance=provenance
    )
    assert payload["epoch"] == 3
    for left, right in zip(model.parameters(), reloaded.parameters(), strict=True):
        torch.testing.assert_close(left, right)
    torch.testing.assert_close(standardization.mean, reloaded_standardization.mean)

    mismatched = copy.deepcopy(provenance)
    mismatched["splits"]["test"]["ordered_rows_sha256"] = "other"
    with pytest.raises(ValueError, match="provenance"):
        load_checkpoint(path, expected_provenance=mismatched)
    with pytest.raises(ValueError, match="provenance"):
        validate_provenance(provenance, mismatched)

    recorded_run = {
        **provenance,
        "run_identity": {"beta_max": 0.01, "pairing": "matched"},
        "latent_standardization": standardization.metadata(),
    }
    changed_run = copy.deepcopy(recorded_run)
    changed_run["run_identity"]["beta_max"] = 0.1
    with pytest.raises(ValueError, match="provenance"):
        validate_provenance(recorded_run, changed_run)
