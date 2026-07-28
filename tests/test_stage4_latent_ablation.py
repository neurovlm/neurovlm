from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from neurovlm.atlas_free_text import AtlasFreeTextEmbeddingLookup
from neurovlm.experiments.stage4_latent_ablation import (
    AblationCheckpointManager,
    LatentTransform,
    Stage4AblationTrainConfig,
    compute_stage4_ablation_loss,
    encode_stage1_latents,
    latent_ablation_metrics,
    split_fingerprint,
    text_cache_identity,
    train_stage4_ablation,
    validate_checkpoint_binding,
)
from neurovlm.pipelines import sha256_file


def _latents(n: int = 64, d: int = 8) -> torch.Tensor:
    generator = torch.Generator().manual_seed(7)
    basis = torch.randn(n, d, generator=generator)
    scales = torch.linspace(0.02, 2.0, d)
    return basis * scales + torch.linspace(-1.0, 1.0, d)


@pytest.mark.parametrize("kind", ["raw", "standardized", "full_whitening"])
def test_lossless_transforms_round_trip(kind: str) -> None:
    latents = _latents()
    transform = LatentTransform.fit(latents, kind, epsilon=1e-5)
    restored = transform.inverse(transform.transform(latents))
    assert transform.active_dim == latents.shape[1]
    assert torch.allclose(restored, latents, atol=2e-5, rtol=2e-5)
    reloaded = LatentTransform.from_payload(transform.to_payload())
    assert reloaded.metadata()["state_sha256"] == transform.metadata()["state_sha256"]
    assert torch.allclose(reloaded.inverse(reloaded.transform(latents)), restored)


def test_pca_99_5_keeps_384_output_convention_and_reports_approximation() -> None:
    generator = torch.Generator().manual_seed(9)
    leading = torch.randn(128, 3, generator=generator)
    trailing = 1e-4 * torch.randn(128, 5, generator=generator)
    latents = torch.cat([leading, trailing], dim=1)
    transform = LatentTransform.fit(latents, "pca_99_5", retained_variance=0.995)
    represented = transform.transform(latents)
    assert represented.shape == latents.shape
    assert transform.active_dim < latents.shape[1]
    assert torch.count_nonzero(represented[:, transform.active_dim :]) == 0
    assert transform.inverse(represented).shape == latents.shape
    assert transform.metadata()["projector_output_dim"] == latents.shape[1]


@pytest.mark.parametrize(
    ("variant", "kind"),
    [
        ("baseline_raw", "raw"),
        ("standardized_mse", "standardized"),
        ("standardized_cosine", "standardized"),
        ("standardized_cosine_norm", "standardized"),
        ("full_whitening", "full_whitening"),
        ("pca_99_5", "pca_99_5"),
    ],
)
def test_every_loss_decodes_raw_latents_with_fixed_projector_width(
    variant: str, kind: str
) -> None:
    latent = _latents(n=16)
    transform = LatentTransform.fit(latent, kind)

    class RecordingDecoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.received = None

        def forward(self, value):
            self.received = value
            return value[:, :, None, None, None]

    decoder = RecordingDecoder()
    projector_output = transform.transform(latent).clone().requires_grad_(True)
    target_volume = latent[:, :, None, None, None]
    output = compute_stage4_ablation_loss(
        variant,
        projector_output,
        latent,
        target_volume,
        transform=transform,
        decoder=decoder,
    )
    assert decoder.received is not None
    assert decoder.received.shape == latent.shape
    assert output.raw_prediction_latent.shape[1] == latent.shape[1]
    output.total.backward()
    assert projector_output.grad is not None


def test_latent_metric_schema_and_per_dimension_rows() -> None:
    target = _latents()
    prediction = target * 0.5
    transform = LatentTransform.fit(target, "standardized")
    metrics, rows = latent_ablation_metrics(
        target,
        prediction,
        transform=transform,
        nearest_reference=target[:20],
        prediction_chunk_size=7,
        reference_chunk_size=8,
    )
    assert {
        "raw_latent_mse",
        "transformed_latent_mse",
        "latent_cosine_similarity",
        "predicted_target_latent_variance_ratio",
        "predicted_target_latent_norm_ratio",
        "global_explained_variance",
        "mean_per_dimension_r_squared",
        "highest_target_variance_quartile_mean_r_squared",
        "target_prediction_per_dimension_variance_correlation",
        "distance_to_nearest_real_ae_latent",
        "distance_to_mean_target_latent",
    } <= metrics.keys()
    assert len(rows) == target.shape[1]
    assert metrics["predicted_target_latent_variance_ratio"] == pytest.approx(0.25)


class _Dataset:
    split = "train"
    rows = [
        {
            "map_id": "map-0",
            "split": "train",
            "source": "pubmed",
            "positive_texts": [{"text_id": "text-0", "text": "zero"}],
        },
        {
            "map_id": "map-1",
            "split": "train",
            "source": "pubmed",
            "positive_texts": [{"text_id": "text-1", "text": "one"}],
        },
    ]
    _tensor_indices = [4, 2]


def test_split_and_text_cache_identities_are_order_sensitive() -> None:
    fingerprint = split_fingerprint(_Dataset())
    assert fingerprint["n"] == 2
    assert fingerprint["split"] == "train"
    embeddings = torch.zeros(2, 768)
    embeddings[0, 0] = 1
    embeddings[1, 1] = 1
    lookup = AtlasFreeTextEmbeddingLookup(embeddings, ["text-0", "text-1"])
    identity = text_cache_identity(lookup)
    assert identity["dimension"] == 768
    reversed_lookup = AtlasFreeTextEmbeddingLookup(
        embeddings.flip(0), ["text-1", "text-0"]
    )
    assert (
        text_cache_identity(reversed_lookup)["ordered_text_ids_sha256"]
        != identity["ordered_text_ids_sha256"]
    )


def test_checkpoint_manager_binds_and_resumes_exact_state(tmp_path: Path) -> None:
    binding = {
        "ae": {"state_sha256": "ae"},
        "text_cache": {"tensor_sha256": "text"},
        "transform": {"state_sha256": "transform"},
        "splits": {"train": "fingerprint"},
    }
    architecture = {"projector": [768, 512, 384], "decoder_input": 384}
    config = {"variant": "standardized_mse", "seed": 42}
    projector = nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=1e-3)
    manager = AblationCheckpointManager(
        tmp_path,
        binding=binding,
        architecture=architecture,
        config=config,
    )
    manager.save_last(projector, optimizer, epoch=1, metrics={"val_top5_dice": 0.2})
    manager.update_best(
        "val_top5_dice",
        0.2,
        projector,
        optimizer,
        epoch=1,
        metrics={"val_top5_dice": 0.2},
    )
    original = {name: value.clone() for name, value in projector.state_dict().items()}
    with torch.no_grad():
        for parameter in projector.parameters():
            parameter.add_(1)
    payload = manager.resume(projector, optimizer)
    assert payload is not None and payload["epoch"] == 1
    assert all(torch.equal(projector.state_dict()[name], value) for name, value in original.items())
    manifest = json.loads((tmp_path / "checkpoint_manifest.json").read_text())
    assert {"last", "top5_dice"} <= manifest["checkpoints"].keys()

    with pytest.raises(ValueError, match="binding mismatch"):
        AblationCheckpointManager(
            tmp_path,
            binding={**binding, "ae": {"state_sha256": "different"}},
            architecture=architecture,
            config=config,
        )
    with pytest.raises(ValueError, match="provenance binding mismatch"):
        validate_checkpoint_binding({"a": 1}, {"a": 2})


def test_checkpoint_metadata_is_weights_only_safe_and_legacy_torch_version_loads(
    tmp_path: Path,
) -> None:
    binding = {
        "text_cache": {
            "metadata": {"torch_version": torch.__version__},
            "state_sha256": "text",
        }
    }
    architecture = {"projector": [4, 3]}
    config = {"variant": "baseline_raw"}
    projector = nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=1e-3)
    manager = AblationCheckpointManager(
        tmp_path,
        binding=binding,
        architecture=architecture,
        config=config,
    )
    path = manager.save_last(
        projector,
        optimizer,
        epoch=1,
        metrics={"val_top5_dice": 0.2},
    )

    # Newly written metadata contains an exact built-in str and loads with the
    # restricted unpickler without an allowlist.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    recorded_version = payload["binding"]["text_cache"]["metadata"]["torch_version"]
    assert type(recorded_version) is str

    # Recreate the legacy Colab condition while keeping the logical binding
    # hash unchanged, then exercise the manager's narrow compatibility path.
    payload["binding"]["text_cache"]["metadata"]["torch_version"] = torch.__version__
    torch.save(payload, path)
    manifest_path = tmp_path / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["checkpoints"]["last"]["sha256"] = sha256_file(path)
    manifest["checkpoints"]["last"]["size"] = path.stat().st_size
    manifest_path.write_text(json.dumps(manifest))
    resumed = manager.resume(projector, optimizer, path=path)
    assert resumed is not None
    assert str(
        resumed["binding"]["text_cache"]["metadata"]["torch_version"]
    ) == str(torch.__version__)


class _TinyDecoder(nn.Module):
    latent_dim = 384
    output_shape = (2, 2, 2)

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(384, 8)

    def forward(self, value):
        return self.linear(value).reshape(-1, 1, 2, 2, 2)


class _TinyAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Flatten(1), nn.Linear(8, 384))
        self.decoder = _TinyDecoder()


class _TinyPairs(torch.utils.data.Dataset):
    def __init__(self, split: str) -> None:
        self.split = split
        self.rows = [
            {
                "map_id": f"{split}-{index}",
                "split": split,
                "source": "pubmed",
                "positive_texts": [
                    {"text_id": f"text-{index}", "text": f"text {index}"}
                ],
            }
            for index in range(4)
        ]
        self._tensor_indices = list(range(4))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return {
            **self.rows[index],
            "volume": torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)
            / (8 + index),
            "dataset_index": index,
            "tensor_index": index,
            "metadata": self.rows[index],
        }


def test_experiment_trainer_writes_resume_safe_metrics_and_checkpoints(
    tmp_path: Path,
) -> None:
    autoencoder = _TinyAE().eval()
    for parameter in autoencoder.parameters():
        parameter.requires_grad_(False)
    train = _TinyPairs("train")
    val = _TinyPairs("val")
    embeddings = torch.zeros(4, 768)
    embeddings[range(4), range(4)] = 1
    lookup = AtlasFreeTextEmbeddingLookup(
        embeddings,
        [f"text-{index}" for index in range(4)],
    )
    training_latents = encode_stage1_latents(
        autoencoder,
        train,
        lookup,
        device="cpu",
        batch_size=2,
        target_shape=(2, 2, 2),
    )
    transform = LatentTransform.fit(training_latents, "raw")
    result = train_stage4_ablation(
        Stage4AblationTrainConfig(
            variant="baseline_raw",
            epochs=1,
            batch_size=2,
            eval_batch_size=2,
            amp=False,
            early_stopping_patience=None,
        ),
        run_dir=tmp_path,
        autoencoder=autoencoder,
        transform=transform,
        training_latents=training_latents,
        train_dataset=train,
        validation_dataset=val,
        lookup=lookup,
        binding={"identity": "tiny"},
        device="cpu",
        target_shape=(2, 2, 2),
    )
    assert result["epochs_completed"] == 1
    assert (tmp_path / "training_history.csv").is_file()
    assert (tmp_path / "validation_metrics.csv").is_file()
    assert (tmp_path / "per_dimension_latent_diagnostics.csv").is_file()
    assert (tmp_path / "checkpoints" / "last.pt").is_file()
    assert (tmp_path / "checkpoints" / "best_validation_top5_dice.pt").is_file()
    assert not any(parameter.requires_grad for parameter in autoencoder.parameters())
