from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from neurovlm.cnn.architectures import ALE3DCNNAutoEncoder
from neurovlm.cnn.models import (
    CNNTextToBrainModel,
    GenerativeTextToAELatent,
    text_to_brain_from_payload,
)
from neurovlm.data.atlas_free_text import AtlasFreeTextEmbeddingLookup
from neurovlm.evaluation import evaluate_text_to_brain
from neurovlm.training import (
    TextToBrainTrainConfig,
    build_text_to_brain,
    text_to_brain_from_checkpoint,
    text_to_brain_loss,
    train_text_to_brain,
)


class _Pairs(Dataset):
    def __init__(self, split: str, n: int = 3):
        self.rows = [
            {
                "volume": torch.rand(1, 4, 4, 4, generator=torch.Generator().manual_seed(index)),
                "map_id": f"{split}-{index}",
                "positive_texts": [
                    {"text_id": f"text-{index}", "text": f"primary {index}"},
                    {"text_id": "ignored", "text": "not selected"},
                ],
                "metadata": {"source": "pubmed"},
            }
            for index in range(n)
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class _Provider:
    def __init__(self):
        self.train = _Pairs("train")
        self.val = _Pairs("val")
        self.test = _Pairs("test")


def _lookup() -> AtlasFreeTextEmbeddingLookup:
    values = torch.zeros(3, 768)
    values[0, 0] = 1
    values[1, 1] = 1
    values[2, 2] = 1
    return AtlasFreeTextEmbeddingLookup(values, ["text-0", "text-1", "text-2"])


def _autoencoder(shape=(4, 4, 4)) -> ALE3DCNNAutoEncoder:
    return ALE3DCNNAutoEncoder(
        output_shape=shape,
        base_channels=2,
        num_blocks=1,
        latent_dim=384,
        dropout=0.5,
    )


def _config(tmp_path: Path, **overrides) -> TextToBrainTrainConfig:
    values = {
        "domain": "pubmed",
        "output_root": tmp_path,
        "run_id": "tiny-t2b",
        "epochs": 1,
        "batch_size": 2,
        "eval_batch_size": 2,
        "device": "cpu",
        "amp": False,
        "early_stopping_patience": None,
        "preset": "custom",
        "target_shape": (4, 4, 4),
        "base_channels": 2,
        "num_blocks": 1,
    }
    values.update(overrides)
    return TextToBrainTrainConfig(**values)


def _model() -> CNNTextToBrainModel:
    return build_text_to_brain(_config(Path("unused")), autoencoder=_autoencoder())


@pytest.mark.parametrize(
    ("domain", "variant", "internal"),
    [
        ("pubmed", "mixed_baseline", "mixed_to_pubmed"),
        ("nilearn", "mixed_baseline", "mixed_to_nilearn"),
        ("neurovault", "mixed_baseline", "mixed_to_neurovault"),
        ("pubmed", "finetuned", "pubmed"),
        ("nilearn", "finetuned", "nilearn"),
        ("neurovault", "finetuned", "neurovault"),
    ],
)
def test_all_six_branch_names_are_exact(domain: str, variant: str, internal: str) -> None:
    assert TextToBrainTrainConfig(domain=domain, variant=variant).internal_variant == internal


def test_domain_and_exact_projector_are_required() -> None:
    with pytest.raises(TypeError):
        TextToBrainTrainConfig()  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="768 -> 512 -> 384"):
        TextToBrainTrainConfig(domain="pubmed", text_hidden_dim=64)


def test_default_and_finetuned_choose_exact_released_autoencoder(monkeypatch) -> None:
    import neurovlm.models.base as models

    calls = []

    def fake_load_model(**kwargs):
        calls.append(kwargs)
        return _autoencoder((36, 45, 38))

    monkeypatch.setattr(models, "load_model", fake_load_model)
    build_text_to_brain(TextToBrainTrainConfig(domain="nilearn"))
    build_text_to_brain(
        TextToBrainTrainConfig(domain="neurovault", variant="finetuned")
    )
    assert calls == [
        {"family": "cnn", "task": "autoencoder", "variant": "mixed_baseline"},
        {
            "family": "cnn",
            "task": "autoencoder",
            "variant": "finetuned",
            "domain": "neurovault",
        },
    ]


def test_raw_loss_gradients_cross_frozen_decoder_only_into_projector(tmp_path: Path) -> None:
    model = _model()
    assert not model.autoencoder.training
    assert not any(parameter.requires_grad for parameter in model.autoencoder.parameters())
    target = torch.rand(2, 1, 4, 4, 4)
    text = _lookup().embeddings[:2]
    with torch.no_grad():
        brain_z = model.autoencoder.encoder(target)
    text_z = model.text_projection(text)
    raw_prediction = model.autoencoder.decoder(text_z)
    loss, parts = text_to_brain_loss(raw_prediction, target, brain_z, text_z)
    loss.backward()
    assert torch.allclose(parts["loss"], parts["reconstruction_mse"] + parts["latent_mse"])
    assert any(parameter.grad is not None for parameter in model.text_projection.parameters())
    assert all(parameter.grad is None for parameter in model.autoencoder.parameters())
    assert not model.autoencoder.training


def test_evaluation_uses_weights_first_positive_and_bounded_outputs() -> None:
    model = _model()
    with torch.no_grad():
        for parameter in model.autoencoder.decoder.parameters():
            parameter.zero_()
        model.autoencoder.decoder.out.bias.fill_(2.0)
    result = evaluate_text_to_brain(
        model,
        _Pairs("val"),
        lookup=_lookup(),
        target_shape=(4, 4, 4),
        batch_size=2,
        reconstruction_weight=2.0,
        latent_weight=3.0,
        generated_limit=1,
    )
    assert result.n == 3
    assert len(result.generated) == 1
    assert {row["text_id"] for row in result.by_sample} == {
        "text-0",
        "text-1",
        "text-2",
    }
    assert result.summary["loss"] == pytest.approx(
        2 * result.summary["raw_reconstruction_mse"] + 3 * result.summary["latent_mse"]
    )
    assert result.summary["raw_reconstruction_mse"] > result.summary["reconstruction_mse"]
    for metric in ("latent_cosine", "spatial_corr", "top5_dice"):
        assert metric in result.summary


def test_tiny_training_artifacts_projector_only_reload_and_resume(tmp_path: Path) -> None:
    provider = _Provider()
    lookup = _lookup()
    result = train_text_to_brain(
        _config(tmp_path, generated_output_limit=1),
        provider=provider,
        lookup=lookup,
        model=_model(),
    )
    assert result.best_checkpoint.is_file()
    assert (result.run_dir / "checkpoints" / "best_val_top5_dice.pt").is_file()
    for name in ("history.csv", "by_source.csv", "by_sample.csv", "test_summary.csv"):
        assert (result.run_dir / "metrics" / name).is_file()
    assert (result.run_dir / "generated_maps" / "test_predictions.pt").is_file()
    effective = json.loads((result.run_dir / "config" / "effective.json").read_text())
    assert effective["values"]["internal_variant"] == "mixed_to_pubmed"
    assert effective["values"]["autoencoder_frozen"] is True

    payload = torch.load(result.best_checkpoint, map_location="cpu", weights_only=True)
    assert payload["extra"]["autoencoder_source"] == {"kind": "provided_model"}
    assert set(payload["model_state_dict"]) == set(result.model.text_projection.state_dict())
    assert not any(key.startswith("autoencoder.") for key in payload["model_state_dict"])
    reloaded = text_to_brain_from_checkpoint(
        result.best_checkpoint, autoencoder=_autoencoder()
    )
    text = lookup.embeddings[:1]
    with torch.no_grad():
        assert torch.allclose(result.model(text), reloaded(text))

    history_before = list(csv.DictReader((result.run_dir / "metrics" / "history.csv").open()))
    resumed = train_text_to_brain(
        _config(tmp_path, epochs=2, generated_output_limit=1, resume="last.pt"),
        provider=provider,
        lookup=lookup,
        model=_model(),
    )
    history_after = list(csv.DictReader((resumed.run_dir / "metrics" / "history.csv").open()))
    assert resumed.epochs_completed == 2
    assert len(history_after) > len(history_before)
    status = json.loads((resumed.run_dir / "status.json").read_text())
    assert status["resume_count"] == 1


def test_checkpoint_architecture_mismatch_and_legacy_payload_compatibility(tmp_path: Path) -> None:
    projector = GenerativeTextToAELatent()
    legacy = {
        "config": {"generative_text_to_ae_latent": {"in_dim": 768, "hidden_dim": 512}},
        "generative_text_to_ae_latent": projector.state_dict(),
    }
    restored = text_to_brain_from_payload(legacy, _autoencoder())
    assert isinstance(restored, CNNTextToBrainModel)

    checkpoint = tmp_path / "mismatch.pt"
    torch.save(
        {
            "architecture": {
                "architecture": "GenerativeTextToAELatent",
                "text_projection": {"in_dim": 768, "hidden_dim": 512, "latent_dim": 384},
                "autoencoder": {"output_shape": (5, 5, 5), "latent_dim": 384},
            },
            "model_state_dict": projector.state_dict(),
        },
        checkpoint,
    )
    with pytest.raises(ValueError, match="architecture mismatch"):
        text_to_brain_from_checkpoint(checkpoint, autoencoder=_autoencoder())
