from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from neurovlm.evaluation.mlp import evaluate_mlp_text_to_brain
from neurovlm.models import NeuroAutoEncoder, ProjHead
from neurovlm.training.mlp import (
    MLPAutoencoderTrainConfig,
    MLPContrastiveTrainConfig,
    MLPTextToBrainTrainConfig,
    build_mlp_autoencoder,
    build_mlp_contrastive,
    build_mlp_text_to_brain,
    mlp_autoencoder_from_checkpoint,
    mlp_contrastive_from_checkpoint,
    mlp_text_to_brain_from_checkpoint,
    train_mlp_autoencoder,
    train_mlp_contrastive,
    train_mlp_text_to_brain,
)


class _BrainRows(Dataset):
    def __init__(self, offset: int = 0):
        generator = torch.Generator().manual_seed(100 + offset)
        self.values = (torch.rand(6, 8, generator=generator) > 0.7).float()

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return {"brain": self.values[index], "source": "pubmed" if index % 2 else "nilearn",
                "sample_id": f"brain-{index}"}


class _PairedRows(Dataset):
    def __init__(self, autoencoder: NeuroAutoEncoder, offset: int = 0):
        generator = torch.Generator().manual_seed(200 + offset)
        self.brain = (torch.rand(6, 8, generator=generator) > 0.65).float()
        with torch.no_grad():
            self.latent = autoencoder.encoder(self.brain)
        matrix = torch.randn(4, 6, generator=generator)
        self.text = self.latent @ matrix

    def __len__(self):
        return len(self.brain)

    def __getitem__(self, index):
        return {"text_embedding": self.text[index], "brain_embedding": self.latent[index],
                "brain": self.brain[index], "source": "pubmed" if index % 2 else "neurovault",
                "sample_id": f"pair-{index}"}


def _ae_config(tmp_path: Path, **overrides):
    values = dict(output_root=tmp_path, run_id="mlp-ae", device="cpu", epochs=1,
                  batch_size=3, eval_batch_size=3, preset="custom", dim_neuro=8,
                  dim_h0=6, dim_h1=5, dim_latent=4)
    values.update(overrides)
    return MLPAutoencoderTrainConfig(**values)


def _t2b_config(tmp_path: Path, **overrides):
    values = dict(output_root=tmp_path, run_id="mlp-t2b", device="cpu", epochs=1,
                  batch_size=3, eval_batch_size=3, preset="custom", text_dim=6,
                  hidden_dim=5, latent_dim=4, brain_dim=8)
    values.update(overrides)
    return MLPTextToBrainTrainConfig(**values)


def _contrastive_config(tmp_path: Path, **overrides):
    values = dict(output_root=tmp_path, run_id="mlp-contrastive", device="cpu", epochs=1,
                  batch_size=3, eval_batch_size=3, preset="custom", text_dim=6,
                  text_hidden_dim=5, brain_dim=4, brain_hidden_dim=4, shared_dim=4,
                  initialize_text_from_mse=False)
    values.update(overrides)
    return MLPContrastiveTrainConfig(**values)


def test_retained_configs_preserve_legacy_architectures_and_losses():
    ae = MLPAutoencoderTrainConfig()
    assert ae.architecture()["dim_neuro"] == 28_542
    assert ae.architecture()["out"] == "logit"
    t2b = MLPTextToBrainTrainConfig()
    assert t2b.architecture()["loss"] == "latent_mse"
    contrastive = MLPContrastiveTrainConfig()
    assert contrastive.temperature == pytest.approx(0.07)
    assert contrastive.initialize_text_from_mse
    with pytest.raises(ValueError, match="preset='custom'"):
        MLPAutoencoderTrainConfig(dim_neuro=8)


def test_mlp_autoencoder_artifacts_reload_and_legacy_model_compatibility(tmp_path: Path):
    provider = {"train": _BrainRows(0), "val": _BrainRows(1), "test": _BrainRows(2)}
    result = train_mlp_autoencoder(_ae_config(tmp_path), provider=provider)
    assert result.best_checkpoint.is_file() and result.last_checkpoint.is_file()
    assert (result.run_dir / "metrics/by_source.csv").is_file()
    assert (result.run_dir / "metrics/by_sample.csv").is_file()
    assert json.loads((result.run_dir / "status.json").read_text())["state"] == "completed"
    reloaded = mlp_autoencoder_from_checkpoint(result.best_checkpoint)
    sample = provider["test"].values[:2]
    with torch.no_grad():
        assert torch.allclose(result.model(sample), reloaded(sample))
    # The standardized runner does not alter the legacy probability/logit API.
    assert torch.all((NeuroAutoEncoder(dim_neuro=8, dim_h0=6, dim_h1=5, dim_latent=4)(sample)) >= 0)


def test_mlp_autoencoder_resume_preserves_history_and_validates_architecture(tmp_path: Path, monkeypatch):
    import neurovlm.training.mlp as module

    provider = {"train": _BrainRows(0), "val": _BrainRows(1), "test": _BrainRows(2)}
    original = module._ae_step
    calls = 0

    def interrupt_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_ae_step", interrupt_second)
    with pytest.raises(RuntimeError, match="interrupted"):
        train_mlp_autoencoder(_ae_config(tmp_path, epochs=2), provider=provider)
    monkeypatch.setattr(module, "_ae_step", original)
    result = train_mlp_autoencoder(_ae_config(tmp_path, epochs=2, resume="last.pt"), provider=provider)
    rows = list(csv.DictReader((result.run_dir / "metrics/history.csv").open()))
    assert {row["epoch"] for row in rows if row["epoch"]} == {"1", "2"}
    assert json.loads((result.run_dir / "status.json").read_text())["resume_count"] == 1

    malformed = torch.load(result.best_checkpoint, weights_only=True)
    malformed["architecture"]["dim_latent"] = 7
    path = tmp_path / "bad-mlp-ae.pt"
    torch.save(malformed, path)
    with pytest.raises((RuntimeError, ValueError)):
        mlp_autoencoder_from_checkpoint(path)


def test_mlp_text_to_brain_uses_latent_mse_freezes_ae_and_reports_decoded_metrics(tmp_path: Path):
    ae = build_mlp_autoencoder(_ae_config(tmp_path))
    before = {name: value.detach().clone() for name, value in ae.state_dict().items()}
    provider = {"train": _PairedRows(ae, 0), "val": _PairedRows(ae, 1), "test": _PairedRows(ae, 2)}
    model = build_mlp_text_to_brain(
        _t2b_config(tmp_path), autoencoder=ae,
        text_projection=ProjHead(6, 5, 4, seed=3),
    )
    result = train_mlp_text_to_brain(_t2b_config(tmp_path), provider=provider, model=model)
    assert "latent_mse" in result.test_metrics
    assert "decoded_bce_with_logits" in result.test_metrics
    assert all(parameter.grad is None for parameter in ae.parameters())
    assert all(torch.equal(before[name], value) for name, value in ae.state_dict().items())
    payload = torch.load(result.best_checkpoint, weights_only=True)
    assert set(payload["model_state_dict"]) == set(model.text_projection.state_dict())
    reloaded = mlp_text_to_brain_from_checkpoint(result.best_checkpoint, autoencoder=ae)
    with torch.no_grad():
        assert torch.allclose(result.model(provider["test"].text), reloaded(provider["test"].text))
    latent_only = [{"text_embedding": provider["test"].text[0],
                    "brain_embedding": provider["test"].latent[0]}]
    evaluation = evaluate_mlp_text_to_brain(reloaded, latent_only)
    assert "latent_mse" in evaluation.summary
    assert not any(name.startswith("decoded_") for name in evaluation.summary)


def test_mlp_contrastive_artifacts_reload_and_both_heads_receive_gradients(tmp_path: Path):
    ae = build_mlp_autoencoder(_ae_config(tmp_path))
    provider = {"train": _PairedRows(ae, 0), "val": _PairedRows(ae, 1), "test": _PairedRows(ae, 2)}
    model = build_mlp_contrastive(_contrastive_config(tmp_path))
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    result = train_mlp_contrastive(_contrastive_config(tmp_path), provider=provider, model=model)
    assert "mean_mrr" in result.test_metrics
    assert any(not torch.equal(initial[name], value) for name, value in model.brain_projection.state_dict(prefix="brain_projection.").items())
    assert any(not torch.equal(initial[name], value) for name, value in model.text_projection.state_dict(prefix="text_projection.").items())
    reloaded = mlp_contrastive_from_checkpoint(result.best_checkpoint)
    with torch.no_grad():
        expected = result.model(provider["test"].latent, provider["test"].text)
        actual = reloaded(provider["test"].latent, provider["test"].text)
    assert all(torch.allclose(left, right) for left, right in zip(expected, actual))
    rows = list(csv.DictReader((result.run_dir / "metrics/history.csv").open()))
    assert {row["split"] for row in rows} >= {"train", "val", "test"}
