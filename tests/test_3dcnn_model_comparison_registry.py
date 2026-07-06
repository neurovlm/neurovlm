from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
THREEDCNN = REPO_ROOT / "experiments" / "3dcnn"
if str(THREEDCNN) not in sys.path:
    sys.path.insert(0, str(THREEDCNN))
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import neurovlm.retrieval_resources as rr
from atlas_free_cnn.evaluation import model_comparison_adapters as adapters
from atlas_free_cnn.evaluation import model_comparison_registry as registry

FLAT_DIM = 10
LATENT_DIM = 5
SHARED_DIM = 4
TEXT_DIM = 6
VOXELS = 8  # 1 x 2 x 2 x 2 volume


class _FakeMLPAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(FLAT_DIM, LATENT_DIM)
        self.decoder = nn.Linear(LATENT_DIM, FLAT_DIM)


class _FakeCNNAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(VOXELS, LATENT_DIM))
        decoder = nn.Sequential(nn.Linear(LATENT_DIM, VOXELS), nn.Unflatten(1, (1, 2, 2, 2)))
        decoder.latent_dim = LATENT_DIM
        self.decoder = decoder


@pytest.fixture(autouse=True)
def _clear_hf_caches():
    for name in [
        "_load_autoencoder",
        "_load_masker",
        "_proj_head_image_infonce",
        "_proj_head_text_infonce",
        "_proj_head_text_mse",
        "_load_mixed_ae",
        "_load_pubmed_finetuned_ae",
        "_load_nilearn_finetuned_ae",
        "_load_neurovault_finetuned_ae",
        "_load_cnn_contrastive_checkpoint_path",
        "_load_cnn_t2b_checkpoint_path",
    ]:
        func = getattr(rr, name, None)
        cache_clear = getattr(func, "cache_clear", None)
        if cache_clear is not None:
            cache_clear()
    yield


def test_registry_covers_all_expected_model_ids() -> None:
    expected = {
        "mlp_neurovlm",
        "cnn_ae_mixed",
        "cnn_ae_pubmed",
        "cnn_ae_nilearn",
        "cnn_ae_neurovault",
        "cnn_contrastive_mixed_to_pubmed",
        "cnn_contrastive_mixed_to_nilearn",
        "cnn_contrastive_mixed_to_neurovault",
        "cnn_contrastive_pubmed",
        "cnn_contrastive_nilearn",
        "cnn_contrastive_neurovault",
        "cnn_t2b_mixed_to_pubmed",
        "cnn_t2b_mixed_to_nilearn",
        "cnn_t2b_mixed_to_neurovault",
        "cnn_t2b_pubmed",
        "cnn_t2b_nilearn",
        "cnn_t2b_neurovault",
    }
    assert set(registry.MODEL_IDS) == expected


def test_registry_marks_baseline_vs_specialized_branch() -> None:
    assert registry.MODEL_SPECS["cnn_contrastive_mixed_to_pubmed"].branch == "baseline"
    assert registry.MODEL_SPECS["cnn_contrastive_mixed_to_pubmed"].domain == "pubmed"
    assert registry.MODEL_SPECS["cnn_contrastive_pubmed"].branch == "specialized"
    assert registry.MODEL_SPECS["cnn_contrastive_pubmed"].domain == "pubmed"
    assert registry.MODEL_SPECS["cnn_t2b_mixed_to_neurovault"].branch == "baseline"
    assert registry.MODEL_SPECS["cnn_t2b_neurovault"].branch == "specialized"


def test_registry_resolves_uploaded_models(monkeypatch) -> None:
    monkeypatch.setattr(rr, "_load_autoencoder", lambda: _FakeMLPAutoencoder())
    monkeypatch.setattr(rr, "_load_masker", lambda: object())
    monkeypatch.setattr(rr, "_proj_head_image_infonce", lambda: nn.Linear(LATENT_DIM, SHARED_DIM))
    monkeypatch.setattr(rr, "_proj_head_text_infonce", lambda: nn.Linear(TEXT_DIM, SHARED_DIM))
    monkeypatch.setattr(rr, "_proj_head_text_mse", lambda: nn.Linear(TEXT_DIM, LATENT_DIM))
    for name in [
        "_load_mixed_ae",
        "_load_pubmed_finetuned_ae",
        "_load_nilearn_finetuned_ae",
        "_load_neurovault_finetuned_ae",
    ]:
        monkeypatch.setattr(rr, name, lambda: _FakeCNNAutoencoder())

    manifest = registry.resolve_model_registry(
        ("mlp_neurovlm", "cnn_ae_mixed", "cnn_ae_pubmed", "cnn_ae_nilearn", "cnn_ae_neurovault")
    )

    for model_id in ["mlp_neurovlm", "cnn_ae_mixed", "cnn_ae_pubmed", "cnn_ae_nilearn", "cnn_ae_neurovault"]:
        assert manifest[model_id]["status"] == "resolved"
        assert manifest[model_id]["error"] is None


def test_registry_reports_missing_checkpoint_instead_of_crashing(monkeypatch) -> None:
    def _raise(variant: str) -> str:
        raise FileNotFoundError(f"{variant} checkpoint not uploaded yet")

    monkeypatch.setattr(rr, "_load_cnn_contrastive_checkpoint_path", _raise)
    monkeypatch.setattr(rr, "_load_cnn_t2b_checkpoint_path", _raise)

    manifest = registry.resolve_model_registry(("cnn_contrastive_mixed_to_pubmed", "cnn_t2b_pubmed"))

    assert manifest["cnn_contrastive_mixed_to_pubmed"]["status"] == "missing_checkpoint"
    assert "not uploaded yet" in manifest["cnn_contrastive_mixed_to_pubmed"]["error"]
    assert manifest["cnn_contrastive_mixed_to_pubmed"]["checkpoint_path"] is None
    assert manifest["cnn_t2b_pubmed"]["status"] == "missing_checkpoint"


def test_write_resolved_registry_produces_valid_json(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(rr, "_load_autoencoder", lambda: _FakeMLPAutoencoder())
    monkeypatch.setattr(rr, "_load_masker", lambda: object())
    monkeypatch.setattr(rr, "_proj_head_image_infonce", lambda: nn.Linear(LATENT_DIM, SHARED_DIM))
    monkeypatch.setattr(rr, "_proj_head_text_infonce", lambda: nn.Linear(TEXT_DIM, SHARED_DIM))
    monkeypatch.setattr(rr, "_proj_head_text_mse", lambda: nn.Linear(TEXT_DIM, LATENT_DIM))
    for name in [
        "_load_mixed_ae",
        "_load_pubmed_finetuned_ae",
        "_load_nilearn_finetuned_ae",
        "_load_neurovault_finetuned_ae",
    ]:
        monkeypatch.setattr(rr, name, lambda: _FakeCNNAutoencoder())

    def _raise(variant: str) -> str:
        raise FileNotFoundError("not uploaded yet")

    monkeypatch.setattr(rr, "_load_cnn_contrastive_checkpoint_path", _raise)
    monkeypatch.setattr(rr, "_load_cnn_t2b_checkpoint_path", _raise)

    import json

    out_path = registry.write_resolved_registry(tmp_path / "model_registry_resolved.json")
    manifest = json.loads(out_path.read_text())

    assert set(manifest) == set(registry.MODEL_IDS)
    assert manifest["mlp_neurovlm"]["status"] == "resolved"
    assert manifest["cnn_contrastive_mixed_to_pubmed"]["status"] == "missing_checkpoint"


def test_mlp_autoencoder_adapter_encode_decode_shapes(monkeypatch) -> None:
    monkeypatch.setattr(rr, "_load_autoencoder", lambda: _FakeMLPAutoencoder())
    monkeypatch.setattr(rr, "_load_masker", lambda: object())

    adapter = adapters.MLPAutoencoderAdapter()
    flat = torch.randn(3, FLAT_DIM)

    latent = adapter.encode(flat)
    recon = adapter.decode(latent)

    assert latent.shape == (3, LATENT_DIM)
    assert recon.shape == (3, FLAT_DIM)


def test_cnn_autoencoder_adapter_encode_decode_shapes(monkeypatch) -> None:
    monkeypatch.setattr(rr, "_load_mixed_ae", lambda: _FakeCNNAutoencoder())

    adapter = adapters.CNNAutoencoderAdapter("mixed")
    volume = torch.randn(2, VOXELS)

    latent = adapter.encode(volume)
    recon = adapter.decode(latent)

    assert latent.shape == (2, LATENT_DIM)
    assert recon.shape == (2, 1, 2, 2, 2)


def test_cnn_autoencoder_adapter_rejects_unknown_domain() -> None:
    with pytest.raises(ValueError, match="Unknown CNN AE domain"):
        adapters.CNNAutoencoderAdapter("unknown")


def test_mlp_contrastive_adapter_shared_space_shapes_and_unit_norm(monkeypatch) -> None:
    monkeypatch.setattr(rr, "_load_autoencoder", lambda: _FakeMLPAutoencoder())
    monkeypatch.setattr(rr, "_proj_head_image_infonce", lambda: nn.Linear(LATENT_DIM, SHARED_DIM))
    monkeypatch.setattr(rr, "_proj_head_text_infonce", lambda: nn.Linear(TEXT_DIM, SHARED_DIM))

    adapter = adapters.MLPContrastiveAdapter()
    flat = torch.randn(3, FLAT_DIM)
    text = torch.randn(3, TEXT_DIM)

    brain_shared = adapter.encode_brain_to_shared(flat)
    text_shared = adapter.encode_text_to_shared(text)

    assert brain_shared.shape == (3, SHARED_DIM)
    assert text_shared.shape == (3, SHARED_DIM)
    assert torch.allclose(brain_shared.norm(dim=-1), torch.ones(3), atol=1e-5)
    assert torch.allclose(text_shared.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_cnn_contrastive_adapter_uses_stage3_evaluator(monkeypatch) -> None:
    monkeypatch.setattr(rr, "_load_cnn_contrastive_checkpoint_path", lambda variant: f"/fake/{variant}.pt")

    def fake_load_stage3_evaluator(checkpoint_path, device):
        assert checkpoint_path == "/fake/pubmed.pt"
        return nn.Linear(VOXELS, SHARED_DIM), nn.Linear(TEXT_DIM, SHARED_DIM)

    monkeypatch.setattr(adapters, "load_stage3_evaluator", fake_load_stage3_evaluator)

    adapter = adapters.CNNContrastiveAdapter("pubmed")
    volume = torch.randn(2, VOXELS)
    text = torch.randn(2, TEXT_DIM)

    brain_shared = adapter.encode_brain_to_shared(volume)
    text_shared = adapter.encode_text_to_shared(text)

    assert brain_shared.shape == (2, SHARED_DIM)
    assert text_shared.shape == (2, SHARED_DIM)


def test_cnn_contrastive_adapter_baseline_variant_uses_mixed_to_domain_checkpoint(monkeypatch) -> None:
    monkeypatch.setattr(rr, "_load_cnn_contrastive_checkpoint_path", lambda variant: f"/fake/{variant}.pt")

    def fake_load_stage3_evaluator(checkpoint_path, device):
        assert checkpoint_path == "/fake/mixed_to_pubmed.pt"
        return nn.Linear(VOXELS, SHARED_DIM), nn.Linear(TEXT_DIM, SHARED_DIM)

    monkeypatch.setattr(adapters, "load_stage3_evaluator", fake_load_stage3_evaluator)

    adapter = adapters.CNNContrastiveAdapter("mixed_to_pubmed")
    assert adapter.encode_brain_to_shared(torch.randn(2, VOXELS)).shape == (2, SHARED_DIM)


def test_cnn_t2b_adapter_specialized_variant_uses_domain_finetuned_ae(monkeypatch) -> None:
    monkeypatch.setattr(rr, "_load_pubmed_finetuned_ae", lambda: _FakeCNNAutoencoder())
    monkeypatch.setattr(rr, "_load_cnn_t2b_checkpoint_path", lambda variant: f"/fake/{variant}.pt")

    def fake_load_stage4_projector(checkpoint_path, device, *, latent_dim):
        assert checkpoint_path == "/fake/pubmed.pt"
        assert latent_dim == LATENT_DIM
        return nn.Linear(TEXT_DIM, latent_dim), {}

    monkeypatch.setattr(adapters, "load_stage4_projector", fake_load_stage4_projector)

    adapter = adapters.CNNTextToBrainAdapter("pubmed")
    text = torch.randn(2, TEXT_DIM)

    generated = adapter.generate(text)

    assert generated.shape == (2, 1, 2, 2, 2)


def test_cnn_t2b_adapter_baseline_variant_uses_mixed_ae(monkeypatch) -> None:
    monkeypatch.setattr(rr, "_load_mixed_ae", lambda: _FakeCNNAutoencoder())
    monkeypatch.setattr(rr, "_load_cnn_t2b_checkpoint_path", lambda variant: f"/fake/{variant}.pt")

    def fake_load_stage4_projector(checkpoint_path, device, *, latent_dim):
        assert checkpoint_path == "/fake/mixed_to_pubmed.pt"
        assert latent_dim == LATENT_DIM
        return nn.Linear(TEXT_DIM, latent_dim), {}

    monkeypatch.setattr(adapters, "load_stage4_projector", fake_load_stage4_projector)

    adapter = adapters.CNNTextToBrainAdapter("mixed_to_pubmed")
    text = torch.randn(2, TEXT_DIM)

    generated = adapter.generate(text)

    assert generated.shape == (2, 1, 2, 2, 2)


def test_cnn_contrastive_and_t2b_adapters_reject_unknown_variant() -> None:
    with pytest.raises(ValueError, match="Unknown CNN stage variant"):
        adapters.CNNContrastiveAdapter("unknown")
    with pytest.raises(ValueError, match="Unknown CNN stage variant"):
        adapters.CNNTextToBrainAdapter("unknown")
