from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
THREEDCNN = REPO_ROOT / "experiments" / "3dcnn"
if str(THREEDCNN) not in sys.path:
    sys.path.insert(0, str(THREEDCNN))
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from atlas_free_cnn.evaluation import compare_ae_reconstruction as compare
from atlas_free_cnn.training.datasets import UnifiedMapTextDataset


def _write_tiny_dataset(tmp_path: Path) -> Path:
    tensor_path = tmp_path / "volumes.pt"
    volumes = torch.tensor(
        [
            [[[[0.0, 1.0], [0.2, 0.0]], [[0.0, 0.7], [0.0, 0.1]]]],
            [[[[1.0, 0.0], [0.0, 0.3]], [[0.4, 0.0], [0.8, 0.0]]]],
        ],
        dtype=torch.float32,
    )
    torch.save({"volumes": volumes}, tensor_path)
    jsonl_path = tmp_path / "test.jsonl"
    rows = [
        {
            "map_id": "pmid-1",
            "source": "pubmed",
            "pmid": "1",
            "tensor_path": str(tensor_path),
            "tensor_index": 0,
        },
        {
            "map_id": "pmid-2",
            "source": "pubmed",
            "pmid": "2",
            "tensor_path": str(tensor_path),
            "tensor_index": 1,
        },
    ]
    jsonl_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return jsonl_path


class _IdentityCNNAEAdapter:
    def __init__(self, domain: str, *, device: str | torch.device = "cpu") -> None:
        self.domain = domain
        self.device = torch.device(device)

    def encode(self, volume_batch: torch.Tensor) -> torch.Tensor:
        return volume_batch

    def decode(self, latent_batch: torch.Tensor) -> torch.Tensor:
        return latent_batch


def test_cnn_ae_reconstruction_rows_use_native_volume_metrics(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(compare.adapters, "CNNAutoencoderAdapter", _IdentityCNNAEAdapter)
    dataset = UnifiedMapTextDataset(_write_tiny_dataset(tmp_path))

    rows = compare.evaluate_cnn_ae_on_dataset(
        model_id="cnn_ae_mixed",
        dataset_name="pubmed",
        dataset=dataset,
        device="cpu",
        batch_size=2,
        target_shape=(2, 2, 2),
        include_voxel_auroc=False,
    )

    assert len(rows) == 2
    assert {row["comparison_space"] for row in rows} == {"native_atlas_free_volume"}
    assert all(row["supported"] for row in rows)
    assert all(row["reconstruction_mse"] == 0.0 for row in rows)
    assert all(row["model_domain"] == "mixed" for row in rows)


class _IdentityMLPAEAdapter:
    def __init__(self, *, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)

    def encode(self, flat_batch: torch.Tensor) -> torch.Tensor:
        return flat_batch

    def decode(self, latent_batch: torch.Tensor) -> torch.Tensor:
        return latent_batch.clamp(0.0, 1.0)


def test_mlp_on_atlas_free_dataset_uses_masker_crop_bridge(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(compare.adapters, "MLPAutoencoderAdapter", _IdentityMLPAEAdapter)
    dataset = UnifiedMapTextDataset(_write_tiny_dataset(tmp_path))

    rows = compare.evaluate_mlp_on_atlas_free_dataset(
        dataset_name="neurovault",
        dataset=dataset,
        device="cpu",
        batch_size=2,
        target_shape=(36, 45, 38),
        include_voxel_auroc=False,
    )

    assert len(rows) == 2
    assert {row["comparison_space"] for row in rows} == {"atlas_free_volume_via_mlp_masker_crop"}
    assert all(row["supported"] for row in rows)
    # Identity encode/decode over a binarized target should reconstruct exactly.
    assert all(row["reconstruction_mse"] == 0.0 for row in rows)


def test_mlp_flat_dataset_is_pubmed_only(tmp_path: Path) -> None:
    rows = compare.evaluate_mlp_flat_dataset(dataset_name="neurovault", device="cpu")

    assert len(rows) == 1
    assert not rows[0]["supported"]
    assert rows[0]["unsupported_reason"] == "handled_via_atlas_free_volume_via_mlp_masker_crop_instead"


def test_summary_counts_supported_and_unsupported_rows() -> None:
    rows = [
        {
            "dataset": "pubmed",
            "model_id": "cnn_ae_mixed",
            "comparison_space": "native_atlas_free_volume",
            "supported": True,
            "reconstruction_mse": 0.25,
        },
        {
            "dataset": "pubmed",
            "model_id": "cnn_ae_mixed",
            "comparison_space": "native_atlas_free_volume",
            "supported": True,
            "reconstruction_mse": 0.75,
        },
        {
            "dataset": "pubmed",
            "model_id": "mlp_neurovlm",
            "comparison_space": "native_atlas_free_volume",
            "supported": False,
            "unsupported_reason": "no_conversion",
        },
    ]

    summary = compare.summarize_rows(rows)

    cnn = next(row for row in summary if row["model_id"] == "cnn_ae_mixed")
    mlp = next(row for row in summary if row["model_id"] == "mlp_neurovlm")
    assert cnn["n_supported"] == 2
    assert cnn["reconstruction_mse_mean"] == 0.5
    assert mlp["n_unsupported"] == 1
    assert mlp["unsupported_reasons"] == "no_conversion"
