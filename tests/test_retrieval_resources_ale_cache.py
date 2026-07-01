from __future__ import annotations

from pathlib import Path

import pytest
import torch

from neurovlm import retrieval_resources as rr


def test_ale_only_cache_path_uses_uploaded_hf_filenames(monkeypatch) -> None:
    calls = []

    def fake_download(repo_id: str, filename: str, repo_type: str = "dataset") -> str:
        calls.append((repo_id, filename, repo_type))
        return f"/hf-cache/{filename}"

    monkeypatch.setattr(rr, "_download_from_hf", fake_download)
    rr._load_ale_only_cache_path.cache_clear()

    atlas = rr._load_ale_only_cache_path("atlas_free")
    difumo = rr._load_ale_only_cache_path("difumo_compatible")

    assert atlas == "/hf-cache/atlas_free_4mm_fwhm9_crop_float16.pt"
    assert difumo == "/hf-cache/difumo_compatible_4mm_fwhm9_crop_float16.pt"
    assert calls == [
        ("neurovlm/atlas_free_cnn_dataset", "atlas_free_4mm_fwhm9_crop_float16.pt", "dataset"),
        ("neurovlm/atlas_free_cnn_dataset", "difumo_compatible_4mm_fwhm9_crop_float16.pt", "dataset"),
    ]


def test_ale_only_cache_rejects_unuploaded_variants() -> None:
    with pytest.raises(ValueError, match="Only the uploaded"):
        rr._load_ale_only_cache_path("atlas_free", kernel_fwhm_mm=6)

    with pytest.raises(ValueError, match="Unknown ALE-only cache mode"):
        rr._load_ale_only_cache_path("unknown")


def test_ale_only_cache_loader_reads_hf_payload(monkeypatch, tmp_path: Path) -> None:
    payload_path = tmp_path / "atlas_free_4mm_fwhm9_crop_float16.pt"
    torch.save({"volumes": torch.zeros(1, 1, 2, 2, 2), "pmids": ["1"]}, payload_path)

    def fake_download(repo_id: str, filename: str, repo_type: str = "dataset") -> str:
        return str(payload_path)

    monkeypatch.setattr(rr, "_download_from_hf", fake_download)
    rr._load_ale_only_cache_path.cache_clear()
    rr._load_ale_only_cache.cache_clear()

    payload = rr._load_ale_only_cache("atlas_free")

    assert torch.equal(payload["volumes"], torch.zeros(1, 1, 2, 2, 2))
    assert payload["pmids"] == ["1"]
