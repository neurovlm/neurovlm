from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from neurovlm import retrieval_resources as rr
from neurovlm.atlas_free_dataset import (
    AtlasFreeCNNDataProvider,
    AtlasFreeCNNDataset,
    atlas_free_cnn_splits,
)


def _payload(map_ids: list[str]) -> dict:
    volumes = torch.arange(len(map_ids) * 24, dtype=torch.float16).reshape(len(map_ids), 1, 2, 3, 4)
    return {
        "version": 1,
        "volumes": volumes,
        "map_ids": map_ids,
        "shape": list(volumes.shape),
    }


def _row(map_id: str, index: int, split: str, source: str = "pubmed") -> dict:
    return {
        "map_id": map_id,
        "tensor_index": index,
        "split": split,
        "source": source,
        "shape": [2, 3, 4],
        "positive_texts": [{"text_id": f"{map_id}::0", "text": f"text for {map_id}"}],
        "tensor_path": "experiments/3dcnn/atlas_free_cnn/cache/does-not-exist.pt",
        "nifti_path": "experiments/3dcnn/atlas_free_cnn/cache/does-not-exist.nii.gz",
    }


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _clear_resource_caches() -> None:
    rr._load_atlas_free_cnn_split_path.cache_clear()
    rr._load_atlas_free_cnn_split_rows.cache_clear()
    rr._load_atlas_free_cnn_volumes.cache_clear()


def test_retrieval_resources_resolve_all_canonical_hf_split_paths(monkeypatch) -> None:
    calls: list[tuple[str, str, str]] = []

    def fake_download(repo_id: str, filename: str, repo_type: str = "dataset") -> str:
        calls.append((repo_id, filename, repo_type))
        return f"/hf-cache/{filename}"

    monkeypatch.setattr(rr, "_download_from_hf", fake_download)
    _clear_resource_caches()

    paths = [rr._load_atlas_free_cnn_split_path(split) for split in ("train", "val", "test")]

    assert paths == [
        "/hf-cache/train.jsonl",
        "/hf-cache/val.jsonl",
        "/hf-cache/test.jsonl",
    ]
    assert calls == [
        ("neurovlm/atlas_free_cnn_dataset", "train.jsonl", "dataset"),
        ("neurovlm/atlas_free_cnn_dataset", "val.jsonl", "dataset"),
        ("neurovlm/atlas_free_cnn_dataset", "test.jsonl", "dataset"),
    ]


def test_default_dataset_uses_canonical_hf_resources_and_ignores_legacy_paths(
    monkeypatch, tmp_path: Path
) -> None:
    split_path = _write_jsonl(tmp_path / "test.jsonl", [_row("map-b", 1, "test")])
    volume_path = tmp_path / "atlas_free_cnn_volumes.pt"
    payload = _payload(["map-a", "map-b"])
    torch.save(payload, volume_path)
    calls: list[tuple[str, str, str]] = []

    def fake_download(repo_id: str, filename: str, repo_type: str = "dataset") -> str:
        calls.append((repo_id, filename, repo_type))
        return str(split_path if filename == "test.jsonl" else volume_path)

    monkeypatch.setattr(rr, "_download_from_hf", fake_download)
    _clear_resource_caches()

    dataset = AtlasFreeCNNDataset("test")
    item = dataset[0]

    assert calls == [
        ("neurovlm/atlas_free_cnn_dataset", "test.jsonl", "dataset"),
        ("neurovlm/atlas_free_cnn_dataset", "atlas_free_cnn_volumes.pt", "dataset"),
    ]
    assert item["map_id"] == "map-b"
    assert item["volume"].dtype == torch.float32
    assert torch.equal(item["volume"], payload["volumes"][1].float())
    assert item["metadata"]["tensor_path"].startswith("experiments/3dcnn/")
    assert item["positive_texts"] == dataset.positive_texts[0]


def test_provider_exposes_cached_split_views_and_deterministic_limit(monkeypatch) -> None:
    payload = _payload(["train-a", "train-b", "val-a", "val-b", "test-a", "test-b"])
    rows = {
        "train": (_row("train-a", 0, "train"), _row("train-b", 1, "train", "neurovault_maps")),
        "val": (_row("val-a", 2, "val"), _row("val-b", 3, "val", "neurovault_maps")),
        "test": (_row("test-a", 4, "test"), _row("test-b", 5, "test", "neurovault_maps")),
    }
    volume_calls = 0
    split_calls: list[str] = []

    def load_volumes() -> dict:
        nonlocal volume_calls
        volume_calls += 1
        return payload

    def load_rows(split: str):
        split_calls.append(split)
        return rows[split]

    monkeypatch.setattr(rr, "_load_atlas_free_cnn_volumes", load_volumes)
    monkeypatch.setattr(rr, "_load_atlas_free_cnn_split_rows", load_rows)

    provider = AtlasFreeCNNDataProvider(limit=1)

    assert provider.train is provider.train
    assert [provider.train[0]["map_id"], provider.val[0]["map_id"], provider.test[0]["map_id"]] == [
        "train-a",
        "val-a",
        "test-a",
    ]
    assert volume_calls == 1
    assert split_calls == ["train", "val", "test"]

    again = AtlasFreeCNNDataset("train", limit=1)
    assert again[0]["map_id"] == provider.train[0]["map_id"]


def test_source_and_domain_filters_are_applied_before_limit(monkeypatch) -> None:
    payload = _payload(["a", "b", "c"])
    rows = (
        _row("a", 0, "train", "pubmed"),
        _row("b", 1, "train", "neurovault_collection"),
        _row("c", 2, "train", "neurovault_collection"),
    )
    monkeypatch.setattr(rr, "_load_atlas_free_cnn_volumes", lambda: payload)
    monkeypatch.setattr(rr, "_load_atlas_free_cnn_split_rows", lambda split: rows)

    by_source = AtlasFreeCNNDataset("train", source="neurovault_collection", limit=1)
    by_domain = AtlasFreeCNNDataset("train", domain="neurovault")

    assert [row["map_id"] for row in by_source.metadata] == ["b"]
    assert [row["map_id"] for row in by_domain.metadata] == ["b", "c"]


def test_local_split_and_volume_overrides_never_call_hf(monkeypatch, tmp_path: Path) -> None:
    payload = _payload(["train", "val", "test"])
    volume_path = tmp_path / "volumes.pt"
    torch.save(payload, volume_path)
    split_dir = tmp_path / "splits"
    for index, split in enumerate(("train", "val", "test")):
        _write_jsonl(split_dir / f"{split}.jsonl", [_row(split, index, split)])

    def unexpected_hf_call(*args, **kwargs):  # pragma: no cover - assertion helper
        raise AssertionError("Explicit local overrides must not call Hugging Face resources")

    monkeypatch.setattr(rr, "_load_atlas_free_cnn_volumes", unexpected_hf_call)
    monkeypatch.setattr(rr, "_load_atlas_free_cnn_split_rows", unexpected_hf_call)

    splits = atlas_free_cnn_splits(split_dir=split_dir, volume_path=volume_path)

    assert list(splits) == ["train", "val", "test"]
    assert [splits[name][0]["map_id"] for name in splits] == ["train", "val", "test"]


@pytest.mark.parametrize("split", ["validation", "dev", ""])
def test_invalid_split_is_rejected(split: str) -> None:
    with pytest.raises(ValueError, match="Unknown atlas-free CNN split"):
        AtlasFreeCNNDataset(split)


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_invalid_limit_is_rejected(limit) -> None:
    with pytest.raises(ValueError, match="limit must be a positive integer"):
        AtlasFreeCNNDataset("train", limit=limit)


@pytest.mark.parametrize(
    ("rows", "payload", "error", "message"),
    [
        ([_row("a", 2, "train")], _payload(["a"]), IndexError, "outside volume bounds"),
        ([_row("wrong", 0, "train")], _payload(["a"]), ValueError, "does not align"),
        ([_row("a", 0, "test")], _payload(["a"]), ValueError, "loaded as 'train'"),
        ([{**_row("a", 0, "train"), "shape": [9, 9, 9]}], _payload(["a"]), ValueError, "payload shape"),
    ],
)
def test_row_alignment_and_bounds_are_validated(
    monkeypatch, rows: list[dict], payload: dict, error: type[Exception], message: str
) -> None:
    monkeypatch.setattr(rr, "_load_atlas_free_cnn_volumes", lambda: payload)
    monkeypatch.setattr(rr, "_load_atlas_free_cnn_split_rows", lambda split: tuple(rows))

    with pytest.raises(error, match=message):
        AtlasFreeCNNDataset("train")


def test_volume_payload_shape_and_map_id_count_are_validated(monkeypatch) -> None:
    rows = (_row("a", 0, "train"),)
    monkeypatch.setattr(rr, "_load_atlas_free_cnn_split_rows", lambda split: rows)

    monkeypatch.setattr(rr, "_load_atlas_free_cnn_volumes", lambda: {"volumes": torch.zeros(1, 2, 3, 4)})
    with pytest.raises(ValueError, match="N x 1 x D x H x W"):
        AtlasFreeCNNDataset("train")

    bad_ids = _payload(["a"])
    bad_ids["map_ids"] = ["a", "b"]
    monkeypatch.setattr(rr, "_load_atlas_free_cnn_volumes", lambda: bad_ids)
    with pytest.raises(ValueError, match="1 volumes but 2 map_ids"):
        AtlasFreeCNNDataset("train")
