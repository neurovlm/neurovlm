"""Shared helpers for atlas-free CNN notebooks."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def split_dir_has_jsonl(path: str | Path) -> bool:
    path = Path(path)
    return all((path / name).exists() for name in ["train.jsonl", "val.jsonl", "test.jsonl"])


def hf_download_first_available(
    filenames: list[str],
    local_dir: str | Path,
    *,
    dataset_repo: str,
) -> Path:
    from huggingface_hub import hf_hub_download

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    errors = []
    for filename in filenames:
        try:
            path = hf_hub_download(
                repo_id=dataset_repo,
                repo_type="dataset",
                filename=filename,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
            )
            return Path(path)
        except Exception as exc:
            errors.append(f"{filename}: {exc}")
    raise FileNotFoundError("Could not download any candidate from HF:\n" + "\n".join(errors))


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dst.resolve():
        return
    try:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def ensure_hf_unified_splits(
    *,
    dataset_repo: str,
    local_unified_cache_dir: str | Path,
    local_split_dir: str | Path,
    local_pack_dir: str | Path,
) -> Path:
    print(f"Downloading atlas-free CNN split JSONLs from Hugging Face: {dataset_repo}")
    local_unified_cache_dir = Path(local_unified_cache_dir)
    local_split_dir = Path(local_split_dir)
    local_pack_dir = Path(local_pack_dir)
    local_split_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        downloaded = hf_download_first_available(
            [f"splits/{split}.jsonl", f"unified_jsonl_rebuild/splits/{split}.jsonl", f"{split}.jsonl"],
            local_unified_cache_dir,
            dataset_repo=dataset_repo,
        )
        _link_or_copy(downloaded, local_split_dir / f"{split}.jsonl")
    for name in ["train_map_ids.json", "val_map_ids.json", "test_map_ids.json"]:
        try:
            downloaded = hf_download_first_available(
                [f"splits/{name}", f"unified_jsonl_rebuild/splits/{name}", name],
                local_unified_cache_dir,
                dataset_repo=dataset_repo,
            )
            _link_or_copy(downloaded, local_split_dir / name)
        except Exception as exc:
            print(f"Optional split sidecar not downloaded ({name}): {exc}")
    try:
        downloaded_volume = hf_download_first_available(
            ["atlas_free_cnn_volumes.pt", "hf_atlas_free_cnn/atlas_free_cnn_volumes.pt", "hf_atlas_free_cnn_rebuild/atlas_free_cnn_volumes.pt"],
            local_pack_dir,
            dataset_repo=dataset_repo,
        )
        target_volume = local_pack_dir / "atlas_free_cnn_volumes.pt"
        _link_or_copy(downloaded_volume, target_volume)
        print("Volume tensor available at:", target_volume)
    except Exception as exc:
        print("WARNING: split JSONLs downloaded, but volume tensor was not prepared:", exc)
        print("Training will fail unless tensor_path values inside JSONL resolve to an accessible tensor file.")
    return local_split_dir


def unified_split_candidates(repo_dir: str | Path, drive_root: str | Path) -> list[Path]:
    repo_dir = Path(repo_dir)
    drive_root = Path(drive_root)
    return [
        repo_dir / "experiments/3dcnn/atlas_free_cnn/cache/unified_jsonl_rebuild/splits",
        repo_dir / "experiments/3dcnn/atlas_free_cnn/cache/unified_jsonl/splits",
        drive_root / "experiments/3dcnn/atlas_free_cnn/cache/unified_jsonl_rebuild/splits",
        drive_root / "experiments/3dcnn/atlas_free_cnn/cache/unified_jsonl/splits",
        drive_root / "atlas_free_cnn/cache/unified_jsonl_rebuild/splits",
        drive_root / "atlas_free_cnn/cache/unified_jsonl/splits",
        drive_root / "cache/unified_jsonl_rebuild/splits",
        drive_root / "cache/unified_jsonl/splits",
        drive_root / "data_atlas_free_cnn/unified_jsonl_rebuild/splits",
        drive_root / "data_atlas_free_cnn/unified_jsonl/splits",
        drive_root / "data_atlas_free_cnn/cache/unified_jsonl_rebuild/splits",
        drive_root / "data_atlas_free_cnn/cache/unified_jsonl/splits",
        drive_root / "data_ale_3dcnn/unified_jsonl_rebuild/splits",
        drive_root / "data_ale_3dcnn/unified_jsonl/splits",
    ]


def discover_unified_split_dir(
    *,
    repo_dir: str | Path,
    drive_root: str | Path,
    dataset_repo: str,
    local_unified_cache_dir: str | Path,
    local_split_dir: str | Path,
    local_pack_dir: str | Path,
    env_var: str = "NEUROVLM_UNIFIED_SPLIT_DIR",
) -> Path:
    override = os.environ.get(env_var, "").strip()
    candidates = []
    if override:
        candidates.append(Path(override).expanduser())
    candidates.extend(unified_split_candidates(repo_dir, drive_root))
    for candidate in candidates:
        if split_dir_has_jsonl(candidate):
            return candidate
    try:
        hf_split_dir = ensure_hf_unified_splits(
            dataset_repo=dataset_repo,
            local_unified_cache_dir=local_unified_cache_dir,
            local_split_dir=local_split_dir,
            local_pack_dir=local_pack_dir,
        )
        if split_dir_has_jsonl(hf_split_dir):
            return hf_split_dir
    except Exception as exc:
        hf_error = exc
    else:
        hf_error = None
    checked = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Could not find unified dataset split JSONL files locally, and Hugging Face fallback did not produce them. "
        "Expected train.jsonl, val.jsonl, and test.jsonl in one of:\n"
        f"{checked}\n\n"
        f"HF dataset repo tried: {dataset_repo}\n"
        f"HF fallback error: {hf_error}\n\n"
        f"If your splits are elsewhere, set {env_var} before running this cell."
    )
