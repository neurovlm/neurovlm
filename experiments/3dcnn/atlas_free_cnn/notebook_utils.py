"""Shared helpers for atlas-free CNN notebooks."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


DOMAIN_DIRS = {"pubmed": "01_pubmed", "nilearn": "02_nilearn", "neurovault": "03_neurovault"}
SPECIALIZED_BRANCHES = {
    "pubmed": "specialized_mixed_to_pubmed",
    "nilearn": "specialized_mixed_to_nilearn",
    "neurovault": "specialized_mixed_to_neurovault",
}
BRANCH_KINDS = ("baseline", "specialized")
NORMALIZED_STAGE3_DIRNAME = "stage3_normalized_specter"
CORRECTED_STAGE4_DIRNAME = "corrected_stage4_normalized_specter"
LEGACY_NORMALIZED_STAGE4_DIRNAME = "legacy_contrastive_initialized_stage4_normalized_specter"
NORMALIZED_STAGE3_CHECKPOINT = "best_val_normalized_recall_auc.pt"
CORRECTED_STAGE4_CHECKPOINT = "best_val_generation_normalized_auc.pt"
DEFAULT_ATLAS_FREE_HF_REPO = "neurovlm/atlas_free_cnn_dataset"
DEFAULT_TEXT_EMBEDDING_CONVENTION = "normalized_specter2"
LEGACY_TEXT_EMBEDDING_CONVENTION = "legacy_specter2"
NORMALIZED_TEXT_EMBEDDING_CONVENTION = "normalized_specter2"
LEGACY_SPECTER_CACHE_FILENAME = "specter_text_cache.pt"
NORMALIZED_SPECTER_CACHE_STEM = "specter2_stage3_stage4_emptycentered_unitnorm"
NORMALIZED_SPECTER_CACHE_FILENAME = "specter2_stage3_stage4_emptycentered_unitnorm.pt"
NORMALIZED_SPECTER_METADATA_FILENAME = f"{NORMALIZED_SPECTER_CACHE_STEM}_metadata.json"
NORMALIZED_SPECTER_VALIDATION_FILENAME = f"{NORMALIZED_SPECTER_CACHE_STEM}_validation.json"
NORMALIZED_SPECTER_INDEX_FILENAME = f"{NORMALIZED_SPECTER_CACHE_STEM}_index.csv"
NORMALIZED_SPECTER_PREPROCESSING = "empty_string_centered_l2_unit_normalized"
LEGACY_SPECTER_PREPROCESSING = "legacy_existing_cache_convention"
SPECTER2_ENCODER_MODEL = "allenai/specter2_aug2023refresh"
SPECTER2_ADAPTER_NAME = "adhoc_query"
SPECTER2_BASE_MODEL_REPO = "allenai/specter2_aug2023refresh_base"
SPECTER2_ADAPTER_REPO = "allenai/specter2_aug2023refresh_adhoc_query"
TEXT_EMBEDDING_DIM = 768

LOCKED_STAGE1_CHECKPOINT_NAMES = {
    "mixed_stage1a": "best_top1_dice.pt",
    "mixed_to_pubmed_stage1b": "best_top1_dice.pt",
    "mixed_to_nilearn_stage1b": "best_val_loss.pt",
    "mixed_to_neurovault_stage1b": "best_top5_dice.pt",
}

_LOCKED_REGISTRY_VARIANTS = {
    "mixed_stage1a": "mixed_baseline_raw_mse",
    "mixed_to_pubmed_stage1b": "mixed_to_pubmed",
    "mixed_to_nilearn_stage1b": "mixed_to_nilearn",
    "mixed_to_neurovault_stage1b": "mixed_to_neurovault",
}


def run_cmd(cmd: list[str | os.PathLike[str]], cwd: str | Path | None = None, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(map(str, cmd)))
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.returncode != 0:
        if result.stderr.strip():
            print(result.stderr.strip())
        if check:
            raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(map(str, cmd))}")
    return result


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def six_branch_specs() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for domain in ["pubmed", "nilearn", "neurovault"]:
        rows.append(
            {
                "domain": domain,
                "domain_dir": DOMAIN_DIRS[domain],
                "branch_kind": "baseline",
                "branch": "baseline_mixed_stage1a",
                "ae_registry_key": "mixed_stage1a",
            }
        )
        rows.append(
            {
                "domain": domain,
                "domain_dir": DOMAIN_DIRS[domain],
                "branch_kind": "specialized",
                "branch": SPECIALIZED_BRANCHES[domain],
                "ae_registry_key": f"mixed_to_{domain}_stage1b",
            }
        )
    return rows


def stage_output_dir(run_root: str | Path, domain: str, branch: str, stage: str, *, layout: str = "6a_normalized_corrected") -> Path:
    run_root = Path(run_root)
    if layout in {"6a_normalized_corrected", "normalized_specter", "6a_normalized_specter"}:
        dirname = NORMALIZED_STAGE3_DIRNAME if stage == "stage3" else CORRECTED_STAGE4_DIRNAME if stage == "stage4" else stage
    else:
        dirname = stage
    return run_root / DOMAIN_DIRS[domain] / branch / dirname


def stage_checkpoint_path(run_root: str | Path, domain: str, branch: str, stage: str, *, layout: str = "6a_normalized_corrected") -> Path:
    out_dir = stage_output_dir(run_root, domain, branch, stage, layout=layout)
    if stage == "stage3" and layout in {"6a_normalized_corrected", "normalized_specter", "6a_normalized_specter"}:
        return out_dir / "checkpoints" / NORMALIZED_STAGE3_CHECKPOINT
    if stage == "stage4" and layout in {"6a_normalized_corrected", "normalized_specter", "6a_normalized_specter"}:
        return out_dir / "checkpoints" / CORRECTED_STAGE4_CHECKPOINT
    if stage == "stage3":
        return out_dir / "checkpoints" / "best_ale_cnn.pt"
    return out_dir / "checkpoints" / "best_val_loss.pt"


def discover_stage_outputs(run_root: str | Path, *, layout: str = "6a_normalized_corrected") -> list[dict[str, Any]]:
    rows = []
    for spec in six_branch_specs():
        stage3_dir = stage_output_dir(run_root, spec["domain"], spec["branch"], "stage3", layout=layout)
        stage4_dir = stage_output_dir(run_root, spec["domain"], spec["branch"], "stage4", layout=layout)
        rows.append(
            {
                **spec,
                "stage3_dir": str(stage3_dir),
                "stage3_checkpoint": str(stage_checkpoint_path(run_root, spec["domain"], spec["branch"], "stage3", layout=layout)),
                "stage4_dir": str(stage4_dir),
                "stage4_checkpoint": str(stage_checkpoint_path(run_root, spec["domain"], spec["branch"], "stage4", layout=layout)),
            }
        )
    return rows


def split_dir_has_jsonl(path: str | Path) -> bool:
    path = Path(path)
    return all((path / name).exists() for name in ["train.jsonl", "val.jsonl", "test.jsonl"])


def primary_text(row: dict[str, Any]) -> dict[str, str]:
    positives = row.get("positive_texts") or []
    if positives:
        pos = positives[0]
        return {
            "text_id": str(pos.get("text_id") or pos.get("id") or ""),
            "text": str(pos.get("text") or pos.get("title") or ""),
        }
    return {
        "text_id": str(row.get("primary_text_id") or row.get("text_id") or ""),
        "text": str(row.get("primary_text") or row.get("text") or ""),
    }


def split_jsonl_fingerprint(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    rows = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append((str(row.get("map_id", "")), primary_text(row)["text_id"]))
    payload = json.dumps(sorted(rows), sort_keys=True, separators=(",", ":")).encode()
    return {
        "path": str(path),
        "exists": path.exists(),
        "rows": len(rows),
        "sha256": sha256_file(path),
        "fingerprint": hashlib.sha256(payload).hexdigest(),
    }


def split_file_fingerprints(split_dir: str | Path) -> dict[str, dict[str, Any]]:
    split_dir = Path(split_dir)
    return {split: split_jsonl_fingerprint(split_dir / f"{split}.jsonl") for split in ["train", "val", "test"]}


def print_split_files(split_dir: str | Path) -> dict[str, dict[str, Any]]:
    report = split_file_fingerprints(split_dir)
    print("Unified split files used:")
    for split, row in report.items():
        print(f"- {split}: {row['path']} sha256={row['sha256']} fingerprint={row['fingerprint']} rows={row['rows']}")
    return report


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


def _default_text_cache_dir(repo_dir: str | Path | None = None) -> Path:
    if repo_dir is None:
        repo_dir = Path(__file__).resolve().parents[3]
    return Path(repo_dir) / "experiments/3dcnn/atlas_free_cnn/cache/text_embeddings"


def _canonical_text_embedding_convention(convention: str | None) -> str:
    value = (convention or DEFAULT_TEXT_EMBEDDING_CONVENTION).strip().lower()
    aliases = {
        "normalized": NORMALIZED_TEXT_EMBEDDING_CONVENTION,
        "normalized_specter": NORMALIZED_TEXT_EMBEDDING_CONVENTION,
        "normalized_specter2": NORMALIZED_TEXT_EMBEDDING_CONVENTION,
        "legacy": LEGACY_TEXT_EMBEDDING_CONVENTION,
        "legacy_specter": LEGACY_TEXT_EMBEDDING_CONVENTION,
        "legacy_specter2": LEGACY_TEXT_EMBEDDING_CONVENTION,
    }
    if value not in aliases:
        raise ValueError(
            f"Unknown text embedding convention {convention!r}. "
            f"Expected {NORMALIZED_TEXT_EMBEDDING_CONVENTION!r} or {LEGACY_TEXT_EMBEDDING_CONVENTION!r}."
        )
    return aliases[value]


def resolve_text_embedding_cache(
    convention: str | None = DEFAULT_TEXT_EMBEDDING_CONVENTION,
    *,
    repo_dir: str | Path | None = None,
    local_cache_dir: str | Path | None = None,
    hf_repo: str = DEFAULT_ATLAS_FREE_HF_REPO,
    env_override: bool = True,
) -> dict[str, Any]:
    """Return the canonical atlas-free CNN SPECTER2 cache specification.

    Notebook 6a defines the normalized cache filenames and HF layout. Earlier
    notebooks should use this resolver instead of hardcoding a cache path.
    """

    convention = _canonical_text_embedding_convention(convention)
    local_dir = Path(local_cache_dir) if local_cache_dir is not None else _default_text_cache_dir(repo_dir)

    if convention == NORMALIZED_TEXT_EMBEDDING_CONVENTION:
        cache_name = NORMALIZED_SPECTER_CACHE_FILENAME
        local_cache_path = local_dir / cache_name
        override_env_vars = ["NEUROVLM_NORMALIZED_SPECTER_CACHE", "NEUROVLM_TEXT_EMBEDDING_CACHE"]
        spec = {
            "convention": convention,
            "cache_name": cache_name,
            "hf_repo": hf_repo,
            "hf_path": f"text_embeddings/{cache_name}",
            "hf_candidate_paths": [f"text_embeddings/{cache_name}", cache_name],
            "metadata_hf_path": f"text_embeddings/{NORMALIZED_SPECTER_METADATA_FILENAME}",
            "validation_hf_path": f"text_embeddings/{NORMALIZED_SPECTER_VALIDATION_FILENAME}",
            "index_hf_path": f"text_embeddings/{NORMALIZED_SPECTER_INDEX_FILENAME}",
            "metadata_local_path": str(local_dir / NORMALIZED_SPECTER_METADATA_FILENAME),
            "validation_local_path": str(local_dir / NORMALIZED_SPECTER_VALIDATION_FILENAME),
            "index_local_path": str(local_dir / NORMALIZED_SPECTER_INDEX_FILENAME),
            "preprocessing": NORMALIZED_SPECTER_PREPROCESSING,
            "expected_dim": TEXT_EMBEDDING_DIM,
            "expect_unit_norm": True,
            "encoder_model": SPECTER2_ENCODER_MODEL,
            "adapter_name": SPECTER2_ADAPTER_NAME,
            "base_model_hf_repo": SPECTER2_BASE_MODEL_REPO,
            "adapter_hf_repo": SPECTER2_ADAPTER_REPO,
        }
    else:
        cache_name = LEGACY_SPECTER_CACHE_FILENAME
        local_cache_path = local_dir / cache_name
        override_env_vars = ["NEUROVLM_LEGACY_TEXT_EMBEDDING_CACHE", "NEUROVLM_TEXT_EMBEDDING_CACHE"]
        spec = {
            "convention": convention,
            "cache_name": cache_name,
            "hf_repo": hf_repo,
            "hf_path": f"text_embeddings/{cache_name}",
            "hf_candidate_paths": [f"text_embeddings/{cache_name}", cache_name, f"cache/text_embeddings/{cache_name}"],
            "metadata_hf_path": "",
            "validation_hf_path": "",
            "index_hf_path": "",
            "metadata_local_path": "",
            "validation_local_path": "",
            "index_local_path": "",
            "preprocessing": LEGACY_SPECTER_PREPROCESSING,
            "expected_dim": TEXT_EMBEDDING_DIM,
            "expect_unit_norm": False,
            "encoder_model": SPECTER2_ENCODER_MODEL,
            "adapter_name": SPECTER2_ADAPTER_NAME,
            "base_model_hf_repo": SPECTER2_BASE_MODEL_REPO,
            "adapter_hf_repo": SPECTER2_ADAPTER_REPO,
        }

    if env_override:
        for env_var in override_env_vars:
            value = os.environ.get(env_var, "").strip()
            if value:
                local_cache_path = Path(value).expanduser()
                spec["local_cache_override_env"] = env_var
                break
    spec["local_cache_path"] = str(local_cache_path)
    spec["local_cache_dir"] = str(local_cache_path.parent)
    return spec


def print_relevant_text_embedding_hf_files(hf_repo: str = DEFAULT_ATLAS_FREE_HF_REPO) -> list[str]:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo_id=hf_repo, repo_type="dataset")
    print("Potentially relevant Hugging Face files")
    for name in files:
        lower = name.lower()
        if "specter" in lower or lower.endswith(".jsonl") or "manifest" in lower or lower.endswith(".parquet") or "metadata" in lower:
            print("-", name)
    return files


def download_text_embedding_cache(spec: dict[str, Any], *, include_sidecars: bool = True) -> Path:
    """Download the resolved text embedding cache and optional sidecars."""

    target = Path(spec["local_cache_path"]).expanduser()
    downloaded = hf_download_first_available(
        list(spec.get("hf_candidate_paths") or [spec["hf_path"], spec["cache_name"]]),
        target.parent,
        dataset_repo=str(spec["hf_repo"]),
    )
    _link_or_copy(downloaded, target)

    if include_sidecars:
        for key in ["metadata", "validation", "index"]:
            hf_path = str(spec.get(f"{key}_hf_path") or "")
            local_path = str(spec.get(f"{key}_local_path") or "")
            if not hf_path or not local_path:
                continue
            try:
                sidecar = hf_download_first_available(
                    [hf_path, Path(hf_path).name],
                    target.parent,
                    dataset_repo=str(spec["hf_repo"]),
                )
                _link_or_copy(sidecar, Path(local_path))
            except Exception as exc:
                print(f"Optional {spec['convention']} sidecar not downloaded ({Path(hf_path).name}): {exc}")
    return target


def build_normalized_specter2_cache(
    spec: dict[str, Any],
    *,
    repo_dir: str | Path,
    python_executable: str | Path = sys.executable,
    force: bool = False,
) -> Path:
    """Build the canonical normalized SPECTER2 cache using the existing builder."""

    if spec["convention"] != NORMALIZED_TEXT_EMBEDDING_CONVENTION:
        raise ValueError("Only the normalized SPECTER2 convention can be built by this helper.")
    repo_dir = Path(repo_dir)
    target = Path(spec["local_cache_path"])
    cmd: list[str | os.PathLike[str]] = [
        python_executable,
        repo_dir / "experiments/3dcnn/atlas_free_cnn/data_building/build_normalized_specter2_cache.py",
        "--repo-id",
        str(spec["hf_repo"]),
        "--repo-dir",
        str(repo_dir),
        "--local-dir",
        str(repo_dir / "experiments/3dcnn/atlas_free_cnn/cache/hf_normalized_specter2"),
        "--output-dir",
        str(target.parent),
        "--output-basename",
        NORMALIZED_SPECTER_CACHE_STEM,
    ]
    if force:
        cmd.append("--force")
    run_cmd(cmd, cwd=repo_dir)
    return target


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
            print_split_files(candidate)
            return candidate
    try:
        hf_split_dir = ensure_hf_unified_splits(
            dataset_repo=dataset_repo,
            local_unified_cache_dir=local_unified_cache_dir,
            local_split_dir=local_split_dir,
            local_pack_dir=local_pack_dir,
        )
        if split_dir_has_jsonl(hf_split_dir):
            print_split_files(hf_split_dir)
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


def required_text_records_from_jsonls(paths: list[str | Path]) -> dict[str, Any]:
    """Collect primary positive text IDs/text strings from unified split JSONLs."""

    records: list[dict[str, str]] = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                primary = primary_text(row)
                text_id = str(primary.get("text_id") or "")
                text = str(primary.get("text") or "")
                if text_id or text:
                    records.append({"text_id": text_id, "text": text})
    return {
        "records": records,
        "text_ids": {row["text_id"] for row in records if row["text_id"]},
        "texts": {row["text"] for row in records if row["text"]},
    }


def _cache_embeddings_ids_and_texts(payload: Any) -> tuple[Any, set[str], set[str], dict[str, Any]]:
    import torch

    metadata = dict(payload.get("metadata", {})) if isinstance(payload, dict) and isinstance(payload.get("metadata"), dict) else {}
    if isinstance(payload, dict) and torch.is_tensor(payload.get("embeddings")):
        text_ids = {str(v) for v in payload.get("text_ids", []) if str(v)}
        texts = {str(v) for v in payload.get("texts", []) if str(v)}
        return payload["embeddings"].float(), text_ids, texts, metadata
    if isinstance(payload, dict):
        for key in ["processed_embedding_by_text_id", "embedding_by_text_id", "text_id_to_embedding", "embeddings_by_text_id"]:
            value = payload.get(key)
            if isinstance(value, dict) and value:
                return torch.stack([torch.as_tensor(v, dtype=torch.float32).flatten() for v in value.values()]), {str(k) for k in value}, set(), metadata
        for key in ["processed_embedding_by_text", "embedding_by_text", "text_to_embedding", "embeddings_by_text"]:
            value = payload.get(key)
            if isinstance(value, dict) and value:
                return torch.stack([torch.as_tensor(v, dtype=torch.float32).flatten() for v in value.values()]), set(), {str(k) for k in value}, metadata
        records = payload.get("records")
        if isinstance(records, list) and records:
            vectors = []
            text_ids = set()
            texts = set()
            for row in records:
                if not isinstance(row, dict):
                    continue
                vector = row.get("processed_embedding", row.get("embedding"))
                if vector is None:
                    continue
                vectors.append(torch.as_tensor(vector, dtype=torch.float32))
                if row.get("text_id"):
                    text_ids.add(str(row["text_id"]))
                if row.get("text"):
                    texts.add(str(row["text"]))
            if vectors:
                return torch.stack(vectors), text_ids, texts, metadata
    if isinstance(payload, dict) and payload and all(torch.is_tensor(v) for v in payload.values()):
        return torch.stack([v.float().flatten() for v in payload.values()]), set(), {str(k) for k in payload}, metadata
    raise TypeError("Unsupported text embedding cache format")


def validate_text_embedding_cache(
    cache: str | Path | dict[str, Any],
    *,
    convention: str | None = None,
    required_text_ids: set[str] | None = None,
    required_texts: set[str] | None = None,
    expected_dim: int = 768,
    expect_unit_norm: bool | None = None,
    mean_norm_tol: float = 1e-3,
) -> dict[str, Any]:
    import torch

    spec = cache if isinstance(cache, dict) and "local_cache_path" in cache else None
    if spec is not None:
        path = Path(str(spec["local_cache_path"]))
        convention = str(spec.get("convention") or convention or DEFAULT_TEXT_EMBEDDING_CONVENTION)
        expected_dim = int(spec.get("expected_dim", expected_dim))
        if expect_unit_norm is None:
            expect_unit_norm = bool(spec.get("expect_unit_norm", False))
    else:
        path = Path(cache)  # type: ignore[arg-type]
        convention = _canonical_text_embedding_convention(convention)
        if expect_unit_norm is None:
            expect_unit_norm = convention == NORMALIZED_TEXT_EMBEDDING_CONVENTION

    payload = torch.load(path, map_location="cpu", weights_only=False)
    embeddings, text_ids, texts, payload_metadata = _cache_embeddings_ids_and_texts(payload)
    if embeddings.ndim != 2 or embeddings.shape[1] != int(expected_dim):
        raise RuntimeError(f"Wrong cache loaded: expected N x {expected_dim}, got {tuple(embeddings.shape)}")
    if not torch.isfinite(embeddings).all():
        raise RuntimeError("Wrong cache loaded: text embeddings contain NaNs or infinities")
    norms = embeddings.norm(dim=1)
    stats = {
        "path": str(path),
        "sha256": sha256_file(path),
        "convention": _canonical_text_embedding_convention(convention),
        "n": int(embeddings.shape[0]),
        "dim": int(embeddings.shape[1]),
        "norm_mean": float(norms.mean().item()),
        "norm_std": float(norms.std(unbiased=False).item()),
        "norm_min": float(norms.min().item()),
        "norm_max": float(norms.max().item()),
        "fraction_within_1e_3": float((norms.sub(1).abs() <= 1e-3).float().mean().item()),
        "text_id_count": len(text_ids),
        "text_key_count": len(texts),
    }
    if expect_unit_norm:
        if abs(stats["norm_mean"] - 1.0) > float(mean_norm_tol):
            raise RuntimeError(f"Wrong cache loaded: mean norm is not approximately 1.0: {stats}")
        if stats["fraction_within_1e_3"] < 0.999:
            raise RuntimeError(f"Wrong cache loaded: vectors are not approximately unit-normalized: {stats}")
    required_text_ids = {str(v) for v in (required_text_ids or set()) if str(v)}
    if required_text_ids:
        if text_ids:
            missing = sorted(required_text_ids - text_ids)
            if missing:
                raise RuntimeError(f"Text embedding cache missing {len(missing)} required manifest text IDs; first={missing[:5]}")
            stats["required_text_ids_present"] = True
        elif expect_unit_norm:
            raise RuntimeError("Normalized text embedding cache does not expose text IDs for alignment validation.")
        else:
            stats["required_text_ids_present"] = "not_available_in_cache"
    required_texts = {str(v) for v in (required_texts or set()) if str(v)}
    if required_texts:
        if texts:
            missing_texts = sorted(required_texts - texts)
            if missing_texts:
                raise RuntimeError(f"Text embedding cache missing {len(missing_texts)} required manifest text strings; first={missing_texts[:3]}")
            stats["required_texts_present"] = True
        else:
            stats["required_texts_present"] = "not_available_in_cache"

    if spec is not None:
        for key in ["metadata", "validation", "index"]:
            sidecar = str(spec.get(f"{key}_local_path") or "")
            if sidecar and Path(sidecar).exists():
                stats[f"{key}_sha256"] = sha256_file(sidecar)
    return {"stats": stats, "metadata": payload_metadata}


def validate_normalized_specter_cache(
    path: str | Path,
    *,
    required_text_ids: set[str] | None = None,
    expected_dim: int = 768,
    mean_norm_tol: float = 1e-3,
) -> dict[str, Any]:
    return validate_text_embedding_cache(
        path,
        convention=NORMALIZED_TEXT_EMBEDDING_CONVENTION,
        required_text_ids=required_text_ids,
        expected_dim=expected_dim,
        expect_unit_norm=True,
        mean_norm_tol=mean_norm_tol,
    )


def validate_legacy_specter_cache(
    path: str | Path,
    *,
    required_text_ids: set[str] | None = None,
    required_texts: set[str] | None = None,
    expected_dim: int = 768,
) -> dict[str, Any]:
    return validate_text_embedding_cache(
        path,
        convention=LEGACY_TEXT_EMBEDDING_CONVENTION,
        required_text_ids=required_text_ids,
        required_texts=required_texts,
        expected_dim=expected_dim,
        expect_unit_norm=False,
    )


def text_embedding_metadata_fields(spec: dict[str, Any], audit: dict[str, Any] | None = None) -> dict[str, Any]:
    stats = (audit or {}).get("stats", {}) if isinstance(audit, dict) else {}
    return {
        "text_embedding_convention": spec["convention"],
        "text_embedding_cache_name": spec["cache_name"],
        "text_embedding_hf_repo": spec["hf_repo"],
        "text_embedding_hf_path": spec["hf_path"],
        "text_embedding_cache_checksum": stats.get("sha256", ""),
        "text_embedding_metadata_path": spec.get("metadata_hf_path", ""),
        "text_embedding_metadata_checksum": stats.get("metadata_sha256", ""),
        "text_embedding_preprocessing": spec["preprocessing"],
        "text_embedding_dim": int(spec["expected_dim"]),
        "expect_unit_norm": bool(spec["expect_unit_norm"]),
        "text_embedding_cache_local_path": spec["local_cache_path"],
    }


def locked_stage1_checkpoint_selection(ae_run_registry: dict[str, dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, checkpoint_name in LOCKED_STAGE1_CHECKPOINT_NAMES.items():
        variant = _LOCKED_REGISTRY_VARIANTS[key]
        spec = (ae_run_registry or {}).get(variant, {})
        run_dir = Path(str(spec.get("run_dir", ""))).expanduser() if spec.get("run_dir") else None
        checkpoint_path = ""
        checksum = ""
        exists = False
        if run_dir:
            ckpt_dir = run_dir / "checkpoints" if (run_dir / "checkpoints").exists() else run_dir
            candidate = ckpt_dir / checkpoint_name
            checkpoint_path = str(candidate)
            exists = candidate.exists()
            checksum = sha256_file(candidate) if exists else ""
        out[key] = {
            "variant": variant,
            "stage": "stage1a" if key == "mixed_stage1a" else "stage1b",
            "training_domain": "mixed" if key == "mixed_stage1a" else key.removeprefix("mixed_to_").removesuffix("_stage1b"),
            "checkpoint_name": checkpoint_name,
            "checkpoint_path": checkpoint_path,
            "sha256": checksum,
            "exists": exists,
            "locked_downstream_selection": True,
        }
    return out
