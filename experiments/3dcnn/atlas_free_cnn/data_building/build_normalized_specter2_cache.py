#!/usr/bin/env python
"""Build an empty-string-centered, unit-normalized SPECTER2 cache.

The builder encodes the exact primary texts from the three unified split
manifests, subtracts the SPECTER2 empty-string embedding, and L2-normalizes the
result. The output includes ID- and text-keyed indexes used by Stage 3 and 4.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None

EXPECTED_DIM = 768
DEFAULT_REPO_ID = "neurovlm/atlas_free_cnn_dataset"
DEFAULT_ENCODER = "allenai/specter2_aug2023refresh"
DEFAULT_ADAPTER = "adhoc_query"
DEFAULT_OUTPUT_BASENAME = "specter2_stage3_stage4_emptycentered_unitnorm"
EPSILON = 1e-12
RANDOM_SEED = 20260626


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def import_versions() -> dict[str, str]:
    versions = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
    }
    for module_name in ["huggingface_hub", "transformers", "adapters", "pandas", "numpy"]:
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # pragma: no cover - optional dependency reporting
            versions[module_name] = f"unavailable: {exc}"
    return versions


def git_commit(repo_dir: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def hf_model_sha(repo_id: str | None) -> str:
    if not repo_id:
        return ""
    try:
        from huggingface_hub import HfApi

        return str(HfApi().model_info(repo_id=repo_id).sha or "")
    except Exception:
        return ""


def hf_file_table(repo_id: str) -> list[dict[str, Any]]:
    from huggingface_hub import HfApi

    api = HfApi()
    rows = []
    for item in api.list_repo_tree(repo_id=repo_id, repo_type="dataset", recursive=True, expand=True):
        path = getattr(item, "path", "")
        if not path:
            continue
        lfs = getattr(item, "lfs", None) or {}
        rows.append(
            {
                "path": path,
                "size": getattr(item, "size", None),
                "blob_id": getattr(item, "blob_id", None),
                "lfs_sha256": lfs.get("sha256") if isinstance(lfs, dict) else None,
                "relevant": is_relevant_hf_path(path),
            }
        )
    rows.sort(key=lambda r: r["path"])
    return rows


def is_relevant_hf_path(path: str) -> bool:
    name = Path(path).name.lower()
    lower = path.lower()
    return (
        "specter" in lower
        or name in {"train.jsonl", "val.jsonl", "test.jsonl"}
        or name.endswith("_map_ids.json")
        or name in {
            "atlas_free_cnn_rows.parquet",
            "atlas_free_cnn_text_pairs.parquet",
            "atlas_free_cnn_manifest.json",
            "preprocessing_audit.json",
        }
        or "metadata" in lower
        or "manifest" in lower
    )


def print_hf_table(rows: list[dict[str, Any]]) -> None:
    print("\nPotentially relevant Hugging Face files")
    print("path,size,lfs_sha256,blob_id")
    for row in rows:
        if row["relevant"]:
            print(f"{row['path']},{row.get('size')},{row.get('lfs_sha256') or ''},{row.get('blob_id') or ''}")


def download_file(repo_id: str, filename: str, local_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    local_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
    )


def primary_positive(row: dict[str, Any]) -> dict[str, Any]:
    positives = row.get("positive_texts") or []
    if not positives:
        raise ValueError(f"Map row has no positive_texts: {row.get('map_id')}")
    return positives[0]


def domain_from_source(value: str) -> str:
    source = str(value or "").lower()
    if source == "pubmed" or source.startswith("pubmed"):
        return "pubmed"
    if source == "nilearn" or source.startswith("nilearn"):
        return "nilearn"
    if source == "neurovault" or source.startswith("neurovault"):
        return "neurovault"
    if source.startswith("network"):
        return "networks"
    return source or "unknown"


def collect_text_records(split_paths: dict[str, Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_id: dict[str, dict[str, Any]] = {}
    map_rows: list[dict[str, Any]] = []
    duplicate_conflicts: list[dict[str, Any]] = []
    for split, path in split_paths.items():
        for row in read_jsonl(path):
            pos = primary_positive(row)
            text = str(pos.get("text", ""))
            if not text:
                raise ValueError(f"Primary text is empty for map_id={row.get('map_id')} split={split}")
            text_id = str(pos.get("text_id") or pos.get("id") or sha256_bytes(text.encode("utf-8"))[:16])
            domain = domain_from_source(str(row.get("source", pos.get("source", ""))))
            publication_id = str(row.get("pmid") or pos.get("pmid") or row.get("collection_id") or row.get("group_id") or "")
            if text_id in by_id and by_id[text_id]["text"] != text:
                duplicate_conflicts.append(
                    {
                        "text_id": text_id,
                        "first_preview": by_id[text_id]["text"][:160],
                        "second_preview": text[:160],
                    }
                )
            rec = by_id.setdefault(
                text_id,
                {
                    "text_id": text_id,
                    "text": text,
                    "source_domains": set(),
                    "publication_or_group_ids": set(),
                    "map_ids": [],
                    "splits": set(),
                },
            )
            rec["source_domains"].add(domain)
            if publication_id:
                rec["publication_or_group_ids"].add(publication_id)
            rec["map_ids"].append(str(row.get("map_id", "")))
            rec["splits"].add(split)
            map_rows.append(
                {
                    "split": split,
                    "domain": domain,
                    "map_id": str(row.get("map_id", "")),
                    "primary_text_id": text_id,
                    "primary_text_sha256": sha256_bytes(text.encode("utf-8")),
                }
            )
    if duplicate_conflicts:
        raise RuntimeError(f"Duplicate text IDs map to different text strings: {duplicate_conflicts[:5]}")
    records = []
    for rec in by_id.values():
        records.append(
            {
                "text_id": rec["text_id"],
                "text": rec["text"],
                "source_domains": sorted(rec["source_domains"]),
                "publication_or_group_ids": sorted(rec["publication_or_group_ids"]),
                "map_ids": sorted(set(rec["map_ids"])),
                "splits": sorted(rec["splits"]),
            }
        )
    records.sort(key=lambda r: r["text_id"])
    return records, map_rows


def numeric_stats(emb: torch.Tensor, *, seed: int = RANDOM_SEED, pairwise_n: int = 2048) -> dict[str, Any]:
    emb = emb.float()
    norms = emb.norm(dim=1)
    generator = torch.Generator().manual_seed(seed)
    n = emb.shape[0]
    subset_n = min(int(pairwise_n), n)
    idx = torch.randperm(n, generator=generator)[:subset_n]
    sub = F.normalize(emb[idx], dim=1, eps=1e-8)
    sim = sub @ sub.T
    mask = ~torch.eye(subset_n, dtype=torch.bool)
    per_dim_mean = emb.mean(dim=0)
    return {
        "n_vectors": int(emb.shape[0]),
        "embedding_dim": int(emb.shape[1]) if emb.ndim == 2 else None,
        "norm_mean": float(norms.mean().item()),
        "norm_std": float(norms.std(unbiased=False).item()),
        "norm_min": float(norms.min().item()),
        "norm_max": float(norms.max().item()),
        "fraction_norm_within_1e_4_of_1": float((norms.sub(1).abs() <= 1e-4).float().mean().item()),
        "fraction_norm_within_1e_3_of_1": float((norms.sub(1).abs() <= 1e-3).float().mean().item()),
        "nan_count": int(torch.isnan(emb).sum().item()),
        "inf_count": int(torch.isinf(emb).sum().item()),
        "per_dimension_mean_mean": float(per_dim_mean.mean().item()),
        "per_dimension_mean_abs_mean": float(per_dim_mean.abs().mean().item()),
        "mean_pairwise_cosine_sample": float(sim[mask].mean().item()) if subset_n > 1 else None,
        "pairwise_sample_n": int(subset_n),
    }


def split_encoder_name(model_name: str, adapter_name: str | None) -> tuple[str, str | None, str]:
    if model_name.endswith("_candidate"):
        base = model_name[: -len("_candidate")]
        return base, model_name, "candidate"
    if adapter_name:
        base = model_name.removesuffix("_base")
        adapter_id = adapter_name if "/" in adapter_name else f"{base}_{adapter_name}"
        return base, adapter_id, adapter_name
    return model_name.removesuffix("_base"), None, ""


def encode_texts_specter2(
    texts: list[str],
    *,
    model_name: str,
    adapter_name: str | None,
    device: str,
    batch_size: int,
    max_length: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    from adapters import AutoAdapterModel
    from transformers import AutoModel, AutoTokenizer

    base_model, adapter_id, adapter_label = split_encoder_name(model_name, adapter_name)
    base_repo = f"{base_model}_base"
    total_batches = (len(texts) + batch_size - 1) // batch_size
    print(
        f"Loading SPECTER2 encoder: base={base_repo} adapter={adapter_id or '<none>'} "
        f"device={device} texts={len(texts):,} batch_size={batch_size} batches={total_batches:,}",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_repo)
    if adapter_id:
        model = AutoAdapterModel.from_pretrained(base_repo)
        model.load_adapter(adapter_id, source="hf", load_as="specter2", set_active=True)
    else:
        model = AutoModel.from_pretrained(base_repo)
    torch_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device)
    model = model.to(torch_device).eval()
    outputs = []
    batch_starts = range(0, len(texts), batch_size)
    if tqdm is not None:
        batch_starts = tqdm(batch_starts, total=total_batches, desc="Encoding SPECTER2", unit="batch")
    for batch_index, start in enumerate(batch_starts, start=1):
        batch = texts[start : start + batch_size]
        tokens = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        tokens = {k: v.to(torch_device) for k, v in tokens.items()}
        with torch.inference_mode():
            hidden = model(**tokens).last_hidden_state
            pooled = hidden[:, 0].float().cpu()
        outputs.append(pooled)
        if tqdm is None and (batch_index == 1 or batch_index % 25 == 0 or batch_index == total_batches):
            encoded = min(start + len(batch), len(texts))
            print(f"Encoded {encoded:,}/{len(texts):,} SPECTER2 texts ({batch_index:,}/{total_batches:,} batches)", flush=True)
    info = {
        "text_encoder_model_name": model_name,
        "base_model": base_repo,
        "adapter_name": adapter_label,
        "adapter_id": adapter_id,
        "pooling_method": "cls_token",
        "maximum_token_length": max_length,
        "tokenizer_sep_token": tokenizer.sep_token,
    }
    return torch.cat(outputs, dim=0), info


def output_paths(output_dir: Path, basename: str) -> dict[str, Path]:
    return {
        "pt": output_dir / f"{basename}.pt",
        "metadata": output_dir / f"{basename}_metadata.json",
        "index": output_dir / f"{basename}_index.csv",
        "validation": output_dir / f"{basename}_validation.json",
    }


def fail_if_would_overwrite(paths: dict[str, Path], force: bool) -> None:
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing and not force:
        raise FileExistsError("Refusing to overwrite existing normalized cache artifacts:\n" + "\n".join(existing))


def build_payload(
    records: list[dict[str, Any]],
    embeddings: torch.Tensor,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    embeddings = embeddings.float().contiguous()
    text_ids = [r["text_id"] for r in records]
    texts = [r["text"] for r in records]
    source_domains = [";".join(r["source_domains"]) for r in records]
    return {
        "embeddings": embeddings,
        "text_ids": text_ids,
        "texts": texts,
        "source_domains": source_domains,
        "metadata": metadata,
        "embedding_by_text": {text: embeddings[i] for i, text in enumerate(texts)},
        "embedding_by_text_id": {text_id: embeddings[i] for i, text_id in enumerate(text_ids)},
    }


def validate_output(
    records: list[dict[str, Any]],
    map_rows: list[dict[str, Any]],
    centered: torch.Tensor,
    normalized: torch.Tensor,
) -> dict[str, Any]:
    if normalized.ndim != 2 or normalized.shape[1] != EXPECTED_DIM:
        raise RuntimeError(f"Output embeddings must be N x 768, got {tuple(normalized.shape)}")
    if not torch.isfinite(normalized).all():
        raise RuntimeError("Output embeddings contain NaNs or infinities")
    text_ids = [r["text_id"] for r in records]
    if len(text_ids) != len(set(text_ids)):
        raise RuntimeError("Unexpected duplicate text IDs in output records")
    required = {row["primary_text_id"] for row in map_rows}
    missing = sorted(required - set(text_ids))
    if missing:
        raise RuntimeError(f"Output cache missing {len(missing)} required manifest text IDs")
    output_stats = numeric_stats(normalized)
    if output_stats["fraction_norm_within_1e_3_of_1"] < 0.999:
        raise RuntimeError(f"Output vectors are not approximately unit-normalized: {output_stats}")
    validation = {
        "output": output_stats,
        "after_empty_string_centering": numeric_stats(centered),
        "one_output_vector_per_unique_text_id": len(records) == len(text_ids) == len(set(text_ids)),
        "all_required_manifest_text_ids_present": not missing,
        "domain_norm_statistics": {},
    }
    for domain in sorted({d for rec in records for d in rec["source_domains"]}):
        idx = [i for i, rec in enumerate(records) if domain in rec["source_domains"]]
        validation["domain_norm_statistics"][domain] = numeric_stats(normalized[idx])
    return validation


def prepare_index(records: list[dict[str, Any]], embeddings: torch.Tensor) -> list[dict[str, Any]]:
    rows = []
    for rec, emb in zip(records, embeddings):
        rows.append(
            {
                "text_id": rec["text_id"],
                "text_sha256": sha256_bytes(rec["text"].encode("utf-8")),
                "source_domains": ";".join(rec["source_domains"]),
                "publication_or_group_ids": ";".join(rec["publication_or_group_ids"]),
                "n_maps": len(rec["map_ids"]),
                "splits": ";".join(rec["splits"]),
                "embedding_norm": float(emb.norm().item()),
                "text_preview": rec["text"][:240].replace("\n", " "),
            }
        )
    return rows


def maybe_upload(paths: dict[str, Path], repo_id: str, repo_prefix: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    print("\nUpload manifest")
    for local_path in paths.values():
        repo_path = f"{repo_prefix.rstrip('/')}/{local_path.name}"
        print(
            json.dumps(
                {
                    "local_path": str(local_path),
                    "repo_path": repo_path,
                    "size": local_path.stat().st_size,
                    "sha256": sha256_file(local_path),
                },
                indent=2,
            )
        )
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="dataset",
        )
    repo_files = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
    missing = [f"{repo_prefix.rstrip('/')}/{path.name}" for path in paths.values() if f"{repo_prefix.rstrip('/')}/{path.name}" not in repo_files]
    if missing:
        raise RuntimeError(f"Upload verification failed; missing files: {missing}")
    print("Upload verification passed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--repo-dir", default=str(Path.cwd()))
    parser.add_argument("--local-dir", default="experiments/3dcnn/atlas_free_cnn/cache/hf_normalized_specter2")
    parser.add_argument("--output-dir", default="experiments/3dcnn/atlas_free_cnn/cache/text_embeddings")
    parser.add_argument("--output-basename", default=DEFAULT_OUTPUT_BASENAME)
    parser.add_argument("--encoder-model", default=DEFAULT_ENCODER)
    parser.add_argument("--adapter-name", default=DEFAULT_ADAPTER)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--upload-prefix", default="text_embeddings")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_dir = Path(args.repo_dir).expanduser().resolve()
    local_dir = Path(args.local_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    paths = output_paths(output_dir, args.output_basename)
    fail_if_would_overwrite(paths, args.force)

    repo_files = hf_file_table(args.repo_id)
    print_hf_table(repo_files)

    required_names = ["train.jsonl", "val.jsonl", "test.jsonl"]
    optional_names = [
        "atlas_free_cnn_rows.parquet",
        "atlas_free_cnn_text_pairs.parquet",
        "atlas_free_cnn_manifest.json",
        "preprocessing_audit.json",
        "train_map_ids.json",
        "val_map_ids.json",
        "test_map_ids.json",
    ]
    downloaded: dict[str, Path] = {}
    for name in required_names:
        downloaded[name] = download_file(args.repo_id, name, local_dir)
    for name in optional_names:
        if any(row["path"] == name for row in repo_files):
            downloaded[name] = download_file(args.repo_id, name, local_dir)

    split_paths = {split: downloaded[f"{split}.jsonl"] for split in ["train", "val", "test"]}
    records, map_rows = collect_text_records(split_paths)
    print(f"\nUnique primary text IDs: {len(records):,}")
    print("Domain counts:", dict(Counter(d for rec in records for d in rec["source_domains"])))

    all_texts = [rec["text"] for rec in records]
    print(f"Encoding {len(all_texts):,} unique texts plus the empty-string reference.", flush=True)
    encoded, encoder_info = encode_texts_specter2(
        all_texts + [""],
        model_name=args.encoder_model,
        adapter_name=args.adapter_name or None,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    empty_embedding = encoded[-1]
    centered = encoded[:-1].float() - empty_embedding.float().view(1, -1)
    normalized = F.normalize(centered, dim=1, eps=EPSILON).contiguous()
    print("Validating normalized cache.", flush=True)
    validation = validate_output(records, map_rows, centered, normalized)
    empty_checksum = sha256_bytes(empty_embedding.float().cpu().numpy().astype("float32").tobytes())

    metadata = {
        "hugging_face_source_repository": args.repo_id,
        "output_filename": paths["pt"].name,
        "text_encoder_model_name": args.encoder_model,
        "base_model_repository": encoder_info.get("base_model", ""),
        "model_revision_or_commit_hash": hf_model_sha(encoder_info.get("base_model", "")),
        "tokenizer_revision": hf_model_sha(encoder_info.get("base_model", "")),
        "adapter_name": encoder_info.get("adapter_name", args.adapter_name or "candidate"),
        "adapter_id": encoder_info.get("adapter_id", args.encoder_model),
        "adapter_revision_or_commit_hash": hf_model_sha(encoder_info.get("adapter_id", args.encoder_model)),
        "pooling_method": encoder_info.get("pooling_method", "cls_token"),
        "embedding_dimension": EXPECTED_DIM,
        "maximum_token_length": args.max_length,
        "title_abstract_formatting": "exact manifest primary positive_texts[0].text; no additional separator rewriting",
        "empty_string_embedding_checksum": empty_checksum,
        "preprocessing_order": ["subtract_empty_string_embedding", "l2_unit_normalize"],
        "text_embedding_preprocessing": "empty_string_centered_l2_unit_normalized",
        "epsilon": EPSILON,
        "number_of_unique_texts": len(records),
        "domain_counts": dict(Counter(d for rec in records for d in rec["source_domains"])),
        "creation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "software_versions": import_versions(),
        "random_seed_for_validation_samples": RANDOM_SEED,
        "git_commit": git_commit(repo_dir),
        "downloaded_files": {name: str(path) for name, path in downloaded.items()},
    }

    payload = build_payload(records, normalized, metadata)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving normalized cache tensor to {paths['pt']}", flush=True)
    torch.save(payload, paths["pt"])
    metadata["output_checksum"] = sha256_file(paths["pt"])
    print("Writing metadata, index, and validation sidecars.", flush=True)
    write_json(paths["metadata"], metadata)
    write_csv(paths["index"], prepare_index(records, normalized))
    validation.update(
        {
                "metadata_checksum": sha256_file(paths["metadata"]),
            "output_checksum": metadata["output_checksum"],
            "map_to_text_rows": len(map_rows),
        }
    )
    write_json(paths["validation"], validation)

    print("\nPrepared files for upload")
    for local_path in paths.values():
        print(
            json.dumps(
                {
                    "local_path": str(local_path),
                    "repository_path": f"{args.upload_prefix.rstrip('/')}/{local_path.name}",
                    "file_size": local_path.stat().st_size,
                    "sha256": sha256_file(local_path),
                },
                indent=2,
            )
        )
    if args.upload:
        maybe_upload(paths, args.repo_id, args.upload_prefix)


if __name__ == "__main__":
    main()
