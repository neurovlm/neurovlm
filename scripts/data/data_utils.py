"""Helpers shared by the scripts/data stages (CLI flags, artifact dir, batched SPECTER encoding)."""
from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/
from common import data_dir, hf_manifest  # noqa: E402

SPECTER_REPOS = ("allenai/specter2_aug2023refresh_base", "allenai/specter2_aug2023refresh_adhoc_query")


def add_data_args(parser: argparse.ArgumentParser, *, amp: bool = True) -> None:
    """Add --artifact-dir (and --amp-dtype for stages that run SPECTER)."""
    parser.add_argument("--artifact-dir", type=Path, default=None,
                        help="Library artifacts (default: data_dir; in --smoke: <out>/smoke_artifacts).")
    if amp:
        parser.add_argument("--amp-dtype", choices=["bf16", "fp16", "fp32"], default="bf16",
                            help="Autocast dtype for SPECTER on CUDA (fp32 = notebook numerics).")


def resolve_artifact_dir(args: argparse.Namespace) -> Path:
    """--artifact-dir, else <out>/smoke_artifacts in --smoke, else data_dir."""
    if args.artifact_dir is not None:
        path = Path(args.artifact_dir)
    elif args.smoke:
        path = Path(args.out) / "smoke_artifacts"
    else:
        path = data_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


def chunk_bounds(n: int, chunk: int):
    for start in range(0, n, chunk):
        yield start, min(start + chunk, n)


def encode_texts(specter, texts: Sequence[str], *, batch_size: int, amp_dtype: str = "fp32", desc: str = "specter"):
    """Encode texts with a Specter wrapper -> float32 CPU tensor (len(texts), 768).

    Batches are sorted by length (longest first); per-sample results are the same as the notebooks' loop.
    """
    import torch
    from tqdm.auto import tqdm

    texts = list(texts)
    out = torch.empty((len(texts), 768), dtype=torch.float32)
    order = np.argsort([-len(t) for t in texts], kind="stable")
    batches = [order[s:e] for s, e in chunk_bounds(len(order), batch_size)]
    device = specter.device
    if device.type == "cuda" and amp_dtype != "fp32":
        amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16 if amp_dtype == "bf16" else torch.float16)
    else:
        amp = contextlib.nullcontext()
    with torch.inference_mode(), amp:
        for idx in tqdm(batches, desc=desc, total=len(batches)):
            emb = specter([texts[i] for i in idx])
            out[torch.as_tensor(idx)] = emb.detach().float().cpu()
    return out


def hf_cache_pinned(*repos: str, group: str = "regenerate") -> None:
    """Raise unless the HF cache serves every manifest file of ``repos`` in ``group`` at its pinned revision: refs/<ref> set
    to it (scripts/data/fetch_hf.py) and the file present. The neurovlm loaders (load_dataset, load_masker, load_model,
    Specter) take no revision and read the HF cache first, so they then get exactly the pinned files."""
    from huggingface_hub import try_to_load_from_cache
    from huggingface_hub.constants import HF_HUB_CACHE

    bad = []
    for repo in repos:
        rows = [e for e in hf_manifest()["files"] if e["repo"] == repo and e.get("hf_cache") and group in e["groups"]]
        if not rows:
            bad.append(f"{repo}: no {group} entry in scripts/hf_manifest.json")
        for e in rows:
            ref = Path(HF_HUB_CACHE) / f"{e['repo_type']}s--{repo.replace('/', '--')}" / "refs" / e["ref"]
            cur = ref.read_text().strip() if ref.exists() else None
            if cur != e["revision"]:
                bad.append(f"{repo}: HF cache refs/{e['ref']} is {cur}, pinned {e['revision'][:8]}")
            elif not isinstance(try_to_load_from_cache(repo, e["path"], revision=e["revision"], repo_type=e["repo_type"]), str):
                bad.append(f"{repo}/{e['path']}@{e['revision'][:8]} is not in the HF cache")
    if bad:
        raise RuntimeError(f"the HF cache is not at the pinned revisions (run scripts/data/fetch_hf.py):\n  " + "\n  ".join(sorted(set(bad))))
