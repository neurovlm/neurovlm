"""Helpers for run_analysis.py: environment, JSON and PubMed data (pinned HF files, leakage-audit test exclusions)."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# written by core/a4_leakage_audit/run_analysis.py --write-exclusions (see scripts/eval_exclusions/README.md)
EVAL_EXCLUSIONS_DIR = Path(__file__).resolve().parents[5] / "scripts" / "eval_exclusions"


def set_offline_env() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("MPLBACKEND", "Agg")


def seed_all(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_torch(path: Path | str, *, mmap: bool = True):
    import torch

    try:
        return torch.load(Path(path), map_location="cpu", weights_only=False, mmap=mmap)
    except (RuntimeError, ValueError, TypeError):
        return torch.load(Path(path), map_location="cpu", weights_only=False)


def json_safe(value: Any) -> Any:
    import torch

    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if np.isfinite(v) else None
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if torch.is_tensor(value):
        return json_safe(value.detach().cpu().tolist())
    return value


def write_json(path: Path | str, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(obj), indent=2, sort_keys=True))


def get_logger(name: str):
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "urllib3", "huggingface_hub", "sentence_transformers", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger(name)


def load_specter_latents(pmids: np.ndarray):
    """SPECTER2 (adhoc_query) embeddings [N, 768] aligned to ``pmids`` (pinned embedded_text/latent_specter2_adhoc.pt)."""
    import torch
    from _hf_pins import hf_file

    payload = load_torch(hf_file("neurovlm/embedded_text", "latent_specter2_adhoc.pt"))
    lat, pm = payload["latent"].float(), np.asarray(payload["pmid"])
    pos = {int(p): i for i, p in enumerate(pm)}
    idx = torch.as_tensor([pos[int(p)] for p in pmids], dtype=torch.long)
    return lat[idx].clone()


def load_pubmed_text_rows(pmids: np.ndarray, *, min_coverage: float = 0.95):
    """Title (``name``) and abstract (``description``) aligned to ``pmids``.

    The HF table stores pmids as Int64 / str, so both sides are cast to int64 before the join.
    """
    import pandas as pd
    from _hf_pins import load_dataset

    want = pd.Index(np.asarray(pmids).astype(np.int64), name="pmid")
    text = load_dataset("pubmed_text").copy()
    text["pmid"] = pd.to_numeric(text["pmid"], errors="coerce")
    text = text.dropna(subset=["pmid"])
    text["pmid"] = text["pmid"].astype(np.int64)
    rows = text.drop_duplicates("pmid").set_index("pmid").reindex(want)[["name", "description"]].copy()
    for col in ("name", "description"):
        rows[col] = rows[col].fillna("").astype(str).str.strip()
    coverage = float((rows["description"].str.len() > 0).mean()) if len(rows) else 1.0
    if coverage < min_coverage:
        raise RuntimeError(f"only {coverage:.1%} of the requested pmids have an abstract in pubmed_text (pmid dtype mismatch?)")
    out = rows.reset_index()
    out["pmid"] = np.asarray(pmids)
    return out


def split_pmids() -> dict[str, np.ndarray]:
    from _hf_pins import hf_split_pmids

    split = hf_split_pmids()  # split flags of the pinned HF publications.parquet, never the data-dir copy
    return {k: np.asarray(split[k]).astype(np.int64) for k in ("train", "val", "test")}


def heldout_pmids(split: str = "test", *, exclusions: bool = True, log=print) -> tuple[np.ndarray, dict]:
    """Sorted PMIDs of a held-out split minus the leakage-audit exclusions, and the counts for metrics.json."""
    import pandas as pd

    pmids = np.sort(split_pmids()[split])
    keep = np.ones(len(pmids), dtype=bool)
    if exclusions and len(pmids):
        excl = {int(x) for x in pd.read_csv(EVAL_EXCLUSIONS_DIR / "pubmed_exclusions.csv")["pmid"]}
        keep = ~np.isin(pmids, list(excl))
        log(f"eval exclusions: dropped {int((~keep).sum())} of {len(keep)} PubMed {split} PMIDs")
    return pmids[keep], {"enabled": bool(exclusions), f"n_pubmed_{split}_total": int(len(pmids)),
                         f"n_pubmed_{split}_dropped": int((~keep).sum())}


def load_pubmed_maps(pmids: np.ndarray):
    """Raw flat maps [N, 28542] (fp32, CPU) for ``pmids`` from the memory-mapped, pinned embedded_text/pubmed_images.pt."""
    import torch
    from _hf_pins import hf_file

    payload = load_torch(hf_file("neurovlm/embedded_text", "pubmed_images.pt"), mmap=True)
    vectors, pm = payload["neuro_vectors"], np.asarray(payload["pmid"])
    pos = {int(p): i for i, p in enumerate(pm)}
    idx = torch.as_tensor([pos[int(p)] for p in pmids], dtype=torch.long)
    return vectors[idx].float().clone()


def encode_specter(specter, texts: Sequence[str], batch_size: int = 32):
    """SPECTER2 768-d embeddings on CPU."""
    import torch

    with torch.no_grad():
        chunks = [torch.as_tensor(specter([str(t) for t in texts[s:s + batch_size]])).detach().float().cpu()
                  for s in range(0, len(texts), batch_size)]
    return torch.cat(chunks, dim=0)


def env_packages(names: Sequence[str] = ("torch", "transformers", "sentence_transformers", "bert_score", "nilearn", "neuromaps",
                                           "brainspace", "sklearn", "scipy", "numpy", "pandas", "matplotlib", "datasets", "psutil")) -> dict:
    import importlib

    out = {}
    for n in names:
        try:
            out[n] = getattr(importlib.import_module(n), "__version__", "?")
        except Exception:
            out[n] = None
    return out
