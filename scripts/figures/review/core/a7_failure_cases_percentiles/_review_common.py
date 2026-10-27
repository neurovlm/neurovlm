"""Shared helpers for this directory: data access, leakage-audit exclusions, shared-space encoders, brain plots and the
``--model-dir`` loader. Only the installed ``neurovlm`` package and its cached data are used."""
from __future__ import annotations

import contextlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[5]
NEUTRAL = "#8a8985"
INK = "#2b2a28"
MUTED = "#6b6a66"


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


def select_device(name: str = "auto"):
    import torch

    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_torch(path: Path | str, *, mmap: bool = True):
    import torch

    try:
        return torch.load(Path(path), map_location="cpu", weights_only=False, mmap=mmap)
    except (RuntimeError, ValueError, TypeError):  # old (non-zip) format cannot be memory-mapped
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


def read_json(path: Path | str):
    return json.loads(Path(path).read_text())


def get_logger(name: str):
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "urllib3", "huggingface_hub", "sentence_transformers", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger(name)


class Timings:
    """Named wall-clock sections saved to ``timings.json`` (merged with earlier runs)."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.rows: dict[str, dict] = read_json(self.path) if self.path.exists() else {}

    @contextlib.contextmanager
    def time(self, name: str, **meta):
        t0 = time.perf_counter()
        row = dict(meta)
        try:
            yield row
        finally:
            row["seconds"] = time.perf_counter() - t0
            row["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            self.rows[name] = row
            write_json(self.path, self.rows)


def release_cuda_cache() -> str:
    """Free the CUDA cache (so later models stay in real VRAM on WSL2) and describe GPU memory."""
    import gc

    import torch

    gc.collect()
    if not torch.cuda.is_available():
        return "cpu"
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    return (f"GPU free {free / 2**30:.1f}/{total / 2**30:.1f} GiB, torch allocated {torch.cuda.memory_allocated() / 2**30:.1f} GiB, "
            f"reserved {torch.cuda.memory_reserved() / 2**30:.1f} GiB")


# Leakage-audit exclusions (written by a4_leakage_audit/run_analysis.py --write-exclusions): PubMed test/val papers with a
# near-duplicate in the training split, and NeuroVault papers that are also PubMed training papers (with all their images).
EVAL_EXCLUSIONS_DIR = REPO / "scripts" / "eval_exclusions"


def _norm_doi(s) -> str | None:
    if s is None or (isinstance(s, float) and s != s):
        return None
    s = str(s).strip().lower()
    s = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi:\s*", "", s)
    return s or None


def _split_ids(values) -> set[int]:
    out: set[int] = set()
    for v in values:
        for x in str(v).split(";"):
            x = x.strip()
            if x and x.lower() != "nan":
                out.add(int(float(x)))
    return out


def load_eval_exclusions() -> dict:
    import pandas as pd

    pm = pd.read_csv(EVAL_EXCLUSIONS_DIR / "pubmed_exclusions.csv")
    nv = pd.read_csv(EVAL_EXCLUSIONS_DIR / "neurovault_exclusions.csv", dtype={"doi": str, "collection_id": str, "image_ids": str})
    return {"pubmed_pmids": {int(x) for x in pm["pmid"]},
            "neurovault_dois": {_norm_doi(x) for x in nv["doi"] if _norm_doi(x)},
            "neurovault_image_ids": _split_ids(nv["image_ids"].fillna("")),
            "neurovault_collection_ids": _split_ids(nv["collection_id"].fillna(""))}


def pubmed_keep_mask(pmids, *, enabled: bool = True, label: str = "PubMed", log=print) -> np.ndarray:
    """False for the excluded PMIDs (all True when not ``enabled``)."""
    import pandas as pd

    pmids = np.asarray(pmids)
    if not enabled or len(pmids) == 0:
        return np.ones(len(pmids), dtype=bool)
    excl = load_eval_exclusions()["pubmed_pmids"]
    keep = ~pd.to_numeric(pd.Series(pmids.ravel()), errors="coerce").astype("Int64").isin(excl).fillna(False).to_numpy(dtype=bool)
    if log is not None:
        log(f"[eval exclusions] {label}: dropped {int((~keep).sum())} of {len(keep)} PMIDs (leakage audit)")
    return keep


def neurovault_keep_mask(dois=None, *, image_ids=None, collection_ids=None, enabled: bool = True, label: str = "NeuroVault",
                         log=print) -> np.ndarray:
    """Keep-mask over NeuroVault rows; a row is dropped when its DOI, image id or collection id is excluded."""
    import pandas as pd

    n = next(len(x) for x in (dois, image_ids, collection_ids) if x is not None)
    if not enabled or n == 0:
        return np.ones(n, dtype=bool)
    ex = load_eval_exclusions()
    drop = np.zeros(n, dtype=bool)
    if dois is not None:
        drop |= np.array([_norm_doi(d) in ex["neurovault_dois"] for d in dois], dtype=bool)
    if image_ids is not None:
        drop |= pd.to_numeric(pd.Series(np.asarray(image_ids)), errors="coerce").isin(ex["neurovault_image_ids"]).to_numpy(dtype=bool)
    if collection_ids is not None:
        drop |= pd.to_numeric(pd.Series(np.asarray(collection_ids)), errors="coerce").isin(ex["neurovault_collection_ids"]).to_numpy(dtype=bool)
    if log is not None:
        log(f"[eval exclusions] {label}: dropped {int(drop.sum())} of {n} rows (NeuroVault papers in PubMed train)")
    return ~drop


def load_pubmed_text_rows(pmids: np.ndarray):
    """Title (``name``), abstract (``description``) and ``summary`` aligned to ``pmids``."""
    import pandas as pd
    from _hf_pins import load_dataset

    want = pd.Index(np.asarray(pmids).astype(np.int64), name="pmid")

    def by_pmid(df):  # the HF tables store pmids as Int64 / str
        df = df.copy()
        df["pmid"] = pd.to_numeric(df["pmid"], errors="coerce")
        df = df.dropna(subset=["pmid"])
        df["pmid"] = df["pmid"].astype(np.int64)
        return df.drop_duplicates("pmid").set_index("pmid")

    rows = by_pmid(load_dataset("pubmed_text")).reindex(want)[["name", "description"]].copy()
    try:
        rows["summary"] = by_pmid(load_dataset("pubmed_summaries")).reindex(want)["summary"]
    except Exception:  # summaries are optional
        rows["summary"] = ""
    for col in ("name", "description", "summary"):
        rows[col] = rows[col].fillna("").astype(str).str.strip()
    coverage = float((rows["description"].str.len() > 0).mean()) if len(rows) else 1.0
    if coverage < 0.95:
        raise RuntimeError(f"only {coverage:.1%} of the requested pmids have an abstract in pubmed_text (pmid dtype mismatch?)")
    out = rows.reset_index()
    out["pmid"] = np.asarray(pmids)
    return out


def split_pmids() -> dict[str, np.ndarray]:
    p = run_file("pmids_split")
    if p is None:  # split flags of the pinned HF publications.parquet, never the data-dir copy
        from _hf_pins import hf_split_pmids

        return hf_split_pmids()
    split = load_torch(p, mmap=False)
    return {k: np.asarray(split[k]).astype(np.int64) for k in ("train", "val", "test")}


def heldout_pmids(split: str = "test", *, exclusions: bool = True, log=print) -> tuple[np.ndarray, dict]:
    """Sorted PMIDs of a held-out split minus the excluded ones, and the exclusion counts."""
    pmids = np.sort(split_pmids()[split])
    keep = pubmed_keep_mask(pmids, enabled=exclusions, label=f"PubMed {split}", log=log)
    return pmids[keep], {"enabled": bool(exclusions), f"n_pubmed_{split}_total": int(len(pmids)),
                         f"n_pubmed_{split}_dropped": int((~keep).sum())}


def load_pubmed_maps(pmids: np.ndarray):
    """Raw flat maps [N, 28542] (fp32, CPU) for ``pmids``, from the pinned HF pubmed_images.pt (as validate_models.py)."""
    import torch
    from _hf_pins import hf_file

    payload = load_torch(hf_file("neurovlm/embedded_text", "pubmed_images.pt"), mmap=True)
    vectors, pm = payload["neuro_vectors"], np.asarray(payload["pmid"])
    pos = {int(p): i for i, p in enumerate(pm)}
    idx = torch.as_tensor([pos[int(p)] for p in pmids], dtype=torch.long)
    return vectors[idx].float().clone()


def released(name: str, device):
    """Released ``neurovlm`` model in eval mode."""
    from _hf_pins import load_model

    model = load_model(name)
    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    elif hasattr(model, "specter"):
        model.specter.eval()
    return model


def encode_specter(specter, texts: Sequence[str], batch_size: int = 32):
    """SPECTER2 (adhoc_query) 768-d embeddings on CPU."""
    import torch

    chunks = []
    with torch.no_grad():
        for s in range(0, len(texts), batch_size):
            chunks.append(torch.as_tensor(specter([str(t) for t in texts[s:s + batch_size]])).detach().float().cpu())
    return torch.cat(chunks, dim=0)


def shared_text(text_head, emb, device, batch_size: int = 1024):
    """768-d text embeddings (L2-normalised first) -> shared InfoNCE space."""
    import torch
    import torch.nn.functional as F

    out = []
    with torch.no_grad():
        for s in range(0, len(emb), batch_size):
            x = F.normalize(emb[s:s + batch_size].float().to(device), dim=1)
            out.append(F.normalize(text_head(x).float(), dim=1).cpu())
    return torch.cat(out, dim=0)


def shared_brain(image_head, latents, device, batch_size: int = 1024):
    import torch
    import torch.nn.functional as F

    out = []
    with torch.no_grad():
        for s in range(0, len(latents), batch_size):
            out.append(F.normalize(image_head(latents[s:s + batch_size].float().to(device)).float(), dim=1).cpu())
    return torch.cat(out, dim=0)


def encode_latents(encoder, X, device, batch_size: int = 512):
    """Autoencoder latents of raw maps, on CPU."""
    import torch

    out = []
    with torch.no_grad():
        for s in range(0, len(X), batch_size):
            out.append(encoder(torch.as_tensor(X[s:s + batch_size]).float().to(device)).detach().float().cpu())
    return torch.cat(out, dim=0)


def calibrate_maps(maps, rule: str = "global", *, kind: str = "head", factor: float | None = None, eps: float = 1e-8):
    """Calibrate generated maps (last axis = voxels), as scripts/models/_common_models.calibrate_maps: "global" (the paper's
    rule) = times the model's factor (map_factor(kind)), clipped at 1; "max" = divided by their own maximum. Torch or numpy
    in, same type out."""
    if rule == "global":
        factor = map_factor(kind) if factor is None else factor
        if isinstance(maps, np.ndarray):
            return np.minimum(maps * np.float32(factor), 1).astype(np.float32, copy=False)
        return (maps * factor).clamp(max=1.0)
    if isinstance(maps, np.ndarray):
        return (maps / np.maximum(maps.max(axis=-1, keepdims=True), eps)).astype(np.float32, copy=False)
    return maps / maps.amax(dim=-1, keepdim=True).clamp_min(eps)


def map_factor(kind: str = "head") -> float:
    """Global calibration factor of the text-to-brain head ("head") or anatomical head ("adapter"): <model dir>/
    map_calibration.json, or scripts/outputs/released_map_calibration.json for the released models (both written by
    scripts/pipeline/validate_models.py)."""
    import hashlib

    if "map_calibration" not in _MODEL_DIR:
        d = _MODEL_DIR["path"]
        path = d / "map_calibration.json" if d is not None else REPO / "scripts" / "outputs" / "released_map_calibration.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing: run scripts/pipeline/validate_models.py on this model first")
        cal = json.loads(path.read_text())
        for src in cal["source"]["models"].values():  # the factors belong to the model files they were fitted on
            if d is not None and not src.startswith("released:"):
                fname, sha = src.split(":")
                if hashlib.sha256((d / fname).read_bytes()).hexdigest() != sha:
                    raise RuntimeError(f"{path} was fitted on another {fname}; rerun scripts/pipeline/validate_models.py")
        _MODEL_DIR["map_calibration"] = cal
    return float(_MODEL_DIR["map_calibration"][kind])


def percentile_of(value: float, dist: Sequence[float]) -> float:
    """Percentile rank (0-100): % of ``dist`` strictly below ``value`` plus half the ties."""
    d = np.asarray(dist, dtype=np.float64)
    d = d[np.isfinite(d)]
    if not len(d) or not np.isfinite(value):
        return float("nan")
    return float(100.0 * ((d < value).sum() + 0.5 * (d == value).sum()) / len(d))


def apply_style(font_size: float = 8.0) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": font_size, "axes.titlesize": font_size + 1, "axes.labelsize": font_size,
        "xtick.labelsize": font_size, "ytick.labelsize": font_size, "legend.fontsize": font_size,
        "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
        "lines.linewidth": 1.6, "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.edgecolor": NEUTRAL, "xtick.color": "#52514e", "ytick.color": "#52514e", "text.color": INK, "axes.labelcolor": INK,
    })


_TEMPLATE = {}


def flat_to_nifti(vec, masker):
    return masker.inverse_transform(np.asarray(vec, dtype=np.float32).reshape(1, -1))


def draw_brain(img, ax, *, cut_coords=None, threshold=1e-2, display_mode: str = "ortho"):
    """``plot_stat_map`` into an existing axes (as scripts/figures/main/_shared.draw_brain)."""
    from nilearn.datasets import load_mni152_template
    from nilearn.plotting import plot_stat_map

    if "t" not in _TEMPLATE:
        _TEMPLATE["t"] = load_mni152_template(resolution=2)
    ax.set_axis_off()
    try:
        return plot_stat_map(img, bg_img=_TEMPLATE["t"], axes=ax, display_mode=display_mode, cut_coords=cut_coords, cmap="Reds",
                             threshold=threshold, vmin=0.0, vmax=None, colorbar=False, draw_cross=False, annotate=False,
                             black_bg=False, symmetric_cbar=False)
    except Exception as exc:  # degenerate / empty maps
        ax.text(0.5, 0.5, f"(map not renderable: {type(exc).__name__})", ha="center", va="center", fontsize=8, color=MUTED,
                transform=ax.transAxes)
        return None


def cut_coords_for(img):
    from nilearn.plotting import find_xyz_cut_coords

    try:
        return [float(c) for c in find_xyz_cut_coords(img)]
    except Exception:
        return None


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


# --model-dir: models of a retrained run (scripts/pipeline/train_pipeline.py layout) instead of the release; anything the run lacks
# falls back to the release. SPECTER2 and NeuroQwen3-0.6B are frozen: always the released ones. Copied from
# scripts/figures/main/_shared.py and scripts/models/_common_models.py.
DEFAULT_MODEL_DIR = REPO / "scripts" / "outputs" / "runs" / "final"
RUN_FILES = {
    "autoencoder": "autoencoder.pt", "proj_head_text_mse": "proj_head_text_mse.pt", "proj_head_text_infonce": "proj_head_text_infonce.pt",
    "proj_head_image_infonce": "proj_head_image_infonce.pt", "neuro_adapter": "adapter_anat_full.pt",
    "qformer_canonical": "neuro_qformer_canonical/model.safetensors",
    "canonical_semantic_banks": "canonical_semantic_banks.pt", "pmids_split": "pmids_split.pt",
}
_MODEL_DIR: dict = {"path": None, "sources": {}}


def set_model_dir(path) -> Path | None:
    """Use the run directory ``path`` (None = released models)."""
    _MODEL_DIR["path"] = None if path is None else Path(path).resolve()
    _MODEL_DIR.pop("map_calibration", None)
    if _MODEL_DIR["path"] is not None and not _MODEL_DIR["path"].is_dir():
        raise FileNotFoundError(f"--model-dir {path} is not a directory")
    return _MODEL_DIR["path"]


def model_dir() -> Path | None:
    return _MODEL_DIR["path"]


def run_file(name: str) -> Path | None:
    d = _MODEL_DIR["path"]
    if d is None:
        return None
    p = d / RUN_FILES[name]
    return p if p.exists() else None


def _rel(p) -> str:
    p = Path(p).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def note_source(name: str, source) -> None:
    _MODEL_DIR["sources"][name] = source if isinstance(source, str) else _rel(source)


def model_sources() -> dict:
    """Checkpoint of every loaded component (for metrics.json)."""
    return dict(_MODEL_DIR["sources"])


def component(name: str, device):
    """Trained component from the run directory (whole-module .pt) or the release: autoencoder, proj_head_text_mse,
    proj_head_text_infonce, proj_head_image_infonce, neuro_adapter or specter."""
    import torch

    p = run_file(name) if name in RUN_FILES else None
    if p is None:
        if model_dir() is not None and name in RUN_FILES:
            print(f"[model-dir] {model_dir()} has no {RUN_FILES[name]}; using the released {name}")
        note_source(name, f"released:{name}" if name != "specter" else "released:allenai/specter2_aug2023refresh + adhoc_query (frozen)")
        return released(name, device)
    obj = torch.load(p, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):  # raw state dict -> released architecture
        from _hf_pins import load_model

        m = load_model(name)
        m.load_state_dict(obj.get("state_dict", obj), strict=True)
        obj = m
    note_source(name, p)
    return obj.to(device).eval()


def _define_mlp_projector():
    """``ResidualMLPProjector`` of scripts/models/_common_models.py, verbatim (the final run's projectors)."""
    import torch
    import torch.nn.functional as F
    from torch import nn

    class ResidualBlock(nn.Module):
        def __init__(self, width: int, expansion: int = 2, dropout: float = 0.05):
            super().__init__()
            self.norm = nn.LayerNorm(width)
            self.fc1 = nn.Linear(width, width * expansion)
            self.act = nn.GELU()
            self.drop = nn.Dropout(dropout)
            self.fc2 = nn.Linear(width * expansion, width)

        def forward(self, x):
            return x + self.fc2(self.drop(self.act(self.fc1(self.norm(x)))))

    class ResidualMLPProjector(nn.Module):
        def __init__(self, image_dim=384, semantic_dim=384, lm_dim=1024, num_queries=32, *, width=704, depth=2, expansion=2,
                     dropout=0.05, head="tokens", align_hidden=512, **_ignored):
            super().__init__()
            self.image_dim, self.semantic_dim, self.lm_dim, self.num_queries = int(image_dim), int(semantic_dim), int(lm_dim), int(num_queries)
            self.width, self.depth, self.expansion, self.head_kind = int(width), int(depth), int(expansion), head
            self.dropout, self.align_hidden = float(dropout), int(align_hidden)
            self.norm_raw, self.norm_sem = nn.LayerNorm(image_dim), nn.LayerNorm(semantic_dim)
            self.in_proj = nn.Linear(image_dim + semantic_dim, width)
            self.blocks = nn.ModuleList([ResidualBlock(width, expansion, dropout) for _ in range(depth)])
            self.out_norm = nn.LayerNorm(width)
            self.head = nn.Linear(width, num_queries * lm_dim if head == "tokens" else lm_dim, bias=False)
            self.query_pos = nn.Parameter(torch.zeros(1, num_queries, lm_dim))
            self.token_norm = nn.LayerNorm(lm_dim)
            self.align_head = nn.Sequential(nn.LayerNorm(lm_dim), nn.Linear(lm_dim, align_hidden), nn.GELU(), nn.Linear(align_hidden, semantic_dim))

        def forward(self, raw_images, semantic_images):
            dtype = self.in_proj.weight.dtype
            h = self.in_proj(torch.cat([self.norm_raw(raw_images.to(dtype)), self.norm_sem(semantic_images.to(dtype))], dim=-1))
            for block in self.blocks:
                h = block(h)
            t = self.head(self.out_norm(h))
            if self.head_kind == "tokens":
                t = t.view(t.size(0), self.num_queries, self.lm_dim)
            else:
                t = t.unsqueeze(1).expand(-1, self.num_queries, -1)
            return self.token_norm(t + self.query_pos)

        def semantic_from_visual(self, visual_tokens):
            x = visual_tokens.to(dtype=self.in_proj.weight.dtype).mean(dim=1)
            return F.normalize(self.align_head(x).float(), dim=1)

    return ResidualMLPProjector


def load_projector(variant: str, device):
    """Brain-to-text projector of the run (``neuro_qformer_<variant>/``) wrapped in ``NeuroQFormer``, else the released one.
    For the canonical variant the run's canonical_semantic_banks.pt replaces the packaged banks."""
    import torch
    import torch.nn.functional as F

    key = f"qformer_{variant}"
    weights = run_file(key)
    if weights is None:
        from _hf_pins import released_qformer

        note_source(key, f"released:neurovlm/{'NeuroQformer' if variant == 'canonical' else 'NeuroQformer-PubMed'}")
        return released_qformer(variant, device)
    from safetensors.torch import load_file
    from neurovlm.models.qformer import CanonicalProjection, NeuroQFormer

    cfg = read_json(weights.parent / "config.json")
    state = dict(load_file(str(weights)))
    note_bank = ""
    if variant == "canonical" and run_file("canonical_semantic_banks") is not None:
        banks = torch.load(run_file("canonical_semantic_banks"), map_location="cpu", weights_only=False)
        same = True
        for b in ("all", "network", "region", "function"):
            k = f"canonical_projection.{b}_bank"
            new = F.normalize(torch.as_tensor(banks[b]).float(), dim=1)
            old = state.get(k)
            same &= old is not None and old.shape == new.shape and float((old.float() - new).abs().max()) == 0.0
            state[k] = new
        note_bank = " (canonical_semantic_banks.pt " + ("identical to the packaged banks)" if same else "replaced the packaged banks)")
    kind = cfg.get("projector", "qformer")
    block = dict(cfg.get("qformer", {}))
    if kind == "qformer":
        model = NeuroQFormer.from_state_dict_payload({"state_dict": state, "config": block, "canonical_metadata": cfg.get("canonical_metadata", {})},
                                                     map_location=device)
    elif kind == "mlp":
        from neurovlm.models.base import ProjHead

        pcfg = {k: v for k, v in (cfg.get("projector_config") or {}).items() if k not in ("class", "params")}
        projector = _define_mlp_projector()(**pcfg)
        first, last = state["proj_head_image.aligner.0.weight"], state["proj_head_image.aligner.2.weight"]
        head = ProjHead(int(first.shape[1]), int(first.shape[0]), int(last.shape[0]))
        sem_dim = int(pcfg.get("semantic_dim", 384))
        banks = {n: state.get(f"canonical_projection.{n}_bank", torch.empty(0, sem_dim)) for n in CanonicalProjection.bank_names}
        model = NeuroQFormer(qformer=projector, proj_head_image=head, canonical_banks=banks, semantic_dim=sem_dim,
                             projection_temp=block.get("projection_temp", 0.05), canonical_basis=block.get("canonical_basis", "all"),
                             use_canonical_projection=block.get("use_canonical_projection", True))
        model.load_state_dict(state, strict=True)
    else:
        raise ValueError(f"unknown projector {kind!r} in {weights.parent / 'config.json'}")
    note_source(key, f"{_rel(weights.parent)} (projector: {kind}){note_bank}")
    return model.to(device).eval()


def make_nvlm(device, *, ae, text_head, image_head, mse_head, specter):
    """``NeuroVLM`` client that uses the given autoencoder, heads and SPECTER2."""
    from neurovlm import NeuroVLM
    from _hf_pins import hf_cache_pinned

    hf_cache_pinned()  # anything the client still lazy-loads comes from the pinned HF cache
    nvlm = NeuroVLM(device=str(device))
    nvlm._autoencoder = ae
    nvlm._proj_head_text_infonce, nvlm._proj_head_image_infonce = text_head, image_head
    nvlm._proj_head_text_mse = mse_head
    nvlm._specter = specter
    note_source("neuroqwen", "released:neurovlm/NeuroQwen3-0.6B (frozen LM)")
    return nvlm
