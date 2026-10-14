#!/usr/bin/env python
"""NeuroVault retrieval with selection-bias bounds, map-type-aware preprocessing and the dataset-size counts.

Maps are re-read from the raw NIfTIs (NaN -> 0, nearest resampling to the 4 mm mask), branched by map type (signed:
positive tail / |value| / sign split; p-maps inverted to 1 - p; 1 - p maps and masks used directly), then normalised as
in the paper (per-image 99th percentile, clamp to [0, 1]). Maps whose uploaders chose the type 'other' are typed from
their names, descriptions, siblings and values (_other_maps.py); mean or variance images and maps that cannot be typed are
excluded (--other-maps infer, default; keep / exclude are written to other_maps_options.csv). Variant 'released' (Fig. 5b) is the published map tensor with
its p-maps rebuilt from the NIfTIs as 1 - p under the same threshold rules; 'released_raw' is the tensor as published. Target rules: paper_upper (image the model scores highest
for the paper's text), worst_image (lowest), oracle_any_true / per_image (every image a target), worst_case.

    .venv/bin/python scripts/figures/review/core/a3_neurovault_bounds/run_analysis.py --device cpu --force --qc-count-all
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _final_model as M  # noqa: E402
import _other_maps as OM  # noqa: E402

N_VOXELS = 28_542
KS = (1, 5, 10)
EXCLUDED_TYPES = {"parcellation": "not a statistical map (discrete labels)", "anatomical": "structural image, not a statistical map", "variance": "variance map, not a contrast"}
MASK_TYPES = {"ROI/mask"}
P_TYPES = {"P map (given null hypothesis)"}
INV_P_TYPES = {'1-P map ("inverted" probability)'}

# leakage-audit exclusions (written by a4_leakage_audit/run_analysis.py --write-exclusions)
EVAL_EXCLUSIONS_DIR = Path(__file__).resolve().parents[5] / "scripts" / "eval_exclusions"


def _norm_doi(s) -> str | None:
    if s is None or (isinstance(s, float) and s != s):
        return None
    s = str(s).strip().lower()
    s = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi:\s*", "", s)
    return s or None


def _split_ids(values) -> set[int]:
    out = set()
    for v in values:
        for x in str(v).split(";"):
            x = x.strip()
            if x and x.lower() != "nan":
                out.add(int(float(x)))
    return out


def neurovault_keep_mask(dois, *, image_ids=None, collection_ids=None, enabled: bool = True, label: str = "NeuroVault") -> np.ndarray:
    """False for NeuroVault rows (publications or images) of papers that are PubMed training papers."""
    if not enabled:
        return np.ones(len(dois), dtype=bool)
    ex = pd.read_csv(EVAL_EXCLUSIONS_DIR / "neurovault_exclusions.csv", dtype=str)
    ex_dois = {_norm_doi(x) for x in ex["doi"] if _norm_doi(x)}
    drop = np.array([_norm_doi(d) in ex_dois for d in dois], dtype=bool)
    for ids, col in ((image_ids, "image_ids"), (collection_ids, "collection_id")):
        if ids is not None:
            drop |= pd.to_numeric(pd.Series(np.asarray(ids)), errors="coerce").isin(_split_ids(ex[col].fillna(""))).to_numpy(dtype=bool)
    print(f"[eval exclusions] {label}: dropped {int(drop.sum())} of {len(drop)} rows")
    return ~drop


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(arg: str, min_free_gb: float = 3.0) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        if free / 1e9 >= min_free_gb:
            return torch.device("cuda")
        print(f"[device] only {free / 1e9:.1f} GB free on GPU -> using CPU")
    return torch.device("cpu")


class Timings:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.records: dict[str, float] = {}

    @contextmanager
    def time(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.records[name] = round(time.perf_counter() - t0, 3)
            self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.records, indent=2))


def l2n(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def deranged_permutation(n: int, rng: np.random.Generator) -> np.ndarray:
    """Permutation without fixed points."""
    perm = rng.permutation(n)
    fixed = np.flatnonzero(perm == np.arange(n))
    if len(fixed) > 1:
        perm[fixed] = np.roll(perm[fixed], 1)
    elif len(fixed) == 1:
        i = int(fixed[0])
        j = (i + 1) % n
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def map_class(map_type: str) -> str:
    if map_type in EXCLUDED_TYPES:
        return "excluded"
    if map_type in MASK_TYPES:
        return "mask"
    if map_type in P_TYPES:
        return "p_map"
    if map_type in INV_P_TYPES:
        return "inv_p_map"
    return "signed"


def load_metadata() -> dict[str, Any]:
    """Released NeuroVault texts and image metadata, joined with the scraped image and collection tables (pinned HF files)."""
    from _hf_pins import hf_file, load_dataset

    df_text = load_dataset("neurovault_text").reset_index(drop=True)
    meta = load_dataset("neurovault_images_meta").reset_index(drop=True)
    local_img = pd.read_parquet(hf_file("neurovlm/neurovault_maps", "neurovault_images.parquet"))
    local_col = pd.read_parquet(hf_file("neurovlm/neurovault_maps", "neurovault_collections.parquet"))
    cols = ["id", "map_type", "is_thresholded", "modality", "not_mni", "number_of_subjects", "cognitive_paradigm_cogatlas", "analysis_level"]
    meta = meta.merge(local_img[[c for c in cols if c in local_img.columns]], on="id", how="left")
    meta["map_type"] = meta["map_type"].fillna("unknown")
    meta["map_class"] = meta["map_type"].map(map_class)
    meta["pub_index"] = meta["doi"].map({d: i for i, d in enumerate(df_text["doi"])})
    return {"df_text": df_text, "meta": meta, "local_img": local_img, "local_col": local_col}


def count_table(md: dict[str, Any], root: Path, qc_count_all: bool, n_jobs: int, exclusions: bool = True) -> pd.DataFrame:
    """Image counts from the scraped metadata to the evaluated set (both dataset-size definitions)."""
    from _hf_pins import load_dataset

    local_img, local_col, meta = md["local_img"], md["local_col"], md["meta"]
    img = local_img.merge(local_col[["id", "DOI"]].rename(columns={"id": "collection_id", "DOI": "collection_doi"}), on="collection_id", how="left")
    img["collection_doi"] = img["collection_doi"].replace("", np.nan)
    rows = []

    def add(stage: str, frame: pd.DataFrame, note: str) -> None:
        rows.append({"stage": stage, "n_images": int(len(frame)), "n_collections": int(frame["collection_id"].nunique()), "n_dois": int(frame["collection_doi"].nunique()), "note": note})

    add("A. scraped images in collections with a DOI", img, "neurovault_images.parquet (02_data/03, rescrape=False)")
    unthr = img[img["is_thresholded"] == False]  # noqa: E712
    add("B. unthresholded (is_thresholded == False)", unthr, "thresholded maps and maps with unknown threshold status dropped")
    nonexcl = unthr[~unthr["map_type"].isin(EXCLUDED_TYPES)]
    add("C. statistical map types only", nonexcl, "parcellation / anatomical / ROI-mask / variance removed as in 02_data/03 (ROI/mask also removed there)")
    nonexcl = nonexcl[~nonexcl["map_type"].isin(MASK_TYPES)]
    add("C'. ... and ROI/mask removed (notebook rule)", nonexcl, "metadata-eligible set, 'definition 1' (paper: ~4,553 at scrape time)")
    hand = nonexcl[~nonexcl["id"].isin([22136, 22137, 22138])]
    add("D. hand-excluded ids removed", hand, "ids 22136-22138 dropped in 02_data/03")
    if qc_count_all:
        from joblib import Parallel, delayed

        ids = hand["id"].to_numpy()
        flags = Parallel(n_jobs=n_jobs)(delayed(_qc_flags)(root / "neurovault_images" / f"neurovault_img_{i}.nii.gz") for i in tqdm(ids, desc="QC count (all eligible NIfTIs)", leave=False))
        qc = pd.DataFrame(flags, index=ids)
        ok = qc["exists"] & qc["n_unique_ge_10"] & qc["nonzero"] & qc["n_unique_gt_50"]
        add("E. image QC (file present, >10 and >50 unique values, non-zero)", hand[hand["id"].isin(qc.index[ok])], "02_data/03 cells 9-10 (before cluster thresholding)")
    pub_dois = set(load_dataset("pubmed_text")["doi"].dropna().astype(str).str.lower())
    in_pubmed = hand["collection_doi"].astype(str).str.lower().isin(pub_dois)
    add("F. collections whose DOI is in the PubMed corpus removed", hand[~in_pubmed], f"{int(in_pubmed.sum())} images / {hand.loc[in_pubmed, 'collection_doi'].nunique()} DOIs overlap PubMed (any split)")
    rel = meta.assign(collection_doi=meta["doi"])
    add("G. released evaluation set (after QC, cluster threshold, >=10 % active, PubMed-DOI removal)", rel, "images_neurovault.parquet / neurovault_images.pt, 'definition 2' (paper: ~3,000)")
    with_text = rel[rel["pub_index"].notna()]
    add("H. ... with a title + abstract (evaluated)", with_text, f"{md['df_text'].shape[0]} publications in publications_neurovault.parquet")
    # last step: leakage-audit exclusions
    keep = neurovault_keep_mask(with_text["doi"].to_numpy(), image_ids=with_text["id"].to_numpy(), collection_ids=with_text["collection_id"].to_numpy(),
                                enabled=exclusions, label="NeuroVault counts chain (stage H -> I)")
    note = (f"{int((~keep).sum())} images / {with_text.loc[~keep, 'doi'].nunique()} publications dropped: NeuroVault papers that are PubMed "
            "training papers (scripts/eval_exclusions/neurovault_exclusions.csv); the set every NeuroVault metric is computed on"
            if exclusions else "not applied (--no-exclusions)")
    add("I. ... after leakage exclusions (evaluated)", with_text[keep], note)
    return pd.DataFrame(rows)


def _qc_flags(path: Path) -> dict[str, Any]:
    """Image QC of 02_data/03: file present, >= 10 and > 50 unique values, non-zero."""
    import nibabel as nib

    if not path.exists():
        return {"exists": False, "n_unique_ge_10": False, "nonzero": False, "n_unique_gt_50": False}
    try:
        arr = np.nan_to_num(np.asarray(nib.load(str(path)).dataobj, dtype=np.float32))
        u = np.unique(arr).size
        return {"exists": True, "n_unique_ge_10": bool(u >= 10), "nonzero": bool(np.any(arr != 0)), "n_unique_gt_50": bool(u > 50)}
    except Exception:
        return {"exists": True, "n_unique_ge_10": False, "nonzero": False, "n_unique_gt_50": False}


def _mask_vector(path: Path, mask_data: np.ndarray, mask_affine: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Raw signed voxel vector inside the NeuroVLM mask (NaN -> 0, nearest resampling) and its stats."""
    import nibabel as nib
    from nilearn.image import resample_to_img

    img = nib.load(str(path))
    arr = np.nan_to_num(np.asarray(img.dataobj, dtype=np.float32))
    if arr.ndim == 4:
        arr = arr[..., 0]
    img = nib.Nifti1Image(arr, img.affine)
    mask_img = nib.Nifti1Image(mask_data.astype(np.uint8), mask_affine)
    res = resample_to_img(img, mask_img, interpolation="nearest", force_resample=True, copy_header=True)
    v = np.asarray(res.dataobj, dtype=np.float32)[mask_data > 0]
    stats = {
        "frac_negative": float((v < 0).mean()),
        "frac_positive": float((v > 0).mean()),
        "frac_in_unit_interval": float(((v > 0) & (v <= 1)).mean()),
        "min": float(v.min()),
        "max": float(v.max()),
        "n_unique": int(min(np.unique(v).size, 100_000)),
        **dict(zip(("q001", "q01", "q99", "q999"), np.quantile(v, [0.001, 0.01, 0.99, 0.999]).astype(float).tolist())),
    }
    return v, stats


def branch(v: np.ndarray, cls: str) -> dict[str, np.ndarray | None]:
    """Preprocessing variants by map class: positive tail, |value|, negative tail (None if not signed)."""
    if cls == "signed":
        return {"pos": np.clip(v, 0, None), "abs": np.abs(v), "neg": np.clip(-v, 0, None)}
    if cls == "p_map":
        q = np.where((v > 0) & (v <= 1), 1.0 - v, 0.0).astype(np.float32)
        return {"pos": q, "abs": q, "neg": None}
    if cls == "inv_p_map":
        q = np.clip(v, 0, None)
        return {"pos": q, "abs": q, "neg": None}
    if cls == "mask":
        b = (v > 0).astype(np.float32)
        return {"pos": b, "abs": b, "neg": None}
    raise ValueError(cls)


def cluster_release(v: np.ndarray, masker: Any) -> np.ndarray:
    """Threshold of the released tensor (scripts/data/03_neurovault.py cell 11): 99th percentile, clusters >= 50 voxels."""
    from nilearn.image import threshold_img

    thr = threshold_img(masker.inverse_transform(v), "99%", cluster_threshold=50, two_sided=False, copy_header=True)
    return np.asarray(masker.transform(thr), dtype=np.float32).reshape(-1)


def released_with_inverted_p(released: torch.Tensor, meta: pd.DataFrame, root: Path, masker: Any) -> tuple[torch.Tensor, pd.DataFrame]:
    """Published tensor with every p-map row rebuilt as in 03_neurovault.py but from 1 - p (p = 0 stays 0). A map with
    no cluster left after the threshold keeps its unthresholded 1 - p (as variant 'pos'): an empty map would drop out
    of every other paper's competitors and inflate the AUCs."""
    out = released.clone()
    mask_data = np.asarray(masker.mask_img_.dataobj) > 0
    rows = []
    for i in np.flatnonzero(meta["map_class"].to_numpy() == "p_map"):
        v, _ = _mask_vector(root / "neurovault_images" / f"neurovault_img_{meta['id'].iloc[i]}.nii.gz", mask_data, masker.mask_img_.affine)
        inv = branch(v, "p_map")["pos"]
        q = cluster_release(inv, masker)
        ok = bool(0 < (q != 0).mean() < 0.1)
        if not ok:
            q = inv
        old = released[i].numpy()
        rows.append({"id": int(meta["id"].iloc[i]), "doi": meta["doi"].iloc[i], "frac_negative": float((v < 0).mean()), "frac_below_0.05": float(((v > 0) & (v < 0.05)).mean()),
                     "active_frac_published": float((old != 0).mean()), "active_frac_inverted": float((q != 0).mean()), "release_threshold_applied": ok,
                     "corr_published_vs_inverted": float(np.corrcoef(old, q)[0, 1]) if old.std() > 0 and q.std() > 0 else float("nan")})
        out[i] = torch.as_tensor(q)
    return out, pd.DataFrame(rows)


def normalise_paper(x: torch.Tensor) -> torch.Tensor:
    """figures/12: divide by the per-image 99th percentile, clamp to [0, 1]."""
    q = torch.quantile(x, 0.99, dim=1, keepdim=True).clamp_min(1e-6)
    return (x / q).clamp(0, 1)


@torch.no_grad()
def encode(ae: Any, x: torch.Tensor, device: torch.device, batch: int = 512) -> torch.Tensor:
    return torch.cat([ae.encoder(x[s : s + batch].to(device)).cpu() for s in range(0, len(x), batch)])


def preprocess_images(meta: pd.DataFrame, root: Path, masker: Any, ae: Any, device: torch.device, cache_dir: Path, n_jobs: int, released: torch.Tensor,
                      released_raw: torch.Tensor, force: bool = False) -> tuple[dict[str, torch.Tensor], pd.DataFrame]:
    """Autoencoder latents per preprocessing variant (cached per image id and model; ``force`` recomputes them) and per-image stats."""
    from joblib import Parallel, delayed

    cache = cache_dir / f"latents_by_variant_{M.model_tag()}.npz"
    ids = meta["id"].to_numpy()
    done: dict[str, Any] = {}
    if cache.exists() and not force:
        z = np.load(cache, allow_pickle=True)
        done = {k: z[k] for k in z.files}
        if "stats" in done and len(done["stats"]) and "q99" not in done["stats"][0]:
            done = {}  # cache from before the value quantiles
    have = set(done.get("ids", np.array([], dtype=np.int64)).tolist())
    todo = np.flatnonzero(~np.isin(ids, list(have)))
    mask_data = np.asarray(masker.mask_img_.dataobj) > 0
    mask_affine = masker.mask_img_.affine
    new_lat: dict[str, list[np.ndarray]] = {"released": [], "released_raw": [], "pos": [], "abs": [], "neg": []}
    new_stats: list[dict[str, Any]] = []
    new_ids: list[int] = []
    chunk = 256
    for s in tqdm(range(0, len(todo), chunk), desc="NIfTI preprocessing", leave=False):
        sel = todo[s : s + chunk]
        paths = [root / "neurovault_images" / f"neurovault_img_{ids[i]}.nii.gz" for i in sel]
        res = Parallel(n_jobs=n_jobs)(delayed(_mask_vector)(p, mask_data, mask_affine) for p in paths)
        vecs = {"pos": [], "abs": [], "neg": []}
        for i, (v, st), p in zip(sel, res, paths):
            cls = meta["map_class"].iloc[i]
            out = branch(v, cls)
            rel = released[i].numpy()
            st.update({"id": int(ids[i]), "map_type": meta["map_type"].iloc[i], "map_class": cls, "file": p.name, "has_negative_tail": bool(st["frac_negative"] >= 0.01 and out["neg"] is not None), "corr_pos_vs_released": float(np.corrcoef(out["pos"], rel)[0, 1]) if out["pos"].std() > 0 and rel.std() > 0 else float("nan")})
            new_stats.append(st)
            new_ids.append(int(ids[i]))
            for k in vecs:
                vecs[k].append(out[k] if out[k] is not None else np.zeros(N_VOXELS, dtype=np.float32))
        for k, lst in vecs.items():
            x = normalise_paper(torch.as_tensor(np.stack(lst), dtype=torch.float32))
            new_lat[k].append(encode(ae, x, device).numpy())
        new_lat["released"].append(encode(ae, normalise_paper(released[torch.as_tensor(sel)].clone()), device).numpy())
        new_lat["released_raw"].append(encode(ae, normalise_paper(released_raw[torch.as_tensor(sel)].clone()), device).numpy())
    if new_ids:
        merged = {"ids": np.concatenate([done.get("ids", np.array([], dtype=np.int64)).astype(np.int64), np.asarray(new_ids, dtype=np.int64)])}
        for k in new_lat:
            prev = done.get(f"lat_{k}", np.zeros((0, 384), dtype=np.float32))
            merged[f"lat_{k}"] = np.concatenate([prev, np.concatenate(new_lat[k])]).astype(np.float32)
        stats_prev = pd.DataFrame(list(done["stats"])) if "stats" in done else pd.DataFrame()
        stats_all = pd.concat([stats_prev, pd.DataFrame(new_stats)], ignore_index=True)
        merged["stats"] = np.asarray(stats_all.to_dict("records"), dtype=object)
        np.savez(cache, **merged)
        done = merged
    pos = {int(i): j for j, i in enumerate(done["ids"].tolist())}
    order = np.asarray([pos[int(i)] for i in ids])
    latents = {k: torch.as_tensor(done[f"lat_{k}"][order]) for k in ("released", "released_raw", "pos", "abs", "neg")}
    stats = pd.DataFrame(list(done["stats"])).set_index("id").loc[ids].reset_index()
    return latents, stats


def paper_groups(meta: pd.DataFrame, pub_idx: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Papers of ``pub_idx`` with at least one map in ``meta`` and the row indices of their maps."""
    q = meta["pub_index"].astype(int).map({p: i for i, p in enumerate(pub_idx)}).to_numpy()
    groups = [np.flatnonzero(q == i) for i in range(len(pub_idx))]
    keep = np.asarray([len(g) > 0 for g in groups], dtype=bool)
    return pub_idx[keep], [g for g, k in zip(groups, keep) if k]


def option_bounds(z_text: torch.Tensor, z_img: dict[str, torch.Tensor], meta: pd.DataFrame, pub_idx: np.ndarray, masks: dict[str, np.ndarray],
                  n_null: int, seed: int) -> pd.DataFrame:
    """AUC and chance of the three target rules (and per map) for each handling of the 'other' maps; rows of ``z_text``
    follow ``pub_idx`` (sorted), rows of ``z_img`` follow ``meta``."""
    rows = []
    for opt, m in masks.items():
        sub = meta[m].reset_index(drop=True)
        pubs, groups = paper_groups(sub, pub_idx)
        zt = z_text[torch.as_tensor(np.searchsorted(pub_idx, pubs))]
        order_key = [sub["id"].to_numpy()[g] for g in groups]
        for var in ("released", "pos"):
            S = (zt @ z_img[var][torch.as_tensor(np.flatnonzero(m))].T).numpy()
            for direction, fn in (("text_to_brain", t2i_definitions), ("brain_to_text", i2t_definitions)):
                defs = fn(S, groups, order_key, np.random.default_rng(seed + 1), 0)
                nulls = [fn(S[deranged_permutation(len(S), np.random.default_rng(seed + 1000 + d))], groups, order_key, np.random.default_rng(seed + 2000 + d), 0)
                         for d in range(n_null)]
                rules = {"upper": "paper_upper", "lower": "worst_image", "multi": "oracle_any_true" if direction == "text_to_brain" else "per_image"}
                if direction == "text_to_brain":
                    rules["per map"] = "per_image"
                for rule, name in rules.items():
                    rows.append({"other_maps": opt, "variant": var, "direction": direction, "rule": rule, "definition": name, "n_publications": len(pubs),
                                 "n_images": int(m.sum()), "recall_curve_auc": rank_metrics(*defs[name][:2])["recall_curve_auc"],
                                 "chance_recall_curve_auc": float(np.mean([rank_metrics(*nl[name][:2])["recall_curve_auc"] for nl in nulls]))})
    return pd.DataFrame(rows)


def rank_metrics(ranks: np.ndarray, sizes: np.ndarray) -> dict[str, float]:
    r, c = ranks.astype(float), sizes.astype(float)
    out = {f"recall@{k}": float((r <= k).mean()) for k in KS}
    out["mrr"] = float((1.0 / r).mean())
    out["median_rank"] = float(np.median(r))
    out["recall_curve_auc"] = float(((c - r + 1.0) / c).mean())
    out["mean_pool_size"] = float(c.mean())
    return out


def t2i_definitions(S: np.ndarray, groups: list[np.ndarray], order_key: list[np.ndarray], rng: np.random.Generator, n_random: int) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Text -> image ranks per target rule: ``name -> (ranks, pool sizes, publication of each rank)``."""
    P, I = S.shape
    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    pubs = np.arange(P)

    def selected_pool(sel: np.ndarray) -> np.ndarray:
        sub = S[:, sel]  # [P, P]
        diag = sub[pubs, pubs]
        return 1 + (sub > diag[:, None]).sum(axis=1)

    best = np.asarray([g[int(np.argmax(S[p, g]))] for p, g in enumerate(groups)])
    worst = np.asarray([g[int(np.argmin(S[p, g]))] for p, g in enumerate(groups)])
    out["paper_upper"] = (selected_pool(best), np.full(P, P), pubs)
    out["worst_image"] = (selected_pool(worst), np.full(P, P), pubs)
    if n_random:  # --first-random-rules
        first = np.asarray([g[int(np.argmin(order_key[p]))] for p, g in enumerate(groups)])
        out["first_image"] = (selected_pool(first), np.full(P, P), pubs)
        rr, pp = [], []
        for _ in range(n_random):
            sel = np.asarray([rng.choice(g) for g in groups])
            rr.append(selected_pool(sel))
            pp.append(pubs)
        out["random_image"] = (np.concatenate(rr), np.full(P * n_random, P), np.concatenate(pp))
    any_r, worst_r, per_r, per_c, per_p = [], [], [], [], []
    for p, g in enumerate(groups):
        row = S[p]
        others = np.ones(I, dtype=bool)
        others[g] = False
        other_scores = row[others]
        pool = int(others.sum()) + 1
        true_scores = row[g]
        any_r.append(1 + int((other_scores > true_scores.max()).sum()))
        worst_r.append(len(g) + int((other_scores > true_scores.min()).sum()))
        per_r.extend((1 + (other_scores[None, :] > true_scores[:, None]).sum(axis=1)).tolist())
        per_c.extend([pool] * len(g))
        per_p.extend([p] * len(g))
    sizes_any = np.asarray([I - len(g) + 1 for g in groups])
    out["oracle_any_true"] = (np.asarray(any_r), sizes_any, pubs)
    out["worst_case"] = (np.asarray(worst_r), np.full(P, I), pubs)
    out["per_image"] = (np.asarray(per_r), np.asarray(per_c), np.asarray(per_p))
    return out


def i2t_definitions(S: np.ndarray, groups: list[np.ndarray], order_key: list[np.ndarray], rng: np.random.Generator, n_random: int) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Image -> text ranks per target rule; every image ranks the P texts."""
    P, I = S.shape
    ST = S.T  # [I, P]
    img_pub = np.empty(I, dtype=np.int64)
    for p, g in enumerate(groups):
        img_pub[g] = p
    true = ST[np.arange(I), img_pub]
    rank_img = 1 + (ST > true[:, None]).sum(axis=1)
    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    pubs = np.arange(P)
    out["per_image"] = (rank_img, np.full(I, P), img_pub)
    best = np.asarray([g[int(np.argmax(S[p, g]))] for p, g in enumerate(groups)])
    worst = np.asarray([g[int(np.argmin(S[p, g]))] for p, g in enumerate(groups)])
    out["paper_upper"] = (rank_img[best], np.full(P, P), pubs)
    out["worst_image"] = (rank_img[worst], np.full(P, P), pubs)
    out["oracle_any_true"] = (np.asarray([rank_img[g].min() for g in groups]), np.full(P, P), pubs)
    out["worst_case"] = (np.asarray([rank_img[g].max() for g in groups]), np.full(P, P), pubs)
    if n_random:  # --first-random-rules
        first = np.asarray([g[int(np.argmin(order_key[p]))] for p, g in enumerate(groups)])
        out["first_image"] = (rank_img[first], np.full(P, P), pubs)
        rr = [rank_img[np.asarray([rng.choice(g) for g in groups])] for _ in range(n_random)]
        out["random_image"] = (np.concatenate(rr), np.full(P * n_random, P), np.tile(pubs, n_random))
    return out


def cluster_bootstrap(ranks: np.ndarray, sizes: np.ndarray, units: np.ndarray, n_boot: int, seed: int) -> dict[str, tuple[float, float]]:
    """95 % CIs, resampling publications."""
    rng = np.random.default_rng(seed)
    pubs = np.unique(units)
    by_pub = {p: np.flatnonzero(units == p) for p in pubs}
    keys = [f"recall@{k}" for k in KS] + ["mrr", "median_rank", "recall_curve_auc"]
    acc: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(n_boot):
        draw = rng.choice(pubs, size=len(pubs), replace=True)
        idx = np.concatenate([by_pub[p] for p in draw])
        m = rank_metrics(ranks[idx], sizes[idx])
        for k in keys:
            acc[k].append(m[k])
    return {k: (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))) for k, v in acc.items()}


def normalised_curve(ranks: np.ndarray, sizes: np.ndarray, grid: np.ndarray) -> np.ndarray:
    u = ranks / sizes
    return np.array([(u <= t).mean() for t in grid])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=HERE / "outputs")
    p.add_argument("--model-dir", type=Path, default=M.DEFAULT_MODEL_DIR, help="Run directory with the trained models (default: runs/final).")
    p.add_argument("--device", default="auto", help="Device (auto/cpu/cuda).")
    p.add_argument("--max-samples", type=int, default=None, help="Number of publications (default: all).")
    p.add_argument("--smoke", action="store_true", help="48 publications, 100 bootstrap draws, 5 random / null draws.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--n-random-draws", type=int, default=20)
    p.add_argument("--first-random-rules", action="store_true", help="Also evaluate the first-image and random-image rules.")
    p.add_argument("--n-null-draws", type=int, default=20)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--qc-count-all", action="store_true", help="Also run the image-QC count over every eligible NIfTI (~4.6k files).")
    p.add_argument("--force", action="store_true", help="Recompute even if outputs exist, including the cached NIfTI latents.")
    p.add_argument("--no-exclusions", dest="exclusions", action="store_false", help="Keep the leakage-audit NeuroVault items.")
    p.add_argument("--other-maps", choices=("infer", "keep", "exclude"), default="infer",
                   help="Maps of NeuroVault type 'other': typed by _other_maps.py, mean / variance images and untyped maps dropped (infer), "
                        "all kept as signed maps (keep, the submission) or all dropped (exclude).")
    args = p.parse_args()
    if args.smoke:
        args.max_samples = args.max_samples or 48
        args.n_boot = min(args.n_boot, 100)
        args.n_random_draws = min(args.n_random_draws, 5)
        args.n_null_draws = min(args.n_null_draws, 5)
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = out / "cache"
    cache_dir.mkdir(exist_ok=True)
    timings = Timings(out / "timings.json")
    t_start = time.perf_counter()
    device = resolve_device(args.device)
    if (out / "bounds_metrics.csv").exists() and not args.force:
        print("[skip] outputs exist; use --force")
        return
    rng = np.random.default_rng(args.seed)
    M.from_args(args)
    from _hf_pins import load_dataset, load_latent, load_masker
    from neurovlm.data.loaders import get_data_dir

    root = get_data_dir()
    print(f"[config] device={device} model={M.model_tag()} smoke={args.smoke} max_samples={args.max_samples} data_dir={root}")

    with timings.time("metadata"):
        md = load_metadata()
        counts = count_table(md, root, args.qc_count_all, args.n_jobs, exclusions=args.exclusions)
        counts.to_csv(out / "counts_table.csv", index=False)
    print(counts[["stage", "n_images", "n_collections", "n_dois"]].to_string(index=False))
    meta_all, df_text = md["meta"], md["df_text"]
    excluded = meta_all[meta_all["map_class"] == "excluded"]
    meta_all = meta_all[(meta_all["map_class"] != "excluded") & meta_all["pub_index"].notna()].reset_index(drop=True)
    # leakage-audit exclusions (publications and all their images) before everything else
    n_images_before, n_pub_before = len(meta_all), len(df_text)
    pub_keep = neurovault_keep_mask(df_text["doi"].to_numpy(), enabled=args.exclusions, label="NeuroVault publications")
    img_keep = neurovault_keep_mask(meta_all["doi"].to_numpy(), image_ids=meta_all["id"].to_numpy(), collection_ids=meta_all["collection_id"].to_numpy(),
                                    enabled=args.exclusions, label="NeuroVault images")
    img_keep &= meta_all["pub_index"].astype(int).isin(np.flatnonzero(pub_keep)).to_numpy()
    meta_all = meta_all[img_keep].reset_index(drop=True)
    exclusions = {"enabled": bool(args.exclusions), "source": str(EVAL_EXCLUSIONS_DIR),
                  "n_neurovault_publications_before": int(n_pub_before), "n_neurovault_publications_dropped": int((~pub_keep).sum()),
                  "n_neurovault_images_before": int(n_images_before), "n_neurovault_images_dropped": int(n_images_before - len(meta_all)),
                  "n_pubmed_test_dropped": 0, "n_pubmed_val_dropped": 0, "note": "PubMed is not used by this analysis"}

    pool = np.flatnonzero(pub_keep)  # indices into df_text / the neurovault_text latents
    n_pub_total = len(pool)
    pub_idx = pool
    if args.max_samples:
        pub_idx = np.sort(pool[rng.permutation(n_pub_total)[: args.max_samples]])
    meta = meta_all[meta_all["pub_index"].isin(pub_idx)].reset_index(drop=True)
    pub_idx, _ = paper_groups(meta, pub_idx)
    print(f"[data] publications={len(pub_idx)} (of {n_pub_total} after exclusions) images={len(meta)} (released {len(meta_all)} after exclusions; "
          f"excluded map types in release: {len(excluded)}; leakage exclusions dropped {exclusions['n_neurovault_publications_dropped']} publications / "
          f"{exclusions['n_neurovault_images_dropped']} images)")

    with timings.time("load_models"):
        ae = M.component("autoencoder", device)
        text_head, image_head = M.component("proj_head_text_infonce", device), M.component("proj_head_image_infonce", device)
        masker = load_masker()
    with timings.time("load_released_images"):
        images = load_dataset("neurovault_images").float()
        meta_hf = load_dataset("neurovault_images_meta").reset_index(drop=True)
        row_of = {int(i): r for r, i in enumerate(meta_hf["id"].to_numpy())}
        released_raw = images[torch.as_tensor([row_of[int(i)] for i in meta["id"]])].clone()
        del images
        released, p_rebuilt = released_with_inverted_p(released_raw, meta, root, masker)
        p_rebuilt.to_csv(out / "p_maps_inverted.csv", index=False)
    print(f"[p-maps] {len(p_rebuilt)} p-maps in {p_rebuilt['doi'].nunique() if len(p_rebuilt) else 0} papers rebuilt as 1 - p for variant 'released'")
    with timings.time("preprocess"):
        latents, stats = preprocess_images(meta, root, masker, ae, device, cache_dir, args.n_jobs, released, released_raw, force=args.force)
        del released, released_raw
    print(f"[sanity] median corr(pos, released) = {stats['corr_pos_vs_released'].median():.4f}; images with a negative tail (>=1 % voxels): {int(stats['has_negative_tail'].sum())}")

    with timings.time("embeddings"):
        text_lat = load_latent("neurovault_text").float()[torch.as_tensor(pub_idx)]
        with torch.no_grad():
            z_text = l2n(text_head(l2n(text_lat).to(device))).cpu()
            z_img = {k: l2n(image_head(v.to(device))).cpu() for k, v in latents.items()}

    # maps of NeuroVault type 'other': kind from names, descriptions, siblings and values; bounds under each handling
    with timings.time("other_maps"):
        vals = meta[["id", "collection_id", "doi", "map_type"]].merge(stats[["id", "frac_negative", "min", "max", "q001", "q01", "q99", "q999"]], on="id", how="left")
        other = OM.other_map_kinds(vals, md["local_img"], md["local_col"])
        other.to_csv(out / "other_maps.csv", index=False)
        OM.name_rule_check(vals, md["local_img"]).to_csv(out / "other_maps_rule_check.csv", index=False)
        is_other = meta["map_type"].eq("other").to_numpy()
        eval_type = dict(zip(other["id"], other["map_type_eval"]))
        masks = {"keep": np.ones(len(meta), dtype=bool), "infer": ~is_other | meta["id"].map(eval_type).notna().to_numpy(), "exclude": ~is_other}
        option_bounds(z_text, z_img, meta, pub_idx, masks, args.n_null_draws, args.seed).to_csv(out / "other_maps_options.csv", index=False)
        sel = np.flatnonzero(masks[args.other_maps])
        n_drop_img, n_pub_sup = len(meta) - len(sel), len(pub_idx)
        meta, stats = meta.iloc[sel].reset_index(drop=True), stats.iloc[sel].reset_index(drop=True)
        z_img = {k: v[torch.as_tensor(sel)] for k, v in z_img.items()}
        if args.other_maps == "infer":
            meta["map_type"] = [eval_type.get(i) or t for i, t in zip(meta["id"], meta["map_type"])]
            stats["map_type"] = meta["map_type"].to_numpy()
        pub_sup = pub_idx
        pub_idx, groups = paper_groups(meta, pub_sup)
        z_text = z_text[torch.as_tensor(np.searchsorted(pub_sup, pub_idx))]
        meta["q"] = meta["pub_index"].astype(int).map({p: i for i, p in enumerate(pub_idx)})
        order_key = [meta["id"].to_numpy()[g] for g in groups]
    print(f"[other maps] {int(is_other.sum())} maps of type 'other': " + ", ".join(f"{k} {v}" for k, v in other["kind"].value_counts().items())
          + f"; --other-maps {args.other_maps}: {n_drop_img} images / {n_pub_sup - len(pub_idx)} publications dropped -> {len(meta)} images, {len(pub_idx)} publications")
    if args.other_maps != "keep":
        rule = "typed from names and values; mean or variance images and untyped maps removed" if args.other_maps == "infer" else "all removed"
        row = {"stage": f"J. ... maps of NeuroVault type 'other' {rule} (--other-maps {args.other_maps})", "n_images": len(meta),
               "n_collections": int(meta["collection_id"].nunique()), "n_dois": int(meta["doi"].nunique()),
               "note": f"{n_drop_img} images / {n_pub_sup - len(pub_idx)} publications dropped (outputs/other_maps.csv); the set every metric here is computed on"}
        counts = pd.concat([counts, pd.DataFrame([row])], ignore_index=True)
        counts.to_csv(out / "counts_table.csv", index=False)
    stats = stats.merge(meta[["id", "doi", "q"]], on="id", how="left")
    stats.to_csv(out / "image_preprocessing_stats.csv", index=False)
    mt = meta.groupby("map_type").agg(n_images=("id", "size"), n_publications=("doi", "nunique")).reset_index()
    mt["map_class"] = mt["map_type"].map(map_class)
    mt["preprocessing"] = mt["map_class"].map({"signed": "positive tail (paper) | |value| | sign split", "p_map": "inverted (1 - p) before normalisation", "inv_p_map": "used directly", "mask": "binary, used directly"})
    mt["share_images"] = mt["n_images"] / mt["n_images"].sum()
    mt.sort_values("n_images", ascending=False).to_csv(out / "map_type_counts.csv", index=False)
    # candidates per variant; signsplit = positive maps plus the negative tails (when present)
    neg_ok = stats["has_negative_tail"].to_numpy()
    variant_sets: dict[str, dict[str, Any]] = {}
    for var in ("released", "released_raw", "pos", "abs"):
        variant_sets[var] = {"z": z_img[var], "groups": groups, "order_key": order_key, "map_type": meta["map_type"].to_numpy(), "tail": np.array(["pos"] * len(meta))}
    extra = np.flatnonzero(neg_ok)
    z_ss = torch.cat([z_img["pos"], z_img["neg"][torch.as_tensor(extra)]])
    g_ss = [np.concatenate([g, len(meta) + np.flatnonzero(np.isin(extra, g))]) for g in groups]
    ok_ss = [np.concatenate([k, k[np.isin(g, extra)]]) for g, k in zip(groups, order_key)]
    variant_sets["signsplit"] = {"z": z_ss, "groups": g_ss, "order_key": ok_ss, "map_type": np.concatenate([meta["map_type"].to_numpy(), meta["map_type"].to_numpy()[extra]]), "tail": np.array(["pos"] * len(meta) + ["neg"] * len(extra))}

    grid = np.linspace(0.0, 1.0, 201)[1:]
    rows, curves, per_type_rows, per_unit = [], [], [], []
    with timings.time("definitions"):
        for var, vs in tqdm(variant_sets.items(), desc="variants", leave=False):
            S = (z_text @ vs["z"].T).numpy()
            for direction, fn in (("text_to_brain", t2i_definitions), ("brain_to_text", i2t_definitions)):
                n_rand = args.n_random_draws if args.first_random_rules else 0
                defs = fn(S, vs["groups"], vs["order_key"], np.random.default_rng(args.seed + 1), n_rand)
                nulls = []
                for d in range(args.n_null_draws):
                    perm = deranged_permutation(S.shape[0], np.random.default_rng(args.seed + 1000 + d))
                    nulls.append(fn(S[perm], vs["groups"], vs["order_key"], np.random.default_rng(args.seed + 2000 + d), min(n_rand, 1)))
                for name, (ranks, sizes, units) in defs.items():
                    m = rank_metrics(ranks, sizes)
                    ci = cluster_bootstrap(ranks, sizes, units, args.n_boot, args.seed)
                    null_m = pd.DataFrame([rank_metrics(nl[name][0], nl[name][1]) for nl in nulls]).mean()
                    rec = {"variant": var, "direction": direction, "definition": name, "n_queries": int(len(ranks)) if name != "random_image" else int(len(ranks) // args.n_random_draws), "n_publications": int(len(np.unique(units))), **m}
                    for k, (lo, hi) in ci.items():
                        rec[f"{k}_ci_lo"], rec[f"{k}_ci_hi"] = lo, hi
                    rec.update({f"chance_{k}": float(null_m[k]) for k in ("recall@1", "recall@5", "recall@10", "mrr", "median_rank", "recall_curve_auc")})
                    rows.append(rec)
                    rc = normalised_curve(ranks, sizes, grid)
                    nc = np.mean([normalised_curve(nl[name][0], nl[name][1], grid) for nl in nulls], axis=0)
                    curves.extend({"variant": var, "direction": direction, "definition": name, "normalized_k": float(t), "recall": float(r), "chance": float(c)} for t, r, c in zip(grid, rc, nc))
                    if name == "per_image":
                        # text -> brain ranks come publication by publication in group order
                        img_index = np.concatenate(vs["groups"]) if direction == "text_to_brain" else np.arange(len(vs["map_type"]))
                        per_unit.append(pd.DataFrame({"variant": var, "direction": direction, "candidate_index": img_index, "map_type": vs["map_type"][img_index], "tail": vs["tail"][img_index], "publication": units, "rank": ranks, "pool_size": sizes}))
                        for mtype, sub in pd.DataFrame({"map_type": vs["map_type"][img_index], "rank": ranks, "size": sizes, "unit": units}).groupby("map_type"):
                            mm = rank_metrics(sub["rank"].to_numpy(), sub["size"].to_numpy())
                            per_type_rows.append({"variant": var, "direction": direction, "map_type": mtype, "n_images": int(len(sub)), "n_publications": int(sub["unit"].nunique()), **mm})
    table = pd.DataFrame(rows)
    table.to_csv(out / "bounds_metrics.csv", index=False)
    pd.DataFrame(curves).to_csv(out / "recall_curves.csv", index=False)
    pd.DataFrame(per_type_rows).to_csv(out / "per_map_type_metrics.csv", index=False)
    pd.concat(per_unit, ignore_index=True).to_parquet(out / "per_image_ranks.parquet", index=False)
    # map types picked by the contrastive selection
    S_pos = (z_text @ z_img["pos"].T).numpy()
    best = np.asarray([g[int(np.argmax(S_pos[p, g]))] for p, g in enumerate(groups)])
    first = np.asarray([g[int(np.argmin(order_key[p]))] for p, g in enumerate(groups)])
    sel = pd.DataFrame({"map_type": meta["map_type"].to_numpy()})
    sel_tab = pd.DataFrame({"all_images": sel["map_type"].value_counts(normalize=True), "paper_selected": pd.Series(meta["map_type"].to_numpy()[best]).value_counts(normalize=True), "first_image": pd.Series(meta["map_type"].to_numpy()[first]).value_counts(normalize=True)}).fillna(0.0)
    sel_tab.reset_index().rename(columns={"index": "map_type"}).to_csv(out / "selection_map_type_shares.csv", index=False)
    pd.DataFrame({"doi": df_text["doi"].to_numpy()[pub_idx], "n_images": [len(g) for g in groups], "paper_selected_image_id": meta["id"].to_numpy()[best], "first_image_id": meta["id"].to_numpy()[first], "paper_selected_map_type": meta["map_type"].to_numpy()[best]}).to_csv(out / "publication_targets.csv", index=False)

    config = {
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "device": str(device),
        **M.model_info(),
        "data_dir": str(root),
        "n_publications": int(len(pub_idx)),
        "n_publications_total": int(n_pub_total),
        "n_images": int(len(meta)),
        "n_images_released": int(len(meta_all)),
        "other_maps": args.other_maps,
        "other_map_kinds": {k: int(v) for k, v in other["kind"].value_counts().items()},
        "n_other_maps_dropped": int(n_drop_img),
        "n_publications_dropped_other_maps": int(n_pub_sup - len(pub_idx)),
        "exclusions": exclusions,
        "n_negative_tail_images": int(neg_ok.sum()),
        "excluded_map_types": EXCLUDED_TYPES,
        "n_p_maps_inverted": int(len(p_rebuilt)),
        "variants": {"released": "published neurovault_images tensor, p-maps rebuilt from the NIfTI as 1 - p with the release threshold (Fig. 5b)", "released_raw": "published neurovault_images tensor as released (negatives clipped, p-maps not inverted)", "pos": "positive tail from raw NIfTI with map-type branching (p-maps inverted)", "abs": "|value| for signed maps", "signsplit": "positive maps plus negative-tail maps (>=1 % negative voxels) as extra candidates"},
        "definitions": {"paper_upper": "contrastive-selected image per publication (target dependent, UPPER bound)", "worst_image": "image the model scores lowest for the publication text, evaluated like paper_upper (LOWER bound)", "oracle_any_true": "all images candidates; any image of the publication in top k", "first_image": "lowest NeuroVault id per publication (--first-random-rules only)", "random_image": f"random image per publication, {args.n_random_draws} draws (--first-random-rules only)", "per_image": "each image a target among all images (siblings not counted as competitors)", "worst_case": "rank of the last image of the publication"},
        "rank_convention": "1 + number of competitors with strictly larger cosine",
        "chance": f"texts deranged across publications, {args.n_null_draws} draws",
        "wall_time_s": round(time.perf_counter() - t_start, 1),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, default=str))
    timings.records["total_wall_time_s"] = config["wall_time_s"]
    timings.records["exclusions"] = exclusions
    timings.save()
    show = table[table["variant"] == "pos"][["direction", "definition", "n_queries", "mean_pool_size", "recall@1", "recall@5", "recall@10", "mrr", "median_rank", "recall_curve_auc", "chance_recall_curve_auc"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"[done] {config['wall_time_s']} s -> {out}")


if __name__ == "__main__":
    main()
