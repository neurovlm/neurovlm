#!/usr/bin/env python
"""Reviewer 3.5: text-to-brain maps from network-only, anatomy-only or cognition-only text, scored against references
NeuroVLM never produced (Yeo-7 and network-atlas maps, Harvard-Oxford / AAL ROIs, Neurosynth maps of the same term).
Models of --model-dir (default runs/final: text-to-brain head v2); generated maps are divided by their maximum.
  .venv/bin/python scripts/figures/review/core/a6_modality_restricted_t2b/run_analysis.py [--smoke] [--device cpu]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _a6_common as C  # noqa: E402
import _final_model as M  # noqa: E402

log = C.get_logger("a6_modality_t2b")
ICA_ATLASES = ("UKBICA", "HCPICA")
LEVEL_OF = {"yeo7": "network", "network_atlases": "network", "harvard_oxford": "anatomy", "aal": "anatomy",
            "cogatlas_neurosynth": "cognition"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=HERE / "outputs", help="Output directory.")
    M.add_model_args(p)
    p.add_argument("--device", default="auto", help="Device (auto/cpu/cuda).")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--smoke", action="store_true", help="At most 12 items per source.")
    p.add_argument("--max-samples", type=int, default=None, help="Max items per source.")
    p.add_argument("--neurosynth-dir", type=Path, default=C.default_neurosynth_dir(), help="Neurosynth v7 data (nimare layout).")
    p.add_argument("--min-studies", type=int, default=50, help="Min Neurosynth studies per Cognitive Atlas term.")
    p.add_argument("--force", action="store_true", help="No effect (kept for run_all.sh).")
    a = p.parse_args()
    if a.smoke and not a.max_samples:
        a.max_samples = 12
    return a


def ho_text(name: str) -> str:
    """Harvard-Oxford label as plain text ("Heschl's Gyrus (includes H1 and H2)" -> "Heschl's Gyrus")."""
    m = re.match(r"(.*?)\s*\(formerly (.*)\)", name)
    return m.group(2) if m else re.sub(r"\s*\(.*\)", "", name).strip()


def network_atlas_items(cache_dir: Path) -> dict:
    """Labelled network-atlas maps (binary as-is, continuous binarised at the 95th percentile) and their names."""
    cache = cache_dir / "network_atlas_maps.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}
    import nibabel as nib
    from nilearn.image import resample_img
    from _hf_pins import load_dataset

    masker = C.get_masker()
    nets = load_dataset("networks")
    lab = load_dataset("network_test_set_labels").drop_duplicates("raw_network_label").set_index("raw_network_label")
    ids, keys, names, masks = [], [], [], []
    for atlas in nets:
        if atlas in ICA_ATLASES:
            continue
        for raw, payload in nets[atlas].items():
            if raw not in lab.index or str(lab.loc[raw, "network_key"]) == "unknown":
                continue
            arr = np.asarray(payload["array"], np.float32)
            if arr.ndim == 4:
                arr = arr[..., 0]
            img = nib.Nifti1Image(arr, affine=payload["affine"])
            if len(np.unique(arr)) == 2:
                res = resample_img(img, masker.affine_, interpolation="nearest", force_resample=True, copy_header=True)
            else:
                res = resample_img(img, masker.affine_, force_resample=True, copy_header=True)
                a = res.get_fdata()
                a[a < 0] = 0.0
                thr = np.percentile(a.flatten(), 95)
                res = nib.Nifti1Image((a >= thr).astype(np.float32), affine=masker.affine_)
            ids.append(f"{atlas}:{raw}")
            keys.append(str(lab.loc[raw, "network_key"]))
            names.append(str(lab.loc[raw, "network_name"]))
            masks.append(np.asarray(masker.transform(res)).reshape(-1) > 0)
    out = {"ids": np.array(ids), "keys": np.array(keys), "names": np.array(names), "masks": np.stack(masks)}
    np.savez_compressed(cache, **out)
    return out


def build_items(args) -> dict[str, dict]:
    cache_dir = args.out / "cache"
    srcs: dict[str, dict] = {}
    y = C.yeo7(cache_dir)
    srcs["yeo7"] = {"ids": list(y["names"]), "text": [n[:1].upper() + n[1:] for n in y["names"]], "ref": y["masks"],
                    "binary": True, "group": list(y["names"])}
    na = network_atlas_items(cache_dir)
    srcs["network_atlases"] = {"ids": list(na["ids"]), "text": [f"{n} network" for n in na["names"]], "ref": na["masks"],
                               "binary": True, "group": list(na["keys"])}
    ho = C.harvard_oxford(cache_dir)
    srcs["harvard_oxford"] = {"ids": list(ho["names"]), "text": [ho_text(n) for n in ho["names"]], "ref": ho["masks"],
                              "binary": True, "group": list(ho["names"])}
    al = C.aal(cache_dir)
    srcs["aal"] = {"ids": list(al["names"]), "text": list(al["text"]), "ref": al["masks"], "binary": True,
                   "group": list(al["names"])}
    ns = C.Neurosynth(args.neurosynth_dir, cache_dir)
    cog_file = C.data_dir() / "cogatlas.parquet"
    if cog_file.exists():
        cog = pd.read_parquet(cog_file)
    else:
        from _hf_pins import load_dataset

        cog = load_dataset("cogatlas")
    terms = sorted({str(t).strip().lower() for t in cog["term"].dropna()})
    keep = [t for t in terms if ns.has(t) and ns.count(t) >= args.min_studies]
    srcs["cogatlas_neurosynth"] = {"ids": keep, "text": keep, "ref": np.stack([ns.get(t) for t in keep]), "binary": False,
                                   "group": keep, "n_studies": [ns.count(t) for t in keep]}
    meta = {"neurosynth_source": ns.source, "n_cogatlas_terms": len(terms), "n_cogatlas_in_neurosynth": int(sum(ns.has(t) for t in terms)),
            "min_studies": args.min_studies}
    if args.max_samples:
        for k, s in srcs.items():
            n = min(args.max_samples, len(s["ids"]))
            if k == "network_atlases":  # keep >= 2 networks
                order = np.argsort(s["group"], kind="stable")
                idx = np.sort(np.concatenate([order[:n // 2], order[-(n - n // 2):]]))
            else:
                idx = np.arange(n)
            for f in ("ids", "text", "group", "n_studies"):
                if f in s:
                    s[f] = [s[f][i] for i in idx]
            s["ref"] = s["ref"][idx]
    return srcs, meta


def binarise_pred(P: np.ndarray, pct: float = C.DICE_PCT) -> np.ndarray:
    return P > np.percentile(P, pct, axis=1, keepdims=True)


def dice_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A, B = A.astype(np.float32), B.astype(np.float32)
    inter = A @ B.T
    return 2 * inter / np.clip(A.sum(1)[:, None] + B.sum(1)[None, :], 1, None)


def topk_dice_matrix(P: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Size-matched Dice: for reference j, the |R_j| highest voxels of prediction i vs R_j."""
    sizes = R.sum(1).astype(int)
    order = np.argsort(-P, axis=1)
    out = np.zeros((len(P), len(R)), np.float32)
    for k in np.unique(sizes):
        if k == 0:
            continue
        sel = np.zeros_like(P, dtype=bool)
        np.put_along_axis(sel, order[:, :k], True, axis=1)
        cols = np.flatnonzero(sizes == k)
        out[:, cols] = dice_matrix(sel, R[cols].astype(bool))
    return out


def score_source(name: str, src: dict, P: np.ndarray, rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    """Per-item scores; nulls average over references of other groups. Also returns the Pearson matrix."""
    R = src["ref"]
    group = np.asarray(src["group"])
    Rb = R.astype(bool) if src["binary"] else R > np.percentile(R, C.DICE_PCT, axis=1, keepdims=True)
    D = dice_matrix(binarise_pred(P), Rb)
    Dk = topk_dice_matrix(P, Rb) if src["binary"] else np.full_like(D, np.nan)
    Rr = C.pearson_matrix(P, R.astype(np.float32))
    n = len(P)
    same = group[:, None] == group[None, :]
    rows = []
    perm = C.deranged_indices(n, rng)
    for i in range(n):
        other = ~same[i]
        r = {"source": name, "level": LEVEL_OF[name], "id": src["ids"][i], "text": src["text"][i], "group": group[i],
             "dice": float(D[i, i]), "dice_topk": float(Dk[i, i]), "pearson": float(Rr[i, i]),
             "dice_null": float(D[i, other].mean()) if other.any() else np.nan,
             "dice_topk_null": float(Dk[i, other].mean()) if other.any() else np.nan,
             "pearson_null": float(Rr[i, other].mean()) if other.any() else np.nan,
             "dice_shuffled": float(D[i, perm[i]]), "pearson_shuffled": float(Rr[i, perm[i]]),
             # rank of the matching reference among itself and the references of other groups
             "rank": int(1 + np.sum(Rr[i, other] > Rr[i, i])), "n_candidates": int(1 + other.sum())}
        if "n_studies" in src:
            r["n_studies"] = int(src["n_studies"][i])
        rows.append(r)
    df = pd.DataFrame(rows)
    df["top1"] = df["rank"] == 1
    df["norm_rank"] = (df["rank"] - 1) / np.clip(df["n_candidates"] - 1, 1, None)
    return df, Rr


def laterality(srcs: dict, preds: dict) -> pd.DataFrame:
    """For lateralised items: mean prediction in the named-side ROI vs its contralateral twin."""
    rows = []
    for name in ("aal", "harvard_oxford"):
        ids = list(srcs[name]["ids"])
        R = srcs[name]["ref"].astype(bool)
        for i, rid in enumerate(ids):
            if name == "aal":
                m = re.match(r"(.*)_([LR])$", rid)
                if not m:
                    continue
                twin = f"{m.group(1)}_{'R' if m.group(2) == 'L' else 'L'}"
                side = m.group(2)
            else:
                m = re.match(r"^(Left|Right) (.*)$", rid)
                if not m:
                    continue
                twin = f"{'Right' if m.group(1) == 'Left' else 'Left'} {m.group(2)}"
                side = m.group(1)[0]
            if twin not in ids:
                continue
            j = ids.index(twin)
            p = preds[name][i]
            a, b = float(p[R[i]].mean()), float(p[R[j]].mean())
            rows.append({"source": name, "id": rid, "side": side, "mean_named": a, "mean_contra": b,
                         "laterality_index": (a - b) / (a + b) if a + b > 0 else 0.0})
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    out = []
    for key, g in df.groupby(by, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        s = dict(zip(by, key))
        s["n"] = len(g)
        for m in ("dice", "dice_topk", "pearson"):
            s[f"{m}_matched"] = float(g[m].mean())
            s[f"{m}_null"] = float(g[f"{m}_null"].mean())
            lo, hi = C.bootstrap_ci(g[m])
            s[f"{m}_ci_lo"], s[f"{m}_ci_hi"] = lo, hi
        for m in ("dice", "pearson"):
            t = C.paired_test(g[m], g[f"{m}_null"])
            s[f"{m}_t"], s[f"{m}_p"] = t.get("t", np.nan), t.get("p_t", np.nan)
            s[f"{m}_frac_gt_null"] = float((g[m] > g[f"{m}_null"]).mean())
        s["top1_acc"] = float(g["top1"].mean())
        s["chance_top1"] = float((1 / g["n_candidates"]).mean())
        s["median_rank"] = float(g["rank"].median())
        s["mean_norm_rank"] = float(g["norm_rank"].mean())
        out.append(s)
    return pd.DataFrame(out)


def main() -> None:
    args = parse_args()
    C.set_offline_env()
    C.seed_all(args.seed)
    M.from_args(args)
    args.out.mkdir(parents=True, exist_ok=True)
    device = C.select_device(args.device)
    timings = C.Timings(args.out / "timings.json")
    rng = np.random.default_rng(args.seed)
    log.info("device=%s model=%s smoke=%s out=%s", device, M.model_tag(), args.smoke, args.out)
    with timings.time("build_items"):
        srcs, meta = build_items(args)
    log.info("items: %s", {k: len(v["ids"]) for k, v in srcs.items()})
    with timings.time("load_models", device=str(device)):
        models = C.load_t2b_models(device)
    preds, frames, pearson = {}, [], {}
    for name, src in srcs.items():
        with timings.time(f"t2b_{name}", n=len(src["ids"])):
            P = C.text_to_brain(models, src["text"], device)
        preds[name] = P
        df, pearson[name] = score_source(name, src, P, rng)
        frames.append(df)
        log.info("%s: dice %.3f (null %.3f) pearson %.3f (null %.3f) top1 %.2f", name, df["dice"].mean(), df["dice_null"].mean(),
                 df["pearson"].mean(), df["pearson_null"].mean(), df["top1"].mean())
    items = pd.concat(frames, ignore_index=True)
    items.to_csv(args.out / "per_item.csv", index=False)
    by_source = summarise(items, ["level", "source"])
    by_level = summarise(items, ["level"])
    by_source.to_csv(args.out / "summary_by_source.csv", index=False)
    by_level.to_csv(args.out / "summary_by_level.csv", index=False)
    lat = laterality(srcs, preds)
    lat.to_csv(args.out / "laterality.csv", index=False)
    # predictions and references, plus NIfTI files of the first 3 predictions per source
    np.savez_compressed(args.out / "predictions.npz",
                        **{f"pred__{k}": v.astype(np.float16) for k, v in preds.items()},
                        **{f"ref__{k}": srcs[k]["ref"].astype(np.float16) for k in srcs},
                        **{f"ids__{k}": np.array(srcs[k]["ids"]) for k in srcs},
                        **{f"pearson__{k}": pearson[k].astype(np.float32) for k in srcs})
    nii = args.out / "nifti"
    nii.mkdir(exist_ok=True)
    for k, v in preds.items():
        for i in range(min(3, len(v))):
            stem = re.sub(r"[^A-Za-z0-9]+", "_", f"{k}_{srcs[k]['ids'][i]}").strip("_")
            C.unmask(v[i]).to_filename(nii / f"pred_{stem}.nii.gz")
    res = {"config": {**vars(args), "out": str(args.out), "neurosynth_dir": str(args.neurosynth_dir), "device": str(device),
                      "dice_pct": C.DICE_PCT, **M.model_info()},
           "meta": meta, "n_items": {k: len(v["ids"]) for k, v in srcs.items()},
           "by_level": by_level.to_dict(orient="records"), "by_source": by_source.to_dict(orient="records"),
           "laterality": {"n": len(lat), "mean_index": float(lat["laterality_index"].mean()) if len(lat) else None,
                          "frac_positive": float((lat["laterality_index"] > 0).mean()) if len(lat) else None},
           "versions": C.installed_versions()}
    C.write_json(args.out / "results.json", res)
    log.info("by level:\n%s", by_level[["level", "n", "dice_matched", "dice_null", "pearson_matched", "pearson_null", "top1_acc", "chance_top1"]])
    log.info("done -> %s", args.out)


if __name__ == "__main__":
    main()
