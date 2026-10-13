#!/usr/bin/env python
"""Percentiles of the paper's qualitative examples (3.8) and failure-case sets (2.m6), final model.
Each example of Figs. 2, 3 and S4 (from the figures' *_text.json) is scored with its figure's recipe, ported from
scripts/figures/main, and ranked within the same score over the whole set (leakage-audit exclusions applied).
Generated maps are divided by their maximum before they are encoded, scored or drawn (scripts/README.md, "Map calibration").

    run_analysis.py --device cuda --gen-batch 64
    run_analysis.py --smoke --device cpu
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _figure_constants import COHORTS, KEEP_TEXT_IDXS, NETS  # noqa: E402
from _review_common import (  # noqa: E402
    DEFAULT_MODEL_DIR, RUN_FILES, Timings, calibrate_maps, component, encode_latents, encode_specter, env_packages, get_logger, heldout_pmids,
    load_projector, load_pubmed_maps, load_pubmed_text_rows, make_nvlm, model_dir, model_sources, neurovault_keep_mask, percentile_of,
    release_cuda_cache, seed_all, select_device, set_model_dir, set_offline_env, shared_brain, shared_text, write_json,
)

set_offline_env()
log = get_logger("a7_failure_cases_percentiles")
SIDECAR_DIR_DEFAULT = HERE.parents[2] / "main"  # scripts/figures/main (final paper figures)
SIDECAR_STEMS = {"newgen": "fig3_newgen", "networks_examples": "fig2_network_examples", "clip_ica_hcp_ukb": "figS4_clip_ica"}
VALIDATION_FINAL = HERE.parents[4] / "scripts" / "outputs" / "validation" / "final"  # validator's stage-2 PubMed test captions
BASES = ("network", "region", "function")
ICA_ATLASES = tuple(c for c, _ in COHORTS)
FIG3_GROUPS = ["High", "High", "Medium", "Medium", "Low", "Low"]
T2B_LABEL = {"head": "text-to-brain head", "adapter": "NeuroAdapter"}


def pearson_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    return (a * b).sum(axis=1) / np.clip(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12, None)


def text_shared(texts, specter, text_head, device):
    """L2-normalised SPECTER2 embeddings and their shared-space vectors."""
    import torch.nn.functional as F

    e = F.normalize(encode_specter(specter, list(texts)), dim=1)
    return e, shared_text(text_head, e, device)


def brain_shared(X, ae, image_head, device):
    import torch

    return shared_brain(image_head, encode_latents(ae.encoder, torch.as_tensor(np.asarray(X, dtype=np.float32)), device), device)


@functools.cache
def model_fingerprint() -> str:
    """First 12 hex digits of the sha256 over the run files (RUN_FILES) of --model-dir; "released" without one."""
    if model_dir() is None:
        return "released"
    h = hashlib.sha256()
    for rel in sorted(RUN_FILES.values()):
        f = model_dir() / rel
        if f.exists():
            h.update(rel.encode())
            with open(f, "rb") as fh:
                while chunk := fh.read(1 << 24):
                    h.update(chunk)
    return h.hexdigest()[:12]


def cached_generation(path: Path, ids: list[str], recipe: dict, fn, *, force: bool) -> list[str]:
    """Generations cached by (item id, recipe, model fingerprint); only missing ids are generated (all with ``force``)."""
    key = json.dumps({**recipe, "model": model_fingerprint()}, sort_keys=True)
    have = {}
    if path.exists() and not force:
        d = pd.read_csv(path, keep_default_na=False)
        d = d[d["recipe"] == key]
        have = dict(zip(d["id"].astype(str), d["generated"].astype(str)))
    todo = [i for i in ids if i not in have]
    if todo:
        have.update(dict(zip(todo, fn(todo))))
        rows = pd.DataFrame({"id": list(have), "generated": list(have.values()), "recipe": key})
        if path.exists():
            old = pd.read_csv(path, keep_default_na=False)
            rows = pd.concat([old[old["recipe"] != key], rows], ignore_index=True)
        rows.to_csv(path, index=False)
    return [have[i] for i in ids]


def t2b_maps(models: dict, e, device, t2b_model: str, batch: int = 64) -> np.ndarray:
    """Calibrated maps from L2-normalised SPECTER2 embeddings: head = sigmoid(AE decoder(proj_head_text_mse(e))) (the paper's
    model), adapter = sigmoid(NeuroAdapter(e)); calibrated with that model's global factor."""
    import torch

    def f(x):
        return models["adapter"](x) if t2b_model == "adapter" else models["ae"].decoder(models["mse_head"](x))

    with torch.no_grad():
        kind = "adapter" if t2b_model == "adapter" else "head"
        return torch.cat([calibrate_maps(torch.sigmoid(f(e[s:s + batch].to(device))).float(), kind=kind).cpu()
                          for s in range(0, len(e), batch)]).numpy()


def batched_generate(nvlm, X: np.ndarray, batch: int, **kw) -> list[str]:
    import torch
    from tqdm.auto import tqdm
    from _hf_pins import pin_neuroqwen

    pin_neuroqwen(nvlm)  # NeuroQwen3 at its pinned revision, not the client's lazy main-branch load
    out = []
    for s in tqdm(range(0, len(X), batch), desc="generate", leave=False):
        r = nvlm.generate_text(torch.from_numpy(np.asarray(X[s:s + batch], dtype=np.float32)), **kw)
        out.extend([r] if isinstance(r, str) else list(r))
    return out


def load_neurovault():
    """NeuroVault papers, maps (scaled by their 99th percentile, clamped to [0, 1]) and map metadata."""
    from _hf_pins import load_dataset

    df_text = load_dataset("neurovault_text").reset_index(drop=True)
    images = load_dataset("neurovault_images").float()
    images = (images / images.quantile(0.99, dim=1).unsqueeze(1)).clamp(0, 1)
    meta = load_dataset("neurovault_images_meta").reset_index(drop=True)
    return df_text, images, meta


def fig3(sidecar: dict, models: dict, nvlm_fn, *, device, args, out: Path, timings: Timings, excl_record: dict):
    """Fig. 3: every NeuroVault paper, scored on the DOI's map most similar to the paper text (figure rule)."""
    import torch

    with timings.time("fig3_data"):
        df_text, images, meta = load_neurovault()
        # exclusions on the full tables, before any truncation
        use_excl = not args.no_exclusions
        pub_ok = neurovault_keep_mask(df_text["doi"].to_numpy(), enabled=use_excl, label="NeuroVault publications (Fig 3 pool)", log=log.info)
        img_ok = neurovault_keep_mask(meta["doi"].to_numpy(), image_ids=meta["id"].to_numpy(), collection_ids=meta["collection_id"].to_numpy(),
                                      enabled=use_excl, label="NeuroVault images (Fig 3 pool)", log=log.info)
        n_img_dropped = int((~img_ok).sum())
        ok = torch.where(~images.isnan().any(dim=1))[0]
        images = images[ok].numpy()
        meta = meta.iloc[ok.numpy()].reset_index(drop=True)
        img_ok = img_ok[ok.numpy()]
        texts = (df_text["title"].astype(str) + " [SEP] " + df_text["abstract"].astype(str)).tolist()
    excluded_t = set(np.flatnonzero(~pub_ok).tolist())
    flagged = sorted(set(KEEP_TEXT_IDXS) & excluded_t)
    excl_record.update({"n_neurovault_publications_total": int(len(pub_ok)), "n_neurovault_publications_dropped": int((~pub_ok).sum()),
                        "n_neurovault_images_dropped": n_img_dropped, "n_neurovault_finite_images_dropped": int((~img_ok).sum()),
                        "dropped_dois": sorted(str(df_text.loc[t, "doi"]) for t in excluded_t),
                        "flagged_examples": [{"text_idx": int(t), "doi": str(df_text.loc[t, "doi"]), "title": str(df_text.loc[t, "title"])}
                                             for t in flagged]})
    if flagged:
        log.warning("[eval exclusions] Fig 3 examples %s are excluded: ranked against the cleaned pool and flagged", flagged)
    else:
        log.info("[eval exclusions] none of the Fig 3 examples %s is excluded", list(KEEP_TEXT_IDXS))
    pool = [t for t in range(len(df_text)) if t not in excluded_t]
    keep = sorted(set(pool) | set(KEEP_TEXT_IDXS))
    if args.max_samples:
        extra = [t for t in pool if t not in KEEP_TEXT_IDXS][: args.max_samples]
        keep = sorted(set(KEEP_TEXT_IDXS) | set(extra))
    meta_doi = meta["doi"].to_numpy()

    def cands(t):
        """Finite maps of the paper's DOI (excluded maps only for an excluded figure example)."""
        c = meta_doi == df_text.loc[t, "doi"]
        return np.flatnonzero(c if t in excluded_t else c & img_ok)

    cand_imgs = sorted({int(i) for t in keep for i in cands(t)})
    with timings.time("fig3_embed", n_texts=len(keep), n_images=len(cand_imgs)):
        e, text_c = text_shared([texts[t] for t in keep], models["specter"], models["text_head"], device)
        im_pred = t2b_maps(models, e, device, args.t2b_model)
        gen_img_c = brain_shared(im_pred, models["ae"], models["image_head"], device)
        true_img_c_sub = brain_shared(images[cand_imgs], models["ae"], models["image_head"], device)
    pos_img = {j: k for k, j in enumerate(cand_imgs)}
    rows = []
    for k, t in enumerate(keep):
        cand = cands(t)
        if not len(cand):
            continue
        sims = (text_c[k] @ true_img_c_sub[[pos_img[int(c)] for c in cand]].T).numpy()
        j = int(cand[int(np.argmax(sims))])
        tc = true_img_c_sub[pos_img[j]]
        rows.append({"text_idx": t, "image_idx": j, "doi": str(df_text.loc[t, "doi"]), "title": str(df_text.loc[t, "title"]),
                     "abstract": str(df_text.loc[t, "abstract"]), "image_cos_cross": float(text_c[k] @ gen_img_c[k]),
                     "image_cos_same": float(gen_img_c[k] @ tc), "true_pair_cos": float(text_c[k] @ tc),
                     "t2b_voxel_pearson": float(pearson_rows(im_pred[k:k + 1], images[j:j + 1])[0]), "excluded_leakage": t in excluded_t,
                     "_k": k})
    d = pd.DataFrame(rows)
    # brain-to-text over the whole set (figure recipe)
    recipe = dict(basis="all", num_beams=3, projection_temp=0.035, repetition_penalty=1.5, seed=0, max_new_tokens=256)
    if args.smoke:
        recipe.update(num_beams=1, max_new_tokens=args.max_new_tokens)
    gen_ids = d["text_idx"].astype(str).tolist()
    if args.smoke:  # the examples keep their sidecar texts
        gen_ids = [i for i in gen_ids if int(i) not in KEEP_TEXT_IDXS]
    by_id = d.set_index(d["text_idx"].astype(str))

    def _gen(ids):
        X = images[by_id.loc[ids, "image_idx"].to_numpy()].copy()
        X[X < 0.25] = 0.0
        return batched_generate(nvlm_fn(), X, args.gen_batch, **recipe)

    with timings.time("fig3_generate", n=len(gen_ids), **{k: v for k, v in recipe.items() if k != "basis"}):
        gen = cached_generation(out / "generated_neurovault.csv", gen_ids, recipe, _gen, force=args.force)
    d["generated_text"] = None
    d.loc[d["text_idx"].astype(str).isin(gen_ids), "generated_text"] = gen
    side = {r["text_idx"]: r for r in sidecar["rows"]}
    d["is_example"] = d["text_idx"].isin(list(side))
    d["sidecar_generated_text"] = d["text_idx"].map(lambda t: side.get(t, {}).get("generated_text"))
    for src in ("generated_text", "sidecar_generated_text"):
        has = d[src].notna()
        if has.any():
            ge, gt = text_shared(d.loc[has, src].tolist(), models["specter"], models["text_head"], device)
            tc = brain_shared(images[d.loc[has, "image_idx"].to_numpy()], models["ae"], models["image_head"], device)
            ks = d.loc[has, "_k"].to_numpy()
            suffix = "" if src == "generated_text" else "_sidecar_text"
            d.loc[has, f"text_cos_cross{suffix}"] = (gt * tc).sum(dim=1).numpy()
            d.loc[has, f"text_cos_same{suffix}"] = (gt * text_c[ks]).sum(dim=1).numpy()
            d.loc[has, f"text_cos_specter{suffix}"] = (ge * e[ks]).sum(dim=1).numpy()
    maps = {"true": images[d["image_idx"].to_numpy()], "generated": im_pred[d["_k"].to_numpy()]}
    return d.drop(columns=["_k"]), maps


def network_maps(nets: dict, items: list[tuple[str, str]]) -> np.ndarray:
    """Atlas network maps (array / max, nearest resampling to the mask), flattened."""
    import nibabel as nib
    from nilearn.image import resample_to_img
    from _hf_pins import load_masker

    masker = load_masker()
    X = []
    for atlas, name in items:
        payload = nets[atlas][name]
        arr = np.asarray(payload["array"], dtype="float32")
        if arr.ndim == 4:
            arr = arr[..., 0]
        img = nib.Nifti1Image(arr / max(float(arr.max()), 1e-8), payload["affine"])
        res = resample_to_img(img, masker.mask_img, interpolation="nearest", force_resample=True, copy_header=True)
        X.append(np.asarray(masker.transform(res), dtype=np.float32).reshape(-1))
    return np.stack(X)


def fig2(sidecar: dict, models: dict, nvlm_fn, *, device, args, out: Path, timings: Timings):
    """Fig. 2: every non-ICA atlas network."""
    from _hf_pins import load_dataset

    ex = [(r["atlas"], r["map"]) for r in sidecar["rows"]]
    with timings.time("fig2_maps"):
        nets = load_dataset("networks")
        items = [(a, n) for a in nets if a not in ICA_ATLASES for n in nets[a]]
        if args.max_samples:
            items = sorted(set(ex) | set([it for it in items if it not in ex][: args.max_samples]), key=items.index)
        X = network_maps(nets, items)
    lab = load_dataset("network_test_set_labels").drop_duplicates("raw_network_label").set_index("raw_network_label")
    ids = [f"{a}:{n}" for a, n in items]
    true_c = brain_shared(X, models["ae"], models["image_head"], device)
    d = pd.DataFrame({"id": ids, "atlas": [a for a, _ in items], "map": [n for _, n in items]})
    d["is_example"] = [it in ex for it in items]

    def t2b(texts):  # always the text-to-brain head, as Fig. 2
        e, tc = text_shared(texts, models["specter"], models["text_head"], device)
        pred = t2b_maps(models, e, device, "head")
        return pred, tc, brain_shared(pred, models["ae"], models["image_head"], device)

    # text-to-brain: the distribution uses "name [SEP] long definition" of labelled maps, the examples the figure's input text
    lab_text = []
    for _, n in items:
        if n in lab.index and str(lab.loc[n, "network_key"]) != "unknown":
            lab_text.append(f"{lab.loc[n, 'network_name']} [SEP] {lab.loc[n, 'long_definition']}")
        else:
            lab_text.append(None)
    d["label_text"] = lab_text
    has = d["label_text"].notna().to_numpy()
    pred_lab = np.zeros_like(X)
    if has.any():
        pred, tc, gc = t2b(d.loc[has, "label_text"].tolist())
        pred_lab[has] = pred
        d.loc[has, "t2b_cos_cross"] = (tc * gc).sum(dim=1).numpy()
        d.loc[has, "t2b_cos_same"] = (gc * true_c[has]).sum(dim=1).numpy()
        d.loc[has, "t2b_voxel_pearson"] = pearson_rows(pred, X[has])
    ex_rows = {(r["atlas"], r["map"]): r for r in sidecar["rows"]}
    ex_idx = [i for i, it in enumerate(items) if it in ex_rows]
    pred_fig = np.zeros_like(X)
    if ex_idx:
        ftexts = [f"{ex_rows[items[i]]['title']} [SEP] {ex_rows[items[i]]['input_text']}" for i in ex_idx]
        pred, tc, gc = t2b(ftexts)
        pred_fig[ex_idx] = pred
        d.loc[ex_idx, "figure_input_text"] = ftexts
        d.loc[ex_idx, "t2b_cos_cross_figure_text"] = (tc * gc).sum(dim=1).numpy()
        d.loc[ex_idx, "t2b_cos_same_figure_text"] = (gc * true_c[ex_idx]).sum(dim=1).numpy()
        d.loc[ex_idx, "t2b_voxel_pearson_figure_text"] = pearson_rows(pred, X[ex_idx])

    # brain-to-text, one generation per basis
    bases = ("network",) if args.smoke else BASES
    gen_ids = [i for i, it in zip(ids, items) if not (args.smoke and it in ex_rows)]
    pos = {i: k for k, i in enumerate(ids)}
    for bi, basis in enumerate(BASES):
        if basis in bases:
            recipe = dict(basis=basis, prefix_text=f"[{basis.upper()}]", num_beams=5, max_new_tokens=256, do_sample=False, seed=bi, projection_temp=0.05)
            if args.smoke:
                recipe.update(num_beams=1, max_new_tokens=args.max_new_tokens)
            with timings.time(f"fig2_generate_{basis}", n=len(gen_ids)):
                gen = cached_generation(out / "generated_networks.csv", gen_ids, recipe,
                                        lambda todo: batched_generate(nvlm_fn(), X[[pos[i] for i in todo]], args.gen_batch, **recipe), force=args.force)
            d[f"generated_{basis}"] = None
            d.loc[d["id"].isin(gen_ids), f"generated_{basis}"] = gen
        d[f"sidecar_generated_{basis}"] = [ex_rows.get(it, {}).get("generated", {}).get(basis, {}).get("raw") for it in items]
        for src, suffix in ((f"generated_{basis}", ""), (f"sidecar_generated_{basis}", "_sidecar_text")):
            if src not in d:
                continue
            m = d[src].notna().to_numpy()
            if m.any():
                _, gt = text_shared(d.loc[m, src].tolist(), models["specter"], models["text_head"], device)
                d.loc[m, f"b2t_cos_{basis}{suffix}"] = (gt * true_c[m]).sum(dim=1).numpy()
    maps = {"true": X, "generated_label_text": pred_lab, "generated_figure_text": pred_fig}
    return d, maps


def apply_thresh(x: np.ndarray) -> np.ndarray:
    """|x| / 99.9th percentile, values below the 95th percentile zeroed, clipped at 1 (figS4_clip_ica.py)."""
    x = np.abs(x.copy())
    x = x / np.percentile(x, 99.9, axis=1)[:, None]
    x[x < np.percentile(x, 95.0, axis=1)[:, None]] = 0
    x[x > 1.0] = 1.0
    return x.astype("float32")


def figS4(sidecar: dict, models: dict, nvlm_fn, *, device, args, out: Path, timings: Timings):
    """Fig. S4: the 40 UKB + HCP ICA components (12 mm smoothing), top-1 label similarity and generated text."""
    import nibabel as nib
    from nilearn.image import resample_to_img, smooth_img
    from _hf_pins import load_dataset, load_masker

    masker = load_masker()
    nets = load_dataset("networks")
    label_names = list(NETS)
    _, lab_c = text_shared([f"{k} [SEP] {v}" for k, v in NETS.items()], models["specter"], models["text_head"], device)
    frames, thr_all = [], []
    with timings.time("figS4_components"):
        for cohort, _ in COHORTS:
            names = list(nets[cohort])
            raw = []
            for n in names:
                img = nib.Nifti1Image(np.asarray(nets[cohort][n]["array"]), affine=nets[cohort][n]["affine"])
                res = smooth_img(resample_to_img(img, masker.mask_img, force_resample=True, copy_header=True), fwhm=12)
                raw.append(np.asarray(masker.transform(res), dtype=np.float32).reshape(-1))
            thr = apply_thresh(np.stack(raw))
            emb = brain_shared(thr, models["ae"], models["image_head"], device)
            cos = (emb @ lab_c.T).numpy()
            frames.append(pd.DataFrame({"id": [f"{cohort}:{n}" for n in names], "cohort": cohort, "component": names, "index": range(len(names)),
                                        "top1_label": [label_names[i] for i in cos.argmax(1)], "top1_cos": cos.max(1),
                                        "emb": list(emb.numpy())}))
            thr_all.append(thr)
    d = pd.concat(frames, ignore_index=True)
    X = np.concatenate(thr_all)
    ex = {(c, r["component"]): r for c, rows in sidecar["cohorts"].items() for r in rows}
    keys = list(zip(d["cohort"], d["component"]))
    d["is_example"] = [k in ex for k in keys]
    d["sidecar_top1_cos"] = [ex.get(k, {}).get("top_sims", [np.nan])[0] for k in keys]
    d["sidecar_generated"] = [ex.get(k, {}).get("generated") for k in keys]
    d["reference_network"] = [ex.get(k, {}).get("ref") for k in keys]
    recipe = dict(seed=123, basis="network", canonical_basis="all", do_sample=False, num_beams=10, max_new_tokens=128, projection_temp=0.05,
                  repetition_penalty=1.1)
    if args.smoke:
        recipe.update(num_beams=1, max_new_tokens=args.max_new_tokens)
    ids = d["id"].tolist()
    gen_ids = [i for i, isx in zip(ids, d["is_example"]) if not args.smoke or not isx]
    if args.max_samples:
        gen_ids = gen_ids[: args.max_samples]
    pos = {i: k for k, i in enumerate(ids)}
    with timings.time("figS4_generate", n=len(gen_ids)):
        gen = cached_generation(out / "generated_ica.csv", gen_ids, recipe,
                                lambda todo: batched_generate(nvlm_fn(), X[[pos[i] for i in todo]], args.gen_batch, **recipe), force=args.force)
    d["generated"] = None
    d.loc[d["id"].isin(gen_ids), "generated"] = gen
    emb = np.stack(d["emb"].to_numpy())
    for src, suffix in (("generated", ""), ("sidecar_generated", "_sidecar_text")):
        m = d[src].notna().to_numpy()
        if m.any():
            _, gt = text_shared(d.loc[m, src].tolist(), models["specter"], models["text_head"], device)
            d.loc[m, f"b2t_cos{suffix}"] = (gt.numpy() * emb[m]).sum(axis=1)
    return d.drop(columns=["emb"]), {"true": X}


def pubmed_cases(models: dict, *, device, args, timings: Timings):
    """Every PubMed test map (exclusions applied). Brain-to-text: contrastive cosine of the validator's stage-2 caption with
    the true text (title [SEP] summary, as Fig. 4b). Text-to-brain: voxel Pearson r of the map generated from
    title [SEP] abstract with the true map."""
    import torch.nn.functional as F

    pmids, excl = heldout_pmids("test", exclusions=True, log=log.info)
    cache = VALIDATION_FINAL / "df_gen_canonical.parquet"
    gen = pd.read_parquet(cache)
    if args.max_samples:
        pmids = pmids[: args.max_samples]
    gen = gen.set_index(gen["pmid"].astype(str)).reindex([str(p) for p in pmids])
    if gen["text"].isna().any():
        raise RuntimeError(f"{cache} does not cover the evaluated PubMed test-set papers (run scripts/validate_models.py --run <model dir> first)")
    rows = load_pubmed_text_rows(pmids)
    truth = [f"{n} [SEP] {s_}" for n, s_ in zip(rows["name"], rows["summary"])]
    t2b_input = [f"{n} [SEP] {d}" for n, d in zip(rows["name"], rows["description"])]
    X = load_pubmed_maps(pmids)
    with timings.time("pubmed_embed", n=len(pmids)):
        _, z_cap = text_shared(gen["text"].astype(str).tolist(), models["specter"], models["text_head"], device)
        _, z_truth = text_shared(truth, models["specter"], models["text_head"], device)
        e_in, _ = text_shared(t2b_input, models["specter"], models["text_head"], device)
        z_map = brain_shared(X.numpy(), models["ae"], models["image_head"], device)
        pred = t2b_maps(models, F.normalize(e_in, dim=1), device, args.t2b_model, batch=256)
    d = pd.DataFrame({"pmid": pmids.astype(np.int64), "title": rows["name"], "summary": rows["summary"], "abstract": rows["description"],
                      "caption": gen["text"].astype(str).to_numpy(),
                      "b2t_text_cos": (z_cap * z_truth).sum(dim=1).numpy(), "b2t_caption_map_cos": (z_cap * z_map).sum(dim=1).numpy(),
                      "t2b_pearson": pearson_rows(np.asarray(pred, dtype=np.float64), np.asarray(X.numpy(), dtype=np.float64))})
    d["pct_b2t_text_cos"] = [percentile_of(v, d["b2t_text_cos"]) for v in d["b2t_text_cos"]]
    d["pct_t2b_pearson"] = [percentile_of(v, d["t2b_pearson"]) for v in d["t2b_pearson"]]
    return d, {"true": X.numpy(), "generated": pred}, excl


def neurovault_best_case(models: dict, nvlm_fn, *, device, args, out: Path, timings: Timings):
    """NeuroVault papers scored by their BEST map, since only some of a paper's maps show its main contrast.
    Text-to-brain: cosine of the map generated from title [SEP] abstract with each of the paper's maps. Brain-to-text: the
    Fig. 3 recipe on every map, cosine of each generated text with its map. Best map = max."""
    df_text, images, meta = load_neurovault()
    pub_ok = neurovault_keep_mask(df_text["doi"].to_numpy(), enabled=True, label="NeuroVault publications (best case)", log=log.info)
    img_ok = neurovault_keep_mask(meta["doi"].to_numpy(), image_ids=meta["id"].to_numpy(), collection_ids=meta["collection_id"].to_numpy(),
                                  enabled=True, label="NeuroVault images (best case)", log=log.info)
    finite = ~images.isnan().any(dim=1).numpy()
    images = images.numpy()
    pubs = [t for t in range(len(df_text)) if pub_ok[t]]
    meta_doi = meta["doi"].to_numpy()
    cand = {t: np.flatnonzero((meta_doi == df_text.loc[t, "doi"]) & finite & img_ok) for t in pubs}
    pubs = [t for t in pubs if len(cand[t])]
    if args.max_samples:
        pubs = pubs[: args.max_samples]
    img_idx = np.array(sorted({int(i) for t in pubs for i in cand[t]}))
    texts = [f"{df_text.loc[t, 'title']} [SEP] {df_text.loc[t, 'abstract']}" for t in pubs]
    with timings.time("nv_best_embed", n_pubs=len(pubs), n_images=len(img_idx)):
        e, text_c = text_shared(texts, models["specter"], models["text_head"], device)
        gen_map = t2b_maps(models, e, device, args.t2b_model)
        z_gen_map = brain_shared(gen_map, models["ae"], models["image_head"], device)
        z_img = brain_shared(images[img_idx], models["ae"], models["image_head"], device)
    pos = {int(j): k for k, j in enumerate(img_idx)}
    recipe = dict(basis="all", num_beams=3, projection_temp=0.035, repetition_penalty=1.5, seed=0, max_new_tokens=256)
    if args.smoke:
        recipe.update(num_beams=1, max_new_tokens=args.max_new_tokens)
    ids = [str(int(meta.loc[j, "id"])) for j in img_idx]

    def _gen(todo):
        jj = [img_idx[ids.index(i)] for i in todo]
        Xg = images[jj].copy()
        Xg[Xg < 0.25] = 0.0
        return batched_generate(nvlm_fn(), Xg, args.gen_batch, **recipe)

    with timings.time("nv_best_generate", n=len(ids), **{k: v for k, v in recipe.items() if k != "basis"}):
        gen_text = cached_generation(out / "generated_neurovault_allimages.csv", ids, recipe, _gen, force=args.force)
    _, z_txt = text_shared(gen_text, models["specter"], models["text_head"], device)
    b2t_img = (z_txt * z_img).sum(dim=1).numpy()
    rows, per_image = [], []
    for k, t in enumerate(pubs):
        kk = [pos[int(j)] for j in cand[t]]
        t2b = (z_gen_map[k:k + 1] @ z_img[kk].T).numpy()[0]
        b2t = b2t_img[kk]
        jt, jb = int(cand[t][int(np.argmax(t2b))]), int(cand[t][int(np.argmax(b2t))])
        for j, a, b in zip(cand[t], t2b, b2t):
            per_image.append({"text_idx": t, "image_idx": int(j), "image_id": int(meta.loc[j, "id"]), "t2b_cos": float(a), "b2t_cos": float(b)})
        rows.append({"text_idx": t, "doi": str(df_text.loc[t, "doi"]), "title": str(df_text.loc[t, "title"]), "abstract": str(df_text.loc[t, "abstract"]),
                     "n_images": len(cand[t]), "t2b_best_cos": float(t2b.max()), "t2b_best_image_idx": jt, "t2b_best_image_id": int(meta.loc[jt, "id"]),
                     "t2b_best_image_contrast": str(meta.loc[jt, "contrast_definition"]),
                     "b2t_best_cos": float(b2t.max()), "b2t_best_image_idx": jb, "b2t_best_image_id": int(meta.loc[jb, "id"]),
                     "b2t_best_image_contrast": str(meta.loc[jb, "contrast_definition"]),
                     "b2t_best_text": gen_text[ids.index(str(int(meta.loc[jb, "id"])))], "_k": k})
    d = pd.DataFrame(rows)
    d["pct_t2b_best_cos"] = [percentile_of(v, d["t2b_best_cos"]) for v in d["t2b_best_cos"]]
    d["pct_b2t_best_cos"] = [percentile_of(v, d["b2t_best_cos"]) for v in d["b2t_best_cos"]]
    return d, pd.DataFrame(per_image), {"images": images, "generated": gen_map}


def percentile_table(d3, d2, ds4, sidecar3: dict) -> pd.DataFrame:
    """One row per (example, metric). ``value`` uses the figure's own input / generated text, ``sidecar_value`` is the number
    stored in the figure sidecar (port check), ``value_alt`` the item scored the distribution's way."""
    rows = []
    s3 = {r["text_idx"]: r for r in sidecar3["rows"]}

    def add(fig, example, metric, value, dist, *, sidecar=np.nan, alt=np.nan, alt_kind="", note="", excluded=False, rank_value=None):
        if excluded:
            note = (note + "; " if note else "") + "EXCLUDED item (leakage audit): ranked against the cleaned pool, not part of it"
        dist = np.asarray(dist, dtype=float)
        dist = dist[np.isfinite(dist)]
        sidecar = float(sidecar) if sidecar is not None else np.nan
        pct = percentile_of(value if rank_value is None else rank_value, dist)
        rows.append({"figure": fig, "example": example, "metric": metric, "value": value, "percentile": pct,
                     "n_distribution": len(dist), "dist_median": float(np.median(dist)) if len(dist) else np.nan,
                     "sidecar_value": sidecar, "abs_diff": abs(value - sidecar) if np.isfinite(value) and np.isfinite(sidecar) else np.nan,
                     "value_alt": alt, "value_alt_kind": alt_kind, "note": note, "excluded_leakage": bool(excluded)})

    side = d3[d3["is_example"]].set_index("text_idx")
    pool3 = d3[~d3["excluded_leakage"].astype(bool)]  # excluded examples are ranked, not counted
    # Fig. 3's own pool (sidecar; texts generated in batches of six like its rows): its cosines rank the rows, so the
    # percentiles are the figure's; metrics the sidecar does not store are ranked within this script's pool
    fig_pool = sidecar3.get("pool", {}).get("scores", {})
    in_fig = "percentile = sidecar value within the Fig. 3 pool (fig3_newgen_text.json)"
    for rank, t in enumerate(KEEP_TEXT_IDXS):
        if t not in side.index:
            continue
        r = side.loc[t]
        sv = s3.get(t, {})
        name = f"row {rank + 1} ({FIG3_GROUPS[rank]}): {r['title'][:50]}"
        for metric in ("image_cos_cross", "image_cos_same", "true_pair_cos", "t2b_voxel_pearson"):
            ours = metric in fig_pool and metric in sv
            add("Fig 3", name, metric, r[metric], fig_pool[metric] if ours else pool3[metric], sidecar=sv.get(metric, np.nan),
                excluded=bool(r["excluded_leakage"]), rank_value=sv[metric] if ours else None, note=in_fig if ours else "")
        for metric in ("text_cos_cross", "text_cos_same", "text_cos_specter"):
            ours = metric in fig_pool and metric in sv
            add("Fig 3", name, metric, r.get(f"{metric}_sidecar_text", np.nan),
                fig_pool[metric] if ours else (pool3[metric] if metric in pool3 else []), sidecar=sv.get(metric, np.nan),
                alt=r.get(metric, np.nan), alt_kind="regenerated text (same recipe)", rank_value=sv[metric] if ours else None,
                note="value = figure's generated text" + ("; " + in_fig if ours else ""), excluded=bool(r["excluded_leakage"]))
    for _, r in d2[d2["is_example"]].iterrows():
        name = f"{r['atlas']}:{r['map']}"
        for metric in ("t2b_cos_cross", "t2b_cos_same", "t2b_voxel_pearson"):
            if f"{metric}_figure_text" in r:
                add("Fig 2", name, metric, r[f"{metric}_figure_text"], d2[metric] if metric in d2 else [], alt=r.get(metric, np.nan),
                    alt_kind="label-definition input text", note="value = figure input text; distribution = label-definition texts")
        for b in BASES:
            v = r.get(f"b2t_cos_{b}_sidecar_text", np.nan)
            if np.isfinite(v):
                add("Fig 2", name, f"b2t_cos_{b}", v, d2[f"b2t_cos_{b}"] if f"b2t_cos_{b}" in d2 else [], alt=r.get(f"b2t_cos_{b}", np.nan),
                    alt_kind="regenerated text (same recipe)", note="value = figure's generated text")
    for _, r in ds4[ds4["is_example"]].iterrows():
        name = f"{r['cohort']}:{r['component']} ({r['reference_network']})"
        add("Fig S4", name, "top1_label_cos", r["top1_cos"], ds4["top1_cos"], sidecar=r["sidecar_top1_cos"])
        add("Fig S4", name, "b2t_cos", r.get("b2t_cos_sidecar_text", np.nan), ds4["b2t_cos"] if "b2t_cos" in ds4 else [],
            alt=r.get("b2t_cos", np.nan), alt_kind="regenerated text (same recipe)", note="value = figure's generated text")
    return pd.DataFrame(rows)


def captions(pt) -> str:
    """First version of captions.tex (make_figure.py rewrites it in panel order)."""
    def fmt(p):
        return "--" if not np.isfinite(p) else f"{int(round(p))}"

    lines = ["% Auto-generated by scripts/figures/review/core/a7_failure_cases_percentiles/run_analysis.py -- percentile ranks of the",
             "% qualitative examples within the full evaluation distribution of the same cosine similarity.", ""]
    f3 = pt[pt["figure"] == "Fig 3"]
    if len(f3):
        t2b = f3[f3.metric == "image_cos_cross"]
        b2t = f3[f3.metric == "text_cos_cross"]
        n1 = int(t2b["n_distribution"].max()) if len(t2b) else 0
        lines.append("% Fig. 3 caption addition")
        lines.append(f"Rows (top to bottom) lie at the {', '.join(fmt(p) for p in t2b['percentile'])} percentiles of the text-to-brain cosine "
                     f"distribution and at the {', '.join(fmt(p) for p in b2t['percentile'])} percentiles of the brain-to-text cosine distribution "
                     f"across all {n1} NeuroVault papers.")
        fl = f3[f3["excluded_leakage"].astype(bool)]["example"].unique()
        if len(fl):
            lines.append(f"% NOTE: example(s) {'; '.join(fl)} are leakage-audit exclusions (NeuroVault paper also a PubMed training paper); "
                         "ranked against the cleaned distribution, not part of it.")
        lines.append("")
    f2 = pt[pt["figure"] == "Fig 2"]
    if len(f2):
        t = f2[f2.metric == "t2b_cos_same"]
        b = f2[f2.metric == "b2t_cos_network"]
        lines.append("% Fig. 2 caption addition")
        lines.append(f"Relative to all {int(f2['n_distribution'].max())} atlas networks, the generated maps rank at the "
                     f"{', '.join(fmt(p) for p in t['percentile'])} percentiles (cosine to the reference network) and the predicted network "
                     f"labels at the {', '.join(fmt(p) for p in b['percentile'])} percentiles (cosine of generated text to map).")
        lines.append("")
    fs = pt[pt["figure"] == "Fig S4"]
    if len(fs):
        t = fs[fs.metric == "top1_label_cos"]
        b = fs[fs.metric == "b2t_cos"]
        lines.append("% Fig. S4 caption addition")
        lines.append(f"Shown components rank at the {', '.join(fmt(p) for p in t['percentile'])} percentiles of top-1 label similarity and the "
                     f"{', '.join(fmt(p) for p in b['percentile'])} percentiles of generated-text similarity among all {int(t['n_distribution'].max())} "
                     f"UKB and HCP ICA components.")
    return "\n".join(lines) + "\n"


def build_panels(dpm, maps_pm, d2, maps2, dnv, maps_nv, *, n_cases: int, n_random: int, seed: int, t2b_label: str) -> tuple[dict, dict]:
    """Cases of the failure-case pages (one list per set) and their maps (float16, keyed ``<set>_<rank>_<kind>``)."""
    sets, arrays = {}, {}

    def add_map(key, vec):
        arrays[key] = np.asarray(vec, dtype=np.float16)
        return key

    def caption_display(c: str) -> str:
        parts = str(c).split(" [SEP] ")
        if len(parts) == 3:
            return "\n".join(f"[{b.upper()}] {t.strip()}" for b, t in zip(BASES, parts))
        return str(c)

    n_pm = len(dpm)

    def pm_case(name, i, rank, kind):
        r = dpm.iloc[i]
        maps = [(add_map(f"{name}_{rank}_true", maps_pm["true"][i]), "true map")]
        if kind in ("t2b", "both"):
            maps.append((add_map(f"{name}_{rank}_gen", maps_pm["generated"][i]), f"generated map ({t2b_label})"))
        metrics = []
        if kind in ("b2t", "both"):
            metrics.append(("brain-to-text: caption vs true text, contrastive cosine", float(r["b2t_text_cos"]), float(r["pct_b2t_text_cos"]), n_pm))
        if kind in ("t2b", "both"):
            metrics.append(("text-to-brain: voxel Pearson r, generated vs true map", float(r["t2b_pearson"]), float(r["pct_t2b_pearson"]), n_pm))
        true_label = "Reference (title + summary)" if kind != "t2b" else "Input text (title + abstract)"
        true_text = f"{r['title']}\n{r['summary']}" if kind != "t2b" else f"{r['title']}\n{r['abstract']}"
        return {"rank": rank, "id": f"PMID {int(r['pmid'])}", "title": str(r["title"]), "true_label": true_label, "true_text": true_text,
                "generated_label": "Generated caption (stage 2, complete)" if kind != "t2b" else None,
                "generated_text": caption_display(r["caption"]) if kind != "t2b" else None, "metrics": metrics, "maps": maps}

    order = np.argsort(dpm["b2t_text_cos"].to_numpy())
    sets["pubmed_b2t_worst"] = [pm_case("pubmed_b2t_worst", int(i), k + 1, "b2t") for k, i in enumerate(order[:n_cases])]
    order = np.argsort(dpm["t2b_pearson"].to_numpy())
    sets["pubmed_t2b_worst"] = [pm_case("pubmed_t2b_worst", int(i), k + 1, "t2b") for k, i in enumerate(order[:n_cases])]
    rnd = np.random.default_rng(seed).choice(n_pm, size=min(n_random, n_pm), replace=False)
    sets["pubmed_random"] = [pm_case("pubmed_random", int(i), k + 1, "both") for k, i in enumerate(sorted(rnd))]

    v = d2["b2t_cos_network"].to_numpy(dtype=float)
    ok = np.flatnonzero(np.isfinite(v))
    cases = []
    for k, i in enumerate(ok[np.argsort(v[ok])][:n_cases]):
        r = d2.iloc[int(i)]
        gen = str(r.get("generated_network") if isinstance(r.get("generated_network"), str) else r.get("sidecar_generated_network"))
        gen = re.sub(r"^\s*\[[A-Z]+\]\s*", "", gen).strip()
        ref = r.get("label_text")
        cases.append({"rank": k + 1, "id": f"{r['atlas']}: {r['map']}", "title": f"{r['atlas']}: {r['map']}",
                      "true_label": "Reference label (network_test_set_labels)",
                      "true_text": ref.replace(" [SEP] ", "\n", 1) if isinstance(ref, str) else "(no reference description for this map)",
                      "generated_label": "Generated [NETWORK] title + text (complete)", "generated_text": gen,
                      "metrics": [("brain-to-text: [NETWORK] text vs map, contrastive cosine", float(v[i]), percentile_of(v[i], v[ok]), int(len(ok)))],
                      "maps": [(add_map(f"networks_bottom_{k + 1}_true", maps2["true"][int(i)]), "network map")]})
    sets["networks_bottom"] = cases

    n_nv = len(dnv)
    for name, col, kind in (("neurovault_b2t_best", "b2t_best_cos", "b2t"), ("neurovault_t2b_best", "t2b_best_cos", "t2b")):
        cases = []
        for k, i in enumerate(np.argsort(dnv[col].to_numpy())[:n_cases]):
            r = dnv.iloc[int(i)]
            img = int(r[f"{kind}_best_image_idx"])
            maps = []
            if kind == "t2b":
                maps.append((add_map(f"{name}_{k + 1}_gen", maps_nv["generated"][int(r["_k"])]), f"generated map ({t2b_label})"))
            contrast = str(r[f"{kind}_best_image_contrast"]).strip()
            tag = f"id {int(r[f'{kind}_best_image_id'])}" + ("" if contrast.lower() in ("", "nan", "none") else f", {contrast}")
            n_img = int(r["n_images"])
            maps.append((add_map(f"{name}_{k + 1}_true", maps_nv["images"][img]),
                         f"true map: the paper's only map ({tag})" if n_img == 1 else f"true map: best of the paper's {n_img} maps ({tag})"))
            label = ("brain-to-text: best map's generated text vs that map, contrastive cosine" if kind == "b2t"
                     else "text-to-brain: generated map vs best true map, contrastive cosine")
            cases.append({"rank": k + 1, "id": str(r["doi"]), "title": str(r["title"]), "true_label": "Paper (title + abstract)",
                          "true_text": f"{r['title']}\n{r['abstract']}",
                          "generated_label": "Generated text from the best map (complete)" if kind == "b2t" else None,
                          "generated_text": str(r["b2t_best_text"]) if kind == "b2t" else None,
                          "metrics": [(label, float(r[col]), float(r[f"pct_{col}"]), n_nv)], "maps": maps})
        sets[name] = cases
    return sets, arrays


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=HERE / "outputs")
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Run directory with the models (default: runs/final).")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--sidecar-dir", type=Path, default=SIDECAR_DIR_DEFAULT, help="Directory with the paper figures' *_text.json.")
    p.add_argument("--max-samples", type=int, default=None, help="Extra (non-example) items per set (default all; 2 in --smoke).")
    p.add_argument("--gen-batch", type=int, default=32, help="Generation batch (<= 128).")
    p.add_argument("--max-new-tokens", type=int, default=12, help="New tokens in --smoke only.")
    p.add_argument("--n-cases", type=int, default=6, help="Cases per failure set.")
    p.add_argument("--n-random", type=int, default=6, help="Random PubMed cases.")
    p.add_argument("--t2b-model", choices=["head", "adapter"], default="head",
                   help="Text-to-brain model: head (proj_head_text_mse + AE decoder, as Figs. 2-4) or adapter (NeuroAdapter).")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force", action="store_true", help="Regenerate cached texts.")
    p.add_argument("--no-exclusions", action="store_true", help="Keep the leakage-audit NeuroVault papers in the Fig. 3 pool.")
    args = p.parse_args(argv)
    if args.gen_batch > 128:
        raise SystemExit("--gen-batch must be <= 128")
    if args.smoke:
        args.max_samples = args.max_samples or 2
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    timings = Timings(out / "timings.json")
    seed_all(args.seed)
    device = select_device(args.device)
    set_model_dir(args.model_dir)
    t0 = time.perf_counter()
    side = {k: json.loads((args.sidecar_dir / f"{stem}_text.json").read_text()) for k, stem in SIDECAR_STEMS.items()}
    with timings.time("load_models", device=str(device)):
        models = {"ae": component("autoencoder", device), "text_head": component("proj_head_text_infonce", device),
                  "image_head": component("proj_head_image_infonce", device), "mse_head": component("proj_head_text_mse", device),
                  "specter": component("specter", device)}
        if args.t2b_model == "adapter":
            models["adapter"] = component("neuro_adapter", device)
    state: dict = {}

    def nvlm_fn():
        """NeuroVLM client with the run's autoencoder, heads and stage-2 projector (loaded on first use)."""
        if "nvlm" not in state:
            nv = make_nvlm(device, ae=models["ae"], text_head=models["text_head"], image_head=models["image_head"], mse_head=models["mse_head"],
                           specter=models["specter"])
            nv._neuro_qformer, nv._neuro_qformer_variant = load_projector("canonical", device), "canonical"
            state["nvlm"] = nv
        return state["nvlm"]

    exclusions: dict = {"enabled": not args.no_exclusions}
    log.info("model dir %s | sidecars %s | eval exclusions %s | text-to-brain model %s", model_dir(), args.sidecar_dir,
             "ON" if exclusions["enabled"] else "OFF", args.t2b_model)
    d3, _ = fig3(side["newgen"], models, nvlm_fn, device=device, args=args, out=out, timings=timings, excl_record=exclusions)
    in_pool3 = ~d3["excluded_leakage"].astype(bool).to_numpy()
    d2, maps2 = fig2(side["networks_examples"], models, nvlm_fn, device=device, args=args, out=out, timings=timings)
    ds4, _ = figS4(side["clip_ica_hcp_ukb"], models, nvlm_fn, device=device, args=args, out=out, timings=timings)
    dnv, dnv_img, maps_nv = neurovault_best_case(models, nvlm_fn, device=device, args=args, out=out, timings=timings)
    state.pop("nvlm", None)
    log.info("released CUDA cache after generation (%s)", release_cuda_cache())
    dpm, maps_pm, excl_pm = pubmed_cases(models, device=device, args=args, timings=timings)

    pt = percentile_table(d3, d2, ds4, side["newgen"])
    pt.to_csv(out / "example_percentiles.csv", index=False)
    (out / "captions.tex").write_text(captions(pt))
    long = []
    for fig, d, cols in (("Fig 3", d3[in_pool3], ["image_cos_cross", "image_cos_same", "true_pair_cos", "t2b_voxel_pearson", "text_cos_cross", "text_cos_same", "text_cos_specter"]),
                         ("Fig 2", d2, ["t2b_cos_cross", "t2b_cos_same", "t2b_voxel_pearson", *[f"b2t_cos_{b}" for b in BASES]]),
                         ("Fig S4", ds4, ["top1_cos", "b2t_cos"])):
        idcol = "text_idx" if fig == "Fig 3" else "id"
        for c in cols:
            if c in d:
                for i, v, isx in zip(d[idcol], d[c], d["is_example"]):
                    if pd.notna(v):
                        long.append({"figure": fig, "item": str(i), "metric": c, "value": float(v), "is_example": bool(isx)})
    pd.DataFrame(long).to_parquet(out / "distributions.parquet", index=False)
    d3.to_parquet(out / "fig3_items.parquet", index=False)
    d2.to_parquet(out / "fig2_items.parquet", index=False)
    ds4.to_parquet(out / "figS4_items.parquet", index=False)
    dpm.to_parquet(out / "pubmed_items.parquet", index=False)
    dnv.drop(columns=["_k"]).to_parquet(out / "neurovault_best_case.parquet", index=False)
    dnv_img.to_parquet(out / "neurovault_best_case_images.parquet", index=False)
    sets, arrays = build_panels(dpm, maps_pm, d2, maps2, dnv, maps_nv, n_cases=args.n_cases, n_random=args.n_random, seed=args.seed,
                                t2b_label=T2B_LABEL[args.t2b_model])
    write_json(out / "panels.json", sets)
    np.savez_compressed(out / "panel_maps.npz", **arrays)
    check = pt.dropna(subset=["abs_diff"])
    write_json(out / "metrics.json", {
        "smoke": args.smoke, "device": str(device), "model_dir": str(model_dir()) if model_dir() else None, "model_sources": model_sources(),
        "sidecar_dir": str(args.sidecar_dir), "text_to_brain_model": f"{args.t2b_model} ({T2B_LABEL[args.t2b_model]})",
        "n": {"fig3": len(d3), "fig3_distribution": int(in_pool3.sum()), "fig2": len(d2), "figS4": len(ds4), "pubmed_test": len(dpm),
              "neurovault_best_case_papers": len(dnv), "neurovault_best_case_images": len(dnv_img)},
        "failure_sets": {k: [{"id": c["id"], "metrics": c["metrics"]} for c in v] for k, v in sets.items()},
        "exclusions": {**exclusions, "pubmed": excl_pm, "n_fig3_items_flagged_not_in_distribution": int((~in_pool3).sum()),
                       "source": "scripts/eval_exclusions/{pubmed,neurovault}_exclusions.csv (leakage audit)"},
        "max_abs_diff_vs_figure_sidecar": float(check["abs_diff"].max()) if len(check) else None,
        "median_percentile_by_figure_metric": pt.groupby(["figure", "metric"])["percentile"].median().to_dict(),
        "packages": env_packages(), "wall_seconds_this_run": time.perf_counter() - t0})
    with pd.option_context("display.width", 220, "display.max_colwidth", 60):
        print(pt[["figure", "example", "metric", "value", "sidecar_value", "value_alt", "percentile", "n_distribution"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    for k, v in sets.items():
        print(k, [(c["id"], [(round(m[1], 3), round(m[2], 2)) for m in c["metrics"]]) for c in v])
    return 0


if __name__ == "__main__":
    sys.exit(main())
