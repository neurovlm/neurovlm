#!/usr/bin/env python
"""Fig. 4: text-to-brain (row a) and brain-to-text (row b) metrics on the PubMed test set, matched vs deranged null.

    .venv/bin/python scripts/figures/main/fig4_generative_metrics.py [--rows a|b|ab] [--no-spin]
"""
from __future__ import annotations

import numpy as np
from _shared import (
    BLUE, FIG_FULL_W, MATCH_COLOR, NULL_COLOR, RELEASED, STATUS_BAD, banner, binom_log10_sf, cache_file,
    calibrate_maps, cohen_d, deranged_perm, fig_parser, finalize, finish, format_p, free_gpu, get_logger, hf_cache_pinned, hf_pinned, hf_pinned_dir, load_component,
    load_pt, make_nvlm, map_cal_tag, map_factor,
    match_null_box, note_source, panel_label, pubmed_keep, pubmed_test_mask, recall_axes, run_file, save_pt,
    select_device, sig_text, strip_steering_tags, timed, to_device_batches, welch_log10p,
)

STEM = "fig4_generative_metrics"
log = get_logger(STEM)
BASES = ("network", "region", "function")
DICE_PCTS = (70, 75, 80, 85, 90, 95)
COMPONENTS = ("specter", "autoencoder", "latent_neuro", "proj_head_text_mse", "proj_head_text_infonce",
              "proj_head_image_infonce", "qformer_canonical", "canonical_semantic_banks", "neuroqwen")
_MASKER = None  # per worker, for the spin test


def parse_args():
    p = fig_parser("Fig. 4: text-to-brain and brain-to-text generative metrics (PubMed test).")
    p.add_argument("--rows", default="ab", help="a: text-to-brain, b: brain-to-text, ab: both.")
    p.add_argument("--gen-batch-size", type=int, default=128, help="Brain-to-text generation batch size.")
    p.add_argument("--bert-batch-size", type=int, default=32)
    p.add_argument("--dice-pct", type=float, default=None, help="Fixed Dice percentile (default: the best of 70-95).")
    p.add_argument("--no-spin", action="store_true", help="Skip the spin test.")
    p.add_argument("--spin-n-perm", type=int, default=100)
    p.add_argument("--spin-density", default="3k")
    p.add_argument("--spin-max-samples", type=int, default=None, help="Cap the number of spin-tested maps.")
    return finalize(p.parse_args())


def _notebook_deranged_perm(n: int, rng: np.random.Generator) -> np.ndarray:
    """Derangement as in docs/figures/24 (retry permutations, then roll)."""
    if n < 2:
        return np.arange(n)
    for _ in range(100):
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm
    return np.roll(np.arange(n), 1)


def _zscore_rows(x):
    x = np.asarray(x, dtype="float32")
    x = x - x.mean(axis=1, keepdims=True)
    return x / np.clip(x.std(axis=1, keepdims=True), 1e-6, None)


def _match_null_correlation(pred, true, *, max_voxels=5000, seed=13):
    """Pearson or Spearman (whichever has the larger match - null gap) on max_voxels random voxels."""
    from scipy.stats import rankdata

    n, d = pred.shape
    if max_voxels is not None and max_voxels < d:
        vox = np.random.default_rng(seed).choice(d, size=int(max_voxels), replace=False)
        pred, true = pred[:, vox], true[:, vox]
    null_idx = deranged_perm(n, np.random.default_rng(seed))
    denom = float(pred.shape[1])
    pz, tz = _zscore_rows(pred), _zscore_rows(true)
    pear_m = np.einsum("ij,ij->i", pz, tz) / denom
    pear_n = np.einsum("ij,ij->i", pz, tz[null_idx]) / denom
    prz = _zscore_rows(rankdata(pred, method="average", axis=1).astype("float32"))
    trz = _zscore_rows(rankdata(true, method="average", axis=1).astype("float32"))
    spear_m = np.einsum("ij,ij->i", prz, trz) / denom
    spear_n = np.einsum("ij,ij->i", prz, trz[null_idx]) / denom
    if (np.nanmean(spear_m) - np.nanmean(spear_n)) > (np.nanmean(pear_m) - np.nanmean(pear_n)):
        return "Spearman", spear_m, spear_n
    return "Pearson", pear_m, pear_n


def _dice_curves(pred, true, null_idx, pcts):
    out = {}
    for pct in pcts:
        pb = pred > np.percentile(pred, pct, axis=1, keepdims=True)
        tb = true > np.percentile(true, pct, axis=1, keepdims=True)
        tbn = tb[null_idx]
        match = 2.0 * (pb & tb).sum(1) / np.maximum(pb.sum(1) + tb.sum(1), 1)
        null = 2.0 * (pb & tbn).sum(1) / np.maximum(pb.sum(1) + tbn.sum(1), 1)
        out[float(pct)] = {"match": match.astype("float32"), "null": null.astype("float32")}
    return out


def _prefetch_spin_atlases(density: str) -> None:
    """Download the neuromaps files the spin workers read (fsaverage spheres, regfusion), once in the parent: parallel
    first downloads into an empty neuromaps-data race on the same .part file."""
    from neuromaps.datasets import fetch_fsaverage, fetch_regfusion

    fetch_fsaverage(density=density)
    fetch_regfusion("fsaverage")


def _spin_one(pred_vec, true_vec, *, pct, n_perm, random_state, density, method):
    global _MASKER
    from _shared import load_masker
    from neurovlm.metrics.text_to_brain import mni152_to_fsaverage_arrays, nct_dice_spin_test_surface

    if _MASKER is None:
        _MASKER = load_masker()
    try:
        pl, pr = mni152_to_fsaverage_arrays(_MASKER.inverse_transform(pred_vec.reshape(1, -1)), density=density, method=method)
        tl, tr = mni152_to_fsaverage_arrays(_MASKER.inverse_transform(true_vec.reshape(1, -1)), density=density, method=method)
        res = nct_dice_spin_test_surface(pl, pr, tl, tr, pct=pct, n_perm=n_perm, random_state=random_state, density=density)
        return {"dice_surface": float(res.dice_pct), "p": float(res.spin_p_value), "method": str(res.spin_method), "error": None}
    except Exception as exc:  # reported in the JSON
        return {"dice_surface": np.nan, "p": np.nan, "method": None, "error": f"{type(exc).__name__}: {exc}"}


def _t2b_records_for_run(args, device):
    """build_pubmed_t2b_eval for a --model-dir run: the run's test split and the run autoencoder's latents
    (latent_neuro.pt, else re-encoded), so the target sigmoid(decoder(latent)) is a reconstruction of the true map."""
    import torch
    from _shared import load_dataset, load_latent

    df = load_dataset("pubmed_text")
    df_test = df[pubmed_test_mask(args, df, log)].reset_index(drop=True)
    lat_rel, lp = load_latent("pubmed_images")  # HF row order = the released recipe's order
    lp = np.asarray(lp)
    m = np.isin(lp, df_test["pmid"].values)
    pm = lp[m]
    ae_file, lat_file = run_file(args.model_dir, "autoencoder"), run_file(args.model_dir, "latent_neuro")
    latents = None
    if ae_file is None:
        latents = torch.as_tensor(lat_rel[m]).float()
        note_source("latent_neuro", RELEASED["latent_neuro"])
    elif lat_file is not None:
        payload = torch.load(lat_file, map_location="cpu", weights_only=False)
        pos = {int(p): i for i, p in enumerate(np.asarray(payload["pmid"]))}
        if all(int(p) in pos for p in pm):
            latents = payload["latent"][torch.as_tensor([pos[int(p)] for p in pm])].float()
            note_source("latent_neuro", lat_file)
    if latents is None:
        images, pi = load_dataset("pubmed_images")
        rows = np.flatnonzero(np.isin(np.asarray(pi), pm))
        ae = load_component("autoencoder", args.model_dir, log).to(device).eval()
        latents = to_device_batches(ae.encoder, images[torch.as_tensor(rows)], device, batch_size=512)
        note_source("latent_neuro", f"re-encoded with {ae_file}")
        del images, ae
    rows = df_test.set_index("pmid")
    return [{"pmid": p, "latent": latents[i], "short_gt": str(rows.loc[p, "name"]), "long_gt": str(rows.loc[p, "description"])}
            for i, p in enumerate(pm) if p in rows.index]


def compute_t2b(args, device):
    """Row a: "<title> [SEP] <abstract>" -> head map vs target sigmoid(decoder(true latent)); retrieval, contrastive,
    Dice, correlation and an NCT-style spin test (fsaverage, cortical targets only)."""
    cache = cache_file(args, "fig4_t2b" + map_cal_tag(args), excl=True)
    if cache.exists() and not args.force:
        log.info("cache hit %s", cache)
        return load_pt(cache)
    import torch
    from _shared import load_masker
    from neurovlm.evaluation.notebook_utils import build_pubmed_t2b_eval
    from neurovlm.metrics.text_to_brain import (
        build_surface_eligibility_mask, decode_latents_to_brain, encode_text_batched,
        generated_image_text_retrieval_curve, surface_metric_eligibility,
    )

    with timed("a) data + generation", log, args.cache_dir, stage=STEM):
        if args.model_dir is None:
            hf_cache_pinned()  # the library builder loads pubmed_text / latents through the HF cache
            records = build_pubmed_t2b_eval(max_samples=None, use_summaries=False)  # HF test flag + released latents
            note_source("latent_neuro", RELEASED["latent_neuro"])
        else:
            records = _t2b_records_for_run(args, device)
        ok = pubmed_keep(args, [r["pmid"] for r in records], key="PubMed test (text-to-brain)", log=log)
        records = [r for r, k in zip(records, ok) if k]
        n_total = len(records)
        if args.max_samples and args.max_samples < n_total:
            sel = np.sort(np.random.default_rng(args.seed).choice(n_total, args.max_samples, replace=False))
            records = [records[i] for i in sel]
        n = len(records)
        texts = [f"{r['short_gt']} [SEP] {r['long_gt']}" for r in records]
        pmids = [str(r["pmid"]) for r in records]
        latents = torch.stack([torch.as_tensor(np.asarray(r["latent"]), dtype=torch.float32) for r in records])
        nvlm = make_nvlm(device, args.model_dir, log, generator=False)
        preds = []
        for s in range(0, n, 256):
            res = nvlm.text(texts[s:s + 256]).to_brain(head="mse", adapter=False)
            preds.append(res.generated_flatmaps.detach().float().cpu())
        pred = calibrate_maps(torch.cat(preds), args.map_calibration, factor=map_factor(args, "head", device)).numpy().astype("float32")
        true = decode_latents_to_brain(latents, nvlm._autoencoder.decoder, batch_size=256)
        text_latents = encode_text_batched(nvlm, texts, batch_size=32)

    with timed("a) retrieval + contrastive", log, args.cache_dir, stage=STEM):
        auc, curve_df, rank_df = generated_image_text_retrieval_curve(
            nvlm, pred, texts, text_latents=text_latents, image_batch_size=256, text_encode_batch_size=32, seed=13)
        del nvlm
        free_gpu()

    with timed("a) dice + correlation", log, args.cache_dir, stage=STEM):
        null_idx = deranged_perm(n, np.random.default_rng(13))
        dice = _dice_curves(pred, true, null_idx, DICE_PCTS)
        if args.dice_pct is not None:
            best_pct = float(args.dice_pct)
            if best_pct not in dice:
                dice.update(_dice_curves(pred, true, null_idx, (best_pct,)))
        else:
            best_pct = max(dice, key=lambda k: float(np.nanmean(dice[k]["match"])))
        corr_title, corr_m, corr_n = _match_null_correlation(pred, true, max_voxels=5000, seed=13)

    spin = {"enabled": not args.no_spin, "n_perm": args.spin_n_perm, "density": args.spin_density, "pct": 90,
            "p": np.full(n, np.nan, dtype="float32"), "dice_surface": np.full(n, np.nan, dtype="float32"),
            "eligible": np.zeros(n, dtype=bool), "cortical_fraction": np.full(n, np.nan, dtype="float32"),
            "tested_idx": np.array([], dtype=int), "errors": []}
    if not args.no_spin:
        with timed("a) spin test", log, args.cache_dir, stage=STEM, n_perm=args.spin_n_perm):
            from joblib import Parallel, delayed, parallel_config
            from neurovlm.metrics.text_to_brain import precompute_spin_permutations

            masker = load_masker()
            surf_mask, surf_method = build_surface_eligibility_mask(masker)
            for i in range(n):
                ok, frac, _ = surface_metric_eligibility(
                    "pubmed", true[i], surface_mask=surf_mask, dice_pct=90, pubmed_surface_filter_enabled=True,
                    pubmed_min_cortical_top_mass_fraction=0.80)
                spin["eligible"][i] = bool(ok)
                spin["cortical_fraction"][i] = float(frac) if np.isfinite(frac) else np.nan
            idx = np.flatnonzero(spin["eligible"])
            if args.spin_max_samples and len(idx) > args.spin_max_samples:
                idx = np.sort(np.random.default_rng(args.seed).choice(idx, args.spin_max_samples, replace=False))
            spin["tested_idx"] = idx
            spin["surface_mask_method"] = surf_method
            log.info("spin test: %d/%d cortical maps, %d permutations, %d workers", len(idx), n, args.spin_n_perm, args.n_jobs)
            _prefetch_spin_atlases(args.spin_density)
            precompute_spin_permutations(density=args.spin_density, n_perm=args.spin_n_perm, random_state=13)
            kw = dict(pct=90, n_perm=args.spin_n_perm, random_state=13, density=args.spin_density, method="nearest")
            with parallel_config(backend="loky", inner_max_num_threads=1):
                results = Parallel(n_jobs=max(1, args.n_jobs), batch_size=4)(
                    delayed(_spin_one)(pred[i], true[i], **kw) for i in idx)
            for i, r in zip(idx, results):
                spin["p"][i] = r["p"]
                spin["dice_surface"][i] = r["dice_surface"]
                if r["error"]:
                    spin["errors"].append({"row": int(i), "error": r["error"]})
            if spin["errors"]:
                log.warning("spin test failed for %d maps (first: %s)", len(spin["errors"]), spin["errors"][0]["error"])

    payload = {
        "n": n, "n_total": n_total, "pmids": pmids, "map_calibration": args.map_calibration,
        "retrieval": {"auc": float(auc), "normalized_k": curve_df["normalized_k"].to_numpy(),
                      "recall": curve_df["recall_at_normalized_k"].to_numpy()},
        "contrastive": {"match": rank_df["matched_contrastive_sim"].to_numpy(), "null": rank_df["null_contrastive_sim"].to_numpy(),
                        "ranks": rank_df["rank"].to_numpy()},
        "dice": dice, "dice_pct": float(best_pct),
        "correlation": {"title": corr_title, "match": corr_m, "null": corr_n},
        "spin": spin,
    }
    save_pt(cache, payload)
    return payload


def compute_b2t(args, device):
    """Row b (docs/figures/24): network/region/function captions joined with [SEP] vs title [SEP] LLM summary;
    retrieval, contrastive, BERTScore, MiniLM cosine and the per-map contrastive null test."""
    cache = cache_file(args, "fig4_b2t", excl=True)
    if cache.exists() and not args.force:
        log.info("cache hit %s", cache)
        return load_pt(cache)
    import pandas as pd
    import torch
    from _shared import load_dataset
    from neurovlm.metrics.common import project_brain_latents_to_shared, project_text_latents_to_shared
    from neurovlm.metrics.text_to_brain import encode_text_batched

    with timed("b) data", log, args.cache_dir, stage=STEM):
        images, pmids = load_dataset("pubmed_images")
        pmids = np.asarray(pmids)
        df_pubs = load_dataset("pubmed_text").copy()
        df_sum = load_dataset("pubmed_summaries")
        # summaries joined on pmid (as strings: the two parquets use different dtypes), not positionally as in
        # notebook 24; rows without a summary keep their abstract
        summ = df_sum.dropna(subset=["summary"]).drop_duplicates("pmid")
        summ = summ.set_index(summ["pmid"].astype(str))["summary"].astype(str).str.strip("\n").str.strip(" \n")
        abstract = df_pubs["description"].copy()
        df_pubs["description"] = df_pubs["pmid"].astype(str).map(summ)
        n_missing = int(df_pubs["description"].isna().sum())
        if n_missing > 0.05 * len(df_pubs):
            raise RuntimeError(f"{n_missing}/{len(df_pubs)} PubMed rows have no LLM summary; pmid join failed")
        if n_missing:
            log.warning("%d PubMed rows have no LLM summary; using the abstract for those", n_missing)
            df_pubs["description"] = df_pubs["description"].fillna(abstract)
        test = df_pubs[pubmed_test_mask(args, df_pubs, log)].drop_duplicates("pmid").set_index("pmid", drop=False)
        m = pd.Series(pmids).isin(test.index).to_numpy().copy()  # writable (pandas copy-on-write views are read-only)
        rows_t = np.flatnonzero(m)
        m[rows_t[~pubmed_keep(args, pmids[rows_t], key="PubMed test (brain-to-text)", log=log)]] = False
        pmids_t, images_t = pmids[m], images[m].float()
        del images
        aligned = test.loc[pmids_t].reset_index(drop=True)
        n_total = len(pmids_t)
        if args.max_samples and args.max_samples < n_total:
            sel = np.sort(np.random.default_rng(args.seed).choice(n_total, args.max_samples, replace=False))
            pmids_t, images_t, aligned = pmids_t[sel], images_t[torch.as_tensor(sel)], aligned.iloc[sel].reset_index(drop=True)
        n = len(pmids_t)
        truth = (aligned["name"].fillna("").astype(str) + " [SEP] " + aligned["description"].fillna("").astype(str)).tolist()

    gen_cache = cache_file(args, "fig4_b2t_generation", "parquet", excl=True)
    nvlm = make_nvlm(device, args.model_dir, log)
    df_gen = pd.read_parquet(gen_cache) if gen_cache.exists() and not args.force else None
    if df_gen is not None and df_gen["pmid"].astype(str).tolist() != [str(p) for p in pmids_t]:
        log.warning("generation cache %s does not match the current sample; regenerating", gen_cache)
        df_gen = None
    if df_gen is None:
        with timed("b) generation", log, args.cache_dir, stage=STEM, n=n, batch=args.gen_batch_size):
            outs = {b: [] for b in BASES}
            for s in range(0, n, args.gen_batch_size):
                batch = images_t[s:s + args.gen_batch_size]
                for basis in BASES:
                    out = nvlm.generate_text(batch, seed=123 + s, basis=basis, do_sample=False, num_beams=1,
                                             max_new_tokens=128, projection_temp=0.035, repetition_penalty=1.25)
                    outs[basis].extend([out] if isinstance(out, str) else list(out))
                log.info("generated %d/%d", min(s + args.gen_batch_size, n), n)
            df_gen = pd.DataFrame({"pmid": [str(p) for p in pmids_t], **outs})
            for b in BASES:
                df_gen[b] = df_gen[b].map(strip_steering_tags).str.replace("\n\n", "\n", regex=True)
            df_gen["text"] = df_gen["network"] + " [SEP] " + df_gen["region"] + " [SEP] " + df_gen["function"]
            df_gen["truth"] = truth
            df_gen.to_parquet(gen_cache)
    generated = df_gen["text"].astype(str).tolist()
    null_idx = _notebook_deranged_perm(n, np.random.default_rng(0))

    with timed("b) contrastive metrics", log, args.cache_dir, stage=STEM):
        brain_latents = torch.cat([nvlm._encode_brain_flat(images_t[s:s + 256]).detach().cpu() for s in range(0, n, 256)])
        text_latents = encode_text_batched(nvlm, generated, batch_size=32)
        z_text = project_text_latents_to_shared(nvlm, text_latents, batch_size=256).float().cpu()
        z_brain = project_brain_latents_to_shared(nvlm, brain_latents, batch_size=256).float().cpu()
        scores = z_text @ z_brain.T
        matched = scores.diag()
        null = scores[torch.arange(n), torch.as_tensor(null_idx)]
        order = scores.argsort(dim=1, descending=True)
        first_hits = order.eq(torch.arange(n).view(-1, 1)).int().argmax(dim=1)
        recall = (torch.cumsum(torch.bincount(first_hits, minlength=n).float(), 0) / n).numpy()
        normalized_k = np.arange(1, n + 1) / n
        null_ge = (scores >= matched[:, None]).sum(dim=1).numpy() - 1
        pvals = (1.0 + null_ge) / float(n)
        del nvlm
        free_gpu()

    bert_cache = cache_file(args, "fig4_b2t_bert", excl=True)
    if bert_cache.exists() and not args.force:
        bert = load_pt(bert_cache)
    else:
        with timed("b) BERTScore (matched + null)", log, args.cache_dir, stage=STEM):
            from bert_score import score as bert_score
            from bert_score.utils import model2layers

            # the pinned snapshot dir and bert_score's layer for deberta-xlarge-mnli (bert_score takes no revision)
            deberta = {"model_type": str(hf_pinned_dir("microsoft/deberta-xlarge-mnli")),
                       "num_layers": model2layers["microsoft/deberta-xlarge-mnli"]}
            _, _, f1 = bert_score(cands=generated, refs=truth, lang="en", **deberta,
                                  batch_size=args.bert_batch_size, device=str(device), verbose=False)
            _, _, f1n = bert_score(cands=generated, refs=[truth[j] for j in null_idx], lang="en", **deberta,
                                   batch_size=args.bert_batch_size, device=str(device), verbose=False)
            bert = {"match": f1.float().numpy(), "null": f1n.float().numpy()}
            save_pt(bert_cache, bert)
            free_gpu()

    with timed("b) MiniLM semantic similarity", log, args.cache_dir, stage=STEM):
        from sentence_transformers import SentenceTransformer

        minilm = "sentence-transformers/all-MiniLM-L6-v2"
        st = SentenceTransformer(minilm, revision=hf_pinned(minilm), device=str(device))
        g = st.encode(generated, batch_size=32, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        t = st.encode(truth, batch_size=32, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        sem = {"match": np.einsum("ij,ij->i", g, t), "null": np.einsum("ij,ij->i", g, t[null_idx])}
        del st
        free_gpu()

    payload = {
        "n": n, "n_total": n_total, "pmids": [str(p) for p in pmids_t],
        "retrieval": {"auc": float(recall.mean()), "normalized_k": normalized_k, "recall": recall},
        "contrastive": {"match": matched.numpy(), "null": null.numpy()},
        "bertscore": bert, "semantic": sem,
        "null_test": {"p": pvals},
        "generation": {"num_beams": 1, "max_new_tokens": 128, "projection_temp": 0.035, "repetition_penalty": 1.25,
                       "cache": str(gen_cache)},
    }
    save_pt(cache, payload)
    return payload


def render(args, a, b):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    plt.rcParams["savefig.bbox"] = "standard"  # the saved figure is exactly FIG_FULL_W wide
    fs = args.font_size
    W = FIG_FULL_W
    # inches: a square recall panel then four box panels per row; one legend (Match / Null / p=0.05) on top replaces
    # the boxes' tick labels
    ax_h, legend_h, title_h, below_h, row_gap = 0.9, 0.2, 0.36, 0.37, 0.06
    x_recall, margin = 0.6, 0.44  # left edge of the recall axes; y label + ticks left of each box panel
    box_w = (W - 0.18 - x_recall - ax_h - 4 * margin) / 4  # the right margin holds the last two-line title
    n_rows = (a is not None) + (b is not None)
    H = legend_h + n_rows * (title_h + ax_h + below_h) + (n_rows - 1) * row_gap
    fig = plt.figure(figsize=(W, H))
    fig.legend(handles=[Patch(facecolor=MATCH_COLOR, alpha=0.8, label="Match"),
                        Patch(facecolor=NULL_COLOR, alpha=0.8, label="Null"),
                        Line2D([], [], color=STATUS_BAD, ls="--", lw=0.9, label="p = 0.05")],
               loc="upper right", bbox_to_anchor=(1.0, 1.0), ncol=3, fontsize=fs, handlelength=1.4,
               columnspacing=1.2, borderaxespad=0.1)
    stats, null_tests = {}, {}

    def box_axes(y0):
        """The five axes of a row whose axes bottom is y0 inches."""
        rects = [(x_recall, ax_h)] + [(x_recall + ax_h + margin + k * (box_w + margin), box_w) for k in range(4)]
        return [fig.add_axes([x / W, y0 / H, w / W, ax_h / H]) for x, w in rects]

    def p_panel(ax, pvals, n_grid, title, note, key):
        """Per-map permutation p-values (blue) next to their exact null (grey, p uniform on {1/n_grid, ..., 1});
        tested with a one-sided binomial test of #(p < 0.05)."""
        pv = np.asarray(pvals, dtype=float)
        pv = pv[np.isfinite(pv)]
        if len(pv) == 0:
            ax.text(0.5, 0.5, note, ha="center", va="center", transform=ax.transAxes, fontsize=fs, wrap=True)
            ax.set_xticks([])
            ax.set_ylabel("−log10 p")
            ax.set_title(title)
            return
        vals = -np.log10(np.clip(pv, 1e-300, None))
        grid = np.arange(1, n_grid + 1) / float(n_grid)
        null = -np.log10(grid)
        p0 = float((grid < 0.05).mean())
        k = int((pv < 0.05).sum())
        lp = binom_log10_sf(k, len(pv), p0)
        null_tests[key] = {
            "test": "one-sided exact binomial test of #(p < 0.05) against the exact null rate (p uniform on {1/N..1})",
            "n": int(len(pv)), "N_candidates": int(n_grid), "min_attainable_p": 1.0 / n_grid,
            "frac_p_lt_0_05": k / len(pv), "null_frac_p_lt_0_05": p0,
            "median_p": float(np.median(pv)), "null_median_p": float(np.median(grid)),
            "frac_at_min_p": float((pv <= (1.0 / n_grid) * (1 + 1e-6)).mean()),  # p stored as float32
            "mean_neglog10_p": float(vals.mean()), "null_mean_neglog10_p": float(null.mean()),
            "cohen_d": cohen_d(vals, null), "log10_p": lp, "p": format_p(lp)}
        match_null_box(ax, vals, null, ylabel="−log10 p", title=title, annotation=sig_text(lp), annotation_size=fs)
        thr = -np.log10(0.05)
        ax.axhline(thr, color=STATUS_BAD, ls="--", lw=0.9, zorder=0)  # p = 0.05, named in the legend

    def row(ridx, data, label, letter, kind):
        y0 = H - legend_h - ridx * (title_h + ax_h + below_h + row_gap) - title_h - ax_h
        axes = box_axes(y0)
        ax = axes[0]
        # two-line title: the square panel (recall_axes) is narrower than the one-line title
        recall_axes(ax, title=("Generated-image" if kind == "a" else "Generated-text") + f"\nretrieval (n={data['n']})")
        ax.plot(data["retrieval"]["normalized_k"], data["retrieval"]["recall"], color=BLUE, lw=1.3)
        # single series: no legend box; AUC/n annotation + labelled chance diagonal instead
        ax.text(0.97, 0.05, f"AUC = {data['retrieval']['auc']:.3f}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=fs, color=BLUE)
        ax.text(0.7, 0.7, "chance", ha="center", va="center", fontsize=fs, color="0.3",
                rotation=45, rotation_mode="anchor",  # square panel (recall_axes): the diagonal is at 45 degrees
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
        # panel letter at the left edge of the figure, above the rotated row label and level with the panel titles
        panel_label(fig, letter, 0.01 / W, (y0 + ax_h + title_h - 0.02) / H, size_delta=1.0)
        fig.text(0.07 / W, (y0 + ax_h / 2) / H, label, rotation=90, ha="center", va="center", fontsize=fs + 2)
        if kind == "a":
            specs = [("contrastive", "Contrastive", "Cosine similarity"),
                     ("dice", f"Dice\n(top {100 - data['dice_pct']:.0f}%)", "Dice"),
                     ("correlation", data["correlation"]["title"], "Correlation")]
        else:
            specs = [("contrastive", "Contrastive", "Cosine similarity"), ("bertscore", "BERTScore", "F1 score"),
                     ("semantic", "Semantic", "Cosine similarity")]
        for ax, (key, title, ylab) in zip(axes[1:4], specs):
            d = data["dice"][data["dice_pct"]] if key == "dice" else data[key]
            w = welch_log10p(d["match"], d["null"])
            stats[f"{kind}:{key}"] = w
            match_null_box(ax, d["match"], d["null"], ylabel=ylab, title=title, annotation=sig_text(w["log10_p"]),
                           annotation_size=fs)
        if kind == "a":
            sp = data["spin"]
            n_tested = int(len(sp["tested_idx"]))
            p_panel(axes[4], sp["p"], int(sp["n_perm"]) + 1, f"Spin test\n(n={n_tested})",
                    "spin test\ndisabled" if not sp["enabled"] else "no cortical\nmaps tested", "a:spin")
        else:
            p_panel(axes[4], data["null_test"]["p"], int(data["n"]), "Contrastive\nnull test", "n/a", "b:null_test")
        for ax in axes[1:]:
            ax.set_xticks([])  # Match / Null: the legend
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
            ax.tick_params(axis="y", pad=2)
            ax.yaxis.labelpad = 2

    if a is not None:
        row(0, a, "Text-to-brain", "a", "a")
    if b is not None:
        row(1, b, "Brain-to-text", "b", "b")
    return fig, stats, null_tests


def main():
    args = parse_args()
    banner(log, args, STEM)
    device = select_device(args.device)
    a = compute_t2b(args, device) if "a" in args.rows else None
    b = compute_b2t(args, device) if "b" in args.rows else None
    with timed("render", log, args.cache_dir, stage=STEM):
        fig, stats, null_tests = render(args, a, b)
        names = {"a:contrastive": "a Contrastive", "a:dice": "a Dice", "a:correlation": "a Pearson/Spearman",
                 "a:spin": "a Spin test", "b:contrastive": "b Contrastive", "b:bertscore": "b BERTScore",
                 "b:semantic": "b Semantic", "b:null_test": "b Contrastive null test"}
        per_panel = {names[k]: {"p": v["p"], "log10_p": round(v["log10_p"], 2), "cohen_d": round(v["cohen_d"], 2),
                                "test": "Welch t (match vs deranged null)" if k in stats else "exact binomial vs uniform-p null",
                                "below_1e-100": bool(v["log10_p"] < -100)}
                     for k, v in {**stats, **null_tests}.items()}
        dump = {"p_values_per_panel": per_panel, "welch_tests": stats, "null_tests": null_tests}
        if a is not None:
            sp = a["spin"]
            pv = sp["p"][np.isfinite(sp["p"])]
            dump["text_to_brain"] = {
                "n": a["n"], "n_total": a["n_total"], "retrieval_auc": a["retrieval"]["auc"], "dice_pct": a["dice_pct"],
                "dice_mean_by_pct": {str(k): float(np.nanmean(v["match"])) for k, v in a["dice"].items()},
                "correlation": a["correlation"]["title"],
                "spin": {"enabled": bool(sp["enabled"]), "n_tested": int(len(sp["tested_idx"])),
                         "n_eligible": int(sp["eligible"].sum()), "n_perm": sp["n_perm"], "density": sp["density"],
                         "frac_p_lt_0_05": float((pv < 0.05).mean()) if len(pv) else None,
                         "n_errors": len(sp["errors"])}}
        if b is not None:
            pv = np.asarray(b["null_test"]["p"])
            dump["brain_to_text"] = {"n": b["n"], "n_total": b["n_total"], "retrieval_auc": b["retrieval"]["auc"],
                                     "frac_p_lt_0_05": float((pv < 0.05).mean()), "generation": b["generation"]}
        finish(fig, STEM, args, log, text_dump=dump,
               components=COMPONENTS + (("pmids_split",) if args.model_dir else ()))


if __name__ == "__main__":
    main()
