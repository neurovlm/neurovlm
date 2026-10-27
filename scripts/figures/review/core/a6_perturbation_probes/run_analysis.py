#!/usr/bin/env python
"""Text-to-brain perturbation probes (comment 3.4): word substitution, composition ('A and B'), context (vs SPECTER2
and word2vec ridge controls) and negation, scored against Neurosynth term maps and Harvard-Oxford ROIs. Models of
--model-dir (default runs/final: text-to-brain head v2); every generated map is calibrated with the head's global factor.

Run: python run_analysis.py [--smoke] [--device cpu]   (outputs -> ./outputs)
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

log = C.get_logger("a6_perturbation")

FRAMES = [
    "{X}",
    "Neural correlates of {x}.",
    "Participants underwent fMRI during {x}.",
    "This study examines brain activation associated with {x} in healthy adults.",
    "We used functional MRI to identify the regions that support {x}.",
    "A meta-analysis of neuroimaging studies of {x}.",
    "Brain responses to {x} were measured in 24 healthy volunteers using fMRI.",
    "Healthy adults completed a task involving {x} while brain activity was recorded.",
]

SUBSTITUTION = [
    # pair, phrase A, phrase B, Neurosynth term A, term B, Harvard-Oxford ROIs A, ROIs B
    ("auditory/visual", "auditory stimulation", "visual stimulation", "auditory", "visual",
     ["Heschl's Gyrus (includes H1 and H2)", "Planum Temporale", "Superior Temporal Gyrus, posterior division"],
     ["Occipital Pole", "Intracalcarine Cortex", "Lateral Occipital Cortex, inferior division"]),
    ("motor/language", "motor processing", "language processing", "motor", "language",
     ["Precentral Gyrus", "Postcentral Gyrus", "Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)"],
     ["Inferior Frontal Gyrus, pars triangularis", "Inferior Frontal Gyrus, pars opercularis",
      "Middle Temporal Gyrus, posterior division"]),
    ("faces/places", "face perception", "place perception", "faces", "place",
     ["Temporal Occipital Fusiform Cortex", "Occipital Fusiform Gyrus", "Right Amygdala"],
     ["Parahippocampal Gyrus, posterior division", "Lingual Gyrus", "Precuneous Cortex"]),
    ("reward/pain", "reward processing", "pain processing", "reward", "pain",
     ["Left Accumbens", "Right Accumbens", "Left Caudate", "Right Caudate", "Frontal Medial Cortex"],
     ["Insular Cortex", "Parietal Opercular Cortex", "Left Thalamus", "Right Thalamus"]),
    ("working/episodic memory", "working memory", "episodic memory", "working memory", "episodic memory",
     ["Middle Frontal Gyrus", "Superior Parietal Lobule", "Paracingulate Gyrus"],
     ["Left Hippocampus", "Right Hippocampus", "Parahippocampal Gyrus, anterior division"]),
]

COMPOSITION = [
    # name, A, B, combined, Neurosynth A, B
    ("faces+places", "faces", "places", "faces and places", "faces", "place"),
    ("auditory+visual", "auditory stimulation", "visual stimulation", "auditory and visual stimulation", "auditory", "visual"),
    ("reward+pain", "reward", "pain", "reward and pain", "reward", "pain"),
    ("motor+language", "motor processing", "language processing", "motor and language processing", "motor", "language"),
]
COMP_FRAMES = ["{X}", "Neural correlates of {x}.", "Brain activity during {x} in healthy adults.", "An fMRI study of {x}."]

CONTEXT = {
    # keyword: [(context phrase, Neurosynth reference term)]
    "memory": [("memory encoding of word lists", "encoding"), ("spatial memory and navigation", "navigation"),
               ("working memory maintenance", "working memory"), ("emotional memory for fearful events", "fear"),
               ("autobiographical memory retrieval", "autobiographical")],
    "attention": [("visual spatial attention", "spatial attention"), ("attention to emotional faces", "emotional faces"),
                  ("sustained attention", "sustained attention"), ("auditory attention to speech", "speech")],
    "control": [("response inhibition and cognitive control", "response inhibition"), ("emotion regulation", "emotion regulation"),
                ("motor control of finger movements", "motor control")],
    "reward": [("monetary reward anticipation", "reward anticipation"), ("social reward", "social"),
               ("reward learning", "learning")],
    "words": [("reading words", "reading"), ("remembering words", "encoding"), ("hearing spoken words", "speech")],
}
CTX_FRAMES = ["{X}", "Participants performed a task involving {x} during fMRI."]

NEGATION = [
    # id, affirmative, negated, Neurosynth term the affirmative names
    ("motor", "Participants performed a motor task.", "Participants did not perform a motor task.", "motor"),
    ("auditory", "Auditory stimulation activated the auditory cortex.",
     "Auditory stimulation did not activate the auditory cortex.", "auditory"),
    ("default mode", "Increased activity in the default mode network.", "Decreased activity in the default mode network.",
     "default mode"),
    ("faces", "Responses to faces.", "Responses to stimuli that were not faces.", "faces"),
    ("amygdala/fear", "The amygdala responded to fearful faces.", "The amygdala did not respond to fearful faces.", "amygdala"),
    ("pain", "Participants received painful heat stimulation.", "Participants received non-painful warm stimulation.", "pain"),
    ("reward", "Activation associated with reward.", "Absence of activation associated with reward.", "reward"),
    ("auditory-not-visual", "Auditory but not visual stimulation.", "Visual but not auditory stimulation.", "auditory"),
]
NEG_CORPUS = {
    "negated activation": r"\b(?:did not|does not|do not|failed to|no)\s+(?:significant(?:ly)?\s+)?(?:activat\w*|engag\w*|recruit\w*)",
    "no difference / no effect": r"\bno\s+(?:significant\s+)?(?:difference|effect|change)s?\b",
    "deactivation / decreased activity": r"\bdeactivat\w*|\bdecreased\s+(?:activity|activation|bold)",
    "affirmative activation": r"\b(?:activat(?:ed|ion|ing)|increased\s+(?:activity|activation|bold)|engag(?:ed|ement))\b",
}


def fill(frame: str, phrase: str) -> str:
    """Put a phrase into a sentence frame ({X}: capitalised, {x}: as is)."""
    return frame.replace("{X}", phrase[:1].upper() + phrase[1:]).replace("{x}", phrase)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=HERE / "outputs", help="Output directory.")
    M.add_model_args(p)
    p.add_argument("--device", default="auto", help="Device (auto/cpu/cuda).")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--smoke", action="store_true", help="Quick run: 2 frames per probe, 1500 ridge training papers.")
    p.add_argument("--max-samples", type=int, default=None, help="Cap on PubMed papers for the ridge controls and corpus counts.")
    p.add_argument("--neurosynth-dir", type=Path, default=C.default_neurosynth_dir(), help="Neurosynth v7 data (nimare layout).")
    p.add_argument("--w2v-path", type=Path, default=Path.home() / "gensim-data" / "word2vec-google-news-300" / "word2vec-google-news-300.gz",
                   help="word2vec binary.")
    p.add_argument("--w2v-limit", type=int, default=1_000_000, help="Most frequent word2vec entries loaded.")
    p.add_argument("--force", action="store_true", help="Refit the ridge controls even if cached.")
    p.add_argument("--no-exclusions", action="store_true",
                   help="Keep the leakage-audit PubMed val PMIDs in the ridge validation set.")
    a = p.parse_args()
    if a.smoke and not a.max_samples:
        a.max_samples = 1500
    return a


class Predictor:
    """Cached text -> map predictions for one model."""

    def __init__(self, fn):
        self.fn, self.cache = fn, {}

    def __call__(self, texts):
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if todo:
            for t, m in zip(todo, self.fn(todo)):
                self.cache[t] = m
        return np.stack([self.cache[t] for t in texts])


def roi_mask(ho: dict, names: list[str]) -> np.ndarray:
    idx = [list(ho["names"]).index(n) for n in names]
    return ho["masks"][idx].any(axis=0)


def mass_fraction(pred: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Share of the map mass inside ROI_A among ROI_A + ROI_B (each normalised by ROI size)."""
    a = float(pred[mask_a].mean())
    b = float(pred[mask_b].mean())
    return a / (a + b) if a + b > 0 else 0.5


def run_substitution(pred: Predictor, ns: C.Neurosynth, ho: dict, frames: list[str]):
    rows, maps = [], {}
    contrasts = {p[0]: ns.get(p[3]) - ns.get(p[4]) for p in SUBSTITUTION}
    for pair, pa, pb, ta, tb, ra, rb in SUBSTITUTION:
        nsa, nsb = ns.get(ta), ns.get(tb)
        ma, mb = roi_mask(ho, ra), roi_mask(ho, rb)
        A = pred([fill(f, pa) for f in frames])
        B = pred([fill(f, pb) for f in frames])
        maps[f"sub|{pair}|A"], maps[f"sub|{pair}|B"] = A.mean(0), B.mean(0)
        for fi, (a, b) in enumerate(zip(A, B)):
            r = {"pair": pair, "frame": fi, "text_A": fill(frames[fi], pa), "text_B": fill(frames[fi], pb)}
            for cond, m in (("A", a), ("B", b)):
                r[f"r_{cond}_nsA"], r[f"r_{cond}_nsB"] = C.pearson(m, nsa), C.pearson(m, nsb)
                r[f"dice_{cond}_nsA"], r[f"dice_{cond}_nsB"] = C.dice_pct(m, nsa), C.dice_pct(m, nsb)
                r[f"massfrac_{cond}"] = mass_fraction(m, ma, mb)
                r[f"dice_{cond}_roiA"], r[f"dice_{cond}_roiB"] = C.dice_mask(m, ma), C.dice_mask(m, mb)
            r["shift_r_ns"] = 0.5 * ((r["r_A_nsA"] - r["r_A_nsB"]) + (r["r_B_nsB"] - r["r_B_nsA"]))
            r["shift_dice_ns"] = 0.5 * ((r["dice_A_nsA"] - r["dice_A_nsB"]) + (r["dice_B_nsB"] - r["dice_B_nsA"]))
            r["shift_dice_roi"] = 0.5 * ((r["dice_A_roiA"] - r["dice_A_roiB"]) + (r["dice_B_roiB"] - r["dice_B_roiA"]))
            r["shift_massfrac"] = r["massfrac_A"] - r["massfrac_B"]
            diff = a - b
            r["diffmap_r_ns"] = C.pearson(diff, contrasts[pair])
            r["diffmap_r_ns_otherpairs"] = float(np.mean([C.pearson(diff, v) for k, v in contrasts.items() if k != pair]))
            r["diffmap_r_roi"] = C.pearson(diff, ma.astype(np.float32) - mb.astype(np.float32))
            r["map_r_A_B"] = C.pearson(a, b)
            rows.append(r)
    df = pd.DataFrame(rows)
    summ = []
    for pair, g in df.groupby("pair", sort=False):
        s = {"pair": pair, "n_frames": len(g)}
        for k in ("shift_r_ns", "shift_dice_ns", "shift_dice_roi", "shift_massfrac", "diffmap_r_ns", "diffmap_r_ns_otherpairs",
                  "diffmap_r_roi", "map_r_A_B"):
            s[k] = float(g[k].mean())
        s["frac_frames_shift_correct"] = float((g["shift_r_ns"] > 0).mean())
        s["frac_frames_mass_correct"] = float((g["shift_massfrac"] > 0).mean())
        summ.append(s)
    allrow = {"pair": "all", "n_frames": len(df)}
    for k in summ[0]:
        if k not in ("pair", "n_frames"):
            allrow[k] = float(np.mean([s[k] for s in summ]))
    summ.append(allrow)
    t = C.paired_test(df["diffmap_r_ns"], df["diffmap_r_ns_otherpairs"])
    return df, pd.DataFrame(summ), maps, {"diffmap_matched_vs_otherpairs": t}


def run_composition(pred: Predictor, ns: C.Neurosynth, frames: list[str]):
    from scipy.optimize import nnls

    rows, maps = [], {}
    for name, pa, pb, pc, ta, tb in COMPOSITION:
        nsa, nsb = ns.get(ta), ns.get(tb)
        A, B, Cm = (pred([fill(f, p) for f in frames]) for p in (pa, pb, pc))
        maps[f"comp|{name}|A"], maps[f"comp|{name}|B"], maps[f"comp|{name}|AB"] = A.mean(0), B.mean(0), Cm.mean(0)
        for fi, (a, b, c) in enumerate(zip(A, B, Cm)):
            union = C.top_pct(a) | C.top_pct(b)
            inter = C.top_pct(a) & C.top_pct(b)
            vmax, vmean = np.maximum(a, b), 0.5 * (a + b)
            r = {"composition": name, "frame": fi, "text": fill(frames[fi], pc),
                 "union_frac_voxels": float(union.mean()), "intersection_frac_voxels": float(inter.mean())}
            for lab, m in (("AB", c), ("A", a), ("B", b)):
                r[f"dice_union_{lab}"] = C.dice_topk_mask(m, union)
            r["dice90_AB_max"], r["dice90_AB_mean"] = C.dice_pct(c, vmax), C.dice_pct(c, vmean)
            r["dice90_AB_A"], r["dice90_AB_B"] = C.dice_pct(c, a), C.dice_pct(c, b)
            r["r_AB_max"], r["r_AB_mean"], r["r_AB_A"], r["r_AB_B"] = C.pearson(c, vmax), C.pearson(c, vmean), C.pearson(c, a), C.pearson(c, b)
            r["r_A_B"] = C.pearson(a, b)
            X = np.stack([a, b, np.ones_like(a)], axis=1).astype(np.float64)
            w, _ = nnls(X, c.astype(np.float64))
            resid = c - X @ w
            r["w_A"], r["w_B"], r["w_0"] = float(w[0]), float(w[1]), float(w[2])
            r["nnls_r2"] = float(1 - resid.var() / max(c.var(), 1e-12))
            r["r_AB_nsA"], r["r_AB_nsB"] = C.pearson(c, nsa), C.pearson(c, nsb)
            r["r_A_nsA"], r["r_B_nsB"] = C.pearson(a, nsa), C.pearson(b, nsb)
            r["r_A_nsB"], r["r_B_nsA"] = C.pearson(a, nsb), C.pearson(b, nsa)
            r["retained_A"] = r["r_AB_nsA"] / r["r_A_nsA"] if r["r_A_nsA"] > 0 else np.nan
            r["retained_B"] = r["r_AB_nsB"] / r["r_B_nsB"] if r["r_B_nsB"] > 0 else np.nan
            rows.append(r)
    df = pd.DataFrame(rows)
    keys = [c for c in df.columns if c not in ("composition", "frame", "text")]
    summ = df.groupby("composition", sort=False)[keys].mean().reset_index()
    summ["union_closer_than_both_singles"] = df.groupby("composition", sort=False).apply(
        lambda g: float(((g["dice_union_AB"] > g["dice_union_A"]) & (g["dice_union_AB"] > g["dice_union_B"])).mean()),
        include_groups=False).to_numpy()
    tests = {"dice_union_AB_vs_best_single": C.paired_test(df["dice_union_AB"], np.maximum(df["dice_union_A"], df["dice_union_B"]))}
    return df, summ, maps, tests


def run_context(predictors: dict, ns: C.Neurosynth, frames: list[str]):
    rows, items = [], []
    for kw, ctxs in CONTEXT.items():
        refs = np.stack([ns.get(t) for _, t in ctxs])
        for model, pred in predictors.items():
            M = np.stack([pred([fill(f, phrase) for f in frames]).mean(0) for phrase, _ in ctxs])  # frame-averaged
            R = C.pearson_matrix(M, refs)  # [contexts, refs]
            D = C.pearson_matrix(M, M)
            iu = np.triu_indices(len(ctxs), 1)
            for i, (phrase, term) in enumerate(ctxs):
                off = np.delete(R[i], i)
                items.append({"keyword": kw, "model": model, "context": phrase, "reference": term, "r_own": float(R[i, i]),
                              "r_other_mean": float(off.mean()), "specificity": float(R[i, i] - off.mean()),
                              "identified": bool(np.argmax(R[i]) == i), "dice_own": C.dice_pct(M[i], refs[i])})
            rows.append({"keyword": kw, "model": model, "n_contexts": len(ctxs),
                         "context_spread": float(np.mean(1 - D[iu])),
                         "specificity": float(np.mean([R[i, i] - np.delete(R[i], i).mean() for i in range(len(ctxs))])),
                         "identification_acc": float(np.mean([np.argmax(R[i]) == i for i in range(len(ctxs))])),
                         "r_own_mean": float(np.mean(np.diag(R)))})
    df_kw, df_items = pd.DataFrame(rows), pd.DataFrame(items)
    summ = df_items.groupby("model", sort=False).agg(n=("specificity", "size"), r_own=("r_own", "mean"),
                                                     r_other=("r_other_mean", "mean"), specificity=("specificity", "mean"),
                                                     identification_acc=("identified", "mean"), dice_own=("dice_own", "mean")).reset_index()
    spread = df_kw.groupby("model", sort=False)["context_spread"].mean()
    summ["context_spread"] = summ["model"].map(spread)
    tests = {}
    base = df_items[df_items.model == "neurovlm"].reset_index(drop=True)
    for m in predictors:
        if m != "neurovlm":
            other = df_items[df_items.model == m].reset_index(drop=True)
            tests[f"specificity_neurovlm_vs_{m}"] = C.paired_test(base["specificity"], other["specificity"])
    return df_kw, df_items, summ, tests


def run_negation(pred: Predictor, ns: C.Neurosynth):
    rows, maps = [], {}
    for nid, aff, neg, term in NEGATION:
        a, n = pred([aff, neg])
        ref = ns.get(term)
        maps[f"neg|{nid}|aff"], maps[f"neg|{nid}|neg"] = a, n
        rows.append({"probe": nid, "affirmative": aff, "negated": neg, "reference": term,
                     "r_aff_neg": C.pearson(a, n), "dice90_aff_neg": C.dice_pct(a, n),
                     "r_aff_ref": C.pearson(a, ref), "r_neg_ref": C.pearson(n, ref),
                     "delta_r_ref": C.pearson(n, ref) - C.pearson(a, ref),
                     "dice_aff_ref": C.dice_pct(a, ref), "dice_neg_ref": C.dice_pct(n, ref)})
    return pd.DataFrame(rows), maps


def corpus_negation_counts(abstracts: pd.Series) -> pd.DataFrame:
    txt = abstracts.fillna("").str.lower()
    rows = []
    for name, pat in NEG_CORPUS.items():
        hit = txt.str.contains(pat, regex=True)
        rows.append({"pattern": name, "regex": pat, "n_abstracts": int(hit.sum()), "frac_abstracts": float(hit.mean())})
    return pd.DataFrame(rows)


def fit_ridge(X: np.ndarray, Y: np.ndarray, Xv: np.ndarray, Yv: np.ndarray, alphas=(1.0, 10.0, 100.0, 1e3, 1e4, 1e5)):
    """Closed-form ridge on standardised inputs; alpha chosen by validation latent MSE."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    ym = Y.mean(0)
    Xs, Xvs = (X - mu) / sd, (Xv - mu) / sd
    G = Xs.T @ Xs
    best = None
    for a in alphas:
        W = np.linalg.solve(G + a * np.eye(G.shape[0]), Xs.T @ (Y - ym))
        mse = float(np.mean((Xvs @ W + ym - Yv) ** 2))
        if best is None or mse < best[0]:
            best = (mse, a, W)
    mse, a, W = best
    return {"mu": mu, "sd": sd, "ym": ym, "W": W, "alpha": a, "val_mse": mse,
            "val_r2": float(1 - mse / np.mean((Yv - Yv.mean(0)) ** 2))}


def ridge_predict(model: dict, X: np.ndarray) -> np.ndarray:
    return ((X - model["mu"]) / model["sd"]) @ model["W"] + model["ym"]


class W2V:
    def __init__(self, path: Path, limit: int):
        from gensim.models import KeyedVectors

        self.kv = KeyedVectors.load_word2vec_format(str(path), binary=True, limit=limit)

    def __call__(self, texts) -> np.ndarray:
        out = np.zeros((len(texts), self.kv.vector_size), np.float32)
        for i, t in enumerate(texts):
            vecs = []
            for w in re.findall(r"[A-Za-z][A-Za-z\-']*", str(t)):
                for cand in (w, w.lower(), w.capitalize()):
                    if cand in self.kv.key_to_index:
                        vecs.append(self.kv[cand])
                        break
            if vecs:
                v = np.mean(vecs, axis=0)
                out[i] = v / (np.linalg.norm(v) + 1e-8)
        return out


def build_controls(args, models, device, timings) -> tuple[dict, dict, dict]:
    """SPECTER2 and word2vec ridge controls: text -> autoencoder latent, fit on the train split, alpha on val."""
    import torch

    info = {}
    # exclusions change the validation set, so they get their own cache file
    cache = args.out / "cache" / f"ridge_controls_{M.model_tag()}_n{args.max_samples or 'all'}{'' if args.no_exclusions else '_excl'}.npz"
    lat, pm = C.load_pubmed_latents()
    masks = C.split_masks(pm)
    tr = C.take_prefix(masks["train"], args.max_samples)  # training set: never filtered
    n_va = max(200, (args.max_samples or 0) // 8) if args.max_samples else None
    # drop val PMIDs with a near-duplicate in the training split (leakage audit) before taking the val prefix
    v_keep = C.pubmed_keep_mask(pm[masks["val"]], enabled=not args.no_exclusions, label="PubMed val (ridge alpha selection)",
                                log=log.info)
    dropped = pm[masks["val"]][~v_keep]
    excl = {"enabled": not args.no_exclusions, "source": "scripts/eval_exclusions/pubmed_exclusions.csv",
            "n_pubmed_val_pool": int(masks["val"].sum()), "n_pubmed_val_dropped": int(len(dropped)),
            "n_pubmed_val_dropped_from_unfiltered_prefix": int(np.isin(pm[C.take_prefix(masks["val"], n_va)], dropped).sum()),
            "n_pubmed_test_dropped": 0, "n_neurovault_publications_dropped": 0, "n_neurovault_images_dropped": 0,
            "note": "PubMed test and NeuroVault items are not used by this analysis; the ridge training set is not filtered",
            "pubmed_val_dropped_pmids": sorted(int(x) for x in dropped)}
    va = C.take_prefix(masks["val"] & ~np.isin(pm, dropped), n_va)
    w2v = None
    if cache.exists() and not args.force:
        z = np.load(cache, allow_pickle=True)
        ridge = {k: {kk: z[f"{k}_{kk}"] for kk in ("mu", "sd", "ym", "W")} for k in ("specter", "w2v")}
        info = z["info"].item()
    else:
        with timings.time("controls_data", n_train=int(tr.sum()), n_val=int(va.sum())):
            rows = C.load_pubmed_text_rows(pm[tr | va])
            text = (rows["name"] + " " + rows["description"]).tolist()
            sel_tr = tr[tr | va]
            Y = lat[tr | va].numpy()
            S = torch.nn.functional.normalize(C.load_specter_latents(pm[tr | va]), dim=1).numpy()
            # check: pre-computed SPECTER2 embeddings vs the live encoder
            live = C.encode_specter(models["specter"], [f"{n}{models['specter'].sep_token}{d}" for n, d in
                                                        zip(rows["name"][:8], rows["description"][:8])]).numpy()
            info["specter_precomputed_vs_live_cos"] = float(np.mean(np.sum(live * S[:8], axis=1)))
        with timings.time("w2v_load", limit=args.w2v_limit):
            w2v = W2V(args.w2v_path, args.w2v_limit)
        with timings.time("w2v_embed_corpus", n=len(text)):
            Wv = w2v(text)
        ridge = {}
        for name, X in (("specter", S), ("w2v", Wv)):
            with timings.time(f"ridge_{name}"):
                r = fit_ridge(X[sel_tr], Y[sel_tr], X[~sel_tr], Y[~sel_tr])
            ridge[name] = r
            info[f"{name}_alpha"], info[f"{name}_val_latent_r2"] = r["alpha"], r["val_r2"]
            log.info("ridge %s: alpha=%g val latent R2=%.3f", name, r["alpha"], r["val_r2"])
        info.update(n_train=int(tr.sum()), n_val=int(va.sum()), w2v_limit=args.w2v_limit, w2v_path=str(args.w2v_path))
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, info=np.array(info, dtype=object),
                            **{f"{k}_{kk}": v[kk] for k, v in ridge.items() for kk in ("mu", "sd", "ym", "W")})
    ae = models["autoencoder"]

    def spec_fn(texts):
        X = C.encode_specter(models["specter"], texts).numpy()
        return C.decode_latents(ae, ridge_predict(ridge["specter"], X), device)

    def w2v_fn(texts):
        nonlocal w2v
        if w2v is None:  # cached ridge: load word2vec only when needed
            with timings.time("w2v_load_probe", limit=args.w2v_limit):
                w2v = W2V(args.w2v_path, args.w2v_limit)
        return C.decode_latents(ae, ridge_predict(ridge["w2v"], w2v(texts)), device)

    return {"specter_ridge": spec_fn, "w2v_ridge": w2v_fn}, info, excl


def main() -> None:
    args = parse_args()
    C.set_offline_env()
    C.seed_all(args.seed)
    M.from_args(args)
    args.out.mkdir(parents=True, exist_ok=True)
    device = C.select_device(args.device)
    timings = C.Timings(args.out / "timings.json")
    frames = FRAMES[:2] if args.smoke else FRAMES
    comp_frames = COMP_FRAMES[:2] if args.smoke else COMP_FRAMES
    log.info("device=%s model=%s smoke=%s out=%s", device, M.model_tag(), args.smoke, args.out)

    with timings.time("load_references"):
        ns = C.Neurosynth(args.neurosynth_dir, args.out / "cache")
        ho = C.harvard_oxford(args.out / "cache")
    with timings.time("load_models", device=str(device)):
        models = C.load_t2b_models(device)
    nvlm = Predictor(lambda t: C.text_to_brain(models, t, device))

    results: dict = {"config": {"smoke": args.smoke, "seed": args.seed, "device": str(device), "dice_pct": C.DICE_PCT,
                                "frames": frames, "comp_frames": comp_frames, "ctx_frames": CTX_FRAMES,
                                "neurosynth_source": ns.source, "max_samples": args.max_samples,
                                "no_exclusions": args.no_exclusions, **M.model_info()},
                     "versions": C.installed_versions()}
    all_maps: dict[str, np.ndarray] = {}

    with timings.time("substitution", n_frames=len(frames)):
        sub_df, sub_sum, maps, tests = run_substitution(nvlm, ns, ho, frames)
        all_maps.update(maps)
        results["substitution_tests"] = tests
        sub_df.to_csv(args.out / "substitution_per_frame.csv", index=False)
        sub_sum.to_csv(args.out / "substitution_summary.csv", index=False)
    with timings.time("composition", n_frames=len(comp_frames)):
        comp_df, comp_sum, maps, tests = run_composition(nvlm, ns, comp_frames)
        all_maps.update(maps)
        results["composition_tests"] = tests
        comp_df.to_csv(args.out / "composition_per_frame.csv", index=False)
        comp_sum.to_csv(args.out / "composition_summary.csv", index=False)
    with timings.time("negation"):
        neg_df, maps = run_negation(nvlm, ns)
        all_maps.update(maps)
        neg_df.to_csv(args.out / "negation_items.csv", index=False)

    controls, info, excl = build_controls(args, models, device, timings)
    results["controls"] = info
    results["exclusions"] = excl
    log.info("eval exclusions: enabled=%s, PubMed val dropped %d of %d (%d of them in the unfiltered ridge val prefix)",
             excl["enabled"], excl["n_pubmed_val_dropped"], excl["n_pubmed_val_pool"],
             excl["n_pubmed_val_dropped_from_unfiltered_prefix"])
    predictors = {"neurovlm": nvlm, **{k: Predictor(fn) for k, fn in controls.items()}}
    with timings.time("context"):
        ctx_kw, ctx_items, ctx_sum, tests = run_context(predictors, ns, CTX_FRAMES)
        results["context_tests"] = tests
        ctx_kw.to_csv(args.out / "context_per_keyword.csv", index=False)
        ctx_items.to_csv(args.out / "context_per_item.csv", index=False)
        ctx_sum.to_csv(args.out / "context_summary.csv", index=False)
        for kw, ctxs in CONTEXT.items():
            for phrase, _ in ctxs:
                for m in ("neurovlm", "w2v_ridge"):
                    all_maps[f"ctx|{kw}|{phrase}|{m}"] = predictors[m]([fill(f, phrase) for f in CTX_FRAMES]).mean(0)

    with timings.time("corpus_negation_counts"):
        lat, pm = C.load_pubmed_latents()
        tr = C.take_prefix(C.split_masks(pm)["train"], args.max_samples)
        rows = C.load_pubmed_text_rows(pm[tr])
        corpus = corpus_negation_counts(rows["name"] + " " + rows["description"])
        corpus["n_total"] = len(rows)
        corpus.to_csv(args.out / "corpus_negation_counts.csv", index=False)

    # Neurosynth maps for the figure; NIfTI of the frame-averaged substitution/composition maps
    for pair in SUBSTITUTION:
        for t in pair[3:5]:
            all_maps[f"ns|{t}"] = ns.get(t)
    for kw, ctxs in CONTEXT.items():
        for _, t in ctxs:
            all_maps[f"ns|{t}"] = ns.get(t)
    np.savez_compressed(args.out / "maps.npz", keys=np.array(list(all_maps)), maps=np.stack(list(all_maps.values())).astype(np.float16))
    nii_dir = args.out / "nifti"
    nii_dir.mkdir(exist_ok=True)
    for k, v in all_maps.items():
        if k.startswith(("sub|", "comp|")):
            C.unmask(v).to_filename(nii_dir / (re.sub(r"[^A-Za-z0-9]+", "_", k).strip("_") + ".nii.gz"))

    results["summary"] = {
        "substitution": sub_sum.to_dict(orient="records"), "composition": comp_sum.to_dict(orient="records"),
        "context": ctx_sum.to_dict(orient="records"), "negation": neg_df.drop(columns=["affirmative", "negated"]).to_dict(orient="records"),
        "corpus": corpus.to_dict(orient="records"),
    }
    results["config"].update(M.model_info())
    C.write_json(args.out / "results.json", results)
    log.info("substitution (all): %s", sub_sum.iloc[-1].to_dict())
    log.info("context: %s", ctx_sum.to_dict(orient="records"))
    log.info("done -> %s", args.out)


if __name__ == "__main__":
    main()
