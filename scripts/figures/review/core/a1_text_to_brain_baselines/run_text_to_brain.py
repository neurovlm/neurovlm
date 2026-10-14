#!/usr/bin/env python
"""Text-to-brain maps of the PubMed test papers by NeuroVLM and baselines, scored with the paper's Dice / Pearson recipe.

Target = autoencoder reconstruction of the paper's map; Dice of 70th-percentile masks; voxel Pearson r. Baselines:
Text2Brain, NeuroQuery, Neurosynth, ridge (SPECTER2 -> latent), atlas labelling, domain and global mean maps, nearest
training paper. Each test paper is flagged when it is in the NeuroQuery corpus (NeuroQuery's training set and the pool of
Text2Brain's training set) or in the Neurosynth v7 database.

    .venv/bin/python scripts/figures/review/core/a1_text_to_brain_baselines/run_text_to_brain.py --model-dir RUN
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts" / "pipeline"))
from _baselines import NeuroQueryTextToMap, NeurosynthTextToMap  # noqa: E402
from _text2brain import MIRROR, WEIGHTS, Text2BrainTextToMap  # noqa: E402
from common import apply_pubmed_exclusions  # noqa: E402

N_VOXELS = 28_542
PAPER_PCT, TOP_PCT = 70, 90
NULL_SEED = 13
RIDGE_ALPHAS = (0.01, 0.1, 0.3, 1.0, 3.0, 10.0)  # as scripts/models/09_projection_head.py
N_DOMAINS, KMEANS_SEED = 10, 0

# Atlas labelling: Harvard-Oxford maxprob-thr25 labels (left + right merged) and a synonym table, copied from
# a6_steering_tokens/run_correctness.py without its non-Harvard-Oxford regions (cerebellum, vermis, hypothalamus, STN),
# plus the Harvard-Oxford names it lacked (paracingulate, subcallosal, planum polare, occipital pole) and word
# boundaries on "lingual" (not "bilingual") and "striate" (not "extrastriate").
HO_CORT = {
    "FP": "Frontal Pole", "INS": "Insular Cortex", "SFG": "Superior Frontal Gyrus", "MFG": "Middle Frontal Gyrus",
    "IFGt": "Inferior Frontal Gyrus, pars triangularis", "IFGo": "Inferior Frontal Gyrus, pars opercularis",
    "PreCG": "Precentral Gyrus", "TP": "Temporal Pole", "STGa": "Superior Temporal Gyrus, anterior division",
    "STGp": "Superior Temporal Gyrus, posterior division", "MTGa": "Middle Temporal Gyrus, anterior division",
    "MTGp": "Middle Temporal Gyrus, posterior division", "MTGto": "Middle Temporal Gyrus, temporooccipital part",
    "ITGa": "Inferior Temporal Gyrus, anterior division", "ITGp": "Inferior Temporal Gyrus, posterior division",
    "ITGto": "Inferior Temporal Gyrus, temporooccipital part", "PostCG": "Postcentral Gyrus", "SPL": "Superior Parietal Lobule",
    "SMGa": "Supramarginal Gyrus, anterior division", "SMGp": "Supramarginal Gyrus, posterior division", "AG": "Angular Gyrus",
    "LOCs": "Lateral Occipital Cortex, superior division", "LOCi": "Lateral Occipital Cortex, inferior division",
    "ICC": "Intracalcarine Cortex", "FMC": "Frontal Medial Cortex",
    "SMA": "Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)", "SubC": "Subcallosal Cortex",
    "PaCG": "Paracingulate Gyrus", "ACCg": "Cingulate Gyrus, anterior division", "PCCg": "Cingulate Gyrus, posterior division",
    "PCun": "Precuneous Cortex", "Cun": "Cuneal Cortex", "FOrb": "Frontal Orbital Cortex",
    "PHGa": "Parahippocampal Gyrus, anterior division", "PHGp": "Parahippocampal Gyrus, posterior division", "LG": "Lingual Gyrus",
    "TFa": "Temporal Fusiform Cortex, anterior division", "TFp": "Temporal Fusiform Cortex, posterior division",
    "TOF": "Temporal Occipital Fusiform Cortex", "OF": "Occipital Fusiform Gyrus", "FO": "Frontal Opercular Cortex",
    "CO": "Central Opercular Cortex", "PO": "Parietal Opercular Cortex", "PP": "Planum Polare",
    "HG": "Heschl's Gyrus (includes H1 and H2)", "PT": "Planum Temporale", "SCC": "Supracalcarine Cortex", "OP": "Occipital Pole",
}
HO_SUB = {"THA": "Thalamus", "CAU": "Caudate", "PUT": "Putamen", "PAL": "Pallidum", "HIP": "Hippocampus", "AMY": "Amygdala",
          "NAc": "Accumbens", "BS": "Brain-Stem"}
# (name, regex on the lower-cased text, atlas parts), specific before general: a match is blanked before the next pattern
REGIONS = [
    ("dorsal anterior cingulate cortex", r"dorsal anterior cingulate|\bdacc\b", "ACCg PaCG"),
    ("subgenual cingulate cortex", r"subgenual( anterior)?( cingulate)?|\bsgacc\b", "SubC ACCg"),
    ("paracingulate gyrus", r"paracingulate", "PaCG"),
    ("subcallosal cortex", r"subcallosal", "SubC"),
    ("anterior cingulate cortex", r"anterior cingulate|\bacc\b", "ACCg PaCG"),
    ("posterior cingulate cortex", r"posterior cingulate|\bpcc\b", "PCCg"),
    ("retrosplenial cortex", r"retrosplenial", "PCCg PCun ICC"),
    ("cingulate cortex", r"cingulate", "ACCg PCCg PaCG"),
    ("pre-supplementary motor area", r"pre-?\s?supplementary motor|\bpre-?sma\b", "SMA SFG PaCG"),
    ("supplementary motor area", r"supplementary motor|\bsma\b|juxtapositional", "SMA"),
    ("frontal eye fields", r"frontal eye fields?|\bfef\b", "PreCG MFG SFG"),
    ("dorsolateral prefrontal cortex", r"dorsolateral (pre)?frontal|\bdlpfc\b", "MFG SFG"),
    ("ventrolateral prefrontal cortex", r"ventrolateral prefrontal|\bvlpfc\b", "IFGt IFGo FO FOrb"),
    ("ventromedial prefrontal cortex", r"ventromedial prefrontal|\bvmpfc\b", "FMC SubC PaCG FP"),
    ("medial prefrontal cortex", r"medial prefrontal|\bmpfc\b|frontal medial", "FMC PaCG SFG FP SubC"),
    ("orbitofrontal cortex", r"orbito-?frontal|orbital frontal|frontal orbital|\bofc\b", "FOrb FMC"),
    ("frontal pole", r"(lateral )?frontopolar|frontal pole", "FP"),
    ("prefrontal cortex", r"prefrontal|\bpfc\b", "FP SFG MFG IFGt IFGo FMC FOrb PaCG SubC"),
    ("inferior frontal gyrus", r"inferior frontal|\bifg\b|broca'?s?|pars opercularis|pars triangularis", "IFGt IFGo"),
    ("frontal operculum", r"frontal opercul\w*", "FO"),
    ("middle frontal gyrus", r"middle frontal|\bmfg\b", "MFG"),
    ("superior frontal gyrus", r"superior frontal|\bsfg\b", "SFG"),
    ("frontal lobe", r"frontal (lobes?|cortex)", "FP SFG MFG IFGt IFGo PreCG FMC FOrb FO SMA PaCG SubC"),
    ("premotor cortex", r"premotor|\bpmc\b|\bpmd\b|\bpmv\b", "PreCG SFG MFG"),
    ("foot motor cortex", r"foot motor", "PreCG SMA"),
    ("paracentral lobule", r"paracentral", "PreCG PostCG SMA"),
    ("primary motor cortex", r"((primary|hand|face) )?motor cortex|precentral|\bm1\b", "PreCG"),
    ("secondary somatosensory cortex", r"secondary somatosensory|\bs2\b|\bsii\b", "PO CO"),
    ("primary somatosensory cortex", r"(primary )?somatosensory cortex|postcentral|\bs1\b", "PostCG"),
    ("primary gustatory cortex", r"(primary )?gustatory cortex", "INS FO"),
    ("vestibular cortex", r"vestibular cortex", "PO INS"),
    ("insula", r"(anterior |posterior )?insula\w*|interoceptive cortex", "INS"),
    ("olfactory cortex", r"olfactory cortex|piriform", "FOrb PHGa AMY"),
    ("parietal operculum", r"parietal opercul\w*", "PO"),
    ("central operculum", r"central opercul\w*|rolandic opercul\w*", "CO"),
    ("temporoparietal junction", r"temporo-?parietal junction|\btpj\b", "SMGp AG STGp"),
    ("wernicke's area", r"wernicke'?s?", "STGp PT SMGp"),
    ("intraparietal sulcus", r"intraparietal|\bips\b", "SPL SMGa SMGp AG"),
    ("superior parietal lobule", r"superior parietal|\bspl\b", "SPL"),
    ("inferior parietal lobule", r"inferior parietal|\bipl\b", "SMGa SMGp AG"),
    ("supramarginal gyrus", r"supramarginal|\bsmg\b", "SMGa SMGp"),
    ("angular gyrus", r"angular gyrus", "AG"),
    ("posterior parietal cortex", r"posterior parietal|\bppc\b", "SPL SMGa SMGp AG PCun"),
    ("precuneus", r"precune\w*", "PCun"),
    ("parietal cortex", r"parietal (cortex|lobes?)", "SPL SMGa SMGp AG PostCG PCun PO"),
    ("primary auditory cortex", r"primary auditory( cortex)?|heschl'?s?|\ba1\b", "HG"),
    ("planum temporale", r"planum temporale", "PT"),
    ("planum polare", r"planum polare", "PP"),
    ("auditory cortex", r"(voice-selective |secondary )?auditory (association )?cortex", "HG PT PP STGa STGp"),
    ("superior temporal sulcus", r"superior temporal sulcus|\bsts\b", "STGa STGp MTGa MTGp"),
    ("superior temporal gyrus", r"superior temporal|\bstg\b", "STGa STGp PT PP HG"),
    ("middle temporal visual area (MT)", r"middle temporal visual|\bmt\+?(?![a-z])|\bv5\b|motion-selective visual( cortex)?", "MTGto LOCi ITGto"),
    ("middle temporal gyrus", r"middle temporal|\bmtg\b", "MTGa MTGp MTGto"),
    ("inferior temporal gyrus", r"inferior temporal|\bitg\b", "ITGa ITGp ITGto"),
    ("temporal pole", r"temporal pole|anterior temporal lobe|\batl\b", "TP"),
    ("visual word form area", r"visual word form area|\bvwfa\b", "TOF TFp ITGto"),
    ("fusiform gyrus", r"fusiform|\bffa\b", "TFa TFp TOF OF"),
    ("parahippocampal gyrus", r"parahippocampal( place)?|\bppa\b|entorhinal|perirhinal", "PHGa PHGp"),
    ("medial temporal lobe", r"medial temporal lobe|\bmtl\b", "HIP AMY PHGa PHGp TFa"),
    ("hippocampus", r"\bhippocamp\w*", "HIP"),
    ("amygdala", r"amygdal\w*", "AMY"),
    ("temporal cortex", r"temporal (cortex|lobes?)", "TP STGa STGp MTGa MTGp MTGto ITGa ITGp ITGto TFa TFp PP PT HG"),
    ("primary visual cortex", r"primary visual( cortex)?|\bstriate cortex|calcarine( cortex| sulcus)?|\bv1\b", "ICC SCC OP"),
    ("occipital pole", r"occipital pole", "OP"),
    ("lateral occipital cortex", r"lateral occipital|\bloc\b", "LOCs LOCi"),
    ("extrastriate visual cortex", r"extrastriate( visual)?|color-selective visual|\bv[234]\b", "LOCs LOCi OF LG Cun"),
    ("cuneus", r"\bcuneus|\bcuneal", "Cun"),
    ("lingual gyrus", r"\blingual", "LG"),
    ("occipital cortex", r"occipital|visual cortex", "LOCs LOCi ICC SCC Cun LG OF OP"),
    ("thalamus", r"\bthalam\w*|pulvinar", "THA"),
    ("ventral striatum", r"ventral striatum|(nucleus )?accumbens", "NAc CAU PUT"),
    ("caudate", r"caudate", "CAU"),
    ("putamen", r"putamen", "PUT"),
    ("globus pallidus", r"(globus )?pallid\w*", "PAL"),
    ("striatum", r"striatum|striatal", "CAU PUT NAc"),
    ("basal ganglia", r"basal ganglia", "CAU PUT PAL NAc"),
    ("brainstem", r"brain-?\s?stem|midbrain|\bpons\b|medulla|periaqueductal|\bpag\b", "BS"),
]
REGION_TRAIL = r"(\s+(cortex|gyrus|gyri|area|areas|lobule|sulcus|nucleus|nuclei|region|lobe|complex|junction|formation|hemispheres?|place))*"
REGION_RE = [(re.compile(r"(" + rx + r")" + REGION_TRAIL), parts.split()) for _, rx, parts in REGIONS]


def deranged_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    """Permutation without fixed points (copy of scripts/pipeline/validate_models.py)."""
    perm = rng.permutation(n)
    fixed = np.flatnonzero(perm == np.arange(n))
    if len(fixed) > 1:
        perm[fixed] = np.roll(perm[fixed], 1)
    elif len(fixed) == 1:
        i = fixed[0]
        j = (i + 1) % n
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def zrows(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=1, keepdims=True)
    return x / np.clip(x.std(axis=1, keepdims=True), 1e-6, None)


def binarise(x: np.ndarray, pct: float) -> np.ndarray:
    return x > np.percentile(x, pct, axis=1, keepdims=True)


def dice_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    inter = np.logical_and(a, b).sum(axis=1).astype(np.float64)
    den = a.sum(axis=1) + b.sum(axis=1)
    return np.where(den > 0, 2.0 * inter / np.maximum(den, 1), 1.0)


def score(pred: np.ndarray, dec: np.ndarray, raw: np.ndarray, pair: np.ndarray, chunk: int = 512) -> dict[str, np.ndarray]:
    """Metrics of pred[i] against target pair[i]: decoded target (paper) and raw map."""
    out = {k: np.empty(len(pred)) for k in ("paper_dice", "paper_pearson", "dice_top10_raw", "pearson_raw")}
    for s in range(0, len(pred), chunk):
        p = pred[s:s + chunk].astype(np.float32)
        t, r = dec[pair[s:s + chunk]], raw[pair[s:s + chunk]]
        pz = zrows(p)
        out["paper_dice"][s:s + chunk] = dice_rows(binarise(p, PAPER_PCT), binarise(t, PAPER_PCT))
        out["paper_pearson"][s:s + chunk] = np.einsum("ij,ij->i", pz, zrows(t)) / p.shape[1]
        out["dice_top10_raw"][s:s + chunk] = dice_rows(binarise(p, TOP_PCT), binarise(r, TOP_PCT))
        out["pearson_raw"][s:s + chunk] = np.einsum("ij,ij->i", pz, zrows(r)) / p.shape[1]
    return out


def regions_in(text: str) -> list[int]:
    """Indices into REGIONS of the regions named in ``text`` (each phrase counted once; copied from a6_steering_tokens)."""
    t, hits = " " + str(text).lower() + " ", []
    for k, (rx, _) in enumerate(REGION_RE):
        if rx.search(t):
            hits.append(k)
            t = rx.sub(lambda m: " " * len(m.group(0)), t)
    return hits


def atlas_parts(cache: Path) -> dict[str, np.ndarray]:
    """{code: bool mask [V]}: Harvard-Oxford maxprob-thr25 2 mm labels, nearest-neighbour resampled to the mask grid."""
    path = cache / "atlas_harvard_oxford_parts.npz"
    if not path.exists():
        from nilearn import datasets
        from nilearn.image import resample_to_img

        from _hf_pins import load_masker

        masker = load_masker()

        def labels(img):
            res = resample_to_img(img, masker.mask_img_, interpolation="nearest", force_resample=True, copy_header=True)
            return np.rint(np.asarray(masker.transform(res)).ravel()).astype(np.int64)

        cort = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm", verbose=0)
        sub = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm", verbose=0)
        lc, ls = labels(cort.maps), labels(sub.maps)
        clab, slab = [str(x) for x in cort.labels], [str(x) for x in sub.labels]
        out = {k: lc == clab.index(v) for k, v in HO_CORT.items()}
        out |= {k: np.isin(ls, [i for i, n in enumerate(slab) if n.endswith(v)]) for k, v in HO_SUB.items()}
        np.savez_compressed(path, **out)
    z = np.load(path)
    return {k: z[k] for k in z.files}


def load(model_dir: Path) -> dict:
    """Texts, SPECTER2 embeddings, latents and map rows per split (sorted pmids; leakage exclusions on val/test)."""
    from _hf_pins import hf_file, load_dataset

    lat = torch.load(model_dir / "latent_neuro.pt", map_location="cpu", weights_only=False)
    latent, pm = lat["latent"].float(), np.asarray(lat["pmid"]).astype(np.int64)
    sp = torch.load(hf_file("neurovlm/embedded_text", "latent_specter2_adhoc.pt"), map_location="cpu", weights_only=False)
    pos_t = {int(p): i for i, p in enumerate(np.asarray(sp["pmid"]))}
    nv = torch.load(hf_file("neurovlm/embedded_text", "pubmed_images.pt"), map_location="cpu", weights_only=False, mmap=True)
    pos_m = {int(p): i for i, p in enumerate(np.asarray(nv["pmid"]))}
    df = load_dataset("pubmed_text").assign(pmid=lambda d: d["pmid"].astype(np.int64)).drop_duplicates("pmid").set_index("pmid")
    split = torch.load(model_dir / "pmids_split.pt", map_location="cpu", weights_only=False)
    pos_l = {int(p): i for i, p in enumerate(pm)}
    out = {"maps": nv["neuro_vectors"]}
    for name in ("train", "val", "test"):
        ids = np.sort(np.asarray(split[name]).astype(np.int64))
        ids = ids[np.isin(ids, pm)]
        if name != "train":
            ids = ids[apply_pubmed_exclusions(ids, label=f"PubMed {name}")]
        rows = df.loc[ids]
        out[name] = {"pmid": ids, "text": (rows["name"].fillna("").astype(str) + " [SEP] " + rows["description"].fillna("").astype(str)).tolist(),
                     "text_emb": sp["latent"][torch.as_tensor([pos_t[p] for p in ids])].float(),
                     "latent": latent[torch.as_tensor([pos_l[p] for p in ids])],
                     "map_index": torch.as_tensor([pos_m[p] for p in ids])}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, default=REPO / "scripts/outputs/runs/final", help="Training run directory.")
    p.add_argument("--out", type=Path, default=HERE / "outputs")
    p.add_argument("--device", default="cpu", help="Device (cpu/cuda).")
    p.add_argument("--nq-jobs", type=int, default=8, help="NeuroQuery workers (only when its map cache is missing).")
    p.add_argument("--t2b-device", default="cpu", help="Text2Brain device (only when its map cache is missing; CPU ~20 min).")
    args = p.parse_args()
    t0 = time.perf_counter()
    torch.set_grad_enabled(False)
    dev = torch.device(args.device)
    cache = args.out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    from _hf_pins import load_masker

    d = load(args.model_dir)
    tr, va, te = d["train"], d["val"], d["test"]
    n = len(te["pmid"])
    print(f"[data] train {len(tr['pmid'])} val {len(va['pmid'])} test {n} ({time.perf_counter() - t0:.0f} s)")
    ae = torch.load(args.model_dir / "autoencoder.pt", map_location="cpu", weights_only=False).to(dev).eval()
    head = torch.load(args.model_dir / "proj_head_text_mse.pt", map_location="cpu", weights_only=False).to(dev).eval()
    head_v1_path = args.model_dir / "proj_head_text_bce.pt"
    head_v1 = torch.load(head_v1_path, map_location="cpu", weights_only=False).to(dev).eval() if head_v1_path.exists() else None

    def decode(z, bs=512):
        return torch.cat([torch.sigmoid(ae.decoder(z[s:s + bs].to(dev))).float().cpu() for s in range(0, len(z), bs)]).numpy()

    def reconstruct(X):  # sigmoid(decoder(encoder(X))), the paper's target
        return decode(torch.cat([ae.encoder(X[s:s + 512].to(dev)).cpu() for s in range(0, len(X), 512)]))

    def maps(split):
        return d["maps"][split["map_index"]].float()

    X_te = maps(te)
    raw = X_te.numpy()
    dec_true = reconstruct(X_te)
    q = F.normalize(te["text_emb"], dim=1).to(dev)
    preds: dict[str, np.ndarray] = {"neurovlm": decode(head(q))}
    if head_v1 is not None:
        preds["neurovlm_v1"] = decode(head_v1(q))

    # NeuroQuery maps are cached (computed in parallel when missing)
    nq_path = cache / "neuroquery_pubmed_test.npy"
    if nq_path.exists() and np.array_equal(np.load(cache / "neuroquery_pubmed_test_pmid.npy").astype(np.int64), te["pmid"]):
        preds["neuroquery"] = np.load(nq_path)
    else:
        from joblib import Parallel, delayed

        def work(idx):
            nqm = NeuroQueryTextToMap(load_masker())
            assert nqm.build(), nqm.status
            return nqm.predict([te["text"][i] for i in idx])

        res = Parallel(n_jobs=args.nq_jobs)(delayed(work)(c) for c in np.array_split(np.arange(n), args.nq_jobs))
        preds["neuroquery"] = np.concatenate(res).astype(np.float32)
        np.save(nq_path, preds["neuroquery"])
        np.save(cache / "neuroquery_pubmed_test_pmid.npy", te["pmid"])
    ns = NeurosynthTextToMap(Path.home() / "nimare_data", load_masker(), cache, torch.device("cpu"))
    assert ns.build(), ns.status
    preds["neurosynth"], n_terms = ns.predict(te["text"])

    # Text2Brain maps are cached the same way (one text at a time)
    t2b_path = cache / "text2brain_pubmed_test.npy"
    if t2b_path.exists() and np.array_equal(np.load(cache / "text2brain_pubmed_test_pmid.npy").astype(np.int64), te["pmid"]):
        preds["text2brain"] = np.load(t2b_path)
    else:
        t2b = Text2BrainTextToMap(load_masker(), torch.device(args.t2b_device))
        t2b.build()
        preds["text2brain"] = t2b.predict(te["text"])
        np.save(t2b_path, preds["text2brain"])
        np.save(cache / "text2brain_pubmed_test_pmid.npy", te["pmid"])
    print(f"[maps] NeuroVLM, NeuroQuery, Neurosynth, Text2Brain ({time.perf_counter() - t0:.0f} s)")

    # test papers in the external training corpora (PMID match)
    from neuroquery.datasets import fetch_neuroquery_model

    nq_corpus = pd.read_csv(Path(fetch_neuroquery_model()) / "corpus_metadata.csv")["pmid"].astype(np.int64)
    ns_db = pd.read_csv(ns._files()["coordinates"].parent / "data-neurosynth_version-7_metadata.tsv.gz", sep="\t")["id"]
    in_nq, in_ns = np.isin(te["pmid"], nq_corpus), np.isin(te["pmid"], ns_db.astype(np.int64))

    # ridge SPECTER2 -> latent: closed form on the training split, alpha by validation paper Dice
    Xtr, Ytr = F.normalize(tr["text_emb"], dim=1).double(), tr["latent"].double()
    mx, my = Xtr.mean(0), Ytr.mean(0)
    Xc = Xtr - mx
    G, B = Xc.T @ Xc, Xc.T @ (Ytr - my)

    def ridge_w(alpha):
        return torch.linalg.solve(G + alpha * torch.eye(G.shape[0], dtype=torch.float64), B)

    X_va = maps(va)
    dec_va = reconstruct(X_va)
    q_va = F.normalize(va["text_emb"], dim=1).double()
    ident = np.arange(len(q_va))
    ridge_val = {}
    for alpha in RIDGE_ALPHAS:
        sc = score(decode(((q_va - mx) @ ridge_w(alpha) + my).float()), dec_va, X_va.numpy(), ident)
        ridge_val[alpha] = {k: float(v.mean()) for k, v in sc.items()}
    alpha = max(ridge_val, key=lambda a: ridge_val[a]["paper_dice"])
    preds["ridge"] = decode(((q.cpu().double() - mx) @ ridge_w(alpha) + my).float())
    val_neurovlm = {k: float(v.mean()) for k, v in score(decode(head(q_va.float().to(dev))), dec_va, X_va.numpy(), ident).items()}
    del X_va, dec_va
    print(f"[ridge] alpha {alpha} (val {ridge_val[alpha]}); NeuroVLM val {val_neurovlm}")

    # domains: k-means (k = 10) on the training-map latents. A query gets the mean training map of the domain whose
    # centroid in SPECTER2 space (mean embedding of its training papers) is closest; the global mean map pools all domains.
    from sklearn.cluster import KMeans

    dom = KMeans(N_DOMAINS, random_state=KMEANS_SEED, n_init=10).fit_predict(tr["latent"].numpy())
    t_tr = F.normalize(tr["text_emb"], dim=1)
    cent = F.normalize(torch.stack([t_tr[torch.as_tensor(dom == k)].mean(0) for k in range(N_DOMAINS)]), dim=1)
    dom_q = (q.cpu() @ cent.T).argmax(1).numpy()
    acc = torch.zeros(N_DOMAINS, N_VOXELS, dtype=torch.float64)
    for s in range(0, len(dom), 4096):
        acc.index_add_(0, torch.as_tensor(dom[s:s + 4096]), d["maps"][tr["map_index"][s:s + 4096]].double())
    dom_n = np.bincount(dom, minlength=N_DOMAINS)
    mean_map = (acc.sum(0) / len(dom)).float().numpy()
    preds["mean_domain"] = (acc / torch.as_tensor(dom_n)[:, None]).float().numpy()[dom_q]
    preds["mean_train"] = np.repeat(mean_map[None], n, 0)

    # atlas labelling: union of the Harvard-Oxford regions named in the title + abstract; ties (inside and outside the
    # union) broken by the global mean map, so a query naming no region gets the global mean map
    parts = atlas_parts(cache)
    hits = [regions_in(t) for t in te["text"]]
    union = np.zeros((n, N_VOXELS), dtype=bool)
    for i, h in enumerate(hits):
        for k in h:
            for code in REGION_RE[k][1]:
                union[i] |= parts[code]
    preds["atlas"] = (union + 0.01 * mean_map / mean_map.max()).astype(np.float32)
    matched = np.array([len(h) > 0 for h in hits])

    # nearest training paper in SPECTER2 space (csv only; retrieval is the subject of comment 3.2)
    nn = (q.cpu() @ t_tr.T).argmax(1)
    preds["nn_text"] = d["maps"][tr["map_index"][nn]].float().numpy()
    print(f"[baselines] domains {dom_n.tolist()}, atlas regions named in {matched.sum()} / {n} queries ({time.perf_counter() - t0:.0f} s)")

    # null: NeuroVLM map of paper i scored against the target of a deranged paper
    null = deranged_indices(n, np.random.default_rng(NULL_SEED))
    frames = []
    jobs = [(k, v, np.arange(n)) for k, v in preds.items()] + [("null_shuffled", preds["neurovlm"], null)]
    for name, pred, pair in jobs:
        sc = score(pred, dec_true, raw, pair)
        frames.append(pd.DataFrame({"method": name, "pmid": te["pmid"], **sc, "in_neuroquery_corpus": in_nq, "in_neurosynth_db": in_ns}))
        print(f"[score] {name:14s} " + "  ".join(f"{k} {v.mean():.4f}" for k, v in sc.items()) + f"  ({time.perf_counter() - t0:.0f} s)")
    per = pd.concat(frames, ignore_index=True)
    args.out.mkdir(parents=True, exist_ok=True)
    per.to_parquet(args.out / "per_query_metrics.parquet", index=False)
    cfg = {"model_dir": str(args.model_dir), "n_test": n, "n_train": len(tr["pmid"]), "n_val": len(va["pmid"]),
           "neurovlm_model": str(args.model_dir / "proj_head_text_mse.pt"), "neurovlm_v1_model": str(head_v1_path) if head_v1 else None,
           "paper_dice_pct": PAPER_PCT, "top_pct": TOP_PCT, "null_seed": NULL_SEED,
           "ridge": {"alpha": alpha, "alphas": list(RIDGE_ALPHAS), "selection": "validation paper Dice",
                     "val": {str(a): v for a, v in ridge_val.items()}},
           "neurovlm_val": val_neurovlm, "neurosynth_median_terms": float(np.median(n_terms)),
           "text2brain": {"weights": f"{MIRROR}/{WEIGHTS}", "input": "title [SEP] abstract, '/' removed, one text at a time",
                          "device": args.t2b_device, "torch_threads": torch.get_num_threads(), "torch": torch.__version__,
                          "transformers": __import__("transformers").__version__},
           "test_in_training_corpus": {"neuroquery_corpus": int(in_nq.sum()), "neurosynth_v7": int(in_ns.sum()),
                                       "either": int((in_nq | in_ns).sum()), "neither": int((~in_nq & ~in_ns).sum()),
                                       "rule": "PMID match; the NeuroQuery corpus is NeuroQuery's training set and the pool "
                                               "Text2Brain's training split was drawn from (split not released)"},
           "domains": {"rule": "k-means on training-map latents; query -> nearest domain centroid of SPECTER2 embeddings",
                       "k": N_DOMAINS, "seed": KMEANS_SEED, "train_sizes": dom_n.tolist(),
                       "test_sizes": np.bincount(dom_q, minlength=N_DOMAINS).tolist()},
           "atlas": {"atlas": "Harvard-Oxford maxprob-thr25 2 mm, cortical + subcortical", "n_matched": int(matched.sum()),
                     "frac_matched": float(matched.mean()), "median_regions_if_matched": float(np.median([len(h) for h in hits if h])),
                     "median_union_voxels_if_matched": float(np.median(union[matched].sum(1))),
                     "paper_dice_matched": {m: float(per.loc[per.method == m, "paper_dice"].to_numpy()[matched].mean()) for m in ("atlas", "neurovlm")},
                     "region_counts": {REGIONS[k][0]: int(c) for k, c in zip(*np.unique(np.concatenate([h for h in hits if h]), return_counts=True))}},
           "wall_time_s": round(time.perf_counter() - t0, 1)}
    (args.out / "run_config.json").write_text(json.dumps(cfg, indent=2))
    print(json.dumps({k: cfg[k] for k in ("ridge", "neurovlm_val", "wall_time_s")}, indent=1))


if __name__ == "__main__":
    main()
