#!/usr/bin/env python
"""Stage 11: synthetic anatomical (text, brain-map) pairs for the NeuroAdapter (port of notebook 03_models/11).

    .venv/bin/python scripts/models/11_anatomical_atlases.py --run-name <run> --inputs-from <run with pmids_split.pt>

Uses training-split papers only (--splits all = the notebook: all papers, outputs suffixed _allsplits). The split is the
run's pmids_split.pt (08), else the split flags of the pinned HF publications.parquet. Reads the external LLM-written
synth.parquet pinned on HF (neurovlm/synthetic_pairs) unless --synth-table. --install copies the outputs into data_dir
under their own names (synth_text.parquet, never over the pinned input synth.parquet).
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/models/
from _common_models import *  # noqa: E402,F403

STAGE = "11_anatomical_atlases"
log = get_logger(STAGE)

SEED = 17
MIN_DF_CANDIDATES = 25                                # cell 8
MIN_DF, MAX_DF_FRAC, TOP_K_NGRAMS = 20, 0.2, 50_000   # cell 10
ATLAS_FWHM = 8                                        # cell 15
COMPOSE_TOP_K = 10                                    # cells 22/23
N_SMOKE_DOCS = 2000
N_PUBMED_MAPS = 30784  # pubmed_images rows (all splits), the --splits all reference for the smoke scaling

# cell 20 stop-list: drop n-gram labels matching any of STOP_PATTERNS or equal to one of STOP_WORDS
STOP_PATTERNS = [
    "the", "during", "appied", "with", "activity", "between", " of ", "but", "can", "change", "suggest",
    "exhibited", "finally", "functional mri", "functional connectivity fc", "functional connectivity rsfc",
    "functional near", "functional near infrared", "functional near infrared spectroscopy fnirs",
    "functionally defined", "functioning", "functions", "indexed", "induced", "making", "mean", "maps", "loss",
    "aim", "using", "examined", "present", "use", "total", "size", "similar", "which", "who", "will", "still",
    "static", "states", "shown", "sample", "measure", "measured", "measures", "^lobes$", "hemisphere", "five",
    "follow", "evaulate", "^even$", "derived", "creates", "dataset", "cross", "^central$", "centers", "categories",
    "assess", "any", "approach", "revealed", "among", "after", "additionally$", "assessment", "ability", "accurate",
    "studies", "have", "participated", "^based", "estimates", "bodies", "^brain activation$", "^brain areas$",
    "^brain functional$", "^brain networks$", "^brain structure$", "^spontaneous brain$", "when", "unclear",
    "characteristics", "stonger", "years", "under", "baseline", "^cessation$", "cohort", "comparison", "compelling",
    "^delay mid$", "^monetary incentive delay mid$", "influence", "obtaining", "pathway", "resolution", "^second$",
    "^secondary$", "modelled", "^red$", "performs", "stronger", "thus", "^triangular$", "systems", "ultra", "we",
    "three", "^clinical cognitive$", "^neurocognitive$", "consistent", "interaction", "included", "remains",
    "^affect$", "attended", "before", "arising", "determine", "factors", "dots", "matched healthy",
    "fifteen healthy", "first", "generic", "fmri data", "rs fmri", "^state fmri$", "increases", "indicated",
    "order", "^lobe$", "^deep$", "^form$", "^lateral$", "^rat", "^structure$", "^sulcus$", "^gyrus$", "^gyri$",
    "common", "structures", "^cognition$", "^object$", "^objective$", "^objects$",
]
STOP_WORDS = [
    "nuclei", "several", "behavior", "complex", "actions", "pre", "anatomical", "component", "cluster", "clusters",
    "signal", "regulation", "multi", "post", "body", "value", "working", "lobule", "lobules", "error",
]


def cached(path, fn):
    """Load the pickle at ``path`` if it exists, else compute ``fn()`` and store it."""
    if path.exists():
        log.info("cached: %s", path.name)
        with path.open("rb") as fh:
            return pickle.load(fh)
    out = fn()
    with path.open("wb") as fh:
        pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return out


def split_counts(pmids, split):
    """Number of PMIDs in each split ("none": in no split, i.e. > 100 peaks or an all-zero map, see 08)."""
    pm = np.asarray(pmids).astype(np.int64)
    out = {k: int(np.isin(pm, v).sum()) for k, v in split.items()}
    out["none"] = int(len(pm) - sum(out.values()))
    return out


def build_atlas_matrix(masker, *, fwhm):
    """Cell 15: AAL-SPM12 parcels -> smoothed, peak-normalised vectors in masker space."""
    import nibabel as nib
    import torch
    from nilearn.image import load_img, resample_to_img, smooth_img

    atlas = fetch_aal()
    atlas_img = resample_to_img(load_img(atlas.maps), masker.mask_img_, interpolation="nearest", force_resample=True,
                                copy_header=True)
    atlas_data = atlas_img.get_fdata()
    vectors, labels = [], []
    for index, label in zip(atlas.indices, atlas.labels, strict=True):
        region_data = (atlas_data == float(index)).astype(np.float32)
        if not region_data.any():
            continue
        region_img = smooth_img(nib.Nifti1Image(region_data, atlas_img.affine, atlas_img.header), fwhm=fwhm)
        vector = np.asarray(masker.transform(region_img)).reshape(-1).astype(np.float32)
        peak = float(vector.max(initial=0.0))
        if peak <= 0:
            continue
        vectors.append(vector / peak)
        labels.append(str(label))
    return torch.from_numpy(np.stack(vectors)), np.asarray(labels)


def stoplist_filter(terms):
    import pandas as pd

    terms = pd.Series(terms)
    for pat in STOP_PATTERNS:
        terms = terms[~terms.str.contains(pat)]
    for word in STOP_WORDS:
        terms = terms[~terms.str.contains(f"^{word}$")]
    return terms


def compose_images(df_rows, text_series, t, w, asort, images_agg, *, top_k, label):
    """Cells 22/23: per row, sum the top-k weighted parcel maps of every matched term, / 99th percentile, clamp to [0, 1].

    ``df_rows`` and ``text_series`` are selected positionally (synth.parquet has a non-unique index).
    """
    import pandas as pd
    import torch

    text_series = pd.Series(text_series).fillna("").astype(str).str.lower()
    if len(t):
        isin = np.column_stack([text_series.str.contains(term, regex=False, na=False).to_numpy(dtype=bool) for term in t])
    else:
        isin = np.zeros((len(text_series), 0), bool)
    matched = isin.sum(axis=1) > 0
    sub = df_rows.iloc[np.flatnonzero(matched)].reset_index(drop=True)
    isin = isin[matched]
    text = np.array(text_series)[matched].tolist()
    images_gen = torch.zeros((len(sub), images_agg.shape[1]))
    for idx in range(len(sub)):
        terms = np.flatnonzero(isin[idx])
        parcels = np.array([asort[i][:top_k] for i in terms])
        ws = torch.from_numpy(np.array([w[i][asort[i][:top_k]] for i in terms]))
        x = (images_agg[parcels] * ws.unsqueeze(2)).sum(dim=(0, 1))
        images_gen[idx] = (x / x.quantile(0.99)).clamp(0, 1)
    keep = ~images_gen.isnan().any(dim=1)
    images_gen = images_gen[keep]
    text = np.array(text)[keep.numpy()].tolist()
    sub = sub[keep.numpy()].reset_index(drop=True)
    log.info("%s: %d rows with matched terms -> %d finite maps", label, int(matched.sum()), len(sub))
    return sub, images_gen, text


def main(argv=None):
    p = base_parser("03_models/11: synthesise anatomical (text, brain-map) training pairs.", stage=STAGE)
    add_model_args(p)
    p.add_argument("--synth-table", type=Path, default=None, help="External LLM-written synth.parquet (title/description).")
    p.add_argument("--n-jobs", type=int, default=-1, help="joblib workers (-1 = all cores).")
    p.add_argument("--splits", choices=["train", "all"], default="train",
                   help="train: training-split papers only; all: every paper, as the notebook (outputs get an _allsplits suffix).")
    p.add_argument("--split-file", type=Path, default=None,
                   help="pmids_split.pt (default: --inputs-from, run dir, else the pinned HF publications.parquet flags).")
    args = finalize_args(p.parse_args(argv))
    run_dir = resolve_run_dir(args)
    setup_torch(SEED, args.deterministic)
    seed_all(SEED)
    import pandas as pd
    import torch
    import _anatomical_atlases_lib as L
    from neurovlm.data import load_dataset, load_masker

    n_jobs = os.cpu_count() if args.n_jobs in (-1, 0) else args.n_jobs
    sfx = "" if args.splits == "train" else "_allsplits"
    cache = run_dir / f"cache{sfx}"
    cache.mkdir(exist_ok=True)
    meta = dict(stage=STAGE, smoke=args.smoke, run=args.run_name, splits=args.splits)
    outputs = {name: run_dir / name.replace(".", f"{sfx}.", 1) for name in (
        "df_region_terms.parquet", "images_agg.pt", "images_synth.pt", "synth_text.parquet", "anatomical_atlases.lineage.json")}
    if artifacts_exist(*outputs.values(), force=args.force):
        log.info("artifacts exist in %s; skipping (use --force to recompute)", run_dir)
        if args.install:
            install_artifacts({"images_synth.pt": outputs["images_synth.pt"], "synth_text.parquet": outputs["synth_text.parquet"]}, log)
        return 0
    # external LLM-generated title/description table, no in-tree producer
    synth_path = args.synth_table or hf_pinned("neurovlm/synthetic_pairs", "synth.parquet")
    source_synth_sha256 = sha256_file(synth_path)
    log.info("synth table: %s", synth_path)
    split, split_source, split_sha256 = pmid_split(run_dir=run_dir, args=args, logger=log, explicit=args.split_file)

    # cell 4: texts + maps
    with timed("load_inputs", log, out=args.out, **meta):
        df_text = load_dataset("pubmed_text")
        images, pmids_all = load_dataset("pubmed_images")
        images = torch.as_tensor(images).float().cpu()
        df_text = df_text[df_text["pmid"].isin(pmids_all)].sort_values("pmid").reset_index(drop=True)
        assert (df_text["pmid"].to_numpy() == np.asarray(pmids_all)).all()
        splits_info = {"mode": args.splits, "split_file": split_source, "split_file_sha256": split_sha256,
                       "pubmed_maps_by_split": split_counts(df_text["pmid"], split)}
        # training-split papers only (the notebook used all of them, leaking val/test maps into images_agg)
        if args.splits == "train":
            pm = df_text["pmid"].to_numpy().astype(np.int64)
            if args.smoke and len(split["train"]) < N_SMOKE_DOCS:
                # 08's smoke split only covers its 64 smoke papers: keep every paper except its val/test ones
                keep = ~np.isin(pm, np.concatenate([split["val"], split["test"]]))
                splits_info["smoke_rule"] = "smoke split too small: every publication except the split's val/test ones"
            else:
                keep = np.isin(pm, split["train"])
            df_text, images = df_text[keep].reset_index(drop=True), images[torch.from_numpy(keep)]
        splits_info["used_by_split"] = split_counts(df_text["pmid"], split)
        log.info("--splits %s (%s): PubMed maps per split %s -> publications used %s", args.splits, split_source,
                 splits_info.get("pubmed_maps_by_split"), splits_info["used_by_split"])
        if args.splits == "train" and (splits_info["used_by_split"]["val"] or splits_info["used_by_split"]["test"] or not len(df_text)):
            raise RuntimeError(f"split filter failed: {splits_info['used_by_split']}")
        n_avail = n_docs = len(df_text)
        if args.smoke and args.max_samples == args.caps.max_samples:
            n_docs = N_SMOKE_DOCS  # the generic 64-row smoke cap is too small for n-gram statistics
        elif args.max_samples:
            n_docs = args.max_samples
        n_docs = min(n_docs, len(df_text))
        if n_docs < len(df_text):
            df_text, images = df_text.iloc[:n_docs].reset_index(drop=True), images[:n_docs]
            splits_info["used_by_split"] = split_counts(df_text["pmid"], split) if split is not None else {"all": len(df_text)}
            log.info("subset (smoke/--max-samples): publications used %s", splits_info["used_by_split"])
        splits_info["n_docs"] = n_docs
        # the df thresholds only scale down for smoke/--max-samples subsets
        scale = n_docs / (n_avail if args.splits == "train" else N_PUBMED_MAPS)
        min_df_cand = MIN_DF_CANDIDATES if scale >= 1 else max(2, round(MIN_DF_CANDIDATES * scale))
        min_df = MIN_DF if scale >= 1 else max(2, round(MIN_DF * scale))
        log.info("documents: %d (min_df candidates %d, min_df %d) | n_jobs %d", n_docs, min_df_cand, min_df, n_jobs)
    key = f"train_n{n_docs}" if args.splits == "train" else f"n{n_docs}"  # cache file tag

    # cell 8: n-gram mining
    abstracts = (df_text["name"] + "." + df_text["description"]).tolist()
    pmids = df_text["pmid"].tolist()
    records = list(zip(pmids, abstracts))
    miner = L.FastNgramMiner(n_min=1, n_max=7)

    def _mine():
        df_rows = miner.count_document_frequency_parallel(records, n_jobs=n_jobs, chunk_docs=20_000)
        candidate_hashes = [row["hash"] for row in df_rows if row["df"] >= min_df_cand]
        examples = miner.collect_phrase_examples_parallel(records, candidate_hashes, n_jobs=n_jobs, chunk_docs=20_000, max_examples_per_hash=3)
        incidence = miner.build_candidate_incidence_parallel(abstracts, candidate_hashes, n_jobs=n_jobs, chunk_docs=5_000)
        return {"df_rows": df_rows, "candidate_hashes": candidate_hashes, "examples": examples, "incidence": incidence}

    with timed("ngram_mining", log, out=args.out, n_docs=n_docs, n_jobs=n_jobs, **meta):
        mined = cached(cache / f"ngram_mining_{key}.pkl", _mine)
        log.info("candidates with df >= %d: %d", min_df_cand, len(mined["candidate_hashes"]))

    # cell 9: spatial-coherence scores
    with timed("coherence_scores", log, out=args.out, **meta):
        scores = cached(cache / f"coherence_scores_{key}.pkl",
                        lambda: L.score_spatial_coherence_from_incidence(incidence=mined["incidence"], maps=images.numpy(),
                                                                         candidate_hashes=mined["candidate_hashes"],
                                                                         candidate_block_size=2048))

    # cell 10: filtered n-gram matrix
    with timed("ngram_matrix", log, out=args.out, **meta):
        labels, X_bool, _ = cached(
            cache / f"ngram_matrix_{key}.pkl",
            lambda: L.make_filtered_ngram_matrix(scores=scores, examples=mined["examples"], incidence=mined["incidence"],
                                                 candidate_hashes=mined["candidate_hashes"], min_df=min_df, max_df_frac=MAX_DF_FRAC,
                                                 top_k=TOP_K_NGRAMS, min_n=1, max_n=7))
        log.info("n-gram features: %d", len(labels))

    # cells 15-16: binarised AAL parcels
    with timed("atlas", log, out=args.out, **meta):
        def _atlas():
            image_t, text_t = build_atlas_matrix(load_masker(), fwhm=ATLAS_FWHM)
            image_b = (image_t > 0.1).long()
            m = image_b.sum(dim=1) != 0
            return image_b[m], text_t[m]

        (run_dir / "cache").mkdir(exist_ok=True)
        image_b, text_b = cached(run_dir / "cache" / "atlas_aal_spm12.pkl", _atlas)  # split-independent, shared
        parcel_names = [f"atlas::{name}" for name in text_b.tolist()]
        log.info("parcels: %d", len(parcel_names))

    # cells 17-18: parcel n-gram statistics + per-parcel aggregate maps
    with timed("parcel_ngrams", log, out=args.out, **meta):
        df_region_terms = L.make_parcel_ngram_dataframe(
            brain_maps=images, atlas_masks=image_b, atlas_parcel_names=parcel_names, labels=labels, X_bool=X_bool, pmids=pmids,
            min_overlap=0.005, top_k_parcels_per_paper=50, top_k_ngrams_per_parcel=50, max_pmids_per_parcel=500)
        df_region_terms.to_parquet(outputs["df_region_terms.parquet"])
        images_agg = torch.zeros((len(df_region_terms), images.shape[1]), dtype=torch.float32)
        for i, parcel_pmids in enumerate(df_region_terms["list_of_pmids"]):
            x = images[torch.from_numpy(df_text["pmid"].isin(parcel_pmids).to_numpy())].sum(dim=0)
            peak = x.max()
            if peak > 0:
                images_agg[i] = x / peak
        torch.save(images_agg, outputs["images_agg.pt"])
        log.info("df_region_terms %s; images_agg %s", df_region_terms.shape, tuple(images_agg.shape))
        pooled = np.unique([pm for x in df_region_terms["list_of_pmids"] for pm in x])
        pooled_by_split = split_counts(pooled, split) if split is not None else {"all": len(pooled)}
        splits_info["pooled_publications_by_split"] = pooled_by_split
        log.info("publications pooled into the parcel maps (unique, by split): %s", pooled_by_split)
        if args.splits == "train" and (pooled_by_split["val"] or pooled_by_split["test"]):
            raise RuntimeError(f"held-out publications in the pooled parcel maps: {pooled_by_split}")

    # cells 20-21: stop-list + term x parcel weights
    with timed("term_weights", log, out=args.out, **meta):
        all_terms = np.unique([j for i in df_region_terms["list_of_ngrams"] for j in i])
        t = all_terms[pd.Series(all_terms).isin(stoplist_filter(all_terms))]
        df_synth = pd.read_parquet(synth_path)
        # w[i, j] = rank_score of term i in parcel j (0 if it is not one of the parcel's n-grams)
        w = np.zeros((len(t), len(df_region_terms)))
        for j, stats in enumerate(df_region_terms["ngram_stats"]):
            rank_score = {s["ngram"]: s["rank_score"] for s in stats}
            for i, term in enumerate(t):
                if term in rank_score:
                    w[i, j] = rank_score[term]
        asort = np.argsort(w)[:, ::-1]
        t = pd.Series(t).fillna("").astype(str).str.lower().to_numpy()
        log.info("terms after stop-list: %d (of %d); synth rows: %d", len(t), len(all_terms), len(df_synth))

    # cells 22-24: compose the synthetic maps; titles/descriptions with "network" match on the full text, the rest on the title
    with timed("compose", log, out=args.out, **meta):
        t_s = (df_synth["title"] + " [SEP] " + df_synth["description"]).str.lower()
        mm = t_s.str.contains("network").to_numpy()  # positional masks: synth.parquet has a non-unique index
        _df_synth, images_gen, text = compose_images(df_synth[mm], t_s[mm], t, w, asort, images_agg, top_k=COMPOSE_TOP_K, label="network rows")
        t_t = df_synth["title"].str.lower()
        mm = ~t_t.str.contains("network").to_numpy()
        _df_synth_t, _images_gen, _text = compose_images(df_synth[mm], t_t[mm], t, w, asort, images_agg, top_k=COMPOSE_TOP_K, label="other rows")
        text = text + _text
        images_gen = torch.vstack((images_gen, _images_gen))
        df_synth_f = pd.concat((_df_synth, _df_synth_t)).reset_index(drop=True)
        df_synth_f["text"] = text

    # cell 25: outputs
    if len(images_gen) == 0:
        raise RuntimeError("No synthetic rows were produced (no synth titles matched the mined terms).")
    torch.save(images_gen, outputs["images_synth.pt"])
    df_synth_f.to_parquet(outputs["synth_text.parquet"])
    lineage = {
        "producer": "scripts/models/11_anatomical_atlases.py (port of docs/03_models/11_anatomical_atlases.ipynb)",
        "seed": SEED, "n_docs": n_docs, "min_df_candidates": min_df_cand, "min_df": min_df, "splits": splits_info,
        "source_synth": str(synth_path), "source_synth_sha256": source_synth_sha256,
        "pubmed_pmids_sha256": hashlib.sha256("\n".join(map(str, pmids)).encode("utf-8")).hexdigest(),
        "outputs": {k: sha256_file(v) for k, v in outputs.items() if k != "anatomical_atlases.lineage.json" and v.exists()},
    }
    outputs["anatomical_atlases.lineage.json"].write_text(json.dumps(lineage, indent=2) + "\n")
    log.info("saved %d aligned examples -> %s, %s", len(df_synth_f), outputs["images_synth.pt"], outputs["synth_text.parquet"])
    write_stage_summary(run_dir, STAGE + sfx, {"splits": splits_info, "n_docs": n_docs, "n_candidates": len(mined["candidate_hashes"]),
                                               "n_features": len(labels), "n_parcels": len(parcel_names), "n_terms": int(len(t)),
                                               "n_synth_rows": int(len(df_synth_f)), "n_jobs": n_jobs, "lineage": lineage})
    if args.install:
        install_artifacts({"images_synth.pt": outputs["images_synth.pt"], "synth_text.parquet": outputs["synth_text.parquet"],
                           "df_region_terms.parquet": outputs["df_region_terms.parquet"], "images_agg.pt": outputs["images_agg.pt"]}, log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
