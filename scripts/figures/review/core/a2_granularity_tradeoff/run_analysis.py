#!/usr/bin/env python
"""Generate vs retrieve, study-specific details and collapse of the brain-to-text captions (reviewer 3.2), final model.
Models: stage1 (grounded), stage2 (canonical projection, the paper) and stage2_raw (no projection) on the PubMed test maps
and one NeuroVault map per paper; PubMed stage1/stage2 captions come from scripts/outputs/validation/final.

    run_analysis.py --device cuda
    run_analysis.py --smoke --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lexicon as LEX  # noqa: E402
from _review_common import (  # noqa: E402
    BASES, DEFAULT_MODEL_DIR, PAPER_TEMP, REPO, BertScore, Timings, component, data_dir, deranged_indices, distinct_n, encode_latents,
    encode_specter, env_packages, get_logger, heldout_pmids, load_minilm, load_neurovault_one_per_doi, load_projector, load_pubmed_latents,
    load_pubmed_maps, load_pubmed_text_rows, load_specter_latents, log_phase, make_nvlm, mean_pairwise_cosine, model_dir, model_sources,
    nearest_cosine, nvlm_generate, read_json, release_cuda_cache, repo_relative, seed_all, select_device, set_model_dir, set_offline_env,
    shared_brain, shared_text, split_pmids, strip_steering_tags, text_to_brain_maps, tokenize_words, write_json,
)

set_offline_env()
log = get_logger("a2_granularity_tradeoff")
VALIDATION_FINAL = REPO / "scripts" / "outputs" / "validation" / "final"
STAGE2_TEXTS = HERE / "outputs" / "cache" / "stage2_training_texts.parquet"
DOMAINS = ("pubmed", "neurovault")
MODELS = {
    "stage1": {"variant": "pubmed", "bases": ("all",), "projection": False},
    "stage2": {"variant": "canonical", "bases": BASES, "projection": True},
    "stage2_raw": {"variant": "canonical", "bases": BASES, "projection": False},
}
VALIDATOR_CACHE = {("pubmed", "stage1"): "df_gen_pubmed.parquet", ("pubmed", "stage2"): "df_gen_canonical.parquet"}
GEN_KW = dict(do_sample=False, num_beams=1, max_new_tokens=128, projection_temp=PAPER_TEMP, repetition_penalty=1.25)

# (query, part A, part B): concepts rarely studied together, so retrieval can only return an existing map for one of them
COMPOSITIONAL_QUERIES = [
    ("Fear conditioning to visual motion stimuli engaging the amygdala and motion-sensitive area MT.",
     "Fear conditioning engaging the amygdala.", "Visual motion processing in motion-sensitive area MT."),
    ("Tactile working memory for vibration frequency in dorsolateral prefrontal cortex and primary somatosensory cortex.",
     "Working memory maintenance in dorsolateral prefrontal cortex.", "Tactile vibration discrimination in primary somatosensory cortex."),
    ("Reward anticipation during sentence comprehension recruiting the ventral striatum and left temporal language cortex.",
     "Monetary reward anticipation in the ventral striatum.", "Sentence comprehension in left superior and middle temporal cortex."),
    ("Perceiving faces of people in pain engages the fusiform face area together with the anterior insula.",
     "Face perception in the fusiform face area.", "Pain processing in the anterior insula."),
    ("Learning finger sequences while navigating a virtual maze engages the cerebellum and the hippocampus.",
     "Motor sequence learning in the cerebellum.", "Spatial navigation in the hippocampus."),
    ("Odor-evoked autobiographical memory retrieval in piriform cortex and hippocampus.",
     "Olfactory processing in piriform cortex.", "Autobiographical memory retrieval in the hippocampus."),
    ("Reasoning about other people's beliefs about numbers engages the temporoparietal junction and intraparietal sulcus.",
     "Theory of mind in the temporoparietal junction.", "Numerical magnitude processing in the intraparietal sulcus."),
    ("Speech production while attending to one's own heartbeat engages Broca's area and the insula.",
     "Overt speech production in Broca's area.", "Interoceptive attention to heartbeat in the insula."),
    ("Saccadic eye movements toward sound sources engage the frontal eye fields and planum temporale.",
     "Saccadic eye movements in the frontal eye fields.", "Auditory spatial localization in the planum temporale."),
    ("Reading words while tapping the fingers engages the visual word form area and primary motor cortex.",
     "Visual word recognition in the left fusiform gyrus.", "Finger tapping in primary motor cortex."),
]

# LLM judge: used only if it gets all 10 obvious cases right (5 with study-specific facts, 5 pure map descriptions)
JUDGE_PROMPT = ("A brain activation map cannot reveal study-specific facts such as the number or age of participants, patient groups, "
                "the experimental task, stimuli, scanner settings or statistics. Answer Yes if the description states such a "
                "study-specific fact, otherwise No.\n\n"
                "Description: The visual network spans occipital cortex and supports the perception of form and motion.\nAnswer: No\n\n"
                "Description: Sixteen healthy volunteers performed a working memory task in a 3T scanner (p < 0.001).\nAnswer: Yes\n\n"
                "Description: {text}\nAnswer:")
JUDGE_TESTS = [
    ("Twenty-four right-handed participants (mean age 23 years) were scanned at 3 Tesla while performing a Stroop task.", True),
    ("Patients with schizophrenia showed significantly reduced activation compared with 20 healthy controls (FDR p < 0.05).", True),
    ("In this study we recruited 30 children aged 8-12 years who viewed 120 pictures from the IAPS set.", True),
    ("We found that older adults showed greater prefrontal activity during a 2-back task, t(28) = 3.1.", True),
    ("Here we report data from 45 patients with major depressive disorder scanned on a Siemens Trio scanner (TR = 2 s).", True),
    ("The dorsal attention network includes the frontal eye fields and intraparietal sulcus and supports goal-directed attention.", False),
    ("The hippocampus lies in the medial temporal lobe and is associated with episodic memory and spatial navigation.", False),
    ("Primary motor cortex in the precentral gyrus supports the execution of voluntary movements.", False),
    ("The default mode network links medial prefrontal cortex, posterior cingulate cortex and angular gyrus; it supports self-referential thought.", False),
    ("Language processing involves left inferior frontal gyrus and superior temporal cortex and supports comprehension and production.", False),
]


def load_domain(name: str, max_n: int | None) -> dict:
    """Evaluated maps of one domain (exclusions applied before the ``max_n`` truncation)."""
    if name == "pubmed":
        pmids, excl = heldout_pmids("test", exclusions=True, log=log.info)
        if max_n:
            pmids = pmids[:max_n]
        rows = load_pubmed_text_rows(pmids)
        return {"X": load_pubmed_maps(pmids), "ids": [str(p) for p in pmids], "pmids": pmids,
                "truth": [f"{n} [SEP] {s}" for n, s in zip(rows["name"], rows["summary"])],
                "truth_nl": [f"{n}\n{s}" for n, s in zip(rows["name"], rows["summary"])],
                "background": [f"{n} [SEP] {d}" for n, d in zip(rows["name"], rows["description"])], "exclusions": excl}
    d = load_neurovault_one_per_doi(max_n, log=log.info)
    return {"X": d["X"], "ids": d["ids"], "pmids": None, "truth": d["truth"], "truth_nl": [t.replace(" [SEP] ", "\n", 1) for t in d["truth"]],
            "background": d["truth"], "exclusions": d["exclusions"]}


def postprocess(df, model: str):
    """As validate_models.eval_qformer: strip the [LABEL] tags, collapse blank lines, join the bases with [SEP]."""
    if MODELS[model]["variant"] == "canonical":
        for b in BASES:
            df[b] = df[b].astype(str).map(strip_steering_tags).str.replace("\n\n", "\n", regex=True)
        df["text"] = df["network"] + " [SEP] " + df["region"] + " [SEP] " + df["function"]
    else:
        df["text"] = df["all"].astype(str).str.replace("\n\n", "\n", regex=True)
    return df


def generate(nvlm, projector, model: str, X, ids, *, gen_batch: int, max_new_tokens: int, tag: str):
    import pandas as pd

    spec = MODELS[model]
    outs = {b: [] for b in spec["bases"]}
    kw = {**GEN_KW, "max_new_tokens": max_new_tokens}
    t0 = time.perf_counter()
    for start in range(0, len(X), gen_batch):
        xb = X[start:start + gen_batch]
        for b in spec["bases"]:
            if spec["variant"] == "pubmed":
                g = nvlm_generate(nvlm, projector, "pubmed", xb, basis=None, prefix_text="", use_canonical_projection=False, seed=123 + start, **kw)
            else:
                g = nvlm_generate(nvlm, projector, "canonical", xb, basis=b, use_canonical_projection=spec["projection"], seed=123 + start, **kw)
            outs[b].extend(g)
        log.info("%s generated %d/%d (%.0f s)", tag, min(start + gen_batch, len(X)), len(X), time.perf_counter() - t0)
    return postprocess(pd.DataFrame({"id": ids, **outs}), model), time.perf_counter() - t0


def generations(dom: str, data: dict, *, out: Path, nvlm_fn, projectors: dict, args, prev_info: dict | None):
    """``{model: DataFrame(id, bases..., text)}`` and their sources: cached csv, else validator parquet, else generate."""
    import pandas as pd

    res, info = {}, {}
    for model in MODELS:
        path = out / f"generated_{dom}_{model}.csv"
        df, src = None, None
        if path.exists() and not args.force:
            d = pd.read_csv(path, keep_default_na=False, dtype={"id": str})
            if list(d["id"]) == data["ids"]:
                df, src = d, (prev_info or {}).get(model) or f"cached {path.name}"
        if df is None and (dom, model) in VALIDATOR_CACHE and not args.smoke:
            v = pd.read_parquet(VALIDATION_FINAL / VALIDATOR_CACHE[(dom, model)])
            if list(v["pmid"].astype(str)) == data["ids"]:
                df = v.rename(columns={"pmid": "id"})
                df = df[["id", *MODELS[model]["bases"], "text"]].copy()
                src = f"validator cache {repo_relative(VALIDATION_FINAL / VALIDATOR_CACHE[(dom, model)])} (generation batch as recorded in validation/final)"
        if df is None:
            with log_phase(log, f"{dom}/{model} generation", n=len(data["ids"]), gen_batch=args.gen_batch):
                df, sec = generate(nvlm_fn(), projectors[MODELS[model]["variant"]], model, data["X"], data["ids"], gen_batch=args.gen_batch,
                                   max_new_tokens=args.max_new_tokens, tag=f"{dom}/{model}")
            src = f"generated here (batch {args.gen_batch}, {sec:.0f} s)"
        df["truth"] = data["truth"]
        df.to_csv(path, index=False)
        res[model], info[model] = df, src
        log.info("%s/%s: %s", dom, model, src)
    return res, info


def batch_size_check(nvlm, projector, data, cached, *, n: int, gen_batch: int) -> dict:
    """bf16 greedy decoding depends on the batch size: regenerate ``n`` cached PubMed stage-2 captions at ``gen_batch``."""
    df, _ = generate(nvlm, projector, "stage2", data["X"][:n], data["ids"][:n], gen_batch=gen_batch, max_new_tokens=GEN_KW["max_new_tokens"],
                     tag="batch-size check")
    a, b = df["text"].tolist(), cached["text"].tolist()[:n]
    la, lb = LEX.count_frame(a), LEX.count_frame(b)
    return {"n": n, "gen_batch": gen_batch, "identical_frac": float(np.mean([x == y for x, y in zip(a, b)])),
            "lexicon_any_regenerated": float(la["any"].mean()), "lexicon_any_cached": float(lb["any"].mean()),
            "n_words_regenerated": float(np.mean([len(tokenize_words(t)) for t in a])), "n_words_cached": float(np.mean([len(tokenize_words(t)) for t in b])),
            "texts_regenerated": a[:3], "texts_cached": b[:3]}


def training_bank(*, text_head, image_head, device, cache: Path, max_train: int | None, seed: int, force: bool = False) -> dict:
    """PubMed training split: shared-space text and map embeddings and captions (title [SEP] summary), cached."""
    import torch
    import torch.nn.functional as F

    if cache.exists() and not force:
        return torch.load(cache, weights_only=False)
    lat, pm = load_pubmed_latents()
    train = set(split_pmids()["train"].tolist())
    idx = np.flatnonzero([int(p) in train for p in pm])
    if max_train and len(idx) > max_train:
        idx = np.sort(np.random.default_rng(seed).choice(idx, size=max_train, replace=False))
    pmids = pm[idx]
    rows = load_pubmed_text_rows(pmids)
    spec = F.normalize(load_specter_latents(pmids), dim=1)
    bank = {"pmids": pmids, "text_shared": shared_text(text_head, spec, device), "brain_latent": lat[idx].clone(),
            "brain_shared": shared_brain(image_head, lat[idx], device),
            "caption": [f"{n} [SEP] {s}" for n, s in zip(rows["name"], rows["summary"])],
            "caption_nl": [f"{n}\n{s}" for n, s in zip(rows["name"], rows["summary"])]}
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, cache)
    return bank


def stage2_corpus() -> list[str]:
    """Stage-2 training texts ("Title\\nbody", before 14's top-50 filter), cached: build_text_table of
    scripts/models/14_qformer_anat.py on the pinned text_synth_less.parquet (data dir, scripts/data/fetch_hf.py)."""
    import importlib.util

    import pandas as pd
    from _hf_pins import _manifest, _sha256

    if not STAGE2_TEXTS.exists():
        e = next(e for e in _manifest()["files"] if e["repo"] == "neurovlm/synthetic_pairs" and e["path"] == "text_synth_less.parquet")
        tsl = data_dir() / e["local_name"]
        if _sha256(str(tsl), tsl.stat().st_size, tsl.stat().st_mtime) != e["sha256"]:
            raise RuntimeError(f"{tsl} is not the pinned text_synth_less.parquet (run scripts/data/fetch_hf.py)")
        spec = importlib.util.spec_from_file_location("stage14", REPO / "scripts" / "models" / "14_qformer_anat.py")
        stage14 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stage14)
        df, sep = stage14.build_text_table(tsl)
        body = df["text"].str.replace(".*\n", "", regex=True)
        STAGE2_TEXTS.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"label": df["text"].str.split("]").str[0].str.strip("["), "title": df["title"], "body": body, "text_sep": sep,
                      "text": df["title"] + "\n" + body}).to_parquet(STAGE2_TEXTS, index=False)
    return pd.read_parquet(STAGE2_TEXTS)["text"].astype(str).tolist()


def pr_in_training_basis(Z: np.ndarray, V: np.ndarray) -> float:
    """Participation ratio (sum v)^2 / sum v^2 of the variances v of the centred set ``Z`` along the training PCs ``V``."""
    Z = np.asarray(Z, dtype=np.float64)
    P = (Z - Z.mean(axis=0, keepdims=True)) @ V
    v = P.var(axis=0)
    return float(v.sum() ** 2 / max((v ** 2).sum(), 1e-12))


def text_units(df, model: str) -> tuple[list[str], list[str], list[str]]:
    """Texts for the collapse analysis: stage 1 = the caption, stage 2 = each basis text."""
    if MODELS[model]["variant"] == "pubmed":
        return df["id"].tolist(), ["all"] * len(df), df["text"].astype(str).tolist()
    ids, bases, texts = [], [], []
    for b in BASES:
        ids += df["id"].tolist()
        bases += [b] * len(df)
        texts += df[b].astype(str).str.strip().tolist()
    return ids, bases, texts


def wilcoxon_p(a, b) -> float:
    from scipy.stats import wilcoxon

    try:
        return float(wilcoxon(np.asarray(a), np.asarray(b)).pvalue)
    except Exception:
        return float("nan")


def score_domain(dom, data, gens, *, bank, st, specter, text_head, image_head, ae, bert, V, s2_emb, bank_emb_nl, device, seed):
    """All metrics of one domain: every model, the retrieved captions and the real texts."""
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from neurovlm.metrics.semantic import exact_pmid_retrieval_metrics

    n = len(data["ids"])
    truth = data["truth"]
    null = deranged_indices(n, np.random.default_rng(seed))
    with log_phase(log, f"{dom}: map + truth embeddings", n=n):
        z_map = shared_brain(image_head, encode_latents(ae.encoder, data["X"], device), device)
        z_truth = shared_text(text_head, F.normalize(encode_specter(specter, truth), dim=1), device)
    # retrieval: the training caption nearest to the map in the contrastive space
    ret_cos, ret_idx = nearest_cosine(z_map, bank["text_shared"])
    retrieved = [bank["caption"][i] for i in ret_idx]
    gen_texts = {m: g["text"].astype(str).tolist() for m, g in gens.items()}
    with log_phase(log, f"{dom}: MiniLM", texts=n * (len(MODELS) + 2)):
        allt = [t for m in MODELS for t in gen_texts[m]] + retrieved + list(truth)
        e = st.encode(allt, batch_size=128, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        emb = {m: e[k * n:(k + 1) * n] for k, m in enumerate(MODELS)}
        emb["retrieved"], emb["truth"] = e[len(MODELS) * n:(len(MODELS) + 1) * n], e[(len(MODELS) + 1) * n:]
    bert_f1 = None
    if bert is not None:
        with log_phase(log, f"{dom}: BERTScore", pairs=n * (len(MODELS) + 1)):
            cands = [t for m in MODELS for t in gen_texts[m]] + retrieved
            f = bert.f1_many(cands, {"truth": list(truth) * (len(MODELS) + 1)}, desc=f"{dom} BERTScore")
            if f is not None:
                bert_f1 = {m: f["truth"][k * n:(k + 1) * n] for k, m in enumerate([*MODELS, "retrieved"])}
    with log_phase(log, f"{dom}: contrastive text embeddings of the generated / retrieved captions"):
        z_gen = {m: shared_text(text_head, F.normalize(encode_specter(specter, gen_texts[m]), dim=1), device) for m in MODELS}
        z_gen["retrieved"] = bank["text_shared"][torch.as_tensor(ret_idx)]
    # collapse bank: both training corpora (MiniLM; PubMed captions joined with "\n")
    union = np.concatenate([bank_emb_nl, s2_emb])
    n_pub = len(bank_emb_nl)
    per, summary, collapse = {}, {}, []
    for m in [*MODELS, "retrieved"]:
        texts = gen_texts[m] if m in MODELS else retrieved
        p = pd.DataFrame({"id": data["ids"], "text": texts, "n_words": [len(tokenize_words(t)) for t in texts]})
        lex = LEX.count_frame(texts)
        for c in [*LEX.CATEGORIES, "total", "any"]:
            p[f"lex_{c}"] = lex[c].to_numpy()
        p["minilm_cos_truth"] = np.einsum("ij,ij->i", emb[m], emb["truth"])
        p["minilm_cos_truth_null"] = np.einsum("ij,ij->i", emb[m], emb["truth"][null])
        p["contrastive_text_cos_truth"] = (z_gen[m] * z_truth).sum(dim=1).numpy()
        if bert_f1 is not None:
            p["bertscore_f1_truth"] = bert_f1[m]
        scores = z_gen[m] @ z_map.T
        p["contrastive_cos_map"] = scores.diag().numpy()
        p["contrastive_cos_map_null"] = scores[torch.arange(n), torch.as_tensor(null)].numpy()
        if m == "retrieved":
            p["retrieved_pmid"] = bank["pmids"][ret_idx].astype(np.int64)
            p["retrieval_cos_map_text"] = ret_cos
        per[m] = p
        s = {"n": n, "n_words_mean": float(p["n_words"].mean()), "lex_any_frac": float(p["lex_any"].mean()), "lex_total_mean": float(p["lex_total"].mean()),
             "lex_total_per_100_words": float(100 * p["lex_total"].sum() / max(p["n_words"].sum(), 1)),
             **{f"lex_{c}_frac": float((p[f"lex_{c}"] > 0).mean()) for c in LEX.CATEGORIES},
             **{f"lex_{c}_mean": float(p[f"lex_{c}"].mean()) for c in LEX.CATEGORIES},
             "minilm_cos_truth_mean": float(p["minilm_cos_truth"].mean()), "minilm_cos_truth_median": float(p["minilm_cos_truth"].median()),
             "minilm_cos_truth_null_mean": float(p["minilm_cos_truth_null"].mean()),
             "contrastive_text_cos_truth_mean": float(p["contrastive_text_cos_truth"].mean()),
             "contrastive_cos_map_mean": float(p["contrastive_cos_map"].mean()), "contrastive_cos_map_null_mean": float(p["contrastive_cos_map_null"].mean()),
             "recall_auc": float(exact_pmid_retrieval_metrics(scores)["normalized_k_recall_curve_auc"]) if n >= 2 else float("nan"),
             "pairwise_minilm_cos": mean_pairwise_cosine(emb[m]), "distinct_1": distinct_n(texts, 1), "distinct_2": distinct_n(texts, 2),
             "unique_frac_text": len(set(texts)) / max(n, 1), "pr_training_pca": pr_in_training_basis(z_gen[m].numpy(), V)}
        if bert_f1 is not None:
            s["bertscore_f1_truth_mean"] = float(np.mean(bert_f1[m]))
        if m in MODELS:
            for b in MODELS[m]["bases"]:
                s[f"unique_frac_{b}"] = float(gens[m][b].astype(str).nunique() / max(n, 1))
            ids, bases, units = text_units(gens[m], m)
            ue = st.encode(units, batch_size=128, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            nn_cos, nn_idx = nearest_cosine(ue, union)
            collapse.append(pd.DataFrame({"model": m, "id": ids, "basis": bases, "nn_cos": nn_cos, "nn_source": np.where(nn_idx < n_pub, "pubmed_train", "stage2_texts"),
                                          "nn_index": nn_idx}))
            s.update({"collapse_nn_cos_median": float(np.median(nn_cos)), "collapse_nn_cos_mean": float(np.mean(nn_cos)),
                      "collapse_near_copy_frac_095": float((nn_cos >= 0.95).mean()), "collapse_nn_in_stage2_frac": float((nn_idx >= n_pub).mean()),
                      "collapse_distinct_nn_frac": float(len(np.unique(nn_idx)) / max(len(nn_idx), 1)), "collapse_n_units": int(len(nn_idx))})
        summary[m] = s
    # retrieval vs generation (paired, same maps)
    for m in MODELS:
        d = per[m]["minilm_cos_truth"].to_numpy() - per["retrieved"]["minilm_cos_truth"].to_numpy()
        summary[m]["gen_minus_retrieved_minilm_mean"] = float(d.mean())
        summary[m]["gen_gt_retrieved_minilm_frac"] = float((d > 0).mean())
        summary[m]["gen_vs_retrieved_minilm_wilcoxon_p"] = wilcoxon_p(per[m]["minilm_cos_truth"], per["retrieved"]["minilm_cos_truth"])
        if bert_f1 is not None:
            summary[m]["gen_minus_retrieved_bertscore_mean"] = float(np.mean(bert_f1[m] - bert_f1["retrieved"]))
            summary[m]["gen_vs_retrieved_bertscore_wilcoxon_p"] = wilcoxon_p(bert_f1[m], bert_f1["retrieved"])
    # real held-out texts: background lexicon, collapse reference, participation ratio
    bg = LEX.count_frame(data["background"])
    re_ = st.encode(data["truth_nl"], batch_size=128, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    rn_cos, rn_idx = nearest_cosine(re_, union)
    collapse.append(pd.DataFrame({"model": "real", "id": data["ids"], "basis": "all", "nn_cos": rn_cos,
                                  "nn_source": np.where(rn_idx < n_pub, "pubmed_train", "stage2_texts"), "nn_index": rn_idx}))
    if data["pmids"] is not None:
        z_real = shared_text(text_head, F.normalize(load_specter_latents(data["pmids"]), dim=1), device)
    else:
        z_real = shared_text(text_head, F.normalize(encode_specter(specter, data["background"]), dim=1), device)
    summary["real"] = {"n": n, "lex_any_frac": float(bg["any"].mean()), "lex_total_mean": float(bg["total"].mean()),
                       "lex_total_per_100_words": float(100 * bg["total"].sum() / max(sum(len(tokenize_words(t)) for t in data["background"]), 1)),
                       **{f"lex_{c}_frac": float((bg[c] > 0).mean()) for c in LEX.CATEGORIES}, **{f"lex_{c}_mean": float(bg[c].mean()) for c in LEX.CATEGORIES},
                       "collapse_nn_cos_median": float(np.median(rn_cos)), "collapse_nn_cos_mean": float(np.mean(rn_cos)),
                       "collapse_near_copy_frac_095": float((rn_cos >= 0.95).mean()), "pr_training_pca": pr_in_training_basis(z_real.numpy(), V),
                       "pairwise_minilm_cos": mean_pairwise_cosine(emb["truth"]), "note": "lexicon on title [SEP] abstract; collapse on the reference text"}
    top = {m: LEX.top_terms(gen_texts[m]) for m in MODELS}
    top["real"] = LEX.top_terms(data["background"])
    return per, summary, pd.concat(collapse, ignore_index=True), top


def gen_vs_retrieval(*, specter, text_head, mse_head, ae, bank, device, n_queries: int | None):
    """Text-to-brain: generated map of a compositional query vs the training map retrieved for it, each compared with the
    generated maps of its two parts."""
    import pandas as pd
    import torch.nn.functional as F

    queries = COMPOSITIONAL_QUERIES[: n_queries or len(COMPOSITIONAL_QUERIES)]
    nq = len(queries)
    e = F.normalize(encode_specter(specter, [q for q, _, _ in queries] + [a for _, a, _ in queries] + [b for _, _, b in queries]), dim=1)
    maps = text_to_brain_maps(e, mse_head, ae.decoder, device)
    gen_full, gen_a, gen_b = maps[:nq], maps[nq:2 * nq], maps[2 * nq:]
    z_q = shared_text(text_head, e[:nq], device)
    ret_cos, ret_idx = (z_q @ bank["brain_shared"].T).max(dim=1)
    ret_pmids = bank["pmids"][ret_idx.numpy()]
    ret_maps = load_pubmed_maps(ret_pmids)

    def pearson_rows(a, b):
        a = a - a.mean(dim=1, keepdim=True)
        b = b - b.mean(dim=1, keepdim=True)
        return (F.normalize(a, dim=1) * F.normalize(b, dim=1)).sum(dim=1).numpy()

    rows = pd.DataFrame({"query": [q for q, _, _ in queries], "part_a": [a for _, a, _ in queries], "part_b": [b for _, _, b in queries],
                         "retrieved_pmid": ret_pmids.astype(np.int64), "cos_query_retrieved_map": ret_cos.numpy(),
                         "pearson_generated_vs_part_a": pearson_rows(gen_full, gen_a), "pearson_generated_vs_part_b": pearson_rows(gen_full, gen_b),
                         "pearson_retrieved_vs_part_a": pearson_rows(ret_maps, gen_a), "pearson_retrieved_vs_part_b": pearson_rows(ret_maps, gen_b)})
    rows["composition_min_generated"] = rows[["pearson_generated_vs_part_a", "pearson_generated_vs_part_b"]].min(axis=1)
    rows["composition_min_retrieved"] = rows[["pearson_retrieved_vs_part_a", "pearson_retrieved_vs_part_b"]].min(axis=1)
    return rows


def llm_judge(device, per_pubmed: dict, *, n: int, seed: int, repo: str = "Qwen/Qwen3-0.6B-Base") -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from _hf_pins import hf_revision

    try:
        rev = hf_revision(repo)  # pinned in scripts/hf_manifest.json
        tok = AutoTokenizer.from_pretrained(repo, revision=rev)
        lm = AutoModelForCausalLM.from_pretrained(repo, revision=rev, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32).to(device).eval()
    except Exception as exc:  # not cached
        return {"used": False, "reason": f"{repo} not loadable offline: {type(exc).__name__}: {exc}"}
    yes, no = tok(" Yes", add_special_tokens=False)["input_ids"][0], tok(" No", add_special_tokens=False)["input_ids"][0]

    def p_yes(texts):
        out = []
        with torch.no_grad():
            for t in texts:
                ids = tok(JUDGE_PROMPT.format(text=str(t).replace("\n", " ")[:2000]), return_tensors="pt").to(device)
                logits = lm(**ids).logits[0, -1].float()
                out.append(float(torch.softmax(logits[[yes, no]], dim=0)[0]))
        return np.asarray(out)

    py = p_yes([t for t, _ in JUDGE_TESTS])
    truth = np.array([y for _, y in JUDGE_TESTS])
    correct = int(((py > 0.5) == truth).sum())
    res = {"model": repo, "test_cases": [{"text": t, "expected": bool(y), "p_yes": float(p)} for (t, y), p in zip(JUDGE_TESTS, py)],
           "test_correct": correct, "test_n": len(JUDGE_TESTS)}
    if correct < len(JUDGE_TESTS):
        res.update(used=False, reason=f"unreliable: {correct}/{len(JUDGE_TESTS)} obvious test cases correct (all 10 required); spot check skipped")
    else:
        res["used"] = True
        rng = np.random.default_rng(seed)
        for m, p in per_pubmed.items():
            idx = rng.choice(len(p), size=min(n, len(p)), replace=False)
            sub = p.iloc[idx]
            j = p_yes(sub["text"].tolist()) > 0.5
            lexf = sub["lex_any"].to_numpy(dtype=bool)
            res[m] = {"n": int(len(idx)), "judge_rate": float(j.mean()), "lexicon_rate_same_captions": float(lexf.mean()),
                      "agreement": float((j == lexf).mean()), "ids": sub["id"].tolist(), "judge_yes": j.tolist()}
    del lm
    release_cuda_cache()
    return res


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=HERE / "outputs")
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Run directory with the models (default: runs/final).")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--domains", default=",".join(DOMAINS), help="Comma list of domains.")
    p.add_argument("--max-samples", type=int, default=None, help="Cap maps per domain (default all).")
    p.add_argument("--max-train", type=int, default=None, help="Cap PubMed training captions in the retrieval / collapse bank (default all).")
    p.add_argument("--gen-batch", type=int, default=128, help="Generation batch (<= 128).")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--bert-model", choices=["deberta", "roberta", "none"], default="deberta", help="BERTScore model.")
    p.add_argument("--batch-check", type=int, default=256, help="Captions regenerated to measure the batch-size effect (0 = off).")
    p.add_argument("--judge-n", type=int, default=60, help="Captions per model for the LLM-judge spot check.")
    p.add_argument("--no-judge", action="store_true", help="Skip the LLM judge.")
    p.add_argument("--smoke", action="store_true", help="4 maps per domain, 12 tokens, 2,000 training captions (CPU).")
    p.add_argument("--force", action="store_true", help="Regenerate cached outputs.")
    args = p.parse_args(argv)
    if args.gen_batch > 128:
        raise SystemExit("--gen-batch must be <= 128")
    if args.smoke:
        args.max_samples, args.max_new_tokens, args.max_train = args.max_samples or 4, 12, args.max_train or 2000
        args.bert_model, args.batch_check, args.no_judge = "none", 0, True
    out = args.out
    (out / "cache").mkdir(parents=True, exist_ok=True)
    timings = Timings(out / "timings.json")
    seed_all(args.seed)
    import pandas as pd

    device = select_device(args.device)
    set_model_dir(args.model_dir)
    log.info("device %s | model dir %s | domains %s | max/domain %s | smoke %s", device, model_dir(), args.domains, args.max_samples, args.smoke)
    t_start = time.perf_counter()
    with timings.time("load_models"), log_phase(log, "load models"):
        ae = component("autoencoder", device)
        text_head, image_head = component("proj_head_text_infonce", device), component("proj_head_image_infonce", device)
        mse_head = component("proj_head_text_mse", device)
        specter = component("specter", device)
        projectors = {"pubmed": load_projector("pubmed", device), "canonical": load_projector("canonical", device)}
    state: dict = {}

    def nvlm_fn():
        if "nvlm" not in state:
            state["nvlm"] = make_nvlm(device, ae=ae, text_head=text_head, image_head=image_head, mse_head=mse_head)
        return state["nvlm"]

    # generation first, then free the CUDA cache for the scorers
    domains = [d for d in args.domains.split(",") if d]
    prev = read_json(out / "metrics.json") if (out / "metrics.json").exists() and not args.force else {}
    data, gens, gen_info, exclusions = {}, {}, {}, {}
    for dom in domains:
        with timings.time(f"load_{dom}"):
            data[dom] = load_domain(dom, args.max_samples)
        exclusions[dom] = {k: v for k, v in data[dom]["exclusions"].items() if k != "dropped_dois"}
        with timings.time(f"generate_{dom}"):
            gens[dom], gen_info[dom] = generations(dom, data[dom], out=out, nvlm_fn=nvlm_fn, projectors=projectors, args=args,
                                                   prev_info=prev.get("generation_sources", {}).get(dom))
    bcheck = prev.get("batch_size_check")
    if args.batch_check and "pubmed" in domains and not bcheck and "validator" in gen_info["pubmed"]["stage2"]:
        with timings.time("batch_size_check"), log_phase(log, "batch-size check", n=args.batch_check):
            bcheck = batch_size_check(nvlm_fn(), projectors["canonical"], data["pubmed"], gens["pubmed"]["stage2"], n=args.batch_check,
                                      gen_batch=args.gen_batch)
    state.pop("nvlm", None)
    log.info("released CUDA cache after generation (%s)", release_cuda_cache())

    with timings.time("training_bank"), log_phase(log, "training bank"):
        bank = training_bank(text_head=text_head, image_head=image_head, device=device, max_train=args.max_train, seed=args.seed, force=args.force,
                             cache=out / "cache" / f"train_bank_{model_dir().name}{'_n' + str(args.max_train) if args.max_train else ''}.pt")
        Ztr = bank["text_shared"].numpy().astype(np.float64)
        Ztr = Ztr - Ztr.mean(axis=0, keepdims=True)
        lam, V = np.linalg.eigh(Ztr.T @ Ztr / len(Ztr))
        lam, V = lam[::-1], V[:, ::-1]
        st = load_minilm(device)
        s2_texts = stage2_corpus()
        s2_emb = st.encode(s2_texts, batch_size=128, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        bank_emb_nl = st.encode(bank["caption_nl"], batch_size=256, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    bert = BertScore(args.bert_model, device, log) if args.bert_model != "none" else None
    results, top_terms = {}, {}
    lex_rows = []
    for dom in domains:
        with timings.time(f"score_{dom}"), log_phase(log, f"{dom}: scoring"):
            per, summ, coll, top = score_domain(dom, data[dom], gens[dom], bank=bank, st=st, specter=specter, text_head=text_head, image_head=image_head,
                                                ae=ae, bert=bert, V=V, s2_emb=s2_emb, bank_emb_nl=bank_emb_nl, device=device, seed=args.seed)
        for m, d in per.items():
            d.to_parquet(out / f"per_text_{dom}_{m}.parquet", index=False)
        coll.to_parquet(out / f"collapse_{dom}.parquet", index=False)
        results[dom], top_terms[dom] = summ, top
        for m, s in summ.items():
            if m == "retrieved":
                continue
            for c in [*LEX.CATEGORIES, "total"]:
                lex_rows.append({"domain": dom, "model": m, "category": c, "frac_with_element": s[f"lex_{c}_frac"] if c != "total" else s["lex_any_frac"],
                                 "mean_count": s[f"lex_{c}_mean"] if c != "total" else s["lex_total_mean"], "n": s["n"]})
    pd.DataFrame(lex_rows).to_csv(out / "lexicon_summary.csv", index=False)
    (out / "lexicon_top_terms.json").write_text(json.dumps(top_terms, indent=1))

    with timings.time("gen_vs_retrieval"), log_phase(log, "text-to-brain: 10 compositional queries"):
        gvr = gen_vs_retrieval(specter=specter, text_head=text_head, mse_head=mse_head, ae=ae, bank=bank, device=device,
                               n_queries=3 if args.smoke else None)
    gvr.to_csv(out / "gen_vs_retrieval.csv", index=False)
    judge = {"used": False, "reason": "--no-judge"}
    if not args.no_judge and "pubmed" in domains:
        with timings.time("llm_judge"), log_phase(log, "LLM judge test + spot check"):
            per_pub = {m: pd.read_parquet(out / f"per_text_pubmed_{m}.parquet") for m in MODELS}
            judge = llm_judge(device, per_pub, n=args.judge_n, seed=args.seed)
    write_json(out / "llm_judge.json", judge)

    cols = ["n", "n_words_mean", "lex_any_frac", "lex_total_mean", "lex_total_per_100_words", "recall_auc", "contrastive_cos_map_mean",
            "contrastive_cos_map_null_mean", "minilm_cos_truth_mean", "bertscore_f1_truth_mean", "contrastive_text_cos_truth_mean",
            "gen_minus_retrieved_minilm_mean", "gen_gt_retrieved_minilm_frac", "gen_minus_retrieved_bertscore_mean", "pairwise_minilm_cos",
            "distinct_1", "distinct_2", "unique_frac_text", "collapse_nn_cos_median", "collapse_near_copy_frac_095", "collapse_nn_in_stage2_frac",
            "collapse_distinct_nn_frac", "pr_training_pca"]
    table = pd.DataFrame([{"domain": d, "model": m, **{c: s.get(c, np.nan) for c in cols}} for d, r in results.items() for m, s in r.items()])
    table.to_csv(out / "table.csv", index=False)
    write_json(out / "metrics.json", {
        "results": results, "generation_sources": gen_info, "batch_size_check": bcheck, "llm_judge_used": judge.get("used"),
        "gen_vs_retrieval": {"n_queries": len(gvr), "composition_min_generated_mean": float(gvr["composition_min_generated"].mean()),
                             "composition_min_retrieved_mean": float(gvr["composition_min_retrieved"].mean())},
        "exclusions": exclusions, "model_dir": repo_relative(model_dir()), "model_sources": model_sources(),
        "decoding": {**GEN_KW, "max_new_tokens": args.max_new_tokens, "gen_batch_new": args.gen_batch},
        "pca": {"fitted_on": f"contrastive-space embeddings of the {len(bank['pmids'])} PubMed training captions (precomputed SPECTER2 -> run text head)",
                "explained_top10": float(lam[:10].sum() / lam.sum()), "pr_training_set": float(lam.sum() ** 2 / (lam ** 2).sum())},
        "collapse_bank": {"pubmed_train_captions": len(bank["caption_nl"]), "stage2_texts": len(s2_texts), "embedding": "all-MiniLM-L6-v2"},
        "lexicon_categories": list(LEX.CATEGORIES), "smoke": args.smoke, "device": str(device), "packages": env_packages(),
        "wall_seconds_this_run": time.perf_counter() - t_start})
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(table.set_index(["domain", "model"]).T.to_string(float_format=lambda v: f"{v:.3f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
