#!/usr/bin/env python
"""Comment 2.m5 (the NeuroQwen3-0.6B pretraining corpus holds 2,906 of the 2,971 evaluated test papers): do the stage-2
captions copy the paper's own abstract?

Stage-2 captions ([NETWORK] + [REGION] + [FUNCTION], joined as in validation) of the 2,971 PubMed test maps come from
scripts/outputs/validation/final/df_gen_canonical.parquet; 2,971 training papers (seed 0; with a map, an abstract and a
PubMed summary) are captioned here with exactly the validation path and recipe (validate_models.Bundle, greedy, 128 new
tokens, projection temperature 0.035, repetition penalty 1.25, batches of 128 in PMID order, seed 123 + batch start).
Each caption is compared with its paper's title + abstract (the NeuroQwen pretraining text; neurovlm/neuro_image_papers
publications.parquet at the pinned revision), with its PubMed summary (the stage-13 training target) and, as a null, with
the abstract / summary of another paper of the same group (derangement, seed 0). Words = rouge_score's tokenizer
(lower-cased alphanumeric runs). Per caption: longest common run of words (LCS substring), share of the caption's word
8-grams found in the reference, ROUGE-L F1. Test vs train: medians, 95th percentiles, two-sided Mann-Whitney U.

Imports scripts/validate_models.py so the training-sample captions use the validation generation path verbatim.
    .venv/bin/python scripts/figures/review/core/a4_memorization_check/run_analysis.py --device cuda   # GPU ~5 min, then CPU ~1 min
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parents[3]
sys.path[:0] = [str(SCRIPTS), str(SCRIPTS / "models")]
from common import hf_pinned, setup_torch  # noqa: E402

VALIDATION_FINAL = SCRIPTS / "outputs" / "validation" / "final"
BASES = ("network", "region", "function")
RECIPE = dict(do_sample=False, num_beams=1, max_new_tokens=128, projection_temp=0.035, repetition_penalty=1.25)  # eval_qformer
NGRAM = 8
METRICS = ("lcs_words", "frac_8gram", "rougeL_f")
REFS = ("abstract", "summary", "abstract_null", "summary_null")


@functools.cache
def _tokenizer():
    from rouge_score.tokenizers import DefaultTokenizer

    return DefaultTokenizer(use_stemmer=False)


def words(text: str) -> list[str]:
    return _tokenizer().tokenize(str(text))


def longest_common_run(a: list[str], b: list[str]) -> tuple[int, tuple]:
    """(length, words) of the longest run of consecutive words shared by a and b (binary search on the run length)."""
    lo, hi, span = 0, min(len(a), len(b)), ()
    while lo < hi:
        k = (lo + hi + 1) // 2
        grams = {tuple(b[i:i + k]) for i in range(len(b) - k + 1)}
        hit = next((tuple(a[i:i + k]) for i in range(len(a) - k + 1) if tuple(a[i:i + k]) in grams), None)
        if hit is None:
            hi = k - 1
        else:
            lo, span = k, hit
    return lo, span


def lcs_len(a: list[str], b: list[str]) -> int:
    """Longest common subsequence length (bit-parallel, Hyyro 2004); equals rouge_score's dynamic programme."""
    if not a or not b:
        return 0
    masks: dict = {}
    for i, t in enumerate(a):
        masks[t] = masks.get(t, 0) | (1 << i)
    full = v = (1 << len(a)) - 1
    for t in b:
        u = v & masks.get(t, 0)
        v = ((v + u) | (v - u)) & full
    return len(a) - bin(v).count("1")


def overlap(cap: list[str], ref: list[str]) -> dict:
    run, span = longest_common_run(cap, ref)
    ref_grams = {tuple(ref[i:i + NGRAM]) for i in range(len(ref) - NGRAM + 1)}
    cap_grams = [tuple(cap[i:i + NGRAM]) for i in range(len(cap) - NGRAM + 1)]
    lcs = lcs_len(cap, ref)
    p, r = (lcs / len(cap) if cap else 0.0), (lcs / len(ref) if ref else 0.0)
    return {"lcs_words": run, "span": " ".join(span), "frac_8gram": float(np.mean([g in ref_grams for g in cap_grams])) if cap_grams else np.nan,
            "rougeL_f": 0.0 if p + r == 0 else 2 * p * r / (p + r)}


def derangement(n: int, rng) -> np.ndarray:
    while True:
        p = rng.permutation(n)
        if not (p == np.arange(n)).any():
            return p


def caption_train_sample(args, pmids: np.ndarray, log) -> pd.DataFrame:
    """Stage-2 captions of ``pmids`` (sorted) with validate_models.eval_qformer's generation loop."""
    import torch
    import validate_models as V
    from _common_models import strip_steering_tags

    cache = args.out / "cache" / f"train_captions_n{len(pmids)}_seed{args.seed}_b{args.gen_batch}.parquet"
    if cache.exists() and not args.force:
        df = pd.read_parquet(cache)
        if df["pmid"].astype(str).tolist() == [str(p) for p in pmids]:
            log(f"cached training-sample captions {cache.name}")
            return df
    device = torch.device(args.device)
    bundle = V.Bundle(args.model_dir, device, released=False)
    nvlm = bundle.nvlm(generator=True)
    nvlm._neuro_qformer, nvlm._neuro_qformer_variant = bundle.qformer("canonical").to(device).eval(), "canonical"
    vectors, vpmids = V.load_neuro_vectors()
    pos = {int(p): i for i, p in enumerate(vpmids)}
    X = vectors[torch.as_tensor([pos[int(p)] for p in pmids])].float().clone()
    del vectors
    outputs, t0 = {b: [] for b in BASES}, time.perf_counter()
    for start in range(0, len(X), args.gen_batch):
        batch = X[start:start + args.gen_batch]
        for basis in BASES:
            gen = nvlm.generate_text(batch, basis=basis, seed=123 + start, **RECIPE)
            outputs[basis].extend([gen] if isinstance(gen, str) else gen)
        log(f"captioned {min(start + args.gen_batch, len(X))}/{len(X)} training papers ({time.perf_counter() - t0:.0f} s)")
    df = pd.DataFrame({"pmid": [str(p) for p in pmids], **outputs})
    for b in BASES:
        df[b] = df[b].map(strip_steering_tags).str.replace("\n\n", "\n", regex=True)
    df["text"] = df["network"] + " [SEP] " + df["region"] + " [SEP] " + df["function"]
    df["title_constraint"] = "free" if bundle.free_titles else "vocab"
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    meta = {"sources": bundle.sources, "recipe": RECIPE, "gen_batch": args.gen_batch, "seconds": time.perf_counter() - t0}
    cache.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, default=SCRIPTS / "outputs" / "runs" / "final", help="Training run (final model).")
    p.add_argument("--test-captions", type=Path, default=VALIDATION_FINAL / "df_gen_canonical.parquet",
                   help="Stage-2 test captions written by validate_models.py.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0, help="Training-sample and null seed.")
    p.add_argument("--gen-batch", type=int, default=128, help="Generation batch (validation uses 128).")
    p.add_argument("--dapt-corpus", type=Path, default=None,
                   help="Optional NeuroQwen pretraining corpus (pubmed_neuroimaging.parquet, pmid column) to flag membership.")
    p.add_argument("--out", type=Path, default=HERE / "outputs")
    p.add_argument("--force", action="store_true", help="Regenerate the training-sample captions.")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    setup_torch(0)
    import torch
    from neurovlm.data import load_dataset

    test = pd.read_parquet(args.test_captions)
    test_titles = test["title_constraint"].iloc[0] if "title_constraint" in test else "free"
    pub_path = hf_pinned("neurovlm/neuro_image_papers", "publications.parquet")
    pub = pd.read_parquet(pub_path, columns=["pmid", "name", "description"]).drop_duplicates("pmid").set_index("pmid")
    summ = load_dataset("pubmed_summaries")
    summ = summ.assign(pmid=summ["pmid"].astype(np.int64)).drop_duplicates("pmid").set_index("pmid")["summary"]
    split = torch.load(args.model_dir / "pmids_split.pt", weights_only=False)
    import validate_models as V

    _, vpmids = V.load_neuro_vectors()

    def has_text(pm):
        d = pub["description"].reindex(pm).fillna("").astype(str).str.strip().ne("").to_numpy()
        s = summ.reindex(pm).fillna("").astype(str).str.strip().ne("").to_numpy()
        return d & s

    test_pm = test["pmid"].astype(np.int64).to_numpy()
    pool = np.sort(np.intersect1d(np.asarray(split["train"], dtype=np.int64), np.asarray(vpmids, dtype=np.int64)))
    pool = pool[has_text(pool)]
    rng = np.random.default_rng(args.seed)
    train_pm = np.sort(rng.choice(pool, size=len(test_pm), replace=False))
    log(f"test {len(test_pm)} captions ({test_titles} titles); training pool {len(pool)}, sample {len(train_pm)} (seed {args.seed})")
    train = caption_train_sample(args, train_pm, log)

    in_dapt = None
    if args.dapt_corpus is not None and Path(args.dapt_corpus).exists():
        in_dapt = set(pd.to_numeric(pd.read_parquet(args.dapt_corpus, columns=["pmid"])["pmid"], errors="coerce").dropna().astype(np.int64))

    rows, null_rng = [], np.random.default_rng(args.seed)
    for group, df in (("test", test), ("train", train)):
        pm = df["pmid"].astype(np.int64).to_numpy()
        caps = [words(t.replace(" [SEP] ", "\n")) for t in df["text"].fillna("").astype(str)]
        abstracts = [words(f"{n}\n\n{d}") for n, d in zip(pub["name"].reindex(pm).fillna(""), pub["description"].reindex(pm).fillna(""))]
        summaries = [words(s) for s in summ.reindex(pm).fillna("")]
        has_abs = pub["description"].reindex(pm).fillna("").astype(str).str.strip().ne("").to_numpy()
        perm = derangement(len(pm), null_rng)
        for i, cap in enumerate(caps):
            row = {"group": group, "pmid": int(pm[i]), "caption_words": len(cap), "has_abstract": bool(has_abs[i]),
                   "has_summary": bool(len(summaries[i]) > 0), "in_dapt_corpus": None if in_dapt is None else int(pm[i]) in in_dapt}
            for ref, words_ref in (("abstract", abstracts[i]), ("summary", summaries[i]), ("abstract_null", abstracts[perm[i]]),
                                   ("summary_null", summaries[perm[i]])):
                row |= {f"{ref}_{k}": v for k, v in overlap(cap, words_ref).items()}
            rows.append(row)
        log(f"{group}: overlaps of {len(caps)} captions")
    per = pd.DataFrame(rows)
    per.to_parquet(args.out / "per_caption.parquet")

    from scipy.stats import mannwhitneyu

    table, summary = [], {"n": {}, "test_titles": test_titles, "train_titles": str(train["title_constraint"].iloc[0])}
    for ref in REFS:
        need = "has_abstract" if ref.startswith("abstract") else "has_summary"
        t, r = per[(per.group == "test") & per[need]], per[(per.group == "train") & per[need]]
        summary["n"][ref] = {"test": len(t), "train": len(r)}
        for m in METRICS:
            a, b = t[f"{ref}_{m}"].to_numpy(float), r[f"{ref}_{m}"].to_numpy(float)
            u = mannwhitneyu(a, b, alternative="two-sided")
            table.append({"reference": ref, "metric": m, "n_test": len(a), "n_train": len(b),
                          "test_median": float(np.median(a)), "train_median": float(np.median(b)),
                          "test_p95": float(np.percentile(a, 95)), "train_p95": float(np.percentile(b, 95)),
                          "test_mean": float(a.mean()), "train_mean": float(b.mean()), "test_max": float(a.max()), "train_max": float(b.max()),
                          "mannwhitney_u": float(u.statistic), "p_two_sided": float(u.pvalue),
                          "auc_test_gt_train": float(u.statistic / (len(a) * len(b)))})
    tab = pd.DataFrame(table)
    tab.to_csv(args.out / "memorization_table.csv", index=False)
    for ref in ("abstract", "summary"):
        for g in ("test", "train"):
            d = per[(per.group == g) & per["has_abstract" if ref == "abstract" else "has_summary"]]
            summary[f"{g}_{ref}"] = {"captions_with_any_8gram": int((d[f"{ref}_frac_8gram"] > 0).sum()),
                                     "captions_lcs_ge_8": int((d[f"{ref}_lcs_words"] >= 8).sum()),
                                     "captions_lcs_ge_12": int((d[f"{ref}_lcs_words"] >= 12).sum()),
                                     "null_captions_with_any_8gram": int((d[f"{ref}_null_frac_8gram"] > 0).sum()),
                                     "null_captions_lcs_ge_8": int((d[f"{ref}_null_lcs_words"] >= 8).sum())}
    if in_dapt is not None:
        summary["in_dapt_corpus"] = {g: int(per[per.group == g]["in_dapt_corpus"].sum()) for g in ("test", "train")}
        sub = per[per["in_dapt_corpus"].astype(bool) & per["has_abstract"]]
        a, b = sub[sub.group == "test"]["abstract_frac_8gram"], sub[sub.group == "train"]["abstract_frac_8gram"]
        summary["in_dapt_corpus_abstract_frac_8gram"] = {"test_median": float(a.median()), "train_median": float(b.median()),
                                                          "test_p95": float(a.quantile(0.95)), "train_p95": float(b.quantile(0.95)),
                                                          "p_two_sided": float(mannwhitneyu(a, b, alternative="two-sided").pvalue)}

    def fmt(v, m):
        return f"{v:.0f}" if m == "lcs_words" else f"{v:.3f}"

    names = {"lcs_words": "longest common run (words)", "frac_8gram": "caption 8-grams in reference", "rougeL_f": "ROUGE-L F1"}
    lines = ["# Stage-2 captions vs the paper's own text: test (2,971) vs training (2,971) papers", "",
             f"Test captions: `{args.test_captions.relative_to(SCRIPTS.parent) if args.test_captions.is_relative_to(SCRIPTS.parent) else args.test_captions}` "
             f"({test_titles} titles); training captions generated here with the same recipe. Reference = the paper's title + abstract "
             "(NeuroQwen3 pretraining text) or its PubMed summary (stage-13 target); null = another paper's text of the same group.", "",
             "| reference | metric | test median | train median | test 95th pct | train 95th pct | test max | train max | Mann-Whitney p |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        lines.append(f"| {r['reference'].replace('_null', ' (other paper)')} | {names[r['metric']]} | {fmt(r['test_median'], r['metric'])} | "
                     f"{fmt(r['train_median'], r['metric'])} | {fmt(r['test_p95'], r['metric'])} | {fmt(r['train_p95'], r['metric'])} | "
                     f"{fmt(r['test_max'], r['metric'])} | {fmt(r['train_max'], r['metric'])} | {r['p_two_sided']:.2g} |")
    lines += ["", "Captions with at least one 8-gram of their own abstract / with a run of >= 8 words: "
              + "; ".join(f"{g} {summary[f'{g}_abstract']['captions_with_any_8gram']} / {summary[f'{g}_abstract']['captions_lcs_ge_8']}"
                          f" (other paper's abstract: {summary[f'{g}_abstract']['null_captions_with_any_8gram']} / {summary[f'{g}_abstract']['null_captions_lcs_ge_8']})"
                          for g in ("test", "train")) + "."]
    lines += ["", "## Longest runs shared with the own abstract", ""]
    for g in ("test", "train"):
        top = per[per.group == g].sort_values("abstract_lcs_words", ascending=False).head(8)
        lines += [f"**{g}**", ""] + [f"- PMID {r.pmid}: {r.abstract_lcs_words} words: \"{r.abstract_span}\"" for r in top.itertuples()] + [""]
    (args.out / "memorization_table.md").write_text("\n".join(lines) + "\n")
    summary |= {"sources": {"test_captions": str(args.test_captions), "publications": f"neurovlm/neuro_image_papers@{hf_pinned('neurovlm/neuro_image_papers')[:8]}",
                            "summaries": "neurovlm.data.load_dataset('pubmed_summaries')", "model_dir": str(args.model_dir)},
                "seconds": round(time.time() - t_start, 1)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("\n".join(lines[:4 + 2 + len(table)]))


if __name__ == "__main__":
    main()
