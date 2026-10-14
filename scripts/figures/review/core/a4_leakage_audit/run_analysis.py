#!/usr/bin/env python
"""Leakage audit (comments 2.4, 1.2): held-out PubMed and NeuroVault papers vs the training data by DOI/PMID, fuzzy
title, MiniLM title+abstract cosine and identical / near-identical coordinate tables.

    python run_analysis.py --force [--write-exclusions scripts/eval_exclusions]
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import sys
import time
import unicodedata
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from _hf_pins import MINILM, hf_file, hf_split_pmids, sentence_transformer

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

TIMINGS: dict[str, float] = {}


@contextmanager
def timed(name: str):
    t = time.perf_counter()
    yield
    TIMINGS[name] = round(time.perf_counter() - t, 3)


def data_dir() -> Path:
    """Where scripts/data/fetch_hf.py puts the pinned NeuroVault and synthetic tables (their manifest local_name)."""
    from neurovlm.data import get_data_dir

    return Path(get_data_dir())


_WS = re.compile(r"\s+")
_NONALNUM = re.compile(r"[^a-z0-9 ]+")


def norm_text(s: Any) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = html.unescape(str(s))
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = _NONALNUM.sub(" ", s.lower())
    return _WS.sub(" ", s).strip()


def norm_doi(s: Any) -> str | None:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    s = str(s).strip().lower()
    s = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi:\s*", "", s)
    return s or None


def load_pubmed(max_samples: int | None = None, seed: int = 0) -> pd.DataFrame:
    """PubMed publications with their split; ``max_samples`` keeps N test + N val + 10N train + N unsplit."""
    pubs = pd.read_parquet(hf_file("neurovlm/neuro_image_papers", "publications.parquet"))
    split = hf_split_pmids()  # split flags of the pinned HF publications.parquet, never the data-dir copy
    lab = pd.Series("unsplit", index=pubs["pmid"])
    for k in ("train", "val", "test"):
        lab.loc[lab.index.isin(split[k])] = k
    pubs["split"] = lab.to_numpy()
    pubs["doi_n"] = pubs["doi"].map(norm_doi)
    pubs["title_n"] = pubs["name"].map(norm_text)
    pubs["abstract_n"] = pubs["description"].map(norm_text)
    pubs = pubs.rename(columns={"name": "title", "description": "abstract"})[["pmid", "doi", "doi_n", "title", "abstract", "title_n", "abstract_n", "split"]]
    if max_samples:
        rng = np.random.default_rng(seed)
        keep = []
        for k, n in (("test", max_samples), ("val", max_samples), ("train", 10 * max_samples), ("unsplit", max_samples)):
            idx = np.flatnonzero((pubs["split"] == k).to_numpy())
            keep.append(rng.choice(idx, size=min(n, len(idx)), replace=False))
        pubs = pubs.iloc[np.sort(np.concatenate(keep))].reset_index(drop=True)
    return pubs


def load_neurovault() -> pd.DataFrame:
    rel = pd.read_parquet(hf_file("neurovlm/embedded_text", "publications_neurovault.parquet"))
    rel["source"] = "neurovault_released"
    dd = data_dir()
    loc = pd.read_parquet(dd / "neurovault_publications.parquet")
    loc["source"] = "neurovault_local_publications"
    c = pd.read_parquet(dd / "neurovault_collections.parquet", columns=["id", "name", "DOI", "preprint_DOI", "description"])
    a = pd.DataFrame({"doi": c["DOI"], "title": c["name"], "abstract": c["description"], "source": "neurovault_collection_doi", "collection_id": c["id"]})
    b = pd.DataFrame({"doi": c["preprint_DOI"], "title": c["name"], "abstract": c["description"], "source": "neurovault_collection_preprint_doi",
                      "collection_id": c["id"]}).dropna(subset=["doi"])
    nv = pd.concat([rel[["doi", "title", "abstract", "source"]], loc[["doi", "title", "abstract", "source"]], a, b], ignore_index=True)
    nv["doi_n"] = nv["doi"].map(norm_doi)
    nv["title_n"] = nv["title"].map(norm_text)
    nv["abstract_n"] = nv["abstract"].map(norm_text)
    return nv


def load_synthetic() -> pd.DataFrame:
    """Synthetic fine-tuning texts (synth.parquet) and canonical prototypes (text_synth_less.parquet)."""
    frames = []
    for name in ("synth.parquet", "text_synth_less.parquet"):
        d = pd.read_parquet(data_dir() / name, columns=["title", "description"])
        d["source"] = name
        frames.append(d)
    s = pd.concat(frames, ignore_index=True).rename(columns={"description": "abstract"})
    s["title_n"] = s["title"].map(norm_text)
    s["abstract_n"] = s["abstract"].map(norm_text)
    return s


def check_exact(pubs: pd.DataFrame, pubs_all: pd.DataFrame, nv: pd.DataFrame, qa: pd.DataFrame, coords: pd.DataFrame, flagged: list[dict]) -> dict[str, Any]:
    """(a) Exact PMID / DOI overlaps: within PubMed, with NeuroVault, the coordinate file and pubmed_summary_qa."""
    res: dict[str, Any] = {}
    held = pubs[pubs["split"].isin(["test", "val"])]
    res["pmid_duplicates"] = int(pubs["pmid"].duplicated().sum())
    # DOI shared by several PMIDs: list every pair
    d = pubs.dropna(subset=["doi_n"])
    groups = d.groupby("doi_n")["pmid"].apply(list)
    multi = groups[groups.map(len) > 1]
    n_cross = 0
    for doi, pmids in multi.items():
        rows = d[d["pmid"].isin(pmids)]
        splits = rows["split"].tolist()
        cross = len({s for s in splits if s in ("test", "val")}) > 0 and len(set(splits)) > 1
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows.iloc[i], rows.iloc[j]
                flagged.append(dict(check="a_exact_doi", id_a=a["pmid"], split_a=a["split"], id_b=b["pmid"], split_b=b["split"], score=1.0,
                                    strict=True, cross_split=cross, key=doi, text_a=a["title"][:120], text_b=b["title"][:120]))
        n_cross += int(cross)
    res["dois_shared_by_multiple_pmids"] = int(len(multi))
    res["dois_shared_across_splits_(held_out_involved)"] = n_cross
    # NeuroVault DOIs vs PubMed DOIs. 02_data/03 drops every collection whose DOI is in PubMed, so collection-DOI hits
    # are informational; hits in the released / local publication tables are leaks.
    for src, grp in nv.dropna(subset=["doi_n"]).groupby("source"):
        hit = pubs[pubs["doi_n"].isin(set(grp["doi_n"]))]
        res[f"{src}_dois_in_pubmed"] = {k: int((hit["split"] == k).sum()) for k in ("train", "val", "test", "unsplit")}
        strict = not src.startswith("neurovault_collection")
        for _, r in hit.iterrows():
            flagged.append(dict(check=f"a_exact_doi_{src}", id_a=r["pmid"], split_a=r["split"], id_b=r["doi_n"], split_b=src, score=1.0,
                                strict=strict, cross_split=True, key=r["doi_n"], text_a=r["title"][:120], text_b=""))
    # coordinate file: DOI disagreements and unknown PMIDs
    cd = coords.drop_duplicates("pmid")[["pmid", "doi"]].merge(pubs[["pmid", "doi_n", "split"]], on="pmid")
    cd["doi_c"] = cd["doi"].map(norm_doi)
    res["coordinate_doi_disagrees_with_publication_doi"] = int((cd["doi_c"].fillna("") != cd["doi_n"].fillna("")).sum())
    res["coordinate_pmids_not_in_publications"] = int((~coords["pmid"].isin(pubs_all["pmid"])).sum())
    # LLM-summary QA set (stage-1 training data) vs held-out
    qa_pm = qa[qa["pmid"].isin(held["pmid"])]
    res["pubmed_summary_qa_rows_in_test"] = int(qa["pmid"].isin(pubs.loc[pubs["split"] == "test", "pmid"]).sum())
    res["pubmed_summary_qa_rows_in_val"] = int(qa["pmid"].isin(pubs.loc[pubs["split"] == "val", "pmid"]).sum())
    res["pubmed_summary_qa_rows_in_train"] = int(qa["pmid"].isin(pubs.loc[pubs["split"] == "train", "pmid"]).sum())
    qa_doi = qa["doi"].map(norm_doi)
    res["pubmed_summary_qa_dois_in_held_out"] = int(qa_doi.isin(set(held["doi_n"].dropna())).sum())
    sp = pubs.set_index("pmid")["split"]
    for _, r in qa_pm.iterrows():
        flagged.append(dict(check="a_exact_pmid_pubmed_summary_qa", id_a=r["pmid"], split_a=sp[r["pmid"]], id_b="pubmed_summary_qa", split_b="synthetic_qa",
                            score=1.0, strict=False, cross_split=True, key=str(r["pmid"]), text_a=str(r.get("title", ""))[:120], text_b=""))
    return res


def check_titles(pubs: pd.DataFrame, nv: pd.DataFrame, synth: pd.DataFrame, thr: int, flagged: list[dict], per_test: pd.DataFrame) -> dict[str, Any]:
    """(b) Exact normalised-title duplicates, and fuzzy titles of held-out papers vs train, NeuroVault and synthetic texts."""
    from rapidfuzz import fuzz, process

    res: dict[str, Any] = {}
    train = pubs[pubs["split"] == "train"].reset_index(drop=True)
    held = pubs[pubs["split"].isin(["test", "val"])].reset_index(drop=True)
    dup = pubs[pubs["title_n"].duplicated(keep=False) & (pubs["title_n"].str.len() > 0)]
    res["exact_normalised_title_duplicate_groups"] = int(dup["title_n"].nunique())
    res["exact_normalised_title_duplicates_cross_split"] = sum(
        1 for _, g in dup.groupby("title_n") if len(set(g["split"])) > 1 and g["split"].isin(["test", "val"]).any())

    def cdist_flag(q: pd.DataFrame, c: pd.DataFrame, tag: str, id_col_c: str = "pmid", split_c: str | None = None, scorer=fuzz.token_set_ratio):
        """Flag every (q, c) title pair scoring >= thr; return each q row's max score."""
        S = process.cdist(q["title_n"].tolist(), c["title_n"].tolist(), scorer=scorer, workers=-1, dtype=np.float32)
        mx = S.max(axis=1)
        ii, jj = np.nonzero(S >= thr)
        for i, j in zip(ii, jj):
            a, b = q.iloc[i], c.iloc[j]
            sort_score = fuzz.token_sort_ratio(a["title_n"], b["title_n"])
            strict = sort_score >= 90 and len(a["title_n"]) >= 25
            flagged.append(dict(check=tag, id_a=a["pmid"], split_a=a["split"], id_b=b[id_col_c], split_b=(b["split"] if split_c is None else b[split_c]),
                                score=float(S[i, j]), score2=float(sort_score), strict=bool(strict), cross_split=True,
                                key="", text_a=a["title"][:120], text_b=str(b["title"])[:120]))
        return mx
    mx_train = cdist_flag(held, train, "b_title_fuzzy_vs_train")
    per_test.loc[held["pmid"], "max_title_token_set_vs_train"] = mx_train
    nv_rel = nv[nv["source"] == "neurovault_released"].reset_index(drop=True)
    nv_rel = nv_rel.assign(pmid=nv_rel["doi_n"])
    cdist_flag(held, nv_rel, "b_title_fuzzy_vs_neurovault", split_c="source")
    syn = synth.assign(pmid=synth["source"] + ":" + synth.index.astype(str)).reset_index(drop=True)
    cdist_flag(held, syn, "b_title_fuzzy_vs_synthetic_prototypes", split_c="source", scorer=fuzz.token_sort_ratio)
    res["held_out_titles_with_token_set_ge_thr_vs_train"] = int((mx_train >= thr).sum())
    return res


def embed_minilm(texts: list[str], cache: Path, key: str, device: str, batch: int = 128) -> np.ndarray:
    """Normalised MiniLM embeddings, cached under ``key`` and a hash of the texts."""
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / f"minilm_{key}_{hashlib.sha1(chr(10).join(texts).encode()).hexdigest()[:10]}.npy"
    if f.exists():
        return np.load(f)
    model = sentence_transformer(MINILM, device)
    model.max_seq_length = 256
    emb = model.encode(texts, batch_size=batch, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True)
    np.save(f, emb.astype(np.float32))
    return emb


def embed_pubmed(pubs: pd.DataFrame, args, cache: Path, device: str) -> np.ndarray:
    key = f"pubmed_{len(pubs)}_{args.seed}" if args.max_samples else "pubmed_full"
    return embed_minilm((pubs["title"].fillna("") + ". " + pubs["abstract"].fillna("")).tolist(), cache, key, device)


def check_embeddings(pubs, nv, synth, args, cache: Path, device: str, flagged: list[dict], per_test: pd.DataFrame) -> dict[str, Any]:
    """(c) MiniLM title+abstract cosine of held-out papers vs train, NeuroVault and synthetic texts."""
    import torch

    res: dict[str, Any] = {}
    thr = args.embed_threshold
    E = embed_pubmed(pubs, args, cache, device)
    tr = (pubs["split"] == "train").to_numpy()
    tr_idx = np.flatnonzero(tr)
    held_idx = np.flatnonzero(pubs["split"].isin(["test", "val"]).to_numpy())
    Et = torch.from_numpy(E[tr])
    hit_rows = []
    mx = np.zeros(len(held_idx), dtype=np.float32)
    for s in range(0, len(held_idx), 512):
        idx = held_idx[s:s + 512]
        S = torch.from_numpy(E[idx]) @ Et.T
        mx[s:s + 512] = S.max(dim=1).values.numpy()
        ii, jj = torch.nonzero(S >= thr, as_tuple=True)
        for i, j in zip(ii.tolist(), jj.tolist()):
            hit_rows.append((idx[i], tr_idx[j], float(S[i, j])))
    per_test.loc[pubs["pmid"].iloc[held_idx], "max_minilm_cos_vs_train"] = mx
    for i, j, sc in hit_rows:
        a, b = pubs.iloc[i], pubs.iloc[j]
        flagged.append(dict(check="c_minilm_abstract_vs_train", id_a=a["pmid"], split_a=a["split"], id_b=b["pmid"], split_b=b["split"], score=sc,
                            strict=True, cross_split=True, key="", text_a=a["title"][:120], text_b=b["title"][:120]))
    res["held_out_items_with_cos_ge_thr_vs_train"] = int((mx >= thr).sum())
    res["held_out_max_cos_quantiles"] = {q: float(np.quantile(mx, q)) for q in (0.5, 0.9, 0.99, 1.0)}
    # vs the released NeuroVault publications
    nv_rel = nv[nv["source"] == "neurovault_released"].reset_index(drop=True)
    En = embed_minilm((nv_rel["title"].fillna("") + ". " + nv_rel["abstract"].fillna("")).tolist(), cache, "neurovault_released", device)
    S = E[held_idx] @ En.T
    ii, jj = np.nonzero(S >= thr)
    for i, j in zip(ii, jj):
        a, b = pubs.iloc[held_idx[i]], nv_rel.iloc[j]
        flagged.append(dict(check="c_minilm_abstract_vs_neurovault", id_a=a["pmid"], split_a=a["split"], id_b=b["doi_n"], split_b="neurovault_released",
                            score=float(S[i, j]), strict=True, cross_split=True, key="", text_a=a["title"][:120], text_b=str(b["title"])[:120]))
    res["held_out_items_with_cos_ge_thr_vs_neurovault"] = int((S.max(axis=1) >= thr).sum())
    # vs the synthetic texts
    Es = embed_minilm((synth["title"].fillna("") + ". " + synth["abstract"].fillna("")).tolist(), cache, f"synthetic_{len(synth)}", device)
    S = E[held_idx] @ Es.T
    res["held_out_max_cos_vs_synthetic_prototypes"] = float(S.max())
    res["held_out_items_with_cos_ge_thr_vs_synthetic"] = int((S.max(axis=1) >= thr).sum())
    ii, jj = np.nonzero(S >= thr)
    for i, j in zip(ii, jj):
        a, b = pubs.iloc[held_idx[i]], synth.iloc[j]
        flagged.append(dict(check="c_minilm_abstract_vs_synthetic", id_a=a["pmid"], split_a=a["split"], id_b=f"{b['source']}:{j}", split_b=b["source"],
                            score=float(S[i, j]), strict=True, cross_split=True, key="", text_a=a["title"][:120], text_b=str(b["title"])[:120]))
    return res


def check_coordinates(pubs: pd.DataFrame, coords: pd.DataFrame, thr: float, flagged: list[dict], per_test: pd.DataFrame) -> dict[str, Any]:
    """(d) Identical rounded coordinate sets, and near-duplicates by Jaccard on a sparse PMID x peak matrix."""
    import scipy.sparse as sp

    res: dict[str, Any] = {}
    c = coords[coords["pmid"].isin(pubs["pmid"])].copy()
    for k in ("x", "y", "z"):
        c[k] = np.rint(c[k].astype(float)).astype(int)
    c = c.drop_duplicates(["pmid", "x", "y", "z"])
    split = pubs.set_index("pmid")["split"]
    title = pubs.set_index("pmid")["title"]
    sets = c.sort_values(["x", "y", "z"]).groupby("pmid").apply(lambda g: hash(tuple(map(tuple, g[["x", "y", "z"]].to_numpy()))))
    size = c.groupby("pmid").size()
    grp = sets.groupby(sets).apply(lambda s: list(s.index))
    grp = grp[grp.map(len) > 1]
    res["identical_coordinate_set_groups_any_size"] = int(len(grp))
    res["identical_coordinate_set_groups_lt_3_peaks_(ignored)"] = int(sum(1 for g in grp if size[g[0]] < 3))
    grp = grp[[size[g[0]] >= 3 for g in grp]]
    res["identical_coordinate_set_groups"] = int(len(grp))
    res["identical_coordinate_set_groups_cross_split"] = 0
    res["identical_coordinate_set_groups_min_peaks_ge_5"] = int(sum(1 for g in grp if size[g[0]] >= 5))
    for g in grp:
        sp_set = {split[p] for p in g}
        cross = len(sp_set) > 1 and bool(sp_set & {"test", "val"})
        res["identical_coordinate_set_groups_cross_split"] += int(cross)
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                flagged.append(dict(check="d_identical_coordinate_set", id_a=g[i], split_a=split[g[i]], id_b=g[j], split_b=split[g[j]], score=1.0,
                                    strict=bool(size[g[i]] >= 5), cross_split=cross, key=f"n_peaks={int(size[g[i]])}", text_a=title[g[i]][:120], text_b=title[g[j]][:120]))
    # near duplicates: Jaccard on peaks shared by <= 200 papers
    pm_codes, pm_uniq = pd.factorize(c["pmid"])
    key = c["x"].astype(str) + "," + c["y"].astype(str) + "," + c["z"].astype(str)
    co_codes, co_uniq = pd.factorize(key)
    A = sp.csr_matrix((np.ones(len(c), dtype=np.int32), (pm_codes, co_codes)), shape=(len(pm_uniq), len(co_uniq)))
    col_counts = np.asarray(A.sum(axis=0)).ravel()
    keep_cols = col_counts <= 200
    res["distinct_peaks"] = int(len(co_uniq))
    res["peaks_shared_by_gt_200_publications_(ignored)"] = int((~keep_cols).sum())
    Af = A[:, keep_cols]
    n_i = np.asarray(Af.sum(axis=1)).ravel()
    inter = (Af @ Af.T).tocoo()
    m = (inter.row < inter.col) & (inter.data >= 3)
    r, cidx, d = inter.row[m], inter.col[m], inter.data[m]
    jac = d / (n_i[r] + n_i[cidx] - d)
    ok = (jac >= thr) & (np.minimum(n_i[r], n_i[cidx]) >= 5)
    pm_arr = np.asarray(pm_uniq)
    held_mask = split.reindex(pm_arr).isin(["test", "val"]).to_numpy()
    train_mask = (split.reindex(pm_arr) == "train").to_numpy()
    # per held-out max Jaccard vs train
    mxj = np.zeros(len(pm_arr), dtype=np.float32)
    sel = (held_mask[r] & train_mask[cidx]) | (held_mask[cidx] & train_mask[r])
    for rr, cc, jj in zip(r[sel], cidx[sel], jac[sel]):
        h = rr if held_mask[rr] else cc
        mxj[h] = max(mxj[h], jj)
    per_test.loc[pm_arr[held_mask], "max_coord_jaccard_vs_train"] = mxj[held_mask]
    n_cross = 0
    for rr, cc, jj, dd in zip(r[ok], cidx[ok], jac[ok], d[ok]):
        a, b = pm_arr[rr], pm_arr[cc]
        cross = len({split[a], split[b]}) > 1 and bool({split[a], split[b]} & {"test", "val"})
        n_cross += int(cross)
        flagged.append(dict(check="d_coordinate_jaccard", id_a=a, split_a=split[a], id_b=b, split_b=split[b], score=float(jj), strict=True, cross_split=cross,
                            key=f"shared={int(dd)}", text_a=title[a][:120], text_b=title[b][:120]))
    res["near_duplicate_pairs_jaccard_ge_thr"] = int(ok.sum())
    res["near_duplicate_pairs_cross_split"] = n_cross
    return res


_PUBMED_URL = re.compile(r"(?:ncbi\.nlm\.nih\.gov/(?:pubmed|pmc/articles/pmid)/|pubmed\.ncbi\.nlm\.nih\.gov/|[?&]pmid=|\bpmid[:\s]+)(\d{6,9})", re.IGNORECASE)


def neurovault_publications(nv: pd.DataFrame) -> pd.DataFrame:
    """One row per NeuroVault publication (released 312 + local-only 21), with its collection ids, alternative DOIs and
    image ids (released ``images_neurovault.parquet``; local ``neurovault_images.parquet`` counted per collection)."""
    pubs = nv[nv["source"].isin(["neurovault_released", "neurovault_local_publications"])].dropna(subset=["doi_n"]).copy()
    pubs["in_released"] = pubs["doi_n"].isin(set(nv.loc[nv["source"] == "neurovault_released", "doi_n"].dropna()))
    pubs = pubs.sort_values("source", ascending=False).drop_duplicates("doi_n").reset_index(drop=True)  # released row wins
    pubs["source"] = np.where(pubs["in_released"], "neurovault_released", "neurovault_local_only")
    coll = nv[nv["source"].isin(["neurovault_collection_doi", "neurovault_collection_preprint_doi"])].dropna(subset=["doi_n"])
    coll_by_doi = coll.groupby("doi_n")["collection_id"].apply(lambda s: sorted({int(x) for x in s.dropna()}))
    img_rel = pd.read_parquet(hf_file("neurovlm/embedded_text", "images_neurovault.parquet"), columns=["id", "collection_id", "doi"])
    img_rel["doi_n"] = img_rel["doi"].map(norm_doi)
    rel_ids = img_rel.groupby("doi_n")["id"].apply(lambda s: [int(x) for x in s])
    rel_coll = img_rel.groupby("doi_n")["collection_id"].apply(lambda s: sorted({int(x) for x in s}))
    li = pd.read_parquet(data_dir() / "neurovault_images.parquet", columns=["id", "collection_id"])
    loc_ids = li.groupby("collection_id")["id"].apply(lambda s: [int(x) for x in s])
    cols, imgs, n_local = [], [], []
    for d in pubs["doi_n"]:
        c = sorted(set(rel_coll.get(d, [])) | set(coll_by_doi.get(d, [])))
        cols.append(c)
        imgs.append(rel_ids.get(d, []))
        n_local.append(int(sum(len(loc_ids.get(ci, [])) for ci in c)))
    pubs["collection_ids"], pubs["image_ids"], pubs["n_images_local_table"] = cols, imgs, n_local
    pubs["n_images"] = pubs["image_ids"].map(len)
    # alternative DOIs of the same paper: the collection's DOI / preprint DOI
    cdoi = nv[nv["source"].isin(["neurovault_collection_doi", "neurovault_collection_preprint_doi"])].dropna(subset=["doi_n", "collection_id"])
    by_coll = cdoi.groupby("collection_id")["doi_n"].apply(set)
    pubs["alt_dois"] = [sorted(set().union(*[by_coll.get(ci, set()) for ci in c]) - {d}) if c else [] for c, d in zip(pubs["collection_ids"], pubs["doi_n"])]
    return pubs


def check_neurovault_vs_train(pubs: pd.DataFrame, pubs_all: pd.DataFrame, nv: pd.DataFrame, coords: pd.DataFrame, args, cache: Path, device: str,
                              flagged: list[dict]) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """(f) Every NeuroVault publication vs the PubMed train split -> (results, per-publication max similarity, exclusions)."""
    from rapidfuzz import fuzz, process

    res: dict[str, Any] = {}
    nvp = neurovault_publications(nv)
    train = pubs[pubs["split"] == "train"].reset_index(drop=True)
    res["neurovault_publications"] = {"total": int(len(nvp)), "released": int(nvp["in_released"].sum()), "local_only": int((~nvp["in_released"]).sum()),
                                      "released_images": int(nvp["n_images"].sum())}
    res["train_publications_compared"] = int(len(train))
    hits: list[dict] = []
    train_title = train.set_index("pmid")["title"]

    def add(i, check, train_pmid, score, score2=None, key=""):
        r = nvp.iloc[i]
        t = train_title.get(train_pmid, "")
        hits.append(dict(nv_index=i, check=check, train_pmid=train_pmid, score=score))
        flagged.append(dict(check=check, id_a=r["doi_n"], split_a=r["source"], id_b=train_pmid, split_b="train", score=float(score),
                            score2=(float(score2) if score2 is not None else np.nan), strict=True, cross_split=True, key=key,
                            text_a=str(r["title"])[:120], text_b=str(t)[:120]))

    # exact DOI: publication / collection / preprint DOI vs train publication DOIs + coordinate-file DOIs of train PMIDs
    train_doi = {}
    for p, d in zip(train["pmid"], train["doi_n"]):
        if d:
            train_doi.setdefault(d, set()).add(int(p))
    tc = coords[coords["pmid"].isin(set(train["pmid"]))].drop_duplicates("pmid")
    for p, d in zip(tc["pmid"], tc["doi"].map(norm_doi)):
        if d:
            train_doi.setdefault(d, set()).add(int(p))
    for i, r in nvp.iterrows():
        for d in [r["doi_n"], *r["alt_dois"]]:
            for p in sorted(train_doi.get(d, ())):
                add(i, "f_nv_exact_doi_vs_train", p, 1.0, key=d)
    # exact PMID: NeuroVault has no PMID field; resolve it from DOIs (all PubMed splits) and PubMed links in the collection
    # metadata, then test membership in the train split
    all_doi = {}
    for p, d in zip(pubs_all["pmid"], pubs_all["doi_n"]):
        if d:
            all_doi.setdefault(d, set()).add(int(p))
    cu = coords.drop_duplicates("pmid")
    for p, d in zip(cu["pmid"], cu["doi"].map(norm_doi)):
        if d:
            all_doi.setdefault(d, set()).add(int(p))
    url_pmids: dict[int, set[int]] = {}
    c = pd.read_parquet(data_dir() / "neurovault_collections.parquet", columns=["id", "paper_url", "full_dataset_url", "description"])
    for _, cr in c.iterrows():
        found = {int(m) for f in ("paper_url", "full_dataset_url", "description") for m in _PUBMED_URL.findall(str(cr[f] or ""))}
        if found:
            url_pmids[int(cr["id"])] = found
    res["collections_with_pubmed_links"] = len(url_pmids)
    split_all = pubs_all.set_index("pmid")["split"]
    train_set = set(train["pmid"].astype(int))
    nv_pmids = []
    for i, r in nvp.iterrows():
        pm = set()
        for d in [r["doi_n"], *r["alt_dois"]]:
            pm |= all_doi.get(d, set())
        for ci in r["collection_ids"]:
            pm |= url_pmids.get(ci, set())
        nv_pmids.append(sorted(pm))
        for p in sorted(pm & train_set):
            add(i, "f_nv_exact_pmid_vs_train", p, 1.0, key=f"split={split_all.get(p, '?')}")
    nvp["pmid_resolved"] = nv_pmids
    res["neurovault_publications_with_resolved_pmid"] = int(sum(bool(x) for x in nv_pmids))
    res["neurovault_resolved_pmids_by_split"] = {k: int(sum(split_all.get(p) == k for x in nv_pmids for p in x)) for k in ("train", "val", "test", "unsplit")}
    # titles
    thr = args.title_threshold
    S = process.cdist(nvp["title_n"].tolist(), train["title_n"].tolist(), scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
    mx_title = S.max(axis=1)
    am_title = S.argmax(axis=1)
    mx_sort = np.array([fuzz.token_sort_ratio(a, train["title_n"].iloc[j]) for a, j in zip(nvp["title_n"], am_title)], dtype=np.float32)
    n_cand = 0
    for i, j in zip(*np.nonzero(S >= thr)):
        a, b = nvp.iloc[i], train.iloc[j]
        sort_score = fuzz.token_sort_ratio(a["title_n"], b["title_n"])
        strict = sort_score >= 90 and len(a["title_n"]) >= 25
        n_cand += 1
        if strict:
            add(i, "f_nv_title_vs_train", int(b["pmid"]), float(S[i, j]), score2=sort_score)
        else:
            flagged.append(dict(check="f_nv_title_vs_train", id_a=a["doi_n"], split_a=a["source"], id_b=int(b["pmid"]), split_b="train", score=float(S[i, j]),
                                score2=float(sort_score), strict=False, cross_split=True, key="token-subset only", text_a=str(a["title"])[:120], text_b=str(b["title"])[:120]))
    res["title_candidate_pairs_token_set_ge_thr"] = n_cand
    # abstract embeddings
    mx_cos = np.full(len(nvp), np.nan, dtype=np.float32)
    if not args.skip_embeddings:
        E = embed_pubmed(pubs, args, cache, device)
        tr_idx = np.flatnonzero((pubs["split"] == "train").to_numpy())
        En = embed_minilm((nvp["title"].fillna("") + ". " + nvp["abstract"].fillna("")).tolist(), cache, f"neurovault_all_{len(nvp)}", device)
        C = En @ E[tr_idx].T
        mx_cos = C.max(axis=1)
        for i, j in zip(*np.nonzero(C >= args.embed_threshold)):
            add(i, "f_nv_minilm_vs_train", int(pubs["pmid"].iloc[tr_idx[j]]), float(C[i, j]))
        res["max_cos_quantiles"] = {q: float(np.quantile(mx_cos, q)) for q in (0.5, 0.9, 0.99, 1.0)}
    per_nv = pd.DataFrame({"doi": nvp["doi_n"], "source": nvp["source"], "n_images": nvp["n_images"], "max_title_token_set_vs_train": mx_title,
                           "token_sort_of_best_title": mx_sort, "best_title_train_pmid": [int(train["pmid"].iloc[j]) for j in am_title],
                           "max_minilm_cos_vs_train": mx_cos})
    # exclusion table: NeuroVault publications with any strict hit
    h = pd.DataFrame(hits, columns=["nv_index", "check", "train_pmid", "score"])
    rows = []
    for i, g in h.groupby("nv_index"):
        r = nvp.iloc[int(i)]
        reasons = sorted(set(g["check"]))
        sc = {c: float(g.loc[g["check"] == c, "score"].max()) for c in reasons}
        rows.append(dict(doi=r["doi_n"], pmid=";".join(map(str, r["pmid_resolved"])), collection_id=";".join(map(str, r["collection_ids"])),
                         image_ids=";".join(map(str, r["image_ids"])), n_images=int(r["n_images"]), n_images_local_table=int(r["n_images_local_table"]),
                         in_released=bool(r["in_released"]), reasons=";".join(reasons),
                         matched_train_pmid=";".join(map(str, sorted({int(x) for x in g["train_pmid"]}))),
                         score=max(sc.values()), title_token_set=float(mx_title[int(i)]), title_token_sort=float(mx_sort[int(i)]),
                         minilm_cos=float(mx_cos[int(i)]), title=str(r["title"])[:200]))
    excl = pd.DataFrame(rows, columns=["doi", "pmid", "collection_id", "image_ids", "n_images", "n_images_local_table", "in_released", "reasons",
                                       "matched_train_pmid", "score", "title_token_set", "title_token_sort", "minilm_cos", "title"])
    res["excluded_publications"] = {"total": int(len(excl)), "released": int(excl["in_released"].sum()) if len(excl) else 0,
                                    "local_only": int((~excl["in_released"]).sum()) if len(excl) else 0,
                                    "released_images": int(excl["n_images"].sum()) if len(excl) else 0,
                                    "by_reason": {c: int(excl["reasons"].str.contains(c).sum()) for c in
                                                  ("f_nv_exact_doi_vs_train", "f_nv_exact_pmid_vs_train", "f_nv_title_vs_train", "f_nv_minilm_vs_train")} if len(excl) else {}}
    return res, per_nv, excl


def summarise(flagged: pd.DataFrame, n_test: int, n_val: int, thresholds: dict[str, Any], nv_images: dict[str, int]) -> pd.DataFrame:
    """One row per check: pairs flagged, held-out items affected, NeuroVault publications / images (f checks)."""
    rows = []
    descr = {
        "a_exact_doi": "same DOI, different PMIDs (PubMed)",
        "a_exact_pmid_pubmed_summary_qa": "held-out PMID present in pubmed_summary_qa (synthetic QA set; info)",
        "a_exact_doi_neurovault_released": "held-out/any DOI in the released NeuroVault publications",
        "a_exact_doi_neurovault_local_publications": "DOI in the local NeuroVault publication table",
        "a_exact_doi_neurovault_collection_doi": "DOI among scraped NeuroVault collections (input of 02_data/03; these collections are dropped there; info)",
        "a_exact_doi_neurovault_collection_preprint_doi": "preprint DOI among scraped NeuroVault collections (info)",
        "b_title_fuzzy_vs_train": f"title token_set_ratio >= {thresholds['title']} vs train",
        "b_title_fuzzy_vs_neurovault": f"title token_set_ratio >= {thresholds['title']} vs NeuroVault",
        "b_title_fuzzy_vs_synthetic_prototypes": f"title token_sort_ratio >= {thresholds['title']} vs synthetic prototypes",
        "c_minilm_abstract_vs_train": f"MiniLM cosine >= {thresholds['embed']} vs train",
        "c_minilm_abstract_vs_neurovault": f"MiniLM cosine >= {thresholds['embed']} vs NeuroVault",
        "c_minilm_abstract_vs_synthetic": f"MiniLM cosine >= {thresholds['embed']} vs synthetic prototypes",
        "d_identical_coordinate_set": "identical rounded coordinate set",
        "d_coordinate_jaccard": f"coordinate-set Jaccard >= {thresholds['jaccard']} (>= 5 peaks)",
        "f_nv_exact_doi_vs_train": "NeuroVault publication / collection DOI in PubMed train (publication or coordinate-file DOI)",
        "f_nv_exact_pmid_vs_train": "NeuroVault publication resolves to a PubMed train PMID",
        "f_nv_title_vs_train": f"NeuroVault title token_set_ratio >= {thresholds['title']} vs train (strict: token_sort >= 90)",
        "f_nv_minilm_vs_train": f"NeuroVault title+abstract MiniLM cosine >= {thresholds['embed']} vs train",
    }
    core = ["a_exact_doi", "b_title_fuzzy_vs_train", "c_minilm_abstract_vs_train", "d_identical_coordinate_set", "d_coordinate_jaccard",
            "f_nv_exact_doi_vs_train", "f_nv_exact_pmid_vs_train", "f_nv_title_vs_train", "f_nv_minilm_vs_train"]
    checks = sorted(set(core + flagged["check"].unique().tolist()))
    for ch in checks:
        f = flagged[flagged["check"] == ch]
        held_a = f[f["split_a"].isin(["test", "val"]) & f["cross_split"]]
        held_b = f[f["split_b"].isin(["test", "val"]) & f["cross_split"]]
        test_items = set(held_a.loc[held_a["split_a"] == "test", "id_a"]) | set(held_b.loc[held_b["split_b"] == "test", "id_b"])
        val_items = set(held_a.loc[held_a["split_a"] == "val", "id_a"]) | set(held_b.loc[held_b["split_b"] == "val", "id_b"])
        strict = f[f["strict"]]
        t_strict = set(strict.loc[strict["split_a"] == "test", "id_a"]) | set(strict.loc[strict["split_b"] == "test", "id_b"])
        nv_strict = set(strict.loc[strict["split_a"].astype(str).str.startswith("neurovault"), "id_a"]) if ch.startswith("f_") else set()
        rows.append(dict(check=ch, description=descr.get(ch, ch), pairs_flagged=int(len(f)), pairs_cross_split=int(f["cross_split"].sum()),
                         test_items_affected=len(test_items), pct_test=round(100 * len(test_items) / max(1, n_test), 2),
                         val_items_affected=len(val_items), pct_val=round(100 * len(val_items) / max(1, n_val), 2),
                         test_items_strict=len(t_strict),
                         neurovault_pubs_strict=len(nv_strict) if ch.startswith("f_") else None,
                         neurovault_images_strict=int(sum(nv_images.get(d, 0) for d in nv_strict)) if ch.startswith("f_") else None))
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # every flag is recorded in summary.json["args"]
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "outputs", help="Output directory.")
    p.add_argument("--device", default="auto", help="Device for MiniLM (auto = cpu).")
    p.add_argument("--max-samples", type=int, default=None, help="Subsample N test + N val + 10N train papers.")
    p.add_argument("--smoke", action="store_true", help="Same as --max-samples 300.")
    p.add_argument("--seed", type=int, default=0, help="Subsampling seed.")
    p.add_argument("--title-threshold", type=int, default=95, help="Fuzzy title threshold (token_set_ratio).")
    p.add_argument("--embed-threshold", type=float, default=0.95, help="MiniLM cosine threshold.")
    p.add_argument("--jaccard-threshold", type=float, default=0.8, help="Coordinate-set Jaccard threshold.")
    p.add_argument("--skip-embeddings", action="store_true", help="Skip the MiniLM checks.")
    p.add_argument("--write-exclusions", type=Path, default=None, help="Also write the exclusion lists into this directory.")
    p.add_argument("--force", action="store_true", help="Recompute even if outputs exist.")
    args = p.parse_args(argv)
    if args.smoke and not args.max_samples:
        args.max_samples = 300
    if args.write_exclusions and (args.max_samples or args.skip_embeddings):
        p.error("--write-exclusions needs a full run (no --smoke / --max-samples / --skip-embeddings)")
    out: Path = args.out
    cache = out / "cache"
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    if (out / "summary_table.csv").exists() and not args.force and not args.smoke and not args.max_samples and not args.write_exclusions:
        print("outputs exist; use --force")
        return 0
    device = "cpu" if args.device == "auto" else args.device
    t_all = time.perf_counter()
    flagged: list[dict] = []
    results: dict[str, Any] = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}}

    with timed("load"):
        pubs_all = load_pubmed()
        pubs = load_pubmed(args.max_samples, args.seed)
        nv = load_neurovault()
        synth = load_synthetic()
        qa = pd.read_parquet(hf_file("neurovlm/pubmed_summary_qa", "pubmed_summary_qa.parquet"), columns=["pmid", "doi", "title"])
        coords = pd.read_parquet(hf_file("neurovlm/neuro_image_papers", "coordinates.parquet"), columns=["pmid", "doi", "x", "y", "z"])
    n_test, n_val = int((pubs["split"] == "test").sum()), int((pubs["split"] == "val").sum())
    results["n"] = {"publications": len(pubs), "test": n_test, "val": n_val, "train": int((pubs["split"] == "train").sum()),
                    "neurovault_rows": {k: int(v) for k, v in nv["source"].value_counts().items()}, "synthetic_rows": {k: int(v) for k, v in synth["source"].value_counts().items()},
                    "subsampled": bool(args.max_samples)}
    per_test = pd.DataFrame(index=pubs.loc[pubs["split"].isin(["test", "val"]), "pmid"].to_numpy())
    per_test["split"] = pubs.set_index("pmid")["split"].reindex(per_test.index)

    with timed("a_exact_ids"):
        results["a_exact"] = check_exact(pubs, pubs_all, nv, qa, coords, flagged)
    with timed("b_titles"):
        results["b_titles"] = check_titles(pubs, nv, synth, args.title_threshold, flagged, per_test)
    if not args.skip_embeddings:
        with timed("c_embeddings"):
            results["c_embeddings"] = check_embeddings(pubs, nv, synth, args, cache, device, flagged, per_test)
    with timed("d_coordinates"):
        results["d_coordinates"] = check_coordinates(pubs, coords, args.jaccard_threshold, flagged, per_test)
    with timed("f_neurovault_vs_train"):
        results["f_neurovault_vs_train"], per_nv, nv_excl = check_neurovault_vs_train(pubs, pubs_all, nv, coords, args, cache, device, flagged)

    fl = pd.DataFrame(flagged).sort_values(["check", "score"], ascending=[True, False])
    fl.to_csv(out / "flagged_pairs.csv", index=False)
    thresholds = {"title": args.title_threshold, "embed": args.embed_threshold, "jaccard": args.jaccard_threshold}
    summ = summarise(fl, n_test, n_val, thresholds, dict(zip(per_nv["doi"], per_nv["n_images"])))
    summ.to_csv(out / "summary_table.csv", index=False)
    # removal list: held-out items with a strict cross-split flag (not the info-only QA-set or collection-DOI flags)
    rm_rows = []
    strict = fl[fl["strict"] & fl["cross_split"] & ~fl["check"].isin(["a_exact_pmid_pubmed_summary_qa",
                                                                  "a_exact_doi_neurovault_collection_doi", "a_exact_doi_neurovault_collection_preprint_doi"])]
    for _, r in strict.iterrows():
        for side, other in (("a", "b"), ("b", "a")):
            if r[f"split_{side}"] in ("test", "val"):
                rm_rows.append(dict(pmid=r[f"id_{side}"], split=r[f"split_{side}"], reason=r["check"], score=r["score"], matched=r[f"id_{other}"], matched_split=r[f"split_{other}"]))
    rm = pd.DataFrame(rm_rows, columns=["pmid", "split", "reason", "score", "matched", "matched_split"])
    rm_agg = rm.groupby(["pmid", "split"]).agg(reasons=("reason", lambda s: ";".join(sorted(set(s)))), max_score=("score", "max"),
                                               matched=("matched", lambda s: ";".join(map(str, sorted(set(map(str, s)))))[:200]),
                                               matched_source=("matched_split", lambda s: ";".join(sorted({f"pubmed_{x}" if x in ("train", "val", "test", "unsplit") else str(x) for x in s})))
                                               ).reset_index() if len(rm) else rm.assign(reasons=None, max_score=None, matched_source=None)
    rm_agg[["pmid", "split", "reasons", "max_score", "matched"]].to_csv(out / "removal_list.csv", index=False)
    per_test.reset_index(names="pmid").to_csv(out / "max_similarity_per_test.csv", index=False)
    per_nv.to_csv(out / "max_similarity_neurovault.csv", index=False)
    nv_excl.to_csv(out / "neurovault_exclusions.csv", index=False)
    results["removal_list"] = {"test": int((rm_agg["split"] == "test").sum()) if len(rm_agg) else 0, "val": int((rm_agg["split"] == "val").sum()) if len(rm_agg) else 0}
    results["neurovault_exclusions"] = results["f_neurovault_vs_train"]["excluded_publications"]
    if args.write_exclusions:
        write_exclusions(args.write_exclusions, rm_agg, nv_excl, results, out)
    TIMINGS["total"] = round(time.perf_counter() - t_all, 3)
    results["timings_s"] = TIMINGS
    (out / "summary.json").write_text(json.dumps(results, indent=2, default=str))
    (out / "timings.json").write_text(json.dumps(TIMINGS, indent=2))
    with pd.option_context("display.width", 240, "display.max_columns", 20):
        print(summ[["check", "pairs_flagged", "pairs_cross_split", "test_items_affected", "pct_test", "val_items_affected", "test_items_strict",
                    "neurovault_pubs_strict", "neurovault_images_strict"]].to_string(index=False))
    print("removal list:", results["removal_list"])
    print("neurovault exclusions:", results["neurovault_exclusions"])
    print("timings:", TIMINGS)
    return 0


def write_exclusions(dest: Path, rm_agg: pd.DataFrame, nv_excl: pd.DataFrame, results: dict[str, Any], out: Path) -> None:
    """Write {pubmed,neurovault}_exclusions.csv and provenance.json (read by every evaluation from scripts/eval_exclusions/)."""
    dest.mkdir(parents=True, exist_ok=True)
    pm = pd.DataFrame({"pmid": rm_agg["pmid"].astype(int), "split": rm_agg["split"], "reasons": rm_agg["reasons"],
                       "matched_pmid": rm_agg["matched"], "source": rm_agg["matched_source"], "max_score": rm_agg["max_score"].astype(float).round(4)})
    pm = pm.sort_values(["split", "pmid"], ascending=[False, True]).reset_index(drop=True)  # test first
    pm.to_csv(dest / "pubmed_exclusions.csv", index=False)
    nv = nv_excl.drop(columns=["title"]).copy()
    nv["score"] = nv["score"].astype(float).round(4)
    for c in ("title_token_set", "title_token_sort", "minilm_cos"):
        nv[c] = nv[c].astype(float).round(4)
    nv = nv.sort_values(["in_released", "doi"], ascending=[False, True]).reset_index(drop=True)
    nv.to_csv(dest / "neurovault_exclusions.csv", index=False)
    a = results["args"]
    prov = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "generator": "scripts/figures/review/core/a4_leakage_audit/run_analysis.py --write-exclusions " + str(dest),
        "audit_outputs": os.path.relpath(Path(out).resolve(), Path(__file__).resolve().parents[5]),  # repo-relative
        "thresholds": {"title_token_set_ratio": a["title_threshold"], "title_strict_token_sort_ratio": 90, "title_strict_min_chars": 25,
                       "minilm_cosine": a["embed_threshold"], "coordinate_jaccard": a["jaccard_threshold"], "coordinate_min_peaks": 5,
                       "minilm_model": "sentence-transformers/all-MiniLM-L6-v2 (title + '. ' + abstract, max_seq_length 256)"},
        "pubmed": {"test": int((pm["split"] == "test").sum()), "val": int((pm["split"] == "val").sum()),
                   "split_sizes": {"test": results["n"]["test"], "val": results["n"]["val"], "train": results["n"]["train"]}},
        "neurovault": results["f_neurovault_vs_train"],
    }
    (dest / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    print("wrote", dest / "pubmed_exclusions.csv", f"({len(pm)} rows)", dest / "neurovault_exclusions.csv", f"({len(nv)} rows)")


if __name__ == "__main__":
    sys.exit(main())
