#!/usr/bin/env python
"""Retrieval metrics of the contrastive space (recall@k, R@x % of the pool, MRR, median rank, AUC), with hard negatives.

Candidate pools per query: all test items; only items of the query's cognitive domain (nearest Cognitive Atlas concept,
concepts k-means clustered); a random pool of the same size. Rank = 1 + candidates with a strictly larger cosine.

    .venv/bin/python scripts/figures/review/core/a1_retrieval_metrics_table/run_analysis.py --device cpu --force
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _final_model as M  # noqa: E402

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


def pubmed_keep_mask(pmids, *, enabled: bool = True, label: str = "PubMed") -> np.ndarray:
    """False for the leakage-audit PMIDs."""
    pmids = np.asarray(pmids)
    if not enabled:
        return np.ones(len(pmids), dtype=bool)
    excl = {int(x) for x in pd.read_csv(EVAL_EXCLUSIONS_DIR / "pubmed_exclusions.csv")["pmid"]}
    keep = ~pd.to_numeric(pd.Series(pmids.ravel()), errors="coerce").astype("Int64").isin(excl).fillna(False).to_numpy(dtype=bool)
    print(f"[eval exclusions] {label}: dropped {int((~keep).sum())} of {len(keep)} PMIDs")
    return keep


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


def load_pubmed(args: argparse.Namespace, rng: np.random.Generator) -> dict[str, Any]:
    from _hf_pins import load_dataset, load_latent

    df = load_dataset("pubmed_text")
    df = df.assign(pmid=df["pmid"].astype(int))
    text_lat, text_pmid = load_latent("pubmed_text")
    img_lat, img_pmid = M.pubmed_latents()
    text_pos = {int(p): i for i, p in enumerate(np.asarray(text_pmid))}
    img_pos = {int(p): i for i, p in enumerate(np.asarray(img_pmid))}

    def rows_for(flag: str) -> pd.DataFrame:
        sub = df[df[flag].fillna(False).astype(bool)]
        keep = sub["pmid"].map(lambda p: p in text_pos and p in img_pos)
        return sub[keep.values].sort_values("pmid").reset_index(drop=True)

    train, test = rows_for("train"), rows_for("test")  # train is never filtered
    # exclusions before --max-samples sampling
    n_test_before = len(test)
    test = test[pubmed_keep_mask(test["pmid"].to_numpy(), enabled=args.exclusions, label="PubMed test pairs")].reset_index(drop=True)
    n_total_test = len(test)
    if args.max_samples:
        test = test.iloc[np.sort(rng.permutation(len(test))[: args.max_samples])].reset_index(drop=True)

    def gather(frame: pd.DataFrame) -> dict[str, Any]:
        pm = frame["pmid"].to_numpy()
        return {"key": pm.astype(str), "text_emb": text_lat[[text_pos[p] for p in pm]].float(), "img_latent": img_lat[[img_pos[p] for p in pm]].float()}

    return {"train": gather(train), "test": gather(test), "n_total_test": n_total_test, "n_test_before_exclusions": n_test_before,
            "n_test_dropped": int(n_test_before - n_total_test)}


def load_neurovault(args: argparse.Namespace, rng: np.random.Generator, ae: Any, device: torch.device) -> dict[str, Any]:
    """NeuroVault texts and image latents; images normalised by their 99th percentile and clamped (figures/12)."""
    from _hf_pins import load_dataset, load_latent

    df_text = load_dataset("neurovault_text").reset_index(drop=True)
    meta = load_dataset("neurovault_images_meta").reset_index(drop=True)
    text_lat = load_latent("neurovault_text").float()
    images = load_dataset("neurovault_images").float()
    # exclusions before --max-samples sampling and the best-image selection
    pub_keep = neurovault_keep_mask(df_text["doi"].to_numpy(), enabled=args.exclusions, label="NeuroVault publications")
    img_keep = neurovault_keep_mask(meta["doi"].to_numpy(), image_ids=meta["id"].to_numpy(), collection_ids=meta["collection_id"].to_numpy(),
                                    enabled=args.exclusions, label="NeuroVault images")
    pool = np.flatnonzero(pub_keep)
    n_total = len(pool)
    idx = pool
    if args.max_samples:
        idx = np.sort(pool[rng.permutation(n_total)[: max(8, args.max_samples * 3 // 4)]])
    doi_text = df_text["doi"].to_numpy()
    meta_doi = meta["doi"].to_numpy()
    cands = [np.flatnonzero((meta_doi == doi_text[i]) & img_keep) for i in idx]
    keep = np.asarray([len(c) > 0 for c in cands])
    idx, cands = idx[keep], [c for c, k in zip(cands, keep) if k]
    rows = np.concatenate(cands)
    sub = images[torch.as_tensor(rows)].clone()
    for s in range(0, len(sub), 256):
        q = torch.quantile(sub[s : s + 256], 0.99, dim=1, keepdim=True).clamp_min(1e-6)
        sub[s : s + 256] = (sub[s : s + 256] / q).clamp(0, 1)
    with torch.no_grad():
        lat = torch.cat([ae.encoder(sub[s : s + 512].to(device)).cpu() for s in range(0, len(sub), 512)])
    # candidates as positions in `lat`
    offsets = np.cumsum([0] + [len(c) for c in cands])
    local = [np.arange(offsets[i], offsets[i + 1]) for i in range(len(cands))]
    image_ids = meta["id"].to_numpy()[rows]
    return {
        "key": doi_text[idx].astype(str),
        "text_emb": text_lat[torch.as_tensor(idx)],
        "img_latent_all": lat,
        "image_ids_all": image_ids,
        "candidates": local,
        "n_total": n_total,
        "n_images": int(len(rows)),
        "n_publications_dropped": int((~pub_keep).sum()),
        "n_images_dropped": int((~img_keep).sum()),
        "n_images_total": int(img_keep.sum()),
    }


def fit_domains(method: str, n_domains: int, seed: int, train_text_emb: torch.Tensor) -> dict[str, Any]:
    """Domains: Cognitive Atlas concepts k-means clustered, text -> nearest concept ("cogatlas"), or k-means on the
    training texts ("kmeans")."""
    from sklearn.cluster import KMeans

    from _hf_pins import load_latent

    ca_lat, ca_terms = load_latent("cogatlas")
    ca = l2n(ca_lat.float())
    terms = [str(t) for t in ca_terms]
    if method == "cogatlas":
        km = KMeans(n_clusters=n_domains, random_state=seed, n_init=10).fit(ca.numpy())
        term_dom = km.labels_.astype(int)
        centroids = l2n(torch.as_tensor(km.cluster_centers_, dtype=torch.float32))
        names = []
        for d in range(n_domains):
            members = np.flatnonzero(term_dom == d)
            order = members[np.argsort(-(ca[members] @ centroids[d]).numpy())][:3]
            names.append(" / ".join(terms[j] for j in order))

        def assign(emb: torch.Tensor) -> pd.DataFrame:
            sims = l2n(emb) @ ca.T
            best = sims.argmax(dim=1).numpy()
            return pd.DataFrame({"domain": term_dom[best], "nearest_concept": [terms[j] for j in best], "concept_sim": sims.max(dim=1).values.numpy()})

    else:
        km = KMeans(n_clusters=n_domains, random_state=seed, n_init=10).fit(l2n(train_text_emb).numpy())
        centroids = l2n(torch.as_tensor(km.cluster_centers_, dtype=torch.float32))
        sims = centroids @ ca.T
        names = [" / ".join(terms[j] for j in torch.topk(sims[d], 3).indices.tolist()) for d in range(n_domains)]

        def assign(emb: torch.Tensor) -> pd.DataFrame:
            e = l2n(emb)
            dom = (e @ centroids.T).argmax(dim=1).numpy()
            cs = e @ ca.T
            best = cs.argmax(dim=1).numpy()
            return pd.DataFrame({"domain": dom, "nearest_concept": [terms[j] for j in best], "concept_sim": cs.max(dim=1).values.numpy()})

    return {"assign": assign, "names": names, "method": method, "n_domains": n_domains}


def ranks_from_sim(sim: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """1-indexed rank of the diagonal item among the candidates flagged in ``mask`` (strict >)."""
    diag = np.diag(sim)
    better = sim > diag[:, None]
    if mask is not None:
        better &= mask
    return 1 + better.sum(axis=1)


PCT_KS = {"1pct": 0.01, "5pct": 0.05, "50pct": 0.50}  # recall at a fraction of the candidate pool


def pct_k(sizes: np.ndarray, frac: float) -> np.ndarray:
    """Per-query cut-off k_i = max(1, floor(frac * |C_i| + 0.5))."""
    return np.maximum(1, np.floor(frac * sizes.astype(float) + 0.5))


def metrics_from_ranks(ranks: np.ndarray, sizes: np.ndarray, ks: tuple[int, ...]) -> dict[str, float]:
    r = ranks.astype(float)
    c = sizes.astype(float)
    out = {f"recall@{k}": float((r <= k).mean()) for k in ks}
    for name, frac in PCT_KS.items():
        kk = pct_k(c, frac)
        out[f"recall@{name}"] = float((r <= kk).mean())
        out[f"chance_recall@{name}"] = float((kk / c).mean())
        out[f"k@{name}"] = float(np.median(kk))
    out["mrr"] = float((1.0 / r).mean())
    out["median_rank"] = float(np.median(r))
    out["mean_rank"] = float(r.mean())
    out["recall_curve_auc"] = float(((c - r + 1.0) / c).mean())
    out["mean_candidates"] = float(c.mean())
    # chance levels for the same pool sizes
    for k in ks:
        out[f"chance_recall@{k}"] = float(np.minimum(k / c, 1.0).mean())
    out["chance_mrr"] = float(np.mean([np.sum(1.0 / np.arange(1, int(n) + 1)) / n for n in c]))
    out["chance_recall_curve_auc"] = float(((c + 1.0) / (2.0 * c)).mean())
    out["chance_median_rank"] = float(np.median((c + 1.0) / 2.0))
    return out


def bootstrap(ranks: np.ndarray, sizes: np.ndarray, ks: tuple[int, ...], n_boot: int, seed: int) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    n = len(ranks)
    keys = [f"recall@{k}" for k in ks] + [f"recall@{k}" for k in PCT_KS] + ["mrr", "median_rank", "recall_curve_auc"]
    acc: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m = metrics_from_ranks(ranks[idx], sizes[idx], ks)
        for k in keys:
            acc[k].append(m[k])
    return {k: (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))) for k, v in acc.items()}


def recall_curve(ranks: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Recall at each absolute k in ``grid``."""
    return np.array([(ranks <= k).mean() for k in grid])


def random_matched_masks(n: int, sizes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """[n, n] mask: per row the diagonal plus (size_i - 1) random other candidates."""
    noise = rng.random((n, n))
    np.fill_diagonal(noise, -1.0)  # diagonal always selected
    order = np.argsort(noise, axis=1)
    mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        mask[i, order[i, : int(sizes[i])]] = True
    return mask


def evaluate_pair_set(name: str, z_text: torch.Tensor, z_img: torch.Tensor, domain: np.ndarray, keys: np.ndarray,
                      args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame]:
    """Metrics, recall curves and per-query ranks for both directions and all candidate pools."""
    sim_t2i = (z_text @ z_img.T).numpy()
    n = sim_t2i.shape[0]
    same = domain[:, None] == domain[None, :]
    sizes_hard = same.sum(axis=1)
    rows, curves, per_query = [], [], []
    grid = np.unique(np.concatenate([np.arange(1, min(n, 50) + 1), np.unique(np.round(np.logspace(0, np.log10(n), 60)).astype(int))]))
    for direction, sim in (("text_to_brain", sim_t2i), ("brain_to_text", sim_t2i.T)):
        sets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        sets["all"] = (ranks_from_sim(sim), np.full(n, n))
        sets["hard_negatives"] = (ranks_from_sim(sim, same), sizes_hard)
        draws = []
        for d in range(args.n_random_draws):
            mask = random_matched_masks(n, sizes_hard, np.random.default_rng(args.seed + 100 * d + (0 if direction == "text_to_brain" else 1)))
            draws.append(ranks_from_sim(sim, mask))
        sets["random_matched"] = (np.concatenate(draws), np.tile(sizes_hard, args.n_random_draws))
        for cset, (ranks, sizes) in sets.items():
            m = metrics_from_ranks(ranks, sizes, args.ks)
            ci = bootstrap(ranks, sizes, args.ks, args.n_boot, args.seed) if cset != "random_matched" else {}
            rec = {"dataset": name, "direction": direction, "candidate_set": cset, "n_queries": int(n), **m}
            for k, (lo, hi) in ci.items():
                rec[f"{k}_ci_lo"], rec[f"{k}_ci_hi"] = lo, hi
            rows.append(rec)
            rc = recall_curve(ranks, grid)
            curves.extend({"dataset": name, "direction": direction, "candidate_set": cset, "k": int(k), "normalized_k": float(k / n), "recall": float(r), "chance": float(np.minimum(k / sizes, 1.0).mean())} for k, r in zip(grid, rc))
            if cset != "random_matched":
                per_query.append(pd.DataFrame({"dataset": name, "direction": direction, "candidate_set": cset, "key": keys, "rank": ranks, "n_candidates": sizes, "domain": domain}))
    return rows, curves, pd.concat(per_query, ignore_index=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=HERE / "outputs")
    p.add_argument("--model-dir", type=Path, default=M.DEFAULT_MODEL_DIR, help="Run directory with the trained models (default: runs/final).")
    p.add_argument("--device", default="auto", help="Device (auto/cpu/cuda).")
    p.add_argument("--max-samples", type=int, default=None, help="PubMed test pairs (NeuroVault uses 3/4 of it).")
    p.add_argument("--smoke", action="store_true", help="64 PubMed pairs, 48 NeuroVault publications, 200 bootstrap draws.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--datasets", default="pubmed,neurovault")
    p.add_argument("--ks", default="1,5,10")
    p.add_argument("--n-domains", type=int, default=10)
    p.add_argument("--domain-method", default="cogatlas", choices=["cogatlas", "kmeans"])
    p.add_argument("--n-random-draws", type=int, default=5)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--force", action="store_true", help="Recompute even if outputs exist.")
    p.add_argument("--no-exclusions", dest="exclusions", action="store_false", help="Keep the leakage-audit PubMed / NeuroVault items.")
    args = p.parse_args()
    args.ks = tuple(int(k) for k in args.ks.split(","))
    if args.smoke:
        args.max_samples = args.max_samples or 64
        args.n_boot = min(args.n_boot, 200)
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    timings = Timings(out / "timings.json")
    t_start = time.perf_counter()
    device = resolve_device(args.device)
    if (out / "retrieval_metrics.csv").exists() and not args.force:
        print("[skip] outputs exist; use --force")
        return
    rng = np.random.default_rng(args.seed)
    M.from_args(args)
    print(f"[config] device={device} model={M.model_tag()} smoke={args.smoke} max_samples={args.max_samples} domains={args.domain_method}x{args.n_domains}")

    from neurovlm.metrics.retrieval import retrieval_metrics as lib_retrieval_metrics

    with timings.time("load_pubmed"):
        pub = load_pubmed(args, rng)
    print(f"[data] PubMed train={len(pub['train']['key'])} test={len(pub['test']['key'])} (of {pub['n_total_test']} after exclusions; {pub['n_test_dropped']} test-set papers dropped)")
    with timings.time("load_models"):
        ae = M.component("autoencoder", device)
        text_head, image_head = M.component("proj_head_text_infonce", device), M.component("proj_head_image_infonce", device)

    @torch.no_grad()
    def project_text(emb: torch.Tensor) -> torch.Tensor:
        return torch.cat([l2n(text_head(l2n(emb[s : s + 4096]).to(device))).cpu() for s in range(0, len(emb), 4096)])

    @torch.no_grad()
    def project_image(lat: torch.Tensor) -> torch.Tensor:
        return torch.cat([l2n(image_head(lat[s : s + 4096].to(device))).cpu() for s in range(0, len(lat), 4096)])

    with timings.time("domains"):
        domains = fit_domains(args.domain_method, args.n_domains, args.seed, pub["train"]["text_emb"])
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    per_query_frames = []
    crosscheck: dict[str, Any] = {}
    counts: dict[str, Any] = {}
    exclusions: dict[str, Any] = {"enabled": bool(args.exclusions), "source": str(EVAL_EXCLUSIONS_DIR), "n_pubmed_test_before": int(pub["n_test_before_exclusions"]),
                                  "n_pubmed_test_dropped": int(pub["n_test_dropped"]), "n_pubmed_val_dropped": 0,
                                  "note": "val split is not used by this analysis; the PubMed training split (domain fit) is not filtered"}

    if "pubmed" in datasets:
        with timings.time("pubmed"):
            te = pub["test"]
            z_t, z_i = project_text(te["text_emb"]), project_image(te["img_latent"])
            dom = domains["assign"](te["text_emb"])
            dom.insert(0, "key", te["key"])
            dom["domain_label"] = [domains["names"][d] for d in dom["domain"]]
            dom.to_csv(out / "domain_assignment_pubmed.csv", index=False)
            r, c, pq = evaluate_pair_set("pubmed", z_t, z_i, dom["domain"].to_numpy(), te["key"], args)
            rows += r
            curves += c
            per_query_frames.append(pq)
            lib = lib_retrieval_metrics(z_t, z_i, ks=args.ks)
            crosscheck["pubmed_text_to_brain_library"] = {k: float(v) for k, v in lib.items()}
            counts["pubmed"] = {"n_pairs": int(len(te["key"])), "n_total_test": pub["n_total_test"]}
    if "neurovault" in datasets:
        with timings.time("load_neurovault"):
            nv = load_neurovault(args, rng, ae, device)
        print(f"[data] NeuroVault publications={len(nv['key'])} (of {nv['n_total']} after exclusions) images={nv['n_images']}")
        with timings.time("neurovault"):
            z_t = project_text(nv["text_emb"])
            z_all = project_image(nv["img_latent_all"])
            dom = domains["assign"](nv["text_emb"])
            dom.insert(0, "key", nv["key"])
            dom["domain_label"] = [domains["names"][d] for d in dom["domain"]]
            dom.to_csv(out / "domain_assignment_neurovault.csv", index=False)
            sim_all = (z_t @ z_all.T).numpy()
            best = np.asarray([c[int(np.argmax(sim_all[i, c]))] for i, c in enumerate(nv["candidates"])])
            first = np.asarray([c[int(np.argmin(nv["image_ids_all"][c]))] for c in nv["candidates"]])
            pd.DataFrame({"doi": nv["key"], "best_image_id": nv["image_ids_all"][best], "first_image_id": nv["image_ids_all"][first], "n_images": [len(c) for c in nv["candidates"]]}).to_csv(out / "neurovault_target_selection.csv", index=False)
            for tag, sel in (("neurovault", best), ("neurovault_first", first)):
                r, c, pq = evaluate_pair_set(tag, z_t, z_all[torch.as_tensor(sel)], dom["domain"].to_numpy(), nv["key"], args)
                rows += r
                curves += c
                per_query_frames.append(pq)
            lib = lib_retrieval_metrics(z_t, z_all[torch.as_tensor(best)], ks=args.ks)
            crosscheck["neurovault_text_to_brain_library"] = {k: float(v) for k, v in lib.items()}
            counts["neurovault"] = {"n_publications": int(len(nv["key"])), "n_total_publications": nv["n_total"], "n_images": nv["n_images"]}
            exclusions.update({"n_neurovault_publications_dropped": nv["n_publications_dropped"], "n_neurovault_images_dropped": nv["n_images_dropped"],
                               "n_neurovault_publications_after": nv["n_total"], "n_neurovault_images_after": nv["n_images_total"]})

    table = pd.DataFrame(rows)
    table.to_csv(out / "retrieval_metrics.csv", index=False)
    pd.DataFrame(curves).to_csv(out / "recall_curves.csv", index=False)
    pd.concat(per_query_frames, ignore_index=True).to_parquet(out / "per_query_ranks.parquet", index=False)
    summary = pd.concat([pd.read_csv(out / f"domain_assignment_{d}.csv").assign(dataset=d) for d in ["pubmed", "neurovault"] if (out / f"domain_assignment_{d}.csv").exists()])
    summary.groupby(["dataset", "domain", "domain_label"]).size().rename("n").reset_index().to_csv(out / "domain_sizes.csv", index=False)
    (out / "library_crosscheck.json").write_text(json.dumps(crosscheck, indent=2))
    config = {
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "device": str(device),
        **M.model_info(),
        "counts": counts,
        "exclusions": exclusions,
        "domain_names": domains["names"],
        "rank_convention": "1 + number of candidates with strictly larger cosine than the true item",
        "neurovault_target": "neurovault = contrastive-selected image per publication (paper recipe, upper bound); neurovault_first = lowest NeuroVault image id per publication (target independent)",
        "wall_time_s": round(time.perf_counter() - t_start, 1),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, default=str))
    timings.records["total_wall_time_s"] = config["wall_time_s"]
    timings.records["exclusions"] = exclusions
    timings.save()
    show = ["dataset", "direction", "candidate_set", "n_queries", "mean_candidates"] + [f"recall@{k}" for k in args.ks] + [f"recall@{k}" for k in PCT_KS] + ["mrr", "median_rank", "recall_curve_auc"]
    print(table[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"[done] {config['wall_time_s']} s -> {out}")


if __name__ == "__main__":
    main()
