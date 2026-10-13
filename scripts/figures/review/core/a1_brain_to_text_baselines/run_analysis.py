#!/usr/bin/env python
"""Brain-to-text: NeuroVLM generation vs nearest-neighbour retrieval, scored against the paper's reference texts with
MiniLM cosine and BERTScore F1 (neither is part of NeuroVLM), plus each text's similarity to its nearest training text.

    .venv/bin/python scripts/figures/review/core/a1_brain_to_text_baselines/run_analysis.py --device cuda --force
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _final_model as M  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("a1_brain_to_text_baselines")
GEN_KWARGS = dict(do_sample=False, num_beams=1, max_new_tokens=128, projection_temp=0.035, repetition_penalty=1.25)
BASES = ("network", "region", "function")
# caption cleanup as scripts/models/_common_models.py (validate_models.py)
STEERING_TAG_RE = re.compile(r"(?i)[ \t]*\[(?:network|region|function)\][ \t]*")


def strip_steering_tags(text: str) -> str:
    """Drop the [NETWORK] / [REGION] / [FUNCTION] tags the LM sometimes writes into a caption, with the spaces around
    them (a tag between two words leaves one space); lines without a tag are left as they are."""
    return "\n".join(STEERING_TAG_RE.sub(" ", line).strip() if STEERING_TAG_RE.search(line) else line
                     for line in str(text).split("\n"))


# leakage-audit exclusions (written by a4_leakage_audit/run_analysis.py --write-exclusions)
EVAL_EXCLUSIONS_DIR = Path(__file__).resolve().parents[5] / "scripts" / "eval_exclusions"


def pubmed_keep_mask(pmids, *, enabled: bool = True, label: str = "PubMed") -> np.ndarray:
    """False for the leakage-audit PMIDs."""
    pmids = np.asarray(pmids)
    if not enabled:
        return np.ones(len(pmids), dtype=bool)
    excl = {int(x) for x in pd.read_csv(EVAL_EXCLUSIONS_DIR / "pubmed_exclusions.csv")["pmid"]}
    keep = ~pd.to_numeric(pd.Series(pmids.ravel()), errors="coerce").astype("Int64").isin(excl).fillna(False).to_numpy(dtype=bool)
    print(f"[eval exclusions] {label}: dropped {int((~keep).sum())} of {len(keep)} PMIDs")
    return keep


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
        if path.exists():
            try:
                self.records = json.loads(path.read_text())
            except json.JSONDecodeError:
                self.records = {}

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


@contextmanager
def log_phase(name: str, **info) -> Iterator[None]:
    extra = ", ".join(f"{k}={v}" for k, v in info.items())
    log.info("start %s%s", name, f" ({extra})" if extra else "")
    t0 = time.perf_counter()
    yield
    log.info("%s done in %.1fs", name, time.perf_counter() - t0)


def release_cuda_cache() -> str:
    """Free the CUDA cache and describe GPU memory.

    Without this, a large model loaded after generation (deberta-xlarge) can land in shared system memory on WSL2 and
    run very slowly."""
    import gc

    gc.collect()
    if not torch.cuda.is_available():
        return "cpu"
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    return (f"GPU free {free / 2**30:.1f}/{total / 2**30:.1f} GiB, torch allocated {torch.cuda.memory_allocated() / 2**30:.1f} GiB, "
            f"reserved {torch.cuda.memory_reserved() / 2**30:.1f} GiB")


def unique_index(text_lists: Sequence[Sequence[str]]) -> tuple[list[str], dict[str, int]]:
    """Unique texts (first-seen order) and text -> row, so every text is encoded once."""
    uniq = list(dict.fromkeys(str(t) for tx in text_lists for t in tx))
    return uniq, {t: i for i, t in enumerate(uniq)}


def l2n(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def deranged_permutation(n: int, rng: np.random.Generator) -> np.ndarray:
    """Permutation without fixed points (the paper's null)."""
    perm = rng.permutation(n)
    fixed = np.flatnonzero(perm == np.arange(n))
    if len(fixed) > 1:
        perm[fixed] = np.roll(perm[fixed], 1)
    elif len(fixed) == 1:
        i = int(fixed[0])
        j = (i + 1) % n
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def bootstrap_mean_ci(values: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def holm_adjust(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return out
    pv = p[ok]
    order = np.argsort(pv)
    m = len(pv)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pv[i])
        adj[i] = min(1.0, running)
    out[ok] = adj
    return out


def ranking_metrics(sims: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    """Rank of the true (diagonal) item per query row; recall@k, MRR, median rank, AUC."""
    n = sims.shape[0]
    diag = np.diag(sims)
    ranks = 1 + (sims > diag[:, None]).sum(axis=1)
    ks = np.arange(1, n + 1)
    curve = np.array([(ranks <= k).mean() for k in ks])
    out = {
        "recall@1": float((ranks <= 1).mean()),
        "recall@5": float((ranks <= 5).mean()),
        "recall@10": float((ranks <= 10).mean()),
        "mrr": float((1.0 / ranks).mean()),
        "median_rank": float(np.median(ranks)),
        "normalized_recall_auc": float(curve.mean()),
        "n_candidates": int(n),
    }
    return out, ranks


def load_pubmed(args: argparse.Namespace, rng: np.random.Generator) -> dict[str, Any]:
    from _hf_pins import load_dataset, load_latent

    df = load_dataset("pubmed_text")
    df = df.assign(pmid=df["pmid"].astype(int))
    summ = load_dataset("pubmed_summaries")[["pmid", "summary"]].assign(pmid=lambda d: d["pmid"].astype(int)).drop_duplicates("pmid").set_index("pmid")["summary"]
    df["summary"] = df["pmid"].map(summ)
    text_lat, text_pmid = load_latent("pubmed_text")
    img_lat, img_pmid = M.pubmed_latents()
    maps, map_pmid = load_dataset("pubmed_images")
    text_pos = {int(p): i for i, p in enumerate(np.asarray(text_pmid))}
    img_pos = {int(p): i for i, p in enumerate(np.asarray(img_pmid))}
    map_pos = {int(p): i for i, p in enumerate(np.asarray(map_pmid))}

    def rows_for(flag: str) -> pd.DataFrame:
        sub = df[df[flag].fillna(False).astype(bool)]
        keep = sub["pmid"].map(lambda p: p in text_pos and p in img_pos and p in map_pos)
        sub = sub[keep.values]
        if args.reference == "summary":
            sub = sub[sub["summary"].notna()]
        return sub.sort_values("pmid").reset_index(drop=True)

    train = rows_for("train")  # never filtered
    test = rows_for("test")
    # exclusions before --max-samples sampling
    n_test_before = len(test)
    test = test[pubmed_keep_mask(test["pmid"].to_numpy(), enabled=args.exclusions, label="PubMed test queries")].reset_index(drop=True)
    exclusions = {"enabled": bool(args.exclusions), "source": str(EVAL_EXCLUSIONS_DIR), "n_pubmed_test_before": int(n_test_before),
                  "n_pubmed_test_dropped": int(n_test_before - len(test)), "n_pubmed_val_dropped": 0,
                  "n_neurovault_publications_dropped": 0, "n_neurovault_images_dropped": 0,
                  "note": "val split and NeuroVault are not used by this analysis; the training bank is not filtered"}
    n_total_test = len(test)
    if args.max_samples:
        sel = np.sort(rng.permutation(len(test))[: args.max_samples])
        test = test.iloc[sel].reset_index(drop=True)
    long_col = "summary" if args.reference == "summary" else "description"

    def gather(frame: pd.DataFrame) -> dict[str, Any]:
        pm = frame["pmid"].to_numpy()
        return {
            "pmid": pm,
            "text": (frame["name"].astype(str) + " [SEP] " + frame[long_col].astype(str)).tolist(),
            "text_emb": text_lat[[text_pos[p] for p in pm]].float(),
            "img_latent": img_lat[[img_pos[p] for p in pm]].float(),
            "map_index": np.asarray([map_pos[p] for p in pm]),
        }

    return {"train": gather(train), "test": gather(test), "maps": maps, "n_total_test": n_total_test, "exclusions": exclusions}


@torch.no_grad()
def generate_texts(nvlm: Any, flats: torch.Tensor, pmids: np.ndarray, batch_size: int, seed: int, cache: Path) -> pd.DataFrame:
    """Fig. 4 recipe: three bases, greedy, 128 tokens, joined with [SEP]; resumes from ``cache``."""
    done = pd.read_parquet(cache) if cache.exists() else pd.DataFrame(columns=["pmid", *BASES, "text"])
    done = done.set_index("pmid") if len(done) else done
    done_ids = set(int(x) for x in done.index.tolist()) if len(done) else set()
    todo = [i for i, p in enumerate(pmids) if int(p) not in done_ids]
    rows: list[dict[str, Any]] = []
    for start in tqdm(range(0, len(todo), batch_size), desc="NeuroVLM brain-to-text", leave=False):
        idx = todo[start : start + batch_size]
        batch = flats[idx]
        outs = {}
        for basis in BASES:
            texts = nvlm.generate_text(batch, seed=seed + start, basis=basis, **GEN_KWARGS)
            if isinstance(texts, str):
                texts = [texts]
            outs[basis] = [strip_steering_tags(t).strip() for t in texts]
        for j, i in enumerate(idx):
            rows.append({"pmid": int(pmids[i]), **{b: outs[b][j] for b in BASES}})
        if start % (batch_size * 8) == 0 and rows:
            _flush(done, rows, cache)
    done = _flush(done, rows, cache)
    return done.loc[[int(p) for p in pmids]].reset_index()


def _flush(done: pd.DataFrame, rows: list[dict[str, Any]], cache: Path) -> pd.DataFrame:
    if rows:
        new = pd.DataFrame(rows)
        new["text"] = new[list(BASES)].agg(" [SEP] ".join, axis=1)
        new = new.set_index("pmid")
        done = pd.concat([done, new]) if len(done) else new
        done = done[~done.index.duplicated(keep="last")]
        done.reset_index().to_parquet(cache, index=False)
    return done


class Bank:
    def __init__(self, nvlm: Any, pub: dict[str, Any], device: torch.device, n_domains: int, seed: int) -> None:
        from sklearn.cluster import KMeans

        self.nvlm = nvlm
        self.device = device
        self.train = pub["train"]
        self.train_texts = np.asarray(self.train["text"], dtype=object)
        self.img_latent = l2n(self.train["img_latent"])
        self.img_shared = self.project_image(self.train["img_latent"])
        self.text_shared = self.project_text(self.train["text_emb"])
        km = KMeans(n_clusters=n_domains, random_state=seed, n_init=10).fit(self.img_latent.numpy())
        self.centroids = l2n(torch.as_tensor(km.cluster_centers_, dtype=torch.float32))
        self.train_domain = km.labels_.astype(int)
        self.medoid = np.array([int((self.img_latent[self.train_domain == d] @ self.centroids[d]).argmax()) for d in range(n_domains)])
        self.medoid = np.array([np.where(self.train_domain == d)[0][m] for d, m in enumerate(self.medoid)])

    @torch.no_grad()
    def project_image(self, latent: torch.Tensor) -> torch.Tensor:
        self.nvlm._ensure_projection_heads()
        return torch.cat([l2n(self.nvlm._proj_head_image_infonce(latent[s : s + 4096].to(self.device))).cpu() for s in range(0, len(latent), 4096)])

    @torch.no_grad()
    def project_text(self, emb: torch.Tensor) -> torch.Tensor:
        self.nvlm._ensure_projection_heads()
        return torch.cat([l2n(self.nvlm._proj_head_text_infonce(l2n(emb[s : s + 4096]).to(self.device))).cpu() for s in range(0, len(emb), 4096)])

    def nn_ae_latent(self, q_latent: torch.Tensor) -> np.ndarray:
        return (l2n(q_latent) @ self.img_latent.T).argmax(dim=1).numpy()

    def nn_contrastive_image(self, q_latent: torch.Tensor) -> np.ndarray:
        return (self.project_image(q_latent) @ self.img_shared.T).argmax(dim=1).numpy()

    def nn_contrastive_xmodal(self, q_latent: torch.Tensor) -> np.ndarray:
        return (self.project_image(q_latent) @ self.text_shared.T).argmax(dim=1).numpy()

    def domain_medoid(self, q_latent: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        dom = (l2n(q_latent) @ self.centroids.T).argmax(dim=1).numpy()
        return self.medoid[dom], dom


class Embedder:
    def __init__(self, name: str, device: torch.device, batch_size: int = 128) -> None:
        from _hf_pins import sentence_transformer

        self.name = name
        self.model = sentence_transformer(name, device)  # at the revision pinned in scripts/hf_manifest.json
        self.batch_size = batch_size

    def __call__(self, texts: list[str]) -> torch.Tensor:
        emb = self.model.encode([str(t) for t in texts], batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return torch.as_tensor(emb, dtype=torch.float32)

    def close(self) -> None:
        del self.model


@torch.no_grad()
def specter_shared(nvlm: Any, texts: list[str], device: torch.device, batch_size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    nvlm._ensure_projection_heads()
    raw, shared = [], []
    for s in tqdm(range(0, len(texts), batch_size), desc=f"SPECTER2 encode ({len(texts)} texts)", leave=False):
        e = nvlm._encode_text([str(t) for t in texts[s : s + batch_size]])
        raw.append(l2n(e).cpu())
        shared.append(l2n(nvlm._proj_head_text_infonce(l2n(e).to(device))).cpu())
    return torch.cat(raw), torch.cat(shared)


class BertScore:
    """BERTScore F1 with one model for the whole run (kept on the CPU between uses).

    Every unique sentence is embedded once, then each (candidates, references) list is greedy-matched with bert_score's
    own functions, so F1 equals ``bert_score.score(cands, refs)[2]`` up to float noise."""

    def __init__(self, model_type: str, device: torch.device, batch_size: int) -> None:
        from bert_score import BERTScorer
        from _hf_pins import bertscore_model

        self.model_type, self.device = model_type, torch.device(device)
        # pinned snapshot dir + bert_score's layer for the model (BERTScorer takes no revision)
        self.scorer = BERTScorer(**bertscore_model(model_type), lang="en", batch_size=batch_size, device="cpu")

    @torch.no_grad()
    def f1_pairs(self, pairs: Mapping[str, tuple[Sequence[str], Sequence[str]]]) -> dict[str, np.ndarray]:
        """``{name: F1(cands[i], refs[i])}`` for ``pairs = {name: (cands, refs)}``."""
        from collections import defaultdict

        from bert_score.utils import get_bert_embedding, greedy_cos_idf
        from torch.nn.utils.rnn import pad_sequence

        sc = self.scorer
        tok, model, bs, all_layers, device = sc._tokenizer, sc._model, sc.batch_size, sc.all_layers, self.device
        pairs = {k: ([str(c) for c in cands], [str(r) for r in refs]) for k, (cands, refs) in pairs.items()}
        try:
            mem = release_cuda_cache()
            model.to(device)
            idf = defaultdict(lambda: 1.0)
            idf[tok.sep_token_id] = 0
            idf[tok.cls_token_id] = 0
            # as bert_score.utils.bert_cos_score_idf: unique sentences, longest first, padding cut off
            sents = sorted(set().union(*(set(c) | set(r) for c, r in pairs.values())), key=lambda x: len(x.split(" ")), reverse=True)
            log.info("BERTScore: %s on %s, %d unique texts for %d lists x %s pairs (%s)", self.model_type, device, len(sents), len(pairs),
                     sorted({len(c) for c, _ in pairs.values()}), mem)
            stats = {}
            for s in tqdm(range(0, len(sents), bs), desc="BERTScore: embed", leave=False):
                batch = sents[s:s + bs]
                emb, mask, pidf = get_bert_embedding(batch, model, tok, idf, device=device, all_layers=all_layers)
                emb, mask, pidf = emb.cpu(), mask.cpu(), pidf.cpu()
                for i, sen in enumerate(batch):
                    n_tok = int(mask[i].sum().item())
                    stats[sen] = (emb[i, :n_tok], pidf[i, :n_tok])

            def pad_batch_stats(sen_batch):  # copy of bert_score's pad_batch_stats
                emb = [stats[x][0].to(device) for x in sen_batch]
                w = [stats[x][1].to(device) for x in sen_batch]
                lens = torch.tensor([e.size(0) for e in emb], dtype=torch.long)
                pad_mask = (torch.arange(int(lens.max()), dtype=torch.long).expand(len(lens), int(lens.max())) < lens.unsqueeze(1)).to(device)
                return pad_sequence(emb, batch_first=True, padding_value=2.0), pad_mask, pad_sequence(w, batch_first=True)

            out = {}
            for name, (cands, refs) in pairs.items():
                f1 = []
                for s in tqdm(range(0, len(cands), bs), desc=f"BERTScore: match {name}", leave=False):
                    f1.append(greedy_cos_idf(*pad_batch_stats(refs[s:s + bs]), *pad_batch_stats(cands[s:s + bs]), all_layers)[2].cpu())
                out[name] = torch.cat(f1, dim=1 if all_layers else 0).numpy().astype(float)
            return out
        finally:
            model.to("cpu")
            release_cuda_cache()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=HERE / "outputs")
    p.add_argument("--model-dir", type=Path, default=M.DEFAULT_MODEL_DIR, help="Run directory with the trained models (default: runs/final).")
    p.add_argument("--device", default="auto", help="Device (auto/cpu/cuda).")
    p.add_argument("--max-samples", type=int, default=None, help="Number of test maps (default: all).")
    p.add_argument("--smoke", action="store_true", help="32 test maps, small batches, 2,000 training texts.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reference", default="summary", choices=["summary", "abstract"], help="Reference text: title + LLM summary (paper) or title + abstract.")
    p.add_argument("--gen-batch-size", type=int, default=64)
    p.add_argument("--n-domains", type=int, default=10)
    p.add_argument("--train-subsample", type=int, default=None, help="Training texts for the nearest-training-text check (default: all).")
    p.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--extra-embedder", default=None, help="Optional second embedder, e.g. BAAI/bge-small-en-v1.5 (pin it in scripts/hf_manifest.json first).")
    p.add_argument("--bertscore-model", default="microsoft/deberta-xlarge-mnli")
    p.add_argument("--bertscore-batch-size", type=int, default=32)
    p.add_argument("--skip-bertscore", action="store_true")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--force", action="store_true", help="Recompute even if outputs exist.")
    p.add_argument("--no-exclusions", dest="exclusions", action="store_false", help="Keep the leakage-audit test papers.")
    args = p.parse_args()
    if args.smoke:
        args.max_samples = args.max_samples or 32
        args.gen_batch_size = min(args.gen_batch_size, 8)
        args.train_subsample = args.train_subsample or 2000
        args.n_boot = min(args.n_boot, 200)
        args.bertscore_batch_size = min(args.bertscore_batch_size, 8)
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = out / "cache"
    cache_dir.mkdir(exist_ok=True)
    timings = Timings(out / "timings.json")
    t_start = time.perf_counter()
    device = resolve_device(args.device)
    if (out / "per_sample_metrics.parquet").exists() and not args.force:
        print("[skip] outputs exist; use --force")
        return
    rng = np.random.default_rng(args.seed)
    M.from_args(args)
    print(f"[config] device={device} model={M.model_tag()} smoke={args.smoke} max_samples={args.max_samples} reference={args.reference}")

    with timings.time("load_pubmed"), log_phase("load PubMed texts, latents and maps"):
        pub = load_pubmed(args, rng)
    te, tr = pub["test"], pub["train"]
    n = len(te["pmid"])
    print(f"[data] train={len(tr['pmid'])} test={n} (of {pub['n_total_test']} after exclusions; {pub['exclusions']['n_pubmed_test_dropped']} test-set papers dropped)")
    with timings.time("load_models"), log_phase("load NeuroVLM", device=device, model=M.model_tag()):
        nvlm = M.make_nvlm(device, ae=M.component("autoencoder", device), text_head=M.component("proj_head_text_infonce", device),
                           image_head=M.component("proj_head_image_infonce", device), mse_head=M.component("proj_head_text_mse", device),
                           projector=M.load_projector("canonical", device))

    # generations are cached per PMID under a key of everything that changes greedy outputs (device type, batch, decoding)
    gen_fp = {"device_type": device.type, "gen_batch_size": args.gen_batch_size, "generation_kwargs": GEN_KWARGS, "bases": BASES,
              "model": M.model_info()["model_dir"]}
    gen_key = hashlib.sha256(json.dumps(gen_fp, sort_keys=True).encode()).hexdigest()[:10]
    gen_cache = cache_dir / f"generated_texts_{args.reference}_{gen_key}.parquet"
    (cache_dir / f"generated_texts_{args.reference}_{gen_key}.json").write_text(json.dumps(gen_fp, indent=2))
    with timings.time("generation"), log_phase("NeuroVLM generation", n=n, gen_batch=args.gen_batch_size, cache=gen_cache.name):
        flats = pub["maps"][te["map_index"]].float()
        gen = generate_texts(nvlm, flats, te["pmid"], args.gen_batch_size, args.seed, gen_cache)
    log.info("released CUDA cache after generation (%s)", release_cuda_cache())

    with timings.time("baselines"), log_phase("retrieval / simple baselines", n_domains=args.n_domains):
        bank = Bank(nvlm, pub, device, args.n_domains, args.seed)
        texts: dict[str, list[str]] = {"neurovlm_generated": gen["text"].tolist()}
        src: dict[str, np.ndarray] = {}
        src["nn_ae_latent"] = bank.nn_ae_latent(te["img_latent"])
        src["nn_contrastive_image"] = bank.nn_contrastive_image(te["img_latent"])
        src["nn_contrastive_xmodal"] = bank.nn_contrastive_xmodal(te["img_latent"])
        src["domain_medoid_text"], q_dom = bank.domain_medoid(te["img_latent"])
        src["random_train_text"] = rng.integers(0, len(tr["pmid"]), size=n)
        for k, v in src.items():
            texts[k] = bank.train_texts[v].tolist()
        perm = deranged_permutation(n, np.random.default_rng(args.seed + 7))
        texts["null_shuffled_neurovlm"] = [texts["neurovlm_generated"][i] for i in perm]
        pd.DataFrame({"pmid": tr["pmid"], "split": "train", "domain": bank.train_domain}).to_csv(out / "domain_assignment_train.csv", index=False)
        pd.DataFrame({"pmid": te["pmid"], "split": "test", "domain": q_dom}).to_csv(out / "domain_assignment_test.csv", index=False)
    refs = te["text"]
    texts_df = pd.DataFrame({"pmid": te["pmid"], "reference": refs, **texts, **{f"{k}_train_pmid": tr["pmid"][v] for k, v in src.items()}})
    texts_df.to_parquet(out / "texts_by_method.parquet", index=False)
    texts_df.head(60).to_csv(out / "texts_by_method_sample.csv", index=False)

    per = pd.DataFrame({"pmid": np.tile(te["pmid"], len(texts)), "method": np.repeat(list(texts), n)})
    retrieval_rows = []

    # independent embedders: cosine to the reference, rank of the reference, nearest training text
    embedders = [args.embedder] + ([args.extra_embedder] if args.extra_embedder else [])
    nearest_rows = []
    for name in embedders:
        tag = "minilm" if "MiniLM" in name else re.sub(r"[^a-z0-9]+", "_", name.split("/")[-1].lower())
        with timings.time(f"embedder_{tag}"), log_phase(f"embedder {tag}: methods + references (unique texts, one call) + training corpus"):
            emb = Embedder(name, device)
            uniq, pos = unique_index([refs, *texts.values()])
            E = emb(uniq)
            log.info("%s: %d unique texts for %d methods x %d + %d references", tag, len(uniq), len(texts), n, n)
            ref_e = E[[pos[str(t)] for t in refs]]
            cos_col, rank_col = [], []
            for method, tx in texts.items():
                e = E[[pos[str(t)] for t in tx]]
                cos_col.append((e * ref_e).sum(1).numpy())
                metrics, ranks = ranking_metrics((e @ ref_e.T).numpy())
                retrieval_rows.append({"space": tag, "method": method, **metrics})
                rank_col.append(ranks)
            per[f"{tag}_cos"] = np.concatenate(cos_col)
            per[f"{tag}_rank"] = np.concatenate(rank_col)
            n_tr = len(tr["pmid"])
            sub = np.sort(rng.permutation(n_tr)[: args.train_subsample]) if args.train_subsample and args.train_subsample < n_tr else np.arange(n_tr)
            tr_e = torch.cat([emb(bank.train_texts[sub[s : s + 2048]].tolist()) for s in tqdm(range(0, len(sub), 2048), desc=f"{tag} training corpus", leave=False)])
            gen_e = E[[pos[str(t)] for t in texts["neurovlm_generated"]]]
            for label, e in (("generated", gen_e), ("true_reference", ref_e)):
                sims = e @ tr_e.T
                mx, arg = sims.max(dim=1)
                nearest_rows.append(pd.DataFrame({"pmid": te["pmid"], "space": tag, "kind": label, "nearest_train_sim": mx.numpy(), "nearest_train_pmid": tr["pmid"][sub[arg.numpy()]]}))
            emb.close()
            del emb, tr_e, E
            release_cuda_cache()

    # NeuroVLM contrastive space (the paper's metric, circular) and nearest training text in it
    with timings.time("nvlm_space"), log_phase("NeuroVLM contrastive space (SPECTER2 of unique texts, one pass)"):
        img_shared = bank.project_image(te["img_latent"])
        col = []
        uniq, pos = unique_index(list(texts.values()))
        _, shared_all = specter_shared(nvlm, uniq, device)
        for method, tx in texts.items():
            shared = shared_all[[pos[str(t)] for t in tx]]
            col.append((shared * img_shared).sum(1).numpy())
            if method == "neurovlm_generated":
                sims = shared @ bank.text_shared.T
                mx, arg = sims.max(dim=1)
                nearest_rows.append(pd.DataFrame({"pmid": te["pmid"], "space": "nvlm_text_shared", "kind": "generated", "nearest_train_sim": mx.numpy(), "nearest_train_pmid": tr["pmid"][arg.numpy()]}))
        per["nvlm_cos"] = np.concatenate(col)
        ref_shared = bank.project_text(te["text_emb"])  # stored SPECTER2 embeddings of the true test texts
        sims = ref_shared @ bank.text_shared.T
        mx, arg = sims.max(dim=1)
        nearest_rows.append(pd.DataFrame({"pmid": te["pmid"], "space": "nvlm_text_shared", "kind": "true_reference", "nearest_train_sim": mx.numpy(), "nearest_train_pmid": tr["pmid"][arg.numpy()]}))
    nearest = pd.concat(nearest_rows, ignore_index=True)
    nearest.to_parquet(out / "nearest_train_text_per_sample.parquet", index=False)
    diag = nearest.groupby(["space", "kind"])["nearest_train_sim"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    diag["frac_gt_0.90"] = nearest.groupby(["space", "kind"])["nearest_train_sim"].apply(lambda s: float((s > 0.90).mean()))
    diag["frac_gt_0.95"] = nearest.groupby(["space", "kind"])["nearest_train_sim"].apply(lambda s: float((s > 0.95).mean()))
    diag.reset_index().to_csv(out / "nearest_train_diagnostic.csv", index=False)

    if not args.skip_bertscore:
        log.info("released CUDA cache before BERTScore (%s)", release_cuda_cache())
        with timings.time("bertscore"), log_phase("BERTScore (all methods, every unique text embedded once)", methods=len(texts), n=n):
            bert = BertScore(args.bertscore_model, device, args.bertscore_batch_size)
            f1 = bert.f1_pairs({method: (tx, refs) for method, tx in texts.items()})
            per["bertscore_f1"] = np.concatenate([f1[method] for method in texts])
            del bert

    per.to_parquet(out / "per_sample_metrics.parquet", index=False)
    pd.DataFrame(retrieval_rows).to_csv(out / "retrieval_table.csv", index=False)

    from scipy import stats

    log.info("start statistics (bootstrap CIs, paired tests)")
    metric_cols = [c for c in per.columns if c.endswith("_cos") or c == "bertscore_f1"]
    ref_block = per[per["method"] == "neurovlm_generated"].set_index("pmid")
    rows = []
    for metric in metric_cols:
        block = []
        for method, grp in per.groupby("method"):
            g = grp.set_index("pmid")
            vals = g[metric].to_numpy(float)
            lo, hi = bootstrap_mean_ci(vals, args.n_boot, args.seed)
            rec: dict[str, Any] = {"metric": metric, "method": method, "n": int(np.isfinite(vals).sum()), "mean": float(np.nanmean(vals)), "sd": float(np.nanstd(vals, ddof=1)), "ci_lo": lo, "ci_hi": hi}
            if method != "neurovlm_generated":
                a = ref_block.loc[g.index, metric].to_numpy(float)
                b = vals
                diff = a - b
                rec["diff_vs_ref_mean"] = float(diff.mean())
                rec["diff_ci_lo"], rec["diff_ci_hi"] = bootstrap_mean_ci(diff, args.n_boot, args.seed + 1)
                rec["paired_t_p"] = float(stats.ttest_rel(a, b).pvalue) if np.any(diff != 0) else float("nan")
                try:
                    rec["wilcoxon_p"] = float(stats.wilcoxon(a, b, zero_method="zsplit").pvalue)
                except ValueError:
                    rec["wilcoxon_p"] = float("nan")
            block.append(rec)
        bdf = pd.DataFrame(block)
        bdf["paired_t_p_holm"] = holm_adjust(bdf["paired_t_p"].to_numpy())
        bdf["wilcoxon_p_holm"] = holm_adjust(bdf["wilcoxon_p"].to_numpy())
        rows.append(bdf)
    table = pd.concat(rows, ignore_index=True)
    table.to_csv(out / "results_table.csv", index=False)
    log.info("statistics done")

    config = {
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "device": str(device),
        "n_test": int(n),
        "n_train": int(len(tr["pmid"])),
        "n_total_test": int(pub["n_total_test"]),
        "exclusions": pub["exclusions"],
        **M.model_info(),
        "generation_kwargs": GEN_KWARGS,
        "bases": BASES,
        "reference_definition": f"title [SEP] {'LLM summary (pubmed_summaries, as in docs/figures/24)' if args.reference == 'summary' else 'abstract'}",
        "wall_time_s": round(time.perf_counter() - t_start, 1),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, default=str))
    timings.records["total_wall_time_s"] = config["wall_time_s"]
    timings.records["exclusions"] = pub["exclusions"]
    timings.save()
    print(table[table["metric"].isin(["minilm_cos", "bertscore_f1"])][["metric", "method", "n", "mean", "ci_lo", "ci_hi", "paired_t_p_holm"]].to_string(index=False))
    print(diag[["mean", "50%", "frac_gt_0.90"]].to_string())
    print(f"[done] {config['wall_time_s']} s -> {out}")


if __name__ == "__main__":
    main()
