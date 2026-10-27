"""Shared helpers for this directory: data access, leakage-audit exclusions, shared-space encoders, text metrics and the
``--model-dir`` loader. Only the installed ``neurovlm`` package, its cached data and the HF cache are used."""
from __future__ import annotations

import contextlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[5]
PAPER_TEMP = 0.035  # canonical projection temperature (scripts/pipeline/validate_models.py::eval_qformer)
BASES = ("network", "region", "function")

# caption cleanup as scripts/models/_common_models.py (validate_models.py)
STEERING_TAG_RE = re.compile(r"(?i)[ \t]*\[(?:network|region|function)\][ \t]*")


def strip_steering_tags(text: str) -> str:
    """Drop the [NETWORK] / [REGION] / [FUNCTION] tags the LM sometimes writes into a caption, with the spaces around
    them (a tag between two words leaves one space); lines without a tag are left as they are."""
    return "\n".join(STEERING_TAG_RE.sub(" ", line).strip() if STEERING_TAG_RE.search(line) else line
                     for line in str(text).split("\n"))


def set_offline_env() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("MPLBACKEND", "Agg")


def seed_all(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(name: str = "auto"):
    import torch

    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def data_dir() -> Path:
    from neurovlm.data import data_dir as _dd

    return Path(_dd)


def load_torch(path: Path | str, *, mmap: bool = True):
    import torch

    try:
        return torch.load(Path(path), map_location="cpu", weights_only=False, mmap=mmap)
    except (RuntimeError, ValueError, TypeError):  # old (non-zip) format cannot be memory-mapped
        return torch.load(Path(path), map_location="cpu", weights_only=False)


def json_safe(value: Any) -> Any:
    import torch

    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if np.isfinite(v) else None
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if torch.is_tensor(value):
        return json_safe(value.detach().cpu().tolist())
    return value


def write_json(path: Path | str, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(obj), indent=2, sort_keys=True))


def read_json(path: Path | str):
    return json.loads(Path(path).read_text())


def get_logger(name: str):
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "urllib3", "huggingface_hub", "sentence_transformers", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger(name)


class Timings:
    """Named wall-clock sections saved to ``timings.json`` (merged with earlier runs)."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.rows: dict[str, dict] = read_json(self.path) if self.path.exists() else {}

    @contextlib.contextmanager
    def time(self, name: str, **meta):
        t0 = time.perf_counter()
        row = dict(meta)
        try:
            yield row
        finally:
            row["seconds"] = time.perf_counter() - t0
            row["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            self.rows[name] = row
            write_json(self.path, self.rows)


@contextlib.contextmanager
def log_phase(log, name: str, **info):
    """Log "start <name>" before and "<name> done in <s>" after a block."""
    extra = ", ".join(f"{k}={v}" for k, v in info.items())
    log.info("start %s%s", name, f" ({extra})" if extra else "")
    t0 = time.perf_counter()
    yield
    log.info("%s done in %.1fs", name, time.perf_counter() - t0)


def release_cuda_cache() -> str:
    """Free the CUDA cache and describe GPU memory. Without it, a model loaded after large-batch generation (e.g. the
    BERTScore DeBERTa) can land in shared system memory on WSL2 and run very slowly."""
    import gc

    import torch

    gc.collect()
    if not torch.cuda.is_available():
        return "cpu"
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    return (f"GPU free {free / 2**30:.1f}/{total / 2**30:.1f} GiB, torch allocated {torch.cuda.memory_allocated() / 2**30:.1f} GiB, "
            f"reserved {torch.cuda.memory_reserved() / 2**30:.1f} GiB")


def deranged_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    """Permutation without fixed points (null pairing)."""
    if n < 2:
        return np.arange(n)
    perm = rng.permutation(n)
    fixed = np.flatnonzero(perm == np.arange(n))
    if len(fixed) > 1:
        perm[fixed] = np.roll(perm[fixed], 1)
    elif len(fixed) == 1:
        i = fixed[0]
        j = (i + 1) % n
        perm[i], perm[j] = perm[j], perm[i]
    return perm


# Leakage-audit exclusions (written by a4_leakage_audit/run_analysis.py --write-exclusions): PubMed test/val papers with a
# near-duplicate in the training split, and NeuroVault papers that are also PubMed training papers (with all their images).
EVAL_EXCLUSIONS_DIR = REPO / "scripts" / "eval_exclusions"


def _norm_doi(s) -> str | None:
    if s is None or (isinstance(s, float) and s != s):
        return None
    s = str(s).strip().lower()
    s = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi:\s*", "", s)
    return s or None


def _split_ids(values) -> set[int]:
    out: set[int] = set()
    for v in values:
        for x in str(v).split(";"):
            x = x.strip()
            if x and x.lower() != "nan":
                out.add(int(float(x)))
    return out


def load_eval_exclusions() -> dict:
    import pandas as pd

    pm = pd.read_csv(EVAL_EXCLUSIONS_DIR / "pubmed_exclusions.csv")
    nv = pd.read_csv(EVAL_EXCLUSIONS_DIR / "neurovault_exclusions.csv", dtype={"doi": str, "collection_id": str, "image_ids": str})
    return {"pubmed_pmids": {int(x) for x in pm["pmid"]},
            "neurovault_dois": {_norm_doi(x) for x in nv["doi"] if _norm_doi(x)},
            "neurovault_image_ids": _split_ids(nv["image_ids"].fillna("")),
            "neurovault_collection_ids": _split_ids(nv["collection_id"].fillna(""))}


def pubmed_keep_mask(pmids, *, enabled: bool = True, label: str = "PubMed", log=print) -> np.ndarray:
    """False for the excluded PMIDs (all True when not ``enabled``)."""
    import pandas as pd

    pmids = np.asarray(pmids)
    if not enabled or len(pmids) == 0:
        return np.ones(len(pmids), dtype=bool)
    excl = load_eval_exclusions()["pubmed_pmids"]
    keep = ~pd.to_numeric(pd.Series(pmids.ravel()), errors="coerce").astype("Int64").isin(excl).fillna(False).to_numpy(dtype=bool)
    if log is not None:
        log(f"[eval exclusions] {label}: dropped {int((~keep).sum())} of {len(keep)} PMIDs (leakage audit)")
    return keep


def neurovault_keep_mask(dois=None, *, image_ids=None, collection_ids=None, enabled: bool = True, label: str = "NeuroVault",
                         log=print) -> np.ndarray:
    """Keep-mask over NeuroVault rows; a row is dropped when its DOI, image id or collection id is excluded."""
    import pandas as pd

    n = next(len(x) for x in (dois, image_ids, collection_ids) if x is not None)
    if not enabled or n == 0:
        return np.ones(n, dtype=bool)
    ex = load_eval_exclusions()
    drop = np.zeros(n, dtype=bool)
    if dois is not None:
        drop |= np.array([_norm_doi(d) in ex["neurovault_dois"] for d in dois], dtype=bool)
    if image_ids is not None:
        drop |= pd.to_numeric(pd.Series(np.asarray(image_ids)), errors="coerce").isin(ex["neurovault_image_ids"]).to_numpy(dtype=bool)
    if collection_ids is not None:
        drop |= pd.to_numeric(pd.Series(np.asarray(collection_ids)), errors="coerce").isin(ex["neurovault_collection_ids"]).to_numpy(dtype=bool)
    if log is not None:
        log(f"[eval exclusions] {label}: dropped {int(drop.sum())} of {n} rows (NeuroVault papers in PubMed train)")
    return ~drop


def load_pubmed_latents():
    """Autoencoder latents [N, 384] and pmids: the run's latent_neuro.pt."""
    payload = load_torch(run_file("latent_neuro"))
    return payload["latent"].float().clone(), np.asarray(payload["pmid"])


def load_specter_latents(pmids: np.ndarray):
    """SPECTER2 (adhoc_query) embeddings [N, 768] aligned to ``pmids``."""
    import torch

    local = data_dir() / "latent_specter2_adhoc.pt"
    if local.exists():
        payload = load_torch(local)
        lat, pm = payload["latent"].float(), np.asarray(payload["pmid"])
    else:
        from _hf_pins import load_latent

        lat, pm = load_latent("pubmed_text")
        lat, pm = torch.as_tensor(lat).float().cpu(), np.asarray(pm)
    pos = {int(p): i for i, p in enumerate(pm)}
    idx = torch.as_tensor([pos[int(p)] for p in pmids], dtype=torch.long)
    return lat[idx].clone()


def load_pubmed_text_rows(pmids: np.ndarray):
    """Title (``name``), abstract (``description``) and ``summary`` aligned to ``pmids``."""
    import pandas as pd
    from _hf_pins import load_dataset

    want = pd.Index(np.asarray(pmids).astype(np.int64), name="pmid")

    def by_pmid(df):  # the HF tables store pmids as Int64 / str
        df = df.copy()
        df["pmid"] = pd.to_numeric(df["pmid"], errors="coerce")
        df = df.dropna(subset=["pmid"])
        df["pmid"] = df["pmid"].astype(np.int64)
        return df.drop_duplicates("pmid").set_index("pmid")

    rows = by_pmid(load_dataset("pubmed_text")).reindex(want)[["name", "description"]].copy()
    try:
        rows["summary"] = by_pmid(load_dataset("pubmed_summaries")).reindex(want)["summary"]
    except Exception:  # summaries are optional
        rows["summary"] = ""
    for col in ("name", "description", "summary"):
        rows[col] = rows[col].fillna("").astype(str).str.strip()
    coverage = float((rows["description"].str.len() > 0).mean()) if len(rows) else 1.0
    if coverage < 0.95:
        raise RuntimeError(f"only {coverage:.1%} of the requested pmids have an abstract in pubmed_text (pmid dtype mismatch?)")
    out = rows.reset_index()
    out["pmid"] = np.asarray(pmids)
    return out


def split_pmids() -> dict[str, np.ndarray]:
    """The run's train / val / test PMIDs (pmids_split.pt)."""
    split = load_torch(run_file("pmids_split"), mmap=False)
    return {k: np.asarray(split[k]).astype(np.int64) for k in ("train", "val", "test")}


def heldout_pmids(split: str = "test", *, exclusions: bool = True, log=print) -> tuple[np.ndarray, dict]:
    """Sorted PMIDs of a held-out split minus the excluded ones, and the exclusion counts."""
    pmids = np.sort(split_pmids()[split])
    keep = pubmed_keep_mask(pmids, enabled=exclusions, label=f"PubMed {split}", log=log)
    return pmids[keep], {"enabled": bool(exclusions), f"n_pubmed_{split}_total": int(len(pmids)),
                         f"n_pubmed_{split}_dropped": int((~keep).sum())}


def load_pubmed_maps(pmids: np.ndarray):
    """Raw flat maps [N, 28542] (fp32, CPU) for ``pmids``."""
    import torch

    nv = data_dir() / "neuro_vectors.pt"
    if nv.exists():
        payload = load_torch(nv, mmap=True)
        vectors, pm = payload["neuro_vectors"], np.asarray(payload["pmid"])
    else:
        from _hf_pins import load_dataset

        vectors, pm = load_dataset("pubmed_images")
        vectors, pm = torch.as_tensor(vectors), np.asarray(pm)
    pos = {int(p): i for i, p in enumerate(pm)}
    idx = torch.as_tensor([pos[int(p)] for p in pmids], dtype=torch.long)
    return vectors[idx].float().clone()


def load_neurovault_one_per_doi(max_n: int | None = None, *, log=print) -> dict:
    """NeuroVault maps, one per paper (lowest image id), scaled by their 99th percentile and clamped to [0, 1]
    (validate_models.neurovault_hf); truth = title [SEP] abstract. Excluded papers and all their maps are dropped first."""
    import torch
    from _hf_pins import load_dataset

    df_text = load_dataset("neurovault_text").reset_index(drop=True)
    meta = load_dataset("neurovault_images_meta").reset_index(drop=True)
    images = np.asarray(load_dataset("neurovault_images"), dtype=np.float32)
    finite = ~np.isnan(images).any(axis=1)
    keep_pub = neurovault_keep_mask(df_text["doi"].to_numpy(), label="NeuroVault publications", log=log)
    keep_img = neurovault_keep_mask(meta["doi"].to_numpy(), image_ids=meta["id"].to_numpy(), collection_ids=meta["collection_id"].to_numpy(),
                                    label="NeuroVault images", log=log)
    excl = {"enabled": True, "n_neurovault_publications_total": int(len(df_text)),
            "n_neurovault_publications_dropped": int((~keep_pub).sum()), "n_neurovault_images_total": int(len(meta)),
            "n_neurovault_images_dropped": int((~keep_img).sum()), "n_neurovault_finite_images_dropped": int((finite & ~keep_img).sum()),
            "dropped_dois": sorted(str(d) for d in df_text["doi"][~keep_pub])}
    df_text = df_text[keep_pub].reset_index(drop=True)
    finite = finite & keep_img
    chosen, truth, ids = [], [], []
    for doi, title, abstract in zip(df_text["doi"], df_text["title"], df_text["abstract"]):
        cand = np.flatnonzero((meta["doi"].to_numpy() == doi) & finite)
        if len(cand) == 0:
            continue
        i = int(cand[np.argmin(meta["id"].to_numpy()[cand])])
        chosen.append(i)
        truth.append(f"{title} [SEP] {abstract}")
        ids.append(f"{doi}|{int(meta['id'].iloc[i])}")
        if max_n and len(chosen) >= max_n:
            break
    X = images[chosen]
    X = np.clip(X / np.quantile(X, 0.99, axis=1, keepdims=True), 0, 1)
    return {"X": torch.from_numpy(X.astype(np.float32)), "ids": ids, "truth": truth, "exclusions": excl}


def encode_specter(specter, texts: Sequence[str], batch_size: int = 32):
    """SPECTER2 (adhoc_query) 768-d embeddings on CPU."""
    import torch

    chunks = []
    with torch.no_grad():
        for s in range(0, len(texts), batch_size):
            chunks.append(torch.as_tensor(specter([str(t) for t in texts[s:s + batch_size]])).detach().float().cpu())
    return torch.cat(chunks, dim=0)


def shared_text(text_head, emb, device, batch_size: int = 1024):
    """768-d text embeddings (L2-normalised first) -> shared InfoNCE space."""
    import torch
    import torch.nn.functional as F

    out = []
    with torch.no_grad():
        for s in range(0, len(emb), batch_size):
            x = F.normalize(emb[s:s + batch_size].float().to(device), dim=1)
            out.append(F.normalize(text_head(x).float(), dim=1).cpu())
    return torch.cat(out, dim=0)


def shared_brain(image_head, latents, device, batch_size: int = 1024):
    import torch
    import torch.nn.functional as F

    out = []
    with torch.no_grad():
        for s in range(0, len(latents), batch_size):
            out.append(F.normalize(image_head(latents[s:s + batch_size].float().to(device)).float(), dim=1).cpu())
    return torch.cat(out, dim=0)


def encode_latents(encoder, X, device, batch_size: int = 512):
    """Autoencoder latents of raw maps, on CPU."""
    import torch

    out = []
    with torch.no_grad():
        for s in range(0, len(X), batch_size):
            out.append(encoder(torch.as_tensor(X[s:s + batch_size]).float().to(device)).detach().float().cpu())
    return torch.cat(out, dim=0)


def text_to_brain_maps(text_emb768, mse_head, decoder, device, batch_size: int = 256):
    """Paper text-to-brain path: SPECTER2 (L2) -> proj_head_text_mse -> autoencoder decoder -> sigmoid; CPU [N, 28542]."""
    import torch
    import torch.nn.functional as F

    out = []
    with torch.no_grad():
        for s in range(0, len(text_emb768), batch_size):
            e = F.normalize(text_emb768[s:s + batch_size].float().to(device), dim=1)
            out.append(torch.sigmoid(decoder(mse_head(e)).float()).cpu())
    return torch.cat(out, dim=0)


def load_minilm(device):
    from _hf_pins import MINILM, sentence_transformer

    return sentence_transformer(MINILM, device)


class BertScore:
    """One cached BERTScorer giving per-pair F1. ``f1_many`` embeds every unique sentence once and matches it against
    several reference lists, with bert_score's own embedding, idf (all 1, 0 for [CLS]/[SEP]), batching and
    ``greedy_cos_idf``, so F1 equals ``BERTScorer.score(cands, refs)[2]`` up to float noise. The model waits on the CPU
    between calls."""

    def __init__(self, model: str, device, log=None):
        self.model_type = {"deberta": "microsoft/deberta-xlarge-mnli", "roberta": "roberta-large"}.get(model)
        self.device, self._scorer, self.note, self.log = device, None, None, log

    def _fail(self, exc: Exception) -> None:
        self.note = f"skipped: {type(exc).__name__}: {exc}"
        if self.log:
            self.log.warning("BERTScore %s", self.note)

    def f1_many(self, cands: Sequence[str], refs: Mapping[str, Sequence[str]], *, desc: str = "BERTScore") -> dict[str, np.ndarray] | None:
        """``{name: F1(cands[i], refs[name][i])}`` for every reference list (each as long as ``cands``); None if unavailable."""
        if self.model_type is None or self.note:
            return None
        try:
            import torch
            from collections import defaultdict
            from torch.nn.utils.rnn import pad_sequence
            from tqdm.auto import tqdm

            if self._scorer is None:
                from bert_score import BERTScorer
                from _hf_pins import bertscore_model

                # pinned snapshot dir + bert_score's layer for the model (BERTScorer takes no revision)
                self._scorer = BERTScorer(**bertscore_model(self.model_type), lang="en", batch_size=32, device="cpu")
            from bert_score.utils import get_bert_embedding, greedy_cos_idf
        except Exception as exc:  # not installed / not cached
            self._fail(exc)
            return None
        sc = self._scorer
        tok, model, bs, all_layers = sc._tokenizer, sc._model, sc.batch_size, sc.all_layers
        device = torch.device(self.device)
        cands = [str(c) for c in cands]
        refs = {k: [str(r) for r in v] for k, v in refs.items()}
        if any(len(v) != len(cands) for v in refs.values()):
            raise ValueError("every reference list must be as long as cands")
        try:
            mem = release_cuda_cache()
            model.to(device)
            idf = defaultdict(lambda: 1.0)
            idf[tok.sep_token_id] = 0
            idf[tok.cls_token_id] = 0
            # as bert_score.utils.bert_cos_score_idf: unique sentences, longest first, embedded in batches without padding
            sents = sorted(set(cands).union(*map(set, refs.values())), key=lambda x: len(x.split(" ")), reverse=True)
            if self.log:
                self.log.info("%s: %s on %s, %d unique texts for %d x %d pairs (%s)", desc, self.model_type, device, len(sents), len(refs),
                              len(cands), mem)
            stats = {}
            with torch.no_grad():
                for s in tqdm(range(0, len(sents), bs), desc=f"{desc}: embed", leave=False):
                    batch = sents[s:s + bs]
                    emb, mask, pidf = get_bert_embedding(batch, model, tok, idf, device=device, all_layers=all_layers)
                    emb, mask, pidf = emb.cpu(), mask.cpu(), pidf.cpu()
                    for i, sen in enumerate(batch):
                        n_tok = int(mask[i].sum().item())
                        stats[sen] = (emb[i, :n_tok], pidf[i, :n_tok])

                def pad_batch_stats(sen_batch):  # bert_score.utils.bert_cos_score_idf.pad_batch_stats
                    emb = [stats[x][0].to(device) for x in sen_batch]
                    w = [stats[x][1].to(device) for x in sen_batch]
                    lens = torch.tensor([e.size(0) for e in emb], dtype=torch.long)
                    pad_mask = (torch.arange(int(lens.max()), dtype=torch.long).expand(len(lens), int(lens.max())) < lens.unsqueeze(1)).to(device)
                    return pad_sequence(emb, batch_first=True, padding_value=2.0), pad_mask, pad_sequence(w, batch_first=True)

                out = {}
                for name, rlist in refs.items():
                    f1 = []
                    for s in tqdm(range(0, len(cands), bs), desc=f"{desc}: match {name}", leave=False):
                        f1.append(greedy_cos_idf(*pad_batch_stats(rlist[s:s + bs]), *pad_batch_stats(cands[s:s + bs]), all_layers)[2].cpu())
                    out[name] = torch.cat(f1, dim=1 if all_layers else 0).numpy()
            return out
        except Exception as exc:  # e.g. CUDA OOM
            self._fail(exc)
            return None
        finally:
            model.to("cpu")
            release_cuda_cache()


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokenize_words(text: str) -> list[str]:
    return _TOKEN_RE.findall(str(text).lower())


def distinct_n(texts: Sequence[str], n: int) -> float:
    """Corpus-level distinct-n: unique / total n-grams over all texts (Li et al. 2016)."""
    total, uniq = 0, set()
    for t in texts:
        toks = tokenize_words(t)
        grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
        total += len(grams)
        uniq.update(grams)
    return len(uniq) / total if total else float("nan")


def mean_pairwise_cosine(emb: np.ndarray) -> float:
    emb = np.asarray(emb, dtype=np.float64)
    n = len(emb)
    if n < 2:
        return float("nan")
    s = emb.sum(axis=0)
    return float((s @ s - np.einsum("ij,ij->", emb, emb)) / (n * (n - 1)))


def nearest_cosine(query, bank, *, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Max cosine and argmax of each L2-normalised query row against an L2-normalised bank (chunked, CPU)."""
    import torch

    q = torch.as_tensor(query).float()
    b = torch.as_tensor(bank).float()
    best, arg = [], []
    with torch.no_grad():
        for s in range(0, len(q), chunk):
            v, i = (q[s:s + chunk] @ b.T).max(dim=1)
            best.append(v)
            arg.append(i)
    return torch.cat(best).numpy(), torch.cat(arg).numpy()


def env_packages(names: Sequence[str] = ("torch", "transformers", "sentence_transformers", "bert_score", "nilearn", "neuromaps",
                                           "brainspace", "sklearn", "scipy", "numpy", "pandas", "matplotlib", "datasets", "psutil")) -> dict:
    import importlib

    out = {}
    for n in names:
        try:
            out[n] = getattr(importlib.import_module(n), "__version__", "?")
        except Exception:
            out[n] = None
    return out


# --model-dir: the run directory (scripts/pipeline/train_pipeline.py layout) holding every trained component. SPECTER2 and
# NeuroQwen3-0.6B are frozen: always the released ones.
DEFAULT_MODEL_DIR = REPO / "scripts" / "outputs" / "runs" / "final"
RUN_FILES = {
    "autoencoder": "autoencoder.pt", "proj_head_text_mse": "proj_head_text_mse.pt", "proj_head_text_infonce": "proj_head_text_infonce.pt",
    "proj_head_image_infonce": "proj_head_image_infonce.pt",
    "qformer_canonical": "neuro_qformer_canonical/model.safetensors", "qformer_pubmed": "neuro_qformer_pubmed/model.safetensors",
    "canonical_semantic_banks": "canonical_semantic_banks.pt", "pmids_split": "pmids_split.pt", "latent_neuro": "latent_neuro.pt",
}
_MODEL_DIR: dict = {"path": None, "sources": {}}


def set_model_dir(path) -> Path:
    """Use the run directory ``path``."""
    _MODEL_DIR["path"] = Path(path).resolve()
    if not _MODEL_DIR["path"].is_dir():
        raise FileNotFoundError(f"--model-dir {path} is not a directory")
    return _MODEL_DIR["path"]


def model_dir() -> Path:
    return _MODEL_DIR["path"]


def run_file(name: str) -> Path:
    p = _MODEL_DIR["path"] / RUN_FILES[name]
    if not p.exists():
        raise FileNotFoundError(f"{p} missing (--model-dir must be a complete run)")
    return p


def repo_relative(p) -> str:
    p = Path(p).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def note_source(name: str, source) -> None:
    _MODEL_DIR["sources"][name] = source if isinstance(source, str) else repo_relative(source)


def model_sources() -> dict:
    """Checkpoint of every loaded component (for metrics.json)."""
    return dict(_MODEL_DIR["sources"])


def component(name: str, device):
    """Trained component of the run directory (whole-module .pt): autoencoder, proj_head_text_mse, proj_head_text_infonce or
    proj_head_image_infonce; ``specter`` is the frozen released SPECTER2 (adhoc_query)."""
    import torch

    if name == "specter":
        from _hf_pins import load_model

        note_source(name, "released:allenai/specter2_aug2023refresh + adhoc_query (frozen)")
        model = load_model(name).to(device)
        model.specter.eval()
        return model
    p = run_file(name)
    note_source(name, p)
    return torch.load(p, map_location="cpu", weights_only=False).to(device).eval()


def _define_mlp_projector():
    """``ResidualMLPProjector`` of scripts/models/_common_models.py, verbatim (the final run's projectors)."""
    import torch
    import torch.nn.functional as F
    from torch import nn

    class ResidualBlock(nn.Module):
        def __init__(self, width: int, expansion: int = 2, dropout: float = 0.05):
            super().__init__()
            self.norm = nn.LayerNorm(width)
            self.fc1 = nn.Linear(width, width * expansion)
            self.act = nn.GELU()
            self.drop = nn.Dropout(dropout)
            self.fc2 = nn.Linear(width * expansion, width)

        def forward(self, x):
            return x + self.fc2(self.drop(self.act(self.fc1(self.norm(x)))))

    class ResidualMLPProjector(nn.Module):
        def __init__(self, image_dim=384, semantic_dim=384, lm_dim=1024, num_queries=32, *, width=704, depth=2, expansion=2,
                     dropout=0.05, head="tokens", align_hidden=512, **_ignored):
            super().__init__()
            self.image_dim, self.semantic_dim, self.lm_dim, self.num_queries = int(image_dim), int(semantic_dim), int(lm_dim), int(num_queries)
            self.width, self.depth, self.expansion, self.head_kind = int(width), int(depth), int(expansion), head
            self.dropout, self.align_hidden = float(dropout), int(align_hidden)
            self.norm_raw, self.norm_sem = nn.LayerNorm(image_dim), nn.LayerNorm(semantic_dim)
            self.in_proj = nn.Linear(image_dim + semantic_dim, width)
            self.blocks = nn.ModuleList([ResidualBlock(width, expansion, dropout) for _ in range(depth)])
            self.out_norm = nn.LayerNorm(width)
            self.head = nn.Linear(width, num_queries * lm_dim if head == "tokens" else lm_dim, bias=False)
            self.query_pos = nn.Parameter(torch.zeros(1, num_queries, lm_dim))
            self.token_norm = nn.LayerNorm(lm_dim)
            self.align_head = nn.Sequential(nn.LayerNorm(lm_dim), nn.Linear(lm_dim, align_hidden), nn.GELU(), nn.Linear(align_hidden, semantic_dim))

        def forward(self, raw_images, semantic_images):
            dtype = self.in_proj.weight.dtype
            h = self.in_proj(torch.cat([self.norm_raw(raw_images.to(dtype)), self.norm_sem(semantic_images.to(dtype))], dim=-1))
            for block in self.blocks:
                h = block(h)
            t = self.head(self.out_norm(h))
            if self.head_kind == "tokens":
                t = t.view(t.size(0), self.num_queries, self.lm_dim)
            else:
                t = t.unsqueeze(1).expand(-1, self.num_queries, -1)
            return self.token_norm(t + self.query_pos)

        def semantic_from_visual(self, visual_tokens):
            x = visual_tokens.to(dtype=self.in_proj.weight.dtype).mean(dim=1)
            return F.normalize(self.align_head(x).float(), dim=1)

    return ResidualMLPProjector


def load_projector(variant: str, device):
    """Stage-1 (``pubmed``) or stage-2 (``canonical``) MLP projector of the run (``neuro_qformer_<variant>/``) wrapped in
    ``NeuroQFormer``. For the canonical variant the run's canonical_semantic_banks.pt replaces the packaged banks."""
    import torch
    import torch.nn.functional as F

    key = f"qformer_{variant}"
    weights = run_file(key)
    from safetensors.torch import load_file
    from neurovlm.models.qformer import CanonicalProjection, NeuroQFormer

    cfg = read_json(weights.parent / "config.json")
    state = dict(load_file(str(weights)))
    note_bank = ""
    if variant == "canonical":
        banks = torch.load(run_file("canonical_semantic_banks"), map_location="cpu", weights_only=False)
        same = True
        for b in ("all", "network", "region", "function"):
            k = f"canonical_projection.{b}_bank"
            new = F.normalize(torch.as_tensor(banks[b]).float(), dim=1)
            old = state.get(k)
            same &= old is not None and old.shape == new.shape and float((old.float() - new).abs().max()) == 0.0
            state[k] = new
        note_bank = " (canonical_semantic_banks.pt " + ("identical to the packaged banks)" if same else "replaced the packaged banks)")
    kind = cfg.get("projector", "qformer")
    if kind != "mlp":
        raise ValueError(f"{weights.parent / 'config.json'}: projector {kind!r}, the final run's is 'mlp'")
    from neurovlm.models.base import ProjHead

    block = dict(cfg.get("qformer", {}))
    pcfg = {k: v for k, v in (cfg.get("projector_config") or {}).items() if k not in ("class", "params")}
    projector = _define_mlp_projector()(**pcfg)
    first, last = state["proj_head_image.aligner.0.weight"], state["proj_head_image.aligner.2.weight"]
    head = ProjHead(int(first.shape[1]), int(first.shape[0]), int(last.shape[0]))
    sem_dim = int(pcfg.get("semantic_dim", 384))
    banks = {n: state.get(f"canonical_projection.{n}_bank", torch.empty(0, sem_dim)) for n in CanonicalProjection.bank_names}
    model = NeuroQFormer(qformer=projector, proj_head_image=head, canonical_banks=banks, semantic_dim=sem_dim,
                         projection_temp=block.get("projection_temp", 0.05), canonical_basis=block.get("canonical_basis", "all"),
                         use_canonical_projection=block.get("use_canonical_projection", True))
    model.load_state_dict(state, strict=True)
    note_source(key, f"{repo_relative(weights.parent)} (projector: {kind}){note_bank}")
    return model.to(device).eval()


def make_nvlm(device, *, ae, text_head, image_head, mse_head):
    """``NeuroVLM`` client that uses the given autoencoder and heads."""
    from neurovlm import NeuroVLM
    from _hf_pins import hf_cache_pinned

    hf_cache_pinned()  # anything the client still lazy-loads comes from the pinned HF cache
    nvlm = NeuroVLM(device=str(device))
    nvlm._autoencoder = ae
    nvlm._proj_head_text_infonce, nvlm._proj_head_image_infonce = text_head, image_head
    nvlm._proj_head_text_mse = mse_head
    note_source("neuroqwen", "released:neurovlm/NeuroQwen3-0.6B (frozen LM)")
    return nvlm


def nvlm_generate(nvlm, projector, variant: str, X, **kw) -> list[str]:
    """``NeuroVLM.generate_text`` with ``projector`` as the brain-to-text module (``variant``: pubmed | canonical) and
    NeuroQwen3 at its pinned revision."""
    from _hf_pins import pin_neuroqwen

    nvlm._neuro_qformer, nvlm._neuro_qformer_variant = projector, variant
    pin_neuroqwen(nvlm)
    out = nvlm.generate_text(X, qformer_variant=variant, **kw)
    return [out] if isinstance(out, str) else list(out)
