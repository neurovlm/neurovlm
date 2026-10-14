"""Helpers for the MLP-projector study: PubMed data, eval exclusions, the frozen LM, brain-to-text metrics
(validity, BERTScore) and figure/table styling.  Each a5_* directory has its own copy.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

FIG_FULL_W = 7.0
# colour-blind-safe categorical palette
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
NEUTRAL = "#8a8985"


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


def load_torch(path: Path | str, *, mmap: bool = True):
    import torch

    try:
        return torch.load(Path(path), map_location="cpu", weights_only=False, mmap=mmap)
    except (RuntimeError, ValueError, TypeError):
        return torch.load(Path(path), map_location="cpu", weights_only=False)


def write_json(path: Path | str, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(obj), indent=2, sort_keys=True))


def read_json(path: Path | str):
    return json.loads(Path(path).read_text())


def json_safe(value: Any) -> Any:
    import torch

    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
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


def count_params(module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def sha_key(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12]


def setup_torch(seed: int) -> None:
    """Seeds + the CUDA speed flags of scripts/pipeline/common.py::setup_torch (TF32, cudnn.benchmark), as in scripts/models/13."""
    import torch

    seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True


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
            self.save()

    def save(self) -> None:
        write_json(self.path, self.rows)


@contextlib.contextmanager
def log_phase(log, name: str, **info):
    """Log "start <name>" and "<name> done in Xs" around a block."""
    extra = ", ".join(f"{k}={v}" for k, v in info.items())
    log.info("start %s%s", name, f" ({extra})" if extra else "")
    t0 = time.perf_counter()
    yield
    log.info("%s done in %.1fs", name, time.perf_counter() - t0)


def release_cuda_cache() -> str:
    """Hand the CUDA allocator's cached blocks back to the driver (no-op on CPU) and describe GPU memory.

    Without this, a model loaded after generation can spill into shared system memory on WSL2 and run very slowly.
    """
    import gc

    import torch

    gc.collect()
    if not torch.cuda.is_available():
        return "cpu"
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    return (f"GPU free {free / 2**30:.1f}/{total / 2**30:.1f} GiB, torch allocated {torch.cuda.memory_allocated() / 2**30:.1f} GiB, "
            f"reserved {torch.cuda.memory_reserved() / 2**30:.1f} GiB")


def get_logger(name: str):
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    return logging.getLogger(name)


# PubMed data (HF tables: the inputs the released Q-Former was trained on)
def load_pubmed_latents():
    """Released autoencoder latents [N, 384] + pmids (HF pubmed_images)."""
    import torch
    from _hf_pins import load_latent

    lat, pm = load_latent("pubmed_images")
    return torch.as_tensor(lat).float().cpu().clone(), np.asarray(pm)


def load_specter_latents(pmids: np.ndarray):
    """SPECTER2 (adhoc_query) embeddings [N, 768] from HF pubmed_text, aligned to ``pmids``."""
    import torch
    from _hf_pins import load_latent

    lat, pm = load_latent("pubmed_text")
    lat, pm = torch.as_tensor(lat).float().cpu(), np.asarray(pm)
    pos = {p: i for i, p in enumerate(pm)}
    idx = torch.as_tensor([pos[p] for p in pmids], dtype=torch.long)
    return lat[idx].clone()


def split_masks(pmids: np.ndarray) -> dict[str, np.ndarray]:
    """Train/val/test masks from the paper's split (the pinned HF publications.parquet flags = runs/final pmids_split.pt)."""
    import pandas as pd
    from _hf_pins import hf_split_pmids

    split = hf_split_pmids()  # split flags of the pinned HF publications.parquet, never the data-dir copy
    pm = pd.Series(pmids)
    return {k: pm.isin(np.asarray(split[k])).to_numpy() for k in ("train", "val", "test")}


def load_pubmed_maps(pmids: np.ndarray):
    """Raw flat maps [N, 28542] (fp32, CPU) for ``pmids``, from the pinned HF pubmed_images.pt (as validate_models.py)."""
    import torch
    from _hf_pins import hf_file

    payload = load_torch(hf_file("neurovlm/embedded_text", "pubmed_images.pt"), mmap=True)
    vectors, pm = payload["neuro_vectors"], np.asarray(payload["pmid"])
    pos = {p: i for i, p in enumerate(pm)}
    idx = torch.as_tensor([pos[p] for p in pmids], dtype=torch.long)
    return vectors[idx].float().clone()


# evaluation exclusions: held-out PubMed papers with a near-duplicate in the training split (16 test + 26 val PMIDs),
# written by a4_leakage_audit/run_analysis.py --write-exclusions
EVAL_EXCLUSIONS_DIR = Path(__file__).resolve().parents[5] / "scripts" / "eval_exclusions"


def pubmed_keep_mask(pmids, *, enabled: bool = True, label: str = "PubMed", log=print) -> np.ndarray:
    """False for the excluded PMIDs; all True when ``enabled`` is False."""
    import pandas as pd

    pmids = np.asarray(pmids)
    if not enabled or len(pmids) == 0:
        return np.ones(len(pmids), dtype=bool)
    excl = {int(x) for x in pd.read_csv(EVAL_EXCLUSIONS_DIR / "pubmed_exclusions.csv")["pmid"]}
    keep = ~pd.to_numeric(pd.Series(pmids.ravel()), errors="coerce").astype("Int64").isin(excl).fillna(False).to_numpy(dtype=bool)
    if log is not None:
        log(f"eval exclusions, {label}: dropped {int((~keep).sum())} of {len(keep)} PMIDs")
    return keep


def exclude_heldout_rows(masks: Mapping[str, np.ndarray], pmids, *, enabled: bool = True, splits: Sequence[str] = ("val", "test"),
                         log=print) -> tuple[dict[str, np.ndarray], dict]:
    """Drop the excluded PMIDs from the val/test masks (train untouched). Returns (masks, the "exclusions" record)."""
    pmids = np.asarray(pmids)
    out = {k: np.asarray(v, dtype=bool).copy() for k, v in masks.items()}
    block: dict[str, Any] = {"enabled": bool(enabled), "source": "scripts/eval_exclusions/pubmed_exclusions.csv"}
    if not enabled and log is not None:
        log("eval exclusions disabled (--no-exclusions)")
    for s in splits:
        rows = np.flatnonzero(out[s])
        keep = pubmed_keep_mask(pmids[rows], enabled=enabled, label=f"PubMed {s}", log=log)
        out[s][rows[~keep]] = False
        block[f"pubmed_{s}_dropped"] = int((~keep).sum())
        block[f"pubmed_{s}_kept"] = int(keep.sum())
    return out, block


def same_exclusions(prev: Mapping | None, enabled: bool) -> bool:
    """True when earlier outputs used the same exclusions setting (no record = disabled)."""
    return bool(((prev or {}).get("exclusions") or {}).get("enabled", False)) == bool(enabled)


# models
def released(name: str, device):
    """``neurovlm.models.load_model(name)`` on ``device`` in eval mode."""
    from _hf_pins import load_model

    model = load_model(name)
    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    elif hasattr(model, "specter"):
        model.specter.eval()
    return model


def load_lm(device, dtype=None):
    """Frozen NeuroQwen3-0.6B, tokenizer and input-embedding statistics (scripts/models/13)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from neurovlm.resources.loaders import NEURO_QWEN_REPO_ID
    from _hf_pins import hf_revision

    dtype = dtype or (torch.bfloat16 if device.type == "cuda" else torch.float32)
    rev = hf_revision(NEURO_QWEN_REPO_ID)
    tok = AutoTokenizer.from_pretrained(NEURO_QWEN_REPO_ID, revision=rev, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(NEURO_QWEN_REPO_ID, revision=rev, torch_dtype=dtype, attn_implementation="sdpa").to(device).eval()
    for p in lm.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        emb = lm.get_input_embeddings().weight.detach().float()
        stats = {"mean": emb.mean().item(), "std": emb.std().item(), "norm": emb.norm(dim=1).mean().item()}
    return lm, tok, stats


def match_lm_token_norm(vis, target_norm: float):
    import torch

    target = torch.tensor(target_norm, device=vis.device, dtype=vis.dtype)
    norm = vis.float().norm(dim=-1, keepdim=True).clamp_min(1e-6).to(vis.dtype)
    return vis * (target / norm)


# brain-to-text metrics (scripts/pipeline/validate_models.py)
FABRICATION_PATTERNS = {
    "sample_size": re.compile(r"\b[nN]\s*=\s*\d+|\b\d+\s+(?:participants|subjects|patients|healthy|volunteers|adults|children)\b"),
    "p_value": re.compile(r"\bp\s*[<=>]\s*0?\.\d+|\bp\s*[<=>]\s*\d\.\d+e-\d+", re.IGNORECASE),
    "percentage": re.compile(r"\d+(?:\.\d+)?\s*%"),
    "year": re.compile(r"\b(?:19|20)\d{2}\b"),
    "citation": re.compile(r"\bet al\.?|\[\d+\]"),
}


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


def validity_checks(texts: Sequence[str]) -> dict:
    """Output-validity checks of scripts/pipeline/validate_models.py: degenerate, duplicated, fabricated specifics."""
    stripped = [re.sub(r"^\s*\[[A-Z]+\]\s*", "", str(t)) for t in texts]
    stripped = [re.sub(r"\s+", " ", t).strip() for t in stripped]
    n = max(len(stripped), 1)
    empty = sum(len(t) < 3 for t in stripped)
    lens = [len(t.split()) for t in stripped]

    def repeat_rate(t: str) -> float:
        toks = t.split()
        if len(toks) < 8:
            return 0.0
        grams = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
        return 1.0 - len(set(grams)) / len(grams)

    rep = [repeat_rate(t) for t in stripped]
    counts: dict[str, int] = {}
    for t in stripped:
        counts[t] = counts.get(t, 0) + 1
    duplicated = sum(c for c in counts.values() if c > 1)
    degenerate = sum(1 for t, r in zip(stripped, rep) if len(t) < 3 or r > 0.3 or len(t.split()) < 3)
    fab = {k: float(np.mean([bool(p.search(t)) for t in stripped])) for k, p in FABRICATION_PATTERNS.items()}
    return {
        "validity_n": len(stripped), "validity_frac_empty": empty / n, "validity_frac_degenerate": degenerate / n,
        "validity_frac_duplicated_outputs": duplicated / n, "validity_n_unique": len(counts),
        "validity_frac_unique": len(counts) / n,
        "validity_mean_words": float(np.mean(lens)) if lens else 0.0, "validity_median_words": float(np.median(lens)) if lens else 0.0,
        "validity_mean_4gram_repeat_rate": float(np.mean(rep)) if rep else 0.0,
        "validity_fabrication_rate": fab,
        "validity_fabrication_any": float(np.mean([any(p.search(t) for p in FABRICATION_PATTERNS.values()) for t in stripped])),
        "examples": stripped[:5],
    }


def load_minilm(device):
    from _hf_pins import MINILM, sentence_transformer

    return sentence_transformer(MINILM, device)


class BertScore:
    """BERTScore F1 with one cached scorer kept on the CPU between calls.

    ``f1_many`` embeds every unique sentence once and scores it against several reference lists (bert_score's own
    embedding, idf=False weights and ``greedy_cos_idf``), so F1 equals ``BERTScorer.score(cands, refs)[2]`` up to float noise.
    """

    def __init__(self, model: str, device, log=None):
        self.model_type = {"deberta": "microsoft/deberta-xlarge-mnli", "roberta": "roberta-large"}.get(model)
        self.device, self._scorer, self.note, self.log = device, None, None, log

    def _fail(self, exc: Exception) -> None:
        self.note = f"skipped: {type(exc).__name__}: {exc}"
        if self.log:
            self.log.warning("BERTScore %s", self.note)

    def f1_many(self, cands: Sequence[str], refs: Mapping[str, Sequence[str]], *, desc: str = "BERTScore") -> dict[str, np.ndarray] | None:
        """``{name: F1(cands[i], refs[name][i])}`` for every named reference list (each as long as ``cands``)."""
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
        except Exception as exc:  # optional dependency or missing cache
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
            # bert_cos_score_idf: unique sentences sorted by word count (longest first), embedded in batches, padding cut off
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

                def pad_batch_stats(sen_batch):  # == bert_score.utils.bert_cos_score_idf.pad_batch_stats
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


_BERTSCORERS: dict[tuple[str, str], BertScore] = {}


def cached_bertscore(model: str, device, log=None) -> BertScore:
    """One ``BertScore`` per (model, device) and process."""
    key = (model, str(device))
    if key not in _BERTSCORERS:
        _BERTSCORERS[key] = BertScore(model, device, log)
    return _BERTSCORERS[key]


# figures and tables
def apply_style(font_size: float = 8.0) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": font_size, "axes.titlesize": font_size + 1, "axes.labelsize": font_size,
        "xtick.labelsize": font_size, "ytick.labelsize": font_size, "legend.fontsize": font_size,
        "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
        "lines.linewidth": 1.6, "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.edgecolor": "#8a8985", "xtick.color": "#52514e", "ytick.color": "#52514e",
    })


def save_fig(fig, stem: str, out: Path | str) -> list[Path]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in ("png", "svg"):
        p = out / f"{stem}.{fmt}"
        fig.savefig(p, format=fmt)
        paths.append(p)
    return paths


def write_table(df, stem: str, out: Path | str, *, caption: str, label: str, float_format: str = "%.3f") -> list[Path]:
    """``<out>/<stem>.csv`` + ``<out>/<stem>.tex`` (booktabs)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    csv_path, tex_path = out / f"{stem}.csv", out / f"{stem}.tex"
    df.to_csv(csv_path, index=False, float_format=float_format)
    cols = list(df.columns)
    lines = ["\\begin{table}[t]", "\\centering", "\\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\begin{tabular}{l" + "r" * (len(cols) - 1) + "}", "\\toprule",
             " & ".join(_tex_escape(c) for c in cols) + " \\\\", "\\midrule"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append("--" if not np.isfinite(v) else float_format % v)
            else:
                cells.append(_tex_escape(str(v)))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    tex_path.write_text("\n".join(lines))
    return [csv_path, tex_path]


def _tex_escape(s: str) -> str:
    return s.replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#")
