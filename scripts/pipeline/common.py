"""Shared helpers for the scripts under scripts/: paths, CLI flags, logging, timing, evaluation exclusions, figure style.

Library artifacts live in ``neurovlm.data.data_dir`` (~/.cache/neurovlm); script outputs go to ``--out``
(default scripts/outputs/<stage>). ``--smoke`` caps epochs/batches/samples so a stage runs in seconds.
Training is deterministic by default (``setup_torch``); ``--no-deterministic`` trades bit-exact reruns for speed.
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import json
import logging
import os
import random
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

# cuBLAS reads this when CUDA starts; deterministic mode requires it
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPTS_DIR.parent
DEFAULT_OUT_ROOT = SCRIPTS_DIR / "outputs"
PAPER_FIG_DIR = REPO_ROOT.parent / "paper" / "figures"
EVAL_EXCLUSIONS_DIR = SCRIPTS_DIR / "eval_exclusions"
HF_MANIFEST = SCRIPTS_DIR / "config" / "hf_manifest.json"
# figure sizes (inches), measured with \the\textwidth / \the\textheight: sn-jnl [sn-nature] 372.0 x 552.69 pt,
# response letter (article 11pt, letter paper, 1 in margins) 469.75 x 650.43 pt; a figure is at most 0.6 x text height
FIG_FULL_W = 372.0 / 72.27  # 5.15, paper (sn-jnl) text width
FIG_HALF_W = FIG_FULL_W / 2  # paper figure at 0.5\linewidth
FIG_MAX_H = 0.6 * 552.69 / 72.27  # 4.59, paper
FIG_LETTER_W = 6.5  # response letter text width
FIG_LETTER_MAX_H = 0.6 * 9.0  # 5.4, response letter
FIG_LETTER_PAGE_H = 0.84 * 9.0  # 7.56, a letter figure page of its own (response_letter.tex \rfigpart)


def data_dir() -> Path:
    """Library data dir (~/.cache/neurovlm unless the package says otherwise)."""
    from neurovlm.data import data_dir as dd  # lazy: neurovlm pulls in torch

    path = Path(dd)
    path.mkdir(parents=True, exist_ok=True)
    return path


def fetch_aal():
    """AAL atlas (SPM12) from the copy fetch_hf.py unpacks into the data dir (nilearn's download server fails TLS checks)."""
    from nilearn import datasets

    root = data_dir() / "nilearn_data"
    if not (root / "aal_SPM12" / "aal" / "atlas" / "AAL.nii").exists():
        raise FileNotFoundError(f"{root / 'aal_SPM12'} missing: run scripts/data/fetch_hf.py (neurovlm/atlases)")
    return datasets.fetch_atlas_aal(version="SPM12", data_dir=str(root), verbose=0)


def artifacts_exist(*paths: Path | str, force: bool = False) -> bool:
    """True when every path exists, unless ``force``."""
    return not force and all(Path(p).exists() for p in paths)


def write_json(path: Path | str, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


def read_json(path: Path | str):
    return json.loads(Path(path).read_text())


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


# pinned Hugging Face inputs (scripts/config/hf_manifest.json, written for scripts/data/fetch_hf.py)
@functools.cache
def hf_manifest() -> dict:
    return json.loads(HF_MANIFEST.read_text())


def hf_manifest_sha256() -> str:
    return sha256_file(HF_MANIFEST)


@functools.cache
def _verified(path: str, size: int, mtime: float, digest: str) -> bool:
    return sha256_file(path) == digest


def hf_pinned(repo_id: str, path: str | None = None, repo_type: str | None = None):
    """Pinned revision of ``repo_id`` from scripts/config/hf_manifest.json (the main-branch pin when a repo has several).

    With ``path``: the local file at that revision, sha256-checked. Looked up in the HF cache, then in data_dir under
    the manifest's local_name (what fetch_hf.py writes), else downloaded with hf_hub_download.
    """
    m = hf_manifest()
    rows = [e for e in m["files"] + m.get("pins", []) if e["repo"] == repo_id and repo_type in (None, e["repo_type"])]
    if path is None:
        rows = [e for e in rows if e.get("ref", "main") == "main"] or rows
        revs = {e["revision"] for e in rows}
        if len(revs) != 1:
            raise KeyError(f"{repo_id}: {len(revs)} pinned revisions in {HF_MANIFEST}; expected one (add the repo with its main sha)")
        return revs.pop()
    rows = [e for e in rows if e.get("path") == path]
    if len(rows) != 1:
        raise KeyError(f"{repo_id}/{path} is not pinned in {HF_MANIFEST}; add it (revision and sha256 from the HF tree)")
    e = rows[0]
    from huggingface_hub import hf_hub_download, try_to_load_from_cache

    cands = []
    hit = try_to_load_from_cache(repo_id, path, revision=e["revision"], repo_type=e["repo_type"])
    if isinstance(hit, str):
        cands.append(Path(hit))
    if e.get("local_name"):
        cands.append(data_dir() / e["local_name"])
    for p in cands:
        if p.is_file() and _verified(str(p), p.stat().st_size, p.stat().st_mtime, e["sha256"]):
            return p
    p = Path(hf_hub_download(repo_id, path, repo_type=e["repo_type"], revision=e["revision"]))
    if not _verified(str(p), p.stat().st_size, p.stat().st_mtime, e["sha256"]):
        raise RuntimeError(f"{p} ({repo_id}/{path}@{e['revision'][:8]}) does not match the pinned sha256")
    return p


def hf_pinned_dir(repo_id: str) -> Path:
    """Snapshot dir of ``repo_id`` at its pinned main revision, with every manifest file of that revision checked.

    For loaders without a ``revision`` argument (bert_score). Repos pinned by revision only get their config,
    tokenizer and safetensors files, unchecked.
    """
    rev = hf_pinned(repo_id)
    files = [e for e in hf_manifest()["files"] if e["repo"] == repo_id and e["revision"] == rev]
    if not files:
        from huggingface_hub import snapshot_download

        e = next(e for e in hf_manifest()["pins"] if e["repo"] == repo_id)
        return Path(snapshot_download(repo_id, repo_type=e["repo_type"], revision=rev,
                                      allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors"]))
    for e in files:
        p = hf_pinned(repo_id, e["path"], e["repo_type"])
    return p.parents[len(Path(e["path"]).parts) - 1]  # snapshots/<revision>/


def hf_split_pmids() -> dict:
    """{"train"|"val"|"test": sorted int64 PMIDs} from the split flags of the pinned neuro_image_papers/publications.parquet.

    Set-identical to runs/final/pmids_split.pt (stage 08's split); consumers use them as sets.
    """
    import pandas as pd

    df = pd.read_parquet(hf_pinned("neurovlm/neuro_image_papers", "publications.parquet"), columns=["pmid", "train", "val", "test"])
    return {k: np.sort(df.loc[df[k].fillna(False).astype(bool), "pmid"].to_numpy(dtype=np.int64)) for k in ("train", "val", "test")}


# evaluation exclusions (leakage audit, see scripts/eval_exclusions/README.md)
def norm_doi(s) -> str | None:
    """Lower-case DOI without the ``https://doi.org/`` or ``doi:`` prefix."""
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
    """The excluded PubMed PMIDs and NeuroVault DOIs / image ids / collection ids from scripts/eval_exclusions/."""
    import pandas as pd

    d = EVAL_EXCLUSIONS_DIR
    pm_path, nv_path = d / "pubmed_exclusions.csv", d / "neurovault_exclusions.csv"
    if not pm_path.exists() or not nv_path.exists():
        raise FileNotFoundError(f"{pm_path} / {nv_path} missing; regenerate with "
                                "scripts/figures/review/core/a4_leakage_audit/run_analysis.py --write-exclusions scripts/eval_exclusions "
                                "or pass --no-exclusions")
    pm = pd.read_csv(pm_path)
    nv = pd.read_csv(nv_path, dtype={"doi": str, "pmid": str, "collection_id": str, "image_ids": str, "matched_train_pmid": str})
    return {
        "pubmed_pmids": {int(x) for x in pm["pmid"]},
        "pubmed": pm,
        "neurovault": nv,
        "neurovault_dois": {norm_doi(x) for x in nv["doi"] if norm_doi(x)},
        "neurovault_image_ids": _split_ids(nv["image_ids"].fillna("")),
        "neurovault_collection_ids": _split_ids(nv["collection_id"].fillna("")),
        "dir": d,
    }


def apply_pubmed_exclusions(pmids, *, enabled: bool = True, logger: logging.Logger | None = None, label: str = "PubMed") -> np.ndarray:
    """Keep-mask over ``pmids`` (int or str): False for the excluded PMIDs, all True when not ``enabled``."""
    import pandas as pd

    pmids = np.asarray(pmids)
    if not enabled or len(pmids) == 0:
        if logger is not None and not enabled:
            logger.info("eval exclusions [%s]: disabled (--no-exclusions); all %d PMIDs kept", label, len(pmids))
        return np.ones(len(pmids), dtype=bool)
    excl = load_eval_exclusions()["pubmed_pmids"]
    as_int = pd.to_numeric(pd.Series(pmids.ravel()), errors="coerce").astype("Int64")
    keep = ~as_int.isin(excl).fillna(False).to_numpy(dtype=bool)
    if logger is not None:
        logger.info("eval exclusions [%s]: dropped %d of %d PMIDs (leakage audit; --no-exclusions to keep)", label, int((~keep).sum()), len(keep))
    return keep


def apply_neurovault_exclusions(dois=None, *, image_ids=None, collection_ids=None, enabled: bool = True,
                                logger: logging.Logger | None = None, label: str = "NeuroVault") -> np.ndarray:
    """Keep-mask over NeuroVault items given by DOI, image id and/or collection id; any excluded id drops the item."""
    import pandas as pd

    n = next(len(x) for x in (dois, image_ids, collection_ids) if x is not None)
    if not enabled or n == 0:
        if logger is not None and not enabled:
            logger.info("eval exclusions [%s]: disabled (--no-exclusions); all %d items kept", label, n)
        return np.ones(n, dtype=bool)
    ex = load_eval_exclusions()
    drop = np.zeros(n, dtype=bool)
    if dois is not None:
        drop |= np.array([norm_doi(d) in ex["neurovault_dois"] for d in dois], dtype=bool)
    if image_ids is not None:
        drop |= pd.to_numeric(pd.Series(np.asarray(image_ids)), errors="coerce").isin(ex["neurovault_image_ids"]).to_numpy(dtype=bool)
    if collection_ids is not None:
        drop |= pd.to_numeric(pd.Series(np.asarray(collection_ids)), errors="coerce").isin(ex["neurovault_collection_ids"]).to_numpy(dtype=bool)
    if logger is not None:
        logger.info("eval exclusions [%s]: dropped %d of %d items (NeuroVault papers in PubMed train; --no-exclusions to keep)", label, int(drop.sum()), n)
    return ~drop


# CLI
@dataclass
class SmokeCaps:
    """Caps used with ``--smoke``."""

    epochs: int = 1
    max_batches: int = 1
    max_samples: int = 64


def base_parser(description: str, *, stage: str) -> argparse.ArgumentParser:
    """Parser with the flags every stage shares."""
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_ROOT / stage, help="Output dir for logs, timings and figures.")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:N or mps.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-deterministic", dest="deterministic", action="store_false",
                   help="Allow non-deterministic CUDA kernels and cudnn.benchmark (faster, reruns not bit-identical).")
    p.add_argument("--force", action="store_true", help="Recompute even if cached.")
    p.add_argument("--smoke", action="store_true", help="Tiny run (1 epoch, few batches and samples).")
    p.add_argument("--max-samples", type=int, default=None, help="Limit samples (overrides the smoke default).")
    p.add_argument("--epochs", type=int, default=None, help="Override the stage's epochs.")
    p.add_argument("--batch-size", type=int, default=None, help="Override the stage's batch size.")
    return p


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    """Apply the smoke caps and create ``--out``; call right after ``parse_args``."""
    args.caps = SmokeCaps() if args.smoke else None
    if args.smoke and args.max_samples is None:
        args.max_samples = args.caps.max_samples
    if args.smoke and args.epochs is None:
        args.epochs = args.caps.epochs
    args.out = Path(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    return args


# torch
def select_device(prefer: str = "auto"):
    import torch

    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_torch(seed: int = 0, deterministic: bool = True) -> None:
    """Seed python/numpy/torch and turn on TF32; deterministic kernels unless ``deterministic`` is False (then cudnn.benchmark)."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic
    get_logger("setup_torch").info(
        "seed %d | deterministic %s (use_deterministic_algorithms %s, cudnn.deterministic %s, cudnn.benchmark %s, "
        "CUBLAS_WORKSPACE_CONFIG %s) | TF32 matmul %s, float32 matmul precision %s | torch %s, CUDA %s, cuDNN %s",
        seed, deterministic, torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark, os.environ.get("CUBLAS_WORKSPACE_CONFIG"), torch.backends.cuda.matmul.allow_tf32,
        torch.get_float32_matmul_precision(), torch.__version__, torch.version.cuda, torch.backends.cudnn.version())


def deterministic_settings(deterministic: bool = True) -> dict:
    """The torch settings setup_torch(deterministic=...) applies (stages 13/14 also turn cuDNN attention off)."""
    return {"deterministic": deterministic, "use_deterministic_algorithms": deterministic, "cudnn.deterministic": deterministic,
            "cudnn.benchmark": not deterministic, "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cudnn_sdpa": not deterministic, "tf32_matmul": True, "float32_matmul_precision": "high", "seeded_dataloaders": True}


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def provenance(deterministic: bool | None = None, group: str = "pipeline") -> dict:
    """Run record: pinned HF inputs (manifest sha256 + revisions of ``group``), torch/CUDA/cuDNN, GPU and driver, determinism,
    git commit, command line. ``deterministic`` None = read the live torch flags (this process ran setup_torch)."""
    import torch

    revs: dict = {}
    for e in hf_manifest()["files"]:
        if group in e["groups"]:
            revs.setdefault(e["repo"], {})[e.get("ref", "main")] = e["revision"]
    if deterministic is None:
        det = {"use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
               "cudnn.deterministic": torch.backends.cudnn.deterministic, "cudnn.benchmark": torch.backends.cudnn.benchmark,
               "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
               "float32_matmul_precision": torch.get_float32_matmul_precision()}
    else:
        det = deterministic_settings(deterministic)
    # GPU from nvidia-smi, so a launcher process does not create a CUDA context
    smi = _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    gpus = [dict(zip(("name", "driver"), (x.strip() for x in line.split(",")))) for line in (smi or "").splitlines() if line.strip()]
    commit = _run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])
    changed = _run(["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no"])
    import huggingface_hub
    import transformers

    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "command": shlex.join([sys.executable] + sys.argv),
        "git": {"commit": commit, "modified_files": None if changed is None else len(changed.splitlines())},
        "hf_manifest": {"path": str(HF_MANIFEST.relative_to(REPO_ROOT)), "sha256": hf_manifest_sha256()},
        "hf_revisions": revs,
        "software": {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
                     "cudnn": torch.backends.cudnn.version(), "transformers": transformers.__version__,
                     "huggingface_hub": huggingface_hub.__version__, "numpy": np.__version__},
        "gpu": gpus, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "determinism": det,
    }


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: seed python/numpy in each worker from its torch seed."""
    import torch

    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


# timing
class Timer:
    """``with Timer("name", log) as t: ...``, then ``t.seconds``."""

    def __init__(self, name: str, logger: logging.Logger | None = None):
        self.name, self.logger, self.seconds = name, logger, 0.0

    def __enter__(self):
        self.t0 = time.perf_counter()
        if self.logger:
            self.logger.info("start %s", self.name)
        return self

    def __exit__(self, *exc):
        with contextlib.suppress(Exception):  # wait for pending GPU work
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        self.seconds = time.perf_counter() - self.t0
        if self.logger:
            self.logger.info("%s done in %.1fs", self.name, self.seconds)


@contextlib.contextmanager
def timed(name: str, logger: logging.Logger | None, out: Path | None = None, **meta) -> Iterator[Timer]:
    """Time a block and append ``{name, seconds, **meta, ts}`` to ``out/timings.jsonl``."""
    with Timer(name, logger) as t:
        yield t
    if out is not None:
        record_timing(out, name=name, seconds=t.seconds, **meta)


def record_timing(out: Path, **row) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    row.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S"))
    with (out / "timings.jsonl").open("a") as fh:
        fh.write(json.dumps(row, default=str) + "\n")


# figures
def apply_paper_style(font_size: float = 8.0) -> None:
    """Matplotlib rcParams for print-legible panels at the paper's figure widths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": font_size,
        "axes.titlesize": font_size + 1,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.4,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    })


def save_fig(fig, stem: str, out: Path) -> list[Path]:
    """Save ``fig`` as ``out/stem.png`` and ``out/stem.svg``; returns the paths."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in ("png", "svg"):
        path = out / f"{stem}.{fmt}"
        fig.savefig(path, format=fmt)
        written.append(path)
    return written
