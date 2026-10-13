"""Helpers for the steering-token analyses: released models, atlas and Neurosynth maps on the mask grid, statistics,
plot style, PubMed loaders and the leakage-audit exclusions. Each a6_* directory keeps its own copy.
"""
from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import math
import os
import random
import re
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
N_VOXELS = 28_542
DICE_PCT = 90.0  # the paper's dice_percentile(pct=90)
NEUTRAL = "#8a8985"
INK, INK2 = "#1f1e1c", "#52514e"
# written by core/a4_leakage_audit/run_analysis.py --write-exclusions (see its README)
EVAL_EXCLUSIONS_DIR = HERE.parents[3] / "eval_exclusions"


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
    except (RuntimeError, ValueError, TypeError):
        return torch.load(Path(path), map_location="cpu", weights_only=False)


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    return value


def write_json(path: Path | str, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(obj), indent=2))


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text())


class Timings:
    """Wall-clock time per named step, merged into a JSON file."""

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
            row["seconds"] = round(time.perf_counter() - t0, 3)
            row["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            self.rows[name] = row
            write_json(self.path, self.rows)


def get_logger(name: str):
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    return logging.getLogger(name)


def sha_key(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12]


def installed_versions(names: Sequence[str] = ("torch", "numpy", "pandas", "nilearn", "nibabel", "transformers",
                                               "sentence-transformers", "gensim", "scipy", "matplotlib", "neurovlm")) -> dict:
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for n in names:
        try:
            out[n] = version(n)
        except PackageNotFoundError:
            out[n] = None
    return out


@functools.cache
def get_masker():
    from _hf_pins import load_masker

    return load_masker()


def _label_vectors(atlas_img, values: Sequence[int]) -> np.ndarray:
    """Binary [len(values), V] masks from a label image (nearest-neighbour resampling to the mask grid)."""
    import nibabel as nib
    from nilearn.image import resample_to_img

    masker = get_masker()
    img = nib.load(atlas_img) if isinstance(atlas_img, (str, Path)) else atlas_img
    res = resample_to_img(img, masker.mask_img_, interpolation="nearest", force_resample=True, copy_header=True)
    lab = np.rint(np.asarray(masker.transform(res), dtype=np.float32).reshape(-1)).astype(np.int64)
    return np.stack([(lab == int(v)) for v in values]).astype(bool)


def _cached(cache_dir: Path, stem: str, builder):
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{stem}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    out = builder()
    np.savez_compressed(path, **out)
    return out


YEO7_NAMES = ["visual network", "somatomotor network", "dorsal attention network", "ventral attention network",
              "limbic network", "frontoparietal control network", "default mode network"]


def yeo7(cache_dir: Path) -> dict:
    """Yeo 2011 7-network (thick) parcellation."""

    def build():
        from nilearn import datasets

        y = datasets.fetch_atlas_yeo_2011(n_networks=7, thickness="thick")
        masks = _label_vectors(y.maps, range(1, 8))
        return {"names": np.array(YEO7_NAMES), "masks": masks}

    return _cached(cache_dir, "atlas_yeo7", build)


class Neurosynth:
    """Neurosynth v7 term association maps (chi-square, signed z) on the 4 mm mask grid.

    Built from the public v7 release (nimare layout): a study is active at a voxel if it reports a coordinate within
    10 mm, a term is present if its tf-idf > 0.001. Cached as ``neurosynth_term_zmaps_4mm.npz`` (copied from A1 if there).
    """

    A1_CACHE = HERE.parent / "a1_text_to_brain_baselines" / "outputs" / "cache" / "neurosynth_term_zmaps_4mm.npz"
    VOCAB = "data-neurosynth_version-7_vocab-terms_vocabulary.txt"
    FEATURES = "data-neurosynth_version-7_vocab-terms_source-abstract_type-tfidf_features.npz"

    def __init__(self, ns_dir: Path, cache_dir: Path):
        import scipy.sparse as sp

        self.ns_dir = Path(ns_dir)
        if not (self.ns_dir / self.VOCAB).exists() or not (self.ns_dir / self.FEATURES).exists():
            raise FileNotFoundError(f"Neurosynth v7 files not found under {self.ns_dir}; pass --neurosynth-dir "
                                    "(nimare.extract.fetch_neurosynth(version='7', vocab='terms', source='abstract')).")
        self.vocab = (self.ns_dir / self.VOCAB).read_text().splitlines()
        self.index = {t: i for i, t in enumerate(self.vocab) if not re.fullmatch(r"[0-9]+", t)}
        self.presence = (sp.load_npz(self.ns_dir / self.FEATURES).tocsr() > 1e-3).astype(np.float32)  # [studies, terms]
        self.n_studies = np.asarray(self.presence.sum(axis=0)).ravel().astype(int)

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache = cache_dir / "neurosynth_term_zmaps_4mm.npz"
        self.source = f"cache {cache.name}"
        if not cache.exists() and self.A1_CACHE.exists():
            shutil.copyfile(self.A1_CACHE, cache)
            self.source = f"copied from {self.A1_CACHE.relative_to(HERE.parent)}"
        if cache.exists():
            self.z = np.load(cache)["z"]
            if self.z.shape != (len(self.vocab), N_VOXELS):
                raise ValueError(f"{cache}: shape {self.z.shape} does not match the vocabulary ({len(self.vocab)})")
        else:
            self.source = "built from Neurosynth v7 coordinates"
            self.z = self._build().astype(np.float16)
            np.savez_compressed(cache, z=self.z)

    def _build(self) -> np.ndarray:
        import pandas as pd
        import scipy.sparse as sp
        import torch

        presence = self.presence
        coords = pd.read_csv(self.ns_dir / "data-neurosynth_version-7_coordinates.tsv.gz", sep="\t")
        meta_ids = pd.read_csv(self.ns_dir / "data-neurosynth_version-7_metadata.tsv.gz", sep="\t")["id"].to_numpy()
        n_studies = presence.shape[0]
        study_pos = {s: i for i, s in enumerate(meta_ids)}
        mask_img = get_masker().mask_img_
        inv = np.linalg.inv(mask_img.affine)
        xyz = coords[["x", "y", "z"]].to_numpy(dtype=np.float64)
        ijk = np.rint(inv[:3, :3] @ xyz.T + inv[:3, 3:4]).T.astype(np.int64)
        sid = coords["id"].map(study_pos).to_numpy()
        ok = np.isfinite(sid.astype(float))
        ijk, sid = ijk[ok], sid[ok].astype(int)
        # every voxel within 10 mm (2.5 voxels) of a coordinate
        r = 10.0 / 4.0
        rr = np.arange(-int(math.ceil(r)), int(math.ceil(r)) + 1)
        offs = np.array([(a, b, c) for a in rr for b in rr for c in rr if a * a + b * b + c * c <= r * r], dtype=np.int64)
        lookup = -np.ones(mask_img.shape, dtype=np.int64)
        mask_bool = np.asarray(mask_img.dataobj) > 0
        lookup[mask_bool] = np.arange(int(mask_bool.sum()))
        pts = ijk[:, None, :] + offs[None, :, :]
        shape = np.asarray(mask_img.shape)
        valid = np.all((pts >= 0) & (pts < shape), axis=2)
        pts = np.clip(pts, 0, shape - 1)
        vidx = lookup[pts[..., 0], pts[..., 1], pts[..., 2]]
        valid &= vidx >= 0
        s_rep = np.repeat(sid[:, None], offs.shape[0], axis=1)
        pairs = np.unique(np.stack([s_rep[valid], vidx[valid]], axis=1), axis=0)
        ma = sp.csr_matrix((np.ones(len(pairs), np.float32), (pairs[:, 0], pairs[:, 1])), shape=(n_studies, N_VOXELS))
        # 2x2 chi-square per (term, voxel): a = studies with term and activation
        n_t = self.n_studies.astype(np.float64)
        n_v = np.asarray(ma.sum(axis=0)).ravel().astype(np.float64)
        n = float(n_studies)
        pres_t = torch.as_tensor(presence.T.toarray(), dtype=torch.float32)
        z = np.zeros((len(self.vocab), N_VOXELS), np.float32)
        for s in range(0, N_VOXELS, 4096):
            m = torch.as_tensor(ma[:, s:s + 4096].toarray(), dtype=torch.float32)
            a = (pres_t @ m).double().numpy()
            nt, nv = n_t[:, None], n_v[None, s:s + 4096]
            b, c = nt - a, nv - a
            d = n - nt - nv + a
            with np.errstate(invalid="ignore", divide="ignore"):
                chi2 = n * (a * d - b * c) ** 2 / ((a + b) * (c + d) * (a + c) * (b + d))
                sign = np.sign(a / np.clip(nt, 1, None) - c / np.clip(n - nt, 1, None))
            z[:, s:s + 4096] = np.clip(np.nan_to_num(sign * np.sqrt(np.clip(chi2, 0, None)), nan=0.0), -10, 10)
        return z

    def has(self, term: str) -> bool:
        return term in self.index

    def get(self, term: str) -> np.ndarray:
        return np.asarray(self.z[self.index[term]], dtype=np.float32)

    def count(self, term: str) -> int:
        """Number of studies whose abstract uses ``term``."""
        return int(self.n_studies[self.index[term]])


def default_neurosynth_dir() -> Path:
    return Path(os.environ.get("NEUROSYNTH_DIR", str(Path.home() / "nimare_data" / "neurosynth")))


def top_pct(x: np.ndarray, pct: float = DICE_PCT) -> np.ndarray:
    x = np.asarray(x, np.float32).ravel()
    return x > np.percentile(x, pct)


def zrows(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, np.float64)
    X = X - X.mean(axis=-1, keepdims=True)
    return X / np.clip(X.std(axis=-1, keepdims=True), 1e-12, None)


def pearson_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pearson r between every row of A [n, V] and every row of B [m, V] -> [n, m]."""
    A, B = zrows(np.atleast_2d(A)), zrows(np.atleast_2d(B))
    return (A @ B.T) / A.shape[1]


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
        "axes.edgecolor": NEUTRAL, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.labelcolor": INK,
    })


def _tex_escape(s: str) -> str:
    return (s.replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
            .replace("#", "\\#").replace("[", "{[}").replace("]", "{]}"))


def write_tex(df, path: Path | str, *, caption: str, label: str) -> None:
    """Booktabs LaTeX table of a DataFrame of strings."""
    lines = ["\\begin{table}[t]", "\\centering", "\\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\begin{tabular}{" + "l" * len(df.columns) + "}", "\\toprule",
             " & ".join(_tex_escape(str(c)) for c in df.columns) + " \\\\", "\\midrule"]
    lines += [" & ".join(_tex_escape(str(v)) for v in row) + " \\\\" for row in df.itertuples(index=False)]
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    Path(path).write_text("\n".join(lines))


def load_pubmed_text_rows(pmids: np.ndarray, *, min_coverage: float = 0.95):
    """Title (``name``) and abstract (``description``) for ``pmids``; raises if too few have an abstract."""
    import pandas as pd
    from _hf_pins import load_dataset

    want = pd.Index(np.asarray(pmids).astype(np.int64), name="pmid")
    df = load_dataset("pubmed_text").copy()
    df["pmid"] = pd.to_numeric(df["pmid"], errors="coerce")
    df = df.dropna(subset=["pmid"])
    df["pmid"] = df["pmid"].astype(np.int64)
    rows = df.drop_duplicates("pmid").set_index("pmid").reindex(want)[["name", "description"]].copy()
    for col in ("name", "description"):
        rows[col] = rows[col].fillna("").astype(str).str.strip()
    coverage = float((rows["description"].str.len() > 0).mean()) if len(rows) else 1.0
    if coverage < min_coverage:
        raise RuntimeError(f"only {coverage:.1%} of the requested pmids have an abstract (pmid dtype mismatch?)")
    out = rows.reset_index()
    out["pmid"] = np.asarray(pmids).astype(np.int64)
    return out


def load_pubmed_maps(pmids: np.ndarray):
    """Flat maps [N, 28542] (float32 tensor) for ``pmids`` from ``neuro_vectors.pt``."""
    import torch

    nv = data_dir() / "neuro_vectors.pt"
    if nv.exists():
        payload = load_torch(nv, mmap=True)
        vectors, pm = payload["neuro_vectors"], np.asarray(payload["pmid"]).astype(np.int64)
    else:
        from _hf_pins import load_dataset

        vectors, pm = load_dataset("pubmed_images")
        vectors, pm = torch.as_tensor(vectors), np.asarray(pm).astype(np.int64)
    pos = {int(p): i for i, p in enumerate(pm)}
    idx = torch.as_tensor([pos[int(p)] for p in pmids], dtype=torch.long)
    return vectors[idx].float().clone()


def pubmed_keep_mask(pmids, *, enabled: bool = True, label: str = "PubMed", log=print) -> np.ndarray:
    """False for PMIDs listed in scripts/eval_exclusions/pubmed_exclusions.csv; all True when not enabled."""
    import pandas as pd

    pmids = np.asarray(pmids)
    if not enabled:
        return np.ones(len(pmids), dtype=bool)
    excl = pd.read_csv(EVAL_EXCLUSIONS_DIR / "pubmed_exclusions.csv")["pmid"].astype(np.int64).to_numpy()
    keep = ~np.isin(pmids, excl)
    log(f"[eval exclusions] {label}: dropped {int((~keep).sum())} of {len(keep)} PMIDs (leakage audit; --no-exclusions keeps them)")
    return keep
