"""Helpers for the modality-restricted text-to-brain analysis: the paper's text-to-brain path, atlas and Neurosynth maps
on the mask grid, and statistics. Each a6_* directory keeps its own copy.
"""
from __future__ import annotations

import contextlib
import functools
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
    """Wall-clock time per named step, merged into ``timings.json``."""

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


def encode_specter(specter, texts: Sequence[str]):
    """L2-normalised SPECTER2 embeddings [N, 768] on CPU."""
    import torch
    import torch.nn.functional as F

    chunks = []
    with torch.no_grad():
        for s in range(0, len(texts), 32):
            e = specter([str(t) for t in texts[s:s + 32]]).float()
            chunks.append(F.normalize(e, dim=1).detach().cpu())
    return torch.cat(chunks)


def load_t2b_models(device) -> dict:
    """SPECTER2, the text-to-brain head and the autoencoder of the selected model (_final_model; runs/final: head v2)."""
    import _final_model as M

    return {"specter": M.component("specter", device), "head": M.component("proj_head_text_mse", device),
            "autoencoder": M.component("autoencoder", device)}


def text_to_brain(models: Mapping, texts: Sequence[str], device) -> np.ndarray:
    """The paper's text-to-brain map on the mask grid: sigmoid(decoder(head(SPECTER2(text)))), calibrated (global factor)."""
    import _final_model as M

    return M.generate_maps(encode_specter(models["specter"], texts), models["head"], models["autoencoder"].decoder, device, batch_size=512).numpy()


@functools.cache
def get_masker():
    from _hf_pins import load_masker

    return load_masker()


def unmask(vec: np.ndarray):
    return get_masker().inverse_transform(np.asarray(vec, dtype=np.float32).reshape(1, -1))


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


HO_SUB_KEEP = ("Thalamus", "Caudate", "Putamen", "Pallidum", "Hippocampus", "Amygdala", "Accumbens")


def harvard_oxford(cache_dir: Path) -> dict:
    """Harvard-Oxford maxprob-thr25 (2 mm) ROIs on the mask grid: 48 cortical + 14 lateralised subcortical."""

    def build():
        from nilearn import datasets

        cort = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
        sub = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
        c_names = [str(x) for x in cort.labels][1:]
        c_masks = _label_vectors(cort.maps, range(1, len(c_names) + 1))
        s_idx = [i for i, n in enumerate(sub.labels) if any(k in str(n) for k in HO_SUB_KEEP)]
        s_names = [str(sub.labels[i]) for i in s_idx]
        s_masks = _label_vectors(sub.maps, s_idx)
        names = np.array(c_names + s_names)
        kind = np.array(["cortical"] * len(c_names) + ["subcortical"] * len(s_names))
        return {"names": names, "kind": kind, "masks": np.concatenate([c_masks, s_masks])}

    return _cached(cache_dir, "atlas_harvard_oxford", build)


_AAL_WORDS = {"Sup": "superior", "Mid": "middle", "Inf": "inferior", "Orb": "orbital", "Oper": "opercular",
              "Tri": "triangular", "Med": "medial", "Ant": "anterior", "Post": "posterior", "Supp": "supplementary",
              "Cingulum": "cingulate gyrus", "Cerebelum": "cerebellum", "Crus1": "crus I", "Crus2": "crus II",
              "ParaHippocampal": "parahippocampal gyrus", "Rolandic": "rolandic", "Paracentral": "paracentral",
              "Lobule": "lobule", "Calcarine": "calcarine cortex", "Cuneus": "cuneus", "Lingual": "lingual gyrus",
              "Fusiform": "fusiform gyrus", "Precuneus": "precuneus", "Heschl": "Heschl's gyrus", "Insula": "insula",
              "Hippocampus": "hippocampus", "Amygdala": "amygdala", "Caudate": "caudate", "Putamen": "putamen",
              "Pallidum": "pallidum", "Thalamus": "thalamus", "Precentral": "precentral gyrus",
              "Postcentral": "postcentral gyrus", "Olfactory": "olfactory cortex", "Rectus": "gyrus rectus",
              "SupraMarginal": "supramarginal gyrus", "Angular": "angular gyrus", "Vermis": "cerebellar vermis",
              "Motor": "motor", "Area": "area", "Pole": "pole", "Occipital": "occipital", "Frontal": "frontal",
              "Temporal": "temporal", "Parietal": "parietal"}


def aal_text(label: str) -> str:
    """'Frontal_Sup_Orb_L' -> 'left superior frontal gyrus, orbital part'."""
    parts = str(label).split("_")
    side = {"L": "left", "R": "right"}.get(parts[-1])
    core = parts[:-1] if side else parts
    if core and core[0] in ("Frontal", "Temporal", "Occipital", "Parietal") and len(core) >= 2:
        lobe = _AAL_WORDS[core[0]]
        mods = [_AAL_WORDS.get(p, p.lower()) for p in core[1:]]
        if "pole" in mods:
            text = f"{lobe} pole" + ("" if len(mods) == 1 else f" ({' '.join(m for m in mods if m != 'pole')})")
        else:
            pos = [m for m in mods if m in ("superior", "middle", "inferior", "medial")]
            rest = [m for m in mods if m not in pos]
            text = " ".join(pos + [lobe, "gyrus"]) + (f", {' '.join(rest)} part" if rest else "")
    elif core and core[0] == "Cingulum" and len(core) == 2:
        text = f"{_AAL_WORDS.get(core[1], core[1].lower())} cingulate gyrus"
    elif core == ["Rolandic", "Oper"]:
        text = "rolandic operculum"
    elif core and core[0] == "Cerebelum":
        rest = [_AAL_WORDS.get(p, p) for p in core[1:]]
        text = "cerebellum " + (" ".join(rest) if rest and rest[0].startswith("crus") else "lobule " + "-".join(rest))
    elif core and core[0] == "Vermis":
        text = "cerebellar vermis lobule " + "-".join(core[1:])
    else:
        text = " ".join(_AAL_WORDS.get(p, p.lower()) for p in core)
    text = re.sub(r"\s+", " ", text).strip()
    return f"{side} {text}" if side else text


def fetch_aal():
    """scripts/pipeline/common.fetch_aal: the AAL atlas from the pinned copy in the data dir."""
    import sys

    sys.path.insert(0, str(HERE.parents[3] / "pipeline"))
    from common import fetch_aal as _fetch

    return _fetch()


def aal(cache_dir: Path) -> dict:
    """AAL (SPM12, 116 lateralised regions) with plain-English names."""

    def build():
        a = fetch_aal()
        labels = [str(x) for x in a.labels]
        idx = [str(x) for x in a.indices]
        keep = [i for i, n in enumerate(labels) if n.lower() != "background"]
        names = [labels[i] for i in keep]
        masks = _label_vectors(a.maps, [int(idx[i]) for i in keep])
        return {"names": np.array(names), "text": np.array([aal_text(n) for n in names]), "masks": masks}

    return _cached(cache_dir, "atlas_aal", build)


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


def zrows(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, np.float64)
    X = X - X.mean(axis=-1, keepdims=True)
    return X / np.clip(X.std(axis=-1, keepdims=True), 1e-12, None)


def pearson_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pearson r between every row of A [n, V] and every row of B [m, V] -> [n, m]."""
    A, B = zrows(np.atleast_2d(A)), zrows(np.atleast_2d(B))
    return (A @ B.T) / A.shape[1]


def deranged_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    """Random permutation without fixed points (zeros for n < 2)."""
    if n < 2:
        return np.zeros(n, dtype=int)
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


def bootstrap_ci(x: Sequence[float], *, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], np.float64)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_test(a: Sequence[float], b: Sequence[float]) -> dict:
    """Paired t-test of a vs b ({} for < 3 pairs or a == b)."""
    from scipy import stats

    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if len(a) < 3 or not np.any(a != b):
        return {}
    t = stats.ttest_rel(a, b)
    return {"t": float(t.statistic), "p_t": float(t.pvalue)}
