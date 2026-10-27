"""Neurosynth and NeuroQuery text-to-map baselines used by run_text_to_brain.py.

Neurosynth v7 is fetched by NiMARE into ~/nimare_data and NeuroQuery's model by ``fetch_neuroquery_model`` (both from their
own servers, not pinned on Hugging Face); the Neurosynth term maps are cached in outputs/cache/.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

N_VOXELS = 28_542
_NUMERIC = re.compile(r"^[\d\s.]+$")


class NeurosynthTextToMap:
    """Neurosynth v7 text -> map: tf x idf weighted mean of the term association maps of the terms in the text.

    Association z (4 mm, NeuroVLM mask): signed sqrt of the 2x2 chi-square between term presence (tf-idf > 0.001)
    and voxel activation (MKDA 10 mm spheres), as in Neurosynth.
    """

    def __init__(self, data_dir: Path, masker: Any, cache_dir: Path, device: torch.device) -> None:
        self.data_dir = data_dir
        self.masker = masker
        self.cache_dir = cache_dir
        self.device = device
        self.z: np.ndarray | None = None
        self.vocab: list[str] = []
        self.idf: np.ndarray | None = None
        self.status: dict[str, Any] = {"available": False}

    def _files(self) -> dict[str, Path] | None:
        try:
            from nimare.extract import fetch_neurosynth

            files = fetch_neurosynth(data_dir=str(self.data_dir), version="7", return_type="files", vocab="terms", source="abstract")
            f = files[0]
            return {"coordinates": Path(f["coordinates"]), "features": Path(f["features"][0]["features"]), "vocabulary": Path(f["features"][0]["vocabulary"])}
        except Exception as exc:  # offline / missing package
            self.status["error"] = f"{type(exc).__name__}: {exc}"
            return None

    def build(self) -> bool:
        import scipy.sparse as sp

        cache = self.cache_dir / "neurosynth_term_zmaps_4mm.npz"
        files = self._files()
        if files is None:
            return False
        self.vocab = files["vocabulary"].read_text().splitlines()
        feats = sp.load_npz(files["features"]).tocsr()
        presence = (feats > 1e-3).astype(np.float32)
        df_counts = np.asarray(presence.sum(axis=0)).ravel()
        n_studies = presence.shape[0]
        self.idf = np.log((1 + n_studies) / (1 + df_counts))
        if cache.exists():
            self.z = np.load(cache)["z"].astype(np.float32)
        else:
            coords = pd.read_csv(files["coordinates"], sep="\t")
            meta_ids = pd.read_csv(files["coordinates"].parent / "data-neurosynth_version-7_metadata.tsv.gz", sep="\t")["id"].to_numpy()
            study_pos = {int(s): i for i, s in enumerate(meta_ids)}
            mask_img = self.masker.mask_img_
            inv = np.linalg.inv(mask_img.affine)
            xyz = coords[["x", "y", "z"]].to_numpy(dtype=np.float64)
            ijk = np.rint(inv[:3, :3] @ xyz.T + inv[:3, 3:4]).T.astype(np.int64)
            sid = coords["id"].map(study_pos).to_numpy()
            ok = np.isfinite(sid.astype(float))
            ijk, sid = ijk[ok], sid[ok].astype(int)
            vox = 4.0
            r = 10.0 / vox
            rng_off = np.arange(-int(math.ceil(r)), int(math.ceil(r)) + 1)
            offs = np.array([(a, b, c) for a in rng_off for b in rng_off for c in rng_off if a * a + b * b + c * c <= r * r], dtype=np.int64)
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
            ma = sp.csr_matrix((np.ones(len(pairs), dtype=np.float32), (pairs[:, 0], pairs[:, 1])), shape=(n_studies, N_VOXELS))
            n_t = df_counts.astype(np.float64)
            n_v = np.asarray(ma.sum(axis=0)).ravel().astype(np.float64)
            n = float(n_studies)
            pres_t = torch.as_tensor(presence.T.toarray(), dtype=torch.float32, device=self.device)  # [T, S]
            z = np.zeros((len(self.vocab), N_VOXELS), dtype=np.float32)
            chunk = 4096
            for s in tqdm(range(0, N_VOXELS, chunk), desc="neurosynth association maps", leave=False):
                m_chunk = torch.as_tensor(ma[:, s : s + chunk].toarray(), dtype=torch.float32, device=self.device)
                a = (pres_t @ m_chunk).double().cpu().numpy()  # term & active
                nt = n_t[:, None]
                nv = n_v[None, s : s + chunk]
                b = nt - a
                c = nv - a
                d = n - nt - nv + a
                with np.errstate(invalid="ignore", divide="ignore"):
                    chi2 = n * (a * d - b * c) ** 2 / ((a + b) * (c + d) * (a + c) * (b + d))
                    sign = np.sign(a / np.clip(nt, 1, None) - c / np.clip(n - nt, 1, None))
                zc = np.nan_to_num(sign * np.sqrt(np.clip(chi2, 0, None)), nan=0.0)
                z[:, s : s + chunk] = np.clip(zc, -10, 10).astype(np.float32)
                del m_chunk
            del pres_t
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            np.savez_compressed(cache, z=z.astype(np.float16))
            self.z = z
        self.term_index = {t: i for i, t in enumerate(self.vocab) if not _NUMERIC.match(t)}
        self.max_ngram = max(len(t.split()) for t in self.term_index)
        self.status = {"available": True, "n_terms": len(self.term_index), "n_studies": int(n_studies), "zmap_cache": str(cache)}
        return True

    def term_weights(self, text: str) -> dict[int, float]:
        tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", text.lower())
        counts: dict[int, float] = {}
        for n in range(1, self.max_ngram + 1):
            for i in range(len(tokens) - n + 1):
                term = " ".join(tokens[i : i + n])
                j = self.term_index.get(term)
                if j is not None:
                    counts[j] = counts.get(j, 0.0) + 1.0
        assert self.idf is not None
        return {j: c * float(self.idf[j]) for j, c in counts.items()}

    def predict(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        assert self.z is not None
        out = np.zeros((len(texts), N_VOXELS), dtype=np.float32)
        n_terms = np.zeros(len(texts), dtype=np.int64)
        for i, text in enumerate(tqdm(texts, desc="neurosynth queries", leave=False)):
            w = self.term_weights(text)
            n_terms[i] = len(w)
            if not w:
                continue
            idx = np.fromiter(w.keys(), dtype=np.int64)
            wt = np.fromiter(w.values(), dtype=np.float32)
            out[i] = (wt[:, None] * self.z[idx]).sum(axis=0) / wt.sum()
        return out, n_terms


class NeuroQueryTextToMap:
    def __init__(self, masker: Any) -> None:
        self.masker = masker
        self.model = None
        self.status: dict[str, Any] = {"available": False}

    def build(self) -> bool:
        try:
            from neuroquery import NeuroQueryModel
            from neuroquery.datasets import fetch_neuroquery_model

            path = fetch_neuroquery_model()
            self.model = NeuroQueryModel.from_data_dir(path)
            self.status = {"available": True, "model_dir": str(path)}
            return True
        except Exception as exc:
            self.status = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
            return False

    def predict(self, texts: list[str]) -> np.ndarray:
        from nilearn.image import resample_to_img

        assert self.model is not None
        out = np.zeros((len(texts), N_VOXELS), dtype=np.float32)
        mask_img = self.masker.mask_img_
        for i, text in enumerate(tqdm(texts, desc="neuroquery queries", leave=False)):
            img = self.model(text)["brain_map"]
            if img.shape != mask_img.shape or not np.allclose(img.affine, mask_img.affine):
                img = resample_to_img(img, mask_img, interpolation="continuous", force_resample=True, copy_header=True)
            out[i] = self.masker.transform(img).ravel().astype(np.float32)
        return out
