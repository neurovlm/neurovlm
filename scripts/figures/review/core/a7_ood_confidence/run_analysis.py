#!/usr/bin/env python
"""A7 (comment 1.4): are out-of-distribution (OOD) maps and texts far from the training set in NeuroVLM's embedding spaces?

Final run models, CPU, ~2-4 min.  Method and outputs: README.md.
    CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/figures/review/core/a7_ood_confidence/run_analysis.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _review_common import (  # noqa: E402
    encode_specter, env_packages, get_logger, heldout_pmids, load_pubmed_maps, load_pubmed_text_rows, load_specter_latents,
    seed_all, set_offline_env, write_json,
)

set_offline_env()
log = get_logger("a7_ood")
REPO = HERE.parents[4]
MAP_SETS = ("rot90", "rotation", "spin", "shuffle", "noise", "x100", "x0.01")
TEXT_SETS = ("wikitext", "recipes")
LABEL = {"id_test": "ID: test maps", "rot90": "rotated 90°", "rotation": "rotated (random)", "spin": "surface spin",
         "shuffle": "voxel shuffle", "noise": "white noise", "x100": "map × 100", "x0.01": "map × 0.01",
         "id_test_text": "ID: test abstracts", "wikitext": "WikiText", "recipes": "recipes"}
SCORES = ("input_knn", "recon", "encoder_knn", "encoder_maha", "contrastive_knn", "contrastive_maha",  # pipeline order
          "latent_norm_score", "recon_per_nz", "recon_per_energy", "recon_support")  # AE latent norm; sparsity-normalised errors
SCORE_NAME = {"input_knn": "input kNN", "recon": "AE recon. error", "encoder_knn": "encoder kNN", "encoder_maha": "encoder Mahal.",
              "contrastive_knn": "contrastive kNN", "contrastive_maha": "contrastive Mahal.", "latent_norm_score": "AE latent norm",
              "recon_per_nz": "recon. SSE / non-zero voxels", "recon_per_energy": "recon. SSE / map energy",
              "recon_support": "recon. error on non-zero voxels"}
K = 10
NZ_REL = 1e-3  # "non-zero" = above 0.1 % of the map maximum
RECIPES = [
    "Preheat the oven to 180 C. Cream the butter and sugar until pale, beat in the eggs one at a time, then fold in the flour and bake for 25 minutes.",
    "Bring a large pot of salted water to the boil, cook the spaghetti until al dente, and toss it with garlic, olive oil, chilli flakes and parsley.",
    "Marinate the chicken thighs in yoghurt, lemon juice, cumin and paprika overnight, then grill over high heat until charred and cooked through.",
    "Soak the dried chickpeas overnight, simmer until tender, and blend with tahini, lemon, garlic and a splash of the cooking water to make hummus.",
    "Whisk eggs with a pinch of salt, pour into a hot buttered pan, stir gently over low heat and remove while the curds are still soft.",
    "Sweat the onions, carrots and celery in olive oil, add the lentils and stock, and simmer for forty minutes before seasoning with thyme.",
    "Combine flour, water, salt and yeast, knead for ten minutes, let the dough rise until doubled, shape into a loaf and bake on a hot stone.",
    "Slice the cucumbers thinly, salt them for twenty minutes, rinse, and dress with rice vinegar, sugar, sesame oil and toasted sesame seeds.",
    "Brown the beef in batches, deglaze the pan with red wine, add tomatoes and herbs, and braise covered in a low oven for three hours.",
    "Melt the chocolate with butter over a water bath, whisk in sugar and eggs, fold in a little flour and bake the brownies until just set.",
    "Toast the arborio rice in butter, add white wine, then ladle in hot stock gradually while stirring until the risotto is creamy.",
    "Blanch the green beans for two minutes, plunge them into ice water, and sauté with shallots and toasted almonds just before serving.",
    "Mix mashed ripe bananas with melted butter, sugar, an egg and baking soda, fold in flour and walnuts, and bake in a loaf tin for an hour.",
    "Fry the curry paste in coconut cream until fragrant, add sliced chicken and vegetables, pour in coconut milk and simmer for fifteen minutes.",
    "Roll out the pastry, line a tart tin, blind bake it, then fill with lemon curd and chill until firm before slicing.",
    "Grate the potatoes, squeeze out the liquid, mix with egg and onion, and fry spoonfuls in hot oil until golden on both sides.",
    "Simmer the berries with sugar and lemon juice until thick, test on a cold plate, and pour the jam into sterilised jars.",
    "Coat the tofu cubes in cornflour, fry until crisp, and toss in a sauce of soy, honey, ginger and garlic with spring onions.",
    "Layer the cooked lasagne sheets with meat sauce, béchamel and parmesan, finishing with cheese, and bake until bubbling and browned.",
    "Shake the gin, lime juice and sugar syrup with ice, strain into a chilled glass and garnish with a twist of lime peel.",
]


def load_run_module(run: Path, name: str):
    import torch

    return torch.load(run / f"{name}.pt", map_location="cpu", weights_only=False).float().eval()


def nonzero(X: np.ndarray) -> np.ndarray:
    """Number of voxels above 0.1 % of the map's own maximum (scale-free, so x100 / x0.01 keep their voxel count)."""
    X = np.atleast_2d(X)
    return (X > NZ_REL * X.max(axis=1, keepdims=True)).sum(axis=1)


class Brain:
    def __init__(self):
        from _hf_pins import load_masker

        img = load_masker().mask_img_
        self.affine = img.affine
        self.mask = np.asarray(img.get_fdata()) > 0
        self.ijk = np.argwhere(self.mask)  # C order == NiftiMasker order
        self.xyz = (self.affine @ np.c_[self.ijk, np.ones(len(self.ijk))].T).T[:, :3]
        self.c = self.ijk.mean(axis=0)
        ev, U = np.linalg.eigh(np.cov(self.ijk.T))
        self.A = U @ np.diag(np.sqrt(ev)) @ U.T  # colouring; whitening = inv(A)
        self.Ai = np.linalg.inv(self.A)

    def to_vol(self, flat):
        vol = np.zeros(self.mask.shape, dtype=np.float32)
        vol[self.mask] = flat
        return vol

    def rotate(self, flat, R, *, normalised=True, order=0):
        """Output voxel o samples the input at M (o - c) + c (M = A R^T A^-1 in brain-normalised coordinates)."""
        from scipy.ndimage import affine_transform

        M = self.A @ R.T @ self.Ai if normalised else R.T
        vol = affine_transform(self.to_vol(flat), M, offset=self.c - M @ self.c, order=order, mode="constant", cval=0.0)
        return np.clip(vol[self.mask], 0, None).astype(np.float32)


class Spinner:
    """Nearest-vertex voxel <-> fsaverage-10k mapping + Alexander-Bloch sphere rotation (every mask voxel is spun)."""

    def __init__(self, brain: Brain, max_dist_mm: float = np.inf):
        import nibabel as nib
        from neuromaps.datasets import fetch_atlas
        from scipy.spatial import cKDTree

        atlas = fetch_atlas("fsaverage", "10k", verbose=0)
        self.spheres, mids = [], []
        for h in (0, 1):
            mid = (np.asarray(nib.load(str(atlas["white"][h])).agg_data()[0]) + np.asarray(nib.load(str(atlas["pial"][h])).agg_data()[0])) / 2
            sph = np.asarray(nib.load(str(atlas["sphere"][h])).agg_data()[0], dtype=np.float64)
            self.spheres.append(sph / np.linalg.norm(sph, axis=1, keepdims=True))
            mids.append(mid)
        self.trees = [cKDTree(s) for s in self.spheres]
        allmid = np.concatenate(mids)
        dist, self.vox_vert = cKDTree(allmid).query(brain.xyz)
        self.keep = dist <= max_dist_mm  # previous construction: 6 mm, voxels beyond it were zeroed
        _, self.vert_vox = cKDTree(brain.xyz).query(allmid)
        self.nv = len(allmid)
        self.count = np.bincount(self.vox_vert[self.keep], minlength=self.nv)

    def spin(self, flat, R, *, mean_of_voxels=True):
        """Spin ``flat`` by the 3x3 rotation ``R`` of the left sphere (the right sphere uses its mirror image)."""
        if mean_of_voxels:
            vals = np.bincount(self.vox_vert[self.keep], weights=flat[self.keep], minlength=self.nv) / np.maximum(self.count, 1)
            vals[self.count == 0] = flat[self.vert_vox[self.count == 0]]
        else:  # previous construction: vertex value = value of its nearest voxel
            vals = flat[self.vert_vox]
        Fx = np.diag([-1.0, 1.0, 1.0])
        spun, off = np.empty_like(vals), 0
        for h, rot in enumerate((R, Fx @ R @ Fx)):
            n = len(self.spheres[h])
            _, perm = self.trees[h].query(self.spheres[h] @ rot.T)
            spun[off:off + n] = vals[off:off + n][perm]
            off += n
        out = np.zeros_like(flat)
        out[self.keep] = spun[self.vox_vert[self.keep]]
        return out.astype(np.float32)


def make_ood_maps(X: np.ndarray, brain: Brain, seed: int) -> tuple[dict, dict]:
    """The OOD map sets built from the source maps ``X`` (sources without a valid rotation / spin are dropped)."""
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(seed)
    spinner = Spinner(brain)
    axes = [Rotation.from_euler(a, s * 90, degrees=True).as_matrix() for a in "xyz" for s in (1, -1)]
    out = {k: [] for k in MAP_SETS}
    src = {k: [] for k in MAP_SETS}  # index of the source map
    draws = {"rotation": [], "spin": []}

    def add(name, y, i):
        out[name].append(y)
        src[name].append(i)

    def accept(y, x):  # keeps >= 90 % of the source's non-zero voxels
        return nonzero(y)[0] >= 0.9 * nonzero(x)[0]

    for i, x in enumerate(X):
        for k in rng.permutation(len(axes)):  # 90 deg: six axis/sign options in random order
            y = brain.rotate(x, axes[k])
            if accept(y, x):
                add("rot90", y, i)
                break
        for name, fn in (("rotation", lambda R: brain.rotate(x, R)), ("spin", lambda R: spinner.spin(x, R))):
            for d in range(100):
                y = fn(Rotation.random(random_state=rng).as_matrix())
                if accept(y, x):
                    add(name, y, i)
                    draws[name].append(d + 1)
                    break
        add("shuffle", rng.permutation(x), i)
        g = np.abs(rng.standard_normal(len(x))).astype(np.float32)
        add("noise", g / g.max(), i)
        add("x100", x * 100, i)
        add("x0.01", x * 0.01, i)
    notes = {k: {"n_sources": int(len(X)), "n_kept": len(src[k]), "n_dropped_no_valid_draw": int(len(X) - len(src[k]))} for k in MAP_SETS}
    for k, v in draws.items():
        notes[k]["median_draws"] = float(np.median(v)) if v else None
    return {k: np.stack(v).astype(np.float32) for k, v in out.items()}, {"src": {k: np.asarray(v) for k, v in src.items()}, "notes": notes}


def map_stats(Y: np.ndarray, X_src: np.ndarray) -> dict:
    n_src = nonzero(X_src)
    n = nonzero(Y)
    return {"n": int(len(Y)), "nonzero_frac_median": float(np.median(n / Y.shape[1])),
            "nonzero_vs_source_median": float(np.median(n / np.maximum(n_src, 1))), "nonzero_vs_source_min": float(np.min(n / np.maximum(n_src, 1))),
            "nonzero_voxels_median": int(np.median(n)), "n_all_zero": int((Y.max(axis=1) <= 0).sum()), "mean": float(Y.mean()),
            "std": float(Y.std()), "max_median": float(np.median(Y.max(axis=1)))}


def load_wikitext(n: int, seed: int) -> list[str]:
    """WikiText-2 raw test paragraphs: the parquet behind ``datasets.load_dataset("wikitext", "wikitext-2-raw-v1",
    split="test")``, at its pinned revision."""
    import pandas as pd
    from _hf_pins import hf_file

    text = pd.read_parquet(hf_file("wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet"))["text"]
    paras = [t.strip() for t in text if len(t.strip()) >= 200 and not t.strip().startswith("=")]
    pick = np.random.default_rng(seed).choice(len(paras), size=min(n, len(paras)), replace=False)
    return [" ".join(paras[i].replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".").split()[:120]) for i in sorted(pick)]


def l2(x):
    import torch.nn.functional as F

    return F.normalize(x.float(), dim=1)


def knn_dist(Q, R, k: int = K, *, ref_normalised: bool = False, chunk: int = 2048) -> np.ndarray:
    import torch

    Q = l2(torch.as_tensor(Q))
    R = R if ref_normalised else l2(R)
    out = []
    for s in range(0, len(Q), chunk):
        out.append(1 - torch.topk(Q[s:s + chunk] @ R.T, k, dim=1).values.mean(dim=1))
    return torch.cat(out).numpy()


class Mahalanobis:
    def __init__(self, R):
        from sklearn.covariance import LedoitWolf

        R = R.double().numpy()
        self.mu = R.mean(axis=0)
        lw = LedoitWolf().fit(R - self.mu)
        self.P, self.shrinkage = lw.precision_, float(lw.shrinkage_)

    def __call__(self, Q) -> np.ndarray:
        D = Q.double().numpy() - self.mu
        return np.sqrt(np.einsum("ij,jk,ik->i", D, self.P, D))


def pearson(A, b) -> np.ndarray:
    """Row-wise Pearson correlation of A [n, d] with b [d] or [n, d]."""
    A, b = A - A.mean(dim=-1, keepdim=True), b - b.mean(dim=-1, keepdim=True)
    return ((A * b).sum(dim=-1) / (A.norm(dim=-1) * b.norm(dim=-1))).numpy()


def auroc(neg: np.ndarray, pos: np.ndarray) -> float:
    """Mann-Whitney AUROC (ties count one half)."""
    from scipy.stats import rankdata

    r = rankdata(np.r_[neg, pos])[len(neg):]
    return float((r.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def detect(neg: np.ndarray, pos: np.ndarray) -> float:
    """Share of OOD scores above the 95th percentile of the ID scores (5 % false alarms by construction)."""
    return float((pos > np.quantile(neg, 0.95)).mean())


def boot_ci(neg: np.ndarray, pos: np.ndarray, n_boot: int, rng) -> dict:
    """Detection rate at the ID 95th percentile and AUROC with 95 % bootstrap CIs (ID and OOD both resampled: the threshold varies too)."""
    boots = []
    for _ in range(n_boot):
        n, p = rng.choice(neg, len(neg)), rng.choice(pos, len(pos))
        boots.append((detect(n, p), auroc(n, p)))
    (d_lo, a_lo), (d_hi, a_hi) = np.quantile(boots, [0.025, 0.975], axis=0)
    return {"detection_at_id95": detect(neg, pos), "det_lo": d_lo, "det_hi": d_hi, "auroc": auroc(neg, pos), "ci_lo": a_lo, "ci_hi": a_hi}


def strata(neg, pos, nz_neg, nz_pos, edges) -> tuple[float, list[dict]]:
    """Sparsity control: an ID 95th-percentile threshold per non-zero-voxel stratum (``edges`` = ID quintiles).

    Returns the pooled detection rate (each OOD map against the threshold of its own stratum) and one row per stratum."""
    bn, bp = np.digitize(nz_neg, edges), np.digitize(nz_pos, edges)
    bounds, thr = np.r_[0, edges, np.inf], np.quantile(neg, 0.95)
    rows = []
    for b in range(len(edges) + 1):
        t, p = np.quantile(neg[bn == b], 0.95), pos[bp == b]
        rows.append({"stratum": b + 1, "nonzero_from": bounds[b], "nonzero_to": bounds[b + 1], "n_id": int((bn == b).sum()), "n_ood": len(p),
                     "id_false_alarm_global_threshold": float((neg[bn == b] > thr).mean()), "id_threshold": float(t),
                     "detection_global_threshold": float((p > thr).mean()) if len(p) else np.nan,
                     "detection_stratum_threshold": float((p > t).mean()) if len(p) else np.nan})
    thr_b = np.array([r["id_threshold"] for r in rows])
    return float((pos > thr_b[bp]).mean()), rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, default=REPO / "scripts" / "outputs" / "runs" / "final", help="Run directory (default: runs/final).")
    p.add_argument("--out", type=Path, default=HERE / "outputs", help="Output directory.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n-ood", type=int, default=500, help="Source test maps / WikiText paragraphs per OOD set.")
    p.add_argument("--n-boot", type=int, default=2000, help="Bootstrap resamples for the AUROC CI.")
    p.add_argument("--no-exclusions", action="store_true", help="Keep the leakage-audit test papers.")
    args = p.parse_args(argv)
    seed_all(args.seed)
    import pandas as pd
    import torch

    torch.set_num_threads(min(16, torch.get_num_threads()))
    t0 = time.perf_counter()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    run = args.model_dir.resolve()

    from _hf_pins import load_model

    # models + training reference
    ae, ih, th = (load_run_module(run, n) for n in ("autoencoder", "proj_head_image_infonce", "proj_head_text_infonce"))
    specter = load_model("specter").to("cpu")
    split = torch.load(run / "pmids_split.pt", weights_only=False)
    split = {k: np.asarray(split[k]).astype(np.int64) for k in ("train", "val", "test")}
    lat = torch.load(run / "latent_neuro.pt", map_location="cpu", weights_only=False)
    lat_pm = np.asarray(lat["pmid"]).astype(np.int64)
    is_tr = np.isin(lat_pm, split["train"])
    pm_tr = lat_pm[is_tr]
    with torch.no_grad():
        L_tr = lat["latent"][is_tr].float()
        Z_tr = l2(ih(L_tr))
        E_tr = l2(load_specter_latents(pm_tr))
        Zt_tr = l2(th(E_tr))
    log.info("training reference: %d maps, %d texts", len(L_tr), len(E_tr))

    # ID test maps (exclusions applied) + OOD sources
    test, excl = heldout_pmids("test", exclusions=not args.no_exclusions, log=log.info)
    test = test[np.isin(test, lat_pm)]
    X_id = load_pubmed_maps(test).numpy()
    src_pm = np.sort(rng.choice(test, size=min(args.n_ood, len(test)), replace=False))
    X_src = X_id[np.searchsorted(test, src_pm)]
    brain = Brain()
    t_build = time.perf_counter()
    ood, meta = make_ood_maps(X_src, brain, args.seed)
    log.info("OOD maps built in %.0f s", time.perf_counter() - t_build)

    def embed_maps(X):
        """Autoencoder latent, contrastive embedding and reconstruction sigmoid(decoder(latent))."""
        with torch.no_grad():
            L = ae.encoder(torch.from_numpy(np.ascontiguousarray(X)))
            return L, l2(ih(L)), torch.sigmoid(ae.decoder(L))

    # consistency: the cached training latents are the final encoder's latents
    chk = np.flatnonzero(is_tr)[:: max(1, is_tr.sum() // 64)][:64]
    with torch.no_grad():
        L_chk = ae.encoder(load_pubmed_maps(lat_pm[chk]))
    lat_check = float((L_chk - lat["latent"][chk].float()).abs().max())

    # sanity: held-out maps should be as close to the training set as training maps are to each other
    loo = np.random.default_rng(0).choice(len(Z_tr), 3000, replace=False)
    S = Z_tr[loo] @ Z_tr.T
    S[np.arange(3000), loo] = -2.0
    train_loo_knn = float(np.median(1 - torch.topk(S, K, dim=1).values.mean(dim=1).numpy()))
    del S
    maha = {"contrastive": Mahalanobis(Z_tr), "encoder": Mahalanobis(L_tr)}
    X_tr = load_pubmed_maps(pm_tr)  # input-space reference (23,894 x 28,542, ~2.7 GB)
    train_max = X_tr.max(dim=1).values
    train_value_range = {"min_of_map_maxima": float(train_max.min()), "max_of_map_maxima": float(train_max.max()), "min_value": float(X_tr.min())}
    mean_map = X_tr.mean(dim=0)
    active = (X_tr > NZ_REL * train_max[:, None]).sum(dim=0) / len(X_tr)  # share of training maps in which a voxel is non-zero
    rare = active < torch.quantile(active, 0.25)  # the quarter of voxels least often active in training
    X_tr = l2(X_tr)

    # mechanism: where the encoder puts a map relative to the training latents, and how well the decoder redraws it
    mu = L_tr.mean(dim=0)
    ev, U = torch.linalg.eigh(torch.cov((L_tr - mu).T.double()))  # ascending
    U_detail = U[:, :-10].float()  # all but the 10 leading principal axes of the training latents
    norm_tr = float(L_tr.norm(dim=1).median())

    def map_columns(X, L, R, X_src_rows=None) -> dict:
        cos = torch.nn.functional.cosine_similarity
        Xt = torch.from_numpy(np.ascontiguousarray(X))
        supp = Xt > NZ_REL * Xt.max(dim=1, keepdim=True).values
        nz, sse, energy = supp.sum(dim=1), ((R - Xt) ** 2).sum(dim=1), (Xt ** 2).sum(dim=1)
        norm = L.norm(dim=1)
        return {"recon": (1 - cos(R, Xt, dim=1)).numpy(), "cos_recon": cos(R, Xt, dim=1).numpy(), "recon_per_nz": (sse / nz).numpy(), "recon_per_energy": (sse / energy).numpy(),
                "recon_support": (1 - cos(R * supp, Xt * supp, dim=1)).numpy(), "nonzero": nz.numpy(),
                "latent_norm": norm.numpy(), "latent_norm_score": np.abs(np.log(norm.numpy() / norm_tr)),
                "latent_dist_mean": (L - mu).norm(dim=1).numpy(), "latent_cos_mean": cos(L, mu[None], dim=1).numpy(),
                "latent_detail": ((L - mu) @ U_detail).norm(dim=1).numpy(), "rare_energy_input": ((Xt ** 2 * rare).sum(dim=1) / energy).numpy(),
                "rec_corr_input": pearson(R, Xt), "rec_corr_mean_map": pearson(R, mean_map), "rec_max": R.max(dim=1).values.numpy(),
                "rec_corr_source": np.nan if X_src_rows is None else pearson(R, torch.from_numpy(X_src_rows))}
    maha_t = {"contrastive": Mahalanobis(Zt_tr), "encoder": Mahalanobis(E_tr)}
    rows, emb = [], {}
    for name, X in {"id_test": X_id, **ood}.items():
        L, Z, R = embed_maps(X)
        emb[name] = (L, Z)
        rows.append(pd.DataFrame({"modality": "map", "set": name, "ood": name != "id_test",
                                  "input_knn": knn_dist(X, X_tr, ref_normalised=True, chunk=256), "contrastive_knn": knn_dist(Z, Z_tr),
                                  "contrastive_maha": maha["contrastive"](Z), "encoder_knn": knn_dist(L, L_tr), "encoder_maha": maha["encoder"](L),
                                  **map_columns(X, L, R, None if name == "id_test" else X_src[meta["src"][name]]),
                                  "nonzero_vs_source": np.nan if name == "id_test" else nonzero(X) / np.maximum(nonzero(X_src[meta["src"][name]]), 1),
                                  "item": [str(p) for p in test] if name == "id_test" else [str(src_pm[i]) for i in meta["src"][name]]}))

    del X_tr

    # text
    E_id = l2(load_specter_latents(test))
    texts = {"wikitext": load_wikitext(args.n_ood, args.seed), "recipes": RECIPES}
    rows_id = load_pubmed_text_rows(test[:16])
    live = l2(encode_specter(specter, [f"{a} [SEP] {b}" for a, b in zip(rows_id["name"], rows_id["description"])]))
    specter_check = float((live * E_id[:16]).sum(dim=1).min())  # precomputed == live "title [SEP] abstract"
    for name, E in {"id_test_text": E_id, **{k: l2(encode_specter(specter, v)) for k, v in texts.items()}}.items():
        with torch.no_grad():
            Z = l2(th(E))
        emb[name] = (E, Z)
        rows.append(pd.DataFrame({"modality": "text", "set": name, "ood": name != "id_test_text", "contrastive_knn": knn_dist(Z, Zt_tr),
                                  "contrastive_maha": maha_t["contrastive"](Z), "encoder_knn": knn_dist(E, E_tr), "encoder_maha": maha_t["encoder"](E),
                                  "max_cos_specter2": 1 - knn_dist(E, E_tr, k=1), "max_cos_shared": 1 - knn_dist(Z, Zt_tr, k=1),
                                  "item": [str(p) for p in test] if name == "id_test_text" else texts[name]}))
    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(out / "scores.parquet", index=False)

    # primary: detection rate at the ID 95th percentile (5 % false alarms); secondary: AUROC. Maps also get a sparsity-matched
    # detection rate (ID threshold per quintile of the ID non-zero-voxel count) and the per-stratum rates (sparsity_strata.csv)
    id_nz = df.loc[df["set"] == "id_test", "nonzero"].to_numpy()
    edges = np.quantile(id_nz, [0.2, 0.4, 0.6, 0.8])
    arows, srows = [], []
    for modality, id_name, sets in (("map", "id_test", MAP_SETS), ("text", "id_test_text", TEXT_SETS)):
        for score in SCORES:
            neg = df.loc[df["set"] == id_name, score].to_numpy()
            if np.isnan(neg).all():
                continue
            for s in sets:
                pos = df.loc[df["set"] == s, score].to_numpy()
                r = {"modality": modality, "set": s, "score": score, "n_id": len(neg), "n_ood": len(pos), **boot_ci(neg, pos, args.n_boot, rng),
                     "detection_sparsity_matched": np.nan, "id95_threshold": float(np.quantile(neg, 0.95)),
                     "id_median": float(np.median(neg)), "ood_median": float(np.median(pos))}
                if modality == "map":
                    r["detection_sparsity_matched"], st = strata(neg, pos, id_nz, df.loc[df["set"] == s, "nonzero"].to_numpy(), edges)
                    srows += [{"set": s, "score": score, **x} for x in st]
                arows.append(r)
    det = pd.DataFrame(arows)
    det.to_csv(out / "detection.csv", index=False, float_format="%.4f")

    # cosine scores (higher = more in-distribution): maps cos(map, its autoencoder reconstruction) = 1 - recon; text max cosine to the
    # training abstracts, in the SPECTER2 space (figure) and in the shared space. Flagged = below the held-out 5th percentile (5 % false alarms)
    frows = []
    for modality, id_name, sets, score in (("map", "id_test", MAP_SETS, "cos_recon"), ("text", "id_test_text", TEXT_SETS, "max_cos_specter2"),
                                           ("text", "id_test_text", TEXT_SETS, "max_cos_shared")):
        neg = df.loc[df["set"] == id_name, score].to_numpy()
        for s in (id_name, *sets):
            pos = df.loc[df["set"] == s, score].to_numpy()
            b = boot_ci(-neg, -pos, args.n_boot, rng) if s != id_name else {"detection_at_id95": float((pos < np.quantile(neg, 0.05)).mean())}
            frows.append({"modality": modality, "set": s, "score": score, "n": len(pos), "threshold_id5": float(np.quantile(neg, 0.05)),
                          "median": float(np.median(pos)), "flagged": b["detection_at_id95"], "flagged_lo": b.get("det_lo", np.nan),
                          "flagged_hi": b.get("det_hi", np.nan), "auroc": b.get("auroc", np.nan)})
    pd.DataFrame(frows).to_csv(out / "flagged_cosine.csv", index=False, float_format="%.4f")
    pd.DataFrame(srows).to_csv(out / "sparsity_strata.csv", index=False, float_format="%.4f")

    # mechanism: per-set medians (maps), the training-latent reference, and each rotated map against its own unrotated source
    maps = df[df["modality"] == "map"]
    mech_cols = ["nonzero", "latent_norm", "latent_dist_mean", "latent_cos_mean", "latent_detail", "encoder_maha", "contrastive_knn", "recon",
                 "rec_corr_input", "rec_corr_source", "rec_corr_mean_map", "rec_max", "rare_energy_input"]
    maps.groupby("set", sort=False)[mech_cols].median().to_csv(out / "mechanism.csv", float_format="%.4f")
    id_rec = maps[maps["set"] == "id_test"].set_index("item")["recon"]
    with torch.no_grad():
        dec_mu = torch.sigmoid(ae.decoder(mu[None]))
    from scipy.stats import spearmanr

    mechanism = {"train_latent_norm_median": norm_tr, "train_mean_latent_norm": float(mu.norm()),
                 "train_latent_dist_to_mean_median": float((L_tr - mu).norm(dim=1).median()),
                 "train_latent_cos_to_mean_median": float(torch.nn.functional.cosine_similarity(L_tr, mu[None], dim=1).median()),
                 "train_latent_detail_median": float(((L_tr - mu) @ U_detail).norm(dim=1).median()),
                 "train_latent_variance_share_top1_top10": [float(ev[-1] / ev.sum()), float(ev[-10:].sum() / ev.sum())],
                 "corr_decoded_mean_latent_vs_mean_training_map": float(pearson(dec_mu, mean_map)[0]),
                 "share_recon_above_own_source": {s: float((maps.loc[maps["set"] == s, "recon"].to_numpy()
                                                            > id_rec.loc[maps.loc[maps["set"] == s, "item"]].to_numpy()).mean())
                                                  for s in ("rot90", "rotation", "spin", "shuffle")}}
    sparsity = {"id_nonzero_quintile_edges": edges, "id_nonzero_range": [int(id_nz.min()), int(id_nz.max())],
                "nonzero_vs_source_range": {s: [float(v.min()), float(np.median(v)), float(v.max())] for s in MAP_SETS
                                            for v in [maps.loc[maps["set"] == s, "nonzero_vs_source"].to_numpy()]},
                "spearman_vs_nonzero_id": {c: float(spearmanr(maps.loc[maps["set"] == "id_test", c], id_nz)[0]) for c in ("recon", "latent_norm")}}

    # inspection: collapse, distance to the source embedding, nearest training items of collapsed sets
    from _hf_pins import load_dataset

    nfoci = load_dataset("pubmed_coordinates").groupby("pmid").size()
    nfoci.index = pd.to_numeric(nfoci.index, errors="coerce").astype("Int64")
    Z_src = emb["id_test"][1][np.searchsorted(test, src_pm)]
    insp, nn_rows = [], []
    with torch.no_grad():
        for name, (L, Z) in emb.items():
            n = len(Z)
            pair = float(((Z @ Z.T).sum() - n) / (n * (n - 1)))
            r = {"set": name, "n": n, "mean_pairwise_cos_contrastive": pair}
            if name in MAP_SETS:
                r["cos_to_own_source_contrastive"] = float((Z * Z_src[meta["src"][name]]).sum(dim=1).mean())
                r.update(map_stats(ood[name], X_src[meta["src"][name]]))
            if name == "id_test":
                r.update(map_stats(X_id, X_id))
            if name in MAP_SETS or name == "id_test":
                r["latent_norm_median"] = float(L.norm(dim=1).median())
            insp.append(r)
            if name in MAP_SETS + TEXT_SETS and pair > 0.8:  # collapsed: report the point and its nearest training items
                ref = Z_tr if name in MAP_SETS else Zt_tr
                cen = l2(Z.mean(dim=0, keepdim=True))
                top = torch.topk((cen @ ref.T)[0], 5)
                pms = pm_tr[top.indices.numpy()]
                titles = load_pubmed_text_rows(pms, min_coverage=0)["name"].tolist()
                for rank, (pm, c, t) in enumerate(zip(pms, top.values.numpy(), titles), 1):
                    nn_rows.append({"set": name, "rank": rank, "pmid": int(pm), "cos_to_set_centroid": float(c), "n_foci": int(nfoci.get(int(pm), -1)),
                                    "title": t})
        for special, x in (("all-zero map", np.zeros((1, X_id.shape[1]), np.float32)), ("all-one map", np.ones((1, X_id.shape[1]), np.float32))):
            L, Z, _ = embed_maps(x)
            insp.append({"set": special, "n": 1, "latent_norm_median": float(L.norm()),
                         **{f"cos_to_{k}_centroid": float((l2(emb[k][1].mean(0, keepdim=True)) @ Z.T)[0, 0]) for k in ("noise", "shuffle", "x0.01")}})
    pd.DataFrame(insp).to_csv(out / "inspection.csv", index=False, float_format="%.4f")
    pd.DataFrame(nn_rows).to_csv(out / "nearest_training_items.csv", index=False, float_format="%.4f")
    train_nfoci = float(nfoci.reindex(pd.array(pm_tr, dtype="Int64")).median())

    # the previous OOD construction on the same sources: rigid rotation + linear interpolation; spin with a 6-mm cutoff
    from scipy.spatial.transform import Rotation

    prev_rng = np.random.default_rng(args.seed)
    sp6 = Spinner(brain, max_dist_mm=6.0)
    prev = {"rigid_rotation_linear": np.stack([brain.rotate(x, Rotation.random(random_state=prev_rng).as_matrix(), normalised=False, order=1) for x in X_src]),
            "spin_6mm_cutoff": np.stack([sp6.spin(x, Rotation.random(random_state=prev_rng).as_matrix(), mean_of_voxels=False) for x in X_src])}
    prev_diag = {k: map_stats(v, X_src) for k, v in prev.items()}
    neg = df.loc[df["set"] == "id_test", "contrastive_knn"].to_numpy()
    for k, v in prev.items():
        _, Z, _ = embed_maps(v)
        prev_diag[k]["contrastive_knn_auroc"] = auroc(neg, knn_dist(Z, Z_tr))
    prev_diag["spin_6mm_cutoff"]["voxels_beyond_6mm_zeroed"] = int((~sp6.keep).sum())

    # wide table (every score) + metrics
    wide = []
    for s in MAP_SETS + TEXT_SETS:
        d = det[det["set"] == s].set_index("score")
        row = {"OOD set": LABEL[s], "n": int(d["n_ood"].iloc[0])}
        for score in SCORES:
            row[SCORE_NAME[score]] = f"{100 * d.loc[score, 'detection_at_id95']:.0f}% ({d.loc[score, 'auroc']:.2f})" if score in d.index else "--"
        wide.append(row)
    wide = pd.DataFrame(wide)
    wide.to_csv(out / "a7_ood_table_full.csv", index=False)
    caption = ("OOD detection with the final NeuroVLM models: share of OOD inputs above the 95th percentile of the held-out test-input "
               f"score (maps n = {int((df['set'] == 'id_test').sum())}, abstracts n = {int((df['set'] == 'id_test_text').sum())}; 5\\% "
               "false alarms by construction) and, in parentheses, the AUROC. Reference = training split; kNN = 1 $-$ mean cosine to the "
               "10 nearest training items; Mahal. = Mahalanobis distance (Ledoit--Wolf). Input = raw voxels; encoder = projection-head input "
               "(autoencoder latent for maps, SPECTER2 for text); contrastive = shared InfoNCE space. AE latent norm = $|\\log(\\|z\\| / "
               "\\mathrm{median}_\\mathrm{train}\\|z\\|)|$; SSE = squared reconstruction error; non-zero voxels = above 0.1\\% of the map maximum.")
    tex = ["\\begin{table*}[t]", "\\centering", f"\\caption{{{caption}}}", "\\label{tab:a7_ood_full}", "\\resizebox{\\textwidth}{!}{%",
           "\\begin{tabular}{lr" + "l" * (wide.shape[1] - 2) + "}", "\\toprule", " & ".join(wide.columns) + " \\\\", "\\midrule"]
    for i, r in wide.iterrows():
        if i == len(MAP_SETS):
            tex.append("\\midrule")
        tex.append(" & ".join(str(v).replace("%", "\\%") for v in r.values) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}", ""]
    (out / "a7_ood_table_full.tex").write_text("\n".join(tex))
    np.savez_compressed(out / "ood_map_examples.npz", source=X_src[:5].astype(np.float16),
                        **{k: v[:5].astype(np.float16) for k, v in ood.items()}, **{f"previous_{k}": v[:5].astype(np.float16) for k, v in prev.items()})
    write_json(out / "metrics.json", {
        "models": {"run_dir": str(run.relative_to(REPO)), "specter": "released (frozen) SPECTER2 adhoc_query"},
        "reference": {"split": "train", "n_maps": int(len(L_tr)), "n_texts": int(len(E_tr))},
        "id_control": {"split": "test", "n": int(len(test)), "exclusions": excl},
        "consistency_checks": {"max_abs_diff_cached_train_latents_vs_final_encoder": lat_check,
                               "median_contrastive_knn_train_leave_one_out": train_loo_knn,
                               "median_contrastive_knn_id_test": float(df.loc[df["set"] == "id_test", "contrastive_knn"].median()),
                               "min_cos_precomputed_vs_live_specter_title_sep_abstract": specter_check,
                               "ledoit_wolf_shrinkage": {"map_contrastive": maha["contrastive"].shrinkage, "map_encoder": maha["encoder"].shrinkage,
                                                         "text_contrastive": maha_t["contrastive"].shrinkage, "text_encoder": maha_t["encoder"].shrinkage}},
        "ood_construction": meta["notes"], "mechanism": mechanism, "sparsity": sparsity, "train_median_n_foci": train_nfoci, "train_map_value_range": train_value_range,
        "diagnosis_previous_version": {
            "a_masking_previous_construction_same_sources": prev_diag,
            "b_score": "canon_dist_all = 1 - max cosine between the map/text embedding and the 904 canonical caption prototypes of the "
                       "Q-Former (a cross-modal similarity to a fixed vocabulary), not a density / OOD score of the embedding space",
            "c_reference": "ID controls and the 95th-percentile threshold pooled PubMed test with NeuroVault maps (dense statistical maps, "
                           "50% non-zero voxels vs 19% for PubMed maps); the training set was not the reference of the primary score",
            "d_space": "released autoencoder / heads (not the final run); otherwise the same encoder -> image head -> L2 path"},
        "n_boot": args.n_boot, "k": K, "seed": args.seed, "packages": env_packages(), "wall_seconds": time.perf_counter() - t0})
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(pd.DataFrame(insp).drop(columns=["mean", "std"], errors="ignore").to_string(float_format=lambda v: f"{v:.3f}"))
        print(det.pivot(index="set", columns="score", values="detection_at_id95").to_string(float_format=lambda v: f"{v:.3f}"))
        print(maps.groupby("set", sort=False)[mech_cols].median().to_string(float_format=lambda v: f"{v:.3f}"))
        print(pd.DataFrame(nn_rows).to_string())
        print({k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in prev_diag.items()})
    log.info("done in %.0f s", time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
