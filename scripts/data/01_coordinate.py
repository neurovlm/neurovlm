#!/usr/bin/env python
"""ALE-smooth the PubMed coordinates and project them through DiFuMo-512 (port of docs/02_data/01_coordinate.ipynb).

    .venv/bin/python scripts/data/01_coordinate.py [--smoke] [--force] [--ijk-mode trunc|round]

--ijk-mode sets how an MNI focus is mapped to a voxel of the 4 mm mask before the ALE kernel. trunc (default) truncates
toward zero, as nimare <= 0.15 did when the released neuro_vectors_ale.pt / neuro_vectors.pt were built (2025-12); with
it the ALE maps are bit-identical to the release. round rounds to the nearest voxel, as nimare >= 0.16 does: about two
thirds of the foci move by one voxel (ALE row r ~0.63 vs the release) and 6 more studies end up with an empty map.
The voxel indices are computed here, so the result does not depend on the installed nimare version.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))  # scripts/pipeline/
from common import artifacts_exist, base_parser, finalize_args, get_logger, select_device, setup_torch, timed  # noqa: E402
from data_utils import add_data_args, chunk_bounds, resolve_artifact_dir  # noqa: E402

STAGE = "data_01_coordinate"
FWHM = 9
DIFUMO_DIM = 512
QUANTILE = 0.9999
N_EXAMPLES = 20
log = get_logger(STAGE)


def coords_to_vectors(df_coords, *, fwhm: int, ijk_mode: str = "trunc"):
    """neurovlm.data.coordinates.coords_to_vectors with the voxel indices set here (nimare then skips its own mm2vox):
    ALE kernel maps of each pmid, divided by the kernel peak, clamped to 1; rows in ascending pmid order."""
    import nibabel as nib
    import torch
    from nimare.meta.kernel import ALEKernel
    from nimare.meta.utils import get_ale_kernel
    from neurovlm.data import load_masker

    masker = load_masker()
    df = df_coords[["pmid", "x", "y", "z"]].rename(columns={"pmid": "id"})
    with np.errstate(invalid="ignore"):
        vox = nib.affines.apply_affine(np.linalg.inv(masker.mask_img.affine), df[["x", "y", "z"]].to_numpy(dtype=float))
        df[["i", "j", "k"]] = (np.round(vox) if ijk_mode == "round" else vox).astype(int)  # astype(int) truncates toward zero
    _, kernel = get_ale_kernel(masker.mask_img, fwhm=fwhm)
    maps = torch.from_numpy(ALEKernel(fwhm=fwhm).transform(df, masker=masker, return_type="array")).float() / kernel.max()
    return torch.clamp_max(maps, 1)


def ale_smooth(df_coords, pmids_sorted: np.ndarray, *, fwhm: int, n_jobs: int, ijk_mode: str = "trunc"):
    """coords_to_vectors over pmid chunks in parallel; rows in ascending pmid order (maps are per study, so exact)."""
    import torch
    from joblib import Parallel, delayed

    n_workers = n_jobs if n_jobs > 0 else (os.cpu_count() or 1)
    # >= ~32 studies per chunk so the nimare/masker setup per chunk stays small
    n_chunks = max(1, min(n_workers * 4, (len(pmids_sorted) + 31) // 32))
    chunks = np.array_split(pmids_sorted, n_chunks)
    parts = Parallel(n_jobs=min(n_workers, n_chunks), backend="loky")(
        delayed(coords_to_vectors)(df_coords[df_coords["pmid"].isin(chunk)], fwhm=fwhm, ijk_mode=ijk_mode) for chunk in chunks
    )
    return torch.cat(parts, dim=0)


def fit_difumo(masker):
    from nilearn import datasets, image
    from nilearn.maskers import NiftiMapsMasker

    difumo = datasets.fetch_atlas_difumo(dimension=DIFUMO_DIM)
    atlas_img = image.resample_to_img(
        image.load_img(difumo.maps), masker.mask_img, force_resample=True, interpolation="nearest", copy_header=True
    )
    difumo_masker = NiftiMapsMasker(atlas_img)
    difumo_masker.fit()
    return difumo_masker


def project_difumo(neuro_vectors, pmids, difumo_masker, masker, device, *, chunk_rows: int):
    """Notebook cell 8 in row chunks: project, drop all-zero rows, divide by the row quantile, clamp to [0, 1]."""
    import torch
    from neurovlm.data.coordinates import build_difumo_projection_matrix, difumo_project_voxels

    P, s = build_difumo_projection_matrix(difumo_masker, masker, device=device)
    keep_rows, out_parts = [], []
    with torch.no_grad():
        for start, end in chunk_bounds(len(neuro_vectors), chunk_rows):
            X = neuro_vectors[start:end].to(device=device, dtype=torch.float32, non_blocking=True)
            X_hat = difumo_project_voxels(X, P, s)
            mask = X_hat.sum(dim=1) != 0
            X_hat = X_hat[mask]
            if len(X_hat):
                thresholds = torch.quantile(X_hat, QUANTILE, dim=1)
                X_hat_norm = X_hat / thresholds.unsqueeze(1)
                X_hat_norm[X_hat_norm < 0] = 0
                X_hat_norm[X_hat_norm > 1] = 1
                out_parts.append(X_hat_norm.cpu())
            keep_rows.append(mask.cpu())
    keep = torch.cat(keep_rows).numpy()
    X_hat_norm = torch.cat(out_parts) if out_parts else torch.empty((0, neuro_vectors.shape[1]))
    return X_hat_norm, pmids[keep]


def _render_example(i: int, vec_difumo: np.ndarray, vec_ale: np.ndarray, out: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nilearn.datasets import load_mni152_template
    from nilearn.plotting import plot_stat_map
    from neurovlm.data import load_masker

    masker = load_masker()
    temp = load_mni152_template(resolution=1)
    kw = dict(bg_img=temp, black_bg=False, draw_cross=False, colorbar=False, annotate=False, cmap="hot", vmin=0.1, vmax=1)
    difumo = plot_stat_map(masker.inverse_transform(vec_difumo), **kw)
    plt.savefig(out / f"{i:02d}_difumo.png", dpi=300)
    plt.close()
    plot_stat_map(masker.inverse_transform(vec_ale), cut_coords=difumo.cut_coords, **kw)
    plt.savefig(out / f"{i:02d}_ale.png", dpi=300)
    plt.close()


def render_examples(neuro_vectors, neuro_vectors_ale, out: Path, *, n_examples: int, n_jobs: int):
    """Notebook cells 10-11: seed-0 random studies, DiFuMo and ALE maps cut at the same slices. Skips existing PNGs."""
    import torch
    from joblib import Parallel, delayed

    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    inds = torch.randperm(len(neuro_vectors))[:n_examples]
    todo = [i for i in range(len(inds)) if not (out / f"{i:02d}_difumo.png").exists()]
    if not todo:
        return 0
    workers = max(1, min(len(todo), 8 if n_jobs < 0 else n_jobs))
    Parallel(n_jobs=workers, backend="loky")(
        delayed(_render_example)(i, neuro_vectors[inds[i]].numpy(), neuro_vectors_ale[inds[i]].numpy(), out) for i in todo
    )
    return len(todo)


def main(argv=None):
    p = base_parser("ALE-smooth PubMed coordinates and project them through DiFuMo-512.", stage=STAGE)
    add_data_args(p, amp=False)
    p.add_argument("--n-jobs", type=int, default=-1, help="joblib workers (-1 = all cores).")
    p.add_argument("--fwhm", type=int, default=FWHM, help="ALE kernel FWHM (mm).")
    p.add_argument("--ijk-mode", choices=["trunc", "round"], default="trunc",
                   help="MNI focus -> voxel: trunc = the release (nimare <= 0.15), round = nimare >= 0.16.")
    p.add_argument("--chunk-rows", type=int, default=2048, help="Rows per chunk for the DiFuMo projection.")
    p.add_argument("--n-examples", type=int, default=None, help=f"Example figure pairs (default {N_EXAMPLES}; 2 in --smoke).")
    p.add_argument("--skip-figures", action="store_true", help="Do not render difumo_examples/.")
    args = finalize_args(p.parse_args(argv))
    art = resolve_artifact_dir(args)
    setup_torch(args.seed)
    import torch

    torch.set_float32_matmul_precision("highest")  # no TF32 in the projection matmul (matches the notebook)
    device = select_device(args.device)
    n_examples = args.n_examples if args.n_examples is not None else (2 if args.smoke else N_EXAMPLES)
    meta = dict(stage=STAGE, smoke=args.smoke)

    ale_path, vec_path, fig_dir = art / "neuro_vectors_ale.pt", art / "neuro_vectors.pt", art / "difumo_examples"
    fig_targets = [fig_dir / f"{i:02d}_{k}.png" for i in range(n_examples) for k in ("ale", "difumo")]
    if artifacts_exist(ale_path, vec_path, *([] if args.skip_figures else fig_targets), force=args.force):
        log.info("artifacts exist (%s, %s, %d figures); use --force to recompute", ale_path, vec_path, len(fig_targets))
        return 0

    from neurovlm.data import load_dataset, load_masker

    with timed("load_datasets", log, out=args.out, **meta):
        df_pubs = load_dataset("pubmed_text").sort_values("pmid")
        df_coords = load_dataset("pubmed_coordinates").sort_values("pmid")
        if args.max_samples:
            keep = df_pubs["pmid"].to_numpy()[: args.max_samples]
            df_pubs = df_pubs[df_pubs["pmid"].isin(keep)]
            df_coords = df_coords[df_coords["pmid"].isin(keep)]
        pmids_sorted = df_pubs["pmid"].to_numpy()
        # the notebook pairs the ALE rows (unique coordinate pmids, ascending) with the df_pubs pmids
        if not np.array_equal(pmids_sorted, np.sort(df_coords["pmid"].unique())):
            raise RuntimeError("pubmed_text and pubmed_coordinates have different pmids")
        log.info("publications=%d coordinates=%d", len(df_pubs), len(df_coords))

    if artifacts_exist(ale_path, force=args.force):
        log.info("loading cached %s", ale_path)
        neuro_vectors_ale, pmids = torch.load(ale_path, weights_only=False).values()
        if args.max_samples and len(pmids) != len(pmids_sorted):
            raise RuntimeError(f"{ale_path} has {len(pmids)} rows but --max-samples asked for {len(pmids_sorted)}; use --force")
    else:
        with timed("ale_smoothing", log, out=args.out, n_studies=len(pmids_sorted), ijk_mode=args.ijk_mode, **meta):
            neuro_vectors_ale = ale_smooth(df_coords, pmids_sorted, fwhm=args.fwhm, n_jobs=args.n_jobs, ijk_mode=args.ijk_mode)
            pmids = pmids_sorted
            torch.save({"neuro_vectors": neuro_vectors_ale, "pmid": pmids}, ale_path)
        log.info("saved %s %s", ale_path, tuple(neuro_vectors_ale.shape))

    if artifacts_exist(vec_path, force=args.force):
        log.info("loading cached %s", vec_path)
        neuro_vectors, pmids_kept = torch.load(vec_path, weights_only=False).values()
    else:
        with timed("difumo_fit", log, out=args.out, **meta):
            masker = load_masker()
            difumo_masker = fit_difumo(masker)
        with timed("difumo_projection", log, out=args.out, n_rows=len(neuro_vectors_ale), device=str(device), **meta):
            neuro_vectors, pmids_kept = project_difumo(
                neuro_vectors_ale, pmids, difumo_masker, masker, device, chunk_rows=args.chunk_rows
            )
            torch.save({"neuro_vectors": neuro_vectors, "pmid": pmids_kept}, vec_path)
        log.info("saved %s %s (dropped %d all-zero rows)", vec_path, tuple(neuro_vectors.shape), len(pmids) - len(pmids_kept))

    # align the ALE rows with the kept DiFuMo rows (cell 9)
    mask = pd.Series(pmids).isin(pmids_kept).to_numpy()
    neuro_vectors_ale_aligned = neuro_vectors_ale[torch.as_tensor(mask)]
    assert (pmids_kept == pmids[mask]).all()

    if not args.skip_figures:
        with timed("example_figures", log, out=args.out, n_examples=n_examples, **meta):
            n_new = render_examples(neuro_vectors, neuro_vectors_ale_aligned, fig_dir, n_examples=n_examples, n_jobs=args.n_jobs)
        log.info("rendered %d new example pairs into %s", n_new, fig_dir)
    log.info("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
