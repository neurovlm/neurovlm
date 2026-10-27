#!/usr/bin/env python
"""Fig. 6: autoencoder reconstructions of 8 network, NeuroVault and PubMed test maps (docs/figures/16_qualatative_auto).
The examples and their order are those of the submitted figure, which was assembled by hand from the notebook's
per-example plots (data_dir/ae_qual): the notebook's random draws and network_examples.csv, with the csv's networks 5-8
shown first. One axial view per example, the axial slice of the submitted figure's views.

    .venv/bin/python scripts/figures/main/fig6_autoencoder_qual.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from _shared import (
    FIG_FULL_W, banner, cache_file, data_dir, draw_brain_views, fig_parser, finalize, finish, free_gpu, get_logger,
    load_models, load_network_list, load_pt, mask_vector, neurovault_keep, pubmed_keep, rel_path, save_pt,
    select_device, pubmed_split, timed, to_device_batches,
)

STEM = "fig6_autoencoder_qual"
log = get_logger(STEM)
GROUPS = ("Networks", "NeuroVault", "PubMed")
COMPONENTS = ("autoencoder", "pmids_split")
NETWORK_CSV = Path(__file__).with_name("network_examples.csv")  # the 8 curated networks (notebook 15)
NETWORK_ORDER = [4, 5, 6, 7, 0, 1, 2, 3]  # csv rows in the submitted figure's order (default mode, motor, ... first)


def parse_args():
    p = fig_parser("Fig. 6: autoencoder reconstructions.")
    p.add_argument("--n-cols", type=int, default=8, help="Examples per dataset (<= 8).")
    return finalize(p.parse_args())


def _clean_choice(n: int, k: int, ok: np.ndarray) -> np.ndarray:
    """np.random.choice(np.arange(n), k) as in the notebook; excluded rows are redrawn from the same stream, so the
    notebook's examples come back whenever none of them is excluded."""
    inds = np.random.choice(np.arange(n), k)
    for j in range(k):
        while not ok[inds[j]]:
            inds[j] = np.random.choice(np.arange(n), 1)[0]
    return inds


def compute(args, device):
    """Per dataset the autoencoder input, the map drawn as True and the reconstruction (sigmoid), in the submitted
    figure's order, and the ids of the examples."""
    cache = cache_file(args, "fig6_examples", excl=True)
    if cache.exists() and not args.force:
        log.info("cache hit %s", cache)
        return load_pt(cache)
    import pandas as pd
    import torch
    from nilearn.image import resample_to_img
    from _shared import load_dataset, load_masker

    dd = data_dir()
    masker = load_masker()
    ae = load_models(["autoencoder"], device, args.model_dir, log)["autoencoder"]
    notes = {}
    with timed("select examples", log, args.cache_dir, stage=STEM):
        # PubMed: 8 random test maps (seed 0)
        nv_path = dd / "neuro_vectors.pt"
        if nv_path.exists():  # split: the run's pmids_split.pt, else the pinned HF split flags
            payload = torch.load(nv_path, map_location="cpu", weights_only=False, mmap=True)
            split, _ = pubmed_split(args)
            test_rows = np.flatnonzero(pd.Series(np.asarray(payload["pmid"])).isin(split["test"]).to_numpy())
            ok = pubmed_keep(args, np.asarray(payload["pmid"])[test_rows], key="PubMed test", log=log)
            np.random.seed(0)
            inds = _clean_choice(len(test_rows), 8, ok)
            X_pm = np.asarray(payload["neuro_vectors"][torch.as_tensor(test_rows[inds])], dtype="float32")
            notes["PubMed"] = {"source": "neuro_vectors.pt test split, np.random.seed(0)",
                               "pmids": np.asarray(payload["pmid"])[test_rows[inds]].tolist()}
        else:
            log.warning("neuro_vectors.pt missing: sampling HF pubmed_images with the test flag "
                        "(different ordering -> different random examples than the notebook)")
            images, pi = load_dataset("pubmed_images")
            df = load_dataset("pubmed_text")
            rows = np.flatnonzero(pd.Series(np.asarray(pi)).isin(df.loc[df["test"].fillna(False).astype(bool), "pmid"]).to_numpy())
            ok = pubmed_keep(args, np.asarray(pi)[rows], key="PubMed test", log=log)
            np.random.seed(0)
            inds = _clean_choice(len(rows), 8, ok)
            X_pm = images[torch.as_tensor(rows[inds])].float().numpy()
            notes["PubMed"] = {"source": "HF pubmed_images (test flag)", "pmids": np.asarray(pi)[rows[inds]].tolist()}
        # networks: the 8 maps of network_examples.csv (indices into the non-ICA list); the repo copy, else data_dir's
        nets = load_network_list()
        csv = NETWORK_CSV if NETWORK_CSV.exists() else dd / "network_examples.csv"
        if csv.exists():
            net_inds = pd.read_csv(csv).iloc[:, 0].to_numpy().astype(int)[NETWORK_ORDER]
            notes["Networks"] = {"source": f"{rel_path(csv)}, rows {NETWORK_ORDER} (the submitted figure's order)"}
        else:
            log.warning("%s and %s missing: choosing 8 networks at random (seed 0)", NETWORK_CSV, csv)
            net_inds = np.random.RandomState(0).choice(len(nets), 8, replace=False)
            notes["Networks"] = {"source": "random (network_examples.csv missing)"}
        notes["Networks"]["maps"] = [f"{nets[i][0]}/{nets[i][1]}" for i in net_inds]
        X_net = np.stack([
            mask_vector(masker, resample_to_img(nets[i][2], masker.mask_img, force_resample=True, copy_header=True))
            for i in net_inds
        ]).astype("float32")
        # NeuroVault: 8 random clustered maps (seed 1); the autoencoder sees them binarised, True shows the cluster values
        nvp = dd / "neurovault.pt"
        if nvp.exists():
            nv = torch.load(nvp, map_location="cpu", weights_only=False)
            clust = np.asarray(nv["neuro_clustered"])
            meta = nv["df_neuro"].reset_index(drop=True)
            ok = neurovault_keep(args, meta["doi"].tolist(), key="NeuroVault images", log=log, image_ids=meta["id"].to_numpy())
            np.random.seed(1)
            nv_inds = _clean_choice(len(clust), 8, ok)
            T_nv = clust[nv_inds].astype("float32")
            notes["NeuroVault"] = {"source": "neurovault.pt neuro_clustered, np.random.seed(1)", "rows": nv_inds.tolist(),
                                   "image_ids": meta.loc[nv_inds, "id"].astype(int).tolist(),
                                   "collection_ids": meta.loc[nv_inds, "collection_id"].astype(int).tolist(),
                                   "dois": meta.loc[nv_inds, "doi"].astype(str).tolist()}
        else:
            log.warning("neurovault.pt missing: using HF neurovault_images")
            img = load_dataset("neurovault_images").float()
            meta = load_dataset("neurovault_images_meta").reset_index(drop=True)
            finite = ~img.isnan().any(dim=1)
            img, meta = img[finite], meta[finite.numpy()].reset_index(drop=True)
            ok = neurovault_keep(args, meta["doi"].tolist(), key="NeuroVault images", log=log, image_ids=meta["id"].to_numpy())
            np.random.seed(1)
            nv_inds = _clean_choice(len(img), 8, ok)
            T_nv = img[torch.as_tensor(nv_inds)].numpy()
            notes["NeuroVault"] = {"source": "HF neurovault_images", "rows": nv_inds.tolist(),
                                   "image_ids": meta.loc[nv_inds, "id"].astype(int).tolist()}
        X_nv = (T_nv > 0).astype("float32")
    with timed("reconstruct", log, args.cache_dir, stage=STEM):
        sets = {}
        for name, X, T in (("Networks", X_net, X_net), ("NeuroVault", X_nv, T_nv), ("PubMed", X_pm, X_pm)):
            recon = torch.sigmoid(to_device_batches(ae, X, device, batch_size=8)).numpy().astype("float32")
            sets[name] = {"input": X, "true": T, "pred": recon}
    del ae
    free_gpu()
    payload = {"sets": sets, "notes": notes}
    save_pt(cache, payload)
    return payload


def cluster_99(img):
    """The notebook's display threshold: voxels above the map's 99th percentile, clusters of at least 50 voxels."""
    from nilearn.image import threshold_img

    try:
        return threshold_img(img, "99%", cluster_threshold=50, two_sided=False, copy_header=True)
    except Exception:
        return img


def axial_cut(pred_img) -> float:
    """The submitted figure's slice: z of nilearn's cut for the thresholded reconstruction (plot_stat_map's default
    find_xyz_cut_coords), at which the true map was drawn too."""
    from nilearn.plotting import find_xyz_cut_coords

    return float(find_xyz_cut_coords(pred_img, activation_threshold=1e-6)[2])


def render(args, payload):
    """Per dataset a True and a Predicted row of n_cols examples in one axial view; both maps of an example share the
    cut (axial_cut). Reconstructions are drawn on [0, 1], true maps on [0, their maximum], as the notebook."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from _shared import load_masker

    plt.rcParams["savefig.bbox"] = "standard"  # the saved figure is exactly FIG_FULL_W wide
    masker = load_masker()
    n_cols = min(8, args.n_cols, args.max_samples or 8)
    fs = args.font_size
    left_in, gap_in, pad, group_gap = 0.34, 0.03, 0.025, 0.08
    cell_w = (FIG_FULL_W - left_in - 0.02 - gap_in * (n_cols - 1)) / n_cols
    brain_h = (cell_w - 2 * pad) * 182 / 148  # an axial view at the full cell width (_shared.BRAIN_MM)
    group_h = 2 * brain_h + 3 * pad
    fig_h = len(GROUPS) * group_h + (len(GROUPS) - 1) * group_gap + 0.04
    fig = plt.figure(figsize=(FIG_FULL_W, fig_h))
    W, H = FIG_FULL_W, fig_h
    views = {}
    for gi, name in enumerate(GROUPS):
        true, pred = payload["sets"][name]["true"], payload["sets"][name]["pred"]
        y_group_top = H - 0.02 - gi * (group_h + group_gap)
        fig.text(0.07 / W, (y_group_top - group_h / 2) / H, name, rotation=90, ha="center", va="center", fontsize=fs + 2)
        for r, lab in enumerate(("True", "Predicted")):
            y_mid = y_group_top - pad - r * (brain_h + pad) - brain_h / 2
            fig.text(0.24 / W, y_mid / H, lab, rotation=90, ha="center", va="center", fontsize=fs + 1)
        views[name] = []
        for c in range(n_cols):
            x0 = left_in + c * (cell_w + gap_in)
            pred_img = cluster_99(masker.inverse_transform(pred[c]))
            true_img = masker.inverse_transform(true[c])
            if name == "PubMed":
                true_img = cluster_99(true_img)
            z = axial_cut(pred_img)
            views[name].append({"view": "z", "cut_mm": z})
            for r, (img, vmax) in enumerate(((true_img, None), (pred_img, 1.0))):
                box = (x0 + pad, y_group_top - pad - r * (brain_h + pad) - brain_h, cell_w - 2 * pad, brain_h)
                draw_brain_views(fig, img, box, views="z", cut_coords=[0.0, 0.0, z], threshold=1e-6, vmin=0.0, vmax=vmax)
            fig.add_artist(Rectangle((x0 / W, (y_group_top - group_h) / H), cell_w / W, group_h / H,
                                     transform=fig.transFigure, fill=False, edgecolor="0.25", lw=0.6))
    return fig, {"examples": "the submitted Fig. 6 (paper/figures_back/autoencoder_qual.png), assembled by hand from "
                             "notebook 16's per-example plots (data_dir/ae_qual/ae_<k>_<set>_<orig|pred>.png): networks "
                             "ae_04-07 then ae_00-03, NeuroVault and PubMed ae_00-07",
                 "view_rule": "axial for every example, at z of nilearn's find_xyz_cut_coords of the thresholded "
                              "reconstruction (the cut of the submitted figure's views); the true map is drawn at the "
                              "same slice",
                 "display": "reconstructions and PubMed true maps: voxels above their 99th percentile in clusters of "
                            ">= 50 voxels; network and NeuroVault true maps as they are; reconstructions coloured on "
                            "[0, 1], true maps on [0, their maximum] (as notebook 16)",
                 "views": views}


def main():
    args = parse_args()
    banner(log, args, STEM)
    device = select_device(args.device)
    payload = compute(args, device)
    with timed("render", log, args.cache_dir, stage=STEM):
        fig, display = render(args, payload)
        finish(fig, STEM, args, log, text_dump=payload["notes"] | display, components=COMPONENTS)


if __name__ == "__main__":
    main()
