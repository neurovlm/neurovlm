#!/usr/bin/env python
"""Fig. S3: PubMed test recall@k of NeuroVLM vs NiCLIP vs NeuroConText (docs/figures/23_versus_others.ipynb), on the test
papers in neither baseline's training set. The baselines come from their authors' releases (scripts/data/fetch_baselines.py)
and are evaluated on DiFuMo-512 features of the test maps and 7B embeddings of the test texts (Mistral-7B-v0.1 and
BrainGPT-7B-v0.2 at pinned revisions, one pass each, cached). A baseline that cannot be fetched or evaluated is drawn
from baseline_curves.csv, which this script rewrites whenever both were evaluated.

    .venv/bin/python scripts/figures/main/figS3_recall_comparison.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
from _shared import (
    FIG_HALF_W, SERIES, banner, cache_file, fig_parser, finalize, finish, free_gpu, get_logger, hf_pinned, load_models,
    load_pt, note_source, pubmed_keep, pubmed_test_mask, recall_axes, run_file, save_pt, select_device, timed,
    text_width_in, to_device_batches, trapz_auc,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data"))  # scripts/data/
from fetch_baselines import ensure as fetch_baselines  # noqa: E402

STEM = "figS3_recall_comparison"
log = get_logger(STEM)
MISTRAL = "mistralai/Mistral-7B-v0.1"  # NeuroConText's text encoder, and the base of BrainGPT
BRAINGPT = "BrainGPT/BrainGPT-7B-v0.2"  # NiCLIP's text encoder: a LoRA adapter on Mistral-7B-v0.1
NICLIP_CKPT = "niclip/model-clip_section-body_embedding-BrainGPT-7B-v0.2_best.pth"
NICLIP_SPLIT = "niclip/model-gclda_cogatlasred-task_embedding-BrainGPT-7B-v0.2_section-body.pkl"
NC_REPO = "neurovlm/neurocontext-baseline"  # the checkpoint: NeuroConText's training script on its authors' data (they publish none)
NC_CKPT, NC_TRAIN, NC_PIPE = "neurocontext/best_val.pt", "neurocontext/train_pmids.pkl", "neurocontext/text_preprocessing_pipeline.pkl"
MODELS = ("NeuroVLM", "NiCLIP", "NeuroConText")


def parse_args():
    p = fig_parser("Fig. S3: contrastive recall, NeuroVLM vs NiCLIP vs NeuroConText.")
    p.add_argument("--step", type=int, default=10)
    p.add_argument("--baseline-csv", type=Path, default=Path(__file__).with_name("baseline_curves.csv"),
                   help="Precomputed curves (model,direction,normalized_k,recall) for baselines that cannot be evaluated.")
    p.add_argument("--no-fair-subset", action="store_true", help="Evaluate on the whole test split.")
    return finalize(p.parse_args())


def _build_models():
    """Loaders of the NiCLIP and NeuroConText checkpoints (architectures copied from their repositories)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ResHead(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.gelu, self.fc, self.dropout, self.layer_norm = nn.GELU(), nn.Linear(dim, dim), nn.Dropout(0.1), nn.LayerNorm(dim)

        def forward(self, x):
            return self.layer_norm(x + self.dropout(self.gelu(self.fc(x))))

    class ProjHead(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.projection, self.gelu, self.fc = nn.Linear(in_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim)
            self.dropout, self.layer_norm = nn.Dropout(0.1), nn.LayerNorm(out_dim)

        def forward(self, x):
            p = self.projection(x)
            return self.layer_norm(self.dropout(self.fc(self.gelu(p))) + p)

    class NiImage(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.model = nn.Sequential(ResHead(dim), ResHead(dim), ResHead(dim))

        def forward(self, x):
            return self.model(x)

    class NiText(nn.Module):
        def __init__(self, text_dim, out_dim):
            super().__init__()
            self.model = nn.Sequential(ProjHead(text_dim, out_dim), ResHead(out_dim), ResHead(out_dim))

        def forward(self, x):
            return self.model(x)

    class NiCLIP(nn.Module):
        def __init__(self, text_dim, img_dim):
            super().__init__()
            self.image_model, self.text_model = NiImage(img_dim), NiText(text_dim, img_dim)
            self.logit_scale = nn.Parameter(torch.ones([]))

        def encode_image(self, x):
            return F.normalize(self.image_model(x), dim=1)

        def encode_text(self, x):
            return F.normalize(self.text_model(x), dim=1)

    class NeuroConText(nn.Module):
        def __init__(self, text_dim, img_dim):
            super().__init__()
            self.image_model = nn.Sequential(ResHead(img_dim), ResHead(img_dim), ResHead(img_dim))
            self.text_model = nn.Sequential(ProjHead(text_dim, img_dim), ResHead(img_dim), ResHead(img_dim))
            self.logit_scale = nn.Parameter(torch.ones([]))

        def encode_image(self, x):
            return F.normalize(self.image_model(x), dim=1)

        def encode_text(self, x):
            return F.normalize(self.text_model(x), dim=1)

    def load_niclip(path):
        sd = torch.load(path, weights_only=True, map_location="cpu")
        m = NiCLIP(sd["text_model.model.0.projection.weight"].shape[1], sd["image_model.model.0.fc.weight"].shape[1])
        m.load_state_dict(sd)
        return m.eval()

    def load_neurocontext(path):
        sd = torch.load(path, weights_only=True, map_location="cpu")
        m = NeuroConText(sd["text_model.0.projection.weight"].shape[1], sd["image_model.0.fc.weight"].shape[1])
        m.load_state_dict(sd)
        return m.eval()

    return load_niclip, load_neurocontext


def _niclip_train_pmids(path: Path) -> set[int]:
    """NiCLIP's training PMIDs: the ids ('<pmid>-<experiment>') of the GC-LDA model fit on the checkpoint's training split."""
    import gzip

    with gzip.open(path, "rb") as fh:
        model = pickle.load(fh)  # nimare GCLDAModel
    return {int(str(i).split("-")[0]) for i in model.ids}


def _merge_lora(model, repo: str):
    """Fold the LoRA adapter of ``repo`` into its base model: W += alpha / r · B A on every target module."""
    import json

    import torch
    from huggingface_hub import hf_hub_download

    rev = hf_pinned(repo)
    cfg = json.loads(Path(hf_hub_download(repo, "adapter_config.json", revision=rev)).read_text())
    lora = torch.load(hf_hub_download(repo, "adapter_model.bin", revision=rev), weights_only=True, map_location="cpu")
    params = dict(model.named_parameters())
    with torch.no_grad():
        for key, a in lora.items():
            if ".lora_A." not in key:
                continue
            b = lora[key.replace(".lora_A.", ".lora_B.")]
            w = params[key.replace("base_model.model.", "", 1).replace(".lora_A.weight", ".weight")]
            w += (cfg["lora_alpha"] / cfg["r"] * b.float() @ a.float()).to(w.dtype)


def _llm_embeddings(texts, repo: str, cache_path: Path, device, *, adapter: str | None = None, max_length: int,
                    batch_size: int = 4):
    """Mean last hidden state of each text under the pinned 7B model (bf16), as the notebook's cells 12-13."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rev = hf_pinned(repo)
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(repo, revision=rev, dtype=torch.bfloat16).eval()
    if adapter:
        _merge_lora(model, adapter)
    model.to(device)
    out = []
    for s in range(0, len(texts), batch_size):
        enc = tok(texts[s:s + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            h = model(**enc, output_hidden_states=True).hidden_states[-1].float()
        m = enc["attention_mask"].unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu())
        if (s // batch_size) % 100 == 0:
            log.info("%s embeddings %d/%d", adapter or repo, min(s + batch_size, len(texts)), len(texts))
    del model
    free_gpu()
    emb = torch.cat(out)
    save_pt(cache_path, emb)
    return emb


def _difumo_512(images, cache_path: Path):
    """DiFuMo-512 (2 mm) features of the test maps, in chunks of 100."""
    import torch
    from nilearn import datasets as nl_datasets
    from nilearn.image import concat_imgs
    from nilearn.maskers import NiftiMapsMasker
    from _shared import load_masker

    masker = load_masker()
    difumo = nl_datasets.fetch_atlas_difumo(dimension=512, resolution_mm=2)
    m512 = NiftiMapsMasker(maps_img=difumo.maps).fit()
    feats = []
    for s in range(0, len(images), 100):
        imgs = concat_imgs([masker.inverse_transform(np.asarray(x)) for x in images[s:s + 100]])
        feats.append(torch.as_tensor(m512.transform(imgs), dtype=torch.float32))
        log.info("DiFuMo-512 features %d/%d", min(s + 100, len(images)), len(images))
    out = torch.cat(feats)
    save_pt(cache_path, out)
    return out


def compute(args, device):
    cache = cache_file(args, "figS3_curves", excl=True)
    if cache.exists() and not args.force:
        log.info("cache hit %s", cache)
        return load_pt(cache)
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from _shared import load_dataset, load_latent
    from neurovlm.metrics.retrieval import recall_curve

    curves, notes = {}, {}
    with timed("NeuroVLM test-split embeddings", log, args.cache_dir, stage=STEM):
        df_pubs = load_dataset("pubmed_text")
        df_test = df_pubs[pubmed_test_mask(args, df_pubs, log)].sort_values("pmid").reset_index(drop=True)
        test_pmids = df_test["pmid"].to_numpy()
        lt, pt = load_latent("pubmed_text")
        mt = pd.Series(np.asarray(pt)).isin(test_pmids).to_numpy()
        lt = lt[mt].float()
        heads = load_models(["proj_head_text_infonce", "proj_head_image_infonce"], device, args.model_dir, log)
        images = None
        if run_file(args.model_dir, "autoencoder") is not None:  # the run's autoencoder: re-encode the test maps
            ae = load_models(["autoencoder"], device, args.model_dir, log)["autoencoder"]
            imgs_all, pi = load_dataset("pubmed_images")
            mi = pd.Series(np.asarray(pi)).isin(test_pmids).to_numpy()
            images = imgs_all[mi].float()
            li = to_device_batches(ae.encoder, images, device, batch_size=1024)
        else:
            li, pi = load_latent("pubmed_images")
            note_source("autoencoder", "released:neurovlm/NeuroAutoEncoder (HF pubmed_images latents)")
            mi = pd.Series(np.asarray(pi)).isin(test_pmids).to_numpy()
            li = li[mi].float()
        if not (np.array_equal(np.asarray(pt)[mt], test_pmids) and np.array_equal(np.asarray(pi)[mi], test_pmids)):
            raise RuntimeError("HF latents are not aligned with the sorted PubMed test split")
        with torch.no_grad():
            z_t = F.normalize(heads["proj_head_text_infonce"](F.normalize(lt.to(device), dim=1)), dim=1).cpu()
            z_i = F.normalize(heads["proj_head_image_infonce"](li.to(device)), dim=1).cpu()
        n_test = len(test_pmids)

    # the baselines, fetched from their authors' releases; whatever cannot be fetched is drawn from --baseline-csv
    try:
        files = fetch_baselines()
    except Exception as exc:  # network, a changed file, ...
        log.warning("baselines not fetched: %s; NiCLIP and NeuroConText come from %s", exc, args.baseline_csv)
        files = {}
    try:
        files[NC_CKPT] = Path(hf_pinned(NC_REPO, "best_val.pt", "model"))
    except Exception as exc:
        log.warning("NeuroConText checkpoint not fetched: %s; its curve comes from %s", exc, args.baseline_csv)
    notes["baselines"] = sorted(files)  # manifest names of the files in place

    # fair subset: test papers in neither baseline's training set
    fair = np.ones(n_test, dtype=bool)
    fair_file = Path(args.baseline_csv).with_name("baseline_fair_pmids.txt")  # the subset the precomputed curves are on
    if not args.no_fair_subset and NICLIP_SPLIT in files and NC_TRAIN in files:
        s1 = _niclip_train_pmids(files[NICLIP_SPLIT])
        with files[NC_TRAIN].open("rb") as fh:
            s2 = set(map(int, pickle.load(fh)))
        in1 = np.array([int(p) in s1 for p in test_pmids])
        in2 = np.array([int(p) in s2 for p in test_pmids])
        fair = ~(in1 | in2)
        notes["fair_subset"] = {"n_test": int(n_test), "in_niclip_train": int(in1.sum()), "in_neurocontext_train": int(in2.sum()),
                                "n_fair": int(fair.sum())}
        log.info("fair subset: %d / %d test papers", fair.sum(), n_test)
    elif not args.no_fair_subset and fair_file.exists():
        keep = set(map(int, fair_file.read_text().split()))
        fair = np.array([int(p) in keep for p in test_pmids])
        notes["fair_subset"] = f"read from {fair_file.name} ({int(fair.sum())} papers, the subset of the precomputed curves)"
        log.warning("training pmid lists not fetched; fair subset read from %s", fair_file.name)
    else:
        notes["fair_subset"] = "not applied (training pmid lists unavailable or --no-fair-subset)"
        if not args.no_fair_subset:
            log.warning("training pmid lists not fetched; using the whole test split")
    excl_ok = pubmed_keep(args, test_pmids, key="PubMed test", log=log)
    notes["eval_exclusions_dropped_from_evaluated_subset"] = int((fair & ~excl_ok).sum())
    idx = np.flatnonzero(fair & excl_ok)
    if args.max_samples:
        idx = idx[: args.max_samples]
    n = len(idx)
    x = np.arange(0, n, args.step) / n

    def add_curve(name, zt, zi):
        t2i, i2t = recall_curve(zt, zi, step=args.step)
        curves[name] = {"t2i": t2i.numpy(), "i2t": i2t.numpy(), "auc_t2i": trapz_auc(x, t2i.numpy()),
                        "auc_i2t": trapz_auc(x, i2t.numpy()), "source": "evaluated"}
        log.info("%s n=%d AUC t2i=%.3f i2t=%.3f", name, n, curves[name]["auc_t2i"], curves[name]["auc_i2t"])

    add_curve("NeuroVLM", z_t[idx], z_i[idx])

    # baselines: their checkpoints on DiFuMo-512 features of the test maps and 7B embeddings of the test texts
    load_niclip, load_neurocontext = _build_models()
    texts = (df_test["name"] + " [SEP] " + df_test["description"]).tolist()
    img512 = None
    if NICLIP_CKPT in files or NC_CKPT in files:
        p = cache_file(args, "figS3_img_emb_512", tagged=False)
        if p.exists():
            img512 = load_pt(p)
        else:
            with timed("DiFuMo-512 features", log, args.cache_dir, stage=STEM):
                if images is None:
                    imgs_all, pi = load_dataset("pubmed_images")
                    images = imgs_all[pd.Series(np.asarray(pi)).isin(test_pmids).to_numpy()].float()
                    del imgs_all
                img512 = _difumo_512(images, p)

    def baseline(name, run):
        try:
            with timed(f"{name} evaluation", log, args.cache_dir, stage=STEM):
                run()
        except Exception as exc:
            notes[name] = f"precomputed: {type(exc).__name__}: {exc}"
            log.warning("%s not evaluated (%s); curve from %s", name, exc, args.baseline_csv)

    def run_niclip():
        p = cache_file(args, "figS3_text_braingpt", tagged=False)
        emb = load_pt(p) if p.exists() else _llm_embeddings(texts, MISTRAL, p, device, adapter=BRAINGPT, max_length=4096)
        m = load_niclip(files[NICLIP_CKPT])
        with torch.no_grad():
            add_curve("NiCLIP", m.encode_text(emb[idx]), m.encode_image(img512[idx]))

    def run_neurocontext():
        p = cache_file(args, "figS3_text_mistral", tagged=False)
        emb = load_pt(p) if p.exists() else _llm_embeddings(texts, MISTRAL, p, device, max_length=8192)
        with files[NC_PIPE].open("rb") as fh:
            pipe = pickle.load(fh)  # L2 normalisation, then the StandardScaler fit on the training texts
        txt = torch.tensor(pipe.transform(F.normalize(emb, dim=1).numpy()).astype("float32"))
        m = load_neurocontext(files[NC_CKPT])
        with torch.no_grad():
            add_curve("NeuroConText", m.encode_text(txt[idx]), m.encode_image(img512[idx]))

    if NICLIP_CKPT in files:
        baseline("NiCLIP", run_niclip)
    else:
        notes["NiCLIP"] = "precomputed: checkpoint not fetched"
    if NC_CKPT in files and NC_PIPE in files:
        baseline("NeuroConText", run_neurocontext)
    else:
        notes["NeuroConText"] = "precomputed: checkpoint or text pipeline not fetched"
    # precomputed curves for anything still missing
    if args.baseline_csv and Path(args.baseline_csv).exists():
        df = pd.read_csv(args.baseline_csv)
        for name, g in df.groupby("model"):
            if name in curves:
                continue
            entry = {"source": f"csv:{Path(args.baseline_csv).name} (precomputed curve on the fair subset)"}
            for direction in ("t2i", "i2t"):
                gd = g[g["direction"] == direction].sort_values("normalized_k")
                if len(gd):
                    entry[direction] = {"x": gd["normalized_k"].to_numpy(), "y": gd["recall"].to_numpy()}
                    entry[f"auc_{direction}"] = trapz_auc(gd["normalized_k"], gd["recall"])
            if "t2i" in entry or "i2t" in entry:
                curves[name] = entry
                log.info("%s curves loaded from %s", name, args.baseline_csv)
    if all(curves.get(m, {}).get("source") == "evaluated" for m in MODELS[1:]) and isinstance(notes["fair_subset"], dict):
        rows = [(m, d, float(k), float(r)) for m in MODELS[1:] for d in ("t2i", "i2t") for k, r in zip(x, curves[m][d])]
        pd.DataFrame(rows, columns=["model", "direction", "normalized_k", "recall"]).to_csv(args.baseline_csv, index=False)
        fair_file.write_text("".join(f"{int(p)}\n" for p in test_pmids[fair]))
        log.info("precomputed curves and their subset written to %s, %s", args.baseline_csv, fair_file.name)
    free_gpu()
    payload = {"x": x, "n": n, "n_test": int(n_test), "step": args.step, "curves": curves, "notes": notes}
    save_pt(cache, payload)
    return payload


def auc_table(ax, rows, fs):
    """AUC table in the empty lower right of a square recall panel, below the chance diagonal: per model a colour
    handle, its name and its text→image and image→text AUC; the two AUC columns sit under one "AUC" header that spans
    them and, below it, a solid and a dashed line sample (one per column)."""
    a_in = ax.get_position().width * ax.figure.get_figwidth()
    pitch = fs * 1.3 / 72.0
    num_w = text_width_in("0.00", fs)
    num_r = [a_in - 0.06 - num_w - 0.1, a_in - 0.06]  # right edges of the two AUC columns
    num_c = [r - num_w / 2 for r in num_r]  # their centres
    x_name = num_r[0] - num_w - 0.1 - max(text_width_in(name, fs) for name, *_ in rows)

    def at(x_in, y_in):
        return x_in / a_in, y_in / a_in

    kw = dict(transform=ax.transAxes, fontsize=fs, va="center")
    ax.text(*at(sum(num_c) / 2, 0.05 + (len(rows) + 1.5) * pitch), "AUC", ha="center", **kw)  # over both columns
    y = 0.05 + (len(rows) + 0.5) * pitch
    for c, ls in zip(num_c, ("-", "--")):
        ax.plot(*zip(at(c - 0.1, y), at(c + 0.1, y)), transform=ax.transAxes, color="black", ls=ls, lw=1.0)
    for i, (name, col, auc_t2i, auc_i2t) in enumerate(rows):
        y = 0.05 + (len(rows) - i - 0.5) * pitch
        ax.plot(*zip(at(x_name - 0.2, y), at(x_name - 0.05, y)), transform=ax.transAxes, color=col, lw=1.6,
                solid_capstyle="butt")
        ax.text(*at(x_name, y), name, ha="left", **kw)
        for r, v in zip(num_r, (auc_t2i, auc_i2t)):
            if v is not None:
                ax.text(*at(r, y), f"{v:.2f}", ha="right", **kw)


def render(args, payload):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams["savefig.bbox"] = "standard"  # the saved figure is exactly FIG_HALF_W square
    fs = args.font_size
    w = FIG_HALF_W
    left, bottom, top = 0.46, 0.36, 0.21  # inches: y label and ticks, x label and ticks, title
    side = w - bottom - top  # square axes
    fig = plt.figure(figsize=(w, w))
    ax = fig.add_axes([left / w, bottom / w, side / w, side / w])
    recall_axes(ax, title="PubMed contrastive recall")
    order = [m for m in MODELS if m in payload["curves"]] + [m for m in payload["curves"] if m not in MODELS]
    rows = []
    for k, name in enumerate(order):
        c = payload["curves"][name]
        col = SERIES[k % len(SERIES)]
        rows.append((name, col, c.get("auc_t2i"), c.get("auc_i2t")))
        for direction, ls in (("t2i", "-"), ("i2t", "--")):
            if direction not in c:
                continue
            xy = c[direction]
            xs, ys = (xy["x"], xy["y"]) if isinstance(xy, dict) else (payload["x"], xy)
            ax.plot(xs, ys, ls=ls, color=col, lw=1.2, alpha=0.95 if ls == "-" else 0.75)
    ax.text(0.62, 0.62, "chance", rotation=45, rotation_mode="anchor", ha="center", va="center", fontsize=fs,
            color="0.3", bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    # line styles in the empty top left (the curves rise steeply), the models and their AUCs in the lower right
    ax.legend(handles=[Line2D([], [], color="black", lw=1.0, ls=ls, label=lab)
                       for ls, lab in (("-", "text→image"), ("--", "image→text"))],
              loc="upper left", fontsize=fs, handlelength=1.6, handletextpad=0.4, labelspacing=0.25, borderaxespad=0.2)
    auc_table(ax, rows, fs)
    return fig


def main():
    args = parse_args()
    banner(log, args, STEM)
    device = select_device(args.device)
    payload = compute(args, device)
    with timed("render", log, args.cache_dir, stage=STEM):
        fig = render(args, payload)
        components = ("autoencoder", "proj_head_text_infonce", "proj_head_image_infonce") + (("pmids_split",) if args.model_dir else ())
        finish(fig, STEM, args, log, components=components, text_dump={
            "n": payload["n"], "n_test": payload["n_test"], "notes": payload["notes"],
            "auc": {k: {d: v.get(f"auc_{d}") for d in ("t2i", "i2t")} | {"source": v["source"]}
                    for k, v in payload["curves"].items()}})


if __name__ == "__main__":
    main()
