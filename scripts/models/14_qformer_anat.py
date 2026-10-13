#!/usr/bin/env python
"""Stage 14: stage 2 projector (canonical), fine-tunes stage 13 on curated [NETWORK]/[REGION]/[FUNCTION] texts.

Port of docs/03_models/14_qformer_anat.ipynb. Brain latents are synthesised by the anatomical head (each map divided by
its maximum and binarised at its 95th percentile; --map-calibration none --synth-binarize-pct 0 = notebook). --seed is
the training seed, --split-seed the validation split. File names keep the qformer_ prefix for either projector.

Training recipe (defaults since 2026-10-10, see scripts/README.md "Salience (Fig. 2)"): the canonical projection of the
training semantics uses each row's own level bank, as inference does (--train-bank level); each training map is
re-binarised per batch at a random percentile with voxel dropout (--map-aug); the title tokens weigh 5x in the LM loss
(--title-weight 5). --train-bank all --no-map-aug --title-weight 1 = the previous recipe.

    .venv/bin/python scripts/models/14_qformer_anat.py --run-name <run>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common_models import *  # noqa: E402,F403

STAGE = "14_qformer_anat"
log = get_logger(STAGE)

SEED = 123
CANONICAL_PROJ_TEMP = 0.05
MAX_TOKENS, TRAIN_BATCH, EVAL_BATCH, VAL_FRAC = 256, 18, 18, 0.1
NUM_EPOCHS, LR = 30, 1e-4
SEMANTIC_IMAGE_MIX_P = 0.5  # training input: image semantics with this probability, else text; validation: image
SYNTH_BINARIZE_PCT = 95.0  # NeuroVLM's rule for a continuous brain image; 0 = notebook
MAP_AUG_PCT = (90.0, 97.0)  # --map-aug: binarisation percentile drawn uniformly from this range
MAP_AUG_DROP, MAP_AUG_DROP_P = 0.10, 0.5  # --map-aug: drop this fraction of voxels in a map with this probability
TITLE_WEIGHT = 5.0
TITLE_START = 3  # tokens 0-2 are the [LEVEL] tag; the title runs from here to the first newline
REFS = [["Laird", "Visual1"], ["Glasser", "Language"], ["Shirer", "Auditory"], ["WashU", "LateralSM"],
        ["Du", "FPN-A"], ["YeoLab", "DefaultA"], ["WashU", "Salience"]]


def binarize_maps(maps, pct: float, *, chunk: int = 256):
    """1 where a map is >= its own pct-th percentile, else 0 (as NeuroVLM._encode_brain_image)."""
    import torch

    out = torch.empty_like(maps)
    for s in range(0, len(maps), chunk):
        x = maps[s:s + chunk]
        out[s:s + chunk] = (x >= torch.quantile(x, pct / 100.0, dim=1, keepdim=True)).to(maps.dtype)
    return out


def build_text_table(path: Path):
    """Cells 3-4: curated texts -> (DataFrame with text "[LABEL]Title\\nbody" and title, SPECTER inputs "Title [SEP] body")."""
    import pandas as pd
    import _qformer_anat_data as D

    df = pd.read_parquet(path)
    df["text"] = df["title"].astype(str) + "\n" + df["description"].astype(str)
    df["title_canonical"] = [D.canonical_title[i] for i in df["text"].str.split("\n").str[0]]
    df = df[df["text"].str.len() <= 550]
    df["desc"] = df["text"].str.split("\n").str[1:].str[0]
    df["text"] = df["title_canonical"] + "\n" + df["desc"]
    df["text"] = ["[" + D.concept_level[i].upper() + "]" for i in df["title"]] + df["text"]
    df_pubs = pd.concat((df[["text"]], D.df_func, D.df_func_2, D.df_more), ignore_index=True)
    # cell 4
    body = df_pubs["text"].str.replace(".*\n", "", regex=True)
    df_pubs["title"] = df_pubs["text"].str.split("]").str[1].str.split("\n").str[0]
    t = (df_pubs["title"] + " [SEP] " + body).tolist()
    df_pubs["text"] = df_pubs["text"].str.split("]").str[0] + "]" + df_pubs["title"] + "\n" + body
    log.info("curated rows: %d %s", len(df_pubs), df_pubs["text"].str.split("]").str[0].value_counts().to_dict())
    return df_pubs, t


def stage13_roots(args, run_dir) -> list[Path]:
    """Where to look for stage 13: this run's directory if it has its own stage-13 output, else --inputs-from first."""
    run_dir = Path(run_dir)
    own = (run_dir / "13_qformer_pubmed.summary.json").exists() or (run_dir / "neuro_qformer_pubmed" / "model.safetensors").exists()
    roots = ([run_dir] if own else []) + ([Path(args.inputs_from)] if args.inputs_from else []) + ([] if own else [run_dir])
    return list(dict.fromkeys(roots))


def _package_state(pkg_dir: Path, device):
    """Bare projector state (``qformer.*`` keys of a packaged NeuroQFormer) + its config.json."""
    from safetensors.torch import load_file

    cfg = read_json(pkg_dir / "config.json")
    state = {k[len("qformer."):]: v.to(device) for k, v in load_file(str(pkg_dir / "model.safetensors")).items() if k.startswith("qformer.")}
    return state, cfg


def resolve_grounded_state(args, run_dir, device):
    """Stage 1 projector weights -> (state, source, kind, projector_config or None).

    MLP: the stage-13 package (its best epoch). Q-Former: --grounded-qformer > stage-13 final epoch > release (pinned).
    """
    import torch

    if args.grounded_qformer is not None:
        path = Path(args.grounded_qformer)
        if path.is_dir():  # a neuro_qformer_pubmed/ package
            state, cfg = _package_state(path, device)
            kind = cfg.get("projector", "qformer")
            return state, str(path), kind, (cfg.get("projector_config") or mlp_config_from_state(state)) if kind == "mlp" else None
        state = torch.load(path, map_location=device, weights_only=True)
        kind = state_projector_kind(state)
        return state, str(path), kind, (mlp_config_from_state(state) if kind == "mlp" else None)
    for root in stage13_roots(args, run_dir):
        pkg = root / "neuro_qformer_pubmed"
        upstream = package_projector_kind(pkg)
        if upstream == "mlp" or args.projector == "mlp":
            if not (pkg / "model.safetensors").exists():
                continue
            if upstream != "mlp":
                raise ValueError(f"--projector mlp (the default) but the stage-13 package {pkg} holds a {upstream!r}; rerun 13 with "
                                 "--projector mlp, or pass --projector qformer / auto to fine-tune that Q-Former")
            state, cfg = _package_state(pkg, device)
            meta = cfg.get("training_metadata", {})
            log.info("stage-13 projector: mlp (%s checkpoint %s)", meta.get("packaged_checkpoint", "?"), meta.get("source_checkpoint", "?"))
            return state, str(pkg), "mlp", cfg.get("projector_config") or mlp_config_from_state(state)
        cands = sorted(root.glob("qformer_grounded_permissive_neuro_epoch-*.pt"), key=lambda p: int(p.stem.rsplit("-", 1)[1]))
        if cands:
            # only the final epoch of a completed stage-13 run
            last_epoch = int(cands[-1].stem.rsplit("-", 1)[1])
            summary = root / "13_qformer_pubmed.summary.json"
            if summary.exists():
                meta13 = read_json(summary)
                expected = int(meta13.get("epochs_trained", meta13.get("epochs", last_epoch + 1))) - 1
                if last_epoch != expected:
                    raise RuntimeError(f"stage 13 in {root} is incomplete: newest checkpoint is epoch-{last_epoch}, summary says "
                                       f"{expected + 1} epochs. Rerun 13 (or pass --grounded-qformer).")
            else:
                log.warning("no 13_qformer_pubmed.summary.json next to %s; cannot confirm stage 13 completed", cands[-1].name)
            state = torch.load(cands[-1], map_location=device, weights_only=True)
            return state, str(cands[-1]), state_projector_kind(state), None
    if args.projector == "mlp":
        raise FileNotFoundError(f"--projector mlp: no stage-13 MLP package (neuro_qformer_pubmed/) in {[str(r) for r in stage13_roots(args, run_dir)]}; "
                                "run 13 first, pass --grounded-qformer, or use --projector qformer / auto")
    log.warning("no stage-13 Q-Former in the run; using the released neurovlm/NeuroQformer-PubMed weights (= epoch-29)")
    released = released_qformer("pubmed", "cpu")
    rev = hf_pinned("neurovlm/NeuroQformer-PubMed")[:8]
    return {k: v.to(device) for k, v in released.qformer.state_dict().items()}, f"released:neurovlm/NeuroQformer-PubMed@{rev}", "qformer", None


def main(argv=None):
    p = base_parser("Stage 14: fine-tune the stage 1 projector on curated network/region/function texts.", stage=STAGE)
    add_model_args(p, amp_default="bf16")
    p.add_argument("--text-synth-less", type=Path, default=None, help="text_synth_less.parquet (else the pinned HF file).")
    p.add_argument("--grounded-qformer", type=Path, default=None,
                   help="Stage 13 state dict or neuro_qformer_pubmed/ package (else --inputs-from, run dir, release).")
    p.add_argument("--adapter", type=Path, default=None, help="adapter_anat_full.pt (else --inputs-from, run dir, release).")
    p.add_argument("--notebook-precision", "--deterministic", dest="notebook_precision", action="store_true",
                   help="Notebook's TF32-off precision (kernels are deterministic by default; see --no-deterministic).")
    p.add_argument("--grad-ckpt", choices=["on", "off"], default="off", help="LM gradient checkpointing (notebook: on).")
    p.add_argument("--no-generate", action="store_true", help="Skip the reference-network captions (cells 20-21).")
    p.add_argument("--num-beams", type=int, default=1, help="Beams for the reference captions (notebook: 1).")
    add_projector_args(p, allow_auto=True, mlp_shape=False)
    p.add_argument("--map-calibration", choices=("max", "none"), default="max",
                   help="Scale of the synthesised maps: max = each map divided by its max (runs/final); none = notebook. "
                        "Maps binarised at a percentile do not depend on it.")
    p.add_argument("--synth-binarize-pct", type=float, default=SYNTH_BINARIZE_PCT,
                   help="Binarise each synthesised map at this percentile before the encoder (0 = notebook's continuous maps).")
    p.add_argument("--split-seed", type=int, default=SEED,
                   help="Seed of the validation split (and --max-samples subset); fixed across a --seed sweep.")
    p.add_argument("--train-bank", choices=["level", "all"], default="level",
                   help="Canonical bank of the training semantics: level = each row's [NETWORK]/[REGION]/[FUNCTION] bank "
                        "(as inference); all = the whole bank (previous recipe).")
    p.add_argument("--map-aug", action=argparse.BooleanOptionalAction, default=True,
                   help="Re-binarise each training map per batch at a random 90-97th percentile and drop 10%% of its voxels "
                        "in half the maps (latents and image semantics recomputed; validation unaugmented).")
    p.add_argument("--title-weight", type=float, default=TITLE_WEIGHT,
                   help="LM loss weight of the title tokens (after the [LEVEL] tag, up to the first newline) in training; 1 = uniform.")
    p.set_defaults(seed=SEED)  # --seed = training seed (RNGs, batch order, semantic mixing)
    args = finalize_args(p.parse_args(argv))
    run_dir = resolve_run_dir(args)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    setup_torch(args.seed, args.deterministic)
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from torch.nn.utils.rnn import pad_sequence
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from neurovlm.models import ProjHead
    from neurovlm.models.base import NeuroAutoEncoder
    from neurovlm.models.qformer import NeuroQFormer, QFormer
    from neurovlm.resources.loaders import NEURO_QWEN_REPO_ID

    if args.notebook_precision:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    log.info("training seed %d | validation-split seed %d", args.seed, args.split_seed)

    device = select_device(args.device)
    if device.type != "cuda":
        log.warning("no CUDA: running fp32 on %s (the notebook recipe is bf16 on CUDA)", device)
    train_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    num_epochs = args.epochs or NUM_EPOCHS
    train_batch = args.batch_size or TRAIN_BATCH
    meta = dict(stage=STAGE, smoke=args.smoke, run=args.run_name)
    canonical_path = run_dir / f"qformer_canonical_bottleneck_epoch-{num_epochs - 1}.pt"
    banks_path = run_dir / "canonical_semantic_banks.pt"
    pkg_dir = run_dir / "neuro_qformer_canonical"
    if artifacts_exist(canonical_path, banks_path, pkg_dir / "model.safetensors", force=args.force):
        existing = package_projector_kind(pkg_dir)
        if args.projector not in ("auto", existing):
            raise RuntimeError(f"{pkg_dir} holds a {existing!r} projector but --projector {args.projector} was requested; "
                               "pass --force to retrain in place or use another --run-name")
        log.info("artifacts exist in %s; skipping (use --force to retrain)", run_dir)
        if args.install:
            install_artifacts({canonical_path.name: canonical_path, banks_path.name: banks_path}, log)
        return 0

    # cells 2-4
    with timed("build_texts", log, out=args.out, **meta):
        # external LLM-curated table, no in-tree producer
        tsl_path = args.text_synth_less or hf_pinned("neurovlm/synthetic_pairs", "text_synth_less.parquet")
        log.info("text_synth_less: %s", tsl_path)
        df_pubs, t = build_text_table(tsl_path)
        if args.max_samples and args.max_samples < len(df_pubs):
            keep = np.sort(np.random.default_rng(args.split_seed).choice(len(df_pubs), args.max_samples, replace=False))
            df_pubs, t = df_pubs.iloc[keep].reset_index(drop=True), [t[i] for i in keep]

    # cell 6: SPECTER -> anatomical head -> sigmoid -> calibrate (-> binarise) -> autoencoder encoder
    with timed("synthesise_latents", log, out=args.out, n=len(t), **meta):
        specter = released_model("specter", device)
        # each from the run (--inputs-from, run dir), else the pinned release; data_dir copies are never used
        rr = dict(run_dir=run_dir, args=args, logger=log)
        adapter, adapter_source = run_or_released("neuro_adapter", "adapter_anat_full.pt", device=device, explicit=args.adapter, **rr)
        ae, ae_source = run_or_released("autoencoder", "autoencoder.pt", device="cpu", builder=lambda: NeuroAutoEncoder(seed=0, out="logit"), **rr)
        enc = ae.encoder.to(device).eval()
        proj_head_text, th_source = run_or_released("proj_head_text_infonce", "proj_head_text_infonce.pt", device=device,
                                                    builder=lambda: ProjHead(768, 512, 384), **rr)
        proj_head_image, ih_source = run_or_released("proj_head_image_infonce", "proj_head_image_infonce.pt", device=device,
                                                     builder=lambda: ProjHead(384, 384, 384), **rr)
        log.info("map calibration: %s | binarised at percentile: %s", args.map_calibration, args.synth_binarize_pct or "off")
        batch_size = 2048
        latent_images = torch.zeros((len(t), 384))
        latent_text = torch.zeros((len(t), 768))
        calibrated = []  # fp16 calibrated maps, re-binarised per batch by --map-aug
        with torch.no_grad():
            for i in range(0, len(t), batch_size):
                txt = F.normalize(specter(t[i:i + batch_size]), dim=1)
                latent_text[i:i + len(txt)] = txt.cpu()
                maps = calibrate_maps(torch.sigmoid(adapter(txt)).float(), args.map_calibration)
                if args.map_aug:
                    calibrated.append(maps.half().cpu())
                if args.synth_binarize_pct:
                    maps = binarize_maps(maps, args.synth_binarize_pct)
                latent_images[i:i + len(maps)] = enc(maps).cpu()
        latent_images, latent_text = latent_images.to(device), latent_text.to(device)
        del specter, adapter
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # cell 7: keep rows whose own title is among their top-50 retrieved titles
    with torch.no_grad():
        image_semantic = F.normalize(torch.cat([proj_head_image(latent_images[s:s + 1024]).float().cpu() for s in range(0, len(latent_images), 1024)]), dim=1)
        text_semantic = F.normalize(torch.cat([proj_head_text(latent_text[s:s + 1024]).float().cpu() for s in range(0, len(latent_text), 1024)]), dim=1)
    cs = image_semantic @ text_semantic.T
    k = min(50, cs.shape[1])
    titles = df_pubs["title"].to_numpy()
    s_topk = cs.argsort(dim=1, descending=True).numpy()[:, :k]
    m = (titles[s_topk] == titles[:, None]).any(axis=1)
    df_pubs = df_pubs[m].reset_index(drop=True)
    latent_images = latent_images[torch.from_numpy(m).to(device)]
    maps_store = torch.cat(calibrated)[torch.from_numpy(m)] if args.map_aug else None
    del calibrated
    image_semantic, text_semantic = image_semantic[torch.from_numpy(m)], text_semantic[torch.from_numpy(m)]
    log.info("after self-consistency filter: %d rows %s", len(df_pubs), df_pubs["text"].str.split("]").str[0].value_counts().to_dict())

    # cells 9-10: projector init + canonical banks (``qformer`` is the projector of either kind)
    seed_all(args.seed)
    latent_images = latent_images.float().cpu()
    grounded_state, grounded_source, projector, grounded_cfg = resolve_grounded_state(args, run_dir, device)
    if args.projector not in ("auto", projector):
        raise ValueError(f"--projector {args.projector} but the grounded init {grounded_source} is a {projector!r}")
    if projector == "qformer":
        qformer = QFormer(image_dim=384, semantic_dim=384, lm_dim=1024, num_queries=32, hidden_dim=512, num_heads=8, num_layers=6).to(device).to(train_dtype)
        qformer.load_state_dict(grounded_state)
    else:  # residual MLP: fp32 weights under the bf16 autocast, as in stage 13
        qformer = build_projector("mlp", **{k: v for k, v in grounded_cfg.items() if k not in ("class", "params")})
        qformer.load_state_dict({k: v.float() for k, v in grounded_state.items()}, strict=True)
        qformer = qformer.to(device)
    projector_cfg = projector_config(qformer)
    qformer.train()
    log.info("init from stage 1 %s | projector=%s params %s", grounded_source, projector, f"{projector_cfg['params']:,}")

    bank = F.normalize(text_semantic.float(), dim=1).cpu()
    levels = df_pubs["text"].astype(str).str.extract(r"^\[([A-Z]+)\]", expand=False).str.lower().fillna("unknown").to_numpy()
    canonical_semantic_banks = {"all": bank}
    for level in ("network", "region", "function"):
        idx = np.where(levels == level)[0]
        if len(idx):
            canonical_semantic_banks[level] = bank[idx]

    @torch.no_grad()
    def project_to_canonical(E, x):
        """softmax(x E^T / temp) E, normalised."""
        E = F.normalize(E.to(device=x.device, dtype=torch.float32), dim=1)
        x = F.normalize(x.float(), dim=1)
        w = torch.softmax((x @ E.T) / CANONICAL_PROJ_TEMP, dim=-1)
        return F.normalize(w @ E, dim=1)

    def project_rows(x, row_levels):
        """Each row onto its own level's bank (--train-bank level) or all rows onto the all bank."""
        if args.train_bank == "all":
            return project_to_canonical(bank, x).cpu()
        out = torch.empty(x.shape, dtype=torch.float32)
        for lv in np.unique(row_levels):
            sel = torch.from_numpy(np.flatnonzero(row_levels == lv))
            out[sel] = project_to_canonical(canonical_semantic_banks.get(lv, bank), x[sel]).cpu()
        return out

    train_semantic_images = project_rows(image_semantic, levels)
    train_semantic_text = project_rows(text_semantic, levels)
    log.info("canonical basis sizes: %s | training semantics on the %s bank | map augmentation %s | title weight %g",
             {k: tuple(v.shape) for k, v in canonical_semantic_banks.items()}, args.train_bank, "on" if args.map_aug else "off",
             args.title_weight)

    # cell 11: frozen LM
    with timed("load_lm", log, out=args.out, **meta):
        lm_rev = hf_pinned(NEURO_QWEN_REPO_ID)
        tokenizer = AutoTokenizer.from_pretrained(NEURO_QWEN_REPO_ID, revision=lm_rev, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(NEURO_QWEN_REPO_ID, revision=lm_rev, torch_dtype=train_dtype,
                                                     attn_implementation="sdpa").to(device)
        for prm in model.parameters():
            prm.requires_grad = False
        lm_sdpa_backends(args.deterministic, log)
        if args.grad_ckpt == "on":
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.eval()
        PAD_ID = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        EOS_ID = tokenizer.eos_token_id
        model_dtype = next(model.parameters()).dtype
        with torch.no_grad():
            LM_EMB_NORM = model.get_input_embeddings().weight.detach().float().norm(dim=1).mean().item()

    # cells 12-13: tokens + title-stratified validation split
    text_arr = np.array(df_pubs["text"].astype(str).values, dtype=object)

    def tokenize(texts):
        ids = []
        for text in texts:
            row = tokenizer(str(text), truncation=True, max_length=MAX_TOKENS - 1, add_special_tokens=False)["input_ids"]
            if len(row) == 0 or row[-1] != EOS_ID:
                row.append(EOS_ID)
            ids.append(torch.tensor(row[:MAX_TOKENS], dtype=torch.long))
        return ids

    @torch.no_grad()
    def augment(idx, idx_t):
        """Latents and image semantics of the batch's maps, re-binarised at a random percentile, voxels dropped in half."""
        maps = maps_store[idx_t].to(device).float()
        lo, hi = MAP_AUG_PCT
        pct = torch.rand(len(idx), 1, device=device, generator=aug_gen) * (hi - lo) + lo
        k = ((1.0 - pct / 100.0) * maps.shape[1]).long().clamp_min(1)
        thr = maps.sort(dim=1, descending=True).values.gather(1, k - 1)
        maps = (maps >= thr).float()
        drop = (torch.rand(maps.shape, device=device, generator=aug_gen) < MAP_AUG_DROP) & \
               (torch.rand(len(idx), 1, device=device, generator=aug_gen) < MAP_AUG_DROP_P)
        raw = enc(maps.masked_fill(drop, 0.0)).float()
        sem = project_rows(F.normalize(proj_head_image(raw).float(), dim=1), levels[np.asarray(idx)])
        return raw.cpu(), sem

    def collate(idx, aug=False):
        idx_t = torch.as_tensor(idx, dtype=torch.long)
        input_ids = pad_sequence([all_ids[i] for i in idx], batch_first=True, padding_value=PAD_ID)
        lengths = torch.tensor([len(all_ids[i]) for i in idx], dtype=torch.long)
        attn_mask = (torch.arange(input_ids.size(1))[None, :] < lengths[:, None]).long()
        if aug:
            raw, sem_image = augment(idx, idx_t)
            return raw, train_semantic_text[idx_t], sem_image, input_ids, attn_mask
        return latent_images[idx_t], train_semantic_text[idx_t], train_semantic_images[idx_t], input_ids, attn_mask

    split_titles = np.array([str(x).split("\n", 1)[0] for x in text_arr], dtype=object)
    rng = np.random.default_rng(args.split_seed)  # independent of the training seed
    train_idx, val_idx = [], []
    df_split = pd.DataFrame({"idx": np.arange(len(text_arr), dtype=int), "title": split_titles})
    for _, group in df_split.groupby("title", sort=False, dropna=False):
        idx = group["idx"].to_numpy(dtype=int).copy()
        rng.shuffle(idx)
        if len(idx) <= 1:
            train_idx.extend(idx.tolist())
            continue
        n_val = min(max(1, int(round(len(idx) * VAL_FRAC))), len(idx) - 1)
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    train_idx, val_idx = np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int)
    if len(val_idx) == 0:
        all_idx = np.arange(len(text_arr), dtype=int)
        rng.shuffle(all_idx)
        n_val = min(max(1, int(round(len(text_arr) * VAL_FRAC))), len(text_arr) - 1)
        val_idx, train_idx = all_idx[:n_val], all_idx[n_val:]
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    all_ids = tokenize(text_arr)
    train_gen = torch.Generator().manual_seed(args.seed + 101)
    aug_gen = torch.Generator(device=device).manual_seed(args.seed + 202)  # --map-aug draws only from this generator
    NL_ID = tokenizer("\n", add_special_tokens=False)["input_ids"][0]
    log.info("filtered network dataset: train %d | val %d | title classes %d", len(train_idx), len(val_idx), pd.Series(split_titles).nunique())

    # cell 14: loss
    def match_lm_token_norm(vis):
        target_norm = torch.tensor(LM_EMB_NORM, device=vis.device, dtype=vis.dtype)
        vis_norm = vis.float().norm(dim=-1, keepdim=True).clamp_min(1e-6).to(vis.dtype)
        return vis * (target_norm / vis_norm)

    def lm_loss(raw, sem_text, sem_image, input_ids, attn_mask):
        raw, sem_text, sem_image, input_ids, attn_mask = (x.to(device, non_blocking=True) for x in (raw, sem_text, sem_image, input_ids, attn_mask))
        if qformer.training:
            use_image = torch.rand(sem_text.size(0), 1, device=sem_text.device) < SEMANTIC_IMAGE_MIX_P
            sem = torch.where(use_image, sem_image, sem_text)
        else:
            sem = sem_image
        batch_size = raw.size(0)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda" and args.amp == "bf16")):
            vis_tokens = match_lm_token_norm(qformer(raw.to(train_dtype), sem.to(train_dtype))).to(model_dtype)
            num_q = vis_tokens.size(1)
            with torch.no_grad():
                text_embeds = model.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([vis_tokens, text_embeds.to(model_dtype)], dim=1)
            full_mask = torch.cat([torch.ones(batch_size, num_q, device=device, dtype=torch.long), attn_mask], dim=1)
            labels = torch.cat([torch.full((batch_size, num_q), -100, dtype=torch.long, device=device), input_ids.masked_fill(attn_mask == 0, -100)], dim=1)
            w_text = attn_mask.float()
            if qformer.training and args.title_weight != 1.0:
                nl = (input_ids == NL_ID).long()
                pos = torch.arange(input_ids.size(1), device=device)[None, :]
                title = ((nl.cumsum(1) - nl) == 0) & (pos >= TITLE_START) & (attn_mask == 1)
                w_text = torch.where(title, w_text * args.title_weight, w_text)
            loss_weights = torch.cat([torch.ones(batch_size, num_q, device=device, dtype=torch.float32), w_text], dim=1)
            logits = model(inputs_embeds=inputs_embeds, attention_mask=full_mask, use_cache=False).logits
            shift_logits = logits[:, :-1].float().contiguous()
            shift_labels = labels[:, 1:].contiguous()
            shift_weights = loss_weights[:, 1:].contiguous()
            token_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100, reduction="none").view_as(shift_labels)
            valid = shift_labels.ne(-100)
            return (token_loss * shift_weights * valid).sum() / (shift_weights * valid).sum().clamp_min(1.0)

    def batches(indices, batch_size, shuffle, gen=None):
        idx = torch.as_tensor(indices)
        if shuffle:
            idx = idx[torch.randperm(len(idx), generator=gen)]
        return [idx[s:s + batch_size].tolist() for s in range(0, len(idx), batch_size)]

    @torch.no_grad()
    def mean_lm_loss(indices, max_batches=None):
        qformer.eval()
        model.eval()
        total = torch.zeros((), device=device)
        n = 0
        for b in batches(indices, EVAL_BATCH, shuffle=False):
            total += lm_loss(*collate(b))
            n += 1
            if max_batches is not None and n >= max_batches:
                break
        return float(total) / max(n, 1)

    # cell 15: training loop (the final epoch is saved, as in the notebook)
    seed_all(args.seed)
    optimizer = torch.optim.AdamW(qformer.parameters(), lr=LR, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    history = {"train": [], "val": [], "lr": [], "epoch_seconds": []}
    smoke_cap = args.caps.max_batches if args.smoke else None
    step_times = []
    with timed("train", log, out=args.out, epochs=num_epochs, batch_size=train_batch, amp=args.amp, grad_ckpt=args.grad_ckpt, **meta) as t_train:
        for epoch in range(num_epochs):
            t_epoch = time.perf_counter()
            qformer.train()
            model.eval()
            total = torch.zeros((), device=device)
            epoch_batches = batches(train_idx, train_batch, shuffle=True, gen=train_gen)
            if smoke_cap:
                epoch_batches = epoch_batches[:smoke_cap]
            for bi, b in enumerate(epoch_batches):
                t0 = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                loss = lm_loss(*collate(b, aug=args.map_aug))
                loss.backward()
                optimizer.step()
                total += loss.detach()
                if bi < 3:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    step_times.append(time.perf_counter() - t0)
            train_loss = float(total) / max(len(epoch_batches), 1)
            val_loss = mean_lm_loss(val_idx, max_batches=smoke_cap) if len(val_idx) else float("nan")
            scheduler.step()
            history["train"].append(train_loss)
            history["val"].append(val_loss)
            history["lr"].append(scheduler.get_last_lr()[0])
            history["epoch_seconds"].append(time.perf_counter() - t_epoch)
            log.info("Epoch %3d | train %.4f | val %.4f | lr %.2e | %.1fs", epoch + 1, train_loss, val_loss, history["lr"][-1], history["epoch_seconds"][-1])
            write_json(run_dir / "history_14_qformer_anat.json", history)
    torch.save(qformer.state_dict(), canonical_path)
    torch.save(canonical_semantic_banks, banks_path)  # cell 22
    log.info("saved %s and %s", canonical_path, banks_path)

    # package like neurovlm/NeuroQformer (fp32 weights + frozen image head + canonical banks)
    q_fp32 = build_projector(projector, **{k: v for k, v in projector_cfg.items() if k not in ("class", "params")})
    q_fp32.load_state_dict({k: v.detach().float().cpu() for k, v in qformer.state_dict().items()}, strict=True)
    packaged = NeuroQFormer(qformer=q_fp32, proj_head_image=ProjHead(384, 384, 384), canonical_banks=canonical_semantic_banks,
                            projection_temp=CANONICAL_PROJ_TEMP, canonical_basis="all", use_canonical_projection=True)
    packaged.proj_head_image.load_state_dict(proj_head_image.cpu().state_dict())
    write_projector_package(pkg_dir, packaged, {
        "architectures": ["neurovlm.qformer.NeuroQFormer"], "library_name": "neurovlm", "model_type": "neurovlm_qformer",
        "qformer": {**package_qformer_block(packaged), "projection_temp": CANONICAL_PROJ_TEMP, "use_canonical_projection": True,
                    "canonical_basis": "all", "raw_image_input_mode": "raw"},
        "canonical_metadata": {"class_counts": {k: int(len(v)) for k, v in canonical_semantic_banks.items() if k != "all"},
                               "qformer_source": str(canonical_path), "grounded_init": grounded_source, "adapter": adapter_source,
                               "seed": args.seed, "split_seed": args.split_seed, "map_calibration": args.map_calibration,
                               "synth_binarize_pct": args.synth_binarize_pct, "train_semantic_bank": args.train_bank,
                               "map_aug": args.map_aug, "title_weight": args.title_weight},
        "weights": "model.safetensors",
    })
    log.info("packaged %s (%s projector, final epoch %d)", pkg_dir, projector, num_epochs)

    # cells 18-21: greedy captions of reference networks, written to generations_refs.jsonl
    generations = []
    if not args.no_generate:
        with timed("generate_refs", log, out=args.out, **meta):
            import nibabel as nib
            from nilearn.image import resample_to_img
            from neurovlm.data import load_dataset, load_masker

            imgs = load_dataset("networks")
            masker = load_masker()
            proj_head_image = proj_head_image.to(device)
            qformer.eval()
            model.eval()
            cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
            for atlas_name, map_name in (REFS[:1] if args.smoke else REFS):
                arr = imgs[atlas_name][map_name]["array"]
                im = nib.Nifti1Image(arr / arr.max(), imgs[atlas_name][map_name]["affine"])
                flat = torch.from_numpy(masker.transform(resample_to_img(im, masker.mask_img, interpolation="nearest"))).float().to(device)
                with torch.no_grad():
                    raw_latent = enc(flat).float().reshape(1, -1)
                    image_sem = F.normalize(proj_head_image(raw_latent).float(), dim=1)
                for seed, basis in enumerate(["network", "region", "function"]):
                    sem_in = project_to_canonical(canonical_semantic_banks[basis], image_sem)
                    prefix_text = f"[{basis.upper()}]"
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        vis = match_lm_token_norm(qformer(raw_latent.to(device, train_dtype), sem_in.to(device, train_dtype))).to(model_dtype)
                        prefix_ids = tokenizer(prefix_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
                        inputs_embeds = torch.cat([vis, model.get_input_embeddings()(prefix_ids).to(model_dtype)], dim=1)
                        attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
                        with torch.random.fork_rng(devices=cuda_devices):
                            torch.manual_seed(seed)
                            if torch.cuda.is_available():
                                torch.cuda.manual_seed_all(seed)
                            out_ids = model.generate(inputs_embeds=inputs_embeds, attention_mask=attn, max_new_tokens=16 if args.smoke else 256,
                                                     num_beams=args.num_beams, do_sample=False, repetition_penalty=1.18, no_repeat_ngram_size=4,
                                                     eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id, top_k=None, temperature=None, top_p=None)
                    pred = prefix_text + tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
                    generations.append({"atlas": atlas_name, "network": map_name, "basis": basis, "caption": pred})
            with (run_dir / "generations_refs.jsonl").open("w") as fh:
                for row in generations:
                    fh.write(json.dumps(row) + "\n")
            for row in generations[:6]:
                log.info("%s/%s [%s]: %s", row["atlas"], row["network"], row["basis"], row["caption"][:160])

    write_stage_summary(run_dir, STAGE, {
        "epochs": num_epochs, "seed": args.seed, "split_seed": args.split_seed,
        "val_loss_final": history["val"][-1], "val_loss_min": float(np.nanmin(history["val"])),
        "history": history, "seconds": t_train.seconds, "sec_per_epoch": t_train.seconds / num_epochs,
        "sec_per_step": float(np.median(step_times)) if step_times else None, "steps_per_epoch": int(np.ceil(len(train_idx) / train_batch)),
        "n_rows": {"train": int(len(train_idx)), "val": int(len(val_idx)), "total": int(len(df_pubs))},
        "bank_sizes": {k: int(len(v)) for k, v in canonical_semantic_banks.items()}, "grounded_init": grounded_source,
        "adapter": adapter_source, "autoencoder": ae_source, "proj_head_text_infonce": th_source, "proj_head_image_infonce": ih_source,
        "text_synth_less": str(tsl_path), "map_calibration": args.map_calibration, "synth_binarize_pct": args.synth_binarize_pct,
        "train_semantic_bank": args.train_bank, "map_aug": args.map_aug, "title_weight": args.title_weight, "amp": args.amp, "deterministic": args.deterministic, "notebook_precision": args.notebook_precision,
        "grad_ckpt": args.grad_ckpt,
        "device": str(device), "n_generations": len(generations), "projector": projector, "projector_config": projector_cfg})
    if args.install:
        install_artifacts({canonical_path.name: canonical_path, banks_path.name: banks_path}, log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
