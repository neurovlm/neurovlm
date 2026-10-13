#!/usr/bin/env python
"""Train the anatomical head (InterleavedDecoderAdapter, the released NeuroAdapter: text 768 -> brain logits 28,542) on
the synthetic anatomy pairs of stage 11 (docs/03_models/12_neuroadapter.ipynb).

    .venv/bin/python scripts/models/12_neuroadapter.py --run-name <run> [--smoke]

--recipe anchored (default) adds head v2's loss on PubMed training pairs and keeps the epoch with the best PubMed
validation recall AUC (scripts/README.md, "Anatomical head v2"); --recipe notebook is the notebook recipe.
Writes adapter_anat_full.{pt,safetensors} and adapter_anat_full.lineage.json (+ the cell-10 exports with --derivatives).
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/models/
from _common_models import *  # noqa: E402,F403
from common import apply_pubmed_exclusions  # noqa: E402

STAGE = "12_neuroadapter"
log = get_logger(STAGE)

SEED = 123
N_EPOCHS, LR, TRAIN_BATCH, VAL_BATCH, EVAL_EVERY = 2000, 2e-4, 2048, 2048, 20
SPECTER_BATCH = 512


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_state_dict(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def resolve_base_models(args, run_dir, device):
    """Autoencoder and proj_head_text_mse from a run directory (08/09), else the released models."""
    from neurovlm.models import ProjHead
    from neurovlm.models.base import NeuroAutoEncoder

    ae_path = head_path = None
    if args.base != "released":
        # auto: a run directory, else the release (as the notebook); data_dir copies are never used
        ae_path = run_input("autoencoder.pt", run_dir=run_dir, args=args)
        head_path = run_input("proj_head_text_mse.pt", run_dir=run_dir, args=args)
        if args.base == "run" and (ae_path is None or head_path is None):
            raise FileNotFoundError("--base run requires autoencoder.pt and proj_head_text_mse.pt in the run directory")
    if ae_path is not None and head_path is not None:
        autoencoder = load_module_or_state(ae_path, builder=lambda: NeuroAutoEncoder(seed=0, out="logit"), device=device)
        proj_head = load_module_or_state(head_path, builder=lambda: ProjHead(768, 512, 384), device=device)
        source = {"autoencoder": str(ae_path), "proj_head_text_mse": str(head_path)}
    else:
        autoencoder = released_model("autoencoder", device)
        proj_head = released_model("proj_head_text_mse", device)
        source = {"autoencoder": f"released:neurovlm/NeuroAutoEncoder@{hf_pinned('neurovlm/NeuroAutoEncoder')[:8]}",
                  "proj_head_text_mse": f"released:neurovlm/ProjectionHeads@{hf_pinned('neurovlm/ProjectionHeads')[:8]}"}
    log.info("base models: %s", source)
    return autoencoder.eval(), proj_head.eval(), source


ANCHOR_LOSS = ("latent MSE + 1.0 x (1 - Pearson r of decoder logits vs decoded true latent) + 0.3 x (1 - r of sigmoid maps) on PubMed "
               "training pairs, input noise 0.5 N(0,I)/sqrt(768) (= the text-to-brain head v2 loss of 09)")


def load_anchor_data(args, run_dir, device) -> dict | None:
    """PubMed training pairs for the anchor loss and validation texts for model selection, on device.

    Text: SPECTER2 latents of "name [SEP] description" (latent_specter2_adhoc.pt, what the validator encodes); targets:
    the run's autoencoder latents; split: pmids_split.pt; plus the InfoNCE heads for the validator's recall AUC.
    """
    import torch
    import torch.nn.functional as F
    from neurovlm.models import ProjHead

    # run outputs (08, 09), else the pinned release; never data_dir
    split, split_source, _ = pmid_split(run_dir=run_dir, args=args, logger=log)
    lat_path = run_input("latent_neuro.pt", run_dir=run_dir, args=args) or hf_pinned("neurovlm/embedded_text", "latent_neuro.pt")
    sp_path = hf_pinned("neurovlm/embedded_text", "latent_specter2_adhoc.pt")
    lat, sp = load_torch(lat_path), load_torch(sp_path)
    sp_pos = {int(p): i for i, p in enumerate(np.asarray(sp["pmid"]))}
    lat_pmid = np.asarray(lat["pmid"]).astype(np.int64)
    lat_set = set(lat_pmid.tolist())
    train = set(np.asarray(split["train"]).astype(np.int64).tolist())
    rows = [(i, sp_pos[int(p)]) for i, p in enumerate(lat_pmid) if int(p) in train and int(p) in sp_pos]
    z = torch.as_tensor(lat["latent"])[torch.tensor([r[0] for r in rows])].float()
    x = F.normalize(torch.as_tensor(sp["latent"])[torch.tensor([r[1] for r in rows])].float(), dim=1)
    val = np.sort(np.asarray(split["val"]).astype(np.int64))
    val = val[apply_pubmed_exclusions(val, enabled=True, logger=log, label="PubMed val (stage-12 selection)")]
    val = np.array([p for p in val if int(p) in sp_pos and int(p) in lat_set])
    if len(z) == 0 or len(val) < 2:
        log.warning("anchored recipe: %d PubMed training pairs / %d validation texts available -> notebook recipe instead", len(z), len(val))
        return None
    val_text = F.normalize(torch.as_tensor(sp["latent"])[torch.tensor([sp_pos[int(p)] for p in val])].float(), dim=1)
    rr = dict(run_dir=run_dir, args=args, device=device, logger=log)
    th, _ = run_or_released("proj_head_text_infonce", "proj_head_text_infonce.pt", builder=lambda: ProjHead(768, 512, 384), **rr)
    ih, _ = run_or_released("proj_head_image_infonce", "proj_head_image_infonce.pt", builder=lambda: ProjHead(384, 384, 384), **rr)
    with torch.no_grad():
        val_z_text = F.normalize(th(val_text.to(device)).float(), dim=1)
    log.info("anchor: %d PubMed training pairs; selection: %d PubMed validation texts", len(z), len(val))
    return {"z": z.to(device), "x": x.to(device), "val_text": val_text.to(device), "val_z_text": val_z_text, "image_head": ih.eval(),
            "sources": {"split": split_source, "latents": str(lat_path), "specter": str(sp_path)}}


def _pearson_rows(a, b):
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    return ((a * b).sum(dim=1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-8)).mean()


def anchor_loss(adapter, anchor, batch, device, amp):
    """Head v2's loss on a random batch of PubMed training pairs, through the adapter's projection head and decoder."""
    import torch
    import torch.nn.functional as F

    idx = torch.randint(0, len(anchor["z"]), (batch,), device=device)
    x = anchor["x"][idx]
    x = F.normalize(x + 0.5 * torch.randn_like(x) / math.sqrt(x.shape[1]), dim=1)
    z_true = anchor["z"][idx]
    with autocast_context(device, amp):
        z = adapter.proj_head(x)
        logits = adapter.decode_logits(z).float()
    with torch.no_grad():  # the frozen decoder on the true latent
        target = adapter.dec2(adapter.dec_act1(adapter.dec1(adapter.dec_act0(adapter.dec0(z_true))))).float()
    return (F.mse_loss(z.float(), z_true) + 1.0 * (1 - _pearson_rows(logits, target))
            + 0.3 * (1 - _pearson_rows(torch.sigmoid(logits), torch.sigmoid(target))))


def pubmed_val_recall_auc(adapter, anchor, device) -> float:
    """Adapter recall AUC on the PubMed validation split, each map divided by its own maximum (the selection rule of
    runs/final; the global factor is fitted on the selected adapter)."""
    import torch
    import torch.nn.functional as F

    was_training = adapter.training
    adapter.eval()
    with torch.no_grad():
        z_img = []
        for s in range(0, len(anchor["val_text"]), 512):
            maps = calibrate_maps(torch.sigmoid(adapter(anchor["val_text"][s:s + 512])).float(), "max")
            z_img.append(F.normalize(anchor["image_head"](anchor["encoder"](maps)).float(), dim=1))
        z_img = torch.cat(z_img)
        scores = z_img @ anchor["val_z_text"].T
        n = len(scores)
        ranks = 1 + (scores > scores.diag()[:, None]).sum(dim=1)
        auc = float((torch.cumsum(torch.bincount(ranks - 1, minlength=n).float(), 0) / n).mean())
    adapter.train(was_training)
    return auc


def main(argv=None):
    p = base_parser("03_models/12: train the NeuroAdapter on synthetic anatomical (text, map) pairs.", stage=STAGE)
    add_model_args(p, amp_default="fp32")
    p.add_argument("--images-synth", type=Path, default=None, help="images_synth.pt (default: run dir, then data_dir).")
    p.add_argument("--synth-table", type=Path, default=None,
                   help="Synthetic text table (default: synth_text.parquet of the run or data_dir, else the pinned HF synth.parquet).")
    p.add_argument("--base", choices=["auto", "run", "released"], default="auto", help="Source of the frozen autoencoder and proj_head_text_mse.")
    p.add_argument("--derivatives", action="store_true", help="Also write the notebook's cell-10 exports.")
    p.add_argument("--recipe", choices=["anchored", "notebook"], default="anchored",
                   help="anchored: + head v2 loss on PubMed pairs, best PubMed validation recall AUC; notebook: synthetic BCE only.")
    p.add_argument("--anchor-weight", type=float, default=1.0, help="Weight of the PubMed anchor loss.")
    p.add_argument("--anchor-batch", type=int, default=512, help="PubMed pairs per step in the anchor loss.")
    args = finalize_args(p.parse_args(argv))
    run_dir = resolve_run_dir(args)
    setup_torch(args.seed, args.deterministic)
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from torch.optim import AdamW
    from neurovlm.models.adapter import InterleavedDecoderAdapter

    device = select_device(args.device)
    if device.type != "cuda":
        log.warning("CUDA is unavailable; the 2,000-epoch recipe will be very slow on CPU.")
    n_epochs = args.epochs or N_EPOCHS
    train_bs = args.batch_size or TRAIN_BATCH
    meta = dict(stage=STAGE, smoke=args.smoke, run=args.run_name)
    adapter_path = run_dir / "adapter_anat_full.pt"
    lineage_path = run_dir / "adapter_anat_full.lineage.json"
    installed = {"adapter_anat_full.pt": adapter_path, "adapter_anat_full.lineage.json": lineage_path}
    if artifacts_exist(adapter_path, lineage_path, force=args.force):
        log.info("artifacts exist in %s; skipping (use --force to retrain)", run_dir)
        if args.install:
            install_artifacts(installed, log)
        return 0

    # cell 3: synthetic pairs
    with timed("load_inputs", log, out=args.out, **meta):
        images_path = find_input("images_synth.pt", run_dir=run_dir, args=args, explicit=args.images_synth, stage_hint="11_anatomical_atlases")
        table_path = args.synth_table or find_input("synth_text.parquet", run_dir=run_dir, args=args, required=False)
        if table_path is None:  # the released pairs: pinned synth.parquet, aligned with the release-era images_synth.pt
            table_path = hf_pinned("neurovlm/synthetic_pairs", "synth.parquet")
        images_gen = load_torch(images_path, weights_only=True).float()
        df_synth = pd.read_parquet(table_path)
        text = df_synth["text"].str.replace(" [sep] ", " [SEP] ", regex=False).tolist()
        if len(images_gen) != len(text):
            raise ValueError(f"unaligned inputs: {len(images_gen)} images != {len(text)} texts")
        if args.max_samples and args.max_samples < len(text):
            images_gen, text, df_synth = images_gen[: args.max_samples], text[: args.max_samples], df_synth.iloc[: args.max_samples]
        log.info("synthetic pairs: %s from %s / %s", tuple(images_gen.shape), images_path, table_path)

    # cell 4: unit-norm SPECTER embeddings
    with timed("specter_encode", log, out=args.out, n=len(text), **meta):
        specter = released_model("specter", device)
        specter.specter.eval()
        text_emb = torch.empty((len(text), 768), dtype=torch.float32)
        for start in range(0, len(text), SPECTER_BATCH):
            stop = min(start + SPECTER_BATCH, len(text))
            with torch.inference_mode():
                text_emb[start:stop] = F.normalize(specter(text[start:stop]), dim=1).detach().float().cpu()
        del specter
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # cell 5: random 99 % / 1 % train/validation split
    torch.manual_seed(SEED)
    inds = torch.randperm(len(images_gen))
    n_train_rows = int(len(inds) * 0.99)
    images_train = images_gen[inds[:n_train_rows]].to(device)
    images_val = images_gen[inds[n_train_rows:]].to(device)
    text_train_emb = text_emb[inds[:n_train_rows]].to(device)
    text_val_emb = text_emb[inds[n_train_rows:]].to(device)

    # cell 7: model, optimiser, training loop
    autoencoder, proj_head, base_source = resolve_base_models(args, run_dir, device)
    adapter = InterleavedDecoderAdapter(autoencoder, proj_head, hidden_dim=1024, freeze_pretrained_decoder=True,
                                        freeze_proj_head=False).to(device)
    anchor = load_anchor_data(args, run_dir, device) if args.recipe == "anchored" else None
    if anchor is not None:
        anchor["encoder"] = autoencoder.encoder.eval()  # frozen run encoder (the adapter holds only the decoder)
    recipe_used = "anchored" if anchor is not None else "notebook"
    lineage = {
        "producer": "scripts/models/12_neuroadapter.py (port of docs/03_models/12_neuroadapter.ipynb)",
        "inputs": {str(images_path): sha256_file(images_path), str(table_path): sha256_file(table_path)},
        "base_models": {**base_source, "autoencoder_state_sha256": sha256_state_dict(autoencoder),
                        "proj_head_text_mse_state_sha256": sha256_state_dict(proj_head)},
        "training": {"seed": SEED, "split": "torch.randperm; first 99% train, final 1% validation", "hidden_dim": 1024,
                     "epochs": n_epochs, "learning_rate": LR, "train_batch_size": train_bs, "validation_batch_size": VAL_BATCH,
                     "evaluate_every": EVAL_EVERY, "optimizer": "AdamW", "weight_decay": 0.0,
                     "loss": "binary_cross_entropy_with_logits", "amp": args.amp, "recipe": recipe_used,
                     **({"anchor": {"weight": args.anchor_weight, "batch": args.anchor_batch, "loss": ANCHOR_LOSS,
                                    "n_train_pairs": int(len(anchor["z"])), "n_val": int(len(anchor["val_text"])),
                                    "selection": "best PubMed-validation recall AUC (calibrated maps), every 20 epochs",
                                    "sources": anchor["sources"]}} if anchor else {})},
    }
    opt = AdamW(adapter.trainable_parameters(), lr=LR, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4, threshold=1e-5,
                                                       threshold_mode="abs", cooldown=0, min_lr=1e-5)
    best_val_loss, best_epoch, best_state, curve = math.inf, -1, None, []
    best_score, best_val_auc = math.inf, float("nan")
    n_train, n_val = images_train.shape[0], images_val.shape[0]

    with timed("train", log, out=args.out, epochs=n_epochs, batch_size=train_bs, amp=args.amp, n_train=n_train, **meta) as t:
        for epoch in range(n_epochs):
            adapter.train()
            order = torch.randperm(n_train, device=images_train.device)
            total_loss = torch.zeros((), device=device)
            for start in range(0, n_train, train_bs):
                batch_idx = order[start:start + train_bs]
                x = images_train[batch_idx]
                y = text_train_emb[batch_idx]
                opt.zero_grad(set_to_none=True)
                with autocast_context(device, args.amp):
                    logits = adapter(y)
                loss = F.binary_cross_entropy_with_logits(logits.float(), x)
                if anchor is not None:
                    loss = loss + args.anchor_weight * anchor_loss(adapter, anchor, args.anchor_batch, device, args.amp)
                loss.backward()
                opt.step()
                total_loss += loss.detach() * x.shape[0]
            if epoch % EVAL_EVERY == 0 or epoch == n_epochs - 1:
                train_loss = float(total_loss) / n_train
                adapter.eval()
                val_total = torch.zeros((), device=device)
                with torch.inference_mode():
                    for start in range(0, n_val, VAL_BATCH):
                        x_val = images_val[start:start + VAL_BATCH]
                        val_total += F.binary_cross_entropy_with_logits(adapter(text_val_emb[start:start + VAL_BATCH]), x_val) * x_val.shape[0]
                val_loss = float(val_total) / n_val
                val_auc = pubmed_val_recall_auc(adapter, anchor, device) if anchor is not None else float("nan")
                score = -val_auc if anchor is not None else val_loss  # lower is better
                if score < best_score:
                    best_score, best_val_loss, best_val_auc, best_epoch = score, val_loss, val_auc, epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
                sched.step(val_loss)  # the synthetic validation BCE drives the schedule in both recipes
                curve.append([epoch, train_loss, val_loss, opt.param_groups[0]["lr"], val_auc])
                log.info("epoch %d: train=%.6f val=%.6f pubmed_val_recall_auc=%.4f best@%d lr=%.2e", epoch, train_loss, val_loss, val_auc,
                         best_epoch, opt.param_groups[0]["lr"])

    # cell 8: save the best state + lineage
    if best_state is None:
        raise RuntimeError("Training completed without a validation checkpoint.")
    adapter.load_state_dict(best_state)
    adapter.eval()
    save_module_and_state(adapter, run_dir / "adapter_anat_full", logger=log)
    adapter.to(device)
    lineage["result"] = {"best_epoch": best_epoch, "best_validation_loss": best_val_loss, "best_pubmed_val_recall_auc": best_val_auc,
                         "adapter_state_sha256": sha256_state_dict(adapter), "checkpoint": str(adapter_path),
                         "seconds": t.seconds, "sec_per_epoch": t.seconds / max(n_epochs, 1)}
    lineage_path.write_text(json.dumps(lineage, indent=2) + "\n")
    write_stage_summary(run_dir, STAGE, {"best_epoch": best_epoch, "best_val_loss": best_val_loss, "best_pubmed_val_recall_auc": best_val_auc,
                                         "recipe": recipe_used, "selection": ("best PubMed-validation recall AUC (calibrated maps)" if anchor is not None
                                                                              else "best synthetic-pair validation BCE"), "epochs": n_epochs,
                                         "seconds": t.seconds, "sec_per_epoch": t.seconds / max(n_epochs, 1),
                                         "curve_cols": ["epoch", "train_loss", "val_loss", "lr", "pubmed_val_recall_auc"], "curve": curve,
                                         "amp": args.amp, "device": str(device), "n_train": n_train, "n_val": n_val,
                                         "base_models": base_source})

    # cell 10: derivative exports
    if args.derivatives:
        with timed("derivatives", log, out=args.out, **meta), torch.inference_mode():
            image_gen_f = torch.cat([torch.sigmoid(adapter(text_emb[s:s + 2048].to(device))).cpu() for s in range(0, len(text_emb), 2048)])
            latent_image_synth = encode_in_batches(autoencoder.encoder, image_gen_f, batch_size=2048, device=device)
            torch.save(image_gen_f, run_dir / "images_synth_trained_full.pt")
            df_synth.to_parquet(run_dir / "text_synth_full.parquet")
            torch.save(text_emb, run_dir / "latent_text_synth_full.pt")
            torch.save(latent_image_synth, run_dir / "latent_image_synth_full.pt")
            torch.save(autoencoder.cpu(), run_dir / "autoencoder_full.pt")
            torch.save(proj_head.cpu(), run_dir / "proj_head_full.pt")
    if args.install:
        install_artifacts(installed, log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
