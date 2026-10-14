#!/usr/bin/env python
"""Train the text-to-brain head and the InfoNCE heads (docs/03_models/09_projection_head.ipynb).

    .venv/bin/python scripts/models/09_projection_head.py --run-name <run> [--smoke]

Stage A is the notebook's BCE head (saved as proj_head_text_bce; it initialises the InfoNCE text head). The text-to-brain
head proj_head_text_mse is head v2: the same module trained on latent MSE + decoder-space Pearson terms and kept at its
best validation Dice (scripts/models/README.md, "Text-to-brain head v2"). --text-head v1 saves stage A as proj_head_text_mse.
"""
from __future__ import annotations

import copy
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/models/
from _common_models import *  # noqa: E402,F403
from common import apply_pubmed_exclusions  # noqa: E402

STAGE = "09_projection_head"
log = get_logger(STAGE)

MSE_EPOCHS, MSE_BATCH, MSE_LR = 201, 1024, 5e-5  # stage A, notebook cell 4
NCE_EPOCHS, NCE_BATCH, NCE_LR, NCE_TEMP = 301, 2048, 1e-5, 0.07  # InfoNCE, notebook cell 8
VAL_INTERVAL = 20
HEAD_SEED = 123
# head v2 (tuned on the validation split only)
T2B_EPOCHS, T2B_BATCH, T2B_LR, T2B_WD = 80, 512, 1e-3, 0.01
T2B_NOISE, T2B_W_LOGIT, T2B_W_PROB = 0.5, 1.0, 0.3  # input noise; weights of the logit and probability Pearson terms
T2B_DICE_PCT, T2B_TOP_PCT = 70, 90  # paper Dice vs the decoded target; top-10 % Dice vs the raw map (logged only)
RIDGE_ALPHAS = (0.01, 0.1, 0.3, 1.0, 3.0, 10.0)


def load_inputs(args, run_dir):
    """Notebook cell 2: autoencoder, maps, latents, split and SPECTER2 embeddings, aligned by pmid (CPU)."""
    import pandas as pd
    import torch
    from neurovlm.models.base import NeuroAutoEncoder

    # the decoder (stage A) and the latents (head v2, InfoNCE) must come from the same autoencoder: the run's (08), else the
    # pinned release; data_dir copies are never used
    ae_path = run_input("autoencoder.pt", run_dir=run_dir, args=args)
    if ae_path is not None:
        latent_path = ae_path.parent / "latent_neuro.pt"
        if not latent_path.exists():
            raise FileNotFoundError(f"latent_neuro.pt (08) must sit next to {ae_path}")
        autoencoder = load_module_or_state(ae_path, builder=lambda: NeuroAutoEncoder(seed=0, out="logit"))
        log.info("autoencoder: %s", ae_path)
    else:
        autoencoder = released_model("autoencoder")
        latent_path = hf_pinned("neurovlm/embedded_text", "latent_neuro.pt")
        log.warning("no autoencoder.pt in the run; using the released autoencoder (neurovlm/NeuroAutoEncoder@%s) and its latents",
                    hf_pinned("neurovlm/NeuroAutoEncoder")[:8])
    payload = load_torch(latent_path)
    latent_neuro, pmids_latent = payload["latent"].float(), np.asarray(payload["pmid"])
    log.info("latent_neuro: %s %s", latent_path, tuple(latent_neuro.shape))

    nv_path = args.neuro_vectors or hf_pinned("neurovlm/embedded_text", "pubmed_images.pt")
    payload = load_torch(nv_path)
    neuro_vectors, pmids = payload["neuro_vectors"].float(), np.asarray(payload["pmid"])
    log.info("neuro_vectors: %s %s", nv_path, tuple(neuro_vectors.shape))

    split, _, _ = pmid_split(run_dir=run_dir, args=args, logger=log)
    ids_train, ids_test, ids_val = split["train"], split["test"], split["val"]

    sp_path = hf_pinned("neurovlm/embedded_text", "latent_specter2_adhoc.pt")
    payload = load_torch(sp_path)
    latent_text, pmids_specter = payload["latent"].float(), np.asarray(payload["pmid"])

    # keep the rows of latent_neuro (studies with <= 100 coordinates)
    mask = pd.Series(pmids).isin(pmids_latent).to_numpy()
    neuro_vectors, pmids = neuro_vectors[torch.from_numpy(mask)], pmids[mask]
    assert (pmids == pmids_latent).all(), "neuro_vectors/latent_neuro pmid mismatch"
    mask = pd.Series(pmids_specter).isin(pmids_latent).to_numpy()
    latent_text, pmids_specter = latent_text[torch.from_numpy(mask)], pmids_specter[mask]
    assert (pmids == pmids_specter).all(), "specter/latent_neuro pmid mismatch"

    pm = pd.Series(pmids)
    masks = {"train": pm.isin(ids_train).to_numpy(), "val": pm.isin(ids_val).to_numpy(), "test": pm.isin(ids_test).to_numpy()}
    if args.max_samples and args.max_samples < len(pmids):  # smoke: a proportional prefix of each split
        keep = np.zeros(len(pmids), dtype=bool)
        for m in masks.values():
            keep[np.flatnonzero(m)[:max(2, int(round(args.max_samples * m.mean())))]] = True
        masks = {k: v & keep for k, v in masks.items()}
    log.info("rows train/val/test = %d/%d/%d", masks["train"].sum(), masks["val"].sum(), masks["test"].sum())
    return dict(autoencoder=autoencoder, neuro_vectors=neuro_vectors, latent_neuro=latent_neuro,
                latent_text=latent_text, pmids=pmids, masks=masks, ae_source=str(ae_path or "released"))


def train_text_mse(inputs, device, *, epochs, batch_size, amp, logger, out, meta):
    """Stage A (notebook cell 4): BCE through the frozen decoder, best validation state."""
    import torch
    from torch import nn
    from torch.optim import AdamW
    from neurovlm.models import ProjHead

    m = inputs["masks"]
    splits = {
        "train": {"X": inputs["latent_text"][m["train"]], "y": inputs["neuro_vectors"][m["train"]]},
        "val": {"X": inputs["latent_text"][m["val"]], "y": inputs["neuro_vectors"][m["val"]]},
    }
    placed, on_device = place_splits(splits, device, logger=logger)
    X_train, y_train = placed["train"]["X"].to(device), placed["train"]["y"]
    X_val, y_val = placed["val"]["X"].to(device), placed["val"]["y"]

    head = ProjHead(seed=HEAD_SEED, latent_in_dim=768, hidden_dim=512, latent_out_dim=384)
    decoder = copy.deepcopy(inputs["autoencoder"].decoder).requires_grad_(False)
    model = nn.Sequential(head.to(device), decoder.to(device))
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=MSE_LR)
    best_val, best_state, curve = float("inf"), None, []
    n = len(X_train)
    gen = torch.Generator(device="cpu")

    def val_loss():
        model.eval()
        total = torch.zeros((), device=device)
        with torch.no_grad():
            for s in range(0, len(X_val), 2048):
                yb = y_val[s:s + 2048].to(device, non_blocking=True)
                total += loss_fn(model(X_val[s:s + 2048]), yb) * len(yb)
        return float(total / len(X_val))

    with timed("train_text_mse", logger, out=out, epochs=epochs, batch_size=batch_size, amp=amp, on_device=on_device, **meta) as t:
        for epoch in range(epochs):
            model.train()
            gen.manual_seed(epoch)  # notebook: torch.manual_seed(epoch); randperm(len(X_train))
            perm = torch.randperm(n, generator=gen).to(device)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                yb = y_train[idx] if on_device else y_train[idx.cpu()].to(device, non_blocking=True)
                with autocast_context(device, amp):
                    prediction = model(X_train[idx])
                loss = loss_fn(prediction.float(), yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            if epoch % VAL_INTERVAL == 0 or epoch == epochs - 1:
                vl = val_loss()
                curve.append([epoch, vl])
                logger.info("text_mse epoch %d val loss %.5g", epoch, vl)
                if vl < best_val:
                    best_val = vl
                    best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    if best_state is not None:
        head.load_state_dict(best_state)
    return head.cpu().eval(), {"best_val_loss": best_val, "val_loss_curve": curve, "seconds": t.seconds,
                               "sec_per_epoch": t.seconds / max(epochs, 1), "epochs": epochs, "on_device": on_device}


def _pct_threshold(x, q):
    """Row-wise np.percentile(x, q, axis=1) (linear interpolation)."""
    n = x.shape[1]
    v, _ = x.sort(dim=1)
    pos = q / 100.0 * (n - 1)
    lo, hi = int(np.floor(pos)), int(np.ceil(pos))
    return v[:, lo] + (v[:, hi] - v[:, lo]) * (pos - lo)


def _dice_pct(pred, target, q):
    """Per-row Dice of maps binarised at their own q-th percentile (neurovlm.metrics.text_to_brain.dice_percentile)."""
    import torch

    pb = pred > _pct_threshold(pred, q)[:, None]
    tb = target > _pct_threshold(target, q)[:, None]
    den = (pb.sum(1) + tb.sum(1)).double()
    return torch.where(den > 0, 2 * (pb & tb).sum(1).double() / den.clamp_min(1), torch.ones_like(den))


def _zrows(x):
    x = x - x.mean(1, keepdim=True)
    return x / x.std(1, keepdim=True, unbiased=False).clamp_min(1e-6)


def text_to_brain_metrics(decoder, pred_latent, true_latent, raw_maps=None, *, batch_size=512) -> dict:
    """Paper Dice (70th-percentile masks) and voxel Pearson r of sigmoid(decoder(pred)) vs sigmoid(decoder(true));
    top-10 % Dice vs the raw map when given."""
    import torch

    dice, r, top = [], [], []
    with torch.no_grad():
        for s in range(0, len(pred_latent), batch_size):
            p = torch.sigmoid(decoder(pred_latent[s:s + batch_size]))
            t = torch.sigmoid(decoder(true_latent[s:s + batch_size]))
            dice.append(_dice_pct(p, t, T2B_DICE_PCT))
            r.append((_zrows(p) * _zrows(t)).mean(1))
            if raw_maps is not None:
                top.append(_dice_pct(p, raw_maps[s:s + batch_size].to(p.device, non_blocking=True), T2B_TOP_PCT))
    out = {"dice": float(torch.cat(dice).mean()), "pearson": float(torch.cat(r).mean())}
    if top:
        out["dice_top10_raw"] = float(torch.cat(top).mean())
    return out


def ridge_fit(X, Y, alpha):
    """Closed-form ridge in float64, intercept not penalised; returns (W [768, 384], b [384])."""
    import torch

    Xd, Yd = X.double(), Y.double()
    mx, my = Xd.mean(0), Yd.mean(0)
    Xc = Xd - mx
    W = torch.linalg.solve(Xc.T @ Xc + alpha * torch.eye(X.shape[1], device=X.device, dtype=torch.float64), Xc.T @ (Yd - my))
    return W.float(), (my - mx @ W).float()


def train_text_to_brain(inputs, device, *, epochs, batch_size, seed, logger, out, meta):
    """Head v2: latent MSE + decoder-space Pearson terms, input noise, early stopping on validation Dice."""
    import torch
    from torch.nn import functional as F
    from torch.optim import AdamW
    from neurovlm.models import ProjHead

    # validation without the leakage-audit PMIDs, as every evaluation
    m = inputs["masks"]
    keep_val = apply_pubmed_exclusions(inputs["pmids"][m["val"]], logger=logger, label="PubMed val (head v2 early stopping)")
    tr_rows = torch.from_numpy(np.flatnonzero(m["train"]))
    val_rows = torch.from_numpy(np.flatnonzero(m["val"])[keep_val])
    X_train, Y_train = inputs["latent_text"][tr_rows].to(device), inputs["latent_neuro"][tr_rows].to(device)
    X_val, Y_val = inputs["latent_text"][val_rows].to(device), inputs["latent_neuro"][val_rows].to(device)
    raw_val = inputs["neuro_vectors"][val_rows].float()
    if gpu_can_hold(raw_val.numel() * 4, device):
        raw_val = raw_val.to(device)
    decoder = copy.deepcopy(inputs["autoencoder"].decoder).to(device).eval()
    decoder.requires_grad_(False)

    # reference: closed-form ridge, alpha chosen on the same validation Dice
    ridge = {}
    for alpha in RIDGE_ALPHAS:
        W, b = ridge_fit(X_train, Y_train, alpha)
        ridge[alpha] = text_to_brain_metrics(decoder, X_val @ W + b, Y_val, raw_val)
    ridge_alpha = max(ridge, key=lambda a: ridge[a]["dice"])
    logger.info("ridge reference (val): alpha %s -> %s", ridge_alpha, ridge[ridge_alpha])

    head = ProjHead(seed=HEAD_SEED, latent_in_dim=768, hidden_dim=512, latent_out_dim=384).to(device)
    with torch.no_grad():  # start at the mean training latent (latents are far from 0, mean norm ~35)
        head.aligner[-1].bias += Y_train.mean(0)
    optimizer = AdamW(head.parameters(), lr=T2B_LR, weight_decay=T2B_WD)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    noise_gen = torch.Generator(device=device).manual_seed(seed)
    n = len(X_train)

    def evaluate():
        head.eval()
        with torch.no_grad():
            pred = torch.cat([head(X_val[s:s + 2048]) for s in range(0, len(X_val), 2048)])
        res = text_to_brain_metrics(decoder, pred, Y_val, raw_val)
        res["latent_mse"] = float(((pred - Y_val) ** 2).mean())
        head.train()
        return res

    def snapshot():
        return {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    def corr_loss(a, b):
        return (1 - (_zrows(a) * _zrows(b)).mean(1)).mean()

    best = {"epoch": -1, "val": evaluate(), "state": snapshot()}
    curve = [[0, best["val"]]]
    with timed("train_text_to_brain", logger, out=out, epochs=epochs, batch_size=batch_size, **meta) as t:
        head.train()
        for epoch in range(1, epochs + 1):
            perm = torch.randperm(n, generator=gen).to(device)
            total = torch.zeros((), device=device)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                x, y = X_train[idx], Y_train[idx]
                x = F.normalize(x + T2B_NOISE * torch.randn(x.shape, device=device, generator=noise_gen) / x.shape[1] ** 0.5, dim=1)
                pred = head(x)
                logits = decoder(pred)
                with torch.no_grad():
                    target_logits = decoder(y)
                loss = (F.mse_loss(pred, y) + T2B_W_LOGIT * corr_loss(logits, target_logits)
                        + T2B_W_PROB * corr_loss(torch.sigmoid(logits), torch.sigmoid(target_logits)))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += loss.detach() * len(idx)
            res = evaluate()
            res["train_loss"] = float(total) / n
            curve.append([epoch, res])
            if res["dice"] > best["val"]["dice"]:
                best = {"epoch": epoch, "val": res, "state": snapshot()}
            logger.info("text_to_brain epoch %d: loss %.4f val dice %.4f pearson %.4f top10 %.4f (best %.4f @ %d)", epoch, res["train_loss"],
                        res["dice"], res["pearson"], res.get("dice_top10_raw", float("nan")), best["val"]["dice"], best["epoch"])
    head.load_state_dict(best["state"])
    summary = {"recipe": {"architecture": "ProjHead(768, 512, 384) + frozen autoencoder.decoder", "n_parameters": int(sum(p.numel() for p in head.parameters())),
                          "loss": f"latent MSE + {T2B_W_LOGIT} * (1 - r(decoder logits)) + {T2B_W_PROB} * (1 - r(sigmoid maps))",
                          "input_noise": T2B_NOISE, "optimizer": f"AdamW(lr={T2B_LR}, weight_decay={T2B_WD})", "batch_size": batch_size,
                          "max_epochs": epochs, "selection": f"best validation Dice (p{T2B_DICE_PCT}, decoded target)", "seed": seed,
                          "head_init_seed": HEAD_SEED},
               "best_epoch": best["epoch"], "val_best": best["val"], "val_curve": curve, "n_train": int(n), "n_val": int(len(X_val)),
               "n_val_excluded": int((~keep_val).sum()), "ridge_val": {"alpha": ridge_alpha, **ridge[ridge_alpha]},
               "ridge_val_path": {str(a): v for a, v in ridge.items()}, "seconds": t.seconds, "sec_per_epoch": t.seconds / max(epochs, 1)}
    if inputs.get("stage_a_head") is not None:  # the notebook head on the same validation rows, for the record
        a_head = inputs["stage_a_head"].to(device).eval()
        with torch.no_grad():
            summary["stage_a_val"] = text_to_brain_metrics(decoder, a_head(X_val), Y_val, raw_val)
        inputs["stage_a_head"].cpu()
    logger.info("head v2: best epoch %d val %s | ridge (alpha %s) val %s", best["epoch"], best["val"], ridge_alpha, ridge[ridge_alpha])
    return head.cpu().eval(), summary


def save_head(module, stem: Path, logger):
    """save_module_and_state, replacing symlinks first so a run that links another run's files never writes through."""
    for path in (stem.with_suffix(".pt"), stem.with_suffix(".safetensors")):
        if path.is_symlink():
            logger.info("replacing symlink %s -> %s with a file", path, path.resolve())
            path.unlink()
    save_module_and_state(module, stem, logger=logger)


def train_infonce(inputs, text_head, device, *, epochs, batch_size, lr, logger, out, meta, name="train_infonce"):
    """InfoNCE heads (notebook cell 8, figures/20 cell 6); the final weights are the result."""
    import torch
    from torch.nn import functional as F
    from torch.optim import AdamW
    from neurovlm.models import ProjHead
    from neurovlm.models.losses import InfoNCELoss

    m = inputs["masks"]
    X_train_image = inputs["latent_neuro"][m["train"]].to(device)
    X_train_text = F.normalize(inputs["latent_text"][m["train"]].to(device), dim=1, eps=1e-8)
    X_val_image = inputs["latent_neuro"][m["val"]].to(device)
    X_val_text = F.normalize(inputs["latent_text"][m["val"]].to(device), dim=1, eps=1e-8)

    proj_head_text = copy.deepcopy(text_head).to(device)
    proj_head_image = ProjHead(seed=HEAD_SEED, latent_in_dim=384, hidden_dim=384, latent_out_dim=384).to(device)
    loss_fn = InfoNCELoss(temperature=NCE_TEMP)
    optimizer = AdamW([*proj_head_text.parameters(), *proj_head_image.parameters()], lr=lr)
    curve = []
    n = len(X_train_image)
    with timed(name, logger, out=out, epochs=epochs, batch_size=batch_size, lr=lr, **meta) as t:
        for iepoch in range(epochs):
            proj_head_text.train()
            proj_head_image.train()
            torch.manual_seed(iepoch)
            rand_inds = torch.randperm(n, device=device)
            for i in range(0, n, batch_size):
                idx = rand_inds[i:i + batch_size]
                loss = loss_fn(proj_head_text(X_train_text[idx]), proj_head_image(X_train_image[idx]))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            if iepoch % VAL_INTERVAL == 0 or iepoch == epochs - 1:
                proj_head_text.eval()
                proj_head_image.eval()
                with torch.no_grad():
                    vl = float(loss_fn(proj_head_text(X_val_text), proj_head_image(X_val_image)))
                curve.append([iepoch, vl])
                logger.info("infonce epoch %d val loss %.5g", iepoch, vl)
    return proj_head_text.cpu().eval(), proj_head_image.cpu().eval(), {
        "val_loss_curve": curve, "seconds": t.seconds, "sec_per_epoch": t.seconds / max(epochs, 1), "epochs": epochs}


def main(argv=None):
    p = base_parser("03_models/09: train the text-to-brain head (v2 by default) and the InfoNCE heads.", stage=STAGE)
    add_model_args(p, amp_default="fp32")
    p.add_argument("--neuro-vectors", type=Path, default=None, help="neuro_vectors.pt (default: the pinned HF pubmed_images.pt).")
    p.add_argument("--infonce-epochs", type=int, default=None, help="InfoNCE epochs (default 301; 1 with --smoke).")
    p.add_argument("--text-head", choices=["v2", "v1"], default="v2",
                   help="Head saved as proj_head_text_mse: v2 (default) or v1 (the notebook's BCE head, stage A).")
    p.add_argument("--only-text-head", action="store_true", help="Train only head v2 into an existing run.")
    p.add_argument("--t2b-epochs", type=int, default=None, help="Head v2 epochs (default 80; --epochs with --smoke).")
    args = finalize_args(p.parse_args(argv))
    if args.only_text_head and args.text_head != "v2":
        p.error("--only-text-head requires --text-head v2")
    run_dir = resolve_run_dir(args)
    setup_torch(args.seed, args.deterministic)

    device = select_device(args.device)
    mse_epochs = args.epochs or MSE_EPOCHS
    nce_epochs = args.infonce_epochs or (1 if args.smoke else NCE_EPOCHS)
    t2b_epochs = args.t2b_epochs or (args.epochs if args.smoke else None) or T2B_EPOCHS
    meta = dict(stage=STAGE, smoke=args.smoke, run=args.run_name, text_head=args.text_head)
    if args.only_text_head:
        names = ["proj_head_text_mse"]
    elif args.text_head == "v2":
        names = ["proj_head_text_mse", "proj_head_text_bce", "proj_head_text_infonce", "proj_head_image_infonce"]
    else:
        names = ["proj_head_text_mse", "proj_head_text_infonce", "proj_head_image_infonce"]
    outputs = {f"{n}.pt": run_dir / f"{n}.pt" for n in names}
    summary_path = run_dir / f"{STAGE}.summary.json"
    done = artifacts_exist(*outputs.values(), force=args.force)
    if done and args.text_head == "v2" and not args.force:  # a run trained before head v2 has proj_head_text_mse but no head v2
        prev = read_json(summary_path) if summary_path.exists() else {}
        done = bool(prev.get("text_to_brain"))
        if not done:
            log.info("%s holds no text-to-brain head v2 (summary has no 'text_to_brain'); training it", run_dir)
    if done:
        log.info("artifacts exist in %s; skipping (use --force to retrain)", run_dir)
        if args.install:
            install_artifacts(outputs, log)
        return 0

    with timed("load_inputs", log, out=args.out, **meta):
        inputs = load_inputs(args, run_dir)

    if args.only_text_head:
        prev = read_json(summary_path) if summary_path.exists() else {}
        if not prev.get("text_to_brain") and not (run_dir / "proj_head_text_bce.pt").exists():
            # the run's proj_head_text_mse is its stage-A head: keep it as proj_head_text_bce
            for suffix in (".pt", ".safetensors"):
                src, dst = run_dir / f"proj_head_text_mse{suffix}", run_dir / f"proj_head_text_bce{suffix}"
                if src.is_symlink():
                    dst.symlink_to(Path(src.readlink()))
                elif src.exists():
                    shutil.copy2(src, dst)
                if dst.exists():
                    log.info("stage-A head preserved: %s -> %s", dst, dst.resolve())
        bce_path = run_dir / "proj_head_text_bce.pt"
        if bce_path.exists():
            inputs["stage_a_head"] = load_module_or_state(bce_path)
        head, t2b_summary = train_text_to_brain(inputs, device, epochs=t2b_epochs, batch_size=T2B_BATCH, seed=args.seed,
                                                logger=log, out=args.out, meta=meta)
        save_head(head, run_dir / "proj_head_text_mse", log)
        summary = read_json(summary_path) if summary_path.exists() else {}
        if summary_path.is_symlink():  # a linked summary: keep its content, write our own file
            summary["stage_a_infonce_source"] = str(summary_path.resolve())
            summary_path.unlink()
        summary.update({"text_to_brain": t2b_summary, "text_head": "v2", "device": str(device), "autoencoder_source": inputs["ae_source"],
                        "n_rows": {k: int(v.sum()) for k, v in inputs["masks"].items()}})
        write_stage_summary(run_dir, STAGE, summary)
        if args.install:
            install_artifacts(outputs, log)
        return 0

    head, mse_summary = train_text_mse(inputs, device, epochs=mse_epochs, batch_size=args.batch_size or MSE_BATCH,
                                       amp=args.amp, logger=log, out=args.out, meta=meta)
    t2b_summary = None
    if args.text_head == "v2":
        save_head(head, run_dir / "proj_head_text_bce", log)
        inputs["stage_a_head"] = head
        t2b_head, t2b_summary = train_text_to_brain(inputs, device, epochs=t2b_epochs, batch_size=T2B_BATCH, seed=args.seed,
                                                    logger=log, out=args.out, meta=meta)
        save_head(t2b_head, run_dir / "proj_head_text_mse", log)
    else:
        save_head(head, run_dir / "proj_head_text_mse", log)

    # the InfoNCE text head starts from stage A in both modes
    text_head, image_head, nce_summary = train_infonce(inputs, head, device, epochs=nce_epochs, batch_size=NCE_BATCH,
                                                       lr=NCE_LR, logger=log, out=args.out, meta=meta)
    save_head(text_head, run_dir / "proj_head_text_infonce", log)
    save_head(image_head, run_dir / "proj_head_image_infonce", log)

    if summary_path.is_symlink():
        summary_path.unlink()
    write_stage_summary(run_dir, STAGE, {"text_mse": mse_summary, "text_to_brain": t2b_summary, "text_head": args.text_head,
                                         "infonce": nce_summary, "amp": args.amp, "device": str(device),
                                         "autoencoder_source": inputs["ae_source"],
                                         "n_rows": {k: int(v.sum()) for k, v in inputs["masks"].items()}})
    if args.install:
        install_artifacts(outputs, log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
