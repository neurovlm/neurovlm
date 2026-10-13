#!/usr/bin/env python
"""Loss ablations of the 09 projection heads (reviewer 3.4): text-to-brain losses (stage A and head v2) and contrastive
variants (stage B), --n-seeds seeds each, on the selected model's autoencoder latents (--model-dir, default runs/final),
scored on the PubMed test split. Seed 123 of the paper variants retrains the final model's own heads.

Usage: run_analysis.py --device cuda [--force]   (smoke: --smoke --device cpu)
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _final_model as M  # noqa: E402
from _a5_common import (  # noqa: E402
    Timings, eval_heads_retrieval, eval_text_to_brain_head, exclude_heldout_rows, get_logger, load_pubmed_latents, load_pubmed_maps,
    load_specter_latents, read_json, same_exclusions, seed_all, select_device, set_offline_env, split_masks, subsample_masks,
    train_infonce_heads, train_text_to_brain_head, train_text_to_brain_v2, write_json,
)

set_offline_env()
log = get_logger("a5_loss_ablation")

MSE_EPOCHS, MSE_BATCH, MSE_LR = 201, 1024, 5e-5
NCE_EPOCHS, NCE_BATCH, NCE_LR, NCE_TEMP = 301, 2048, 1e-5, 0.07
T2B_EPOCHS = 80  # head v2
HEAD_SEED = 123
PAPER_DICE_PCT = 70  # paper Dice: both maps thresholded at their 70th percentile
GPU_RATES = {"mse_sec_per_epoch": 0.093, "infonce_sec_per_epoch": 0.017, "head_v2_sec_per_epoch": 0.27}  # 09 runs (scripts/README.md)

CONTRASTIVE_VARIANTS = {
    # name: (kind, temperature, learnable, init_from_stage_a, description)
    "symmetric_t0.07": ("symmetric", 0.07, False, True, "paper: symmetric InfoNCE, tau 0.07, text head init from stage A"),
    "t2i_only": ("t2i", 0.07, False, True, "single-direction CE: text -> image only"),
    "i2t_only": ("i2t", 0.07, False, True, "single-direction CE: image -> text only"),
    "symmetric_t0.03": ("symmetric", 0.03, False, True, "sharper temperature (tau 0.03)"),
    "symmetric_t0.20": ("symmetric", 0.20, False, True, "softer temperature (tau 0.20)"),
    "learnable_temp": ("symmetric", 0.07, True, True, "learnable logit scale from tau 0.07 (library option, unused in the paper)"),
    "cosine_only": ("cosine", None, False, True, "no contrastive cross-entropy: 1 - cos on matched pairs only"),
    "random_text_init": ("symmetric", 0.07, False, False, "paper loss, text head randomly initialised instead of from stage A"),
}
T2B_VARIANTS = {
    "head_v2": "final: head v2, latent MSE + Pearson terms on decoder logits and maps, input noise, best validation Dice",
    "decoder_bce": "submitted: BCE-with-logits through the frozen decoder (28,542-d); initialises the contrastive text head",
    "latent_mse": "library train_mlp_text_to_brain objective: MSE on the 384-d latent",
    "latent_cosine": "1 - cosine on the 384-d latent",
}
C_METRICS = ("t2i_auc", "i2t_auc", "mean_auc", "mean_recall@10", "mean_mrr")
T_METRICS = ("dice_matched", "dice_null", "pearson_matched", "pearson_null", "map_recall_auc", "dice_vs_raw_matched")
SMOKE = {"max_samples": 64, "epochs": 1, "infonce_epochs": 1, "max_batches": 2, "n_seeds": 2}


class ContrastiveVariant(nn.Module):
    """Contrastive loss variants, ``forward(text, image)``; ``symmetric`` equals ``neurovlm.models.losses.InfoNCELoss``."""

    def __init__(self, kind: str, temperature: float | None = 0.07, learnable: bool = False, max_scale_logit: float = 100.0):
        super().__init__()
        self.kind, self.temperature, self.learnable, self.max_scale_logit = kind, temperature, learnable, max_scale_logit
        if learnable:
            self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature), dtype=torch.float32))

    def temperature_value(self) -> float:
        if self.learnable:
            return float(1.0 / self.logit_scale.detach().exp())
        return float(self.temperature) if self.temperature is not None else float("nan")

    def forward(self, text, image):
        text, image = F.normalize(text, dim=1), F.normalize(image, dim=1)
        if self.kind == "cosine":
            return (1.0 - (text * image).sum(dim=1)).mean()
        logits = text @ image.T
        if self.learnable:
            with torch.no_grad():
                self.logit_scale.clamp_(max=math.log(self.max_scale_logit))
            logits = logits * self.logit_scale.exp()
        else:
            logits = logits / self.temperature
        labels = torch.arange(text.size(0), device=text.device)
        if self.kind == "t2i":
            return F.cross_entropy(logits, labels)
        if self.kind == "i2t":
            return F.cross_entropy(logits.T, labels)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def selfcheck_symmetric() -> float:
    """|symmetric variant - library InfoNCELoss| on random data (raises if > 1e-5)."""
    from neurovlm.models.losses import InfoNCELoss

    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(64, 384, generator=g), torch.randn(64, 384, generator=g)
    ref, got = float(InfoNCELoss(0.07)(b, a)), float(ContrastiveVariant("symmetric", 0.07)(a, b))
    if abs(ref - got) > 1e-5:
        raise RuntimeError(f"symmetric variant mismatch vs InfoNCELoss: {ref} vs {got}")
    return abs(ref - got)


def aggregate(runs: dict, variants, seeds, metrics) -> dict:
    """Mean and sd over seeds per variant."""
    out = {}
    for v in variants:
        rs = [runs[f"{v}@{s}"] for s in seeds if f"{v}@{s}" in runs]
        if not rs:
            continue
        agg = {"n_seeds": len(rs)}
        for m in metrics:
            vals = np.asarray([r.get(m, np.nan) for r in rs], dtype=float)
            agg[f"{m}_mean"], agg[f"{m}_sd"] = float(np.nanmean(vals)), float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else float("nan")
        temps = [r.get("training", {}).get("final_temperature") for r in rs]
        temps = [t for t in temps if t is not None and np.isfinite(t)]
        if temps:
            agg["final_temperature_mean"] = float(np.mean(temps))
        out[v] = agg
    return out


def load_stage_a_head(path: Path, seed: int):
    from neurovlm.models import ProjHead

    head = ProjHead(seed=seed, latent_in_dim=768, hidden_dim=512, latent_out_dim=384)
    head.load_state_dict(torch.load(path, weights_only=True))
    return head


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "outputs")
    M.add_model_args(p)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=HEAD_SEED, help="First head seed (paper: 123).")
    p.add_argument("--n-seeds", type=int, default=None, help="Number of seeds (default 3; 2 in --smoke).")
    p.add_argument("--epochs", type=int, default=None, help="Stage-A epochs (default 201; 1 in --smoke).")
    p.add_argument("--infonce-epochs", type=int, default=None, help="Stage-B epochs (default 301; 1 in --smoke).")
    p.add_argument("--max-samples", type=int, default=None, help="Cap rows (split-proportional prefix).")
    p.add_argument("--max-batches", type=int, default=None, help="Cap batches per epoch.")
    p.add_argument("--contrastive", default=",".join(CONTRASTIVE_VARIANTS), help="Contrastive variants (comma list).")
    p.add_argument("--t2b", default=",".join(T2B_VARIANTS), help="Text-to-brain variants (comma list).")
    p.add_argument("--smoke", action="store_true", help="64 rows, 1 epoch x 2 batches, 2 seeds.")
    p.add_argument("--force", action="store_true", help="Recompute finished runs.")
    p.add_argument("--no-exclusions", action="store_true", help="Keep the leakage-audit papers in val/test.")
    args = p.parse_args(argv)
    if args.smoke:
        for k, v in SMOKE.items():
            setattr(args, k, getattr(args, k) or v)
    mse_epochs, nce_epochs = args.epochs or MSE_EPOCHS, args.infonce_epochs or NCE_EPOCHS
    seeds = [args.seed + i for i in range(args.n_seeds or 3)]
    c_variants = [v for v in args.contrastive.split(",") if v]
    t_variants = [v for v in args.t2b.split(",") if v]
    unknown = [v for v in c_variants if v not in CONTRASTIVE_VARIANTS] + [v for v in t_variants if v not in T2B_VARIANTS]
    if unknown:
        p.error(f"unknown variants: {unknown}")
    out = args.out
    M.from_args(args)
    t2b_epochs = 1 if args.smoke else T2B_EPOCHS
    ck_dir = out / "checkpoints" / M.model_tag()
    ck_dir.mkdir(parents=True, exist_ok=True)
    timings = Timings(out / "timings.json")
    seed_all(args.seed)

    device = select_device(args.device)
    check = selfcheck_symmetric()
    log.info("device %s | model %s | symmetric self-check |delta| = %.2e | epochs A/B/v2 %d/%d/%d | seeds %s | smoke %s", device, M.model_tag(), check,
             mse_epochs, nce_epochs, t2b_epochs, seeds, args.smoke)

    with timings.time("load_data"):
        latents, pmids = load_pubmed_latents()
        # exclusions before the --max-samples prefix
        masks, excl = exclude_heldout_rows(split_masks(pmids), pmids, enabled=not args.no_exclusions, log=log.info)
        masks = subsample_masks(masks, args.max_samples)
        keep = masks["train"] | masks["val"] | masks["test"]
        latents, pmids = latents[torch.from_numpy(keep)], pmids[keep]
        masks = {k: v[keep] for k, v in masks.items()}
        text = load_specter_latents(pmids)
        maps = load_pubmed_maps(pmids)
        ae = M.component("autoencoder", device)
        ref_image_head = M.component("proj_head_image_infonce", device)
    tr, va, te = (torch.from_numpy(masks[k]) for k in ("train", "val", "test"))
    log.info("rows train/val/test = %d/%d/%d", tr.sum(), va.sum(), te.sum())
    prev = read_json(out / "metrics.json") if (out / "metrics.json").exists() else {}
    model_name = M.model_info()["model_dir"]
    if (args.force or bool(prev.get("smoke")) != bool(args.smoke) or not same_exclusions(prev, excl["enabled"]) or prev.get("model_dir") != model_name
            or prev.get("dice_pct") != PAPER_DICE_PCT):
        prev = {}
    results = {"runs": {"text_to_brain": prev.get("runs", {}).get("text_to_brain", {}), "contrastive": prev.get("runs", {}).get("contrastive", {})},
               "reference": prev.get("reference"), "smoke": args.smoke, "exclusions": excl, "model_dir": model_name, "dice_pct": PAPER_DICE_PCT}
    ck_tag = "_excl" if excl["enabled"] else ""  # stage-A heads are val-selected, so checkpoints are kept per setting

    # the selected model's own heads as reference
    if results["reference"] is None:
        with timings.time("eval_reference"):
            th, ih, mh = (M.component(k, device) for k in ("proj_head_text_infonce", "proj_head_image_infonce", "proj_head_text_mse"))
            results["reference"] = {"retrieval": eval_heads_retrieval(th, ih, text[te], latents[te], device),
                                    "text_to_brain": eval_text_to_brain_head(mh, ae, text[te], maps[te], device, text_va=text[va],
                                                                             image_head=ref_image_head, pct=PAPER_DICE_PCT)}
        write_json(out / "metrics.json", results)

    for seed in seeds:
        # stage A: text-to-brain objective
        stage_a_heads = {}
        for v in t_variants:
            key, ck = f"{v}@{seed}", ck_dir / f"t2b_{v}_s{seed}{ck_tag}.pt"
            if key in results["runs"]["text_to_brain"] and ck.exists():
                stage_a_heads[v] = load_stage_a_head(ck, seed).eval()
                log.info("t2b %s: done earlier, skipping", key)
                continue
            seed_all(seed)
            with timings.time(f"train_t2b_{v}_s{seed}", epochs=t2b_epochs if v == "head_v2" else mse_epochs, device=str(device)) as trow:
                if v == "head_v2":  # first seed (123): the 09 head v2 (head seed 123, data seed 0)
                    head, summ = train_text_to_brain_v2(text[tr], latents[tr], text[va], latents[va], decoder=ae.decoder, in_dim=768, device=device,
                                                        epochs=t2b_epochs, seed=seeds.index(seed), head_seed=seed, max_batches=args.max_batches, log=log)
                else:
                    head, summ = train_text_to_brain_head(text[tr], maps[tr], text[va], maps[va], decoder=ae.decoder, in_dim=768, device=device,
                                                          epochs=mse_epochs, batch_size=MSE_BATCH, lr=MSE_LR, seed=seed, loss=v, latent_tr=latents[tr],
                                                          latent_va=latents[va], max_batches=args.max_batches, log=log)
                trow.update({k: summ[k] for k in ("sec_per_epoch", "sec_per_step", "steps")})
            torch.save(head.state_dict(), ck)
            stage_a_heads[v] = head
            with timings.time(f"eval_t2b_{v}_s{seed}"):
                r = {"variant": v, "seed": seed, "description": T2B_VARIANTS[v], "training": summ,
                     **eval_text_to_brain_head(head, ae, text[te], maps[te], device, text_va=text[va], image_head=ref_image_head,
                                              pct=PAPER_DICE_PCT)}
            results["runs"]["text_to_brain"][key] = r
            log.info("t2b %-20s dice %.4f (null %.4f) pearson %.4f (null %.4f) map AUC %.4f", key, r["dice_matched"], r["dice_null"], r["pearson_matched"],
                     r["pearson_null"], r.get("map_recall_auc", float("nan")))
            write_json(out / "metrics.json", results)

        # stage B: contrastive objective, text head initialised from this seed's decoder_bce head
        init_head = stage_a_heads.get("decoder_bce")
        if init_head is None:
            ck = ck_dir / f"t2b_decoder_bce_s{seed}{ck_tag}.pt"
            if ck.exists():
                init_head = load_stage_a_head(ck, seed)
            else:
                raise SystemExit(f"stage-B init (seed {seed}) needs the decoder_bce head: include it in --t2b")
        for v in c_variants:
            key = f"{v}@{seed}"
            if key in results["runs"]["contrastive"]:
                log.info("contrastive %s: done earlier, skipping", key)
                continue
            kind, temp, learnable, use_init, desc = CONTRASTIVE_VARIANTS[v]
            seed_all(seed)
            loss_fn = ContrastiveVariant(kind, temp, learnable)
            with timings.time(f"train_contrastive_{v}_s{seed}", epochs=nce_epochs, device=str(device)) as trow:
                th, ih, summ = train_infonce_heads(text[tr], latents[tr], text[va], latents[va], text_head_init=init_head if use_init else None, in_dim=768,
                                                   device=device, loss_fn=loss_fn, epochs=nce_epochs, batch_size=NCE_BATCH, lr=NCE_LR, seed=seed,
                                                   max_batches=args.max_batches, log=log)
                trow.update({k: summ[k] for k in ("sec_per_epoch", "sec_per_step", "steps")})
            torch.save({"text": th.state_dict(), "image": ih.state_dict()}, ck_dir / f"contrastive_{v}_s{seed}{ck_tag}.pt")
            with timings.time(f"eval_contrastive_{v}_s{seed}"):
                r = {"variant": v, "seed": seed, "description": desc, "kind": kind, "temperature": temp, "learnable_temperature": learnable,
                     "init_from_stage_a": use_init, "training": summ, **eval_heads_retrieval(th, ih, text[te], latents[te], device)}
            results["runs"]["contrastive"][key] = r
            log.info("contrastive %-22s t2i AUC %.4f i2t AUC %.4f mean R@10 %.4f", key, r["t2i_auc"], r["i2t_auc"], r["mean_recall@10"])
            write_json(out / "metrics.json", results)

    import pandas as pd

    agg_c = aggregate(results["runs"]["contrastive"], c_variants, seeds, C_METRICS)
    agg_t = aggregate(results["runs"]["text_to_brain"], t_variants, seeds, T_METRICS)
    ref_c = agg_c.get("symmetric_t0.07", {})
    ref_t = agg_t.get("head_v2", {})
    tc = pd.DataFrame([{"variant": v, "description": CONTRASTIVE_VARIANTS[v][4], "temperature": CONTRASTIVE_VARIANTS[v][1], "n_seeds": a["n_seeds"],
                        **{k: a[k] for k in a if k.endswith(("_mean", "_sd"))},
                        "delta_mean_auc_vs_paper": a["mean_auc_mean"] - ref_c.get("mean_auc_mean", np.nan)} for v, a in agg_c.items()])
    tt = pd.DataFrame([{"variant": v, "description": T2B_VARIANTS[v], "n_seeds": a["n_seeds"], **{k: a[k] for k in a if k.endswith(("_mean", "_sd"))},
                        "delta_dice_vs_final": a["dice_matched_mean"] - ref_t.get("dice_matched_mean", np.nan)} for v, a in agg_t.items()])
    tc.to_csv(out / "table_contrastive.csv", index=False)
    tt.to_csv(out / "table_text_to_brain.csv", index=False)
    per_seed_sec = (sum(T2B_EPOCHS * GPU_RATES["head_v2_sec_per_epoch"] if v == "head_v2" else MSE_EPOCHS * GPU_RATES["mse_sec_per_epoch"] for v in t_variants)
                    + len(c_variants) * NCE_EPOCHS * GPU_RATES["infonce_sec_per_epoch"])
    results.update({"aggregate": {"contrastive": agg_c, "text_to_brain": agg_t}, "seeds": seeds, "device": str(device), "model_sources": M.model_sources(),
                    "epochs": {"stage_a": mse_epochs, "stage_b": nce_epochs, "head_v2": t2b_epochs}, "n_rows": {k: int(v.sum()) for k, v in masks.items()},
                    "gpu_rates_sec_per_epoch": GPU_RATES, "symmetric_selfcheck_abs_delta": check,
                    "runtime_estimate_gpu_seconds": {"per_contrastive_variant_per_seed": NCE_EPOCHS * GPU_RATES["infonce_sec_per_epoch"],
                                                     "per_t2b_variant_per_seed": MSE_EPOCHS * GPU_RATES["mse_sec_per_epoch"],
                                                     "all_variants_per_seed": per_seed_sec, "all_variants_all_seeds": per_seed_sec * len(seeds)}})
    write_json(out / "metrics.json", results)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(tc.drop(columns=["description"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(tt.drop(columns=["description"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
