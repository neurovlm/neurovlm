#!/usr/bin/env python
"""5-fold PubMed cross-validation: autoencoder, text-to-brain head and InfoNCE heads per fold (training cells of
docs/figures/20_pubmed_cv.ipynb; the recall evaluation is validate_models.py --components cv).

    .venv/bin/python scripts/models/20_pubmed_cv.py --run-name <run> [--folds 0,1] [--smoke]

Per fold, 5 % of all rows are drawn from the training fold for validation; unlike the notebook the draws are seeded
(1000 + fold for the autoencoder, 2000 + fold for the text-to-brain head). Writes {autoencoder, proj_head_text_mse,
proj_head_text_infonce, proj_head_image_infonce}_kfold-{i}.{pt,safetensors} and training_runs/ to scripts/outputs/runs/<run>/.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/models/
from _common_models import *  # noqa: E402,F403

STAGE = "20_pubmed_cv"
log = get_logger(STAGE)

AE_EPOCHS, AE_BATCH, AE_LR, AE_WD = 101, 256, 5e-5, 0.01
T2B_EPOCHS, T2B_BATCH, T2B_LR, T2B_WD = 501, 1024, 1e-4, 0.01
NCE_EPOCHS, NCE_BATCH, NCE_LR = 301, 2048, 2e-5
N_SPLITS = 5


def _load_stage(name: str, filename: str):
    """Import a sibling stage script (their file names start with a digit)."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fold_val_split(train, test, *, seed):
    """Notebook: a validation set of 5 % of all rows, drawn from the training fold (seeded here)."""
    import torch

    gen = torch.Generator().manual_seed(seed)
    train_inds = torch.randperm(len(train), generator=gen)
    val_size = int((len(train) + len(test)) * 0.05)
    return train[train_inds[val_size:]], train[train_inds[:val_size]]


def main(argv=None):
    p = base_parser("figures/20: 5-fold PubMed CV training (autoencoder, text-to-brain, InfoNCE heads).", stage=STAGE)
    add_model_args(p, amp_default="fp32")
    p.add_argument("--neuro-vectors", type=Path, default=None, help="neuro_vectors.pt (default: the pinned HF pubmed_images.pt).")
    p.add_argument("--folds", default=None, help="Comma-separated folds (default: all; 0 with --smoke).")
    p.add_argument("--ae-epochs", type=int, default=None, help="Autoencoder epochs (default 101).")
    p.add_argument("--t2b-epochs", type=int, default=None, help="Text-to-brain epochs (default 501).")
    p.add_argument("--infonce-epochs", type=int, default=None, help="InfoNCE epochs (default 301).")
    p.add_argument("--eval-batch-size", type=int, default=1024, help="Autoencoder validation/test batch size.")
    p.add_argument("--no-fast-eval", action="store_true", help="Keep the library's CPU rank AUROC in per-epoch validation.")
    args = finalize_args(p.parse_args(argv))
    run_dir = resolve_run_dir(args)
    setup_torch(args.seed, args.deterministic)
    import pandas as pd
    import torch
    from sklearn.model_selection import KFold
    from neurovlm.models import ProjHead
    from neurovlm.training import (MLPAutoencoderTrainConfig, MLPTextToBrainTrainConfig, train_mlp_autoencoder,
                                   train_mlp_text_to_brain)
    from neurovlm.training.mlp import build_mlp_autoencoder

    stage09 = _load_stage("stage09", "09_projection_head.py")
    stage08 = _load_stage("stage08", "08_autoencoder.py")

    device = select_device(args.device)
    ae_epochs = args.ae_epochs or args.epochs or AE_EPOCHS
    t2b_epochs = args.t2b_epochs or args.epochs or T2B_EPOCHS
    nce_epochs = args.infonce_epochs or args.epochs or NCE_EPOCHS
    folds = [int(f) for f in args.folds.split(",")] if args.folds else ([0] if args.smoke else list(range(N_SPLITS)))
    meta = dict(stage=STAGE, smoke=args.smoke, run=args.run_name)
    smoke_caps = dict(max_train_batches=args.caps.max_batches, max_eval_batches=args.caps.max_batches) if args.smoke else {}

    # cell 2: maps of studies with <= 100 coordinates; cell 5: SPECTER2 embeddings aligned to them
    with timed("load_inputs", log, out=args.out, **meta):
        neuro_vectors, pmids = stage08.load_neuro_vectors(args.neuro_vectors, log)
        mask = coordinate_filter(pmids, max_coords=100)
        pmids, neuro_vectors = np.asarray(pmids[mask]), neuro_vectors[torch.from_numpy(mask)]
        if args.max_samples and args.max_samples < len(pmids):
            g = torch.Generator().manual_seed(args.seed)
            keep = torch.randperm(len(pmids), generator=g)[: args.max_samples].sort().values
            pmids, neuro_vectors = pmids[keep.numpy()], neuro_vectors[keep]
        payload = load_torch(hf_pinned("neurovlm/embedded_text", "latent_specter2_adhoc.pt"))
        latent_text, pmids_specter = payload["latent"].float(), np.asarray(payload["pmid"])
        m = pd.Series(pmids_specter).isin(pmids).to_numpy()
        latent_text, pmids_specter = latent_text[torch.from_numpy(m)], pmids_specter[m]
        assert (pmids_specter == pmids).all()
        log.info("rows: %d; folds: %s", len(pmids), folds)
        placed, on_device = place_splits({"all": {"brain": neuro_vectors}}, device, logger=log)
        brain_all = placed["all"]["brain"]
        text_all = latent_text.to(device)
        del neuro_vectors

    splits = list(KFold(n_splits=N_SPLITS, shuffle=True, random_state=0).split(np.arange(len(pmids))))
    summary = {"folds": {}, "device": str(device), "amp": args.amp, "on_device": on_device,
               "epochs": {"ae": ae_epochs, "t2b": t2b_epochs, "infonce": nce_epochs}}

    for idx in folds:
        train, test = (torch.from_numpy(a) for a in splits[idx])
        fold_summary = {}
        ae_out = run_dir / f"autoencoder_kfold-{idx}.pt"
        mse_out = run_dir / f"proj_head_text_mse_kfold-{idx}.pt"
        nce_text_out = run_dir / f"proj_head_text_infonce_kfold-{idx}.pt"
        nce_img_out = run_dir / f"proj_head_image_infonce_kfold-{idx}.pt"

        # cell 3: autoencoder
        if artifacts_exist(ae_out, force=args.force):
            log.info("fold %d: %s exists, skipping AE", idx, ae_out.name)
            autoencoder = load_module_or_state(ae_out, device=device)
        else:
            tr, va = fold_val_split(train, test, seed=1000 + idx)
            config = MLPAutoencoderTrainConfig(
                output_root=run_dir / "training_runs", run_id=f"pubmed-cv-autoencoder-fold-{idx}", seed=idx,
                device=str(device), epochs=ae_epochs, batch_size=args.batch_size or AE_BATCH, learning_rate=AE_LR,
                weight_decay=AE_WD, eval_batch_size=args.eval_batch_size, **smoke_caps)
            fresh_trainer_dir(config.output_root, config.run_id)
            provider = {
                "train": make_batched_loader(BatchedTensorDataset(brain=brain_all[tr.to(brain_all.device)]), batch_size=config.batch_size, shuffle=True, seed=idx),
                "val": make_batched_loader(BatchedTensorDataset(brain=brain_all[va.to(brain_all.device)]), batch_size=args.eval_batch_size, shuffle=False),
                "test": make_batched_loader(BatchedTensorDataset(brain=brain_all[test.to(brain_all.device)]), batch_size=args.eval_batch_size, shuffle=False),
            }
            model = build_mlp_autoencoder(config)
            with timed(f"ae_fold{idx}", log, out=args.out, epochs=ae_epochs, fold=idx, **meta) as t:
                with bf16_training_forward(model, device, args.amp), fast_mlp_eval(device, enabled=not args.no_fast_eval):
                    result = train_mlp_autoencoder(config, provider=provider, model=model)
            autoencoder = result.model
            save_module_and_state(autoencoder, ae_out.with_suffix(""), logger=log)
            autoencoder.to(device)
            fold_summary["autoencoder"] = summarise_trainer_run(result, seconds=t.seconds)
            del provider

        # cell 6: text-to-brain head (MSE to the fold autoencoder's latents)
        autoencoder = autoencoder.to(device).eval()
        latent_neuro = encode_in_batches(autoencoder.encoder, brain_all, batch_size=4096, device=device, out_device=device)
        if artifacts_exist(mse_out, nce_text_out, nce_img_out, force=args.force):
            log.info("fold %d: heads exist, skipping", idx)
            summary["folds"][idx] = fold_summary
            continue
        tr, va = fold_val_split(train, test, seed=2000 + idx)
        tr_d, va_d, te_d = tr.to(device), va.to(device), test.to(device)
        proj_head = ProjHead(seed=idx, latent_in_dim=768, hidden_dim=512, latent_out_dim=384)
        config = MLPTextToBrainTrainConfig(
            output_root=run_dir / "training_runs", run_id=f"pubmed-cv-text-to-brain-fold-{idx}", seed=idx,
            device=str(device), epochs=t2b_epochs, batch_size=T2B_BATCH, learning_rate=T2B_LR, weight_decay=T2B_WD,
            eval_batch_size=4096, **smoke_caps)
        fresh_trainer_dir(config.output_root, config.run_id)
        provider = {
            name: make_batched_loader(
                BatchedTensorDataset(text_embedding=text_all[sel], brain_embedding=latent_neuro[sel]),
                batch_size=config.batch_size if name == "train" else 4096, shuffle=(name == "train"), seed=idx)
            for name, sel in (("train", tr_d), ("val", va_d), ("test", te_d))
        }
        with timed(f"t2b_fold{idx}", log, out=args.out, epochs=t2b_epochs, fold=idx, **meta) as t:
            result = train_mlp_text_to_brain(config, provider=provider, autoencoder=autoencoder, text_projection=proj_head)
        proj_head = result.model.text_projection
        save_module_and_state(proj_head, mse_out.with_suffix(""), logger=log)
        fold_summary["text_to_brain"] = summarise_trainer_run(result, seconds=t.seconds)

        # cell 6 (cont.): InfoNCE heads, text head initialised from the fold's text-to-brain head
        rows = np.arange(len(pmids))
        inputs = {"latent_neuro": latent_neuro.cpu(), "latent_text": text_all.cpu(),
                  "masks": {"train": np.isin(rows, tr.numpy()), "val": np.isin(rows, va.numpy())}}
        text_head, image_head, nce_summary = stage09.train_infonce(
            inputs, proj_head, device, epochs=nce_epochs, batch_size=NCE_BATCH, lr=NCE_LR, logger=log, out=args.out,
            meta={**meta, "fold": idx}, name=f"infonce_fold{idx}")
        save_module_and_state(text_head, nce_text_out.with_suffix(""), logger=log)
        save_module_and_state(image_head, nce_img_out.with_suffix(""), logger=log)
        fold_summary["infonce"] = nce_summary
        summary["folds"][idx] = fold_summary
        write_stage_summary(run_dir, STAGE, summary)

    write_stage_summary(run_dir, STAGE, summary)
    if args.install:
        stems = ("autoencoder", "proj_head_text_mse", "proj_head_text_infonce", "proj_head_image_infonce")
        install_artifacts({f"{s}_kfold-{i}.pt": run_dir / f"{s}_kfold-{i}.pt" for i in folds for s in stems}, log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
