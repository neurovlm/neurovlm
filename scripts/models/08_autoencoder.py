#!/usr/bin/env python
"""Train the MLP autoencoder (28,542 -> 384 -> 28,542) on PubMed coordinate maps (docs/03_models/08_autoencoder.ipynb).

    .venv/bin/python scripts/models/08_autoencoder.py --run-name <run> [--smoke]

Writes pmids_split.pt, autoencoder.{pt,safetensors}, latent_neuro.pt and training_runs/autoencoder/ to scripts/outputs/runs/<run>/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/models/
from _common_models import *  # noqa: E402,F403

STAGE = "08_autoencoder"
log = get_logger(STAGE)

EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY, SEED = 101, 256, 5e-5, 0.01, 0  # notebook cell 6


def load_neuro_vectors(path: Path | None, logger):
    """(neuro_vectors [N, 28542] float32 on CPU, pmids) from ``path``, else the pinned HF pubmed_images.pt (= neuro_vectors.pt)."""
    path = path or hf_pinned("neurovlm/embedded_text", "pubmed_images.pt")
    payload = load_torch(path)
    vectors, pmids = payload["neuro_vectors"], np.asarray(payload["pmid"])
    logger.info("neuro_vectors from %s: %s", path, tuple(vectors.shape))
    return vectors.float(), pmids


def main(argv=None):
    p = base_parser("03_models/08: train the MLP autoencoder and export the PubMed split + latents.", stage=STAGE)
    add_model_args(p, amp_default="fp32")
    p.add_argument("--neuro-vectors", type=Path, default=None, help="neuro_vectors.pt (default: the pinned HF pubmed_images.pt).")
    p.add_argument("--naive", action="store_true", help="Notebook loader (CPU tensors, per-batch copies to the GPU), for timing.")
    p.add_argument("--eval-batch-size", type=int, default=1024, help="Validation/test batch size.")
    p.add_argument("--no-fast-eval", action="store_true", help="Keep the library's CPU rank AUROC in per-epoch validation.")
    args = finalize_args(p.parse_args(argv))
    run_dir = resolve_run_dir(args)
    setup_torch(args.seed, args.deterministic)
    import torch
    from neurovlm.training import MLPAutoencoderTrainConfig, train_mlp_autoencoder
    from neurovlm.training.mlp import build_mlp_autoencoder

    device = select_device(args.device)
    epochs = args.epochs or EPOCHS
    batch_size = args.batch_size or BATCH_SIZE
    meta = dict(stage=STAGE, smoke=args.smoke, run=args.run_name)

    outputs = {name: run_dir / name for name in ("pmids_split.pt", "autoencoder.pt", "latent_neuro.pt")}  # --install copies these
    if artifacts_exist(*outputs.values(), run_dir / "autoencoder.safetensors", force=args.force):
        log.info("artifacts exist in %s; skipping (use --force to retrain)", run_dir)
        if args.install:
            install_artifacts(outputs, log)
        return 0

    # cell 3: load, keep studies with <= 100 coordinates
    with timed("load_inputs", log, out=args.out, **meta):
        neuro_vectors, pmids = load_neuro_vectors(args.neuro_vectors, log)
        mask = coordinate_filter(pmids, max_coords=100)
        pmids = np.asarray(pmids[mask])
        neuro_vectors = neuro_vectors[torch.from_numpy(mask)]
        if args.max_samples:
            g = torch.Generator().manual_seed(args.seed)
            keep = torch.randperm(len(pmids), generator=g)[: args.max_samples].sort().values
            pmids, neuro_vectors = pmids[keep.numpy()], neuro_vectors[keep]
        log.info("studies with <= 100 coordinates: %d", len(pmids))

    # cell 5: split
    train_inds, val_inds, test_inds = notebook_split(len(pmids))
    split_inds = {"train": train_inds, "val": val_inds, "test": test_inds}
    torch.save({"train": pmids[train_inds], "test": pmids[test_inds], "val": pmids[val_inds]}, outputs["pmids_split.pt"])
    log.info("split train/val/test = %d/%d/%d -> %s", len(train_inds), len(val_inds), len(test_inds), outputs["pmids_split.pt"])

    # cell 6: train
    config = MLPAutoencoderTrainConfig(
        output_root=run_dir / "training_runs", run_id="autoencoder", seed=SEED, device=str(device),
        epochs=epochs, batch_size=batch_size, learning_rate=LR, weight_decay=WEIGHT_DECAY,
        eval_batch_size=args.eval_batch_size,
        max_train_batches=args.caps.max_batches if args.smoke else None,
        max_eval_batches=args.caps.max_batches if args.smoke else None,
    )
    fresh_trainer_dir(config.output_root, config.run_id)
    if args.naive:
        provider = {k: neuro_vectors[v] for k, v in split_inds.items()}
        on_device = False
    else:
        placed, on_device = place_splits({k: {"brain": neuro_vectors[v]} for k, v in split_inds.items()}, device, logger=log)
        provider = {k: make_batched_loader(BatchedTensorDataset(**placed[k]), batch_size=batch_size if k == "train" else args.eval_batch_size,
                                           shuffle=k == "train", seed=SEED) for k in split_inds}
    del neuro_vectors

    model = build_mlp_autoencoder(config)
    with timed("train", log, out=args.out, epochs=epochs, batch_size=batch_size, amp=args.amp, on_device=on_device,
               naive=args.naive, device=str(device), **meta) as t:
        with bf16_training_forward(model, device, args.amp), fast_mlp_eval(device, enabled=not args.no_fast_eval):
            result = train_mlp_autoencoder(config, provider=provider, model=model)
    autoencoder = result.model
    log.info("trained %d epochs in %.1fs (%.2fs/epoch); best val loss %.5f; test %s",
             result.epochs_completed, t.seconds, t.seconds / max(result.epochs_completed, 1), result.best_metric,
             {k: round(float(v), 5) for k, v in result.test_metrics.items() if isinstance(v, (int, float))})

    with timed("save_model", log, out=args.out, **meta):
        save_module_and_state(autoencoder, run_dir / "autoencoder", logger=log)

    # cell 9: latents of every kept study, in pmid order
    with timed("encode_latents", log, out=args.out, **meta):
        encoder = autoencoder.to(device).eval().encoder
        if args.naive:
            latent = encode_in_batches(encoder, torch.cat(list(provider.values())), batch_size=4096, device=device)
        else:
            latent = torch.cat([encode_in_batches(encoder, placed[k]["brain"], batch_size=4096, device=device) for k in split_inds])
        latent = latent[np.argsort(np.concatenate([v.numpy() for v in split_inds.values()]))]
        torch.save({"latent": latent.float().cpu(), "pmid": pmids}, outputs["latent_neuro.pt"])
        log.info("latent_neuro %s -> %s", tuple(latent.shape), outputs["latent_neuro.pt"])

    summary = summarise_trainer_run(result, seconds=t.seconds)
    summary.update(n_train=len(train_inds), n_val=len(val_inds), n_test=len(test_inds), amp=args.amp,
                   on_device=on_device, naive=args.naive, device=str(device), epochs=epochs, batch_size=batch_size)
    write_stage_summary(run_dir, STAGE, summary)
    if args.install:
        install_artifacts(outputs, log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
