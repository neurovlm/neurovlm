#!/usr/bin/env python
"""SPECTER2 embeddings of the NeuroWiki articles + the released text projection heads (port of
docs/02_data/04_neurowiki.ipynb). Optional: nothing downstream reads these outputs.

    .venv/bin/python scripts/data/04_neurowiki.py [--smoke] [--force]
"""
from __future__ import annotations

import sys
from hashlib import sha256
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))  # scripts/pipeline/
from common import artifacts_exist, base_parser, finalize_args, get_logger, select_device, setup_torch, timed  # noqa: E402
from data_utils import SPECTER_REPOS, add_data_args, chunk_bounds, encode_texts, hf_cache_pinned, resolve_artifact_dir  # noqa: E402

STAGE = "data_04_neurowiki"
HEADS = ("mse", "infonce")
log = get_logger(STAGE)


def project_heads(latent, heads: dict, device, *, chunk_rows: int = 4096):
    """Notebook cells 6-7: head(emb), unit norm per row, fp32, in row chunks."""
    import torch

    out = {name: torch.empty((len(latent), 384), dtype=torch.float32) for name in heads}
    with torch.inference_mode():
        for start, end in chunk_bounds(len(latent), chunk_rows):
            x = latent[start:end].to(device=device, dtype=torch.float32)
            for name, head in heads.items():
                y = head(x)
                out[name][start:end] = (y / y.norm(dim=1)[:, None]).cpu()
    return out


def main(argv=None):
    p = base_parser("SPECTER2 embeddings of NeuroWiki + projection-head alignment (optional stage).", stage=STAGE)
    add_data_args(p)
    p.add_argument("--shard-size", type=int, default=1024, help="Rows per shard file in <artifact-dir>/specter_wiki/.")
    args = finalize_args(p.parse_args(argv))
    art = resolve_artifact_dir(args)
    setup_torch(args.seed)
    import torch

    device = select_device(args.device)
    batch_size = args.batch_size or 128
    meta = dict(stage=STAGE, smoke=args.smoke)
    latent_path = art / "latent_specter2_wiki.pt"
    aligned_paths = {name: art / f"latent_specter_wiki_aligned_adhoc_{name}.pt" for name in HEADS}
    shard_dir = art / "specter_wiki"
    if artifacts_exist(latent_path, *aligned_paths.values(), force=args.force):
        log.info("artifacts exist (%s, %s); use --force to recompute", latent_path, list(aligned_paths.values()))
        return 0

    from neurovlm.data import load_dataset
    from neurovlm.models import Specter, load_model

    hf_cache_pinned(*SPECTER_REPOS, "neurovlm/neuro_wiki", "neurovlm/ProjectionHeads")  # the neurovlm loaders below take no revision

    with timed("load_dataset", log, out=args.out, **meta):
        df = load_dataset("wiki")
        if args.max_samples:
            df = df.head(args.max_samples).reset_index(drop=True)
        text = list(df["title"] + " [SEP] " + df["summary"])
        ids = [sha256(t.encode("utf-8")).hexdigest()[:8] for t in text]  # replaces the parquet's id column
    log.info("%d wiki articles", len(text))

    # SPECTER2 embeddings, not normalised (cells 4-5); reuse an existing file for the same articles
    latent = None
    if artifacts_exist(latent_path, force=args.force):
        payload = torch.load(latent_path, weights_only=False, map_location="cpu")
        if list(payload["id"]) == ids:
            log.info("reusing %s", latent_path)
            latent = payload["latent"]
        else:
            log.info("%s does not match the current article list; re-encoding", latent_path)
    if latent is None:
        with timed("load_specter", log, out=args.out, **meta):
            specter = Specter(device=str(device))
        with timed("encode", log, out=args.out, n_docs=len(text), batch_size=batch_size, amp=args.amp_dtype, device=str(device), **meta):
            latent = encode_texts(specter, text, batch_size=batch_size, amp_dtype=args.amp_dtype, desc="specter2 wiki")
        del specter
        if device.type == "cuda":
            torch.cuda.empty_cache()
        with timed("write_shards_and_latent", log, out=args.out, **meta):
            shard_dir.mkdir(parents=True, exist_ok=True)
            for start, end in chunk_bounds(len(text), args.shard_size):
                torch.save(
                    {"embeddings": latent[start:end], "id": ids[start:end]},
                    shard_dir / f"encoded_text_specter2_adhoc_{start:04d}.pt",
                    pickle_protocol=5,
                )
            torch.save({"latent": latent, "id": ids}, latent_path)
        log.info("saved %s %s", latent_path, tuple(latent.shape))

    # released projection heads (cells 6-8)
    with timed("load_projection_heads", log, out=args.out, **meta):
        heads = {name: load_model(f"proj_head_text_{name}").to(device).eval() for name in HEADS}
    with timed("project", log, out=args.out, n_docs=len(text), device=str(device), **meta):
        aligned = project_heads(latent, heads, device)
        for name, path in aligned_paths.items():
            torch.save({"latent": aligned[name], "id": ids}, path)
            log.info("saved %s %s", path, tuple(aligned[name].shape))
    log.info("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
