#!/usr/bin/env python
"""SPECTER2 (adhoc_query) embeddings of PubMed title[SEP]abstract (port of docs/02_data/02_pubmed.ipynb).

    .venv/bin/python scripts/data/02_pubmed.py [--smoke] [--force]
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/
from common import base_parser, finalize_args, get_logger, select_device, setup_torch, timed  # noqa: E402
from data_utils import SPECTER_REPOS, add_data_args, chunk_bounds, encode_texts, hf_cache_pinned, resolve_artifact_dir  # noqa: E402

STAGE = "data_02_pubmed"
log = get_logger(STAGE)


def main(argv=None):
    p = base_parser("SPECTER2 (adhoc_query) embeddings of PubMed title[SEP]abstract.", stage=STAGE)
    add_data_args(p)
    p.add_argument("--shard-size", type=int, default=1024, help="Rows per shard file in <artifact-dir>/specter/.")
    args = finalize_args(p.parse_args(argv))
    art = resolve_artifact_dir(args)
    setup_torch(args.seed)
    import torch
    from neurovlm.data import load_dataset
    from neurovlm.models import Specter

    hf_cache_pinned(*SPECTER_REPOS, "neurovlm/neuro_image_papers")  # the neurovlm loaders below take no revision

    device = select_device(args.device)
    batch_size = args.batch_size or 128
    meta = dict(stage=STAGE, smoke=args.smoke)
    out_path = art / "latent_specter2_adhoc.pt"
    shard_dir = art / "specter"

    with timed("load_datasets", log, out=args.out, **meta):
        df_pubs = load_dataset("pubmed_text")
        if args.max_samples:
            df_pubs = df_pubs.head(args.max_samples)

    # incremental mode (the notebook's overwrite=False): only encode pmids missing from the existing file
    suffix = ""
    if not args.force and out_path.exists():
        latent_exist, pmids_exist = torch.load(out_path, weights_only=False).values()
        mask = ~df_pubs["pmid"].isin(pmids_exist)
        if not mask.any():
            log.info("%s already covers all %d pmids; use --force to recompute", out_path, len(df_pubs))
            return 0
        df_pubs = df_pubs[mask]
        new_pmids = np.sort(df_pubs["pmid"].to_numpy())
        suffix = "_" + hashlib.sha256(new_pmids.tobytes()).hexdigest()[:8]  # deterministic shard suffix
        log.info("incremental mode: %d new pmids (suffix %s)", len(df_pubs), suffix)

    papers = [title + "[SEP]" + abstract for title, abstract in zip(df_pubs["name"], df_pubs["description"])]
    pmids_new = df_pubs["pmid"].to_numpy()
    with timed("load_specter", log, out=args.out, **meta):
        specter = Specter(device=str(device))
    with timed("encode", log, out=args.out, n_docs=len(papers), batch_size=batch_size, amp=args.amp_dtype, device=str(device), **meta):
        latent = encode_texts(specter, papers, batch_size=batch_size, amp_dtype=args.amp_dtype, desc="specter2")

    with timed("write_shards", log, out=args.out, **meta):
        shard_dir.mkdir(parents=True, exist_ok=True)
        for start, end in chunk_bounds(len(papers), args.shard_size):
            torch.save(
                {"embeddings": latent[start:end], "pmid": pmids_new[start:end]},
                shard_dir / f"encoded_text_specter2_adhoc_{start:05d}{suffix}.pt",
                pickle_protocol=5,
            )

    # unit norm, append to the existing file, sort by pmid (cells 4-5)
    with timed("normalise_sort_save", log, out=args.out, **meta):
        latent = latent / torch.norm(latent, dim=1)[:, None]
        pmids = pmids_new.astype(int)
        if suffix:
            latent = torch.vstack((latent_exist, latent))
            pmids = np.concatenate((pmids_exist, pmids))
        inds = np.argsort(pmids)
        torch.save({"latent": latent[inds], "pmid": pmids[inds]}, out_path)
    log.info("saved %s %s", out_path, tuple(latent.shape))
    return 0


if __name__ == "__main__":
    sys.exit(main())
