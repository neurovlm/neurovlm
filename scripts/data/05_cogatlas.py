#!/usr/bin/env python
"""SPECTER2 embeddings of the 300 most central Cognitive Atlas concepts (port of docs/02_data/05_cogatlas.ipynb).
Optional: nothing downstream reads latent_cogatlas.pt (the released one is used instead).

    .venv/bin/python scripts/data/05_cogatlas.py [--smoke] [--force]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))  # scripts/pipeline/
from common import artifacts_exist, base_parser, finalize_args, get_logger, select_device, setup_torch, timed  # noqa: E402
from data_utils import SPECTER_REPOS, add_data_args, encode_texts, hf_cache_pinned, resolve_artifact_dir  # noqa: E402

STAGE = "data_05_cogatlas"
TOP_K = 300
DROP_TERMS = ["active cognitive inhibition", "transfer data"]  # bad definitions
log = get_logger(STAGE)


def select_terms(df, df_graph, *, k: int = TOP_K):
    """Notebook cell 2: term[SEP]definition for the k nodes with the highest degree centrality (child -> parent graph)."""
    import networkx as nx

    df["definition"] = df["definition"].str.replace("\n", "").replace("\r", "")
    df["term"] = df["term"].str.lower()
    df = df[~df["term"].isin(DROP_TERMS)]
    df_graph = df_graph[(~df_graph["parent"].isin(DROP_TERMS)) & (~df_graph["child"].isin(DROP_TERMS))]

    G = nx.DiGraph()
    for _, row in df_graph.iterrows():
        G.add_edge(str(row["child"]).strip(), str(row["parent"]).strip())
    centrality = nx.degree_centrality(G)
    top_nodes = sorted(centrality, key=centrality.get, reverse=True)[:k]

    df = df[df["term"].isin(top_nodes)]
    return (df["term"] + "[SEP]" + df["definition"]).tolist()


def main(argv=None):
    p = base_parser("SPECTER2 embeddings of the top-300 Cognitive Atlas concepts (optional stage).", stage=STAGE)
    add_data_args(p)
    p.add_argument("--top-k", type=int, default=TOP_K, help="Most central graph nodes to keep.")
    args = finalize_args(p.parse_args(argv))
    art = resolve_artifact_dir(args)
    setup_torch(args.seed)
    import torch

    device = select_device(args.device)
    batch_size = args.batch_size or 128
    meta = dict(stage=STAGE, smoke=args.smoke)
    out_path = art / "latent_cogatlas.pt"
    if artifacts_exist(out_path, force=args.force):
        log.info("artifact exists (%s); use --force to recompute", out_path)
        return 0

    from neurovlm.data import load_dataset
    from neurovlm.models import Specter

    hf_cache_pinned(*SPECTER_REPOS, "neurovlm/cognitive_atlas")  # the neurovlm loaders below take no revision

    with timed("load_and_select", log, out=args.out, **meta):
        text = select_terms(load_dataset("cogatlas"), load_dataset("cogatlas_graph"), k=args.top_k)
        if args.max_samples:
            text = text[: args.max_samples]
    log.info("%d terms selected (top-%d by degree centrality)", len(text), args.top_k)

    with timed("load_specter", log, out=args.out, **meta):
        specter = Specter(device=str(device))
    with timed("encode", log, out=args.out, n_docs=len(text), batch_size=batch_size, amp=args.amp_dtype, device=str(device), **meta):
        latent = encode_texts(specter, text, batch_size=batch_size, amp_dtype=args.amp_dtype, desc="specter2 cogatlas")
    torch.save(latent, out_path)  # bare tensor, no term ids (as the notebook)
    log.info("saved %s %s", out_path, tuple(latent.shape))
    return 0


if __name__ == "__main__":
    sys.exit(main())
