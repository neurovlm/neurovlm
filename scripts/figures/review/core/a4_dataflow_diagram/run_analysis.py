#!/usr/bin/env python
"""Counts for every dataset -> stage edge of the data-flow diagram (comments 1.2, 2.2), from the released data, the split,
the stage 11/12/14 summaries of --model-dir and the leakage-audit exclusion lists (evaluation counts only).

    .venv/bin/python scripts/figures/review/core/a4_dataflow_diagram/run_analysis.py --model-dir scripts/outputs/runs/final --force
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
QWEN_CORPUS_ARTICLES_MODEL_CARD = 1_200_000
# NeuroQwen3-0.6B domain-adaptation corpus (not released): 1,231,613 PubMed titles and abstracts = 1,219,296 train + 12,317 val
# (DAPT notebook log, 05_dapt_pubmed_neuroimaging.ipynb cell 9). Not filtered for NeuroVLM's held-out papers (it holds 2,906 of the
# 2,971 evaluated test papers).
QWEN_DAPT_RECORDS = 1_231_613
REPO = Path(__file__).resolve().parents[5]
EXCLUSIONS_DIR = REPO / "scripts" / "eval_exclusions"


def hf_file(repo_id: str, filename: str) -> Path:
    """``repo_id/filename`` at the revision pinned in scripts/config/hf_manifest.json, sha256-checked (_hf_pins.py)."""
    from _hf_pins import hf_file as pinned

    return pinned(repo_id, filename)


def pinned_local(repo_id: str, path: str) -> Path:
    """A pinned file that scripts/data/fetch_hf.py copies into the data dir (not the HF cache), sha256-checked."""
    from _hf_pins import _manifest, _sha256
    from neurovlm.data import get_data_dir

    e = next(e for e in _manifest()["files"] if e["repo"] == repo_id and e["path"] == path)
    p = Path(get_data_dir()) / e["local_name"]
    if _sha256(str(p), p.stat().st_size, p.stat().st_mtime) != e["sha256"]:
        raise RuntimeError(f"{p} does not match the pinned {repo_id}/{path} (run scripts/data/fetch_hf.py)")
    return p


def torch_zip_pmids(path: Path) -> np.ndarray:
    """Read only the pickled ``pmid`` array of a torch zip checkpoint, skipping the tensor storages."""
    import pickle
    import zipfile

    class _Stub(pickle.Unpickler):
        def persistent_load(self, pid):
            return None

        def find_class(self, mod, name):
            if mod.startswith("torch"):
                return lambda *a, **k: None
            return super().find_class(mod, name)

    with zipfile.ZipFile(path) as z:
        pkl = next(n for n in z.namelist() if n.endswith("data.pkl"))
        obj = _Stub(z.open(pkl)).load()
    return np.asarray(obj["pmid"])


def rel(p: Path) -> str:
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)


def count_everything(timings: dict[str, float], run: Path, exclusions: bool = True) -> dict[str, Any]:
    from _hf_pins import hf_split_pmids

    t0 = time.perf_counter()
    pubs = pd.read_parquet(hf_file("neurovlm/neuro_image_papers", "publications.parquet"))
    coords = pd.read_parquet(hf_file("neurovlm/neuro_image_papers", "coordinates.parquet"), columns=["pmid"])
    timings["pubmed"] = round(time.perf_counter() - t0, 2)
    c: dict[str, Any] = {}
    c["pubmed_publications"] = int(len(pubs))
    c["pubmed_peaks"] = int(len(coords))
    n = coords.groupby("pmid").size()
    c["pubmed_gt100_peaks"] = int((n > 100).sum())
    c["pubmed_specter2_embeddings"] = int(len(pubs))
    t0 = time.perf_counter()
    split = hf_split_pmids()  # split flags of the pinned HF publications.parquet, never the data-dir copy
    c["split_train"], c["split_val"], c["split_test"] = (int(len(split[k])) for k in ("train", "val", "test"))
    img_pmids = torch_zip_pmids(hf_file("neurovlm/embedded_text", "pubmed_images.pt"))
    c["pubmed_maps_nonzero"] = int(len(img_pmids))
    c["pubmed_maps_all_zero_dropped"] = int(len(pubs) - len(img_pmids))
    # maps available to 11_anatomical_atlases.py (all splits; the run's own counts come from stage11_12_counts)
    in_split = {k: set(np.asarray(split[k]).astype(int).tolist()) for k in ("train", "val", "test")}
    ip = np.asarray(img_pmids).astype(int)
    c["synthetic_source_maps_by_split"] = {k: int(np.isin(ip, list(v)).sum()) for k, v in in_split.items()}
    c["synthetic_source_maps_by_split"]["none"] = int(len(ip) - sum(c["synthetic_source_maps_by_split"].values()))
    timings["splits_images"] = round(time.perf_counter() - t0, 2)
    # summaries / QA for the PubMed Q-Former
    qa = pd.read_parquet(hf_file("neurovlm/pubmed_summary_qa", "pubmed_summary_qa.parquet"), columns=["pmid"])
    c["summaries_qa_rows"] = int(len(qa))
    qa_img = qa[qa["pmid"].isin(set(img_pmids.tolist()))]
    flags = pubs.set_index("pmid")[["train", "val", "test"]].reindex(qa_img["pmid"]).fillna(False).astype(bool)
    c["qformer_pubmed_train"], c["qformer_pubmed_val"], c["qformer_pubmed_test"] = (int(flags[k].sum()) for k in ("train", "val", "test"))
    # synthetic pairs and anatomical head (stages 11 / 12), canonical texts and stage 2 rows (stage 14) of the run
    c.update(stage11_12_counts(run))
    tsl = pinned_local("neurovlm/synthetic_pairs", "text_synth_less.parquet")
    c["text_synth_less_rows"] = int(len(pd.read_parquet(tsl, columns=["title"])))
    c["qformer_canonical_rows"] = canonical_rows(tsl)
    c.update(stage2_rows(run))
    # NeuroVault
    rel_imgs = pd.read_parquet(hf_file("neurovlm/embedded_text", "images_neurovault.parquet"), columns=["id", "doi"])
    rel_pubs = pd.read_parquet(hf_file("neurovlm/embedded_text", "publications_neurovault.parquet"), columns=["doi"])
    c["neurovault_maps"] = int(len(rel_imgs))
    c["neurovault_publications"] = int(len(rel_pubs))
    if exclusions:
        apply_eval_exclusions(c, split, rel_imgs, rel_pubs)
    else:
        c["eval_exclusions"] = {"enabled": False}
    img = pd.read_parquet(hf_file("neurovlm/neurovault_maps", "neurovault_images.parquet"), columns=["id", "is_thresholded", "map_type"])
    c["neurovault_scraped"] = int(len(img))
    s = img[(img["is_thresholded"] == False) & ~img["id"].isin([22136, 22137, 22138])]  # noqa: E712
    c["neurovault_unthresholded_stat_maps"] = int((~s["map_type"].isin(["parcellation", "anatomical", "ROI/mask", "variance"])).sum())
    # networks
    labels = pd.read_parquet(hf_file("neurovlm/embedded_text", "networks_labels.parquet"))
    ica = labels["atlas_name"].isin(["HCPICA", "UKBICA"])
    c["network_maps"] = int(len(labels))
    c["network_maps_labelled"] = int((~ica).sum())
    c["network_ica_components"] = int(ica.sum())
    c["network_atlases"] = int(labels["atlas_name"].nunique())
    c["qwen_corpus_articles_model_card"] = QWEN_CORPUS_ARTICLES_MODEL_CARD
    c["qwen_dapt_records"] = QWEN_DAPT_RECORDS
    return c


def apply_eval_exclusions(c: dict[str, Any], split: dict, rel_imgs: pd.DataFrame, rel_pubs: pd.DataFrame) -> None:
    """Add post-exclusion evaluation counts (``*_eval``) and an ``eval_exclusions`` block to ``c``.

    PubMed: excluded PMIDs in the test / val split. NeuroVault: excluded publications (by DOI) and their images (by DOI or id).
    """
    pm = set(pd.read_csv(EXCLUSIONS_DIR / "pubmed_exclusions.csv", usecols=["pmid"])["pmid"].astype(int))
    drop = {k: int(np.isin(np.asarray(split[k]).astype(int), list(pm)).sum()) for k in ("test", "val")}
    nv = pd.read_csv(EXCLUSIONS_DIR / "neurovault_exclusions.csv", usecols=["doi", "image_ids"], dtype=str)
    dois = set(nv["doi"].str.lower().str.strip())
    ids = {int(x) for s in nv["image_ids"].dropna() for x in s.split(";") if x.strip()}
    pub_drop = int(rel_pubs["doi"].astype(str).str.lower().str.strip().isin(dois).sum())
    img_mask = rel_imgs["doi"].astype(str).str.lower().str.strip().isin(dois) | rel_imgs["id"].isin(ids)
    img_drop = int(img_mask.sum())
    c["split_test_eval"] = c["split_test"] - drop["test"]
    c["split_val_eval"] = c["split_val"] - drop["val"]
    c["neurovault_publications_eval"] = c["neurovault_publications"] - pub_drop
    c["neurovault_maps_eval"] = c["neurovault_maps"] - img_drop
    c["eval_exclusions"] = {"enabled": True, "source": str(EXCLUSIONS_DIR.relative_to(REPO)),
                            "pubmed_test_dropped": drop["test"], "pubmed_val_dropped": drop["val"],
                            "neurovault_publications_dropped": pub_drop, "neurovault_maps_dropped": img_drop,
                            "note": "evaluation counts only; the training splits (and the val split used during training) are unchanged"}


def stage11_12_counts(run: Path) -> dict[str, Any]:
    """Synthetic pairs (stage 11) and anatomical-head rows (stage 12) from the run's summaries."""
    f11, f12 = run / "11_anatomical_atlases.summary.json", run / "12_neuroadapter.summary.json"
    s11, s12 = json.loads(f11.read_text()), json.loads(f12.read_text())
    sp = s11.get("splits", {})
    out = {"synth_rows": int(s11["n_synth_rows"]), "adapter_train": int(s12["n_train"]), "adapter_val": int(s12["n_val"]),
           "synthetic_source_split_mode": sp.get("mode", "all"), "stage11_12_source": [rel(f11), rel(f12)]}
    if sp.get("used_by_split"):
        out["synthetic_source_maps_by_split"] = sp["used_by_split"]
    if sp.get("pooled_publications_by_split"):
        out["synthetic_pooled_publications_by_split"] = sp["pooled_publications_by_split"]
    return out


def stage2_rows(run: Path) -> dict[str, Any]:
    """Stage 2 rows kept by the self-consistency filter of 14_qformer_anat.py, from the run's summary."""
    f = run / "14_qformer_anat.summary.json"
    n = json.loads(f.read_text())["n_rows"]
    return {"stage2_rows_kept": int(n["total"]), "stage2_train": int(n["train"]), "stage2_val": int(n["val"]), "stage2_rows_source": rel(f)}


def canonical_rows(tsl: Path) -> int:
    """Rows of the curated stage 2 table, from build_text_table of scripts/models/14_qformer_anat.py."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("stage14", REPO / "scripts" / "models" / "14_qformer_anat.py")
    stage14 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage14)
    return int(len(stage14.build_text_table(tsl)[0]))


def k(v: int | None) -> str:
    return "n/a" if v is None else f"{v:,}"


def eval_counts(c: dict[str, Any]) -> tuple[int, int, int, str]:
    """(PubMed test, NeuroVault publications, NeuroVault maps) as evaluated, and the post-exclusion marker."""
    if c.get("eval_exclusions", {}).get("enabled"):
        return c["split_test_eval"], c["neurovault_publications_eval"], c["neurovault_maps_eval"], "\u2020"
    return c["split_test"], c["neurovault_publications"], c["neurovault_maps"], ""


def exclusion_footnote(c: dict[str, Any]) -> str:
    """Exclusion footnote for the graph (empty without exclusions)."""
    e = c.get("eval_exclusions", {})
    if not e.get("enabled"):
        return ""
    return (f"\u2020 after the leakage-audit evaluation exclusions ({e['source']}/): PubMed test {k(c['split_test'])} -> "
            f"{k(c['split_test_eval'])} (-{e['pubmed_test_dropped']} publications; val {k(c['split_val'])} -> {k(c['split_val_eval'])} for "
            f"evaluation); NeuroVault {k(c['neurovault_publications'])} -> {k(c['neurovault_publications_eval'])} publications and "
            f"{k(c['neurovault_maps'])} -> {k(c['neurovault_maps_eval'])} maps (released counts first). Training splits unchanged.")


def build_dot(c: dict[str, Any]) -> str:
    """Graphviz source of the detailed graph (edges as in scripts/models/)."""
    tr, va, te = c["split_train"], c["split_val"], c["split_test"]
    te_ev, nv_pubs, nv_maps, mk = eval_counts(c)
    s2 = f"{k(c['stage2_rows_kept'])} kept: train {k(c['stage2_train'])} / val {k(c['stage2_val'])}"
    pool_from = "the train split" if c.get("synthetic_source_split_mode") == "train" else "all splits"
    nodes = {
        # sources
        "pubmed": ("PubMed publications\ncoordinate tables + titles/abstracts", f"{k(c['pubmed_publications'])} publications\n{k(c['pubmed_peaks'])} peaks", "source"),
        "atlas": ("AAL atlas + LLM-written concept texts", f"{k(c['synth_rows'])} / {k(c['text_synth_less_rows'])} texts", "source"),
        "external": ("NeuroVault + network atlases", f"{k(nv_pubs)} publications, {k(nv_maps)} maps{mk}\n{k(c['network_maps'])} network maps", "source"),
        # preprocessing
        "prep": ("Maps, embeddings, split", f"{k(c['pubmed_maps_nonzero'])} ALE/DiFuMo maps; SPECTER2; LLM summaries\n"
                 f"train {k(tr)} | val {k(va)} | test {k(te)}", "prep"),
        "synth": ("Synthetic (text, map) pairs", f"{k(c['synth_rows'])} pairs; {k(c['qformer_canonical_rows'])} canonical texts\n"
                  f"parcel maps pooled from {pool_from}", "prep"),
        # models (training order)
        "ae": ("1 Autoencoder", f"train {k(tr)} / val {k(va)}", "train"),
        "heads": ("2 Projection heads\ntext-to-brain, then contrastive", f"train {k(tr)} / val {k(va)}", "train"),
        "anat": ("3 Anatomical projection head\n(NeuroAdapter in the code)", f"train {k(c.get('adapter_train'))} / val {k(c.get('adapter_val'))} synthetic pairs", "train"),
        "stage1": ("4 Stage-1 projector (MLP)\ngrounded on PubMed", f"train {k(c['qformer_pubmed_train'])} / val {k(c['qformer_pubmed_val'])}", "train"),
        "stage2": ("5 Stage-2 projector (MLP)\ncanonical", f"{k(c['qformer_canonical_rows'])} texts; {s2}", "train"),
        "lm": ("NeuroQwen3-0.6B (frozen)", f"adapted on {c['qwen_dapt_records'] / 1e6:.2f} M titles and abstracts (evaluation papers excluded by the training script)", "frozen"),
        # evaluation
        "eval": ("Held-out evaluation sets", f"PubMed test {k(te_ev)}{mk}; NeuroVault {k(nv_pubs)} / {k(nv_maps)}{mk};\n"
                 f"networks {k(c['network_maps_labelled'])} + {k(c['network_ica_components'])} ICA", "test"),
    }
    edges = [
        ("pubmed", "prep", ""), ("atlas", "synth", ""), ("prep", "synth", f"{pool_from.removeprefix('the ')}, pooled per parcel"),
        ("prep", "ae", "train / val"), ("prep", "heads", "train / val"), ("prep", "stage1", "summaries, train / val"),
        ("ae", "heads", "latents, frozen decoder"), ("heads", "anat", "init: text-to-brain head"), ("synth", "anat", "pairs"),
        ("ae", "stage1", "AE latent"), ("heads", "stage1", "contrastive latent"),
        ("stage1", "stage2", "init"), ("anat", "stage2", "synthetic maps"), ("synth", "stage2", "canonical texts"),
        ("heads", "stage2", "contrastive latent"), ("lm", "stage1", "frozen"), ("lm", "stage2", "frozen"),
        ("prep", "eval", f"test {k(te)}"), ("external", "eval", "evaluation only"),
    ]
    fill = {"source": "#cde2fb", "prep": "#d9f2e8", "train": "#ffffff", "frozen": "#eeeeec", "test": "#fbe3ec"}
    foot = exclusion_footnote(c).replace('"', "'")
    lines = ["digraph neurovlm_dataflow {", '  rankdir=LR; nodesep=0.25; ranksep=0.6; fontname="Helvetica"; fontsize=9;',
             *([f'  labelloc=b; labeljust=l; label="{foot}";'] if foot else []),
             '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=9, color="#555555"];',
             '  edge [fontname="Helvetica", fontsize=8, color="#555555"];']
    for nid, (title, sub, kind) in nodes.items():
        label = title.replace("\n", "\\n") + "\\n" + sub.replace("\n", "\\n")
        lines.append(f'  {nid} [label="{label}", fillcolor="{fill[kind]}"];')
    for a, b, lab in edges:
        lines.append(f'  {a} -> {b} [label="{lab}"];' if lab else f"  {a} -> {b};")
    lines.append("}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "outputs", help="Output directory.")
    p.add_argument("--force", action="store_true", help="Recompute even if dataflow_counts.json exists.")
    p.add_argument("--model-dir", type=Path, default=REPO / "scripts" / "outputs" / "runs" / "final",
                   help="Run directory with the stage 11/12/14 summaries (default: runs/final).")
    p.add_argument("--no-exclusions", action="store_true", help="Do not apply scripts/eval_exclusions/.")
    args = p.parse_args(argv)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / "dataflow_counts.json").exists() and not args.force:
        print("counts exist; use --force")
        return 0
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    c = count_everything(timings, args.model_dir.resolve(), exclusions=not args.no_exclusions)
    (out / "dataflow_counts.json").write_text(json.dumps(c, indent=2))
    (out / "a4_dataflow.dot").write_text(build_dot(c))
    timings["total"] = round(time.perf_counter() - t0, 2)
    (out / "timings.json").write_text(json.dumps(timings, indent=2))
    print(json.dumps(c, indent=1))
    print("timings", timings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
