#!/usr/bin/env python
"""Count publications, peaks and splits from the released data, and the per-stage training rows of the final model from its
run directory (split file and stage summaries, --model-dir, default scripts/outputs/runs/final) (comments 1.2, 2.2, 1.m1).

    python run_analysis.py --force     # writes outputs/counts.json, counts_long.csv, stage_splits.csv, coords_*.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

REPO = Path(__file__).resolve().parents[5]
DEFAULT_MODEL_DIR = REPO / "scripts" / "outputs" / "runs" / "final"

PAPER_NUMBERS = {
    "pubmed_pairs": 30826,
    "pubmed_test": 2987,
    "neurovault_maps_main_text": 3000,
    "neurovault_maps_supplement": 4553,
    "network_images_results": 300,
    "network_images_other": 100,
    "qwen_corpus_articles": 1_200_000,
}
# NeuroQwen3-0.6B domain-adaptation corpus (not released): 1,231,613 PubMed titles and abstracts, split at random
# (train_test_split(test_size=0.01, seed=12345)) into 1,219,296 train / 12,317 val. Source: the DAPT notebook's logged split
# (05_dapt_pubmed_neuroimaging.ipynb, cell 9 output; 01_dapt.ipynb cells 2 and 5). The corpus was not filtered for NeuroVLM's
# held-out papers (2,906 of the 2,971 evaluated test papers are in it).
NEUROQWEN_DAPT_SPLIT = {"train": 1_219_296, "val": 12_317}
TIMINGS: dict[str, float] = {}


@contextmanager
def timed(name: str):
    t = time.perf_counter()
    yield
    TIMINGS[name] = round(time.perf_counter() - t, 3)


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


def rel(p: Path) -> str:
    p = Path(p).resolve()
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)


def summary(run: Path, stage: str) -> dict[str, Any]:
    return json.loads((run / f"{stage}.summary.json").read_text())


def quantiles(s: pd.Series) -> dict[str, float]:
    q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_dict()
    out = {f"p{int(k * 100):02d}": float(v) for k, v in q.items()}
    out.update(mean=float(s.mean()), std=float(s.std()), min=float(s.min()), max=float(s.max()), n=int(len(s)))
    return out


def count_pubmed() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    with timed("load_pubmed"):
        pubs = pd.read_parquet(hf_file("neurovlm/neuro_image_papers", "publications.parquet"))
        coords = pd.read_parquet(hf_file("neurovlm/neuro_image_papers", "coordinates.parquet"),
                                 columns=["pmid", "doi", "x", "y", "z"])
    sec: dict[str, Any] = {}
    sec["publications"] = int(len(pubs))
    sec["unique_pmids"] = int(pubs["pmid"].nunique())
    sec["publications_with_doi"] = int(pubs["doi"].notna().sum())
    sec["publications_without_doi"] = int(pubs["doi"].isna().sum())
    sec["unique_dois"] = int(pubs["doi"].nunique())
    vc = pubs["doi"].value_counts()
    sec["dois_with_multiple_pmids"] = int((vc > 1).sum())
    sec["pmids_sharing_a_doi"] = int(vc[vc > 1].sum())
    sec["publications_with_pmcid"] = int(pubs["pmcid"].notna().sum())
    sec["abstract_empty"] = int((pubs["description"].fillna("").str.strip() == "").sum())
    sec["title_empty"] = int((pubs["name"].fillna("").str.strip() == "").sum())
    for col in ("train", "val", "test"):
        sec[f"flag_{col}"] = int(pubs[col].sum())
    sec["flag_none_(excluded_from_splits)"] = int((~(pubs["train"] | pubs["val"] | pubs["test"])).sum())

    sec["coordinate_rows"] = int(len(coords))
    sec["coordinate_pmids"] = int(coords["pmid"].nunique())
    sec["coordinate_pmids_not_in_publications"] = int((~coords["pmid"].isin(pubs["pmid"])).sum())
    sec["publications_without_coordinates"] = int((~pubs["pmid"].isin(coords["pmid"])).sum())
    cd = coords.drop_duplicates("pmid")[["pmid", "doi"]].merge(pubs[["pmid", "doi"]], on="pmid", suffixes=("_coord", "_pub"))
    sec["coordinate_doi_disagrees_with_publication_doi"] = int((cd["doi_coord"].fillna("") != cd["doi_pub"].fillna("")).sum())
    sec["coordinate_rows_non_integer"] = int(((coords["x"] % 1 != 0) | (coords["y"] % 1 != 0) | (coords["z"] % 1 != 0)).sum())
    sec["coordinate_rows_duplicated_within_pmid"] = int(coords.duplicated(["pmid", "x", "y", "z"]).sum())
    n = coords.groupby("pmid").size().rename("n_coords")
    sec["coords_per_publication"] = quantiles(n)
    sec["publications_gt_100_coords_(excluded_by_03_models/08)"] = int((n > 100).sum())
    sec["publications_le_100_coords"] = int((n <= 100).sum())
    sec["experiment_contrast_table_identifiers_in_released_data"] = (
        "none: coordinates.parquet has only pmid/pmcid/doi/x/y/z; experiments, contrasts and tables are "
        "not distinguishable, every peak of a PMID is pooled into one map"
    )
    sec["negative_effect_handling"] = (
        "no sign/direction metadata is released; activation and deactivation peaks (and every contrast/table) "
        "are pooled per PMID and smoothed identically (ALE kernel, 9 mm FWHM, voxel-wise max over foci)"
    )
    sec["aggregation_rule"] = (
        "scripts/data/01_coordinate.py / neurovlm.data.coordinates.coords_to_vectors: one map per PMID = "
        "nimare ALEKernel(fwhm=9) over all of the PMID's foci (max-combined), divided by the kernel peak, "
        "clamped to <=1; then DiFuMo-512 projection X_hat=(X@P/s)@P.T, drop all-zero rows, per-row 0.9999-quantile "
        "ceiling, clamp to [0,1]"
    )
    per_pub = n.reset_index()
    per_pub["doi"] = per_pub["pmid"].map(pubs.set_index("pmid")["doi"])
    for col in ("train", "val", "test"):
        per_pub[col] = per_pub["pmid"].map(pubs.set_index("pmid")[col]).fillna(False).astype(bool)
    return sec, pubs, per_pub


def count_splits(pubs: pd.DataFrame, per_pub: pd.DataFrame, run: Path) -> dict[str, Any]:
    sec: dict[str, Any] = {}
    with timed("load_splits"):
        import torch

        p = run / "pmids_split.pt"
        split = torch.load(p, weights_only=False)
        img_pmids = torch_zip_pmids(hf_file("neurovlm/embedded_text", "pubmed_images.pt"))
        lat_pmids = torch_zip_pmids(hf_file("neurovlm/embedded_text", "latent_neuro.pt"))
    sec["pubmed_images_rows_(released_pubmed_images.pt)"] = int(len(img_pmids))
    sec["pubmed_images_dropped_all_zero_difumo_rows"] = int(len(pubs) - len(img_pmids))
    sec["latent_neuro_rows_(released_latent_neuro.pt)"] = int(len(lat_pmids))
    sec["pmids_split_file"] = rel(p)
    for k in ("train", "val", "test"):
        sec[f"split_{k}"] = int(len(split[k]))
    sec["split_total"] = int(sum(len(split[k]) for k in ("train", "val", "test")))
    allsplit = np.concatenate([split[k] for k in ("train", "val", "test")])
    sec["split_pmid_overlap_train_val_test"] = int(len(allsplit) - len(np.unique(allsplit)))
    agree = all(set(pubs.loc[pubs[k], "pmid"]) == set(split[k].tolist()) for k in ("train", "val", "test"))
    sec["publications_flags_identical_to_pmids_split"] = bool(agree)
    le100 = set(per_pub.loc[per_pub["n_coords"] <= 100, "pmid"])
    sec["split_pmids_all_have_le_100_coords"] = bool(set(allsplit.tolist()) <= le100)
    sec["split_derivation"] = (
        "03_models/08: pmids of neuro_vectors.pt (non-zero DiFuMo maps) with <=100 coordinates; "
        "sklearn train_test_split(train_size=0.8, random_state=0) then 50/50 (random_state=1) -> 80/10/10"
    )
    nz_le100 = np.intersect1d(img_pmids, np.array(sorted(le100)))
    sec["eligible_pmids_(nonzero_map_and_le_100_coords)"] = int(len(nz_le100))
    return sec


def count_stages(run: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    """Rows per training stage x split of the final model: its pmids_split.pt and stage summaries (*.summary.json)."""
    import torch

    rows: list[dict[str, Any]] = []
    split = torch.load(run / "pmids_split.pt", weights_only=False)
    n_tr, n_va, n_te = (len(split[k]) for k in ("train", "val", "test"))
    s08, s09, s11, s12, s13, s14 = (summary(run, k) for k in ("08_autoencoder", "09_projection_head", "11_anatomical_atlases",
                                                                "12_neuroadapter", "13_qformer_pubmed", "14_qformer_anat"))
    notes: dict[str, Any] = {"model_dir": rel(run), "sources": [rel(run / "pmids_split.pt")] + [
        rel(run / f"{k}.summary.json") for k in ("08_autoencoder", "09_projection_head", "11_anatomical_atlases", "12_neuroadapter",
                                                 "13_qformer_pubmed", "14_qformer_anat")]}

    def add(stage, script, data, train, val, test, unit, note=""):
        rows.append(dict(stage=stage, script=script, data=data, train=int(train), val=int(val), test=int(test),
                         total=int(train) + int(val) + int(test), unit=unit, note=note))

    add("Neuroimage autoencoder", "08_autoencoder", "PubMed coordinate maps (ALE->DiFuMo)", n_tr, n_va, n_te,
        "publications (one map each)", f"pmids_split.pt; {s08['epochs_completed']} epochs, BCE")
    t2b, nr = s09["text_to_brain"], s09["n_rows"]
    add("Text-to-brain head (head v2)", "09_projection_head", "SPECTER2(title[SEP]abstract) -> PubMed maps", t2b["n_train"], nr["val"], nr["test"],
        "publications", f"same split; early stopping on {t2b['n_val']:,} val papers (leakage-audit exclusions), best epoch {t2b['best_epoch']}")
    add("Contrastive projection heads (InfoNCE)", "09_projection_head", "SPECTER2 text x autoencoder image latents", nr["train"], nr["val"], nr["test"],
        "publications", "same split; text head initialised from the stage A (BCE) head")
    sp = s11["splits"]
    add("Anatomically centred projection head (NeuroAdapter in the code; internal)", "11_anatomical_atlases, 12_neuroadapter",
        "synthetic concept texts -> AAL parcel maps pooled from " + ("training-split papers only" if sp["mode"] == "train" else f"{sp['mode']} papers"),
        s12["n_train"], s12["n_val"], 0,
        "synthetic (text, map) pairs", f"{s11['n_synth_rows']:,} pairs from {sp['n_docs']:,} papers; 99/1 random split; "
        f"{s12['recipe']} recipe, best epoch {s12['best_epoch']}")
    notes.update(synth_rows=int(s11["n_synth_rows"]), synth_source_split_mode=sp["mode"], synth_source_maps_by_split=sp["used_by_split"])
    nr = s13["n_rows"]
    add(f"Stage 1 projector ({s13['projector'].upper()}; grounded, PubMed)", "13_qformer_pubmed",
        "autoencoder latents x LLM summaries (pubmed_summary_qa)", nr["train"], nr["val"], nr["test"], "publications",
        f"best epoch {s13['best_epoch']} of {s13['epochs_trained']} (early stopping)")
    qa = pd.read_parquet(hf_file("neurovlm/pubmed_summary_qa", "pubmed_summary_qa.parquet"), columns=["pmid"])
    notes["pubmed_summary_qa_rows"] = int(len(qa))
    curated = stage2_curated_counts(pinned_local("neurovlm/synthetic_pairs", "text_synth_less.parquet"))
    n_cur, nr = int(sum(curated.values())), s14["n_rows"]
    add(f"Stage 2 projector ({s14['projector'].upper()}; canonical)", "14_qformer_anat",
        f"[NETWORK]/[REGION]/[FUNCTION] texts x latents from the anatomically centred head ({nr['total']:,} of {n_cur:,} kept by the "
        "self-consistency filter)", nr["train"], nr["val"], 0, "synthetic texts",
        f"seed {s14['seed']}; synthesised maps calibrated ({s14['map_calibration']}) and binarised at the {s14['synth_binarize_pct']:g}th percentile")
    notes.update(stage2_curated_rows=n_cur, stage2_curated_rows_by_label=curated, stage2_rows_kept=int(nr["total"]),
                 stage2_bank_sizes=s14["bank_sizes"])
    add("LLM domain adaptation (NeuroQwen3-0.6B)", "external (DAPT notebook log)",
        "1.23 M PubMed titles and abstracts, test papers included",
        NEUROQWEN_DAPT_SPLIT["train"], NEUROQWEN_DAPT_SPLIT["val"], 0, "articles",
        "corpus not released; random 1% validation split, seed 12345; not filtered for the held-out papers")
    return notes, pd.DataFrame(rows)


def stage2_curated_counts(tsl_path: Path) -> dict[str, int]:
    """Curated stage 2 rows per label, from build_text_table of scripts/models/14_qformer_anat.py."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("stage14", REPO / "scripts" / "models" / "14_qformer_anat.py")
    stage14 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage14)
    df, _ = stage14.build_text_table(tsl_path)
    labels = df["text"].str.split("]").str[0].str.strip("[").value_counts()
    return {str(k): int(v) for k, v in labels.items()}


def count_neurovault(pubs: pd.DataFrame) -> dict[str, Any]:
    sec: dict[str, Any] = {}
    with timed("load_neurovault"):
        rel_imgs = pd.read_parquet(hf_file("neurovlm/embedded_text", "images_neurovault.parquet"))
        rel_pubs = pd.read_parquet(hf_file("neurovlm/embedded_text", "publications_neurovault.parquet"))
    sec["released_images_(images_neurovault.parquet)"] = int(len(rel_imgs))
    sec["released_publications_(publications_neurovault.parquet)"] = int(len(rel_pubs))
    sec["released_unique_dois_in_images"] = int(rel_imgs["doi"].nunique())
    sec["released_unique_collections"] = int(rel_imgs["collection_id"].nunique())
    sec["released_images_per_publication"] = quantiles(rel_imgs.groupby("doi").size())
    sec["released_publication_dois_in_pubmed_any_split"] = int(rel_pubs["doi"].str.lower().isin(pubs["doi"].dropna().str.lower()).sum())
    sec["released_contrast_definition_empty_pct"] = round(100 * float((rel_imgs["contrast_definition"].fillna("").str.strip() == "").mean()), 1)

    # NeuroVault scrape (input to scripts/data/03_neurovault)
    img = pd.read_parquet(hf_file("neurovlm/neurovault_maps", "neurovault_images.parquet"))
    col = pd.read_parquet(hf_file("neurovlm/neurovault_maps", "neurovault_collections.parquet"))
    sec["scrape_images"] = int(len(img))
    sec["scrape_collections"] = int(img["collection_id"].nunique())
    step1 = img[img["is_thresholded"] == False]  # noqa: E712
    sec["scrape_unthresholded_images"] = int(len(step1))
    step2 = step1[~step1["id"].isin([22136, 22137, 22138])]
    step2 = step2[~step2["map_type"].isin(["parcellation", "anatomical", "ROI/mask", "variance"])]
    sec["scrape_after_map_type_filter_(input_to_03_neurovault)"] = int(len(step2))
    sec["scrape_after_map_type_filter_by_map_type"] = {str(k): int(v) for k, v in step2["map_type"].value_counts().items()}
    sec["released_images_by_map_type"] = {str(k): int(v) for k, v in img.set_index("id").reindex(rel_imgs["id"])["map_type"].value_counts().items()}
    sec["scrape_collections_with_doi"] = int(col["DOI"].notna().sum())
    sec["scrape_collections_with_preprint_doi"] = int(col["preprint_DOI"].notna().sum())
    nv_pubs = pinned_local("neurovlm/neurovault_maps", "derived/neurovault_publications.parquet")
    sec["local_neurovault_publications_(before_pubmed_doi_exclusion)"] = int(len(pd.read_parquet(nv_pubs, columns=["doi"])))
    sec["paper_numbers"] = {
        "main_text_'3,000 group-level activation maps'": "~= 3,183 released maps (312 publications) used for contrastive/generative evaluation",
        "supplement_'n=4553 images'": "not reproducible from released metadata: lies between the 4,611 unthresholded statistic maps that enter "
                                       "03_neurovault and the 3,183 maps that survive loading/unique-value/cluster/>=10 %-active/PubMed-DOI filters; "
                                       "recommend reporting 3,183 maps / 312 publications (autoencoder: all 3,183 maps; contrastive: 312 pairs)",
    }
    return sec


def count_networks() -> dict[str, Any]:
    sec: dict[str, Any] = {}
    with timed("load_networks"):
        labels = pd.read_parquet(hf_file("neurovlm/embedded_text", "networks_labels.parquet"))
        test_labels = pd.read_csv(hf_file("neurovlm/embedded_text", "network_test_set_labels.csv"))
        net_text = pd.read_parquet(hf_file("neurovlm/embedded_text", "network_text.parquet"))
    per_atlas = labels["atlas_name"].value_counts()
    sec["network_maps_total_(networks_labels.parquet)"] = int(len(labels))
    sec["atlases"] = int(per_atlas.size)
    sec["maps_per_atlas"] = {str(k): int(v) for k, v in per_atlas.items()}
    ica = labels["atlas_name"].isin(["HCPICA", "UKBICA"])
    sec["labelled_network_maps_(non-ICA,_used_for_Fig5_confusion)"] = int((~ica).sum())
    sec["unlabelled_ICA_components_(HCP+UKB)"] = int(ica.sum())
    sec["labelled_maps_with_known_standard_label"] = int(((~ica) & (labels["network_standard"] != "unknown")).sum())
    sec["standard_network_classes"] = int(labels.loc[~ica & (labels["network_standard"] != "unknown"), "network_standard"].nunique())
    sec["network_test_set_labels_rows"] = int(len(test_labels))
    sec["canonical_network_texts_(network_text.parquet)"] = int(len(net_text))
    sec["paper_numbers"] = {
        "results_'300 network images'": "not reproducible from the released data: 185 maps in total (145 labelled + 40 ICA components)",
        "'100 network images'": "not reproducible either; closest released counts are 145 labelled maps (8 atlases) or 130 label rows",
    }
    return sec


def flatten(d: dict[str, Any], prefix: str = "") -> list[tuple[str, str, Any]]:
    """(group, item, value) rows; small dicts like per-atlas counts stay one group."""
    out = []
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and all(not isinstance(x, dict) for x in v.values()) and k.endswith(("per_publication", "per_atlas", "by_map_type", "by_label", "paper_numbers")):
            for kk, vv in v.items():
                out.append((key, str(kk), vv))
        elif isinstance(v, dict):
            out.extend(flatten(v, prefix=f"{key}."))
        else:
            out.append((prefix.rstrip("."), str(k), v))
    return out


def write_outputs(out: Path, counts: dict[str, Any], stages: pd.DataFrame, per_pub: pd.DataFrame) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "counts.json").write_text(json.dumps(counts, indent=2, default=str))
    rows = []
    for section, sub in counts.items():
        for sec_key, item, val in flatten(sub):
            rows.append(dict(section=section, group=sec_key, item=item, value=val if not isinstance(val, (dict, list)) else json.dumps(val)))
    pd.DataFrame(rows).to_csv(out / "counts_long.csv", index=False)
    stages.to_csv(out / "stage_splits.csv", index=False)
    per_pub.to_csv(out / "coords_per_publication.csv", index=False)
    bins = np.r_[0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, np.inf]
    hist = pd.cut(per_pub["n_coords"], bins=bins, right=True).value_counts().sort_index()
    pd.DataFrame({"bin": hist.index.astype(str), "n_publications": hist.values}).to_csv(out / "coords_hist.csv", index=False)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "outputs", help="Output directory.")
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Run directory of the final model (split file, stage summaries).")
    p.add_argument("--force", action="store_true", help="Recompute even if counts.json exists.")
    args = p.parse_args(argv)
    out: Path = args.out
    if (out / "counts.json").exists() and not args.force:
        print(f"{out / 'counts.json'} exists; use --force to recompute")
        return 0
    t0 = time.perf_counter()
    counts: dict[str, Any] = {"paper_numbers_checked": PAPER_NUMBERS}
    pub_sec, pubs, per_pub = count_pubmed()
    counts["pubmed"] = pub_sec
    run = args.model_dir.resolve()
    counts["splits"] = count_splits(pubs, per_pub, run)
    with timed("stages"):
        notes, stages = count_stages(run)
    counts["stage_notes"] = notes
    counts["neurovault"] = count_neurovault(pubs)
    counts["networks"] = count_networks()
    counts["timings_s"] = TIMINGS
    TIMINGS["total"] = round(time.perf_counter() - t0, 3)
    write_outputs(out, counts, stages, per_pub)
    (out / "timings.json").write_text(json.dumps(TIMINGS, indent=2))
    print(json.dumps({k: v for k, v in counts["pubmed"].items() if not isinstance(v, (dict, str))}, indent=1))
    print(stages[["stage", "train", "val", "test", "total"]].to_string(index=False))
    print("timings:", TIMINGS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
