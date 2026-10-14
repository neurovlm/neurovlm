#!/usr/bin/env python
"""Build the NeuroVault evaluation set: cleaned abstracts, resampled + cluster-thresholded maps and SPECTER2 text
embeddings (port of docs/02_data/03_neurovault.ipynb). Reads the NeuroVault scrape from --input-dir (default data_dir).

    .venv/bin/python scripts/data/03_neurovault.py [--smoke] [--force] [--download-images]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))  # scripts/pipeline/
from common import artifacts_exist, base_parser, data_dir, finalize_args, get_logger, select_device, setup_torch, timed  # noqa: E402
from data_utils import SPECTER_REPOS, add_data_args, encode_texts, hf_cache_pinned, resolve_artifact_dir  # noqa: E402

STAGE = "data_03_neurovault"
N_VOX = 28542
MISSING_IMAGE_IDS = [22136, 22137, 22138]
DROP_MAP_TYPES = ["parcellation", "anatomical", "ROI/mask", "variance"]
log = get_logger(STAGE)


def rescrape(input_dir: Path):
    """The notebook's rescrape=True branch: NeuroVault collections/images + CrossRef abstracts (slow, network)."""
    import pickle

    import requests
    from tqdm.auto import tqdm

    offsets = list(range(0, 16000, 100))
    db = [requests.get(f"https://neurovault.org/api/collections/?offset={i}") for i in tqdm(offsets, desc="collections")]
    out = [j for i in db for j in i.json()["results"]]
    filtered_out = [i for i in out if i["number_of_images"] > 0 and i["DOI"] is not None]
    with open(input_dir / "neurovault_collections.pkl", "wb") as f:
        pickle.dump(filtered_out, f)

    meta_images = []
    for i in tqdm(filtered_out, desc="images"):
        meta_images.extend(requests.get(f"https://neurovault.org/api/collections/{i['id']}/images/").json()["results"])
    with open(input_dir / "neurovault_images.pkl", "wb") as f:
        pickle.dump(meta_images, f)

    df_collections = pd.DataFrame(filtered_out)
    df_images = pd.DataFrame(meta_images).iloc[:, :67]
    df_images = df_images[df_images["analysis_level"] == "group"].drop_duplicates("id")
    df_collections = df_collections[df_collections["id"].isin(df_images["collection_id"])]
    df_collections.to_parquet(input_dir / "neurovault_collections.parquet")
    df_images.to_parquet(input_dir / "neurovault_images.parquet")

    titles, abstracts = [], []
    for doi in tqdm(df_collections["DOI"], desc="crossref"):
        data = requests.get(f"https://api.crossref.org/works/{doi}").json()
        titles.append(data["message"]["title"][0])
        abstracts.append(data["message"].get("abstract", "No abstract available"))
    df_text = pd.DataFrame(dict(doi=df_collections["DOI"], title=titles, abstract=abstracts))
    df_text["abstract"] = df_text["abstract"].replace("No abstract available", pd.NA)
    df_text.to_parquet(input_dir / "neurovault_abstracts.parquet")


def download_images(df_images: pd.DataFrame, img_dir: Path, *, n_jobs: int):
    """Fetch missing neurovault_img_{id}.nii.gz from the 'file' URLs (not in the notebook)."""
    from joblib import Parallel, delayed

    def _get(img_id, url):
        import requests

        target = img_dir / f"neurovault_img_{img_id}.nii.gz"
        if target.exists() or not isinstance(url, str) or not url:
            return 0
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            target.write_bytes(r.content)
            return 1
        except Exception as exc:  # user-uploaded files
            log.warning("download failed for %s: %s", img_id, exc)
            return 0

    img_dir.mkdir(parents=True, exist_ok=True)
    missing = [(i, u) for i, u in zip(df_images["id"], df_images["file"]) if not (img_dir / f"neurovault_img_{i}.nii.gz").exists()]
    log.info("%d images missing locally; downloading", len(missing))
    got = Parallel(n_jobs=8 if n_jobs < 0 else n_jobs, backend="threading")(delayed(_get)(i, u) for i, u in missing)
    log.info("downloaded %d files", sum(got))


def clean_abstracts(df):
    """Regex clean-up of the scraped abstracts (notebook cell 5)."""
    a = df["abstract"].str.replace(r"</?[^>]+?>", "", regex=True)
    a = a.str.replace("\n", " ").str.replace(r"\s+", " ", regex=True).str.strip(" ").str.strip("\t")
    a = a.str.replace(r"^ ", "", regex=True).str.replace(r"^\t", "", regex=True)
    for header in ("Abstract", "ABSTRACT", "Significance", "Background", "Objective"):
        a = a.str.replace(rf"^{header}\s*", "", regex=True)
    a = a.str.replace("BACKGROUND AND OBJECTIVES: ", "")
    a = a.str.replace("( ", "(").replace(" )", ")")
    a = a.str.replace("’", "'").str.replace("&amp", "&")
    a = a.str.replace("&lt;b&gt;&lt;i&gt;Introduction:&lt;/i&gt;&lt;/b&gt; ", "")
    df["abstract"] = a
    return df


def process_one(img_id, img_dir: Path, masker):
    """Notebook cell 9: load, NaN -> 0, skip near-binary maps, resample, mask, clip negatives."""
    import nibabel as nib
    from nilearn.image import resample_to_img

    try:
        img = nib.load(img_dir / f"neurovault_img_{img_id}.nii.gz")
        img_arr = img.get_fdata()  # some maps have nan outside the brain
        mask = ~np.isfinite(img_arr)
        if np.all(mask):
            print(f"All NaN: {img_id}")
        img_arr[mask] = 0
        if len(np.unique(img_arr.flatten())) < 10:  # needs to be continuous, not binary or integer
            return None
        img = nib.Nifti1Image(img_arr, img.affine)
        arr = masker.transform(resample_to_img(img, masker.mask_img, interpolation="nearest", force_resample=True, copy_header=True))
        arr[arr < 0] = 0
    except Exception:
        # arbitrary user-uploaded files: no clean way to guarantee a valid nii.gz
        return None
    return arr


def _process_chunk(idx_ids, img_dir, masker):
    return [(idx, process_one(img_id, img_dir, masker)) for idx, img_id in idx_ids]


def cluster(arr, masker):
    """Notebook cell 11: threshold at the 99th percentile, drop clusters < 50 voxels."""
    from nilearn.image import threshold_img

    thr_img = threshold_img(masker.inverse_transform(arr), "99%", cluster_threshold=50, two_sided=False, copy_header=True)
    return masker.transform(thr_img).astype(np.float32, copy=False)


def _cluster_chunk(idx_arrs, masker):
    return [(i, cluster(arr, masker)) for i, arr in idx_arrs]


def _chunks(items, n_chunks):
    return [c for c in np.array_split(np.arange(len(items)), max(1, min(n_chunks, len(items)))) if len(c)]


def main(argv=None):
    p = base_parser("Build the NeuroVault evaluation set (maps + text embeddings).", stage=STAGE)
    add_data_args(p)
    p.add_argument("--n-jobs", type=int, default=-1, help="joblib workers (-1 = all cores).")
    p.add_argument("--input-dir", type=Path, default=None, help="NeuroVault parquets/csv and neurovault_images/ (default: data_dir).")
    p.add_argument("--rescrape", action="store_true", help="Re-scrape NeuroVault + CrossRef metadata into --input-dir (slow).")
    p.add_argument("--download-images", action="store_true", help="Download missing nii.gz files listed in neurovault_images.parquet.")
    args = finalize_args(p.parse_args(argv))
    art = resolve_artifact_dir(args)
    input_dir = Path(args.input_dir) if args.input_dir else data_dir()
    img_dir = input_dir / "neurovault_images"
    cache_dir = art / "neurovault_cache"
    setup_torch(args.seed)
    import torch

    device = select_device(args.device)
    batch_size = args.batch_size or 64
    n_workers = args.n_jobs if args.n_jobs > 0 else (os.cpu_count() or 1)
    meta = dict(stage=STAGE, smoke=args.smoke)
    pubs_path, clust_path, bundle_path = art / "neurovault_publications.parquet", art / "neuro_clust.pt", art / "neurovault.pt"
    if artifacts_exist(pubs_path, clust_path, bundle_path, force=args.force):
        log.info("artifacts exist (%s); use --force to recompute", bundle_path)
        return 0

    from joblib import Parallel, delayed
    from neurovlm.data import load_dataset, load_masker
    from neurovlm.models import Specter

    hf_cache_pinned(*SPECTER_REPOS, "neurovlm/neuro_image_papers", "neurovlm/NeuroAutoEncoder")  # the neurovlm loaders below take no revision

    # metadata (cell 3)
    with timed("load_metadata", log, out=args.out, **meta):
        if args.rescrape:
            rescrape(input_dir)
        df_collections = pd.read_parquet(input_dir / "neurovault_collections.parquet")
        df_text = pd.read_parquet(input_dir / "neurovault_abstracts.parquet")  # scraped from crossref
        df_missing = pd.read_csv(input_dir / "neurovault_abstracts_missing.csv")  # missing from crossref, scraped by hand

    # publications (cells 5-6)
    with timed("clean_abstracts", log, out=args.out, **meta):
        df_text = clean_abstracts(df_text)
        df_missing = clean_abstracts(df_missing)
        df_missing = df_missing.merge(df_collections[["DOI", "name"]].rename(columns={"DOI": "doi", "name": "title"}), on="doi", how="left")
        pd.concat((df_text, df_missing)).to_parquet(pubs_path)

    # image table (cell 8)
    df_collections = df_collections.rename(columns=dict(id="collection_id"))
    df_images = pd.read_parquet(input_dir / "neurovault_images.parquet")
    df_images = df_images[df_images["is_thresholded"] == False]  # noqa: E712
    df_images = df_images[~df_images["id"].isin(MISSING_IMAGE_IDS)]
    df_images = df_images[~df_images["map_type"].isin(DROP_MAP_TYPES)]
    df = df_images[["id", "collection_id", "contrast_definition"]].merge(df_collections[["name", "collection_id"]], on="collection_id", how="left")
    if args.max_samples:
        df = df.head(args.max_samples)
    if args.download_images:
        with timed("download_images", log, out=args.out, **meta):
            download_images(df_images[df_images["id"].isin(df["id"])], img_dir, n_jobs=args.n_jobs)
    log.info("%d candidate images", len(df))

    # resample + mask (cell 9), cached; the df.csv round trip is kept so df_neuro matches the notebook
    masker = load_masker()
    cache_dir.mkdir(parents=True, exist_ok=True)
    neuro_cache, df_cache = cache_dir / "neuro.pt", cache_dir / "df.csv"
    if artifacts_exist(neuro_cache, df_cache, force=args.force) and len(pd.read_csv(df_cache)) == len(df):
        log.info("reusing cached %s", neuro_cache)
    else:
        with timed("resample_images", log, out=args.out, n_images=len(df), n_jobs=args.n_jobs, **meta):
            ids = df["id"].to_numpy()
            results = Parallel(n_jobs=n_workers, backend="loky")(
                delayed(_process_chunk)([(int(i), int(ids[i])) for i in c], img_dir, masker) for c in _chunks(ids, n_workers * 8)
            )
            neuro = torch.zeros((len(df), N_VOX), dtype=torch.float32)
            for chunk in results:
                for idx, arr in chunk:
                    if arr is not None:
                        neuro[idx] = torch.from_numpy(np.asarray(arr, dtype=np.float32)).reshape(-1)
            torch.save(neuro, neuro_cache)
            df.to_csv(df_cache)

    # drop empty and near-binary maps (cell 10)
    neuro = torch.load(neuro_cache)
    df = pd.read_csv(df_cache)
    mask = ~((neuro == 0).all(axis=1)).numpy()
    df, neuro = df[mask], neuro[mask]
    mask = np.array([len(np.unique(i)) > 50 for i in neuro], dtype=bool)
    df, neuro = df[mask], neuro[mask]
    log.info("after empty/binary filter: %d maps", len(df))

    # cluster threshold (cell 11), cached
    clust_cache = cache_dir / "neuro_clustered.npy"
    if artifacts_exist(clust_cache, force=args.force) and np.load(clust_cache, mmap_mode="r").shape[0] == len(neuro):
        log.info("reusing cached %s", clust_cache)
    else:
        with timed("cluster_threshold", log, out=args.out, n_images=len(neuro), n_jobs=args.n_jobs, **meta):
            neuro_cp = neuro.detach().cpu().numpy()
            neuro_cp[neuro_cp < 0] = 0
            results = Parallel(n_jobs=n_workers, backend="loky")(
                delayed(_cluster_chunk)([(int(i), neuro_cp[i]) for i in c], masker) for c in _chunks(neuro_cp, n_workers * 8)
            )
            neuro_clustered = np.zeros((len(neuro), N_VOX), dtype=np.float32)
            for chunk in results:
                for i, vec in chunk:
                    neuro_clustered[i] = vec
            np.save(clust_cache, neuro_clustered)
    neuro_clustered = np.load(clust_cache)

    # drop maps active in >= 10% of voxels, then all-zero maps (cell 13)
    mask = (neuro_clustered != 0).mean(axis=1) < 0.1
    df = df[mask].reset_index(drop=True)
    neuro_clustered, neuro = neuro_clustered[mask], neuro[mask]
    mask = ~(neuro_clustered == 0).all(axis=1)
    neuro_clustered, neuro = neuro_clustered[mask], neuro[mask]
    df = df[mask].reset_index(drop=True)
    log.info("after cluster filters: %d maps", len(df))

    # drop collections whose DOI is a PubMed training paper (cell 15)
    with timed("filter_pubmed_dois", log, out=args.out, **meta):
        df_pubmed = load_dataset("pubmed_text")
        df_collections_filt = df_collections[df_collections["collection_id"].isin(df["collection_id"].unique())]
        df_collections_filt = df_collections_filt[~df_collections_filt["DOI"].isin(df_pubmed["doi"])]
        mask = df["collection_id"].isin(df_collections_filt["collection_id"]).to_numpy()
        df, neuro, neuro_clustered = df[mask], neuro[mask], neuro_clustered[mask]
        df_pubs = pd.read_parquet(pubs_path)
        df_pubs = df_pubs[df_pubs["doi"].isin(df_collections_filt[df_collections_filt["collection_id"].isin(df["collection_id"])]["DOI"])]
    log.info("final: %d maps, %d publications", len(df), len(df_pubs))
    torch.save(neuro_clustered, clust_path)

    # text embeddings (cell 18)
    with timed("load_specter", log, out=args.out, **meta):
        specter = Specter(device=str(device))
    titles, abstracts = df_pubs["title"], df_pubs["abstract"]
    with timed("encode_text", log, out=args.out, n_docs=len(titles), amp=args.amp_dtype, device=str(device), **meta):
        kw = dict(batch_size=batch_size, amp_dtype=args.amp_dtype)
        text_emb_titles = encode_texts(specter, titles.tolist(), desc="titles", **kw)
        text_emb_abstracts = encode_texts(specter, abstracts.tolist(), desc="abstracts", **kw)
        text_emb_titles_abstracts = encode_texts(specter, (titles + "[SEP]" + abstracts).tolist(), desc="title[SEP]abstract", **kw)

    # bundle (cell 19)
    with timed("save_bundle", log, out=args.out, **meta):
        torch.save({
            "df_neuro": df.merge(df_collections_filt[["collection_id", "DOI"]].rename(columns={"DOI": "doi"}), on="collection_id", how="left"),
            "df_pubs": df_pubs,
            "neuro": neuro,
            "neuro_clustered": neuro_clustered,
            "text_emb_titles": text_emb_titles,
            "text_emb_abstracts": text_emb_abstracts,
            "text_emb_titles_abstracts": text_emb_titles_abstracts,
        }, bundle_path)
    log.info("saved %s", bundle_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
