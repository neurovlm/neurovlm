# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: .conda
#     language: python
#     name: python3
# ---

# %%
import requests
from tqdm.notebook import tqdm
import pickle
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import torch

import nibabel as nib
from nilearn.image import resample_to_img, threshold_img

from neurovlm.models import Specter
from neurovlm.data import data_dir, load_masker, load_dataset
def select_device() -> str:
    """Prefer Apple MPS, then CUDA, and otherwise use the CPU."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

masker = load_masker()

# %% [markdown]
# # Neurovault
#
# This notebook fetches and processes neurovault data. This notebook is slow since it fetches lots of data.

# %% [markdown]
# ## Fetch
# Pull images and metadata from neurovault.

# %%
rescrape = False

if rescrape:

    # Scrape neurovault, this is slow
    db = []
    offsets = list(range(0, 16000, 100))
    for i in tqdm(offsets, total=len(offsets)):
        pg = requests.get(f"https://neurovault.org/api/collections/?offset={str(i)}")
        db.append(pg)

    db_json = [i.json() for i in db]
    out = [j for i in db_json for j in i["results"]]
    filtered_out = [i for i in out if i["number_of_images"] > 0 and i["DOI"] is not None]

    with open(data_dir / "neurovault_collections.pkl", "wb") as f:
        pickle.dump(filtered_out, f)

    meta_images = []
    for i in tqdm(filtered_out, total=len(filtered_out)):
        meta_images.extend(
            requests.get(f"https://neurovault.org/api/collections/{str(i['id'])}/images/").json()["results"]
        )

    with open(data_dir / "neurovault_images.pkl", "wb") as f:
        pickle.dump(meta_images, f)

    # Preprocess
    df_collections = pd.DataFrame(filtered_out)

    df_images = pd.DataFrame(meta_images)
    df_images = df_images.iloc[:, :67]
    df_images = df_images[(df_images["analysis_level"] == "group")]
    df_images = df_images.drop_duplicates("id")

    df_collections = df_collections[df_collections['id'].isin(df_images['collection_id'])]
    df_collections.to_parquet(data_dir / "neurovault_collections.parquet")
    df_images.to_parquet(data_dir / "neurovault_images.parquet")

    # Scrape abstracts from crossref
    titles = []
    abstracts = []

    for doi in tqdm(df_collections["DOI"], total=len(df_collections)):

        url = f"https://api.crossref.org/works/{doi}"

        response = requests.get(url)
        data = response.json()

        title = data['message']['title'][0]
        abstract = data['message'].get('abstract', 'No abstract available')

        titles.append(title)
        abstracts.append(abstract)

    df_text = pd.DataFrame(dict(
        doi=df_collections["DOI"],
        title=titles,
        abstract=abstracts,
    ))
    df_text["abstract"] = df_text["abstract"].replace("No abstract available", pd.NA)
    df_text.to_parquet(data_dir / "neurovault_abstracts.parquet")
else:
    # Skip scraping
    df_collections = pd.read_parquet(data_dir / "neurovault_collections.parquet")
    df_images = pd.read_parquet(data_dir / "neurovault_images.parquet")
    df_text = pd.read_parquet(data_dir / "neurovault_abstracts.parquet")    # scaped from crossref
    df_missing = pd.read_csv(data_dir / "neurovault_abstracts_missing.csv") # manually scraped, missing from crossref


# %% [markdown]
# ## Preprocessing

# %%
def clean_abstracts(df_abstracts):
    """Regex to clean up noisy abstract formats."""

    # Reduce white spcae, newlines, tabs
    df_abstracts['abstract'] = df_abstracts["abstract"].str.replace(r'</?[^>]+?>', '', regex=True)
    df_abstracts['abstract'] = df_abstracts['abstract'].str.replace("\n", " ")
    df_abstracts["abstract"] = df_abstracts["abstract"].str.replace(r'\s+', ' ', regex=True)
    df_abstracts["abstract"] = df_abstracts["abstract"].str.strip(" ")
    df_abstracts["abstract"] = df_abstracts["abstract"].str.strip("\t")
    df_abstracts["abstract"] = df_abstracts["abstract"].str.replace(r"^ ", "", regex=True)
    df_abstracts["abstract"] = df_abstracts["abstract"].str.replace(r"^\t", "", regex=True)

    # Remove headers
    df_abstracts["abstract"] = df_abstracts["abstract"].str.replace(r"^Abstract\s*", "", regex=True)
    df_abstracts["abstract"] = df_abstracts["abstract"].str.replace(r"^ABSTRACT\s*", "", regex=True)
    df_abstracts["abstract"] = df_abstracts["abstract"].str.replace(r"^Significance\s*", "", regex=True)
    df_abstracts["abstract"] = df_abstracts["abstract"].str.replace(r"^Background\s*", "", regex=True)
    df_abstracts["abstract"] = df_abstracts["abstract"].str.replace(r"^Objective\s*", "", regex=True)
    df_abstracts["abstract"] = df_abstracts["abstract"].str.replace("BACKGROUND AND OBJECTIVES: ", "")

    # Special characters
    df_abstracts["abstract"] = df_abstracts["abstract"].str.replace("( ", "(").replace(" )", ")")
    df_abstracts["abstract"] = df_abstracts["abstract"].str.replace("’", "'")
    df_abstracts["abstract"] = df_abstracts["abstract"].str.replace("&amp", "&")
    df_abstracts["abstract"] = df_abstracts["abstract"].str.replace("&lt;b&gt;&lt;i&gt;Introduction:&lt;/i&gt;&lt;/b&gt; ", "")

    return df_abstracts


# %%
# Clean up
df_text = clean_abstracts(df_text)
df_missing = clean_abstracts(df_missing)
df_missing = df_missing.merge(
    df_collections[["DOI", "name"]].rename(columns={"DOI":"doi", "name":"title"}), on="doi", how="left"
)
df_pubs = pd.concat((df_text, df_missing))
df_pubs.to_parquet(data_dir / "neurovault_publications.parquet")

# %% [markdown]
# ## Images
#
# 1. Resample images to a common space.
# 2. Cluster threshold activation maps.

# %%
# Read
df_collections = pd.read_parquet(data_dir / "neurovault_collections.parquet")
df_collections = df_collections.rename(columns=dict(id="collection_id"))

df_images = pd.read_parquet(data_dir / "neurovault_images.parquet")
df_images = df_images[df_images["is_thresholded"] == False]
df_images = df_images[~df_images["id"].isin([22136, 22137, 22138])] # these ids are missing images
df_images = df_images[~(df_images["map_type"].isin(['parcellation', 'anatomical', "ROI/mask", "variance"]))]

df = df_images[["id", "collection_id", 'contrast_definition']].merge(
    df_collections[["name", "collection_id"]], on="collection_id", how="left")

# %%
masker = load_masker()

def process_one(idx, img_id):
    """Define worker."""
    f = data_dir / "neurovault_images" / f"neurovault_img_{img_id}.nii.gz"

    try:
        img = nib.load(f)

        # some maps have nan for outside of brain
        img_arr = img.get_fdata()
        mask = ~np.isfinite(img_arr)
        if np.all(mask):
            print(f"All NaN: {img_id}")
        img_arr[mask] = 0

        # needs to be continuous, not binary or integer
        if len(np.unique(img_arr.flatten())) < 10:
            return idx, None

        # resample
        img = nib.Nifti1Image(img_arr, img.affine)
        arr = masker.transform(
            resample_to_img(img, masker.mask_img, interpolation="nearest",
            force_resample=True, copy_header=True
        ))

        arr[arr < 0] = 0
        # arr = np.abs(arr)
    except:
        # catch all because these are abitrary user uploaded files,
        # there is no clean way to guarantee valid nii.gz images
        return idx, None

    return idx, arr

# Setup
n = len(df)
neuro = torch.zeros((n, 28542), dtype=torch.float32)

# Process images in parallel
results = Parallel(n_jobs=16, backend="loky", prefer="processes")(
    delayed(process_one)(idx, img_id)
    for idx, img_id in enumerate(tqdm(df["id"].copy(), total=n))
)

# Re-order
neuro = torch.zeros((n, 28542), dtype=torch.float32)
for idx, arr in results:
    if arr is not None:
        neuro[idx] = torch.from_numpy(arr)

torch.save(neuro, "neuro.pt")
df.to_csv("df.csv")

# %%
neuro = torch.load("neuro.pt")
df = pd.read_csv("df.csv")

# Drop empty (unthresholded map is empty) and binary maps
mask = ~((neuro == 0).all(axis=1)).numpy()
df, neuro = df[mask], neuro[mask]

mask = np.array([len(np.unique(i)) > 50 for i in neuro])
df, neuro = df[mask], neuro[mask]
df.shape, neuro.shape


# %%
# Cluster activation maps
def cluster(i, arr):
    thr_img = threshold_img(
        masker.inverse_transform(arr), "99%",
        cluster_threshold=50, two_sided=False, copy_header=True
    )
    vec = masker.transform(thr_img)
    return i, vec.astype(np.float32, copy=False)

neuro_cp = neuro.detach().cpu().numpy()
neuro_cp[neuro_cp < 0] = 0

results = Parallel(n_jobs=16, backend="loky")(
    delayed(cluster)(i, arr) for i, arr in
    enumerate(tqdm(neuro_cp, total=len(neuro_cp)))
)

neuro_clustered = np.zeros((len(neuro), 28542), dtype=np.float32)
for i, vec in results:
    neuro_clustered[i] = vec

np.save("neuro_clustered.npy", neuro_clustered)

# %%
neuro_clustered = np.load("neuro_clustered.npy")

# %%
# Drop overactive examples (whole brain is activated)
mask = (neuro_clustered != 0).mean(axis=1) < 0.1
df = df[mask]
df.reset_index(inplace=True, drop=True)
neuro_clustered = neuro_clustered[mask]
neuro = neuro[mask]

# Drop all zero
mask = ~(neuro_clustered == 0).all(axis=1)
neuro_clustered = neuro_clustered[mask]
neuro = neuro[mask]
df = df[mask]
df.reset_index(inplace=True, drop=True)

# %% [markdown]
# ## Filter
#
# Remove pubmed training ids from this sample.

# %%
df_pubmed = load_dataset("pubmed_text")

df_collections_filt = df_collections[
    df_collections["collection_id"].isin(df['collection_id'].unique())
]

df_collections_filt = df_collections_filt[
    ~(df_collections_filt["DOI"].isin(df_pubmed["doi"]))
]

# Drop
mask = df["collection_id"].isin(df_collections_filt["collection_id"])
df = df[mask]
neuro = neuro[mask]
neuro_clustered = neuro_clustered[mask]

df_pubs = pd.read_parquet(data_dir / "neurovault_publications.parquet")
df_pubs = df_pubs[df_pubs["doi"].isin(
    df_collections_filt[df_collections_filt["collection_id"].isin(
        df["collection_id"]
    )]["DOI"]
)]

# %%
torch.save(neuro_clustered, data_dir / "neuro_clust.pt")

# %% [markdown]
# ## Encode Text

# %%
specter = Specter(
    model="allenai/specter2_aug2023refresh",
    adapter="adhoc_query",
    device=select_device()
)

titles = df_pubs["title"]
abstracts = df_pubs["abstract"]

batch_size = 64
text_emb_titles = torch.zeros((len(titles), 768))
text_emb_abstracts = torch.zeros((len(titles), 768))
text_emb_titles_abstracts = torch.zeros((len(titles), 768))

for i in tqdm(range(0, len(titles), batch_size), total=len(titles)//batch_size):
    with torch.no_grad():
        text_emb_titles[i:i+batch_size] = specter(titles[i:i+batch_size].tolist()).detach()
        text_emb_abstracts[i:i+batch_size] = specter(abstracts[i:i+batch_size].tolist()).detach()
        text_emb_titles_abstracts[i:i+batch_size] = specter(
            (titles[i:i+batch_size] + "[SEP]" + abstracts[i:i+batch_size]).tolist()
        ).detach()

# %%
torch.save({
    "df_neuro": df.merge(
        df_collections_filt[["collection_id", "DOI"]].rename(
            columns={"DOI": "doi"}
        ),
        on="collection_id", how="left"
    ),
    "df_pubs": df_pubs,
    "neuro": neuro,
    "neuro_clustered": neuro_clustered,
    "text_emb_titles": text_emb_titles,
    "text_emb_abstracts": text_emb_abstracts,
    "text_emb_titles_abstracts": text_emb_titles_abstracts
}, data_dir / "neurovault.pt")
