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

# %% [markdown]
# # Tutorial 4: Generative brain-to-text with NeuroQFormer
#
# This tutorial uses the packaged QFormer-based generator:
#
# ```text
# brain map or latent → NeuroQFormer → NeuroQwen → generated neuroscience text
# ```
#
# The examples below use bundled network latents so the notebook is small and deterministic. A CUDA GPU is recommended; CPU works but can be slow because NeuroQwen is loaded for generation.
#

# %%
import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import nibabel as nib
import matplotlib.pyplot as plt
import textwrap
from nilearn.image import resample_to_img, smooth_img, threshold_img
from nilearn.plotting import plot_stat_map

from neurovlm import NeuroVLM
from neurovlm.data import load_dataset, load_masker, load_latent
from neurovlm.models import load_model

device = "cuda" if torch.cuda.is_available() else "cpu"
nvlm = NeuroVLM(device=device)
masker = load_masker()
networks = load_dataset("networks")

# %% [markdown]
# ## 2. Generation
#
# Predict network text description for the Laird `Visual1` map.
#

# %% [markdown]
# ### Input image

# %%
# Load image
X = networks["Laird"]["Visual1"]["array"]
affine = networks["Laird"]["Visual1"]["affine"]

# Use percentile threshold to binarize
X = X / X.max()
X[X > 1] = 1

# Plot
img_visual = nib.Nifti1Image(X, affine=affine)

plot_stat_map(img_visual, draw_cross=False, cmap='hot')

# %% [markdown]
# ### Generated text
#
# Text targets where trained with steering prefixes:
#
# - `[NETWORK]`: network-level generation
# - `[REGION]`: individual region generation
# - `[FUNCTION]`: funtional-level generation
#
# These prefixes are controlled with the `basis` kwarg. If left as `None`, the LLM determines the basis in generation. The same brain latent can be decoded with different explicit basis prefixes. This is useful when you want a network-level label, a likely anatomical region, or a functional interpretation.

# %%
GENERATION_KWARGS = dict(
    max_new_tokens=140,
    num_beams=3,
    do_sample=False,
    seed=12345,
    projection_temp=0.035,
    repetition_penalty=1.18,
    no_repeat_ngram_size=4,
)

text = nvlm.brain(img_visual).to_text(head="qformer", **GENERATION_KWARGS)

for basis in ["network", "region", "function"]:
    text = nvlm.brain(img_visual).to_text(head="qformer", basis=basis, **GENERATION_KWARGS)
    print(text + "\n")


# %% [markdown]
# ## 3. Generate multiple examples
#
# Batching keeps the model load cost fixed and returns one generated string per input latent.
#

# %%
# Prepare images
examples = [
    ("HCPICA", "ICA2"),
    ("Glasser", "Somatomotor"),
    ("Du", "AUD"),
]

images = []
image_arrays = []
for atlas, network in examples:
    # Load image
    X = networks[atlas][network]["array"]
    affine = networks[atlas][network]["affine"]

    # Rescale to (0, 1)
    X = X / X.max()

    # Resample to 4mm space
    img = nib.Nifti1Image(X, affine=affine)
    img = resample_to_img(img, masker.mask_img, interpolation="nearest")

    # Plot
    images.append(img)
    image_arrays.append(masker.transform(images[-1]))

image_arrays = np.vstack(image_arrays)

# %%
# Generate text
generated_text = nvlm.brain(image_arrays).to_text(head="qformer", **GENERATION_KWARGS)

# %%
fig, axes = plt.subplots(3, 2, figsize=(12, 7))

for ax_row, ex, img, text in zip(axes, examples, images, generated_text):
    plot_stat_map(img, axes=ax_row[0], title=f"{ex[0]} {ex[1]}", colorbar=False, draw_cross=False)
    ax_row[1].axis("off")
    ax_row[1].text(0, 1, text, va="top", wrap=True)


# %% [markdown]
# ## 6. NeuroVault examples
#
# This example uses NeuroVault maps. For each search term, candidate papers are matched by title/abstract, then the contrastive image/text model selects the most similar image from those papers. The image column is smoothed and cluster-thresholded before plotting to make the spatial pattern readable.
#
#
# NeuroVault studies have multiple images per study. Below uses the contrastive model to select the most similar image to the study text for demonstration.

# %%
# Load
neurovault_images = load_dataset("neurovault_images")
neurovault_meta = load_dataset("neurovault_images_meta").reset_index(drop=True)
neurovault_text = load_dataset("neurovault_text").reset_index(drop=True)
neurovault_text["text_row"] = neurovault_text.index
neurovault_text_by_doi = neurovault_text.set_index("doi", drop=False)

latent_neurovault_images = load_latent("neurovault_images")
latent_neurovault_text = load_latent("neurovault_text")

# Contrastive model
proj_head_image = load_model("proj_head_image_infonce").to(device).eval()
proj_head_text = load_model("proj_head_text_infonce").to(device).eval()
with torch.no_grad():
    image_latent = F.normalize(
        proj_head_image(latent_neurovault_images.to(device)), dim=1
    ).cpu()

    text_latent = F.normalize(
        proj_head_text(latent_neurovault_text.to(device)), dim=1
    ).cpu()

# Pair images to text
def _contains(series, search_str):
    return series.fillna("").astype(str).str.contains(search_str, case=False, na=False, regex=False)

def select_neurovault_image(search_str):

    papers = neurovault_text[
        _contains(neurovault_text["title"], search_str)
        | _contains(neurovault_text["abstract"], search_str)
    ].copy()
    images = neurovault_meta[neurovault_meta["doi"].isin(papers["doi"])].copy()
    if images.empty:
        raise ValueError(f"No NeuroVault paper/image match contains: {search_str!r}")

    image_idx = images.index.to_numpy(copy=True)
    text_idx = images["doi"].map(papers.set_index("doi")["text_row"]).astype(int).to_numpy(copy=True)
    scores = (image_latent[image_idx] * text_latent[text_idx]).sum(dim=1)
    image_idx = int(image_idx[int(scores.argmax())])

    return image_idx

search_terms = ["motor", "reward", "visual perception"]
neurovault_ids = [select_neurovault_image(term) for term in search_terms]
nv_raw_batch = neurovault_images[neurovault_ids]


# %%
# Cluster images
def cluster_threshold_vector(brain_vec, smoothing_fwhm=8):
    brain_vec = brain_vec.clamp_min(0)
    img = smooth_img(masker.inverse_transform(brain_vec), fwhm=smoothing_fwhm)
    img = threshold_img(
        img,
        "95%",
        cluster_threshold=100,
        two_sided=False,
        copy_header=True,
    )
    vec = torch.from_numpy(masker.transform(img)).float().squeeze(0)
    return vec, img

nv_thresholded = [cluster_threshold_vector(brain_vec) for brain_vec in nv_raw_batch]
nv_batch = torch.stack([vec for vec, _ in nv_thresholded])
nv_images = [img for _, img in nv_thresholded]

# %%
# Generate text from images
nv_generated = nvlm.brain(nv_batch).to_text(head="qformer", **GENERATION_KWARGS)


# %%
# Plot
def wrap_text(text, width=38):
    return "\n".join(textwrap.fill(line, width=width) for line in str(text).splitlines())

fig, axes = plt.subplots(3, 3, figsize=(13, 10), gridspec_kw={"wspace": 0.03, "hspace": 0.12})
text_box = dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="0.75")

for i, (ax_img, ax_true, ax_gen) in enumerate(axes):
    image_idx = neurovault_ids[i]
    map_row = neurovault_meta.iloc[image_idx]
    doi = map_row["doi"]
    row = neurovault_text_by_doi.loc[doi]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    contrast = map_row["contrast_definition"]
    contrast = "not specified" if pd.isna(contrast) else contrast
    true_text = f"Title: {row['title']}\n\nAbstract: {row['abstract'][:200]}..."

    plot_stat_map(
        nv_images[i],
        axes=ax_img,
        colorbar=False,
        draw_cross=False,
        annotate=False,
        threshold=1e-6,
        cmap="Reds",
    )

    ax_true.axis("off")
    ax_gen.axis("off")
    ax_true.set(xlim=(0, 1), ylim=(0, 1))
    ax_gen.set(xlim=(0, 1), ylim=(0, 1))

    if i == 0:
        ax_img.set_title("True brain", pad=10)
        ax_true.set_title("True text", pad=10)
        ax_gen.set_title("Generated text", pad=10)

    ax_true.text(
        0.5, 0.5, wrap_text(true_text),
        ha="center", va="center", transform=ax_true.transAxes,
        fontsize=8.5, bbox=text_box,
    )

    ax_gen.text(
        0.5, 0.5, wrap_text(nv_generated[i]),
        ha="center", va="center", transform=ax_gen.transAxes,
        fontsize=8.5, bbox=text_box,
    )
