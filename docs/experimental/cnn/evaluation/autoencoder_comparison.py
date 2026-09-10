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
# # Autoencoder reconstruction comparison
#
# Compare the released MLP autoencoder with the mixed-baseline CNN on paired test examples. The published test split contains 3,066 PubMed, 79 Nilearn, and 202 NeuroVault maps. Because full PubMed reconstruction is slow on a typical Mac, this notebook evaluates the first 200 PubMed examples and the complete Nilearn and NeuroVault splits. Set `DOMAIN_LIMITS["pubmed"] = None` to run the complete PubMed test split. Set `INCLUDE_FINETUNED = True` only when you explicitly want the domain-fine-tuned CNN branches.
#
# The provider downloads the published root-level split JSONLs and shared volume tensor from Hugging Face. Legacy local paths stored in individual JSONL rows are metadata only and are never loaded.
#
# This is the `paired_atlas_free` protocol. Older PubMed MLP autoencoder comparisons loaded the MLP-native PubMed image resource and selected a different first-N cohort, while the CNN used atlas-free rows. Those older values are valid for their native cohort but are not expected to match this notebook. Here both families receive the same examples; the MLP bridge converts each shared volume to its established masker flat-map representation.

# %%
import matplotlib.pyplot as plt
import pandas as pd
import torch

from neurovlm import AtlasFreeCNNDataProvider, load_pipeline
from neurovlm.evaluation import (
    default_comparison_matrix,
    evaluate_reconstruction_comparison,
)

DOMAINS = ("pubmed", "nilearn", "neurovault")
DOMAIN_LIMITS = {
    "pubmed": 200,       # runtime-conscious default; use None for all 3,066
    "nilearn": None,     # complete test split: 79
    "neurovault": None,  # complete test split: 202
}
INCLUDE_FINETUNED = False  # explicit opt-in
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVALUATION_SCOPE = "PubMed first 200; Nilearn and NeuroVault complete test splits"


# %% [markdown]
# ## Run the comparison
#
# The shared volume payload is about 4.2 GB and is downloaded once into the normal Hugging Face cache. Later providers reuse the cached payload.

# %%
results = []
for domain in DOMAINS:
    selections = default_comparison_matrix(
        "autoencoder",
        domains=(domain,),
        include_finetuned=INCLUDE_FINETUNED,
    )
    provider = AtlasFreeCNNDataProvider(
        domain=domain,
        limit=DOMAIN_LIMITS[domain],
    )
    results.append(evaluate_reconstruction_comparison(
        selections=selections,
        provider=provider,
        device=DEVICE,
    ))

summary = pd.DataFrame(row for result in results for row in result.summary)
by_source = pd.DataFrame(row for result in results for row in result.by_source)
by_sample = pd.DataFrame(row for result in results for row in result.by_sample)
manifest = pd.DataFrame(row for result in results for row in result.manifest)

summary.sort_values(["evaluation_domain", "family", "variant"])


# %% [markdown]
# ## Aggregate metric plots
#
# CNN metrics are computed in native atlas-free volume space; MLP metrics are computed in the established masker flat-map space. The plots compare complete pipelines in their declared spaces, not voxel-identical representations. Lower reconstruction MSE is better; higher spatial correlation and top-5% Dice are better.

# %%
resolved = summary[(summary["status"] == "resolved") & (summary["n"] > 0)].copy()
if resolved.empty:
    raise RuntimeError("No models resolved. Inspect `manifest` for checkpoint errors.")
resolved["model"] = resolved.apply(
    lambda row: f'{row["family"].upper()} · {row["variant"]}', axis=1
)

metrics = (
    ("reconstruction_mse", "Reconstruction MSE ↓"),
    ("spatial_corr", "Spatial correlation ↑"),
    ("top5_dice", "Top-5% Dice ↑"),
)
fig, axes = plt.subplots(1, len(metrics), figsize=(17, 4.5))
for ax, (metric, title) in zip(axes, metrics):
    table = resolved.pivot(index="evaluation_domain", columns="model", values=metric)
    table.plot.bar(ax=ax, rot=0)
    ax.set_title(title)
    ax.set_xlabel("Evaluation domain")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Model", fontsize=8)
fig.suptitle(f"Autoencoder comparison ({EVALUATION_SCOPE})")
fig.tight_layout()
plt.show()


# %% [markdown]
# ## Per-sample distributions
#
# These box plots expose variance hidden by the aggregate means.

# %%
samples = by_sample.copy()
samples["model"] = samples.apply(
    lambda row: f'{row["family"].upper()} · {row["variant"]}', axis=1
)
samples["group"] = samples["evaluation_domain"] + "\n" + samples["model"]
group_names = list(dict.fromkeys(samples["group"]))

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for ax, metric, title in (
    (axes[0], "spatial_corr", "Per-sample spatial correlation ↑"),
    (axes[1], "top5_dice", "Per-sample top-5% Dice ↑"),
):
    values = [samples.loc[samples["group"] == name, metric].dropna() for name in group_names]
    ax.boxplot(values, labels=group_names, showmeans=True)
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=35)
    ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
plt.show()


# %% [markdown]
# ## Qualitative CNN reconstructions
#
# The next three cells show three examples per domain. All-zero maps are skipped—this matters for Nilearn because some very small atlas parcels disappear during 4 mm resampling/cropping. For every non-empty example, the sagittal, coronal, or axial plane with the greatest input activation is selected. Input, reconstruction, and error use independent robust color scales so sparse Nilearn parcels and lower-amplitude reconstructions remain visible; compare spatial structure rather than color magnitude across panels.

# %%
cnn = load_pipeline(family="cnn", task="autoencoder", device=DEVICE)
VISUAL_EXAMPLES_PER_DOMAIN = 3
PLANE_NAMES = ("sagittal", "coronal", "axial")

def _first_nonempty_examples(domain, count=VISUAL_EXAMPLES_PER_DOMAIN):
    data = AtlasFreeCNNDataProvider(domain=domain).test
    examples = []
    skipped = 0
    for index in range(len(data)):
        example = data[index]
        if int(torch.count_nonzero(example["volume"])) == 0:
            skipped += 1
            continue
        examples.append(example)
        if len(examples) == count:
            break
    if len(examples) != count:
        raise RuntimeError(f"Only {len(examples)} non-empty {domain} examples were available")
    return examples, skipped

def _strongest_plane(volume):
    candidates = []
    for axis, plane_name in enumerate(PLANE_NAMES):
        reduce_dims = tuple(dim for dim in range(3) if dim != axis)
        scores = volume.abs().sum(dim=reduce_dims)
        slice_index = int(scores.argmax())
        candidates.append((float(scores[slice_index]), axis, slice_index, plane_name))
    _, axis, slice_index, plane_name = max(candidates)
    return volume.select(axis, slice_index), axis, slice_index, plane_name

def _robust_positive_max(values):
    positive = values[values > 0]
    if not len(positive):
        return max(float(values.abs().max()), 1e-8)
    return max(float(torch.quantile(positive.float(), 0.99)), 1e-8)

def plot_reconstruction_domain(domain, count=VISUAL_EXAMPLES_PER_DOMAIN):
    examples, skipped = _first_nonempty_examples(domain, count)
    inputs = torch.stack([example["volume"] for example in examples])
    reconstructions = cnn.reconstruct(inputs).cpu()[:, 0]
    metadata = []
    fig, axes = plt.subplots(count, 3, figsize=(13, 4 * count))
    for row_index, (example, reconstruction) in enumerate(zip(examples, reconstructions)):
        truth = example["volume"][0].cpu()
        truth_plane, axis, slice_index, plane_name = _strongest_plane(truth)
        reconstruction_plane = reconstruction.select(axis, slice_index)
        error_plane = (reconstruction_plane - truth_plane).abs()
        panels = (
            (truth_plane, "Input", "hot"),
            (reconstruction_plane, "CNN reconstruction", "hot"),
            (error_plane, "Absolute error", "magma"),
        )
        for ax, (plane, title, cmap) in zip(axes[row_index], panels):
            image = ax.imshow(
                plane.T,
                origin="lower",
                cmap=cmap,
                vmin=0.0,
                vmax=_robust_positive_max(plane),
            )
            ax.set_title(f"Example {row_index + 1} · {title}")
            ax.axis("off")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        metadata.append({
            "domain": domain,
            "example": row_index + 1,
            "map_id": example["map_id"],
            "plane": plane_name,
            "slice_index": slice_index,
            "all_zero_maps_skipped_before_selection": skipped,
        })
    fig.suptitle(f"{domain.title()} mixed-baseline CNN reconstructions")
    fig.tight_layout()
    plt.show()
    return pd.DataFrame(metadata)



# %% [markdown]
# ### PubMed: three reconstruction examples

# %%
plot_reconstruction_domain("pubmed")

# %% [markdown]
# ### Nilearn: three non-empty reconstruction examples

# %%
plot_reconstruction_domain("nilearn")

# %% [markdown]
# ### NeuroVault: three reconstruction examples

# %%
plot_reconstruction_domain("neurovault")
