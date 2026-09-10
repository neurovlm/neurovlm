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
# # Contrastive retrieval comparison
#
# Compare the released MLP contrastive pipeline with the mixed-baseline CNN on identical complete PubMed (3,066), Nilearn (79), and NeuroVault (202) test splits. This notebook uses the paired atlas-free protocol; set `INCLUDE_FINETUNED = True` only for an explicit domain-fine-tuning ablation.

# %%
import matplotlib.pyplot as plt
import pandas as pd
import torch

from neurovlm import AtlasFreeCNNDataProvider
from neurovlm.evaluation import (
    default_comparison_matrix,
    evaluate_contrastive_comparison,
)

DOMAINS = ("pubmed", "nilearn", "neurovault")
LIMIT_PER_DOMAIN = None  # full test split; set an integer only for a quick run
INCLUDE_FINETUNED = False  # explicit opt-in
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVALUATION_SCOPE = (
    "full test split"
    if LIMIT_PER_DOMAIN is None
    else f"first {LIMIT_PER_DOMAIN} test pairs per domain"
)


# %% [markdown]
# ## Run the comparison
#
# The provider retrieves the root-level split JSONLs, shared volume tensor, and normalized SPECTER2 text cache from Hugging Face. CNN rows consume that published cache. MLP rows re-encode the same raw positive texts with the released MLP SPECTER2 `adhoc_query` preprocessing. Resources use the normal Hugging Face cache and are reused across domains.

# %%
results = []
for domain in DOMAINS:
    selections = default_comparison_matrix(
        "contrastive",
        domains=(domain,),
        include_finetuned=INCLUDE_FINETUNED,
    )
    provider = AtlasFreeCNNDataProvider(
        domain=domain,
        limit=LIMIT_PER_DOMAIN,
    )
    results.append(evaluate_contrastive_comparison(
        selections=selections,
        provider=provider,
        device=DEVICE,
    ))

summary = pd.DataFrame(row for result in results for row in result.summary)
recall_curves = pd.DataFrame(
    row for result in results for row in result.recall_curves
)
by_sample = pd.DataFrame(row for result in results for row in result.by_sample)
manifest = pd.DataFrame(row for result in results for row in result.manifest)

summary.sort_values(["evaluation_domain", "family", "variant"])


# %% [markdown]
# ### Why this PubMed MLP AUC differs from the older plot
#
# The older PubMed MLP result was **0.831055**, but it evaluated the first 32 aligned examples from the MLP-native PubMed resource and its official test split. The integrated notebook previously displayed **0.718262** for the first 32 PubMed examples in the atlas-free unified test split and shared the CNN-oriented cached text inputs with the MLP row. Restoring family-native MLP text preprocessing raises the verified paired result to **0.744629**. The remaining difference from 0.831055 is expected because the cohort is still different. The AUC implementation is the same; the example cohort is not. The old calculation was valid for its native benchmark, but presenting it as directly paired with the CNN result obscured that protocol difference.
#
# This notebook deliberately keeps the paired protocol so both families see the same map/text pairs. Its rows record `comparison_protocol="paired_atlas_free"`, their declared brain space, and family-specific `text_preprocessing`. Do not use the historical 0.831055 as the expected value for this different cohort.

# %% [markdown]
# ## Normalized recall-curve AUC
#
# AUC integrates Recall@K over normalized K (`K / N`), so comparisons remain interpretable across evaluation-set sizes. Higher is better. Text-to-image (T2I) retrieves brain maps from text; image-to-text (I2T) retrieves text from brain maps.

# %%
resolved = summary[(summary["status"] == "resolved") & (summary["n"] > 0)].copy()
if resolved.empty:
    raise RuntimeError("No models resolved. Inspect `manifest` for checkpoint errors.")
resolved["model"] = resolved.apply(
    lambda row: f'{row["family"].upper()} · {str(row["variant"]).replace("_", " ")}',
    axis=1,
)

auc_metrics = (
    ("t2i_normalized_k_recall_curve_auc", "Text → image normalized AUC"),
    ("i2t_normalized_k_recall_curve_auc", "Image → text normalized AUC"),
    ("mean_normalized_k_recall_curve_auc", "Mean bidirectional normalized AUC"),
)
fig, axes = plt.subplots(1, len(auc_metrics), figsize=(18, 4.8), sharey=True)
for ax, (metric, title) in zip(axes, auc_metrics):
    table = resolved.pivot(
        index="evaluation_domain", columns="model", values=metric
    ).reindex(DOMAINS)
    table.plot.bar(ax=ax, rot=0)
    ax.set_title(title)
    ax.set_xlabel("Evaluation domain")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Model", fontsize=8)
axes[0].set_ylabel("Normalized recall-curve AUC")
fig.suptitle(f"Contrastive retrieval ({EVALUATION_SCOPE})")
fig.tight_layout()
plt.show()


# %% [markdown]
# ## Normalized recall curves
#
# As in the MLP retrieval plots, the dashed diagonal is random chance. Each model label reports its mean bidirectional normalized AUC.

# %%
model_labels = resolved.set_index("model_id")["model"].to_dict()
mean_auc = resolved.set_index("model_id")["mean_normalized_k_recall_curve_auc"].to_dict()

fig, axes = plt.subplots(1, len(DOMAINS), figsize=(18, 5), sharex=True, sharey=True)
for ax, domain in zip(axes, DOMAINS):
    domain_curves = recall_curves[recall_curves["evaluation_domain"] == domain]
    for model_id, curve in domain_curves.groupby("model_id", sort=False):
        curve = curve.sort_values("normalized_k")
        ax.plot(
            curve["normalized_k"],
            curve["mean_recall"],
            linewidth=2,
            label=f'{model_labels[model_id]} (AUC={mean_auc[model_id]:.3f})',
        )
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Random chance")
    ax.set_title(domain.title())
    ax.set_xlabel(r"Normalized K: $K / N$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
axes[0].set_ylabel("Mean bidirectional Recall@K")
fig.suptitle("Normalized contrastive recall curves")
fig.tight_layout()
plt.show()

