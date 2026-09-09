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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Train CNN contrastive and text-to-brain branches
#
# Contrastive retrieval and text-to-brain projection are independent tasks. Choose one of the three domains in one place. Both default to the released mixed-source autoencoder.

# %%
from neurovlm.training import (
    ContrastiveTrainConfig, TextToBrainTrainConfig,
    train_contrastive, train_text_to_brain,
)

DOMAIN = "pubmed"            # pubmed | nilearn | neurovault
VARIANT = "mixed_baseline"  # explicit alternative: finetuned
AE_FROM_RUN = None            # e.g. "runs/<ae-run-id>"


# %% [markdown]
# ## Contrastive retrieval

# %%
contrastive = train_contrastive(ContrastiveTrainConfig(
    domain=DOMAIN, variant=VARIANT, output_root="runs",
    from_run=AE_FROM_RUN,
))
print(contrastive.run_dir)
print(contrastive.run_dir / "metrics/history.csv")
print(contrastive.run_dir / "metrics/curves.csv")

# %% [markdown]
# ## Text-to-brain projection

# %%
text_to_brain = train_text_to_brain(TextToBrainTrainConfig(
    domain=DOMAIN, variant=VARIANT, output_root="runs",
    autoencoder_from_run=AE_FROM_RUN,
))
print(text_to_brain.run_dir)
print(text_to_brain.run_dir / "metrics/history.csv")
print(text_to_brain.run_dir / "metrics/summary.csv")

# %% [markdown]
# ## Resume
#
# To resume either task, keep its original `run_id` and pass its run directory through `resume`. This restores the optimizer, epoch, best-metric state, and accumulated metric history.
