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
# # Train CNN contrastive retrieval on PubMed
#
# This is the integrated version of the original best-recipe experiment. The released mixed-source autoencoder initializes the brain encoder by default; fine-tuning is never selected implicitly.

# %%
from neurovlm.training import ContrastiveTrainConfig, train_contrastive

config = ContrastiveTrainConfig(
    domain="pubmed",             # pubmed | nilearn | neurovault
    variant="mixed_baseline",  # explicit alternative: finetuned
    output_root="runs",
    epochs=100,
)
result = train_contrastive(config)

# %%
print("run:", result.run_dir)
print("best checkpoint:", result.best_checkpoint)
print("epoch metrics:", result.run_dir / "metrics/history.csv")
print("recall curves:", result.run_dir / "metrics/curves.csv")

# %% [markdown]
# ## Explicit local initialization and resume
#
# Leave `from_run` unset for released Hugging Face initialization. Set it only when chaining from a locally trained autoencoder. Resume retains the original `run_id` and uses the existing run directory.

# %%
# config = ContrastiveTrainConfig(domain="pubmed", from_run="runs/<ae-run-id>")
# config = ContrastiveTrainConfig(domain="pubmed", run_id="my-run", resume="runs/my-run")
