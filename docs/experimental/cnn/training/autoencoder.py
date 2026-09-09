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
# # Train the atlas-free CNN autoencoder
#
# The retained mixed-source autoencoder is the default. Domain fine-tuning is an explicit opt-in; the run writes reproducible configuration, provenance, checkpoints, metrics, plots, generated maps, and logs.

# %%
from neurovlm.training import AutoencoderTrainConfig, train_autoencoder

VARIANT = "mixed_baseline"  # change explicitly to "finetuned"
DOMAIN = None                 # finetuned: pubmed, nilearn, or neurovault

config = AutoencoderTrainConfig(
    output_root="runs",
    variant=VARIANT,
    domain=DOMAIN,
    epochs=100,
)
result = train_autoencoder(config)

# %%
print("run:", result.run_dir)
print("best checkpoint:", result.best_checkpoint)
print("epoch metrics:", result.run_dir / "metrics/history.csv")
print("summary metrics:", result.run_dir / "metrics/summary.csv")

# %% [markdown]
# ## Resume or initialize from a local run
#
# Published Hugging Face resources are used automatically. For intentional local chaining, pass `from_run="runs/..."`; to continue an interrupted run, retain its `run_id` and pass `resume="runs/..."`.

# %%
# Example only; uncomment after setting your own run directory.
# chained = AutoencoderTrainConfig(from_run="runs/<run-id>")
# resumed = AutoencoderTrainConfig(run_id="my-run", resume="runs/my-run")
