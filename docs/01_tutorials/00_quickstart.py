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
#     display_name: .env
#     language: python
#     name: python3
# ---

# %%
from neurovlm.data import fetch_data, load_latent
from neurovlm import NeuroVLM

# Fetch models and datasets
fetch_data()

# %% [markdown]
# # Quickstart
#
# This tutorial introduces the high-level, inference-only API. It walks through text-to-brain and brain-to-text generation and retrieval.

# %% [markdown]
# ## Generative models

# %% [markdown]
# ### Text-to-Brain: Generative
#
# The default text-to-brain model is from the PubMed training set, thus inherits biases of that set. For example, anatomy is not always well followed and probabilities are not calibrated. However, this is a base model that can be fined-tuned and calibrated. A calibrator class if provided in:
#
# `from neurovlm.models.adapter import LogitCalibrator`
#
# A anatomy-focused fine-tune adapter of the base model is provided, along with a learned calibrator. To use this model, set `.to_brain(head='mse', adapter=True)`. Custom adapters should be trained per use-case.

# %%
# Initialize with CPU device explicitly
# On Mac, avoid device conflicts by using CPU
nvlm = NeuroVLM(device="cpu")
result = nvlm.text(["vision", "default mode network"]).to_brain(head="mse")
result.to_nifti() # returns list of nib.Nifti1Image

# %%
result.plot(0, threshold=0.25); # plot image for vision

# %%
result.plot(1, threshold=0.15); # plot image for DMN

# %% [markdown]
# Use the anatomy adapter + calibrator head

# %%
result = nvlm.text(["insula", "putamen"]).to_brain(head="mse", adapter=True)

# %%
result.plot(0, threshold=0.5); # plot image for insula

# %%
result.plot(1, threshold=0.5); # plot image for putamen

# %% [markdown]
# ## Brain-to-text: Generative

# %%
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
nvlm = NeuroVLM(device=device)
networks = load_latent("networks_neuro")
result = nvlm.brain([networks["Du"]["AUD"], networks["Du"]["DN-A"]]).to_text()

# %%
# Text generated from the AUD map
print(result[0])

# %%
# Text generated from the DN-A map
print(result[1])

# %% [markdown]
# ## Contrastive models

# %% [markdown]
# ### Text-to-Brain: Contrastive Ranking & Retrieval
#
# Contrastive models are used for ranking and retrieval. We can lookup similar neuroimages in a dataset, given a text query. This works by embedding the text query and comparing a set of image embeddings via cosine similarity.

# %%
# Initialize with CPU device explicitly
nvlm = NeuroVLM(device="cpu")
result = nvlm.text("motor").to_brain(head='infonce')
top = result.top_k(2) # each row pairs to a neuorimage that is most similar to the text query
top

# %% [markdown]
# Each row in the `top` dataframe above, is paried to an image that can be viewed.

# %%
# WashU network atlas
top.plot_row(1, threshold=0.1);

# %%
# NeuroVault
top.plot_row(2, threshold=2.5);

# %%
# PubMed
top.plot_row(4, threshold=0.1);

# %% [markdown]
# ### Brain-to-Text: Contrastive Ranking & Retrieval
#
# Here we use an auditory map as input from the Du atlas. We use the contrastive model to rank the most similar text across the datasets.

# %%
# Transform rank text based on auditory network
nvlm = NeuroVLM(device="cpu")
result = nvlm.brain(networks["Du"]["AUD"]).to_text(head='infonce')

# %%
result.top_k(5, dataset="pubmed")

# %%
result.top_k(5, dataset="pubmed_mesh")

# %%
result.top_k(5, dataset='llm_neuro_terms')

# %%
result.top_k(5, dataset='cogatlas')

# %%
result.top_k(5, dataset='wiki')
