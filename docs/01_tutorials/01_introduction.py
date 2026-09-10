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
# # Tutorial 1: Introduction to NeuroVLM
#
# This tutorial provides an overview of NeuroVLM, a multimodal framework for text-to-brain and brain-to-text applications in neuroimaging. You'll learn about:
#
# 1. Fetching models and datasets from HuggingFace
# 2. Types of models and architectures
# 3. Available datasets
# 4. Basic concepts: text-to-brain and brain-to-text

# %% [markdown]
# ## 1. Setup and Installation
#
# First, let's import the necessary modules and fetch the pre-trained models and datasets.

# %%
import os

os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from neurovlm import NeuroVLM
from neurovlm.data import fetch_data, load_dataset, load_latent

# Fetch all models and datasets from HuggingFace
fetch_data()

# %% [markdown]
# ## 2. Model Architectures
#
# NeuroVLM uses several model architectures working together:
#
# ### Text Encoder (SPECTER2)
# - Encodes scientific text (titles, abstracts) into 768-dimensional embeddings
# - Pre-trained on scientific literature for domain-specific understanding
# - Handles variable-length text input
#
# ### Autoencoder
# - **Encoder**: Compresses 28,542-dimensional brain activation maps to 384-dimensional latent space
# - **Decoder**: Reconstructs brain maps from latent representations
# - Enables efficient storage and manipulation of neuroimaging data
#
# ### Projection Heads
# - **Contrastive (InfoNCE)**: Projects text and brain embeddings into shared space for retrieval
# - **Generative (MSE)**: Projects text embeddings to brain latent space for generation
#
# ### Architecture Overview
#
# ```
# Text → SPECTER2 → Projection Head → Shared Space ← Projection Head ← Autoencoder.Encoder ← Brain
#                                                                                              |
# Text → SPECTER2 → Projection Head → Brain Latent → Autoencoder.Decoder → Generated Brain  ←┘
# ```

# %% [markdown]
# ## 3. Available Datasets
#
# NeuroVLM includes several curated datasets:
#
# ### Text Datasets
# - **PubMed**: ~30K neuroimaging publications with titles and abstracts
# - **NeuroWiki**: Neuroscience concepts from Wikipedia
# - **Cognitive Atlas**: Cognitive concepts, tasks, and disorders
# - **Networks**: Canonical brain network descriptions
#
# ### Brain Image Datasets
# - **PubMed Images**: Brain activation maps from published studies
# - **NeuroVault**: Community-contributed brain maps
# - **Network Atlases**: Canonical brain networks (multiple atlases)
#
# Let's explore these datasets:

# %%
# Load text datasets
publications = load_dataset("pubmed_text")
print(f"PubMed publications: {len(publications)} papers")
print(f"Columns: {list(publications.columns)}")
print("\nExample publication:")
publications.head(10)

# %%
# Cognitive Atlas concepts
cogatlas = load_dataset("cogatlas")
print(f"\nCognitive Atlas concepts: {len(cogatlas)}")
print("\nExample concepts:")
cogatlas.head(10)

# %%
# NeuroWiki
neurowiki = load_dataset("wiki")
print(f"\nNeuroWiki entries: {len(neurowiki)}")
print("\nExample entries:")
neurowiki.head(10)

# %%
# Network atlases
networks = load_dataset("networks_canonical")
print(f"\nCanonical networks: {len(networks)}")
print("\nExample networks:")
networks.head(10)

# %% [markdown]
# ## 4. Text-to-Brain and Brain-to-Text
#
# NeuroVLM supports bidirectional querying:
#
# ### Text-to-Brain
# Given a text query, NeuroVLM can:
# 1. **Generate** brain activation patterns (generative approach)
# 2. **Retrieve** similar brain maps from datasets (contrastive approach)
#
# ### Brain-to-Text
# Given a brain activation map, NeuroVLM can:
# 1. **Retrieve** related scientific text, concepts, or descriptions
# 2. **Generate** text descriptions using language models (see Tutorial 4)

# %% [markdown]
# ## 5. Quick Examples
#
# Let's see both directions in action:

# %%
# Initialize the model
# Note: Models are lazy-loaded on first use (not here).
# The first call to .text() will load SPECTER (~500MB transformer) into RAM,
# which typically takes 1-3 minutes. All subsequent calls are < 5 seconds.
nvlm = NeuroVLM(device="cpu")

print("Model initialized successfully!")

# %% [markdown]
# ### Text-to-Brain: Generate brain maps from text

# %%
# Generate a brain map from text
# First run: loads SPECTER + projection heads + autoencoder into RAM (~1-3 min total)
# Subsequent runs in the same kernel session: < 5 seconds
# All computation runs on CPU - no MPS or CUDA used
result = nvlm.text("visual processing").to_brain(head="mse")

# Plot the generated brain map
result.plot(threshold=0.2);

# %% [markdown]
# ### Text-to-Brain: Retrieve similar brain maps

# %%
# Find brain maps similar to the text query
result = nvlm.text("working memory").to_brain(head="infonce")

# Show top matches
top = result.top_k(3)
top

# %%
# Visualize the top match
top.plot_row(6, threshold=0.1);

# %% [markdown]
# ### Brain-to-Text: Find text descriptions for brain maps

# %%
# Load example network atlases
networks_neuro = load_latent("networks_neuro")

# Use the Default Mode Network as a query
dmn = networks_neuro["Du"]["AUD"]

# Find related text
result = nvlm.brain(dmn).to_text(head="infonce")
top = result.top_k(5).query("cosine_similarity > 0.4") # return up to 5 examples per dataset within threshold
top

# %% [markdown]
# ## 6. Summary
#
# In this tutorial, you learned:
#
# 1. **How to fetch models and datasets** from HuggingFace using `fetch_data()`
# 2. **Model architectures** in NeuroVLM:
#    - SPECTER2 text encoder
#    - Autoencoder for brain compression
#    - Contrastive and generative projection heads
# 3. **Available datasets**:
#    - Text: PubMed, NeuroWiki, Cognitive Atlas, Networks
#    - Brain: PubMed images, NeuroVault, Network atlases
# 4. **Text-to-brain and brain-to-text** concepts
#
# In the following tutorials, you'll learn:
# - **Tutorial 2**: Contrastive retrieval for brain-to-text and text-to-brain
# - **Tutorial 3**: Generative text-to-brain mapping
# - **Tutorial 4**: Generative brain-to-text with LLMs
