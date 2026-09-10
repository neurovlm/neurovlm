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
#     display_name: .env (3.12.3)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tutorial 2: Contrastive Retrieval
#
# This tutorial demonstrates contrastive learning approaches in NeuroVLM for:
#
# 1. **Brain-to-Text**: Labeling brain networks and activation maps
# 2. **Text-to-Brain**: Finding similar brain maps from text queries
#
# We'll cover:
# - Network labeling with canonical atlases
# - ICA component labeling (HCP, UK Biobank)
# - NeuroVault map labeling
# - Text-to-brain retrieval across multiple datasets

# %%
import os

os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from neurovlm import NeuroVLM
from neurovlm.data import load_latent

# Initialize model
nvlm = NeuroVLM()

# %% [markdown]
# ## 1. Brain-to-Text: Network Labeling
#
# Given a brain network or activation map, we can retrieve the most semantically similar text from different datasets. This is useful for:
# - Automatically labeling brain networks
# - Understanding what cognitive processes are associated with activation patterns
# - Validating network definitions against literature
#
# ***Note*** - The threshold serves as a relevance filter for Brain-to-text network labeling — terms appearing in the top-k results are not guaranteed to be meaningfully related to the neuroimaging data. A threshold of 0.4 is generally a reliable cutoff, as most brain-to-text scores above this value correspond to significant associations. While exceptions exist, this filter effectively removes spurious or unrelated terms.

# %% [markdown]
# ### Load Network Atlases
#
# We'll work with canonical brain networks from multiple atlases.

# %%
# Load pre-encoded network atlases
networks = load_latent("networks_neuro")

# Available atlases
print("Available atlases:")
for atlas_name in networks.keys():
    print(f"  {atlas_name}: {len(networks[atlas_name])} networks")

# %% [markdown]
# ### Example: Label the Auditory Network

# %%
# Use the auditory network from Du et al. atlas
auditory_network = networks["Du"]["AUD"]

# Find related text across all datasets
result = nvlm.brain(auditory_network).to_text(head="infonce")

# Show top 5 matches per dataset per dataset within threshold
top = result.top_k(5).query("cosine_similarity > 0.4")
top

# %% [markdown]
# ### Query Specific Datasets
#
# You can also search specific datasets like Cognitive Atlas concepts, tasks, or disorders.

# %%
# Find related cognitive concepts
result = nvlm.brain(auditory_network).to_text(head="infonce")
concepts = result.top_k(5, dataset="cogatlas")
print("\nTop Cognitive Atlas Concepts:")
concepts

# %%
# Find related scientific papers
papers = result.top_k(5, dataset="pubmed")
print("\nTop Related Papers:")
papers

# %%
# Find related Wikipedia entries
wiki = result.top_k(5, dataset="wiki")
print("\nTop NeuroWiki Entries:")
wiki

# %% [markdown]
# ### Example: Label the Default Mode Network

# %%
# Default mode network from Yeo et al. atlas
dmn = networks["YeoLab"]["DefaultA"]

result = nvlm.brain(dmn).to_text(head="infonce")

# top 5 matches per dataset within threshold
top = result.top_k(5).query("cosine_similarity > 0.4")
top

# %% [markdown]
# ### Example: Label a Motor Network

# %%
# Somatomotor network
motor = networks["YeoLab"]["SomMotA"]

result = nvlm.brain(motor).to_text(head="infonce")
top = result.top_k(5).query("cosine_similarity > 0.4")
top

# %% [markdown]
# ## 2. ICA Component Labeling
#
# Independent Component Analysis (ICA) is commonly used to identify brain networks from resting-state fMRI data (e.g., HCP, UK Biobank). NeuroVLM can automatically label these components.

# %%
# Example: Label ICA components from the HCP ICA atlas
ica_component = networks["HCPICA"]["ICA10"]

result = nvlm.brain(ica_component).to_text(head="infonce")
top = result.top_k(3)
print("ICA Component 10 is most similar to:")
top

# %% [markdown]
# ## 3. NeuroVault Map Labeling
#
# For arbitrary brain activation maps (e.g., from your own study or NeuroVault), you can:
# 1. Load the NIfTI image
# 2. Query it against text datasets

# %%
# Example with visual network
visual_network = networks["YeoLab"]["VisualA"]

result = nvlm.brain(visual_network).to_text(head="infonce")
top = result.top_k(5)
print("Visual Network Labels:")
top

# %% [markdown]
# ## 4. Text-to-Brain: Finding Similar Brain Maps
#
# Given a text query, retrieve the most similar brain activation patterns from datasets.
#
# Note - The brain-to-text filtering threshold of 0.4 does not apply to text-brain

# %% [markdown]
# ### Example: Find brain maps for "emotion processing"

# %%
# Search across all brain datasets
result = nvlm.text("emotion processing").to_brain(head="infonce")

# Get top 5 matches
top = result.top_k(5)
top.table = top.table.sort_values('cosine_similarity', ascending=False).reset_index(drop=True)
top

# %% [markdown]
# ### Example: Search specific brain datasets

# %%
# Search only in canonical networks
result = nvlm.text("attention").to_brain(head="infonce", dataset="networks")
top = result.top_k(3)
print("Top attention-related networks:")
top

# %%
# Search only in PubMed activation maps
result = nvlm.text("working memory").to_brain(head="infonce", dataset="pubmed")
top = result.top_k(3)
print("Top working memory studies:")
top

# %% [markdown]
# ### Example: Multi-dataset retrieval

# %%
# Search across multiple datasets
result = nvlm.text("language comprehension").to_brain(
    head="infonce",
    dataset=["networks", "neurovault"]
)
top = result.top_k(5)
top.table = top.table.sort_values('cosine_similarity', ascending=False).reset_index(drop=True)
top

# %%
# Plot top matches from different datasets
top.plot_row(0, threshold=0.1, title="Top match - Networks");

# %%
top.plot_row(5, threshold=0.1, title="Top match - NeuroVault");

# %% [markdown]
# ## 5. Comparing Multiple Queries
#
# You can run multiple queries at once and compare results.

# %%
# Multiple text queries
queries = [
    "visual perception",
    "motor control",
    "executive function"
]

result = nvlm.text(["vision", "default mode network"]).to_brain(head="mse")
result.to_nifti() # returns list of nib.Nifti1Image

# %%
result.plot(0, threshold=0.25); # plot image for vision

# %%
result.plot(1, threshold=0.15); # plot image for DMN

# %% [markdown]
# ## 6. Batch Network Labeling
#
# Label multiple networks at once for systematic comparison.

# %%
# Select multiple networks to label
import torch

network_names = ["VIS-P", "AUD", "SMOT-A", "DN-A", "LANG"]
network_latents = [networks["Du"][name] for name in network_names]

# Stack into batch
batch = torch.stack(network_latents)

# Label all at once
result = nvlm.brain(batch).to_text(head="infonce")

# Show top concept for each network
results = []
for i, name in enumerate(network_names):
    print(f"\n=== {name} ===")
    top = result.top_k(3, query_index=i, dataset="cogatlas")
    results.append(top)

# %%
results[0] # top concepts for "VIS-P"

# %%
results[1] # top concepts for "AUD"

# %%
results[2] # top concepts for SMOT-A network

# %% [markdown]
# ## 7. Summary
#
# In this tutorial, you learned:
#
# 1. **Brain-to-Text** retrieval:
#    - Network labeling using canonical atlases
#    - ICA component labeling
#    - Querying specific datasets (PubMed, Cognitive Atlas, NeuroWiki)
#    
# 2. **Text-to-Brain** retrieval:
#    - Finding similar brain maps from text queries
#    - Searching specific datasets (networks, neurovault, pubmed)
#    - Multi-dataset retrieval
#    
# 3. **Advanced usage**:
#    - Batch processing multiple queries
#    - Systematic network labeling
#
# **Next**: In Tutorial 3, you'll learn how to generate brain activation maps from text using the generative approach.

# %% [markdown]
#
