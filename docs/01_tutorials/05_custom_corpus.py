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
# # Tutorial 5: Searching Your Own Corpus
#
# The `NeuroVLM` high-level API (used in the previous tutorials) searches NeuroVLM's
# built-in datasets — PubMed abstracts, NeuroWiki, Cognitive Atlas, and canonical
# brain networks.  
#
# Sometimes you have **your own text corpus** — a set of study descriptions, clinical
# notes, lab protocols, or any collection of named entries — and you want to rank it
# against a brain map or a text query using the same contrastive model.  
# That is what `neurovlm.retrieval.user` is for.
#
# This tutorial covers:
#
# 1. The key difference between `NeuroVLM` (core) and `user_retrieval`
# 2. Preparing your corpus (format + pre-embedding)
# 3. `search_text_corpus_given_text` — rank your corpus against a text query
# 4. `search_text_corpus_given_neuroimage` — rank your corpus against a brain map
# 5. `generate_llm_response` — get a language-model summary of the top results
# 6. NIfTI resampling helpers

# %% [markdown]
# ## 1. Core API vs `user_retrieval` — what is the difference?
#
# | | `NeuroVLM` (core) | `user_retrieval` |
# |---|---|---|
# | **Corpus** | NeuroVLM built-in datasets (PubMed, NeuroWiki, CogAtlas, networks) | **Your own** DataFrame |
# | **Query types** | Text string, NIfTI, latent tensor, pre-embedded tensor | Text string, NIfTI |
# | **Output** | `TextSearchResult` / `BrainSearchResult` (chainable) | Plain `pandas.DataFrame` |
# | **LLM helper** | `nvlm.generate_llm_response(...)` | `generate_llm_response(context_df, ...)` |
# | **Embeddings** | Loaded automatically from HuggingFace | You pre-embed your corpus with SPECTER |
#
# Both routes use the **same contrastive model weights** underneath, so scores are
# directly comparable.

# %%
import os
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import pandas as pd
import nibabel as nib

from neurovlm.retrieval.user import (
    search_text_corpus_given_text,
    search_text_corpus_given_neuroimage,
    generate_llm_response,
    resample_nifti,
)

# %% [markdown]
# ## 2. Preparing your corpus
#
# Your corpus must be a `pandas.DataFrame` with at least two columns:
#
# - **`name`** — a short label (study title, concept name, …)
# - **`description`** — longer text that will also be embedded
#
# Before calling any search function you also need to **pre-embed** every row
# using SPECTER, the same text encoder NeuroVLM was trained with.  
# Pass the name and description concatenated (or just the description — whatever
# best captures the semantics of that row).
#
# Stack all embeddings into a single `torch.Tensor` of shape `(N, 768)` where
# `N` is the number of rows.

# %%
# ── Build a small toy corpus ────────────────────────────────────────────────
corpus = pd.DataFrame([
    {
        "name": "Default Mode Network",
        "description": (
            "A set of brain regions that show deactivation during externally "
            "directed tasks and activation at rest, associated with "
            "self-referential thinking, mind-wandering, and autobiographical memory."
        ),
    },
    {
        "name": "Working Memory",
        "description": (
            "A cognitive system with limited capacity that temporarily holds "
            "information available for processing, strongly linked to the "
            "dorsolateral prefrontal cortex and parietal regions."
        ),
    },
    {
        "name": "Visual Cortex",
        "description": (
            "The primary and extrastriate visual cortex located in the occipital "
            "lobe, responsible for processing visual information including shape, "
            "color, and motion."
        ),
    },
    {
        "name": "Amygdala",
        "description": (
            "A subcortical structure in the medial temporal lobe critically "
            "involved in emotional processing, threat detection, fear conditioning, "
            "and modulating memory consolidation for emotionally significant events."
        ),
    },
    {
        "name": "Motor Cortex",
        "description": (
            "The primary motor cortex (Brodmann area 4) generates the neural "
            "impulses controlling voluntary movement execution. It is somatotopically "
            "organized, with the homunculus representing body parts along its surface."
        ),
    },
    {
        "name": "Auditory Cortex",
        "description": (
            "Located in the temporal lobe (Heschl's gyrus), the primary auditory "
            "cortex processes sound frequency, rhythm, and speech, with higher-order "
            "regions handling language comprehension."
        ),
    },
    {
        "name": "Hippocampus",
        "description": (
            "A medial temporal lobe structure essential for the formation of new "
            "declarative memories (episodic and semantic), spatial navigation, and "
            "consolidation of short-term to long-term memory."
        ),
    },
    {
        "name": "Cerebellum",
        "description": (
            "Coordinates voluntary movement, balance, and fine motor control. "
            "Also implicated in cognitive functions including timing, language, "
            "and emotional processing."
        ),
    },
])

print(f"Corpus: {len(corpus)} entries")
corpus[["name", "description"]]

# %%
# ── Pre-embed with SPECTER ───────────────────────────────────────────────────
# NeuroVLM ships a SPECTER wrapper — use it so your embeddings live in the
# same space the model was trained in.
from neurovlm.resources.loaders import _load_specter

specter = _load_specter()

# Embed name + description concatenated for each row
texts = [
    f"{row['name']}. {row['description']}"
    for _, row in corpus.iterrows()
]

with torch.no_grad():
    # specter() returns a tuple; index [0] is the [CLS] token embedding
    embeddings = torch.stack([specter(t)[0] for t in texts])  # (N, 768)

print(f"Embeddings shape: {embeddings.shape}")

# %% [markdown]
# > **Tip — large corpora**  
# > For thousands of entries, embed in mini-batches and save the tensor with
# > `torch.save(embeddings, "my_corpus_embeddings.pt")`.  
# > Load it back later with `embeddings = torch.load("my_corpus_embeddings.pt")` —
# > no need to re-embed on every run.

# %% [markdown]
# ## 3. `search_text_corpus_given_text`
#
# Rank your corpus against a **natural-language query**.  
# Under the hood:
#
# 1. The query string is encoded with SPECTER (768-d)
# 2. Both query and corpus embeddings are projected into the shared contrastive
#    space via the text projection head
# 3. Cosine similarity is computed and the top-k rows are returned

# %%
results = search_text_corpus_given_text(
    query="memory and learning",
    corpus_df=corpus,
    corpus_embeddings=embeddings,
    top_k=5,
    show_names=True,
)
results

# %%
# Try a different query
results_motor = search_text_corpus_given_text(
    query="voluntary movement",
    corpus_df=corpus,
    corpus_embeddings=embeddings,
    top_k=3,
)
results_motor

# %% [markdown]
# ### How is this different from `NeuroVLM.to_text`?
#
# ```python
# # Core API — searches the built-in NeuroVLM datasets
# from neurovlm import NeuroVLM
# nvlm = NeuroVLM()
# result = nvlm.text("memory and learning").to_text(head="infonce")
# result.top_k(5)
# ```
#
# The core API automatically loads and projects the NeuroVLM corpora (PubMed,
# NeuroWiki, CogAtlas).  `search_text_corpus_given_text` does the same projection
# but **against your DataFrame** — the model weights are identical.

# %% [markdown]
# ## 4. `search_text_corpus_given_neuroimage`
#
# Rank your corpus against a **brain activation map** (NIfTI).  
# Under the hood:
#
# 1. The NIfTI image is resampled to the project mask
# 2. The flattened brain vector is encoded by the brain autoencoder
# 3. The latent is projected into the shared space via the **image** projection head
# 4. Your corpus embeddings are projected via the **text** projection head
# 5. Cosine similarity is computed
#
# For this example we will use a network atlas to get a real NIfTI image.

# %%
from neurovlm.data import load_latent, load_masker

# Load a canonical network as a latent tensor (already encoded)
# To get a raw NIfTI we decode it back through the masker
networks = load_latent("networks_neuro")

# Grab the Default Mode Network latent from YeoLab
dmn_latent = networks["YeoLab"]["DefaultA"]  # torch.Tensor (384,)
print(f"Network latent shape: {dmn_latent.shape}")

# %%
# Decode back to a NIfTI image so we can test the neuroimage search path
from neurovlm.resources.loaders import _load_autoencoder, _load_masker as _masker_fn

autoencoder = _load_autoencoder()
masker = _masker_fn()

with torch.no_grad():
    flat_decoded = torch.sigmoid(
        autoencoder.decoder(dmn_latent.unsqueeze(0))
    ).squeeze(0).cpu().numpy()  # (28542,)

dmn_nifti = masker.inverse_transform(flat_decoded.reshape(1, -1))
print(f"Decoded NIfTI shape: {dmn_nifti.shape}")

# %%
# Now rank our custom corpus against this brain map
results_brain = search_text_corpus_given_neuroimage(
    neuroimage=dmn_nifti,
    corpus_df=corpus,
    corpus_embeddings=embeddings,
    top_k=5,
    show_names=True,
)
results_brain

# %% [markdown]
# The DMN should rank **Default Mode Network** and possibly **Hippocampus** (memory) near the top.
#
# ### How is this different from `NeuroVLM.brain(...).to_text(head="infonce")`?
#
# ```python
# # Core API — searches NeuroVLM's built-in corpora
# nvlm = NeuroVLM()
# result = nvlm.brain(dmn_nifti).to_text(head="infonce")
# result.top_k(5)
# ```
#
# The core API compares the brain map against PubMed, NeuroWiki, CogAtlas, and networks.
# `search_text_corpus_given_neuroimage` compares it against **your** corpus instead,
# with the same model.

# %% [markdown]
# ## 5. `generate_llm_response`
#
# Pass the ranked results DataFrame into `generate_llm_response` to get a
# natural-language interpretation from a local LLM.
#
# The `query_type` argument tells the LLM whether the context was derived from
# a **brain map** (`"neuroimage"`) or a **text query** (`"text"`), which controls
# the system prompt framing.
#
# Two backends are supported:
# - **`"ollama"`** — requires [Ollama](https://ollama.com) installed and running locally (fast)
# - **`"huggingface"`** — downloads and runs the model in-process (works offline, uses more RAM)

# %%
# ── Text-to-corpus + LLM summary ────────────────────────────────────────────
results_memory = search_text_corpus_given_text(
    query="memory and learning",
    corpus_df=corpus,
    corpus_embeddings=embeddings,
    top_k=4,
)

response = generate_llm_response(
    context_df=results_memory,
    query_type="text",          # context came from a text query
    backend="ollama",
    model_name="qwen2.5:3b-instruct",
    user_prompt="What brain regions are most relevant to memory and learning?",
)
print(response)

# %%
# ── Brain-to-corpus + LLM summary ───────────────────────────────────────────
response_brain = generate_llm_response(
    context_df=results_brain,
    query_type="neuroimage",    # context came from a brain map
    backend="ollama",
    model_name="qwen2.5:3b-instruct",
    user_prompt="What does this brain activation pattern suggest?",
)
print(response_brain)

# %%
# ── HuggingFace backend (no Ollama required) ─────────────────────────────────
# response_hf = generate_llm_response(
#     context_df=results_memory,
#     query_type="text",
#     backend="huggingface",
#     model_name="Qwen/Qwen2.5-0.5B-Instruct",   # ~1 GB
# )
# print(response_hf)

# %% [markdown]
# ### How is this different from `nvlm.generate_llm_response`?
#
# ```python
# # Core API — LLM sees NeuroVLM's built-in retrieval results
# nvlm.brain(dmn_nifti).to_text(head="infonce")
# nvlm.generate_llm_response(backend="ollama", model_name="qwen2.5:3b-instruct")
# ```
#
# `nvlm.generate_llm_response` automatically draws context from the last
# `to_text()` / `to_brain()` call on the built-in corpora.  
# `user_retrieval.generate_llm_response` instead takes an **explicit** DataFrame
# (from your own search), so you fully control what the LLM sees.  
# The LLM backends and model defaults are the same in both paths.

# %% [markdown]
# ## 6. NIfTI resampling helpers
#
# `user_retrieval` also has NIfTI utilities. These are useful when you need to pre-process your own
# brain maps before encoding them.

# %%
from neurovlm.retrieval.user import resample_nifti, resample_networks_to_mask

# resample_nifti — resample any NIfTI to the project mask and return a flat tensor
flat_tensor = resample_nifti(dmn_nifti)
print(f"Flat brain vector shape: {flat_tensor.shape}")  # (28542,)

# This tensor can then be encoded manually:
from neurovlm.resources.loaders import _load_autoencoder
autoencoder = _load_autoencoder()
with torch.no_grad():
    latent = autoencoder.encoder(flat_tensor.unsqueeze(0))  # (1, 384)
print(f"Brain latent shape: {latent.shape}")

# %%
# resample_networks_to_mask — bulk-resample a dict of network arrays
# Useful when you have your own atlas stored as raw numpy arrays

import numpy as np

# Simulate two custom network arrays (shape matches the NIfTI)
shape = dmn_nifti.shape[:3]
affine = dmn_nifti.affine

custom_networks = {
    "MyNet_A": {
        "array": (np.random.rand(*shape) > 0.95).astype(float),
        "affine": affine,
    },
    "MyNet_B": {
        "array": (np.random.rand(*shape) > 0.95).astype(float),
        "affine": affine,
    },
}

resampled = resample_networks_to_mask(custom_networks)
for name, img in resampled.items():
    print(f"{name}: {img.shape}")

# %% [markdown]
# ## 7. Full end-to-end example
#
# Here is a concise version of the complete workflow in one place.

# %%
import torch, pandas as pd
from neurovlm.resources.loaders import _load_specter
from neurovlm.retrieval.user import (
    search_text_corpus_given_text,
    search_text_corpus_given_neuroimage,
    generate_llm_response,
)

# 1. Your corpus
my_corpus = pd.DataFrame([
    {"name": "Concept A", "description": "..."},
    {"name": "Concept B", "description": "..."},
])

# 2. Pre-embed
specter = _load_specter()
texts = [f"{r['name']}. {r['description']}" for _, r in my_corpus.iterrows()]
with torch.no_grad():
    my_embeddings = torch.stack([specter(t)[0] for t in texts])  # (N, 768)

# 3a. Rank against text
results = search_text_corpus_given_text(
    query="your query here",
    corpus_df=my_corpus,
    corpus_embeddings=my_embeddings,
    top_k=5,
)

# 3b. Rank against a NIfTI brain map
# results = search_text_corpus_given_neuroimage(
#     neuroimage=my_nifti,
#     corpus_df=my_corpus,
#     corpus_embeddings=my_embeddings,
#     top_k=5,
# )

# 4. LLM summary
# response = generate_llm_response(
#     context_df=results,
#     query_type="text",          # or "neuroimage"
#     backend="ollama",
#     model_name="qwen2.5:3b-instruct",
# )

results

# %% [markdown]
# ## 8. Summary
#
# | Function | When to use |
# |---|---|
# | `search_text_corpus_given_text` | You have a text query and want to rank your own corpus |
# | `search_text_corpus_given_neuroimage` | You have a brain map and want to rank your own corpus |
# | `generate_llm_response` | You want a natural-language summary of ranked results from your corpus |
# | `resample_nifti` | You need to flatten a NIfTI into the project mask space |
# | `resample_networks_to_mask` | You need to bulk-resample network arrays to the mask affine |
#
# When working with NeuroVLM's built-in datasets (PubMed, CogAtlas, NeuroWiki, canonical
# networks), use `NeuroVLM` from `neurovlm.core` as shown in the earlier tutorials.  
# When you have your **own** corpus, use the functions in `neurovlm.retrieval.user`.
