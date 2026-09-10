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
# # Historical architecture exploration
#
# The original ResNet48/multi-scale-attention notebook documented an exploratory architecture branch. It is retained as design history, not as the supported training recipe. The released pipeline uses the retained four-block atlas-free autoencoder and the package registry so training and inference resolve the same architecture.

# %% [markdown]
# ## Why the retained model is the supported path
#
# The integrated implementation fixes the input space at `(1, 36, 45, 38)`, uses a 384-dimensional latent representation, records architecture metadata in every checkpoint, and validates it on resume. Custom architecture research remains possible by setting `preset="custom"` in a training config, but it does not silently replace released weights.

# %%
from neurovlm.models.registry import resolve_model_spec
from neurovlm.training import AutoencoderTrainConfig

spec = resolve_model_spec(family="cnn", task="autoencoder")
config = AutoencoderTrainConfig()
spec.canonical_name, config.architecture()
