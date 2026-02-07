.. _api_documentation:

=================
API Documentation
=================

API reference for the timescales module.

Table of Contents
=================

.. contents::
   :local:
   :depth: 1

Data
----

Fetches from huggingface and loads as tensors or dataframes.

Fetching
~~~~~~~~

.. currentmodule:: neurovlm.data

.. autosummary::
   :toctree: generated/

   fetch_data
   load_dataset

Embeddings
~~~~~~~~~~

Precomputed latent vectors for text and neuroimages.

.. currentmodule:: neurovlm.data

.. autosummary::
   :toctree: generated/

   load_latent

Masker
~~~~~~

Nifti masker need to resample and mask neuroimages.

.. currentmodule:: neurovlm.data

.. autosummary::
   :toctree: generated/

   load_masker

Models
------

Base models for autoencoder, projection heads, and specter.
Pretrained models return from load_model or calling .from_pretrained on model classes.

.. currentmodule:: neurovlm.models

.. autosummary::
   :toctree: generated/

   NeuroAutoEncoder
   ProjHead
   Specter
   load_model

Loss Functions
--------------

The pretrained models used InfoNCELoss or MSELoss. Additional options include FocalLoss or TruncatedLoss.

.. currentmodule:: neurovlm.loss

.. autosummary::
   :toctree: generated/

   InfoNCELoss
   FocalLoss
   TruncatedLoss

Training
--------

Convenience wrapper for training: a standard PyTorch training loop.

.. currentmodule:: neurovlm.train

.. autosummary::
   :toctree: generated/

   Trainer
   which_device

Metrics
-------

Performance metrics.

.. currentmodule:: neurovlm.metrics

.. autosummary::
   :toctree: generated/

   recall_at_k
   recall_curve
   dice
   dice_top_k
   bernoulli_bce
   bits_per_pixel
