.. _api_documentation:

=================
API Documentation
=================

API reference for the `neurovlm` module.

Table of Contents
=================

.. contents::
   :local:
   :depth: 1

Inference
---------

The legacy high-level interface remains available for text, NIfTI, and
retrieval workflows.

.. currentmodule:: neurovlm.core

.. autosummary::
   :toctree: generated/

   NeuroVLM
   BrainSearchResult
   TextSearchResult
   BrainTopKResult

Structured runtime
~~~~~~~~~~~~~~~~~~

Task-oriented tensor inference selects the family, task, domain, variant, and
released checkpoint or local run explicitly. CNN domain tasks default to the
mixed baseline; fine-tuning is explicit.

.. currentmodule:: neurovlm.runtime

.. autosummary::
   :toctree: generated/

   load_pipeline
   NeuroVLMRuntime
   RuntimeMetadata


Data
----

Fetches from huggingface and loads.

Fetching
~~~~~~~~

.. currentmodule:: neurovlm.data

.. autosummary::
   :toctree: generated/

   fetch_data
   load_dataset

Embeddings
~~~~~~~~~~

Pre-computed latent vectors for text and neuroimages.

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

Atlas-free CNN datasets
~~~~~~~~~~~~~~~~~~~~~~~

Published split JSONLs and their shared volume tensor. Legacy per-row local
paths are ignored.

.. currentmodule:: neurovlm.atlas_free_dataset

.. autosummary::
   :toctree: generated/

   AtlasFreeCNNDataset
   AtlasFreeCNNDataProvider
   atlas_free_cnn_splits

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

The structured selectors are defined in the model registry:

.. currentmodule:: neurovlm.model_registry

.. autosummary::
   :toctree: generated/

   ModelFamily
   ModelTask
   ModelDomain
   ModelVariant
   ModelSpec
   resolve_model_spec

Atlas-Free CNN
~~~~~~~~~~~~~~

Installable 3D CNN architectures and conversion helpers for the MLP and CNN
input spaces. Pretrained instances are returned by ``load_model``.

.. currentmodule:: neurovlm.cnn

.. autosummary::
   :toctree: generated/

   CNNContrastiveModel
   CNNTextToBrainModel
   atlas_free_volume_to_mlp_flat
   mlp_flat_to_atlas_free_volume

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

The original generic PyTorch trainer remains supported.

.. currentmodule:: neurovlm.train

.. autosummary::
   :toctree: generated/

   Trainer
   which_device

Standardized task runners
~~~~~~~~~~~~~~~~~~~~~~~~~

Typed runners share artifact, metric, checkpoint, provenance, and resume
conventions.

.. currentmodule:: neurovlm.training

.. autosummary::
   :toctree: generated/

   AutoencoderTrainConfig
   ContrastiveTrainConfig
   TextToBrainTrainConfig
   MLPAutoencoderTrainConfig
   MLPContrastiveTrainConfig
   MLPTextToBrainTrainConfig
   MLPBrainToTextRetrievalTrainConfig
   BrainToTextGenerationTrainConfig
   train_autoencoder
   train_contrastive
   train_text_to_brain
   train_mlp_autoencoder
   train_mlp_contrastive
   train_mlp_text_to_brain
   train_mlp_brain_to_text_retrieval
   train_brain_to_text_generation

Model comparison
----------------

Shared MLP/CNN reconstruction, retrieval, and generation comparisons. The
default matrix uses mixed-baseline CNN checkpoints; fine-tuned rows are an
explicit opt-in.

.. currentmodule:: neurovlm.evaluation

.. autosummary::
   :toctree: generated/

   ComparisonSelection
   ComparisonResult
   default_comparison_matrix
   evaluate_reconstruction_comparison
   evaluate_contrastive_comparison
   evaluate_text_to_brain_comparison

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
