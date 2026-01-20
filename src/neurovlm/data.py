"""Fetch required data from Hugging Face.

This module provides utilities to download and cache NeuroVLM data from
Hugging Face repositories. All data is cached using the huggingface_hub
default cache mechanism.
"""

import os
from pathlib import Path
from typing import Optional, List
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE

# Hugging Face repository information
REPO_DATASETS = {
    "neuro_image_papers": "neurovlm/neuro_image_papers",
    "neuro_wiki": "neurovlm/neuro_wiki",
    "cognitive_atlas": "neurovlm/cognitive_atlas",
    "embedded_text": "neurovlm/embedded_text",
}

REPO_MODELS = {
    "encoder_and_proj_head": "neurovlm/encoder_and_proj_head",
}


def fetch_data(
    datasets: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    cache_dir: Optional[str] = None,
) -> str:
    """Fetch NeuroVLM data from Hugging Face repositories.

    This function downloads all required datasets and models from Hugging Face
    and caches them locally. By default, it fetches all available repositories.

    Parameters
    ----------
    datasets : list of str, optional
        List of dataset repository keys to download. If None, downloads all datasets.
        Available keys: "neuro_image_papers", "neuro_wiki", "cognitive_atlas", "embedded_text"
    models : list of str, optional
        List of model repository keys to download. If None, downloads all models.
        Available keys: "encoder_and_proj_head"
    cache_dir : str, optional
        Custom cache directory. If None, uses Hugging Face default cache.

    Returns
    -------
    cache_dir : str
        Path to the cache directory where data is stored.

    Examples
    --------
    >>> # Fetch all data
    >>> cache_dir = fetch_data()

    >>> # Fetch only specific datasets
    >>> cache_dir = fetch_data(datasets=["neuro_image_papers", "cognitive_atlas"])

    >>> # Fetch only models
    >>> cache_dir = fetch_data(datasets=[], models=["encoder_and_proj_head"])
    """
    # Use default datasets/models if not specified
    if datasets is None:
        datasets = list(REPO_DATASETS.keys())
    if models is None:
        models = list(REPO_MODELS.keys())

    # Determine cache directory
    if cache_dir is None:
        cache_dir = HUGGINGFACE_HUB_CACHE

    print(f"Downloading NeuroVLM data to: {cache_dir}")

    # Download datasets
    for dataset_key in datasets:
        if dataset_key not in REPO_DATASETS:
            print(f"Warning: Unknown dataset key '{dataset_key}', skipping...")
            continue

        repo_id = REPO_DATASETS[dataset_key]
        print(f"Downloading dataset: {repo_id}")
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                cache_dir=cache_dir,
            )
            print(f"✓ Successfully downloaded {repo_id}")
        except Exception as e:
            print(f"✗ Error downloading {repo_id}: {e}")

    # Download models
    for model_key in models:
        if model_key not in REPO_MODELS:
            print(f"Warning: Unknown model key '{model_key}', skipping...")
            continue

        repo_id = REPO_MODELS[model_key]
        print(f"Downloading model: {repo_id}")
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                cache_dir=cache_dir,
            )
            print(f"✓ Successfully downloaded {repo_id}")
        except Exception as e:
            print(f"✗ Error downloading {repo_id}: {e}")

    print(f"\n✓ Data fetch complete. Cache directory: {cache_dir}")
    return cache_dir


def get_data_dir() -> Path:
    """Return the path to the Hugging Face cache directory.

    This is where all NeuroVLM data from Hugging Face is stored.
    The cache is managed by huggingface_hub and shared across all projects.

    Returns
    -------
    Path
        Path to the Hugging Face cache directory.

    Notes
    -----
    The default cache directory is typically:
    - Linux/Mac: ~/.cache/huggingface/hub
    - Windows: %USERPROFILE%\\.cache\\huggingface\\hub

    You can override this by setting the HF_HOME environment variable.
    """
    return Path(HUGGINGFACE_HUB_CACHE)


def preload_all_data(cache_dir: Optional[str] = None, verbose: bool = True) -> None:
    """Preload all NeuroVLM data by calling all retrieval functions.

    This ensures all data is downloaded and cached before use. It's useful
    for preparing an environment or warming up the cache.

    Parameters
    ----------
    cache_dir : str, optional
        Custom cache directory. If None, uses Hugging Face default cache.
    verbose : bool, optional
        If True, prints progress messages.

    Examples
    --------
    >>> # Preload all data before running experiments
    >>> preload_all_data()
    """
    if verbose:
        print("Preloading all NeuroVLM data from Hugging Face...")

    # Import retrieval functions
    from neurovlm.retrieval_resources import (
        _load_dataframe,
        _load_neuro_wiki,
        _load_cogatlas_dataset,
        _load_cogatlas_task_dataset,
        _load_cogatlas_disorder_dataset,
        _load_latent_text,
        _load_latent_wiki,
        _load_latent_cogatlas,
        _load_latent_cogatlas_disorder,
        _load_latent_cogatlas_task,
        _load_autoencoder,
        _load_masker,
        _load_networks,
        _proj_head_image_infonce,
        _proj_head_mse_sparse_adhoc,
        _proj_head_text_infonce,
        _load_specter,
    )

    loaders = [
        ("Publications dataframe", _load_dataframe),
        ("NeuroWiki dataframe", _load_neuro_wiki),
        ("CogAtlas concepts", _load_cogatlas_dataset),
        ("CogAtlas tasks", lambda: _load_cogatlas_task_dataset(filtered=True)),
        ("CogAtlas disorders", _load_cogatlas_disorder_dataset),
        ("Latent text embeddings", _load_latent_text),
        ("Latent wiki embeddings", _load_latent_wiki),
        ("Latent CogAtlas embeddings", _load_latent_cogatlas),
        ("Latent CogAtlas disorder embeddings", _load_latent_cogatlas_disorder),
        ("Latent CogAtlas task embeddings", _load_latent_cogatlas_task),
        ("Autoencoder model", _load_autoencoder),
        ("Brain masker", _load_masker),
        ("Network atlases", _load_networks),
        ("Image projection head", _proj_head_image_infonce),
        ("MSE projection head", _proj_head_mse_sparse_adhoc),
        ("Text projection head", _proj_head_text_infonce),
        ("SPECTER model", _load_specter),
    ]

    for name, loader in loaders:
        if verbose:
            print(f"Loading {name}...", end=" ")
        try:
            loader()
            if verbose:
                print("✓")
        except Exception as e:
            if verbose:
                print(f"✗ Error: {e}")

    if verbose:
        print("\n✓ All data preloaded successfully!")


# For backward compatibility, keep data_dir
data_dir = get_data_dir()
