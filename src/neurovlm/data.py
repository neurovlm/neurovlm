"""Fetch required data from Hugging Face.

This module provides utilities to download and cache NeuroVLM data from
Hugging Face repositories. All data is cached using the huggingface_hub
default cache mechanism.
"""

import os
from pathlib import Path
from typing import Optional, List
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
import pandas as pd
from neurovlm.retrieval_resources import (
    _load_pubmed_dataframe,
    _load_pubmed_coordinates,
    _load_neuro_wiki,
    _load_neuro_wiki_graph,
    _load_cogatlas_dataset,
    _load_cogatlas_task_dataset,
    _load_cogatlas_disorder_dataset,
    _load_cogatlas_graph_dataset,
    _load_networks,
    _load_networks_labels,
    _load_networks_canonical,
    _load_latent_text,
    _load_latent_neuro,
    _load_latent_networks_neuro,
    _load_latent_networks_canonical_text,
    _load_latent_wiki,
    _load_latent_cogatlas,
    _load_latent_cogatlas_disorder,
    _load_latent_cogatlas_task,
    _load_latent_neurovault_images,
    _load_latent_neurovault_text,
    _load_publications_neurovault_dataframe,
    _load_images_neurovault_dataframe,
    _load_neurovault_images,
    _load_pubmed_images,
    _load_autoencoder,
    _load_masker,
    _proj_head_image_infonce,
    _proj_head_text_mse,
    _proj_head_text_infonce,
    _load_specter
)


# Hugging Face repository information
REPO_DATASETS = {
    "neuro_image_papers": "neurovlm/neuro_image_papers",
    "neuro_wiki": "neurovlm/neuro_wiki",
    "cognitive_atlas": "neurovlm/cognitive_atlas",
    "embedded_text": "neurovlm/embedded_text"
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

    status_width = 0

    def _print_status(message: str) -> None:
        nonlocal status_width
        status_width = max(status_width, len(message))
        # Keep progress on a single line and erase leftovers from prior messages.
        print(f"\r{message.ljust(status_width)}", end="", flush=True)

    _print_status(f"Downloading NeuroVLM data to: {cache_dir}")

    # Download datasets
    for dataset_key in datasets:
        if dataset_key not in REPO_DATASETS:
            _print_status(f"Warning: Unknown dataset key '{dataset_key}', skipping...")
            continue

        repo_id = REPO_DATASETS[dataset_key]
        _print_status(f"Downloading dataset: {repo_id}")
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                cache_dir=cache_dir,
            )
            _print_status(f"Successfully downloaded {repo_id}")
        except Exception as e:
            _print_status(f"Error downloading {repo_id}: {e}")

    # Download models
    for model_key in models:
        if model_key not in REPO_MODELS:
            _print_status(f"Warning: Unknown model key '{model_key}', skipping...")
            continue

        repo_id = REPO_MODELS[model_key]
        _print_status(f"Downloading model: {repo_id}")
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                cache_dir=cache_dir,
            )
            _print_status(f"Successfully downloaded {repo_id}")
        except Exception as e:
            _print_status(f"Error downloading {repo_id}: {e}")

    print(f"\r{' ' * status_width}\r", end="", flush=True)
    print(f"Data fetch complete. Cache directory: {cache_dir}")
    return cache_dir


def get_data_dir() -> Path:
    """Return the path to the Hugging Face cache directory.

    This is where all NeuroVLM intermediate files will be stored.

    The main models/datasets/etc have been committed to huggingface_hub repositories:
        See load_dataset, load_latent, load_mask for functions that load datasets from huggingface_hub.
        See from_pretrained method in models.py for a method that loads models from huggingface_hub.

    Returns
    -------
    Path
        Path to the cache directory.

    Notes
    -----
    The default cache directory: ~/.cache/neurovlm
    """
    cache = Path.home() / ".cache" / "neurovlm"
    cache.mkdir(exist_ok=True, parents=True)
    return cache


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
    loaders = [
        ("Publications dataframe", _load_pubmed_dataframe),
        ("NeuroWiki dataframe", _load_neuro_wiki),
        ("NeuroWiki graph dataframe", _load_neuro_wiki_graph),
        ("CogAtlas concepts", _load_cogatlas_dataset),
        ("CogAtlas tasks", lambda: _load_cogatlas_task_dataset(filtered=True)),
        ("CogAtlas disorders", _load_cogatlas_disorder_dataset),
        ("CogAtlas graph", _load_cogatlas_graph_dataset),
        ("Latent text embeddings", _load_latent_text),
        ("Latent wiki embeddings", _load_latent_wiki),
        ("Latent CogAtlas embeddings", _load_latent_cogatlas),
        ("Latent CogAtlas disorder embeddings", _load_latent_cogatlas_disorder),
        ("Latent CogAtlas task embeddings", _load_latent_cogatlas_task),
        ("Autoencoder model", _load_autoencoder),
        ("Brain masker", _load_masker),
        ("Network atlases", _load_networks),
        ("Image projection head", _proj_head_image_infonce),
        ("Text MSE projection head", _proj_head_text_mse),
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
                print(f"Error: {e}")

    if verbose:
        print("\nAll data preloaded successfully!")


# For backward compatibility, keep data_dir
data_dir = get_data_dir()


def _without_grad(payload):
    """Recursively detach tensors and disable gradients."""
    if isinstance(payload, torch.Tensor):
        out = payload.detach()
        if out.requires_grad:
            out = out.requires_grad_(False)
        return out
    if isinstance(payload, tuple):
        return tuple(_without_grad(item) for item in payload)
    if isinstance(payload, list):
        return [_without_grad(item) for item in payload]
    if isinstance(payload, dict):
        return {key: _without_grad(value) for key, value in payload.items()}
    return payload

# Unified interface for all datasets
def load_dataset(name: str):
    """Alias to _load_* functions in retrieval resources.

    Parameters
    ----------
    name: str, {"publications", "coordinate", "neurowiki", "cogatlas",
                "cogatlas_task", "cogatlas_graph", "cogatlas_disorder", "networks",
                "publications_neurovault", "images_neurovault", "neurovault_images",
                "pubmed_images"}
        Name of dataset.

    Returns
    -------
    dataset

    Notes
    -----

    - "pubmed_text": dataframe that includes pubmed dois, pmids, pmcids, titles, abstracts
    - "pubmed_coordinates": pubmed coordinate tables
    - "pubmed_images": tensor containing pubmed images
    - "neurowiki": wikipedia article titles and descriptions
    - "cogatlas": all cogatlas terms
    - "cogatlas_task": cogatlas task terms
    - "cogatlas_disorder": cogatlas disorder terms
    - "cogatlas_graph": how cogatlas terms are related
    - "networks": dict that contains atlas and network name keys, and .nii.gz keys.
    - "publications_neurovault": neurovault-linked publication metadata (titles, abstracts, dois)
    - "neurovault_text": publication data for each neurovault image
    - "neurovault_images": tensor containing neurovault images
    - "neurovault_images_meta" dataframe the maps each image to a study
    """
    match name:
        case "pubmed_text":
            return _load_pubmed_dataframe()
        case "pubmed_coordinates":
            return _load_pubmed_coordinates()
        case "pubmed_images":
            return _load_pubmed_images()
        case "wiki" | "neurowiki":
            return _load_neuro_wiki()
        case "neurowiki_graph":
            return _load_neuro_wiki_graph()
        case "cogatlas":
            return _load_cogatlas_dataset()
        case "cogatlas_graph":
            return _load_cogatlas_graph_dataset()
        case "cogatlas_task":
            return _load_cogatlas_task_dataset()
        case "cogatlas_disorder":
            return _load_cogatlas_disorder_dataset()
        case "networks":
            return _load_networks()
        case "networks_canonical":
            return _load_networks_canonical()
        case "neurovault_text":
            return _load_publications_neurovault_dataframe()
        case "neurovault_images_meta":
            return _load_images_neurovault_dataframe()
        case "neurovault_images":
            return _load_neurovault_images()
        case _:
            valid_names = ["publications", "pubmed", "coordinates", "pubmed_coordinates", "wiki", "neurowiki",
                           "neurowiki_graph", "cogatlas", "cogatlas_task", "cogatlas_disorder", "networks",
                           "networks_canonical", "publications_neurovault", "images_neurovault",
                           "neurovault_images", "pubmed_images"]
            raise ValueError(f"{name} not in {valid_names}")


def load_latent(name: str):
    """Alias to _load_latent* functions in retrieval resources.

    Parameters
    ----------
    name: str, {"publications", "neurowiki", "cogatlas", "cogatlas_task", "cogatlas_disorder",
                "networks", "latent_neurovault_images", "latent_neurovault_text"}
        Name of dataset.

    Returns
    -------
    latent

    Notes
    -----

    - "pubmed_text": pubmed papers passed through specter.
    - "pubmed_images": encoded neuroimages from pubmed coordinate tables
    - "neurowiki": wikipedia articles pass through specter
    - "cogatlas": cogatlas terms and definitions passed through specter
    - "networks": encoded network images
    - "neurovault_images": latent embeddings for neurovault images
    - "neurovault_text": latent embeddings for neurovault-linked text

    """
    match name:
        case "pubmed_text":
            payload = _load_latent_text()
        case "pubmed_images":
            payload = _load_latent_neuro()
        case "wiki" | "neurowiki":
            payload = _load_latent_wiki()
        case "cogatlas":
            payload = _load_latent_cogatlas()
        case "cogatlas_task":
            payload = _load_latent_cogatlas_task()
        case "cogatlas_disorder":
            payload = _load_latent_cogatlas_disorder()
        case "networks_text":
            payload = _load_latent_networks_canonical_text()
        case "networks_neuro":
            payload = _load_latent_networks_neuro()
        case "neurovault_images":
            payload = _load_latent_neurovault_images()
        case "neurovault_text":
            payload = _load_latent_neurovault_text()
        case _:
            valid_names = ["publications", "pubmed", "neurowiki", "cogatlas",
                           "cogatlas_task", "cogatlas_disorder", "networks_text",
                           "networks_neuro", "latent_neurovault_images",
                           "latent_neurovault_text"]
            raise ValueError(f"{name} not in {valid_names}")
    return _without_grad(payload)


def load_masker():
    """Masker alias."""
    return _load_masker()
