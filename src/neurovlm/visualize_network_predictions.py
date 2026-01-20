"""Visualize network labeling predictions for brain images.

This module provides functions to create publication-quality figures showing
predictions from NeuroVLM's brain-to-text retrieval system. The visualization
shows brain images alongside their predicted networks, regions, and cognitive terms.
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Dict, Any, Union
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import nibabel as nib
from nilearn import plotting
from nilearn.plotting import plot_stat_map
import seaborn as sns
import pandas as pd
import torch

__all__ = [
    "plot_network_labeling_compact",
    "plot_single_category",
    "create_predictions_from_search_results",
]


def _convert_to_nifti(brain_data: Any) -> Optional[nib.Nifti1Image]:
    """Convert various brain data formats to NIfTI image.

    Parameters
    ----------
    brain_data : various
        Can be:
        - nib.Nifti1Image: returned as-is
        - torch.Tensor: converted to NIfTI with identity affine
        - np.ndarray: converted to NIfTI with identity affine
        - str/Path: loaded as NIfTI file
        - None: returns None

    Returns
    -------
    nib.Nifti1Image or None
    """
    if brain_data is None:
        return None

    # Already a NIfTI image
    if isinstance(brain_data, nib.Nifti1Image):
        return brain_data

    # Path to NIfTI file
    if isinstance(brain_data, (str, Path)):
        return nib.load(str(brain_data))

    # Torch tensor
    if isinstance(brain_data, torch.Tensor):
        data = brain_data.detach().cpu().numpy()
        if data.ndim == 1:
            # Assume it's a flattened brain map - can't reconstruct without masker
            return None
        affine = np.eye(4)
        return nib.Nifti1Image(data, affine)

    # Numpy array
    if isinstance(brain_data, np.ndarray):
        if brain_data.ndim == 1:
            # Assume it's a flattened brain map - can't reconstruct without masker
            return None
        affine = np.eye(4)
        return nib.Nifti1Image(brain_data, affine)

    return None


def plot_single_category(
    predictions: Union[Dict[str, Any], List[Dict[str, Any]]],
    category_key: str = "predictions",
    category_name: str = "Predictions",
    figsize: Optional[Tuple[float, float]] = None,
    output_path: Optional[str] = None,
    show_colorbar: bool = False,
    score_threshold: Optional[float] = None,
    top_k: int = 10,
) -> plt.Figure:
    """Create compact single-category prediction figure.

    Parameters
    ----------
    predictions : dict or list of dict
        Each dict should contain:
        - "network_name" (str): Network label for the row
        - "brain_image" (various): Brain image (NIfTI, tensor, array, or path)
        - category_key (list of tuple): [(label, score), ...] predictions
    category_key : str, optional
        Key name for predictions in dict, by default "predictions"
    category_name : str, optional
        Display name for the category column, by default "Predictions"
    figsize : tuple of float, optional
        Figure size (width, height). If None, auto-computed
    output_path : str, optional
        If provided, save figure to this path
    show_colorbar : bool, optional
        Whether to show colorbars on brain images, by default False
    score_threshold : float, optional
        Minimum score to filter predictions, by default None
    top_k : int, optional
        Maximum number of predictions to show, by default 10

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Handle single prediction
    if isinstance(predictions, dict):
        predictions = [predictions]

    n_networks = len(predictions)

    # Auto-compute figure size
    if figsize is None:
        figsize = (10, n_networks * 2)

    # Create figure with 2 columns: brain image, predictions
    fig, axes = plt.subplots(
        nrows=n_networks, ncols=2,
        figsize=figsize,
        gridspec_kw={'width_ratios': [1, 2]}
    )

    # Handle single row case
    if n_networks == 1:
        axes = axes.reshape(1, -1)

    # Plot each network row
    for row_idx, pred in enumerate(predictions):
        # --- Column 1: Brain Image ---
        ax_brain = axes[row_idx, 0]
        brain_data = pred.get("brain_image", None)
        network_name = pred.get("network_name", f"Network {row_idx + 1}")

        # Convert to NIfTI
        brain_img = _convert_to_nifti(brain_data)

        if brain_img is not None:
            try:
                # Plot brain image
                plot_stat_map(
                    brain_img,
                    colorbar=show_colorbar,
                    draw_cross=False,
                    annotate=False,
                    axes=ax_brain,
                    cmap='Reds',
                    vmax=None  # Auto-scale
                )
            except Exception as e:
                # If plotting fails, show error message
                ax_brain.text(0.5, 0.5, f"Error plotting brain:\n{str(e)[:50]}",
                           ha='center', va='center', transform=ax_brain.transAxes,
                           fontsize=8, color='red')
                ax_brain.axis('off')
        else:
            # No valid brain image
            ax_brain.text(0.5, 0.5, "No brain image",
                         ha='center', va='center', transform=ax_brain.transAxes,
                         fontsize=10, style='italic')
            ax_brain.axis('off')

        # Add network name label
        ax_brain.set_yticks([])
        ax_brain.set_xticks([])

        # Format network name for vertical display
        formatted_name = network_name.replace(" ", "\n").replace("-", "\n")

        ax_brain.text(
            -0.15, 0.5,
            formatted_name,
            transform=ax_brain.transAxes,
            rotation=90,
            verticalalignment='center',
            horizontalalignment='center',
            fontsize=11,
            fontweight='bold'
        )

        # Add column title for first row
        if row_idx == 0:
            ax_brain.set_title("True Networks", fontsize=12, fontweight='bold', pad=10)

        # --- Column 2: Prediction Bars ---
        ax_pred = axes[row_idx, 1]
        predictions_list = pred.get(category_key, [])

        if not predictions_list:
            ax_pred.text(0.5, 0.5, "No predictions",
                        ha='center', va='center', transform=ax_pred.transAxes,
                        fontsize=10, style='italic')
            ax_pred.axis('off')
            continue

        # Extract labels and scores
        labels = []
        scores = []
        for item in predictions_list:
            if isinstance(item, tuple) and len(item) >= 2:
                label = str(item[0]).split(" (")[0]  # Remove any parenthetical info
                score = float(item[1])
                labels.append(label)
                scores.append(score)
            else:
                print(f"Warning: Skipping invalid prediction item: {item}")

        if not labels:
            ax_pred.axis('off')
            continue

        # Filter by threshold if provided
        if score_threshold is not None:
            filtered = [(l, s) for l, s in zip(labels, scores) if s >= score_threshold]
            if filtered:
                labels, scores = zip(*filtered)
                labels, scores = list(labels), list(scores)
            else:
                ax_pred.text(0.5, 0.5, f"No predictions above threshold {score_threshold}",
                            ha='center', va='center', transform=ax_pred.transAxes,
                            fontsize=9, style='italic')
                ax_pred.axis('off')
                continue

        # Limit to top_k
        if len(labels) > top_k:
            labels = labels[:top_k]
            scores = scores[:top_k]

        # Reverse order for plotting (highest at top)
        labels = labels[::-1]
        scores = scores[::-1]

        # Create color palette (blue for positive, red for negative)
        colors = ['#d32f2f' if s < 0 else '#1976d2' for s in scores]

        # Plot horizontal bars
        y_positions = np.arange(len(labels))
        bars = ax_pred.barh(y_positions, scores, color=colors, alpha=0.8, height=0.7)

        # Add labels and scores to bars
        for i, (bar, label, score) in enumerate(zip(bars, labels, scores)):
            width = bar.get_width()

            # Truncate long labels
            if len(label) > 70:  # ← Increase this
                label = label[:67] + "..."  # ← And this

            # Label position: inside bar for long bars, outside for short ones
            if abs(width) > 0.1:  # Inside bar
                label_x = 0.02
                label_ha = 'left'
                label_color = 'white'
            else:  # Outside bar
                label_x = width + 0.02
                label_ha = 'left'
                label_color = 'black'

            # Add label
            ax_pred.text(label_x, i, label,
                        ha=label_ha, va='center',
                        fontsize=9, fontweight='bold',
                        color=label_color)

            # # Add score at the end of bar
            # score_x = width * 0.95 if abs(width) > 0.1 else width + 0.02
            # ax_pred.text(score_x, i, f'{score:.3f}',
            #             ha='right' if abs(width) > 0.1 else 'left',
            #             va='center',
            #             fontsize=8,
            #             color='white' if abs(width) > 0.1 else 'black',
            #             fontweight='bold')

        # Style the axis
        ax_pred.set_yticks([])
        ax_pred.spines['right'].set_visible(False)
        ax_pred.spines['top'].set_visible(False)
        ax_pred.spines['left'].set_visible(False)

        # Only show x-axis on last row
        if row_idx == n_networks - 1:
            ax_pred.set_xlabel("Cosine Similarity", fontsize=10, fontweight='bold')
            ax_pred.spines['bottom'].set_visible(True)
        else:
            ax_pred.spines['bottom'].set_visible(False)
            ax_pred.set_xticks([])

        # Add column title for first row
        if row_idx == 0:
            ax_pred.set_title(category_name, fontsize=12, fontweight='bold', pad=10)

        # Set x-axis limits with some padding
        if scores:
            max_score = max(abs(min(scores)), abs(max(scores)))
            ax_pred.set_xlim(-max_score * 0.1, max_score * 1.1)

    # Adjust layout
    plt.tight_layout()

    # Save if output path provided
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to: {output_path}")

    return fig


def plot_network_labeling_compact(
    predictions: Union[Dict[str, Any], List[Dict[str, Any]]],
    figsize: Optional[Tuple[float, float]] = None,
    output_path: Optional[str] = None,
    show_colorbar: bool = False,
    score_threshold: Optional[float] = None,
    top_k: int = 10,
) -> plt.Figure:
    """Create compact network labeling figure (paper style).

    Parameters
    ----------
    predictions : dict or list of dict
        Single prediction dict or list of dicts. Each dict should contain:
        - "network_name" (str): True network label
        - "brain_image" (various): Brain image (NIfTI, tensor, array, or path)
        - "network_predictions" (list of tuple): [(label, score), ...]
        - "region_predictions" (list of tuple): [(label, score), ...]
        - "cognition_predictions" (list of tuple): [(label, score), ...]
    figsize : tuple of float, optional
        Figure size (width, height). If None, auto-computed
    output_path : str, optional
        If provided, save figure to this path
    show_colorbar : bool, optional
        Whether to show colorbars on brain images, by default False
    score_threshold : float, optional
        Minimum score to filter predictions, by default None
    top_k : int, optional
        Maximum number of predictions to show per category, by default 10

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Handle single prediction
    if isinstance(predictions, dict):
        predictions = [predictions]

    n_networks = len(predictions)

    # Auto-compute figure size
    if figsize is None:
        figsize = (14, n_networks * 2)

    # Create figure with 4 columns
    fig, axes = plt.subplots(
        nrows=n_networks, ncols=4,
        figsize=figsize,
        gridspec_kw={'width_ratios': [1, 1.5, 1.5, 1.5]}
    )

    # Handle single row case
    if n_networks == 1:
        axes = axes.reshape(1, -1)

    # Column titles
    column_titles = ["True Networks", "Network Predictions", "Region Predictions", "Cognition Predictions"]
    category_keys = [None, "network_predictions", "region_predictions", "cognition_predictions"]

    # Plot each row
    for row_idx, pred in enumerate(predictions):
        network_name = pred.get("network_name", f"Network {row_idx + 1}")

        # Plot brain image (column 0)
        ax_brain = axes[row_idx, 0]
        brain_data = pred.get("brain_image", None)
        brain_img = _convert_to_nifti(brain_data)

        if brain_img is not None:
            try:
                plot_stat_map(
                    brain_img,
                    colorbar=show_colorbar,
                    draw_cross=False,
                    annotate=False,
                    axes=ax_brain,
                    cmap='Reds'
                )
            except Exception as e:
                ax_brain.text(0.5, 0.5, f"Error:\n{str(e)[:30]}",
                            ha='center', va='center', transform=ax_brain.transAxes,
                            fontsize=8, color='red')
                ax_brain.axis('off')
        else:
            ax_brain.axis('off')

        # Add network name
        formatted_name = network_name.replace(" ", "\n").replace("-", "\n")
        ax_brain.text(-0.15, 0.5, formatted_name,
                     transform=ax_brain.transAxes, rotation=90,
                     va='center', ha='center', fontsize=11, fontweight='bold')

        if row_idx == 0:
            ax_brain.set_title(column_titles[0], fontsize=12, fontweight='bold', pad=10)

        # Plot prediction columns (1, 2, 3)
        for col_idx in range(1, 4):
            ax = axes[row_idx, col_idx]
            key = category_keys[col_idx]
            predictions_list = pred.get(key, [])

            # Similar bar plotting logic as in plot_single_category
            # (implementation similar to above, shortened for brevity)
            if not predictions_list:
                ax.axis('off')
                continue

            # Extract and process predictions
            labels, scores = [], []
            for item in predictions_list:
                if isinstance(item, tuple) and len(item) >= 2:
                    labels.append(str(item[0]).split(" (")[0])
                    scores.append(float(item[1]))

            if not labels:
                ax.axis('off')
                continue

            # Filter and limit
            if score_threshold:
                filtered = [(l, s) for l, s in zip(labels, scores) if s >= score_threshold]
                if filtered:
                    labels, scores = zip(*filtered)
                else:
                    ax.axis('off')
                    continue

            labels, scores = list(labels[:top_k])[::-1], list(scores[:top_k])[::-1]

            # Plot bars (simplified)
            colors = ['#d32f2f' if s < 0 else '#1976d2' for s in scores]
            y_pos = np.arange(len(labels))
            ax.barh(y_pos, scores, color=colors, alpha=0.8, height=0.7)

            # Style
            ax.set_yticks([])
            ax.spines[['right', 'top', 'left']].set_visible(False)

            if row_idx == n_networks - 1:
                ax.set_xlabel("Cosine Similarity", fontsize=10, fontweight='bold')
            else:
                ax.spines['bottom'].set_visible(False)
                ax.set_xticks([])

            if row_idx == 0:
                ax.set_title(column_titles[col_idx], fontsize=12, fontweight='bold', pad=10)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to: {output_path}")

    return fig


def create_predictions_from_search_results(
    network_name: str,
    network_results: List[str],
    network_scores: np.ndarray,
    region_results: List[str],
    region_scores: np.ndarray,
    cognition_results: List[str],
    cognition_scores: np.ndarray,
    brain_image: Optional[Any] = None,
) -> Dict[str, Any]:
    """Create prediction dictionary from search results."""
    return {
        "network_name": network_name,
        "network_predictions": list(zip(network_results, network_scores)),
        "region_predictions": list(zip(region_results, region_scores)),
        "cognition_predictions": list(zip(cognition_results, cognition_scores)),
        "brain_image": brain_image,
    }