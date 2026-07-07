"""Shared plotting helpers for the MLP vs atlas-free CNN model comparison notebooks.

Every notebook imports from here so a given model id always renders in the
same color, bars are ordered the same way, and no chart ends up with two
y-axes or an arbitrary matplotlib color cycle. Colors come from the
validated 8-hue categorical palette in the `dataviz` skill
(`references/palette.md`); only 5 of the 8 slots are used here, one per
model family, so adjacent-hue separation is comfortably above the
colorblind-safety floor.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Validated categorical hues (light-mode steps from references/palette.md).
BLUE = "#2a78d6"
AQUA = "#1baf7a"
YELLOW = "#eda100"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"

# One fixed color per model *family*, reused across every figure in every
# notebook. Family, not the literal model_id string, is what should carry
# color: "mixed baseline" always reads as green whether it is
# `cnn_ae_mixed`, `cnn_contrastive_mixed_to_pubmed`, or `cnn_t2b_mixed`.
FAMILY_COLOR = {
    "mlp": BLUE,
    "cnn_mixed_baseline": GREEN,
    "cnn_pubmed_specialized": YELLOW,
    "cnn_nilearn_specialized": AQUA,
    "cnn_neurovault_specialized": RED,
    "other": VIOLET,
}
FAMILY_LABEL = {
    "mlp": "MLP (NeuroVLM)",
    "cnn_mixed_baseline": "CNN mixed (baseline)",
    "cnn_pubmed_specialized": "CNN PubMed-specialized",
    "cnn_nilearn_specialized": "CNN Nilearn-specialized",
    "cnn_neurovault_specialized": "CNN NeuroVault-specialized",
    "other": "Other",
}
FAMILY_ORDER = list(FAMILY_COLOR)


def model_family(model_id: str) -> str:
    """Map any registry model id to a fixed, cross-notebook color family."""
    if model_id == "mlp_neurovlm":
        return "mlp"
    if "_mixed_to_" in model_id or model_id.endswith("_mixed"):
        return "cnn_mixed_baseline"
    for domain in ("pubmed", "nilearn", "neurovault"):
        if model_id.endswith(f"_{domain}"):
            return f"cnn_{domain}_specialized"
    return "other"


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.yaxis.grid(True, color="#d8d7d2", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0)


def grouped_bar(
    df: pd.DataFrame,
    *,
    category_col: str,
    series_col: str,
    value_col: str,
    ax: plt.Axes | None = None,
    title: str = "",
    ylabel: str = "",
    value_fmt: str = "{:.2f}",
    categories: Sequence[str] | None = None,
    show_legend: bool = True,
) -> plt.Axes:
    """Grouped bar chart: one group per category (e.g. dataset), one bar per
    series (e.g. model_id) colored by fixed model family. Skips missing
    category/series combinations instead of erroring, so partial results
    (some checkpoints unresolved) still render.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    if df.empty or value_col not in df.columns:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color="#8a8a86")
        ax.set_title(title)
        return ax

    categories = list(categories) if categories is not None else sorted(df[category_col].dropna().unique())
    series_ids = sorted(df[series_col].dropna().unique(), key=lambda s: (FAMILY_ORDER.index(model_family(s)), s))

    n_series = max(len(series_ids), 1)
    group_width = 0.8
    bar_width = group_width / n_series
    x = np.arange(len(categories))

    seen_families: dict[str, Any] = {}
    for i, series_id in enumerate(series_ids):
        offsets = x - group_width / 2 + bar_width * (i + 0.5)
        values = []
        for cat in categories:
            match = df[(df[category_col] == cat) & (df[series_col] == series_id)]
            values.append(float(match[value_col].iloc[0]) if len(match) and pd.notna(match[value_col].iloc[0]) else np.nan)
        family = model_family(series_id)
        color = FAMILY_COLOR[family]
        bars = ax.bar(offsets, values, width=bar_width * 0.92, color=color, zorder=3, label=series_id)
        seen_families[family] = bars
        for rect, value in zip(bars, values):
            if np.isnan(value):
                continue
            ax.annotate(
                value_fmt.format(value),
                (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#3a3a37",
                xytext=(0, 2),
                textcoords="offset points",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, loc="left")
    _style_axes(ax)
    if show_legend and len(series_ids) >= 2:
        ax.legend(
            handles=[plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLOR[model_family(s)]) for s in series_ids],
            labels=list(series_ids),
            frameon=False,
            fontsize=8,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
        )
    return ax


def small_multiples_lines(
    df: pd.DataFrame,
    *,
    panel_col: str,
    x_col: str,
    y_col: str,
    series_col: str,
    suptitle: str = "",
    xlabel: str = "",
    ylabel: str = "",
    reference_line: bool = False,
    aggregate: str | None = None,
    show_range: bool = False,
    figsize_per_panel: tuple[float, float] = (4.2, 4.0),
) -> plt.Figure:
    """One line subplot per panel value (e.g. dataset), each line a series
    (e.g. model_id) colored by fixed model family. Used for recall curves
    and Dice-vs-threshold sensitivity curves.

    Set ``aggregate="mean"`` when a single (panel, series, x) combination has
    multiple rows (e.g. one Dice-sensitivity row per sample per threshold) --
    otherwise the line zigzags across per-sample noise instead of tracing the
    trend. ``show_range=True`` additionally shades the min-max band around
    the aggregated line.
    """
    panels = sorted(df[panel_col].dropna().unique()) if not df.empty else []
    n = max(len(panels), 1)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per_panel[0] * n, figsize_per_panel[1]), squeeze=False)
    axes = axes[0]

    if df.empty:
        axes[0].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[0].transAxes, color="#8a8a86")
        fig.suptitle(suptitle)
        return fig

    for ax, panel in zip(axes, panels):
        panel_df = df[df[panel_col] == panel]
        series_ids = sorted(panel_df[series_col].dropna().unique(), key=lambda s: (FAMILY_ORDER.index(model_family(s)), s))
        has_line = False
        for series_id in series_ids:
            sub = panel_df[panel_df[series_col] == series_id].dropna(subset=[x_col, y_col])
            if aggregate is not None and len(sub):
                grouped = sub.groupby(x_col)[y_col]
                agg_sub = grouped.agg(aggregate).reset_index().sort_values(x_col)
                if show_range:
                    lo = grouped.min().reindex(agg_sub[x_col]).to_numpy()
                    hi = grouped.max().reindex(agg_sub[x_col]).to_numpy()
                    ax.fill_between(agg_sub[x_col], lo, hi, color=FAMILY_COLOR[model_family(series_id)], alpha=0.15, zorder=1)
                sub = agg_sub
            else:
                sub = sub.sort_values(x_col)
            if not len(sub):
                continue
            has_line = True
            ax.plot(
                sub[x_col],
                sub[y_col],
                color=FAMILY_COLOR[model_family(series_id)],
                linewidth=2,
                marker="o",
                markersize=4,
                label=series_id,
                zorder=3,
            )
        if not has_line:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color="#8a8a86")
        if reference_line:
            xs = panel_df[x_col]
            if len(xs):
                ax.plot([xs.min(), xs.max()], [xs.min(), xs.max()], color="#b7b6b0", linestyle="--", linewidth=1, zorder=2, label="Chance")
        ax.set_title(str(panel), fontsize=10, loc="left")
        ax.set_xlabel(xlabel)
        _style_axes(ax)
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(), frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle(suptitle, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 0.98, 0.95))
    return fig


def sample_distribution_box(
    df: pd.DataFrame,
    *,
    panel_col: str,
    series_col: str,
    value_col: str,
    suptitle: str = "",
    ylabel: str = "",
    figsize_per_panel: tuple[float, float] = (4.2, 4.0),
) -> plt.Figure:
    """Per-dataset small multiples of a by-sample metric's distribution
    across models, as a boxplot (median/IQR read better than a mean bar for
    noisy per-sample metrics like Pearson r).
    """
    panels = sorted(df[panel_col].dropna().unique()) if not df.empty else []
    n = max(len(panels), 1)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per_panel[0] * n, figsize_per_panel[1]), squeeze=False)
    axes = axes[0]

    if df.empty or value_col not in df.columns:
        axes[0].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[0].transAxes, color="#8a8a86")
        fig.suptitle(suptitle)
        return fig

    for ax, panel in zip(axes, panels):
        panel_df = df[(df[panel_col] == panel) & df[value_col].notna()]
        series_ids = sorted(panel_df[series_col].dropna().unique(), key=lambda s: (FAMILY_ORDER.index(model_family(s)), s))
        data = [panel_df.loc[panel_df[series_col] == s, value_col].to_numpy() for s in series_ids]
        data = [d for d in data if len(d)]
        kept_ids = [s for s, d in zip(series_ids, data) if len(d)]
        if not data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color="#8a8a86")
            ax.set_title(str(panel), fontsize=10, loc="left")
            continue
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False, zorder=3)
        for patch, series_id in zip(bp["boxes"], kept_ids):
            patch.set_facecolor(FAMILY_COLOR[model_family(series_id)])
            patch.set_alpha(0.75)
            patch.set_edgecolor("#3a3a37")
        for median in bp["medians"]:
            median.set_color("#1a1a19")
        ax.set_xticks(range(1, len(kept_ids) + 1))
        ax.set_xticklabels(kept_ids, rotation=30, ha="right", fontsize=8)
        ax.set_title(str(panel), fontsize=10, loc="left")
        _style_axes(ax)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(suptitle, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def coverage_table(df: pd.DataFrame, *, dataset_col: str = "dataset", model_col: str = "model_id", status_col: str = "status") -> pd.DataFrame:
    """Pivot table of run status per (dataset, model) -- makes missing
    checkpoints / unsupported combinations visible at a glance instead of
    silently disappearing from the metric charts above.
    """
    if df.empty or status_col not in df.columns:
        return pd.DataFrame()
    return df.pivot_table(index=model_col, columns=dataset_col, values=status_col, aggfunc="first").fillna("no_row")


def save_report_assets(
    output_dir: str | Path,
    *,
    figures: dict[str, plt.Figure],
    dataframes: dict[str, pd.DataFrame],
    dpi: int = 150,
) -> dict[str, Any]:
    """Save every figure (PNG) and dataframe (CSV) from a notebook run into
    one directory, plus a `manifest.json` listing what was written.

    Intended as the last cell of each comparison notebook so an HTML-report
    builder can read a single, predictable location per notebook rather than
    re-deriving paths or re-running the comparison.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"figures": {}, "dataframes": {}}
    for name, fig in figures.items():
        path = output_dir / f"{name}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        manifest["figures"][name] = str(path)
    for name, df in dataframes.items():
        path = output_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        manifest["dataframes"][name] = str(path)
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    manifest["manifest_path"] = str(manifest_path)
    return manifest
