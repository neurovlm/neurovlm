"""Shared atlas-free CNN workflow conventions.

This module is deliberately free of Colab, Hugging Face download, and notebook
display helpers so core training/evaluation code can depend on it without
depending on notebook support code.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


DOMAIN_DIRS = {"pubmed": "01_pubmed", "nilearn": "02_nilearn", "neurovault": "03_neurovault"}
SPECIALIZED_BRANCHES = {
    "pubmed": "specialized_mixed_to_pubmed",
    "nilearn": "specialized_mixed_to_nilearn",
    "neurovault": "specialized_mixed_to_neurovault",
}
BRANCH_KINDS = ("baseline", "specialized")

NORMALIZED_STAGE3_DIRNAME = "stage3_normalized_specter"
CORRECTED_STAGE4_DIRNAME = "corrected_stage4_normalized_specter"

NORMALIZED_STAGE3_CHECKPOINT = "best_val_normalized_recall_auc.pt"
CORRECTED_STAGE4_CHECKPOINT = "best_val_generation_normalized_auc.pt"
STAGE4_COMPUTE_SEMANTIC_AUC_DURING_TRAINING = False
GENERATION_AUC_VAL_INTERVAL = 5
STAGE4_PRIMARY_SPATIAL_CHECKPOINT = "best_val_top5_dice.pt"
STAGE4_SPATIAL_CORR_CHECKPOINT = "best_val_spatial_corr.pt"
STAGE4_SEMANTIC_CHECKPOINT = "best_val_generation_normalized_auc.pt"

DEFAULT_ATLAS_FREE_HF_REPO = "neurovlm/atlas_free_cnn_dataset"
DEFAULT_TEXT_EMBEDDING_CONVENTION = "normalized_specter2"
NORMALIZED_TEXT_EMBEDDING_CONVENTION = "normalized_specter2"
NORMALIZED_SPECTER_CACHE_STEM = "specter2_stage3_stage4_emptycentered_unitnorm"
NORMALIZED_SPECTER_CACHE_FILENAME = "specter2_stage3_stage4_emptycentered_unitnorm.pt"
NORMALIZED_SPECTER_METADATA_FILENAME = f"{NORMALIZED_SPECTER_CACHE_STEM}_metadata.json"
NORMALIZED_SPECTER_VALIDATION_FILENAME = f"{NORMALIZED_SPECTER_CACHE_STEM}_validation.json"
NORMALIZED_SPECTER_INDEX_FILENAME = f"{NORMALIZED_SPECTER_CACHE_STEM}_index.csv"
NORMALIZED_SPECTER_PREPROCESSING = "empty_string_centered_l2_unit_normalized"
SPECTER2_ENCODER_MODEL = "allenai/specter2_aug2023refresh"
SPECTER2_ADAPTER_NAME = "adhoc_query"
SPECTER2_BASE_MODEL_REPO = "allenai/specter2_aug2023refresh_base"
SPECTER2_ADAPTER_REPO = "allenai/specter2_aug2023refresh_adhoc_query"
TEXT_EMBEDDING_DIM = 768
TEXT_EMBEDDING_CONVENTION_DIR_SUFFIXES = {
    NORMALIZED_TEXT_EMBEDDING_CONVENTION: "normalized_specter",
}

LOCKED_STAGE1_CHECKPOINT_NAMES = {
    "mixed_stage1a": "best_top1_dice.pt",
    "mixed_to_pubmed_stage1b": "best_top1_dice.pt",
    "mixed_to_nilearn_stage1b": "best_val_loss.pt",
    "mixed_to_neurovault_stage1b": "best_top5_dice.pt",
}

LOCKED_REGISTRY_VARIANTS = {
    "mixed_stage1a": "mixed_baseline_raw_mse",
    "mixed_to_pubmed_stage1b": "mixed_to_pubmed",
    "mixed_to_nilearn_stage1b": "mixed_to_nilearn",
    "mixed_to_neurovault_stage1b": "mixed_to_neurovault",
}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ae_branch_spec(domain: str, branch_kind: str) -> dict[str, str]:
    if branch_kind == "baseline":
        return {
            "run": f"mixed_stage1a_on_{domain}",
            "domain": domain,
            "domain_dir": DOMAIN_DIRS[domain],
            "branch_kind": "baseline",
            "type": "baseline",
            "branch": "baseline_mixed_stage1a",
            "ae_registry_key": "mixed_stage1a",
        }
    if branch_kind == "specialized":
        return {
            "run": f"mixed_to_{domain}_stage1b_on_{domain}",
            "domain": domain,
            "domain_dir": DOMAIN_DIRS[domain],
            "branch_kind": "specialized",
            "type": "specialized",
            "branch": SPECIALIZED_BRANCHES[domain],
            "ae_registry_key": f"mixed_to_{domain}_stage1b",
        }
    raise ValueError(f"branch_kind must be one of {BRANCH_KINDS}; got {branch_kind!r}")


def six_branch_specs() -> list[dict[str, str]]:
    """Return the fixed three baseline plus three domain-specialized runs."""

    rows: list[dict[str, str]] = []
    for domain in ["pubmed", "nilearn", "neurovault"]:
        rows.append(_ae_branch_spec(domain, "baseline"))
        rows.append(_ae_branch_spec(domain, "specialized"))
    return rows


def select_six_downstream_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return all six manifest rows in their fixed pipeline order."""

    by_name = {str(run.get("run", "")): run for run in runs}
    selected: list[dict[str, Any]] = []
    missing: list[str] = []
    for spec in six_branch_specs():
        row = by_name.get(spec["run"])
        if row is None:
            missing.append(spec["run"])
            continue
        selected.append({**spec, **row})
    if missing:
        raise KeyError(f"Downstream manifest is missing required six-branch runs: {missing}")
    return selected


def canonical_text_embedding_convention(convention: str | None) -> str:
    value = (convention or DEFAULT_TEXT_EMBEDDING_CONVENTION).strip().lower()
    aliases = {
        "normalized": NORMALIZED_TEXT_EMBEDDING_CONVENTION,
        "normalized_specter": NORMALIZED_TEXT_EMBEDDING_CONVENTION,
        "normalized_specter2": NORMALIZED_TEXT_EMBEDDING_CONVENTION,
    }
    if value not in aliases:
        raise ValueError(
            f"Unknown text embedding convention {convention!r}. "
            f"Expected {NORMALIZED_TEXT_EMBEDDING_CONVENTION!r}."
        )
    return aliases[value]


def text_embedding_convention_dir_suffix(convention: str | None = DEFAULT_TEXT_EMBEDDING_CONVENTION) -> str:
    """Return the output-directory suffix for a SPECTER cache convention."""

    return TEXT_EMBEDDING_CONVENTION_DIR_SUFFIXES[canonical_text_embedding_convention(convention)]


def stage3_dirname_for_text_embedding_convention(convention: str | None = DEFAULT_TEXT_EMBEDDING_CONVENTION) -> str:
    return f"stage3_{text_embedding_convention_dir_suffix(convention)}"


def corrected_stage4_dirname_for_text_embedding_convention(convention: str | None = DEFAULT_TEXT_EMBEDDING_CONVENTION) -> str:
    return f"corrected_stage4_{text_embedding_convention_dir_suffix(convention)}"


def _text_convention_from_layout(layout: str | None, convention: str | None = None) -> str | None:
    if convention is not None:
        return canonical_text_embedding_convention(convention)
    if layout in {"6a_normalized_corrected", "normalized_specter", "6a_normalized_specter"}:
        return NORMALIZED_TEXT_EMBEDDING_CONVENTION
    return None


def stage_output_dir(
    run_root: str | Path,
    domain: str,
    branch: str,
    stage: str,
    *,
    layout: str = "6a_normalized_corrected",
    text_embedding_convention: str | None = None,
) -> Path:
    run_root = Path(run_root)
    convention = _text_convention_from_layout(layout, text_embedding_convention)
    if convention is not None:
        dirname = (
            stage3_dirname_for_text_embedding_convention(convention)
            if stage == "stage3"
            else corrected_stage4_dirname_for_text_embedding_convention(convention)
            if stage == "stage4"
            else stage
        )
    else:
        dirname = stage
    return run_root / DOMAIN_DIRS[domain] / branch / dirname


def stage_checkpoint_path(
    run_root: str | Path,
    domain: str,
    branch: str,
    stage: str,
    *,
    layout: str = "6a_normalized_corrected",
    text_embedding_convention: str | None = None,
) -> Path:
    out_dir = stage_output_dir(run_root, domain, branch, stage, layout=layout, text_embedding_convention=text_embedding_convention)
    convention = _text_convention_from_layout(layout, text_embedding_convention)
    if stage == "stage3":
        return out_dir / "checkpoints" / NORMALIZED_STAGE3_CHECKPOINT
    if stage == "stage4" and convention is not None:
        return out_dir / "checkpoints" / STAGE4_PRIMARY_SPATIAL_CHECKPOINT
    return out_dir / "checkpoints" / "best_val_loss.pt"


def discover_stage_outputs(
    run_root: str | Path,
    *,
    layout: str = "6a_normalized_corrected",
    text_embedding_convention: str | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for spec in six_branch_specs():
        stage3_dir = stage_output_dir(run_root, spec["domain"], spec["branch"], "stage3", layout=layout, text_embedding_convention=text_embedding_convention)
        stage4_dir = stage_output_dir(run_root, spec["domain"], spec["branch"], "stage4", layout=layout, text_embedding_convention=text_embedding_convention)
        rows.append(
            {
                **spec,
                "stage3_dir": str(stage3_dir),
                "stage3_checkpoint": str(
                    stage_checkpoint_path(
                        run_root,
                        spec["domain"],
                        spec["branch"],
                        "stage3",
                        layout=layout,
                        text_embedding_convention=text_embedding_convention,
                    )
                ),
                "stage4_dir": str(stage4_dir),
                "stage4_checkpoint": str(
                    stage_checkpoint_path(
                        run_root,
                        spec["domain"],
                        spec["branch"],
                        "stage4",
                        layout=layout,
                        text_embedding_convention=text_embedding_convention,
                    )
                ),
            }
        )
    return rows
