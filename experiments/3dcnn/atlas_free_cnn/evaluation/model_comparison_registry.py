"""Model registry resolving MLP and atlas-free CNN checkpoints for comparison.

MLP resources and Stage 3/4 checkpoint paths are resolved through the
HuggingFace loaders in ``neurovlm.retrieval_resources``. Stage 1 CNN
autoencoders are resolved by experiment-local adapters because constructing
those models depends on the repo-local ``atlas_free_cnn`` package. All CNN
contrastive (Stage 3) and CNN text-to-brain (Stage 4) checkpoint variants are
uploaded to the ``neurovlm/3d_cnn`` model repo, but resolution still falls back
to a ``missing_checkpoint`` status instead of crashing if a variant is later
renamed or pulled upstream.

Stage 3/4 come in six variants per family: three ``mixed_to_{domain}``
baseline branches (the Stage 1A mixed AE evaluated on that domain) and three
domain-specialized branches (``pubmed``, ``nilearn``, ``neurovault``, each
paired with its own Stage 1B domain-finetuned AE). This mirrors the
baseline/specialized branch split already used for Stage 1 AE checkpoints.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

CNN_AE_DOMAINS = ("mixed", "pubmed", "nilearn", "neurovault")
CNN_STAGE_VARIANTS = (
    "mixed_to_pubmed",
    "mixed_to_nilearn",
    "mixed_to_neurovault",
    "pubmed",
    "nilearn",
    "neurovault",
)

DEFAULT_REGISTRY_OUTPUT_PATH = Path("outputs/model_comparison/model_registry_resolved.json")

_CNN_AE_LOADER_NAMES = {
    "mixed": "_load_mixed_ae",
    "pubmed": "_load_pubmed_finetuned_ae",
    "nilearn": "_load_nilearn_finetuned_ae",
    "neurovault": "_load_neurovault_finetuned_ae",
}


def _variant_domain_and_branch(variant: str) -> tuple[str, str]:
    """Split a Stage 3/4 variant id into its evaluation domain and branch kind."""

    if variant.startswith("mixed_to_"):
        return variant.removeprefix("mixed_to_"), "baseline"
    return variant, "specialized"


def _resolve_mlp() -> dict[str, Any]:
    import neurovlm.retrieval_resources as rr

    rr._load_autoencoder()
    rr._load_masker()
    rr._proj_head_image_infonce()
    rr._proj_head_text_infonce()
    rr._proj_head_text_mse()
    return {"status": "resolved", "checkpoint_path": None, "error": None}


def _resolve_cnn_ae(domain: str) -> dict[str, Any]:
    from atlas_free_cnn.evaluation import model_comparison_adapters as adapters

    loader = getattr(adapters, _CNN_AE_LOADER_NAMES[domain])
    loader()
    return {"status": "resolved", "checkpoint_path": None, "error": None}


def _resolve_cnn_contrastive(variant: str) -> dict[str, Any]:
    import neurovlm.retrieval_resources as rr

    path = rr._load_cnn_contrastive_checkpoint_path(variant)
    return {"status": "resolved", "checkpoint_path": str(path), "error": None}


def _resolve_cnn_t2b(variant: str) -> dict[str, Any]:
    import neurovlm.retrieval_resources as rr

    path = rr._load_cnn_t2b_checkpoint_path(variant)
    return {"status": "resolved", "checkpoint_path": str(path), "error": None}


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    family: str  # "mlp" | "cnn_ae" | "cnn_contrastive" | "cnn_t2b"
    domain: str | None
    branch: str | None  # "mixed" | "specialized" | "baseline" | None (mlp)
    resolver: Callable[[], dict[str, Any]]


def _build_model_specs() -> dict[str, ModelSpec]:
    specs: dict[str, ModelSpec] = {
        "mlp_neurovlm": ModelSpec("mlp_neurovlm", "mlp", None, None, _resolve_mlp),
    }
    for domain in CNN_AE_DOMAINS:
        branch = "mixed" if domain == "mixed" else "specialized"
        specs[f"cnn_ae_{domain}"] = ModelSpec(
            f"cnn_ae_{domain}", "cnn_ae", domain, branch, lambda domain=domain: _resolve_cnn_ae(domain)
        )
    for variant in CNN_STAGE_VARIANTS:
        domain, branch = _variant_domain_and_branch(variant)
        specs[f"cnn_contrastive_{variant}"] = ModelSpec(
            f"cnn_contrastive_{variant}",
            "cnn_contrastive",
            domain,
            branch,
            lambda variant=variant: _resolve_cnn_contrastive(variant),
        )
    for variant in CNN_STAGE_VARIANTS:
        domain, branch = _variant_domain_and_branch(variant)
        specs[f"cnn_t2b_{variant}"] = ModelSpec(
            f"cnn_t2b_{variant}",
            "cnn_t2b",
            domain,
            branch,
            lambda variant=variant: _resolve_cnn_t2b(variant),
        )
    return specs


MODEL_SPECS = _build_model_specs()
MODEL_IDS = tuple(MODEL_SPECS.keys())


def resolve_model_registry(model_ids: tuple[str, ...] | None = None) -> dict[str, dict[str, Any]]:
    """Attempt to resolve every requested model, catching missing checkpoints."""

    manifest: dict[str, dict[str, Any]] = {}
    for model_id in model_ids or MODEL_IDS:
        spec = MODEL_SPECS[model_id]
        try:
            result = spec.resolver()
        except Exception as exc:  # noqa: BLE001 - resolution failures are expected for unreleased models
            result = {"status": "missing_checkpoint", "checkpoint_path": None, "error": str(exc)}
        manifest[model_id] = {
            "family": spec.family,
            "domain": spec.domain,
            "branch": spec.branch,
            **result,
        }
    return manifest


def write_resolved_registry(path: str | Path = DEFAULT_REGISTRY_OUTPUT_PATH) -> Path:
    """Resolve the full registry and write it to a JSON-serializable manifest."""

    manifest = resolve_model_registry()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return path


if __name__ == "__main__":
    written = write_resolved_registry()
    print(f"Wrote model registry manifest to {written}")
