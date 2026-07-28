"""Correctness audits for the atlas-free CNN text-to-brain pipeline.

These helpers deliberately preserve the retained scientific recipe. They test
identity, pairing, frozen-model behavior, gradient flow, latent conditioning,
and output scale without introducing a new loss or architecture.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from neurovlm.atlas_free_text import (
    AtlasFreeTextEmbeddingLookup,
    primary_positive_text,
    primary_positive_text_id,
)
from neurovlm.evaluation.spatial import reconstruction_metrics
from neurovlm.pipelines import (
    atomic_write_csv,
    atomic_write_json,
    sha256_file,
    sha256_state_dict,
)


def tensor_sha256(value: Tensor) -> str:
    """Hash tensor dtype, shape, and contiguous CPU bytes."""

    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def autoencoder_identity(
    autoencoder: nn.Module,
    *,
    checkpoint: str | Path | None = None,
    domain: str | None = None,
    branch: str | None = None,
    selection_metric: str | None = None,
) -> dict[str, Any]:
    """Describe an AE by checkpoint bytes and exact encoder/decoder tensors."""

    encoder = getattr(autoencoder, "encoder", None)
    decoder = getattr(autoencoder, "decoder", None)
    if not isinstance(encoder, nn.Module) or not isinstance(decoder, nn.Module):
        raise TypeError("autoencoder must expose encoder and decoder modules")
    path = None if checkpoint is None else Path(checkpoint)
    dropout = getattr(encoder, "dropout", None)
    if isinstance(dropout, nn.Dropout):
        dropout = float(dropout.p)
    first_block = None
    features = getattr(encoder, "features", None)
    if isinstance(features, nn.Sequential) and len(features):
        first_block = features[0]
    norm = getattr(first_block, "norm", None)
    if isinstance(norm, nn.Module):
        norm = {
            nn.GroupNorm: "group",
            nn.BatchNorm3d: "batch",
            nn.InstanceNorm3d: "instance",
            nn.Identity: "none",
        }.get(type(norm), type(norm).__name__)
    pooling = getattr(first_block, "pool", None)
    if isinstance(pooling, nn.Module):
        pooling = "max" if isinstance(pooling, nn.MaxPool3d) else "stride"
    first_conv = getattr(first_block, "conv", None)
    architecture = {
        "architecture": type(autoencoder).__name__,
        "encoder_architecture": type(encoder).__name__,
        "decoder_architecture": type(decoder).__name__,
        "encoder_arch": (
            "plain" if type(encoder).__name__ == "ALE3DCNNEncoder" else "resnet"
        ),
        "output_shape": tuple(getattr(decoder, "output_shape", ())),
        "latent_dim": getattr(decoder, "latent_dim", None),
        "in_channels": getattr(first_conv, "in_channels", None),
        "base_channels": getattr(decoder, "base_channels", None),
        "num_blocks": getattr(decoder, "num_blocks", None),
        "dropout": dropout,
        "norm": norm,
        "pooling": pooling,
    }
    return {
        "checkpoint_path": None if path is None else str(path.absolute()),
        "checkpoint_sha256": (
            sha256_file(path) if path is not None and path.is_file() else None
        ),
        "state_sha256": sha256_state_dict(autoencoder),
        "encoder_state_sha256": sha256_state_dict(encoder),
        "decoder_state_sha256": sha256_state_dict(decoder),
        "architecture": architecture,
        "domain": domain,
        "branch": branch,
        "selection_metric": selection_metric,
    }


@torch.no_grad()
def ae_ceiling_bypass(
    model: nn.Module,
    target: Tensor,
    *,
    atol: float = 1e-7,
    rtol: float = 1e-6,
) -> dict[str, Any]:
    """Verify that ``z_pred = encoder(target)`` reproduces the AE ceiling."""

    ae = getattr(model, "autoencoder", None)
    projector = getattr(model, "text_projection", None)
    if not isinstance(ae, nn.Module) or not isinstance(projector, nn.Module):
        raise TypeError("model must expose autoencoder and text_projection modules")
    model.eval()
    ae.eval()
    ae.encoder.eval()
    ae.decoder.eval()
    latent = ae.encoder(target)
    reconstruction = ae.decoder(latent)

    # Exercise the ordinary composite-wrapper forward path while bypassing only
    # the projector. The original module is restored even if forward fails.
    model.text_projection = nn.Identity()
    try:
        bypass = model(latent)
    finally:
        model.text_projection = projector

    difference = (bypass - reconstruction).abs()
    reference_metrics = reconstruction_metrics(reconstruction, target)
    bypass_metrics = reconstruction_metrics(bypass, target)
    metric_differences = {
        key: abs(float(bypass_metrics[key]) - float(reference_metrics[key]))
        for key in reference_metrics
    }
    passed = bool(
        torch.allclose(bypass, reconstruction, atol=atol, rtol=rtol)
        and all(value <= max(atol, rtol) for value in metric_differences.values())
    )
    return {
        "max_absolute_voxel_difference": float(difference.max()),
        "mean_absolute_voxel_difference": float(difference.mean()),
        "metric_differences": metric_differences,
        "reference_metrics": reference_metrics,
        "bypass_metrics": bypass_metrics,
        "atol": atol,
        "rtol": rtol,
        "passed": passed,
    }


@torch.no_grad()
def frozen_ae_determinism(
    model: nn.Module,
    target: Tensor,
    *,
    repeats: int = 5,
    atol: float = 0.0,
) -> dict[str, Any]:
    """Test repeatability and the parent-wrapper training-mode invariant."""

    if repeats < 5:
        raise ValueError("The frozen-AE audit requires at least five repeats")
    ae = getattr(model, "autoencoder", None)
    if not isinstance(ae, nn.Module):
        raise TypeError("model must expose an autoencoder module")
    model.train(True)
    stayed_eval_after_parent_train = not ae.training and not ae.encoder.training and not ae.decoder.training
    ae.eval()
    ae.encoder.eval()
    ae.decoder.eval()
    for parameter in ae.parameters():
        parameter.requires_grad_(False)
    latents = [ae.encoder(target).detach().clone() for _ in range(repeats)]
    pairwise = [
        float((latents[left] - latents[right]).abs().max())
        for left in range(repeats)
        for right in range(left + 1, repeats)
    ]
    max_pairwise = max(pairwise, default=0.0)
    frozen = not any(parameter.requires_grad for parameter in ae.parameters())
    passed = stayed_eval_after_parent_train and frozen and max_pairwise <= atol
    return {
        "repeats": repeats,
        "maximum_pairwise_latent_difference": max_pairwise,
        "stayed_eval_after_parent_train": stayed_eval_after_parent_train,
        "all_parameters_frozen": frozen,
        "passed": passed,
    }


def audit_pairings(
    dataset: Any,
    lookup: AtlasFreeTextEmbeddingLookup,
    *,
    minimum: int = 100,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Audit JSONL-row, tensor, primary-text, and cache-index alignment."""

    rows = getattr(dataset, "rows", None)
    tensor_indices = getattr(dataset, "_tensor_indices", None)
    payload_map_ids = getattr(dataset, "_payload_map_ids", None)
    split = str(getattr(dataset, "split", ""))
    if not isinstance(rows, Sequence) or not isinstance(tensor_indices, Sequence):
        raise TypeError("dataset must expose validated rows and tensor indices")
    if len(rows) < minimum:
        raise ValueError(f"Pairing audit requires {minimum} rows, but dataset has {len(rows)}")
    lookup.validate_dataset(rows)
    audit_rows: list[dict[str, Any]] = []
    for dataset_index, row in enumerate(rows[:minimum]):
        text_id = primary_positive_text_id(row)
        embedding = lookup[text_id]
        audit_rows.append(
            {
                "dataset_row_index": dataset_index,
                "tensor_index": int(tensor_indices[dataset_index]),
                "map_id": str(row.get("map_id") or ""),
                "text_id": text_id,
                "primary_text": primary_positive_text(row),
                "text_cache_index": lookup.index(text_id),
                "source": str(row.get("source") or ""),
                "domain": str((row.get("metadata") or row).get("domain") or ""),
                "split": str(row.get("split") or split),
                "embedding_sha256": tensor_sha256(embedding),
                "payload_map_id": (
                    None
                    if payload_map_ids is None
                    else str(payload_map_ids[int(tensor_indices[dataset_index])])
                ),
            }
        )
    passed = all(
        row["map_id"]
        and row["text_id"]
        and row["split"] == split
        and lookup.text_ids[row["text_cache_index"]] == row["text_id"]
        and (
            row["payload_map_id"] is None
            or row["payload_map_id"] == row["map_id"]
        )
        for row in audit_rows
    )
    report = {
        "split": split,
        "dataset_size": len(rows),
        "audited": len(audit_rows),
        "passed": passed,
        "rows": audit_rows,
    }
    if output_dir is not None:
        destination = Path(output_dir)
        atomic_write_json(destination / f"{split}_pairing_audit.json", report)
        atomic_write_csv(destination / f"{split}_pairing_audit.csv", audit_rows)
    return report


def audit_text_preprocessing(lookup: AtlasFreeTextEmbeddingLookup) -> dict[str, Any]:
    """Validate the canonical Stage 4 SPECTER2 cache convention."""

    metadata = dict(lookup.metadata)
    embeddings = lookup.embeddings
    norms = embeddings.norm(dim=1)
    expected = {
        "embedding_dimension": 768,
        "pooling_method": "cls_token",
        "preprocessing_order": [
            "subtract_empty_string_embedding",
            "l2_unit_normalize",
        ],
    }
    checks = {
        "dimension_768": embeddings.ndim == 2 and embeddings.shape[1] == 768,
        "finite": bool(torch.isfinite(embeddings).all()),
        "unit_norm": bool((norms.sub(1).abs() <= 1e-3).float().mean() >= 0.999),
        "cls_pooling": metadata.get("pooling_method") == expected["pooling_method"],
        "single_center_then_normalize": metadata.get("preprocessing_order")
        == expected["preprocessing_order"],
        "adapter_recorded": bool(metadata.get("adapter_id")),
        "model_revision_recorded": bool(metadata.get("model_revision_or_commit_hash")),
        "unique_text_ids": len(set(lookup.text_ids)) == len(lookup.text_ids),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "n": len(lookup),
        "norm": {
            "minimum": float(norms.min()),
            "mean": float(norms.mean()),
            "maximum": float(norms.max()),
        },
        "model": metadata.get("base_model_repository")
        or metadata.get("text_encoder_model_name"),
        "model_revision": metadata.get("model_revision_or_commit_hash"),
        "adapter": metadata.get("adapter_id"),
        "adapter_revision": metadata.get("adapter_revision_or_commit_hash"),
        "pooling": metadata.get("pooling_method"),
        "preprocessing_order": metadata.get("preprocessing_order"),
        "empty_string_embedding_checksum": metadata.get(
            "empty_string_embedding_checksum"
        ),
    }


def audit_raw_latent_path(model: nn.Module) -> dict[str, Any]:
    """Reject transformations that would change the Stage 1 raw latent scale."""

    projector = getattr(model, "text_projection", None)
    ae = getattr(model, "autoencoder", None)
    if not isinstance(projector, nn.Module) or not isinstance(ae, nn.Module):
        raise TypeError("model must expose text_projection and autoencoder")
    forbidden = (nn.LayerNorm, nn.Sigmoid, nn.Tanh, nn.Softmax)
    found = [
        type(module).__name__
        for module in projector.modules()
        if module is not projector and isinstance(module, forbidden)
    ]
    checks = {
        "no_latent_normalization_or_bounding_module": not found,
        "matching_latent_dimension": (
            getattr(ae.decoder, "latent_dim", None)
            == getattr(projector.net[-1], "out_features", None)
        ),
        "decoder_called_directly_by_wrapper": hasattr(model, "forward"),
    }
    return {
        "checks": checks,
        "forbidden_modules": found,
        "passed": all(checks.values()),
    }


def _distribution(value: Tensor) -> dict[str, float]:
    value = value.detach().float().flatten()
    return {
        "minimum": float(value.min()),
        "q05": float(torch.quantile(value, 0.05)),
        "median": float(torch.quantile(value, 0.5)),
        "mean": float(value.mean()),
        "q95": float(torch.quantile(value, 0.95)),
        "maximum": float(value.max()),
        "std": float(value.std(unbiased=False)),
    }


def latent_diagnostics(target: Tensor, prediction: Tensor) -> dict[str, Any]:
    """Return collapse and conditioning diagnostics for raw AE latents."""

    target = torch.as_tensor(target).detach().float().cpu()
    prediction = torch.as_tensor(prediction).detach().float().cpu()
    if target.ndim != 2 or target.shape != prediction.shape:
        raise ValueError("target and prediction must have identical N x D shapes")
    target_mean = target.mean(dim=0)
    prediction_mean = prediction.mean(dim=0)
    target_std = target.std(dim=0, unbiased=False)
    prediction_std = prediction.std(dim=0, unbiased=False)
    residual = target - prediction
    target_centered = target - target_mean
    ss_res = residual.square().sum(dim=0)
    ss_tot = target_centered.square().sum(dim=0)
    r_squared = torch.where(ss_tot > 0, 1 - ss_res / ss_tot, torch.zeros_like(ss_tot))
    explained_variance = 1 - residual.var(dim=0, unbiased=False).sum() / target.var(
        dim=0, unbiased=False
    ).sum().clamp_min(1e-12)
    target_covariance = target_centered.T @ target_centered / max(1, len(target) - 1)
    eigenvalues = torch.linalg.eigvalsh(target_covariance).clamp_min(0).flip(0)
    target_flat_centered = target.flatten() - target.mean()
    prediction_flat_centered = prediction.flatten() - prediction.mean()
    pearson = (
        (target_flat_centered * prediction_flat_centered).sum()
        / (
            target_flat_centered.norm() * prediction_flat_centered.norm()
        ).clamp_min(1e-12)
    )
    target_variance = target_std.square()
    prediction_variance = prediction_std.square()
    target_variance_centered = target_variance - target_variance.mean()
    prediction_variance_centered = prediction_variance - prediction_variance.mean()
    dimension_variance_correlation = (
        (target_variance_centered * prediction_variance_centered).sum()
        / (
            target_variance_centered.norm() * prediction_variance_centered.norm()
        ).clamp_min(1e-12)
    )
    variance_order = target_variance.argsort(descending=True)
    quartile_size = max(1, target.shape[1] // 4)
    high_variance_dimensions = variance_order[:quartile_size]
    low_variance_dimensions = variance_order[-quartile_size:]
    distances = torch.cdist(prediction, target)
    nearest = distances.min(dim=1).values
    distance_to_mean = (prediction - target_mean).norm(dim=1)
    variance_ratio = prediction_std.square().sum() / target_std.square().sum().clamp_min(1e-12)
    return {
        "n": len(target),
        "latent_dim": target.shape[1],
        "target_per_dimension_mean": target_mean.tolist(),
        "target_per_dimension_std": target_std.tolist(),
        "prediction_per_dimension_mean": prediction_mean.tolist(),
        "prediction_per_dimension_std": prediction_std.tolist(),
        "target_norm": _distribution(target.norm(dim=1)),
        "prediction_norm": _distribution(prediction.norm(dim=1)),
        "mean_cosine_similarity": float(
            F.cosine_similarity(prediction, target, dim=1, eps=1e-8).mean()
        ),
        "pearson_correlation": float(pearson),
        "per_dimension_r_squared": r_squared.tolist(),
        "mean_per_dimension_r_squared": float(r_squared.mean()),
        "top_target_variance_quartile_mean_r_squared": float(
            r_squared[high_variance_dimensions].mean()
        ),
        "bottom_target_variance_quartile_mean_r_squared": float(
            r_squared[low_variance_dimensions].mean()
        ),
        "target_prediction_dimension_variance_correlation": float(
            dimension_variance_correlation
        ),
        "global_explained_variance": float(explained_variance),
        "covariance_eigenvalues": eigenvalues.tolist(),
        "predicted_variance_over_target_variance": float(variance_ratio),
        "distance_to_nearest_real_latent": _distribution(nearest),
        "distance_to_mean_target_latent": _distribution(distance_to_mean),
    }


def _top_fraction_distribution(value: Tensor, fraction: float) -> dict[str, float]:
    flat = value.detach().float().reshape(value.shape[0], -1)
    count = max(1, int(math.ceil(flat.shape[1] * fraction)))
    return _distribution(flat.topk(count, dim=1).values)


def volume_scale_diagnostics(target: Tensor, prediction: Tensor) -> dict[str, Any]:
    """Report raw output amplitude and sparsity diagnostics."""

    target = torch.as_tensor(target).detach().float().cpu()
    prediction = torch.as_tensor(prediction).detach().float().cpu()
    if target.shape != prediction.shape or target.ndim < 2:
        raise ValueError("target and prediction must have identical batched shapes")

    def describe(value: Tensor, *, prediction_value: bool) -> dict[str, Any]:
        flat = value.flatten()
        report: dict[str, Any] = {
            **_distribution(flat),
            "positive_fraction": float((flat > 0).float().mean()),
            "nonzero_fraction": float((flat != 0).float().mean()),
            "top_1_percent": _top_fraction_distribution(value, 0.01),
            "top_5_percent": _top_fraction_distribution(value, 0.05),
            "top_10_percent": _top_fraction_distribution(value, 0.10),
        }
        if prediction_value:
            report["negative_fraction_before_clamping"] = float(
                (flat < 0).float().mean()
            )
        return report

    return {
        "target": describe(target, prediction_value=False),
        "prediction": describe(prediction, prediction_value=True),
    }


def _gradient_norm(
    loss: Tensor,
    parameters: Sequence[nn.Parameter],
    *,
    retain_graph: bool,
) -> float:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    total = sum(
        gradient.detach().float().square().sum()
        for gradient in gradients
        if gradient is not None
    )
    return float(total.sqrt())


def loss_gradient_diagnostics(
    model: nn.Module,
    target: Tensor,
    text: Tensor,
    *,
    reconstruction_weight: float = 1.0,
    latent_weight: float = 1.0,
) -> dict[str, Any]:
    """Measure separate effective gradient scales without updating parameters."""

    ae = model.autoencoder
    projector = model.text_projection
    ae.eval()
    projector.train()
    parameters = tuple(parameter for parameter in projector.parameters() if parameter.requires_grad)
    if not parameters:
        raise RuntimeError("Projector has no trainable parameters")
    with torch.no_grad():
        target_latent = ae.encoder(target)
    prediction_latent = projector(text)
    prediction = ae.decoder(prediction_latent)
    reconstruction = F.mse_loss(prediction, target)
    latent = F.mse_loss(prediction_latent, target_latent)
    weighted_reconstruction = reconstruction_weight * reconstruction
    weighted_latent = latent_weight * latent
    reconstruction_gradient = _gradient_norm(
        weighted_reconstruction, parameters, retain_graph=True
    )
    latent_gradient = _gradient_norm(weighted_latent, parameters, retain_graph=True)
    total = weighted_reconstruction + weighted_latent
    total_gradient = _gradient_norm(total, parameters, retain_graph=False)
    return {
        "raw_reconstruction_loss": float(reconstruction.detach()),
        "weighted_reconstruction_contribution": float(
            weighted_reconstruction.detach()
        ),
        "raw_latent_loss": float(latent.detach()),
        "weighted_latent_contribution": float(weighted_latent.detach()),
        "reconstruction_gradient_norm": reconstruction_gradient,
        "latent_gradient_norm": latent_gradient,
        "total_projector_gradient_norm": total_gradient,
        "projector_parameter_count": sum(parameter.numel() for parameter in parameters),
        "projector_in_optimizer_required": True,
        "all_losses_finite": all(
            math.isfinite(value)
            for value in (
                float(reconstruction.detach()),
                float(latent.detach()),
                reconstruction_gradient,
                latent_gradient,
                total_gradient,
            )
        ),
        "gradients_nonzero": reconstruction_gradient > 0
        and latent_gradient > 0
        and total_gradient > 0,
    }


def tiny_overfit_projector(
    model: nn.Module,
    text: Tensor,
    target: Tensor,
    *,
    steps: int = 500,
    learning_rate: float = 1e-3,
    loss_mode: str = "combined",
    shuffled_pairing: bool = False,
    report_every: int = 25,
    foreground_weight: float = 4.0,
) -> dict[str, Any]:
    """Deterministically overfit one fixed paired batch.

    ``loss_mode`` supports the five correctness controls requested by the
    Stage 4 audit: ``latent_only``, ``reconstruction_only``, ``combined``,
    ``standardized_latent``, and ``foreground_combined``.
    """

    allowed = {
        "latent_only",
        "reconstruction_only",
        "combined",
        "standardized_latent",
        "foreground_combined",
    }
    if loss_mode not in allowed:
        raise ValueError(f"loss_mode must be one of {sorted(allowed)}")
    if steps < 1 or report_every < 1 or learning_rate <= 0:
        raise ValueError("steps, report_every, and learning_rate must be positive")
    evaluation_text = torch.as_tensor(text).detach()
    target = torch.as_tensor(target).detach()
    if len(evaluation_text) != len(target):
        raise ValueError("text and target batch sizes must match")
    training_text = (
        evaluation_text.roll(1, dims=0) if shuffled_pairing else evaluation_text
    )

    ae = model.autoencoder
    projector = model.text_projection
    model.train()
    ae.eval()
    for parameter in ae.parameters():
        parameter.requires_grad_(False)
    for parameter in projector.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        projector.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
    )
    with torch.no_grad():
        target_latent = ae.encoder(target)
        ceiling = ae.decoder(target_latent)
    latent_mean = target_latent.mean(dim=0, keepdim=True)
    latent_std = target_latent.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-4)
    initial_parameters = torch.cat(
        [parameter.detach().flatten().cpu() for parameter in projector.parameters()]
    )
    history: list[dict[str, Any]] = []

    def snapshot(step: int, prediction_latent: Tensor, prediction: Tensor) -> None:
        spatial = reconstruction_metrics(prediction, target)
        ceiling_spatial = reconstruction_metrics(ceiling, target)
        history.append(
            {
                "step": step,
                "latent_mse": float(F.mse_loss(prediction_latent, target_latent)),
                "decoded_reconstruction_mse": float(F.mse_loss(prediction, target)),
                "spatial_corr": spatial["spatial_corr"],
                "top5_dice": spatial["top5_dice"],
                "ceiling_spatial_corr": ceiling_spatial["spatial_corr"],
                "ceiling_top5_dice": ceiling_spatial["top5_dice"],
                "target_latent_norm_mean": float(target_latent.norm(dim=1).mean()),
                "prediction_latent_norm_mean": float(
                    prediction_latent.norm(dim=1).mean()
                ),
                "target_voxel_min": float(target.min()),
                "target_voxel_max": float(target.max()),
                "prediction_voxel_min": float(prediction.min()),
                "prediction_voxel_max": float(prediction.max()),
            }
        )

    with torch.no_grad():
        initial_latent = projector(evaluation_text)
        snapshot(0, initial_latent, ae.decoder(initial_latent))
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction_latent = projector(training_text)
        raw_latent = F.mse_loss(prediction_latent, target_latent)
        standardized = F.mse_loss(
            (prediction_latent - latent_mean) / latent_std,
            (target_latent - latent_mean) / latent_std,
        )
        if loss_mode == "latent_only":
            loss = raw_latent
        elif loss_mode == "standardized_latent":
            loss = standardized
        else:
            prediction = ae.decoder(prediction_latent)
            raw_reconstruction = F.mse_loss(prediction, target)
            foreground = (
                (prediction - target).square()
                * (1.0 + foreground_weight * (target > 0).float())
            ).mean()
            if loss_mode == "reconstruction_only":
                loss = raw_reconstruction
            elif loss_mode == "foreground_combined":
                loss = raw_latent + foreground
            else:
                loss = raw_latent + raw_reconstruction
        loss.backward()
        optimizer.step()
        if step == steps or step % report_every == 0:
            with torch.no_grad():
                evaluation_latent = projector(evaluation_text)
                snapshot(step, evaluation_latent, ae.decoder(evaluation_latent))

    final_parameters = torch.cat(
        [parameter.detach().flatten().cpu() for parameter in projector.parameters()]
    )
    return {
        "n": len(target),
        "steps": steps,
        "learning_rate": learning_rate,
        "loss_mode": loss_mode,
        "shuffled_pairing": shuffled_pairing,
        "projector_parameter_update_norm": float(
            (final_parameters - initial_parameters).norm()
        ),
        "history": history,
        "initial": history[0],
        "final": history[-1],
    }


__all__ = [
    "ae_ceiling_bypass",
    "audit_pairings",
    "audit_raw_latent_path",
    "audit_text_preprocessing",
    "autoencoder_identity",
    "frozen_ae_determinism",
    "latent_diagnostics",
    "loss_gradient_diagnostics",
    "tensor_sha256",
    "tiny_overfit_projector",
    "volume_scale_diagnostics",
]
