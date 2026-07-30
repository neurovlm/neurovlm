"""Build the Stage 4 / Stage 3 semantic bridge Colab notebook.

The generated notebook intentionally keeps orchestration in cells while
reusing regression-tested experiment primitives from
``neurovlm.experiments.stage4_semantic_bridge``.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "stage4_stage3_semantic_bridge.ipynb"


def lines(value: str) -> list[str]:
    return dedent(value).lstrip("\n").splitlines(keepends=True)


def markdown(value: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(value)}


def code(value: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(value),
    }


cells = [
    markdown(
        r"""
        # Stage 4 ↔ Stage 3 semantic bridge

        **Question.** Does the strong Stage 3 contrastive space provide a better
        intermediate representation for text-to-brain generation than directly
        regressing normalized SPECTER2 embeddings to raw Stage 1 AE latents?

        Stage 3 retrieval and Stage 4 generation use different spaces. This
        notebook therefore freezes the released Stage 1 AE and Stage 3
        brain/text encoders, then compares six separately labeled paths:

        - **A `direct_baseline`:** normalized SPECTER2 → retained 768→512→384 projector.
        - **B `stage3_text_bridge`:** frozen Stage 3 text semantics → trainable bridge.
        - **C `stage3_brain_bridge_oracle`:** frozen Stage 3 brain semantics → trainable
          bridge. This is a diagnostic oracle, never an inference model.
        - **D `shared_bridge_dual_supervision`:** one bridge supervised with paired text
          and brain semantic vectors.
        - **E `concatenated_text_semantic`:** raw 768-d text + 384-d Stage 3 text semantics.
        - **F `residual_direct_plus_semantic`:** retained direct prediction + semantic residual.

        The primary architecture comparison always optimizes **raw latent MSE +
        decoded-volume MSE**. Standardized-latent, cosine, and norm objectives
        are a separate loss-sensitivity axis and are never pooled into the
        architecture conclusion. Test data are not loaded for model selection;
        only validation-selected finalists are evaluated on test.

        The notebook is resumable from Google Drive. A resumed cache/checkpoint
        must exactly match repository commit, split order, AE file/state, Stage 3
        file/brain/text states, text cache, architecture, and effective config.
        No released Stage 1, Stage 3, or Stage 4 checkpoint is modified.
        """
    ),
    markdown(
        r"""
        ## Interpretation contract

        The final report applies these predeclared rules to validation metrics:

        1. Brain-semantic oracle works but text-semantic does not → Stage 3
           cross-modal alignment is insufficient for precise generation.
        2. Neither works → Stage 3 semantics discard spatial information required
           by the AE decoder.
        3. Both work → the direct SPECTER2→AE projector is the main bottleneck.
        4. Concatenated raw text + semantics is best → the two text
           representations are complementary.

        “Works” is operationalized below by fixed validation thresholds on
        global latent explained variance and spatial correlation. Change those
        thresholds before training if a different scientific criterion is desired.
        Shuffled B and C controls use fixed, split-wide derangements with no
        self-pairs.
        """
    ),
    code(
        r"""
        # Fresh-Colab setup: mount Drive, check out an exact repository ref, install.
        import os
        import subprocess
        import sys
        from pathlib import Path

        IN_COLAB = "google.colab" in sys.modules
        DRIVE_ROOT = Path("/content/drive/MyDrive") if IN_COLAB else Path.cwd()
        if IN_COLAB:
            from google.colab import drive
            drive.mount("/content/drive")

        REPO_URL = "https://github.com/neurovlm/neurovlm.git"
        REPO_REF = os.environ.get("NEUROVLM_REPO_REF", "main")
        EXPECTED_COMMIT = os.environ.get("NEUROVLM_EXPECTED_COMMIT", "").strip() or None
        REPO_DIR = Path("/content/neurovlm") if IN_COLAB else Path.cwd()

        if IN_COLAB:
            if not (REPO_DIR / ".git").exists():
                subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)
            subprocess.run(["git", "fetch", "--tags", "origin"], cwd=REPO_DIR, check=True)
            subprocess.run(["git", "checkout", REPO_REF], cwd=REPO_DIR, check=True)
        resolved_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_DIR,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if EXPECTED_COMMIT is not None and resolved_commit != EXPECTED_COMMIT:
            raise RuntimeError(
                f"Repository commit mismatch: expected {EXPECTED_COMMIT}, got {resolved_commit}"
            )
        if IN_COLAB:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-q",
                    "-e",
                    f"{REPO_DIR}[metrics,viz,notebook]",
                ],
                check=True,
            )
        os.chdir(REPO_DIR)
        print({"repo": str(REPO_DIR), "commit": resolved_commit, "drive": str(DRIVE_ROOT)})
        """
    ),
    code(
        r"""
        # Imports, deterministic execution, CUDA/BF16 policy, and shared helpers.
        import copy
        import json
        import math
        import os
        import platform
        import random
        import tempfile
        import time
        from dataclasses import asdict
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import torch
        import torch.nn.functional as F
        from torch import nn
        from torch.utils.data import DataLoader

        from neurovlm.atlas_free_dataset import AtlasFreeCNNDataProvider
        from neurovlm.atlas_free_text import (
            AtlasFreeContrastiveCollator,
            AtlasFreeTextEmbeddingLookup,
        )
        from neurovlm.evaluation.spatial import reconstruction_metrics
        from neurovlm.evaluation.text_to_brain_audit import (
            audit_pairings,
            audit_text_preprocessing,
            autoencoder_identity,
        )
        from neurovlm.experiments.stage4_latent_ablation import (
            split_fingerprint,
            text_cache_identity,
        )
        from neurovlm.experiments.stage4_semantic_bridge import (
            BRIDGE_ARCHITECTURES,
            BRIDGE_LOSS_VARIANTS,
            BRIDGE_PATHS,
            BridgeCheckpointManager,
            BridgeLossConfig,
            bridge_architecture_record,
            bridge_latent_metrics,
            build_bridge_model,
            compute_bridge_loss,
            fixed_derangement,
            freeze_module,
            semantic_alignment_metrics,
            stage3_identity,
            validate_semantic_embeddings,
        )
        from neurovlm.pipelines import (
            atomic_write_csv,
            atomic_write_json,
            environment_provenance,
            git_provenance,
            sha256_file,
            sha256_state_dict,
            sha256_value,
        )
        from neurovlm.semantic_evaluation import evaluate_semantic_neighbor_retrieval
        import neurovlm.retrieval_resources as rr

        SEED = 42

        def seed_everything(seed=SEED):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        seed_everything()
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        BF16_AVAILABLE = bool(
            DEVICE.type == "cuda" and torch.cuda.is_bf16_supported()
        )
        MIXED_PRECISION_DTYPE = (
            torch.bfloat16
            if BF16_AVAILABLE
            else (torch.float16 if DEVICE.type == "cuda" else torch.float32)
        )
        USE_AMP = DEVICE.type == "cuda"
        torch.set_float32_matmul_precision("high")

        def atomic_torch_save(path, payload):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                torch.save(payload, temporary)
                os.replace(temporary, path)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise

        ENVIRONMENT = {
            **environment_provenance(
                ("neurovlm", "torch", "numpy", "pandas", "matplotlib")
            ),
            "device": str(DEVICE),
            "cuda": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(0)
                if DEVICE.type == "cuda"
                else None
            ),
            "bf16_available": BF16_AVAILABLE,
            "mixed_precision_dtype": str(MIXED_PRECISION_DTYPE),
            "deterministic_algorithms": True,
            "git": git_provenance(REPO_DIR),
        }
        ENVIRONMENT
        """
    ),
    markdown("## Configuration\n\n`FAST_RUN` is a smoke test. Set `FULL_RUN=True` for evidence."),
    code(
        r"""
        # Mixed Stage 1A AE only: domain-finetuned AEs did not improve the
        # reconstruction/contrastive prerequisites and are excluded.
        ALL_BRANCHES = [
            "mixed_to_pubmed",
            "mixed_to_nilearn",
            "mixed_to_neurovault",
        ]
        BRANCHES_TO_RUN = ALL_BRANCHES
        unknown_branches = sorted(set(BRANCHES_TO_RUN) - set(ALL_BRANCHES))
        if unknown_branches:
            raise ValueError(f"Unknown branches: {unknown_branches}")

        SUPPORTED_BRIDGE_ARCHITECTURES = [
            "mlp_512",
            "deep_mlp_1024",
            "residual_mlp_1024",
        ]
        if tuple(SUPPORTED_BRIDGE_ARCHITECTURES) != tuple(BRIDGE_ARCHITECTURES):
            raise RuntimeError("Notebook/package bridge architecture contract changed")
        CHECKPOINT_FILES = {
            "top5_dice": "best_top5_dice.pt",
            "spatial_correlation": "best_spatial_correlation.pt",
            "latent_explained_variance": "best_latent_explained_variance.pt",
            "semantic_normalized_auc": "best_semantic_normalized_auc.pt",
            "last": "last.pt",
        }
        REQUIRED_METRICS = [
            "raw_latent_mse",
            "standardized_latent_mse",
            "latent_cosine",
            "latent_variance_ratio",
            "latent_norm_ratio",
            "global_explained_variance",
            "mean_per_dimension_r_squared",
            "nearest_real_latent_distance",
            "decoded_mse",
            "foreground_mse",
            "spatial_corr",
            "top5_dice",
            "semantic_normalized_auc",
            "stage3_text_brain_matched_cosine",
            "stage3_text_brain_shuffled_cosine",
        ]

        FAST_RUN = True
        FULL_RUN = False
        if FAST_RUN == FULL_RUN:
            raise ValueError("Select exactly one of FAST_RUN and FULL_RUN")

        RUN_SHUFFLED_CONTROLS = True
        RUN_SECONDARY_LOSS_VARIANTS = True
        RUN_FINAL_TEST = True
        RESUME = True

        PRIMARY_BRIDGE_ARCHITECTURES = (
            list(BRIDGE_ARCHITECTURES) if FULL_RUN else ["mlp_512"]
        )
        LOSS_SENSITIVITY_ARCHITECTURE = "mlp_512"
        LOSS_SENSITIVITY_PATHS = [
            "stage3_text_bridge",
            "stage3_brain_bridge_oracle",
        ]
        SECONDARY_LOSS_VARIANTS = [
            "standardized_decoded",
            "standardized_cosine_decoded",
            "standardized_cosine_norm_decoded",
        ]

        EPOCHS = 50 if FULL_RUN else 2
        BATCH_SIZE = 64 if FULL_RUN else 8
        EVAL_BATCH_SIZE = 128 if FULL_RUN else 32
        NUM_WORKERS = 8 if IN_COLAB else 0
        PREFETCH_FACTOR = 4
        LEARNING_RATE = 3e-4
        WEIGHT_DECAY = 1e-4
        GRADIENT_CLIP = 1.0
        MAX_TRAIN_BATCHES = None if FULL_RUN else 2
        MAX_EVAL_BATCHES = None if FULL_RUN else 2
        SEMANTIC_MAX_EXAMPLES = 2048 if FULL_RUN else 16
        GENERATED_EXAMPLES = 6
        FINAL_CHECKPOINT_ROLE = "top5_dice"

        PRIMARY_LOSS = BridgeLossConfig(
            variant="primary_raw_decoded",
            latent_weight=1.0,
            decoded_weight=1.0,
        )
        SECONDARY_LOSS_CONFIGS = {
            "standardized_decoded": BridgeLossConfig(
                variant="standardized_decoded",
                latent_weight=1.0,
                decoded_weight=1.0,
            ),
            "standardized_cosine_decoded": BridgeLossConfig(
                variant="standardized_cosine_decoded",
                latent_weight=1.0,
                decoded_weight=1.0,
                cosine_weight=0.1,
            ),
            "standardized_cosine_norm_decoded": BridgeLossConfig(
                variant="standardized_cosine_norm_decoded",
                latent_weight=1.0,
                decoded_weight=1.0,
                cosine_weight=0.1,
                norm_weight=0.1,
            ),
        }

        # Predeclared interpretation thresholds; validation only.
        WORKS_MIN_GLOBAL_EXPLAINED_VARIANCE = 0.10
        WORKS_MIN_SPATIAL_CORRELATION = 0.10

        EXPERIMENT_NAME = "stage4_stage3_semantic_bridge"
        OUTPUT_ROOT = (
            DRIVE_ROOT / "neurovlm_runs" / EXPERIMENT_NAME / resolved_commit[:12]
        )
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        print({"output_root": str(OUTPUT_ROOT), "branches": BRANCHES_TO_RUN})
        """
    ),
    markdown(
        r"""
        ## Strict branch resources and provenance

        Every branch pairs the released mixed Stage 1A AE with the
        corresponding released Stage 3 domain branch. Domain-finetuned AEs are
        deliberately excluded. The Stage 3 brain encoder and text projector
        share one composite file, but their tensor states are fingerprinted
        separately.
        """
    ),
    code(
        r"""
        BRANCH_SPECS = {
            "mixed_to_pubmed": {"domain": "pubmed", "ae_variant": "mixed"},
            "mixed_to_nilearn": {"domain": "nilearn", "ae_variant": "mixed"},
            "mixed_to_neurovault": {"domain": "neurovault", "ae_variant": "mixed"},
        }

        def load_branch_resources(branch):
            spec = {**BRANCH_SPECS[branch], "branch": branch}
            ae_filename = rr.CNN_AUTOENCODER_FILENAMES[spec["ae_variant"]]
            ae_path = Path(
                rr._download_from_hf(
                    rr.ATLAS_FREE_CNN_MODEL_REPO_ID,
                    ae_filename,
                    repo_type="model",
                )
            )
            stage3_path = Path(rr._load_cnn_contrastive_checkpoint_path(branch))
            autoencoder = freeze_module(rr._load_cnn_autoencoder(spec["ae_variant"]))
            stage3_model = freeze_module(rr._load_cnn_contrastive(branch))
            provider = AtlasFreeCNNDataProvider(domain=spec["domain"])
            return spec, ae_path, stage3_path, autoencoder, stage3_model, provider

        def collect_branch_provenance(
            branch, spec, ae_path, stage3_path, autoencoder, stage3_model, provider, lookup
        ):
            split_records = {
                split: split_fingerprint(getattr(provider, split))
                for split in ("train", "val", "test")
            }
            audits = {}
            audit_dir = OUTPUT_ROOT / branch / "provenance_audits"
            for split in ("train", "val", "test"):
                dataset = getattr(provider, split)
                audits[split] = audit_pairings(
                    dataset,
                    lookup,
                    minimum=min(100, len(dataset)),
                    output_dir=audit_dir,
                )
                if not audits[split]["passed"]:
                    raise RuntimeError(f"{branch}/{split} pairing audit failed")
            text_audit = audit_text_preprocessing(lookup)
            if not text_audit["passed"]:
                raise RuntimeError("Published normalized SPECTER2 audit failed")
            ae_record = autoencoder_identity(
                autoencoder,
                checkpoint=ae_path,
                domain=spec["domain"],
                branch=branch,
            )
            stage3_record = stage3_identity(
                stage3_model,
                checkpoint=stage3_path,
                branch=branch,
            )
            text_cache_path = Path(
                rr._download_from_hf(
                    rr.ATLAS_FREE_CNN_DATASET_REPO,
                    rr.ATLAS_FREE_CNN_NORMALIZED_SPECTER2_FILENAME,
                )
            )
            normalized_cache_record = {
                **text_cache_identity(lookup),
                "checkpoint_path": str(text_cache_path.absolute()),
                "checkpoint_sha256": sha256_file(text_cache_path),
            }
            if ae_record["architecture"]["latent_dim"] != 384:
                raise RuntimeError("Stage 1 AE latent dimension must be 384")
            return {
                "format_version": 1,
                "repo_commit": resolved_commit,
                "branch": spec,
                "stage1_autoencoder": ae_record,
                "stage3": stage3_record,
                "normalized_specter2_cache": normalized_cache_record,
                "splits": split_records,
                "pairing_audits": {
                    key: {k: v for k, v in value.items() if k != "rows"}
                    for key, value in audits.items()
                },
                "text_preprocessing_audit": text_audit,
                "semantic_space": {
                    "dimension": 384,
                    "text": "F.normalize(stage3.text_projection(specter2), dim=-1)",
                    "brain": "F.normalize(stage3.brain_encoder(volume), dim=-1)",
                    "normalization": "l2_unit_normalized",
                },
                "ae_bridge_output": {
                    "dimension": 384,
                    "normalization": None,
                    "convention": "raw_stage1_ae_latent",
                },
            }
        """
    ),
    markdown(
        r"""
        ## Ordered latent/semantic caches

        Cache entries contain only frozen model outputs, never trainable bridge
        predictions. Each split-wide shuffle is a saved derangement. Cache reuse
        is refused unless its full identity binding matches.
        """
    ),
    code(
        r"""
        def make_loader(dataset, lookup, *, batch_size, shuffle, seed):
            lookup.validate_dataset(dataset.rows)
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=NUM_WORKERS,
                collate_fn=AtlasFreeContrastiveCollator(lookup, (36, 45, 38)),
                pin_memory=DEVICE.type == "cuda",
                persistent_workers=NUM_WORKERS > 0,
                generator=torch.Generator().manual_seed(seed),
                **({"prefetch_factor": PREFETCH_FACTOR} if NUM_WORKERS > 0 else {}),
            )

        def cache_binding(provenance, split):
            return {
                "format_version": 1,
                "split": provenance["splits"][split],
                "ae_encoder_state_sha256": provenance["stage1_autoencoder"][
                    "encoder_state_sha256"
                ],
                "stage3_brain_state_sha256": provenance["stage3"]["brain_encoder"][
                    "state_sha256"
                ],
                "stage3_text_state_sha256": provenance["stage3"]["text_projection"][
                    "state_sha256"
                ],
                "text_cache": provenance["normalized_specter2_cache"],
                "semantic_dimension": 384,
                "semantic_normalization": "l2_unit_normalized",
                "target_latent_convention": "raw_stage1_ae_latent",
            }

        @torch.no_grad()
        def build_or_load_frozen_cache(
            branch_dir, split, dataset, lookup, autoencoder, stage3_model, provenance
        ):
            binding = cache_binding(provenance, split)
            binding_sha = sha256_value(binding)
            path = branch_dir / "frozen_caches" / f"{split}_{binding_sha[:16]}.pt"
            if path.exists():
                payload = torch.load(path, map_location="cpu", weights_only=True)
                if payload.get("binding_sha256") != binding_sha:
                    raise ValueError(f"{split} frozen cache binding mismatch")
            else:
                autoencoder.to(DEVICE).eval()
                stage3_model.to(DEVICE).eval()
                targets, text_semantics, brain_semantics = [], [], []
                map_ids, text_ids = [], []
                loader = make_loader(
                    dataset,
                    lookup,
                    batch_size=EVAL_BATCH_SIZE,
                    shuffle=False,
                    seed=SEED,
                )
                for batch in loader:
                    volume = batch["volume"].to(DEVICE, non_blocking=True)
                    raw_text = batch["text_embedding"].to(DEVICE, non_blocking=True)
                    targets.append(autoencoder.encoder(volume).float().cpu())
                    text_semantics.append(stage3_model.encode_text(raw_text).float().cpu())
                    brain_semantics.append(stage3_model.encode_brain(volume).float().cpu())
                    map_ids.extend(str(value) for value in batch["map_id"])
                    text_ids.extend(str(value) for value in batch["text_id"])
                payload = {
                    "format_version": 1,
                    "binding": binding,
                    "binding_sha256": binding_sha,
                    "target_latent": torch.cat(targets),
                    "text_semantic": torch.cat(text_semantics),
                    "brain_semantic": torch.cat(brain_semantics),
                    "derangement": fixed_derangement(len(dataset), SEED + len(split)),
                    "map_ids": map_ids,
                    "text_ids": text_ids,
                }
                atomic_torch_save(path, payload)
            expected_n = len(dataset)
            for key in ("target_latent", "text_semantic", "brain_semantic"):
                if len(payload[key]) != expected_n:
                    raise RuntimeError(f"{split} cache count mismatch for {key}")
            validate_semantic_embeddings(
                payload["text_semantic"], label=f"{split}_stage3_text_semantic"
            )
            validate_semantic_embeddings(
                payload["brain_semantic"], label=f"{split}_stage3_brain_semantic"
            )
            if payload["target_latent"].shape != (expected_n, 384):
                raise RuntimeError("Raw Stage 1 latent cache must be N x 384")
            if torch.equal(
                payload["derangement"], torch.arange(expected_n)
            ) or bool(
                (payload["derangement"] == torch.arange(expected_n)).any()
            ):
                raise RuntimeError("Shuffled-control cache is not a derangement")
            return payload, path

        def cache_indices(batch):
            indices = torch.as_tensor(batch["dataset_index"], dtype=torch.long)
            if bool((indices < 0).any()):
                raise RuntimeError("Dataset indices must be non-negative")
            return indices

        def conditioning_from_cache(batch, cache, *, shuffled_control=None):
            indices = cache_indices(batch)
            selected = (
                cache["derangement"][indices]
                if shuffled_control is not None
                else indices
            )
            return (
                cache["text_semantic"][selected].to(DEVICE, non_blocking=True),
                cache["brain_semantic"][selected].to(DEVICE, non_blocking=True),
                cache["target_latent"][indices].to(DEVICE, non_blocking=True),
            )
        """
    ),
    markdown("## Path dispatch, primary/secondary losses, and training"),
    code(
        r"""
        def predict_raw_latent(
            model, path, raw_text, text_semantic, brain_semantic
        ):
            if path == "direct_baseline":
                return model(raw_text)
            if path == "stage3_text_bridge":
                return model(text_semantic)
            if path == "stage3_brain_bridge_oracle":
                return model(brain_semantic)
            if path == "shared_bridge_dual_supervision":
                return model(text_semantic, brain_semantic)
            if path in {
                "concatenated_text_semantic",
                "residual_direct_plus_semantic",
            }:
                return model(raw_text, text_semantic)
            raise ValueError(path)

        def loss_for_prediction(
            prediction, target_latent, target_volume, decoder, latent_mean, latent_std, loss_config
        ):
            prediction_volume = decoder(prediction)
            result = compute_bridge_loss(
                prediction,
                target_latent,
                prediction_volume,
                target_volume,
                training_latent_mean=latent_mean,
                training_latent_std=latent_std,
                config=loss_config,
            )
            return result, prediction_volume

        def train_epoch(
            model,
            path,
            dataset,
            cache,
            lookup,
            autoencoder,
            optimizer,
            *,
            latent_mean,
            latent_std,
            loss_config,
            shuffled_control,
            epoch,
        ):
            model.train()
            autoencoder.decoder.eval()
            loader = make_loader(
                dataset,
                lookup,
                batch_size=BATCH_SIZE,
                shuffle=True,
                seed=SEED + epoch,
            )
            scaler = torch.amp.GradScaler(
                "cuda",
                enabled=USE_AMP and MIXED_PRECISION_DTYPE == torch.float16,
            )
            totals, n = {}, 0
            for batch_index, batch in enumerate(loader):
                if MAX_TRAIN_BATCHES is not None and batch_index >= MAX_TRAIN_BATCHES:
                    break
                raw_text = batch["text_embedding"].to(DEVICE, non_blocking=True)
                target_volume = batch["volume"].to(DEVICE, non_blocking=True)
                text_semantic, brain_semantic, target_latent = conditioning_from_cache(
                    batch, cache, shuffled_control=shuffled_control
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=DEVICE.type,
                    dtype=MIXED_PRECISION_DTYPE,
                    enabled=USE_AMP,
                ):
                    prediction = predict_raw_latent(
                        model, path, raw_text, text_semantic, brain_semantic
                    )
                    if path == "shared_bridge_dual_supervision":
                        text_loss, _ = loss_for_prediction(
                            prediction[0], target_latent, target_volume,
                            autoencoder.decoder, latent_mean, latent_std, loss_config
                        )
                        brain_loss, _ = loss_for_prediction(
                            prediction[1], target_latent, target_volume,
                            autoencoder.decoder, latent_mean, latent_std, loss_config
                        )
                        loss = (text_loss.total + brain_loss.total) / 2
                        detached = {
                            key: (text_loss.detached()[key] + brain_loss.detached()[key]) / 2
                            for key in text_loss.detached()
                        }
                    else:
                        one_loss, _ = loss_for_prediction(
                            prediction, target_latent, target_volume,
                            autoencoder.decoder, latent_mean, latent_std, loss_config
                        )
                        loss = one_loss.total
                        detached = one_loss.detached()
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                    optimizer.step()
                batch_n = len(raw_text)
                n += batch_n
                for key, value in detached.items():
                    totals[key] = totals.get(key, 0.0) + float(value) * batch_n
            if not n:
                raise RuntimeError("Training produced no examples")
            return {f"train_{key}": value / n for key, value in totals.items()}
        """
    ),
    markdown(
        r"""
        ## Evaluation

        Spatial metrics use the repository convention: generated and target
        maps are made finite and clamped only inside metrics. Training never
        clamps. Semantic normalized recall AUC embeds generated volumes with
        the frozen branch-matched Stage 3 brain encoder and ranks the paired
        Stage 3 text semantics, with nearest raw-SPECTER2 neighbors as additional
        positives. Latent nearest-neighbor distance uses training latents only.
        """
    ),
    code(
        r"""
        @torch.no_grad()
        def evaluate_run(
            model,
            path,
            dataset,
            cache,
            train_cache,
            lookup,
            autoencoder,
            stage3_model,
            *,
            latent_mean,
            latent_std,
            shuffled_control,
            split,
        ):
            model.to(DEVICE).eval()
            autoencoder.to(DEVICE).eval()
            stage3_model.to(DEVICE).eval()
            spatial_totals, n = {}, 0
            predictions, targets = [], []
            secondary_predictions = []
            semantic_brain, semantic_text, semantic_raw, semantic_ids = [], [], [], []
            examples = []
            loader = make_loader(
                dataset,
                lookup,
                batch_size=EVAL_BATCH_SIZE,
                shuffle=False,
                seed=SEED,
            )
            for batch_index, batch in enumerate(loader):
                if MAX_EVAL_BATCHES is not None and batch_index >= MAX_EVAL_BATCHES:
                    break
                raw_text = batch["text_embedding"].to(DEVICE, non_blocking=True)
                target_volume = batch["volume"].to(DEVICE, non_blocking=True)
                text_semantic, brain_semantic, target_latent = conditioning_from_cache(
                    batch, cache, shuffled_control=shuffled_control
                )
                with torch.autocast(
                    device_type=DEVICE.type,
                    dtype=MIXED_PRECISION_DTYPE,
                    enabled=USE_AMP,
                ):
                    output = predict_raw_latent(
                        model, path, raw_text, text_semantic, brain_semantic
                    )
                    if path == "shared_bridge_dual_supervision":
                        prediction, secondary = output
                        secondary_predictions.append(secondary.float().cpu())
                    else:
                        prediction = output
                    prediction_volume = autoencoder.decoder(prediction)
                spatial = reconstruction_metrics(prediction_volume, target_volume)
                batch_n = len(raw_text)
                n += batch_n
                for key, value in spatial.items():
                    spatial_totals[key] = (
                        spatial_totals.get(key, 0.0) + float(value) * batch_n
                    )
                predictions.append(prediction.float().cpu())
                targets.append(target_latent.float().cpu())
                remaining = GENERATED_EXAMPLES - len(examples)
                for index in range(min(remaining, batch_n)):
                    examples.append(
                        {
                            "map_id": str(batch["map_id"][index]),
                            "text_id": str(batch["text_id"][index]),
                            "prediction": prediction_volume[index].float().cpu(),
                            "target": target_volume[index].float().cpu(),
                        }
                    )
                already = sum(len(value) for value in semantic_brain)
                take = min(batch_n, max(0, SEMANTIC_MAX_EXAMPLES - already))
                if take:
                    safe_volume = torch.nan_to_num(
                        prediction_volume[:take],
                        nan=0.0,
                        posinf=1.0,
                        neginf=0.0,
                    ).clamp(0, 1).float()
                    semantic_brain.append(
                        stage3_model.encode_brain(safe_volume).float().cpu()
                    )
                    # Always use the correctly paired text semantics to assess
                    # the generated image, including shuffled-control models.
                    indices = cache_indices(batch)[:take]
                    semantic_text.append(cache["text_semantic"][indices])
                    semantic_raw.append(raw_text[:take].float().cpu())
                    semantic_ids.extend(str(value) for value in batch["map_id"][:take])
            if n < 2:
                raise RuntimeError(f"{split} evaluation requires at least two examples")
            target_tensor = torch.cat(targets)
            prediction_tensor = torch.cat(predictions)
            latent_metrics, per_dimension = bridge_latent_metrics(
                target_tensor,
                prediction_tensor,
                training_mean=latent_mean,
                training_std=latent_std,
                nearest_reference=train_cache["target_latent"],
                distance_device=DEVICE,
            )
            summary = {
                **latent_metrics,
                **{key: value / n for key, value in spatial_totals.items()},
                "decoded_mse": spatial_totals["reconstruction_mse"] / n,
                "n": n,
            }
            if len(semantic_ids) >= 2:
                semantic_metrics, _ = evaluate_semantic_neighbor_retrieval(
                    torch.cat(semantic_brain),
                    torch.cat(semantic_text),
                    semantic_ids,
                    neighbor_text_embeddings=torch.cat(semantic_raw),
                    n_neighbors=min(10, len(semantic_ids) - 1),
                )
                summary["semantic_normalized_auc"] = float(
                    semantic_metrics["semantic_normalized_k_recall_curve_auc"]
                )
            else:
                summary["semantic_normalized_auc"] = float("nan")
            evaluated_indices = torch.arange(n)
            alignment = semantic_alignment_metrics(
                cache["text_semantic"][evaluated_indices],
                cache["brain_semantic"][evaluated_indices],
                shuffled_indices=cache["derangement"][evaluated_indices],
            )
            summary.update(alignment)
            if secondary_predictions:
                secondary_metrics, _ = bridge_latent_metrics(
                    target_tensor,
                    torch.cat(secondary_predictions),
                    training_mean=latent_mean,
                    training_std=latent_std,
                    nearest_reference=train_cache["target_latent"],
                    distance_device=DEVICE,
                )
                summary.update(
                    {f"brain_supervision_{key}": value for key, value in secondary_metrics.items()}
                )
            missing_metrics = sorted(set(REQUIRED_METRICS) - set(summary))
            if missing_metrics:
                raise RuntimeError(
                    f"{split} evaluation omitted required metrics: {missing_metrics}"
                )
            return {
                "summary": summary,
                "per_dimension": per_dimension,
                "examples": examples,
            }
        """
    ),
    code(
        r"""
        def run_id_for(spec):
            suffix = "__shuffled_control" if spec["shuffled_control"] else ""
            return (
                f"{spec['experiment_axis']}__{spec['path']}{suffix}"
                f"__{spec['architecture']}__{spec['loss'].variant}"
            )

        def train_one_run(
            branch_dir,
            spec,
            provenance,
            autoencoder,
            stage3_model,
            provider,
            lookup,
            caches,
            cache_paths,
        ):
            run_id = run_id_for(spec)
            run_dir = branch_dir / "runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            seed_everything(SEED)
            model = build_bridge_model(
                spec["path"], architecture=spec["architecture"]
            ).to(DEVICE)
            architecture = bridge_architecture_record(
                spec["path"], spec["architecture"], model
            )
            effective_config = {
                "run_id": run_id,
                "branch": provenance["branch"],
                "path": spec["path"],
                "architecture": architecture,
                "loss": spec["loss"].effective_dict(),
                "experiment_axis": spec["experiment_axis"],
                "shuffled_control": spec["shuffled_control"],
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "gradient_clip": GRADIENT_CLIP,
                "max_train_batches": MAX_TRAIN_BATCHES,
                "max_eval_batches": MAX_EVAL_BATCHES,
                "seed": SEED,
                "amp_dtype": str(MIXED_PRECISION_DTYPE),
                "stage1_frozen": True,
                "stage3_frozen": True,
                "test_used_for_selection": False,
            }
            binding = {
                "repo_commit": provenance["repo_commit"],
                "stage1_autoencoder": provenance["stage1_autoencoder"],
                "stage3": provenance["stage3"],
                "normalized_specter2_cache": provenance[
                    "normalized_specter2_cache"
                ],
                "splits": provenance["splits"],
                "frozen_cache_files": {
                    key: {
                        "path": str(value),
                        "sha256": sha256_file(value),
                    }
                    for key, value in cache_paths.items()
                },
            }
            atomic_write_json(run_dir / "effective_config.json", effective_config)
            atomic_write_json(run_dir / "binding.json", binding)
            atomic_write_json(run_dir / "architecture.json", architecture)
            manager = BridgeCheckpointManager(
                run_dir,
                binding=binding,
                effective_config=effective_config,
                architecture=architecture,
            )
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
            )
            latent_mean = caches["train"]["target_latent"].mean(dim=0)
            latent_std = caches["train"]["target_latent"].std(
                dim=0, unbiased=False
            ).clamp_min(1e-6)
            history_path = run_dir / "training_history.csv"
            history = (
                pd.read_csv(history_path).to_dict("records")
                if history_path.exists()
                else []
            )
            start_epoch = 1
            if RESUME:
                try:
                    payload = manager.load(
                        "last", model=model, optimizer=optimizer, map_location=DEVICE
                    )
                    start_epoch = int(payload["epoch"]) + 1
                    print(f"Resuming {run_id} at epoch {start_epoch}")
                except FileNotFoundError:
                    pass
            shuffled_kind = (
                spec["path"] if spec["shuffled_control"] else None
            )
            for epoch in range(start_epoch, EPOCHS + 1):
                started = time.perf_counter()
                train_metrics = train_epoch(
                    model,
                    spec["path"],
                    provider.train,
                    caches["train"],
                    lookup,
                    autoencoder,
                    optimizer,
                    latent_mean=latent_mean,
                    latent_std=latent_std,
                    loss_config=spec["loss"],
                    shuffled_control=shuffled_kind,
                    epoch=epoch,
                )
                validation = evaluate_run(
                    model,
                    spec["path"],
                    provider.val,
                    caches["val"],
                    caches["train"],
                    lookup,
                    autoencoder,
                    stage3_model,
                    latent_mean=latent_mean,
                    latent_std=latent_std,
                    shuffled_control=shuffled_kind,
                    split="val",
                )
                row = {
                    "epoch": epoch,
                    "epoch_seconds": time.perf_counter() - started,
                    **train_metrics,
                    **{
                        f"val_{key}": value
                        for key, value in validation["summary"].items()
                    },
                }
                history = [
                    old for old in history if int(old["epoch"]) != epoch
                ] + [row]
                history = sorted(history, key=lambda item: int(item["epoch"]))
                atomic_write_csv(history_path, history)
                manager.save_epoch(
                    model,
                    optimizer,
                    epoch=epoch,
                    metrics=validation["summary"],
                    extra={"history_rows": len(history)},
                )
                print(
                    run_id,
                    epoch,
                    {
                        "top5_dice": validation["summary"]["top5_dice"],
                        "spatial_corr": validation["summary"]["spatial_corr"],
                        "global_explained_variance": validation["summary"][
                            "global_explained_variance"
                        ],
                    },
                )
            manager.load(
                FINAL_CHECKPOINT_ROLE,
                model=model,
                map_location=DEVICE,
            )
            selected_validation = evaluate_run(
                model,
                spec["path"],
                provider.val,
                caches["val"],
                caches["train"],
                lookup,
                autoencoder,
                stage3_model,
                latent_mean=latent_mean,
                latent_std=latent_std,
                shuffled_control=shuffled_kind,
                split="val_selected",
            )
            atomic_write_json(
                run_dir / "validation_selected_metrics.json",
                selected_validation["summary"],
            )
            atomic_write_csv(
                run_dir / "validation_per_dimension.csv",
                selected_validation["per_dimension"],
            )
            model.to("cpu")
            return {
                "run_id": run_id,
                "run_dir": run_dir,
                "spec": spec,
                "model": model,
                "manager": manager,
                "latent_mean": latent_mean,
                "latent_std": latent_std,
                "validation": selected_validation,
            }
        """
    ),
    markdown(
        r"""
        ## Run grid

        The architecture axis contains only `primary_raw_decoded`. The loss
        axis fixes the bridge architecture and changes only the loss. Shuffled
        controls are validation diagnostics for B and C and are not eligible
        as final inference variants.
        """
    ),
    code(
        r"""
        def build_run_grid():
            grid = []
            for path in BRIDGE_PATHS:
                architectures = (
                    ["mlp_512"]
                    if path in {"direct_baseline", "concatenated_text_semantic"}
                    else PRIMARY_BRIDGE_ARCHITECTURES
                )
                for architecture in architectures:
                    grid.append(
                        {
                            "path": path,
                            "architecture": architecture,
                            "loss": PRIMARY_LOSS,
                            "experiment_axis": "primary_architecture",
                            "shuffled_control": False,
                        }
                    )
            if RUN_SHUFFLED_CONTROLS:
                for path in (
                    "stage3_text_bridge",
                    "stage3_brain_bridge_oracle",
                ):
                    for architecture in PRIMARY_BRIDGE_ARCHITECTURES:
                        grid.append(
                            {
                                "path": path,
                                "architecture": architecture,
                                "loss": PRIMARY_LOSS,
                                "experiment_axis": "shuffled_control",
                                "shuffled_control": True,
                            }
                        )
            if RUN_SECONDARY_LOSS_VARIANTS:
                for path in LOSS_SENSITIVITY_PATHS:
                    for variant in SECONDARY_LOSS_VARIANTS:
                        grid.append(
                            {
                                "path": path,
                                "architecture": LOSS_SENSITIVITY_ARCHITECTURE,
                                "loss": SECONDARY_LOSS_CONFIGS[variant],
                                "experiment_axis": "loss_sensitivity",
                                "shuffled_control": False,
                            }
                        )
            ids = [run_id_for(item) for item in grid]
            if len(ids) != len(set(ids)):
                raise RuntimeError("Run grid contains duplicate identities")
            return grid

        RUN_GRID = build_run_grid()
        pd.DataFrame(
            [
                {
                    **{k: v for k, v in item.items() if k != "loss"},
                    "loss": item["loss"].variant,
                }
                for item in RUN_GRID
            ]
        )
        """
    ),
    code(
        r"""
        # Run selected branches. Resources are released after each branch.
        lookup = AtlasFreeTextEmbeddingLookup.published()
        all_results = {}
        all_provenance = {}
        architecture_configs = {}

        root_config = {
            "experiment": EXPERIMENT_NAME,
            "repo_commit": resolved_commit,
            "branches_to_run": BRANCHES_TO_RUN,
            "all_valid_branches": ALL_BRANCHES,
            "mixed_ae_only": True,
            "resource_profile": {
                "batch_size": BATCH_SIZE,
                "eval_batch_size": EVAL_BATCH_SIZE,
                "num_workers": NUM_WORKERS,
                "prefetch_factor": PREFETCH_FACTOR,
            },
            "run_grid": [
                {
                    **{k: v for k, v in item.items() if k != "loss"},
                    "loss": item["loss"].effective_dict(),
                }
                for item in RUN_GRID
            ],
            "environment": ENVIRONMENT,
            "test_used_for_selection": False,
        }
        atomic_write_json(OUTPUT_ROOT / "experiment_config.json", root_config)

        for branch in BRANCHES_TO_RUN:
            print(f"\n===== {branch} =====")
            branch_dir = OUTPUT_ROOT / branch
            branch_dir.mkdir(parents=True, exist_ok=True)
            (
                spec,
                ae_path,
                stage3_path,
                autoencoder,
                stage3_model,
                provider,
            ) = load_branch_resources(branch)
            provenance = collect_branch_provenance(
                branch,
                spec,
                ae_path,
                stage3_path,
                autoencoder,
                stage3_model,
                provider,
                lookup,
            )
            all_provenance[branch] = provenance
            atomic_write_json(branch_dir / "provenance.json", provenance)

            caches, cache_paths = {}, {}
            # Test features are deliberately not materialized until after
            # validation-only finalist selection.
            for split in ("train", "val"):
                caches[split], cache_paths[split] = build_or_load_frozen_cache(
                    branch_dir,
                    split,
                    getattr(provider, split),
                    lookup,
                    autoencoder,
                    stage3_model,
                    provenance,
                )
            cache_stats = {
                "train_target_latent_mean": caches["train"]["target_latent"].mean(
                    dim=0
                ),
                "train_target_latent_std": caches["train"]["target_latent"].std(
                    dim=0, unbiased=False
                ),
            }
            atomic_torch_save(branch_dir / "train_latent_statistics.pt", cache_stats)

            branch_results = []
            for run_spec in RUN_GRID:
                print(f"--- {run_id_for(run_spec)}")
                result = train_one_run(
                    branch_dir,
                    run_spec,
                    provenance,
                    autoencoder,
                    stage3_model,
                    provider,
                    lookup,
                    caches,
                    cache_paths,
                )
                branch_results.append(result)
                architecture_configs[result["run_id"]] = json.loads(
                    (result["run_dir"] / "architecture.json").read_text()
                )
            all_results[branch] = {
                "results": branch_results,
                "resources": {
                    "autoencoder": autoencoder,
                    "stage3_model": stage3_model,
                    "provider": provider,
                    "caches": caches,
                    "cache_paths": cache_paths,
                },
            }
            autoencoder.to("cpu")
            stage3_model.to("cpu")
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
        atomic_write_json(OUTPUT_ROOT / "provenance.json", all_provenance)
        atomic_write_json(
            OUTPUT_ROOT / "bridge_architecture_configs.json", architecture_configs
        )
        """
    ),
    markdown(
        r"""
        ## Validation selection and one-time finalist test evaluation

        One primary-loss finalist per main path and branch is selected by
        validation top-5 Dice (ties: spatial correlation, then latent explained
        variance). Only that finalist's validation-selected top-5 checkpoint is
        evaluated on test. Loss-sensitivity runs and shuffled controls remain
        validation-only.
        """
    ),
    code(
        r"""
        validation_rows = []
        finalist_results = {}
        for branch, bundle in all_results.items():
            for result in bundle["results"]:
                summary = result["validation"]["summary"]
                validation_rows.append(
                    {
                        "branch": branch,
                        "run_id": result["run_id"],
                        "path": result["spec"]["path"],
                        "architecture": result["spec"]["architecture"],
                        "loss": result["spec"]["loss"].variant,
                        "experiment_axis": result["spec"]["experiment_axis"],
                        "shuffled_control": result["spec"]["shuffled_control"],
                        **{f"val_{key}": value for key, value in summary.items()},
                    }
                )
        validation_frame = pd.DataFrame(validation_rows)
        atomic_write_csv(
            OUTPUT_ROOT / "branch_metrics.csv",
            validation_frame.to_dict("records"),
        )

        eligible = validation_frame[
            (validation_frame["experiment_axis"] == "primary_architecture")
            & (~validation_frame["shuffled_control"])
            & (validation_frame["loss"] == "primary_raw_decoded")
        ].copy()
        for (branch, path), group in eligible.groupby(["branch", "path"]):
            selected = group.sort_values(
                [
                    "val_top5_dice",
                    "val_spatial_corr",
                    "val_global_explained_variance",
                ],
                ascending=False,
            ).iloc[0]
            finalist_results[(branch, path)] = next(
                result
                for result in all_results[branch]["results"]
                if result["run_id"] == selected["run_id"]
            )

        selected_ids = {
            (branch, path, result["run_id"])
            for (branch, path), result in finalist_results.items()
        }
        selected_validation = eligible[
            eligible.apply(
                lambda row: (row["branch"], row["path"], row["run_id"])
                in selected_ids,
                axis=1,
            )
        ].copy()

        test_rows, generated_examples = [], {}
        if RUN_FINAL_TEST:
            for branch in BRANCHES_TO_RUN:
                resources = all_results[branch]["resources"]
                branch_dir = OUTPUT_ROOT / branch
                # This is the first point at which frozen test features are
                # materialized: every finalist identity is already fixed.
                test_cache, test_cache_path = build_or_load_frozen_cache(
                    branch_dir,
                    "test",
                    resources["provider"].test,
                    lookup,
                    resources["autoencoder"],
                    resources["stage3_model"],
                    all_provenance[branch],
                )
                resources["caches"]["test"] = test_cache
                resources["cache_paths"]["test"] = test_cache_path
                for path in BRIDGE_PATHS:
                    result = finalist_results[(branch, path)]
                    result["manager"].load(
                        FINAL_CHECKPOINT_ROLE,
                        model=result["model"],
                        map_location="cpu",
                    )
                    test_result = evaluate_run(
                        result["model"],
                        path,
                        resources["provider"].test,
                        test_cache,
                        resources["caches"]["train"],
                        lookup,
                        resources["autoencoder"],
                        resources["stage3_model"],
                        latent_mean=result["latent_mean"],
                        latent_std=result["latent_std"],
                        shuffled_control=None,
                        split="test_finalist",
                    )
                    test_rows.append(
                        {
                            "branch": branch,
                            "path": path,
                            "run_id": result["run_id"],
                            "checkpoint_role": FINAL_CHECKPOINT_ROLE,
                            "test_cache_sha256": sha256_file(test_cache_path),
                            "test_used_for_selection": False,
                            **{f"test_{key}": value for key, value in test_result["summary"].items()},
                        }
                    )
                    generated_examples[(branch, path)] = test_result["examples"]
                    atomic_write_csv(
                        result["run_dir"] / "test_per_dimension_finalist.csv",
                        test_result["per_dimension"],
                    )
                    result["model"].to("cpu")
                resources["autoencoder"].to("cpu")
                resources["stage3_model"].to("cpu")
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

        if RUN_FINAL_TEST:
            test_frame = pd.DataFrame(test_rows)
            final_comparison = selected_validation.merge(
                test_frame,
                on=["branch", "path", "run_id"],
                how="inner",
            )
        else:
            final_comparison = selected_validation
        atomic_write_csv(
            OUTPUT_ROOT / "final_comparison.csv",
            final_comparison.to_dict("records"),
        )
        atomic_write_json(
            OUTPUT_ROOT / "final_comparison.json",
            final_comparison.to_dict("records"),
        )
        final_comparison
        """
    ),
    markdown("## Critical comparisons, controls, plots, and generated examples"),
    code(
        r"""
        semantic_rows, oracle_gap_rows = [], []
        for branch in BRANCHES_TO_RUN:
            branch_rows = validation_frame[
                validation_frame["branch"] == branch
            ]
            primary_selected = final_comparison[
                final_comparison["branch"] == branch
            ].set_index("path")
            text_row = primary_selected.loc["stage3_text_bridge"]
            brain_row = primary_selected.loc["stage3_brain_bridge_oracle"]
            semantic_rows.append(
                {
                    "branch": branch,
                    "matched_cosine": text_row[
                        "val_stage3_text_brain_matched_cosine"
                    ],
                    "shuffled_cosine": text_row[
                        "val_stage3_text_brain_shuffled_cosine"
                    ],
                    "matched_minus_shuffled": text_row[
                        "val_stage3_matched_minus_shuffled_cosine"
                    ],
                    "text_bridge_global_explained_variance": text_row[
                        "val_global_explained_variance"
                    ],
                    "brain_oracle_global_explained_variance": brain_row[
                        "val_global_explained_variance"
                    ],
                    "text_bridge_spatial_corr": text_row["val_spatial_corr"],
                    "brain_oracle_spatial_corr": brain_row["val_spatial_corr"],
                }
            )
            for metric in (
                "val_raw_latent_mse",
                "val_standardized_latent_mse",
                "val_latent_cosine",
                "val_latent_variance_ratio",
                "val_global_explained_variance",
                "val_spatial_corr",
                "val_top5_dice",
            ):
                oracle_gap_rows.append(
                    {
                        "branch": branch,
                        "metric": metric.removeprefix("val_"),
                        "text_bridge": float(text_row[metric]),
                        "brain_oracle": float(brain_row[metric]),
                        "brain_minus_text": float(brain_row[metric] - text_row[metric]),
                    }
                )
        semantic_frame = pd.DataFrame(semantic_rows)
        oracle_gap_frame = pd.DataFrame(oracle_gap_rows)
        atomic_write_csv(
            OUTPUT_ROOT / "text_versus_brain_semantic_comparison.csv",
            semantic_frame.to_dict("records"),
        )
        atomic_write_csv(
            OUTPUT_ROOT / "oracle_gap_metrics.csv",
            oracle_gap_frame.to_dict("records"),
        )

        controls = validation_frame[
            validation_frame["experiment_axis"].isin(
                ["primary_architecture", "shuffled_control"]
            )
            & validation_frame["path"].isin(
                ["stage3_text_bridge", "stage3_brain_bridge_oracle"]
            )
            & (validation_frame["architecture"] == PRIMARY_BRIDGE_ARCHITECTURES[0])
        ].copy()
        atomic_write_csv(
            OUTPUT_ROOT / "shuffled_control_metrics.csv",
            controls.to_dict("records"),
        )

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        plot_gap = oracle_gap_frame[
            oracle_gap_frame["metric"].isin(
                ["global_explained_variance", "spatial_corr", "top5_dice"]
            )
        ]
        for index, (branch, group) in enumerate(plot_gap.groupby("branch")):
            offset = (index - (len(BRANCHES_TO_RUN) - 1) / 2) * 0.12
            axes[0].bar(
                np.arange(len(group)) + offset,
                group["brain_minus_text"],
                width=0.12,
                label=branch,
            )
        axes[0].set_xticks(np.arange(3))
        axes[0].set_xticklabels(
            ["explained variance", "spatial corr", "top-5 Dice"],
            rotation=20,
        )
        axes[0].axhline(0, color="black", linewidth=0.8)
        axes[0].set_title("Brain-oracle minus text-bridge validation gap")
        axes[0].legend(fontsize=7)

        variance_plot = final_comparison.pivot(
            index="path", columns="branch", values="val_global_explained_variance"
        )
        variance_plot.plot(kind="bar", ax=axes[1])
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].set_title("Raw AE latent variance recovered")
        axes[1].set_ylabel("Global explained variance")
        axes[1].tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(OUTPUT_ROOT / "oracle_gap_plots.png", dpi=180)
        fig.savefig(OUTPUT_ROOT / "latent_variance_plots.png", dpi=180)
        plt.show()

        if generated_examples:
            example_branch = BRANCHES_TO_RUN[0]
            path_examples = {
                path: generated_examples[(example_branch, path)]
                for path in BRIDGE_PATHS
            }
            example_count = min(
                GENERATED_EXAMPLES,
                *(len(values) for values in path_examples.values()),
            )
            fig, axes = plt.subplots(
                example_count,
                len(BRIDGE_PATHS) + 1,
                figsize=(3 * (len(BRIDGE_PATHS) + 1), 3 * example_count),
                squeeze=False,
            )
            for row in range(example_count):
                reference = path_examples["direct_baseline"][row]
                target = reference["target"].squeeze().numpy()
                middle = target.shape[-1] // 2
                axes[row, 0].imshow(target[:, :, middle], cmap="magma")
                axes[row, 0].set_title(f"Target\n{reference['map_id']}")
                axes[row, 0].axis("off")
                for column, path in enumerate(BRIDGE_PATHS, start=1):
                    example = path_examples[path][row]
                    if example["map_id"] != reference["map_id"]:
                        raise RuntimeError("Generated example rows are not aligned")
                    prediction = example["prediction"].squeeze().numpy()
                    axes[row, column].imshow(
                        prediction[:, :, middle], cmap="magma"
                    )
                    axes[row, column].set_title(path)
                    axes[row, column].axis("off")
            fig.suptitle(f"Validation-selected test finalists: {example_branch}")
            fig.tight_layout()
            fig.savefig(OUTPUT_ROOT / "generated_examples.png", dpi=180)
            plt.show()
        """
    ),
    markdown("## Final interpretation and report"),
    code(
        r"""
        def path_works(row):
            return bool(
                row["val_global_explained_variance"]
                >= WORKS_MIN_GLOBAL_EXPLAINED_VARIANCE
                and row["val_spatial_corr"]
                >= WORKS_MIN_SPATIAL_CORRELATION
            )

        report_sections = [
            "# Stage 4 / Stage 3 semantic bridge report",
            "",
            f"- Repository commit: `{resolved_commit}`",
            f"- Run mode: `{'FULL_RUN' if FULL_RUN else 'FAST_RUN smoke test'}`",
            f"- Branches: `{', '.join(BRANCHES_TO_RUN)}`",
            "- Selection: validation top-5 Dice, tie-broken by validation spatial "
            "correlation and global explained variance.",
            "- Test split was not used for model or checkpoint selection.",
            "- Main conclusion uses only `primary_raw_decoded`; loss-sensitivity "
            "variants are reported separately.",
            "",
            "## Branch conclusions",
            "",
        ]
        classification_rows = []
        for branch in BRANCHES_TO_RUN:
            selected = final_comparison[
                final_comparison["branch"] == branch
            ].set_index("path")
            text_works = path_works(selected.loc["stage3_text_bridge"])
            brain_works = path_works(
                selected.loc["stage3_brain_bridge_oracle"]
            )
            if brain_works and not text_works:
                classification = (
                    "Stage 3 cross-modal alignment remains insufficient for precise "
                    "generation: the brain-semantic oracle works but the text bridge does not."
                )
            elif not brain_works and not text_works:
                classification = (
                    "Stage 3 semantic embeddings discard raw spatial information needed "
                    "by the Stage 1 decoder: neither semantic path works."
                )
            elif brain_works and text_works:
                classification = "Both semantic paths work; the current direct SPECTER2-to-AE projector is the main bottleneck."
            else:
                classification = (
                    "Text semantics pass the predeclared thresholds while the brain oracle "
                    "does not; this atypical pattern requires audit before a causal claim."
                )
            best_path = selected["val_top5_dice"].astype(float).idxmax()
            complementary = best_path == "concatenated_text_semantic"
            if complementary:
                classification += " Concatenated raw text plus semantic input is best, supporting complementary information."
            direct = selected.loc["direct_baseline"]
            stage3_useful = bool(
                text_works
                or selected.loc[
                    [
                        "shared_bridge_dual_supervision",
                        "concatenated_text_semantic",
                        "residual_direct_plus_semantic",
                    ],
                    "val_top5_dice",
                ].max()
                > direct["val_top5_dice"]
            )
            spatial_preserved = brain_works
            classification_rows.append(
                {
                    "branch": branch,
                    "text_semantic_works": text_works,
                    "brain_semantic_oracle_works": brain_works,
                    "stage3_useful_for_generation": stage3_useful,
                    "stage3_preserves_enough_spatial_information": spatial_preserved,
                    "best_primary_path_by_val_top5_dice": best_path,
                    "concatenated_is_best": complementary,
                    "classification": classification,
                }
            )
            report_sections.extend(
                [
                    f"### {branch}",
                    "",
                    classification,
                    "",
                    f"- Stage 3 useful for generation: **{stage3_useful}**.",
                    f"- Enough spatial information for raw-AE reconstruction: "
                    f"**{spatial_preserved}** (brain-oracle criterion).",
                    f"- Text bridge: explained variance "
                    f"{selected.loc['stage3_text_bridge', 'val_global_explained_variance']:.4f}, "
                    f"spatial correlation "
                    f"{selected.loc['stage3_text_bridge', 'val_spatial_corr']:.4f}.",
                    f"- Brain oracle: explained variance "
                    f"{selected.loc['stage3_brain_bridge_oracle', 'val_global_explained_variance']:.4f}, "
                    f"spatial correlation "
                    f"{selected.loc['stage3_brain_bridge_oracle', 'val_spatial_corr']:.4f}.",
                    f"- Direct baseline: explained variance "
                    f"{direct['val_global_explained_variance']:.4f}, spatial correlation "
                    f"{direct['val_spatial_corr']:.4f}.",
                    f"- Stage 3 matched vs shuffled cosine: "
                    f"{selected.loc['stage3_text_bridge', 'val_stage3_text_brain_matched_cosine']:.4f} "
                    f"vs "
                    f"{selected.loc['stage3_text_bridge', 'val_stage3_text_brain_shuffled_cosine']:.4f}.",
                    "",
                ]
            )

        report_sections.extend(
            [
                "## Scope and caveats",
                "",
                "- The oracle is diagnostic and cannot be used for text-only inference.",
                "- Stage 1 and Stage 3 were frozen; all bridge outputs remained raw and "
                "unnormalized before the frozen decoder.",
                "- Architecture conclusions exclude loss-sensitivity runs.",
                "- A FAST_RUN report is a pipeline smoke test, not scientific evidence. "
                "Use FULL_RUN and an immutable EXPECTED_COMMIT for archival results.",
                "",
                "## Saved artifacts",
                "",
                "- Exact Stage 1/Stage 3/cache/split provenance: `provenance.json`",
                "- Run histories and multi-objective checkpoints: branch `runs/` directories",
                "- Validation/test comparison: `final_comparison.csv` and `.json`",
                "- Text/brain oracle gap: `text_versus_brain_semantic_comparison.csv`",
                "- Controls: `shuffled_control_metrics.csv`",
                "- Plots: `oracle_gap_plots.png`, `latent_variance_plots.png`, "
                "`generated_examples.png`",
            ]
        )
        classification_frame = pd.DataFrame(classification_rows)
        atomic_write_csv(
            OUTPUT_ROOT / "interpretation_classification.csv",
            classification_frame.to_dict("records"),
        )
        report = "\n".join(report_sections) + "\n"
        report_path = OUTPUT_ROOT / "final_report.md"
        report_path.write_text(report, encoding="utf-8")
        print(report)
        print(f"Final report: {report_path}")
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "gpuType": "A100",
            "provenance": [],
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
