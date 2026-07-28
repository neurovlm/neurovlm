from __future__ import annotations

import json
import textwrap
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "stage4_joint_ae_projector_finetuning.ipynb"


def lines(value: str) -> list[str]:
    text = textwrap.dedent(value).strip("\n") + "\n"
    return text.splitlines(keepends=True)


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
        """
        # Stage 4 joint AE/projector fine-tuning

        This experiment tests whether the frozen Stage 1 decoder amplifies small,
        off-manifold Stage 4 latent errors. Every training pair follows two paths:

        1. **AE replay:** image → current encoder → current decoder → reconstruction.
        2. **Text generation:** normalized SPECTER2 → projector → raw AE latent →
           current decoder → generated image.

        An independently loaded, evaluation-only Stage 1 autoencoder is retained
        throughout. Validation re-evaluates its reconstruction ceiling, measures the
        adapted AE against that ceiling, and rejects unsafe checkpoints. The held-out
        test split is not touched until validation has selected checkpoints.

        The default configuration runs all seven requested variants, four calibrated
        noise scales plus a zero-noise replay control, a shuffled-pair control, and
        the 32-example tiny-overfit control. Outputs are written to a new timestamped
        Drive directory; released Stage 1 and Stage 4 checkpoints are read-only.
        """
    ),
    markdown(
        """
        ## 1. Colab, Drive, repository, and dependencies

        Set `REPO_REF` to a branch or tag. For an archival run, also set
        `NEUROVLM_PINNED_COMMIT` to the expected full commit SHA. A mismatch fails
        before any experiment artifacts are created.
        """
    ),
    code(
        """
        from pathlib import Path
        import os, subprocess, sys

        IN_COLAB = "google.colab" in sys.modules
        if IN_COLAB:
            from google.colab import drive
            drive.mount("/content/drive")

        REPO_URL = "https://github.com/neurovlm/neurovlm.git"
        REPO_REF = "neurovlm_experiments"  # configurable branch/tag
        NEUROVLM_PINNED_COMMIT = None  # set a full SHA for the archival run
        UPDATE_LOCAL_CHECKOUT = False
        REPO_DIR = Path("/content/neurovlm" if IN_COLAB else Path.cwd()).resolve()

        if not (REPO_DIR / ".git").is_dir():
            REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)
        if IN_COLAB or UPDATE_LOCAL_CHECKOUT:
            subprocess.run(
                ["git", "fetch", "--all", "--tags", "--prune"],
                cwd=REPO_DIR,
                check=True,
            )
            subprocess.run(["git", "checkout", REPO_REF], cwd=REPO_DIR, check=True)
            remote = f"refs/remotes/origin/{REPO_REF}"
            exists = subprocess.run(
                ["git", "show-ref", "--verify", "--quiet", remote],
                cwd=REPO_DIR,
            ).returncode == 0
            if exists:
                subprocess.run(
                    ["git", "merge", "--ff-only", f"origin/{REPO_REF}"],
                    cwd=REPO_DIR,
                    check=True,
                )

        RESOLVED_COMMIT = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_DIR, text=True
        ).strip()
        if (
            NEUROVLM_PINNED_COMMIT is not None
            and RESOLVED_COMMIT != NEUROVLM_PINNED_COMMIT
        ):
            raise RuntimeError(
                f"Expected {NEUROVLM_PINNED_COMMIT}, resolved {RESOLVED_COMMIT}"
            )
        print("Repository:", REPO_DIR)
        print("Configured ref:", REPO_REF)
        print("Resolved commit:", RESOLVED_COMMIT)
        """
    ),
    code(
        """
        subprocess.run(
            [
                sys.executable, "-m", "pip", "install", "-q", "-U",
                "pip", "setuptools", "wheel",
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable, "-m", "pip", "install", "-q", "-e",
                f"{REPO_DIR}[metrics,viz,notebook]",
            ],
            check=True,
        )
        src_path = str(REPO_DIR / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        os.environ["PYTHONPATH"] = (
            src_path + os.pathsep + os.environ.get("PYTHONPATH", "")
        )
        os.chdir(REPO_DIR)
        """
    ),
    markdown(
        """
        ## 2. Effective experiment configuration

        Conservative learning rates are separated by module role. The default
        alignment target is the immutable original encoder latent, which prevents
        encoder-head adaptation from silently redefining the Stage 4 target. Set
        `LATENT_ALIGNMENT_MODE="standardized"` to standardize the alignment loss
        using train-only Stage 1 latent statistics.

        The explicit objective is

        `generation_latent × latent_alignment`
        `+ generation_image × generation_image_loss`
        `+ replay × AE_reconstruction_replay`
        `+ distill × decoder_output_distillation`
        `+ parameter × parameter_distance_regularization`.

        Every retention decision records `satisfies_1_percent`,
        `satisfies_2_percent`, and `satisfies_5_percent` in addition to the
        configurable hard safety result.
        """
    ),
    code(
        """
        BRANCHES_TO_RUN = ["mixed_to_pubmed"]
        VARIANTS_TO_RUN = [
            "projector_only_baseline",
            "projector_plus_decoder_output",
            "projector_plus_last_decoder_block",
            "projector_plus_decoder_seed",
            "projector_plus_seed_and_last_block",
            "projector_plus_encoder_head_and_decoder",
            "latent_noise_decoder_adaptation",
        ]
        LATENT_NOISE_SCALES = [0.25, 0.5, 1.0, 2.0]
        RUN_ZERO_NOISE_REPLAY_CONTROL = True
        RUN_SHUFFLED_TEXT_PAIR_CONTROL = True
        RUN_TINY_OVERFIT_CONTROL = True
        RUN_FULL_TRAINING = True

        SEED = 42
        PROJECTOR_SEED = 42
        EPOCHS = 50
        BATCH_SIZE = 32
        EVAL_BATCH_SIZE = 32
        NUM_WORKERS = 2
        MAX_TRAIN_BATCHES = None
        MAX_EVAL_BATCHES = None
        DATA_LIMIT = None
        VALIDATE_EVERY_EPOCHS = 1
        EARLY_STOPPING_PATIENCE = 8
        GRADIENT_CLIP = 1.0

        PROJECTOR_LEARNING_RATE = 1e-4
        DECODER_LEARNING_RATE = 1e-5
        ENCODER_HEAD_LEARNING_RATE = 5e-6
        WEIGHT_DECAY = 1e-4

        LOSS_WEIGHTS = {
            "generation_latent": 1.0,
            "generation_image": 1.0,
            "replay": 4.0,
            "distill": 2.0,
            "parameter": 0.0,
            "generation_foreground": 1.0,
            "replay_foreground": 0.0,
        }
        LATENT_ALIGNMENT_MODE = "raw"  # raw | standardized
        ALIGNMENT_TARGET = "original"  # original | adapted

        AMP = True
        AMP_DTYPE = "auto"  # BF16 on A100/H100, FP16 otherwise
        MAXIMUM_AE_TOP5_DEGRADATION_PERCENT = 5.0
        SAFETY_ACTION = "stop"  # stop | reject_checkpoint

        PROJECTOR_INITIALIZATION = "released_stage4"  # released_stage4 | random
        NOISE_DECODER_COMPONENTS = [
            "decoder_seed", "last_decoder_block", "decoder_output"
        ]
        SEMANTIC_MAX_EXAMPLES = 1024
        SEMANTIC_NEIGHBORS = 10
        EXAMPLES_PER_RUN = 6

        TINY_OVERFIT_N = 32
        TINY_OVERFIT_STEPS = 500
        TINY_OVERFIT_LEARNING_RATE = 1e-3
        FAST_DEV_RUN = False

        DRIVE_OUTPUT_BASE = Path(
            "/content/drive/MyDrive/neurovlm/stage4_joint_ae_projector_finetuning"
            if IN_COLAB
            else REPO_DIR / "runs" / "stage4_joint_ae_projector_finetuning"
        )
        RESUME_EXPERIMENT_DIR = None
        AUTO_RESUME_ACTIVE = True

        if FAST_DEV_RUN:
            EPOCHS = 2
            NUM_WORKERS = 0
            DATA_LIMIT = 128
            MAX_TRAIN_BATCHES = 2
            MAX_EVAL_BATCHES = 2
            SEMANTIC_MAX_EXAMPLES = 64
            TINY_OVERFIT_STEPS = 20
            EARLY_STOPPING_PATIENCE = None

        assert VALIDATE_EVERY_EPOCHS == 1
        assert SAFETY_ACTION in {"stop", "reject_checkpoint"}
        assert LATENT_ALIGNMENT_MODE in {"raw", "standardized"}
        assert ALIGNMENT_TARGET in {"original", "adapted"}
        """
    ),
    markdown("## 3. Imports, deterministic seeds, precision, and environment report"),
    code(
        """
        import copy, csv, importlib.metadata, json, math, platform, random, tempfile, textwrap, time
        from dataclasses import asdict
        from datetime import datetime, timezone

        import numpy as np
        import pandas as pd
        import torch
        import torch.nn.functional as F
        from matplotlib import pyplot as plt
        from torch import nn
        from torch.utils.data import DataLoader, Subset

        from neurovlm import retrieval_resources as rr
        from neurovlm.atlas_free_dataset import AtlasFreeCNNDataProvider
        from neurovlm.atlas_free_text import (
            AtlasFreeContrastiveCollator,
            AtlasFreeTextEmbeddingLookup,
        )
        from neurovlm.cnn import GenerativeTextToAELatent, autoencoder_from_payload
        from neurovlm.evaluation.spatial import reconstruction_metrics
        from neurovlm.evaluation.text_to_brain_audit import (
            audit_pairings,
            audit_text_preprocessing,
            autoencoder_identity,
        )
        from neurovlm.experiments.stage4_joint_finetuning import (
            JOINT_FINETUNING_VARIANTS,
            JointLossWeights,
            ae_retention_decision,
            assert_frozen_parameters_unchanged,
            assert_original_untouched,
            checkpoint_binding,
            compute_joint_loss,
            configure_trainable_variant,
            fit_latent_standardization,
            latent_metrics,
            optimizer_group_settings,
            parameter_snapshot,
            trainable_parameter_manifest,
            untouched_autoencoder,
            validate_checkpoint_binding,
        )
        from neurovlm.experiments.stage4_latent_ablation import (
            encode_stage1_latents,
            resolve_amp_dtype,
            split_fingerprint,
            text_cache_identity,
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
        from neurovlm.training.text_to_brain import (
            _autoencoder_state_provenance,
            _text_cache_provenance,
            _validate_recorded_autoencoder_state,
            _validate_recorded_text_cache,
        )

        def seed_everything(seed):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        seed_everything(SEED)
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        MIXED_PRECISION_DTYPE = resolve_amp_dtype(DEVICE, AMP_DTYPE)
        AUTOCAST_ENABLED = (
            AMP and DEVICE.type == "cuda" and MIXED_PRECISION_DTYPE != torch.float32
        )
        PACKAGE_NAMES = [
            "neurovlm", "torch", "numpy", "pandas", "matplotlib",
            "nilearn", "nibabel", "huggingface-hub", "transformers",
        ]
        ENVIRONMENT = {
            **environment_provenance(PACKAGE_NAMES),
            "python_full": sys.version,
            "platform_full": platform.platform(),
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
            "gpu_capability": (
                torch.cuda.get_device_capability(0)
                if torch.cuda.is_available()
                else None
            ),
            "bf16_supported": (
                torch.cuda.is_bf16_supported()
                if torch.cuda.is_available()
                else False
            ),
            "amp_dtype": str(MIXED_PRECISION_DTYPE),
            "git": git_provenance(REPO_DIR),
            "configured_ref": REPO_REF,
            "resolved_commit": RESOLVED_COMMIT,
        }
        print(json.dumps(ENVIRONMENT, indent=2, default=str))
        """
    ),
    markdown("## 4. Branch resources, exact provenance, and data loaders"),
    code(
        """
        BRANCH_SPECS = {
            "mixed_to_pubmed": {
                "domain": "pubmed", "stage1": "1A", "ae_variant": "mixed"
            },
            "pubmed": {
                "domain": "pubmed", "stage1": "1B", "ae_variant": "pubmed"
            },
            "mixed_to_nilearn": {
                "domain": "nilearn", "stage1": "1A", "ae_variant": "mixed"
            },
            "nilearn": {
                "domain": "nilearn", "stage1": "1B", "ae_variant": "nilearn"
            },
            "mixed_to_neurovault": {
                "domain": "neurovault", "stage1": "1A", "ae_variant": "mixed"
            },
            "neurovault": {
                "domain": "neurovault", "stage1": "1B", "ae_variant": "neurovault"
            },
        }
        unknown_branches = sorted(set(BRANCHES_TO_RUN) - set(BRANCH_SPECS))
        unknown_variants = sorted(
            set(VARIANTS_TO_RUN) - set(JOINT_FINETUNING_VARIANTS)
        )
        if unknown_branches:
            raise ValueError(f"Unknown branches: {unknown_branches}")
        if unknown_variants:
            raise ValueError(f"Unknown variants: {unknown_variants}")

        def utc_stamp():
            return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        def resolve_experiment_root():
            DRIVE_OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
            pointer = DRIVE_OUTPUT_BASE / "ACTIVE_EXPERIMENT.json"
            if RESUME_EXPERIMENT_DIR is not None:
                root = Path(RESUME_EXPERIMENT_DIR)
            elif AUTO_RESUME_ACTIVE and pointer.exists():
                active = json.loads(pointer.read_text())
                candidate = Path(active["path"])
                unfinished = active.get("state") != "completed" and candidate.exists()
                root = candidate if unfinished else DRIVE_OUTPUT_BASE / utc_stamp()
            else:
                root = DRIVE_OUTPUT_BASE / utc_stamp()
            root.mkdir(parents=True, exist_ok=True)
            atomic_write_json(
                pointer,
                {"path": str(root), "state": "running", "updated_at": utc_stamp()},
            )
            return root, pointer

        def make_loader(dataset, lookup, *, batch_size, shuffle, seed):
            rows = getattr(dataset, "rows", None)
            if rows is not None:
                lookup.validate_dataset(rows)
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=NUM_WORKERS,
                collate_fn=AtlasFreeContrastiveCollator(lookup, (36, 45, 38)),
                pin_memory=DEVICE.type == "cuda",
                persistent_workers=NUM_WORKERS > 0,
                generator=torch.Generator().manual_seed(seed),
            )

        def load_branch_resources(branch):
            spec = {"branch": branch, **BRANCH_SPECS[branch]}
            ae_filename = rr.CNN_AUTOENCODER_FILENAMES[spec["ae_variant"]]
            ae_path = Path(
                rr._download_from_hf(
                    rr.ATLAS_FREE_CNN_MODEL_REPO_ID,
                    ae_filename,
                    repo_type="model",
                )
            )
            payload = torch.load(ae_path, map_location="cpu", weights_only=True)
            original_ae = untouched_autoencoder(autoencoder_from_payload(payload))
            starting_ae = untouched_autoencoder(autoencoder_from_payload(payload))
            if sha256_state_dict(original_ae) != sha256_state_dict(starting_ae):
                raise RuntimeError("Independent Stage 1 loads do not match")
            if PROJECTOR_INITIALIZATION == "released_stage4":
                released = rr._load_cnn_text_to_brain(branch)
                starting_projector = copy.deepcopy(released.text_projection).cpu()
                del released
            elif PROJECTOR_INITIALIZATION == "random":
                seed_everything(PROJECTOR_SEED)
                starting_projector = GenerativeTextToAELatent(768, 512, 384)
            else:
                raise ValueError(
                    f"Unknown projector initialization {PROJECTOR_INITIALIZATION!r}"
                )
            provider = AtlasFreeCNNDataProvider(
                domain=spec["domain"], limit=DATA_LIMIT
            )
            semantic_model = rr._load_cnn_contrastive(branch).eval()
            for parameter in semantic_model.parameters():
                parameter.requires_grad_(False)
            return (
                spec, ae_path, original_ae, starting_ae,
                starting_projector, provider, semantic_model,
            )

        def build_branch_provenance(
            spec, ae_path, original_ae, starting_ae,
            starting_projector, provider, lookup, audit_dir,
        ):
            ae_source = _autoencoder_state_provenance(
                {
                    "kind": "released_read_only",
                    "path": str(ae_path.resolve()),
                    "file_sha256": sha256_file(ae_path),
                    "branch": spec["branch"],
                    "domain": spec["domain"],
                    "stage1": spec["stage1"],
                    "loader_variant": spec["ae_variant"],
                },
                original_ae,
            )
            _validate_recorded_autoencoder_state(ae_source, original_ae)
            cache = _text_cache_provenance(lookup)
            _validate_recorded_text_cache(cache, _text_cache_provenance(lookup))
            text_audit = audit_text_preprocessing(lookup)
            if not text_audit["passed"]:
                raise RuntimeError(f"Text-cache audit failed: {text_audit}")
            pairings = {}
            for split in ("train", "val", "test"):
                pairings[split] = audit_pairings(
                    getattr(provider, split),
                    lookup,
                    minimum=min(100, len(getattr(provider, split))),
                    output_dir=audit_dir,
                )
                if not pairings[split]["passed"]:
                    raise RuntimeError(f"{split} image/text pairing audit failed")
            return {
                "environment": ENVIRONMENT,
                "git_commit": RESOLVED_COMMIT,
                "branch": spec,
                "original_ae_identity": {
                    **ae_source,
                    **autoencoder_identity(
                        original_ae,
                        checkpoint=ae_path,
                        domain=spec["domain"],
                        branch=spec["branch"],
                    ),
                },
                "starting_ae_state_identity": sha256_state_dict(starting_ae),
                "starting_projector_state_identity": sha256_state_dict(
                    starting_projector
                ),
                "text_cache_identity": {
                    **cache,
                    **text_cache_identity(lookup),
                },
                "split_fingerprints": {
                    split: split_fingerprint(getattr(provider, split))
                    for split in ("train", "val", "test")
                },
                "text_preprocessing_audit": text_audit,
                "pairing_audits": pairings,
                "latent_convention": "raw_384d_stage1_ae_latent",
                "test_used_for_selection": False,
            }

        EXPERIMENT_ROOT, ACTIVE_POINTER = resolve_experiment_root()
        """
    ),
    markdown(
        """
        ## 5. Validation metrics and causal diagnostics

        Each validation pass evaluates generation, the untouched AE ceiling, the
        current image→encoder→decoder replay path, original true latents through the
        adapted decoder, and predicted latents through the original decoder. The
        latter two separate decoder adaptation from encoder target-space movement.
        """
    ),
    code(
        """
        def weighted_update(totals, values, n, prefix=""):
            for name, value in values.items():
                key = f"{prefix}{name}"
                totals[key] = totals.get(key, 0.0) + float(value) * n

        def semantic_auc(brain, text, raw_text, ids):
            n = sum(len(value) for value in brain)
            if n < 2:
                return float("nan")
            neighbors = max(0, min(SEMANTIC_NEIGHBORS, n - 2))
            metrics, _ = evaluate_semantic_neighbor_retrieval(
                torch.cat(brain),
                torch.cat(text),
                ids,
                neighbor_text_embeddings=torch.cat(raw_text),
                n_neighbors=neighbors,
            )
            return float(metrics["semantic_normalized_k_recall_curve_auc"])

        @torch.no_grad()
        def evaluate_joint(
            projector, adapted_ae, original_ae, semantic_model,
            dataset, lookup, *, split, max_batches=None,
        ):
            projector.eval()
            adapted_ae.eval()
            original_ae.eval()
            semantic_model.eval()
            totals, n = {}, 0
            target_latents, predicted_latents, adapted_latents = [], [], []
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
                if max_batches is not None and batch_index >= max_batches:
                    break
                target = batch["volume"].to(DEVICE, non_blocking=True)
                text = batch["text_embedding"].to(DEVICE, non_blocking=True)
                original_latent = original_ae.encoder(target)
                adapted_latent = adapted_ae.encoder(target)
                predicted_latent = projector(text)

                generated = adapted_ae.decoder(predicted_latent)
                generated_original_decoder = original_ae.decoder(predicted_latent)
                original_reconstruction = original_ae.decoder(original_latent)
                adapted_reconstruction = adapted_ae.decoder(adapted_latent)
                adapted_decoder_true_latent = adapted_ae.decoder(original_latent)

                batch_n = len(target)
                weighted_update(
                    totals,
                    reconstruction_metrics(generated, target),
                    batch_n,
                    "generation_",
                )
                weighted_update(
                    totals,
                    reconstruction_metrics(
                        generated_original_decoder, target
                    ),
                    batch_n,
                    "generation_original_decoder_",
                )
                weighted_update(
                    totals,
                    reconstruction_metrics(original_reconstruction, target),
                    batch_n,
                    "original_ae_",
                )
                weighted_update(
                    totals,
                    reconstruction_metrics(adapted_reconstruction, target),
                    batch_n,
                    "adapted_ae_",
                )
                weighted_update(
                    totals,
                    reconstruction_metrics(adapted_decoder_true_latent, target),
                    batch_n,
                    "adapted_decoder_true_latent_",
                )
                totals["adapted_ae_difference_from_original_decoder_output"] = (
                    totals.get(
                        "adapted_ae_difference_from_original_decoder_output", 0.0
                    )
                    + float(
                        F.mse_loss(
                            adapted_reconstruction.float(),
                            original_reconstruction.float(),
                        )
                    )
                    * batch_n
                )
                totals["adapted_decoder_true_latent_difference_from_original"] = (
                    totals.get(
                        "adapted_decoder_true_latent_difference_from_original", 0.0
                    )
                    + float(
                        F.mse_loss(
                            adapted_decoder_true_latent.float(),
                            original_reconstruction.float(),
                        )
                    )
                    * batch_n
                )
                totals["adapted_vs_original_latent_mse"] = (
                    totals.get("adapted_vs_original_latent_mse", 0.0)
                    + float(F.mse_loss(adapted_latent.float(), original_latent.float()))
                    * batch_n
                )
                target_latents.append(original_latent.float().cpu())
                predicted_latents.append(predicted_latent.float().cpu())
                adapted_latents.append(adapted_latent.float().cpu())
                n += batch_n

                semantic_seen = sum(len(value) for value in semantic_brain)
                take = min(batch_n, max(0, SEMANTIC_MAX_EXAMPLES - semantic_seen))
                if take:
                    safe_generated = torch.nan_to_num(
                        generated[:take].float(), nan=0.0, posinf=1.0, neginf=0.0
                    ).clamp(0, 1)
                    semantic_brain.append(
                        semantic_model.encode_brain(safe_generated).float().cpu()
                    )
                    semantic_text.append(
                        semantic_model.encode_text(text[:take]).float().cpu()
                    )
                    semantic_raw.append(text[:take].float().cpu())
                    semantic_ids.extend(
                        str(value) for value in batch["map_id"][:take]
                    )

                remaining = EXAMPLES_PER_RUN - len(examples)
                for index in range(min(remaining, batch_n)):
                    examples.append(
                        {
                            "map_id": str(batch["map_id"][index]),
                            "text_id": str(batch["text_id"][index]),
                            "target": target[index].float().cpu(),
                            "prediction": generated[index].float().cpu(),
                            "original_reconstruction": (
                                original_reconstruction[index].float().cpu()
                            ),
                            "adapted_reconstruction": (
                                adapted_reconstruction[index].float().cpu()
                            ),
                        }
                    )
            if not n:
                raise RuntimeError(f"{split} evaluation produced no examples")
            summary = {name: value / n for name, value in totals.items()}
            summary.update(
                latent_metrics(
                    torch.cat(target_latents), torch.cat(predicted_latents)
                )
            )
            adapted_target_metrics = latent_metrics(
                torch.cat(adapted_latents), torch.cat(predicted_latents)
            )
            summary.update(
                {
                    f"adapted_target_{name}": value
                    for name, value in adapted_target_metrics.items()
                }
            )
            summary["decoded_mse"] = summary["generation_mse"]
            summary["foreground_mse"] = summary["generation_foreground_mse"]
            summary["spatial_correlation"] = summary["generation_spatial_corr"]
            summary["top5_dice"] = summary["generation_top5_dice"]
            summary["semantic_normalized_recall_auc"] = semantic_auc(
                semantic_brain, semantic_text, semantic_raw, semantic_ids
            )
            original_metrics = {
                "top5_dice": summary["original_ae_top5_dice"],
                "spatial_corr": summary["original_ae_spatial_corr"],
                "mse": summary["original_ae_mse"],
            }
            adapted_metrics = {
                "top5_dice": summary["adapted_ae_top5_dice"],
                "spatial_corr": summary["adapted_ae_spatial_corr"],
                "mse": summary["adapted_ae_mse"],
            }
            safety = ae_retention_decision(
                original_metrics,
                adapted_metrics,
                maximum_top5_dice_degradation_percent=(
                    MAXIMUM_AE_TOP5_DEGRADATION_PERCENT
                ),
            )
            return {
                "split": split,
                "n": n,
                "summary": summary,
                "safety": safety,
                "examples": examples,
            }
        """
    ),
    markdown(
        """
        ## 6. Residual calibration, checkpoints, and optimization diagnostics

        The observed residual standard deviation is estimated on validation
        target-versus-starting-projector raw latents. It is used only as a noise
        magnitude—not for selection. Every checkpoint embeds all required static
        identities plus current projector/adapted-AE identities:
        `original_ae_identity`, `starting_ae_state_identity`,
        `current_trainable_module_identity`, `text_cache_identity`,
        `split_fingerprints`, `exact_unfrozen_parameter_names`, `loss_weights`,
        and `optimizer_group_settings`.
        """
    ),
    code(
        """
        @torch.no_grad()
        def estimate_projector_residuals(
            projector, original_ae, dataset, lookup, max_batches=None,
        ):
            projector.eval()
            original_ae.eval()
            residuals = []
            loader = make_loader(
                dataset,
                lookup,
                batch_size=EVAL_BATCH_SIZE,
                shuffle=False,
                seed=SEED,
            )
            for batch_index, batch in enumerate(loader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                target = batch["volume"].to(DEVICE)
                text = batch["text_embedding"].to(DEVICE)
                residuals.append(
                    (
                        projector(text) - original_ae.encoder(target)
                    ).float().cpu()
                )
            values = torch.cat(residuals)
            return {
                "residual_std": values.std(dim=0, unbiased=False).clamp_min(1e-8),
                "global_residual_std": float(values.std(unbiased=False)),
                "residual_mse": float(values.square().mean()),
                "n": len(values),
            }

        def group_norms(groups, field):
            output = {}
            for group_name, values in groups.items():
                terms = []
                for _, parameter in values:
                    tensor = getattr(parameter, field)
                    if tensor is not None:
                        terms.append(tensor.detach().float().square().sum())
                output[group_name] = (
                    float(torch.stack(terms).sum().sqrt()) if terms else 0.0
                )
            return output

        def group_update_norms(groups, before):
            output = {}
            for group_name, values in groups.items():
                terms = [
                    (
                        parameter.detach().float()
                        - before[name].to(parameter.device).float()
                    ).square().sum()
                    for name, parameter in values
                ]
                output[group_name] = (
                    float(torch.stack(terms).sum().sqrt()) if terms else 0.0
                )
            return output

        def group_drift_norms(groups, initial):
            output = {}
            for group_name, values in groups.items():
                terms = [
                    (
                        parameter.detach().float()
                        - initial[name].to(parameter.device).float()
                    ).square().sum()
                    for name, parameter in values
                ]
                output[group_name] = (
                    float(torch.stack(terms).sum().sqrt()) if terms else 0.0
                )
            return output

        def read_csv_rows(path):
            path = Path(path)
            if not path.exists():
                return []
            with path.open(newline="", encoding="utf-8") as stream:
                return list(csv.DictReader(stream))

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

        def build_checkpoint_payload(
            role, epoch, projector, adapted_ae, starting_ae,
            optimizer, provenance, manifest, optimizer_settings,
            metrics, safety, run_config,
        ):
            binding = checkpoint_binding(
                original_ae_identity=provenance["original_ae_identity"],
                starting_ae=starting_ae,
                adapted_ae=adapted_ae,
                projector=projector,
                text_cache_identity=provenance["text_cache_identity"],
                split_fingerprints=provenance["split_fingerprints"],
                unfrozen_parameter_names=manifest["unfrozen_parameter_names"],
                loss_weights=LOSS_WEIGHTS,
                optimizer_groups=optimizer_settings,
            )
            return {
                "format_version": 1,
                "role": role,
                "epoch": int(epoch),
                "projector_state_dict": projector.state_dict(),
                "adapted_ae_state_dict": adapted_ae.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": dict(metrics),
                "safety_rule_decision": dict(safety),
                "run_config": dict(run_config),
                "run_config_sha256": sha256_value(run_config),
                "binding": binding,
                **binding,
            }

        def save_checkpoint(
            run_dir, role, epoch, projector, adapted_ae, starting_ae,
            optimizer, provenance, manifest, optimizer_settings,
            metrics, safety, run_config,
        ):
            filename = {
                "best_generation_top5": "best_generation_top5_safe.pt",
                "best_semantic_auc": "best_semantic_auc_safe.pt",
                "best_ae_preserving": "best_ae_preserving.pt",
                "last": "last.pt",
            }[role]
            path = Path(run_dir) / "checkpoints" / filename
            payload = build_checkpoint_payload(
                role, epoch, projector, adapted_ae, starting_ae,
                optimizer, provenance, manifest, optimizer_settings,
                metrics, safety, run_config,
            )
            atomic_torch_save(path, payload)
            return path, payload

        def load_checkpoint(
            path, projector, adapted_ae, optimizer, expected_binding, run_config,
        ):
            payload = torch.load(path, map_location=DEVICE, weights_only=True)
            validate_checkpoint_binding(payload["binding"], expected_binding)
            if payload["run_config_sha256"] != sha256_value(run_config):
                raise ValueError("Resume run configuration mismatch")
            projector.load_state_dict(payload["projector_state_dict"], strict=True)
            adapted_ae.load_state_dict(payload["adapted_ae_state_dict"], strict=True)
            optimizer.load_state_dict(payload["optimizer_state_dict"])
            current = payload["current_trainable_module_identity"]
            if sha256_state_dict(projector) != current["projector_state_sha256"]:
                raise ValueError("Resumed projector state identity mismatch")
            if sha256_state_dict(adapted_ae) != current["adapted_ae_state_sha256"]:
                raise ValueError("Resumed adapted AE state identity mismatch")
            return payload
        """
    ),
    markdown("## 7. One epoch and one resume-safe experimental run"),
    code(
        """
        def train_one_epoch(
            projector, adapted_ae, original_ae, dataset, lookup,
            optimizer, groups, initial_snapshot, initial_ae_state,
            original_identity, weights, standardization, residual_std,
            *, epoch, replay_noise_scale, shuffled_text,
        ):
            # Epoch-local RNG makes interrupted/resumed execution identical.
            seed_everything(SEED + epoch)
            projector.train()
            adapted_ae.eval()
            original_ae.eval()
            scaler = torch.amp.GradScaler(
                "cuda",
                enabled=(
                    AUTOCAST_ENABLED
                    and MIXED_PRECISION_DTYPE == torch.float16
                ),
            )
            totals, n = {}, 0
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            loader = make_loader(
                dataset,
                lookup,
                batch_size=BATCH_SIZE,
                shuffle=True,
                seed=SEED + epoch,
            )
            for batch_index, batch in enumerate(loader):
                if (
                    MAX_TRAIN_BATCHES is not None
                    and batch_index >= MAX_TRAIN_BATCHES
                ):
                    break
                target = batch["volume"].to(DEVICE, non_blocking=True)
                text = batch["text_embedding"].to(DEVICE, non_blocking=True)
                if shuffled_text:
                    text = text.roll(shifts=1, dims=0)
                optimizer.zero_grad(set_to_none=True)
                before = {
                    name: parameter.detach().cpu().clone()
                    for values in groups.values()
                    for name, parameter in values
                }
                with torch.autocast(
                    device_type=DEVICE.type,
                    dtype=MIXED_PRECISION_DTYPE,
                    enabled=AUTOCAST_ENABLED,
                ):
                    result = compute_joint_loss(
                        projector,
                        adapted_ae,
                        original_ae,
                        text,
                        target,
                        weights=weights,
                        initial_parameters=initial_snapshot,
                        alignment_target=ALIGNMENT_TARGET,
                        latent_standardization=standardization,
                        replay_noise_std=residual_std,
                        replay_noise_scale=replay_noise_scale,
                    )
                if scaler.is_enabled():
                    scaler.scale(result.total).backward()
                    scaler.unscale_(optimizer)
                else:
                    result.total.backward()
                gradient_norms = group_norms(groups, "grad")
                trainable_parameters = [
                    parameter
                    for values in groups.values()
                    for _, parameter in values
                ]
                if GRADIENT_CLIP is not None:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_parameters, GRADIENT_CLIP
                    )
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                update_norms = group_update_norms(groups, before)
                drift_norms = group_drift_norms(groups, initial_snapshot)
                batch_n = len(target)
                values = {
                    "loss": float(result.total.detach()),
                    **{
                        f"raw_{name}": float(value.detach())
                        for name, value in result.components.items()
                    },
                    **{
                        f"weighted_{name}": float(value.detach())
                        for name, value in result.weighted.items()
                    },
                    **{
                        f"gradient_norm_{name}": value
                        for name, value in gradient_norms.items()
                    },
                    **{
                        f"update_norm_{name}": value
                        for name, value in update_norms.items()
                    },
                    **{
                        f"parameter_drift_{name}": value
                        for name, value in drift_norms.items()
                    },
                }
                weighted_update(totals, values, batch_n)
                n += batch_n
            if not n:
                raise RuntimeError("Training produced no batches")
            unfrozen_ae = [
                name
                for name, parameter in adapted_ae.named_parameters()
                if parameter.requires_grad
            ]
            assert_frozen_parameters_unchanged(
                adapted_ae, initial_ae_state, unfrozen_ae
            )
            assert_original_untouched(original_ae, original_identity)
            elapsed = time.perf_counter() - started
            summary = {name: value / n for name, value in totals.items()}
            summary.update(
                {
                    "epoch_time_seconds": elapsed,
                    "peak_gpu_memory_bytes": (
                        torch.cuda.max_memory_allocated()
                        if torch.cuda.is_available()
                        else 0
                    ),
                }
            )
            return summary, n

        def expanded_run_specs():
            specs = []
            for variant in VARIANTS_TO_RUN:
                if variant == "latent_noise_decoder_adaptation":
                    scales = list(LATENT_NOISE_SCALES)
                    if RUN_ZERO_NOISE_REPLAY_CONTROL:
                        scales = [0.0, *scales]
                    for scale in scales:
                        label = str(scale).replace(".", "p")
                        specs.append(
                            {
                                "variant": variant,
                                "run_name": f"{variant}_noise{label}x",
                                "replay_noise_scale": float(scale),
                                "shuffled_text": False,
                                "control": (
                                    "zero_noise_replay"
                                    if scale == 0
                                    else "latent_noise_adaptation"
                                ),
                            }
                        )
                else:
                    specs.append(
                        {
                            "variant": variant,
                            "run_name": variant,
                            "replay_noise_scale": 0.0,
                            "shuffled_text": False,
                            "control": None,
                        }
                    )
            if RUN_SHUFFLED_TEXT_PAIR_CONTROL:
                specs.append(
                    {
                        "variant": "projector_only_baseline",
                        "run_name": "shuffled_text_pair_control",
                        "replay_noise_scale": 0.0,
                        "shuffled_text": True,
                        "control": "shuffled_text_pair",
                    }
                )
            return specs

        def train_run(
            branch_dir, run_spec, original_ae_cpu, starting_ae_cpu,
            starting_projector_cpu, semantic_model_cpu, provider, lookup,
            provenance, standardization_cpu, residual_std_cpu,
        ):
            run_dir = Path(branch_dir) / run_spec["run_name"]
            run_dir.mkdir(parents=True, exist_ok=True)
            seed_everything(PROJECTOR_SEED)
            original_ae = untouched_autoencoder(original_ae_cpu).to(DEVICE)
            starting_ae = untouched_autoencoder(starting_ae_cpu)
            adapted_ae = copy.deepcopy(starting_ae_cpu).to(DEVICE)
            projector = copy.deepcopy(starting_projector_cpu).to(DEVICE)
            semantic_model = copy.deepcopy(semantic_model_cpu).to(DEVICE).eval()
            original_identity = sha256_state_dict(original_ae)
            initial_ae_state = {
                name: value.detach().cpu().clone()
                for name, value in adapted_ae.state_dict().items()
            }
            groups = configure_trainable_variant(
                adapted_ae,
                projector,
                run_spec["variant"],
                noise_decoder_components=NOISE_DECODER_COMPONENTS,
            )
            manifest = trainable_parameter_manifest(
                adapted_ae, projector, groups
            )
            optimizer_groups, optimizer_settings = optimizer_group_settings(
                groups,
                projector_learning_rate=PROJECTOR_LEARNING_RATE,
                decoder_learning_rate=DECODER_LEARNING_RATE,
                encoder_head_learning_rate=ENCODER_HEAD_LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
            )
            optimizer = torch.optim.AdamW(optimizer_groups)
            weights = JointLossWeights(**LOSS_WEIGHTS)
            initial_snapshot = parameter_snapshot(groups)
            standardization = (
                {
                    name: value.to(DEVICE)
                    for name, value in standardization_cpu.items()
                }
                if standardization_cpu is not None
                else None
            )
            residual_std = residual_std_cpu.to(DEVICE)
            run_config = {
                **run_spec,
                "branch": provenance["branch"],
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "eval_batch_size": EVAL_BATCH_SIZE,
                "loss_weights": LOSS_WEIGHTS,
                "alignment_target": ALIGNMENT_TARGET,
                "latent_alignment_mode": LATENT_ALIGNMENT_MODE,
                "latent_standardization_state_identity": (
                    sha256_state_dict(standardization_cpu)
                    if standardization_cpu is not None
                    else None
                ),
                "projector_residual_std_state_identity": sha256_state_dict(
                    {"residual_std": residual_std_cpu}
                ),
                "optimizer_group_settings": optimizer_settings,
                "gradient_clip": GRADIENT_CLIP,
                "amp": AMP,
                "amp_dtype": str(MIXED_PRECISION_DTYPE),
                "seed": SEED,
                "projector_seed": PROJECTOR_SEED,
                "maximum_ae_top5_degradation_percent": (
                    MAXIMUM_AE_TOP5_DEGRADATION_PERCENT
                ),
                "test_used_for_selection": False,
            }
            atomic_write_json(run_dir / "effective_config.json", run_config)
            atomic_write_json(run_dir / "provenance.json", provenance)
            atomic_write_json(
                run_dir / "trainable_parameter_manifest.json", manifest
            )
            atomic_write_csv(
                run_dir / "trainable_parameter_manifest.csv",
                manifest["trainable"],
            )

            expected_binding = checkpoint_binding(
                original_ae_identity=provenance["original_ae_identity"],
                starting_ae=starting_ae,
                adapted_ae=adapted_ae,
                projector=projector,
                text_cache_identity=provenance["text_cache_identity"],
                split_fingerprints=provenance["split_fingerprints"],
                unfrozen_parameter_names=manifest["unfrozen_parameter_names"],
                loss_weights=LOSS_WEIGHTS,
                optimizer_groups=optimizer_settings,
            )
            history = read_csv_rows(run_dir / "history.csv")
            validation_rows = read_csv_rows(run_dir / "validation_metrics.csv")
            safety_rows = read_csv_rows(run_dir / "safety_rule_decisions.csv")
            checkpoint_manifest_path = run_dir / "checkpoint_manifest.json"
            checkpoint_manifest = (
                json.loads(checkpoint_manifest_path.read_text())
                if checkpoint_manifest_path.exists()
                else {"checkpoints": {}, "selection_split": "validation"}
            )
            last_path = run_dir / "checkpoints" / "last.pt"
            start_epoch = 1
            stale_epochs = 0
            best_early = -float("inf")
            if last_path.exists():
                resumed = load_checkpoint(
                    last_path,
                    projector,
                    adapted_ae,
                    optimizer,
                    expected_binding,
                    run_config,
                )
                start_epoch = int(resumed["epoch"]) + 1
                history = [
                    row for row in history if int(row["epoch"]) < start_epoch
                ]
                validation_rows = [
                    row
                    for row in validation_rows
                    if int(row["epoch"]) < start_epoch
                ]
                safety_rows = [
                    row
                    for row in safety_rows
                    if int(row["epoch"]) < start_epoch
                ]
                best_early = -float("inf")
                stale_epochs = 0
                for row in validation_rows:
                    value = float(row["generation_top5_dice"])
                    if value > best_early:
                        best_early = value
                        stale_epochs = 0
                    else:
                        stale_epochs += 1
                if (
                    SAFETY_ACTION == "stop"
                    and not bool(resumed["safety_rule_decision"]["safe"])
                ):
                    start_epoch = EPOCHS + 1

            for epoch in range(start_epoch, EPOCHS + 1):
                train_metrics, train_n = train_one_epoch(
                    projector,
                    adapted_ae,
                    original_ae,
                    provider.train,
                    lookup,
                    optimizer,
                    groups,
                    initial_snapshot,
                    initial_ae_state,
                    original_identity,
                    weights,
                    standardization,
                    residual_std,
                    epoch=epoch,
                    replay_noise_scale=run_spec["replay_noise_scale"],
                    shuffled_text=run_spec["shuffled_text"],
                )
                validation = evaluate_joint(
                    projector,
                    adapted_ae,
                    original_ae,
                    semantic_model,
                    provider.val,
                    lookup,
                    split="val",
                    max_batches=MAX_EVAL_BATCHES,
                )
                summary = validation["summary"]
                safety = validation["safety"]
                history.append(
                    {
                        "epoch": epoch,
                        "n": train_n,
                        **train_metrics,
                    }
                )
                validation_rows.append(
                    {
                        "epoch": epoch,
                        "n": validation["n"],
                        **summary,
                    }
                )
                safety_rows.append({"epoch": epoch, **safety})
                atomic_write_csv(run_dir / "history.csv", history)
                atomic_write_csv(
                    run_dir / "validation_metrics.csv", validation_rows
                )
                atomic_write_csv(
                    run_dir / "safety_rule_decisions.csv", safety_rows
                )

                last_checkpoint, _ = save_checkpoint(
                    run_dir,
                    "last",
                    epoch,
                    projector,
                    adapted_ae,
                    starting_ae,
                    optimizer,
                    provenance,
                    manifest,
                    optimizer_settings,
                    summary,
                    safety,
                    run_config,
                )
                checkpoint_manifest["checkpoints"]["last"] = {
                    "path": str(last_checkpoint.relative_to(run_dir)),
                    "epoch": epoch,
                    "safe": bool(safety["safe"]),
                }

                if safety["safe"]:
                    candidates = {
                        "best_generation_top5": (
                            summary["generation_top5_dice"], "max"
                        ),
                        "best_semantic_auc": (
                            summary["semantic_normalized_recall_auc"], "max"
                        ),
                        "best_ae_preserving": (
                            summary["adapted_ae_top5_dice"], "max"
                        ),
                    }
                    for role, (value, direction) in candidates.items():
                        current = checkpoint_manifest["checkpoints"].get(role)
                        better = (
                            math.isfinite(float(value))
                            and (
                                current is None
                                or (
                                    float(value) > float(current["value"])
                                    if direction == "max"
                                    else float(value) < float(current["value"])
                                )
                            )
                        )
                        if better:
                            path, _ = save_checkpoint(
                                run_dir,
                                role,
                                epoch,
                                projector,
                                adapted_ae,
                                starting_ae,
                                optimizer,
                                provenance,
                                manifest,
                                optimizer_settings,
                                summary,
                                safety,
                                run_config,
                            )
                            checkpoint_manifest["checkpoints"][role] = {
                                "path": str(path.relative_to(run_dir)),
                                "epoch": epoch,
                                "value": float(value),
                                "safe": True,
                            }
                atomic_write_json(
                    checkpoint_manifest_path, checkpoint_manifest
                )

                generation_top5 = float(summary["generation_top5_dice"])
                if generation_top5 > best_early:
                    best_early = generation_top5
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                print(
                    run_spec["run_name"],
                    f"epoch={epoch}",
                    f"gen_top5={generation_top5:.4f}",
                    f"AE_degradation={safety['top5_dice_degradation_percent']:.3f}%",
                    safety["action"],
                )
                if not safety["safe"] and SAFETY_ACTION == "stop":
                    break
                if (
                    EARLY_STOPPING_PATIENCE is not None
                    and stale_epochs >= EARLY_STOPPING_PATIENCE
                ):
                    break

            del original_ae, adapted_ae, projector, semantic_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {
                "run_dir": str(run_dir),
                "run_spec": run_spec,
                "checkpoint_manifest": checkpoint_manifest,
            }
        """
    ),
    markdown(
        """
        ## 8. Required controls

        Controls include exact original-AE bypass, exact true latents through each
        adapted decoder, 32-example overfit, latent-noise reconstruction at 0×,
        0.25×, 0.5×, 1×, and 2× observed residual standard deviation, shuffled
        image/text training, and the explicit zero-noise replay run.
        """
    ),
    code(
        """
        @torch.no_grad()
        def true_latent_bypass_control(
            original_ae, adapted_ae, dataset, lookup, max_batches=1,
        ):
            totals, n = {}, 0
            loader = make_loader(
                dataset,
                lookup,
                batch_size=EVAL_BATCH_SIZE,
                shuffle=False,
                seed=SEED,
            )
            for batch_index, batch in enumerate(loader):
                if batch_index >= max_batches:
                    break
                target = batch["volume"].to(DEVICE)
                true_latent = original_ae.encoder(target)
                original = original_ae.decoder(true_latent)
                adapted = adapted_ae.decoder(true_latent)
                batch_n = len(target)
                weighted_update(
                    totals,
                    reconstruction_metrics(original, target),
                    batch_n,
                    "original_exact_true_latent_",
                )
                weighted_update(
                    totals,
                    reconstruction_metrics(adapted, target),
                    batch_n,
                    "adapted_exact_true_latent_",
                )
                n += batch_n
            return {"n": n, **{name: value / n for name, value in totals.items()}}

        def run_tiny_overfit_control(
            output_dir, original_ae_cpu, starting_ae_cpu,
            starting_projector_cpu, dataset, lookup,
        ):
            output_dir = Path(output_dir)
            seed_everything(PROJECTOR_SEED)
            original_ae = untouched_autoencoder(original_ae_cpu).to(DEVICE)
            adapted_ae = copy.deepcopy(starting_ae_cpu).to(DEVICE)
            projector = copy.deepcopy(starting_projector_cpu).to(DEVICE)
            groups = configure_trainable_variant(
                adapted_ae,
                projector,
                "projector_plus_seed_and_last_block",
            )
            optimizer_groups, _ = optimizer_group_settings(
                groups,
                projector_learning_rate=TINY_OVERFIT_LEARNING_RATE,
                decoder_learning_rate=TINY_OVERFIT_LEARNING_RATE / 10,
                encoder_head_learning_rate=TINY_OVERFIT_LEARNING_RATE / 20,
                weight_decay=0.0,
            )
            optimizer = torch.optim.AdamW(optimizer_groups)
            initial = parameter_snapshot(groups)
            tiny_dataset = Subset(
                dataset, range(min(TINY_OVERFIT_N, len(dataset)))
            )
            batch = next(
                iter(
                    DataLoader(
                        tiny_dataset,
                        batch_size=min(TINY_OVERFIT_N, len(tiny_dataset)),
                        shuffle=False,
                        collate_fn=AtlasFreeContrastiveCollator(
                            lookup, (36, 45, 38)
                        ),
                    )
                )
            )
            target = batch["volume"].to(DEVICE)
            text = batch["text_embedding"].to(DEVICE)
            weights = JointLossWeights(**LOSS_WEIGHTS)
            history = []
            for step in range(1, TINY_OVERFIT_STEPS + 1):
                optimizer.zero_grad(set_to_none=True)
                result = compute_joint_loss(
                    projector,
                    adapted_ae,
                    original_ae,
                    text,
                    target,
                    weights=weights,
                    initial_parameters=initial,
                    alignment_target=ALIGNMENT_TARGET,
                )
                result.total.backward()
                optimizer.step()
                if (
                    step == 1
                    or step == TINY_OVERFIT_STEPS
                    or step % max(1, TINY_OVERFIT_STEPS // 20) == 0
                ):
                    history.append(
                        {
                            "step": step,
                            "loss": float(result.total.detach()),
                            "latent_mse": float(
                                result.components["latent_alignment"].detach()
                            ),
                            **{
                                f"generation_{name}": value
                                for name, value in reconstruction_metrics(
                                    result.generated_volume, target
                                ).items()
                            },
                            **{
                                f"replay_{name}": value
                                for name, value in reconstruction_metrics(
                                    result.clean_replay_volume, target
                                ).items()
                            },
                        }
                    )
            atomic_write_csv(output_dir / "tiny_overfit_history.csv", history)
            atomic_write_json(
                output_dir / "tiny_overfit_summary.json", history[-1]
            )
            return history[-1]

        @torch.no_grad()
        def latent_noise_reconstruction_test(
            adapted_ae, original_ae, dataset, lookup, residual_std, scales,
        ):
            rows = []
            loader = make_loader(
                dataset,
                lookup,
                batch_size=EVAL_BATCH_SIZE,
                shuffle=False,
                seed=SEED,
            )
            cached = []
            for batch_index, batch in enumerate(loader):
                if (
                    MAX_EVAL_BATCHES is not None
                    and batch_index >= MAX_EVAL_BATCHES
                ):
                    break
                target = batch["volume"].to(DEVICE)
                cached.append((target, original_ae.encoder(target)))
            for scale in scales:
                for decoder_name, decoder in (
                    ("original", original_ae.decoder),
                    ("adapted", adapted_ae.decoder),
                ):
                    totals, n = {}, 0
                    seed_everything(SEED)
                    for target, latent in cached:
                        noise = torch.randn_like(latent) * residual_std * float(scale)
                        prediction = decoder(latent + noise)
                        batch_n = len(target)
                        weighted_update(
                            totals,
                            reconstruction_metrics(prediction, target),
                            batch_n,
                        )
                        n += batch_n
                    rows.append(
                        {
                            "noise_scale": float(scale),
                            "decoder": decoder_name,
                            "n": n,
                            **{
                                name: value / n
                                for name, value in totals.items()
                            },
                        }
                    )
            return rows
        """
    ),
    markdown(
        """
        ## 9. Run training and validation selection

        This phase never accesses `provider.test`. The train-only latent
        standardization and validation residual calibration are identity-bound and
        saved for reproducibility.
        """
    ),
    code(
        """
        EFFECTIVE_CONFIG = {
            "branches_to_run": BRANCHES_TO_RUN,
            "variants_to_run": VARIANTS_TO_RUN,
            "latent_noise_scales": LATENT_NOISE_SCALES,
            "run_zero_noise_replay_control": RUN_ZERO_NOISE_REPLAY_CONTROL,
            "run_shuffled_text_pair_control": RUN_SHUFFLED_TEXT_PAIR_CONTROL,
            "run_tiny_overfit_control": RUN_TINY_OVERFIT_CONTROL,
            "run_full_training": RUN_FULL_TRAINING,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "projector_learning_rate": PROJECTOR_LEARNING_RATE,
            "decoder_learning_rate": DECODER_LEARNING_RATE,
            "encoder_head_learning_rate": ENCODER_HEAD_LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "loss_weights": LOSS_WEIGHTS,
            "alignment_target": ALIGNMENT_TARGET,
            "latent_alignment_mode": LATENT_ALIGNMENT_MODE,
            "noise_decoder_components": NOISE_DECODER_COMPONENTS,
            "maximum_ae_top5_degradation_percent": (
                MAXIMUM_AE_TOP5_DEGRADATION_PERCENT
            ),
            "safety_action": SAFETY_ACTION,
            "amp": AMP,
            "amp_dtype": str(MIXED_PRECISION_DTYPE),
            "seed": SEED,
            "projector_seed": PROJECTOR_SEED,
            "test_used_for_selection": False,
            "released_models_overwritten": False,
        }
        atomic_write_json(
            EXPERIMENT_ROOT / "effective_config.json", EFFECTIVE_CONFIG
        )
        atomic_write_json(
            EXPERIMENT_ROOT / "environment.json", ENVIRONMENT
        )

        lookup = AtlasFreeTextEmbeddingLookup.published()
        run_records = []
        root_provenance = {
            "environment": ENVIRONMENT,
            "branches": {},
            "resolved_commit": RESOLVED_COMMIT,
        }

        for branch in BRANCHES_TO_RUN:
            print(f"===== {branch} =====")
            branch_dir = EXPERIMENT_ROOT / branch
            branch_dir.mkdir(parents=True, exist_ok=True)
            audit_dir = branch_dir / "provenance_audits"
            audit_dir.mkdir(exist_ok=True)
            (
                spec, ae_path, original_ae_cpu, starting_ae_cpu,
                starting_projector_cpu, provider, semantic_model_cpu,
            ) = load_branch_resources(branch)
            provenance = build_branch_provenance(
                spec,
                ae_path,
                original_ae_cpu,
                starting_ae_cpu,
                starting_projector_cpu,
                provider,
                lookup,
                audit_dir,
            )
            root_provenance["branches"][branch] = provenance
            atomic_write_json(branch_dir / "provenance.json", provenance)

            train_latents = encode_stage1_latents(
                original_ae_cpu,
                provider.train,
                lookup,
                device=DEVICE,
                batch_size=EVAL_BATCH_SIZE,
                num_workers=NUM_WORKERS,
            )
            standardization_cpu = (
                fit_latent_standardization(train_latents)
                if LATENT_ALIGNMENT_MODE == "standardized"
                else None
            )
            original_ae_probe = untouched_autoencoder(original_ae_cpu).to(DEVICE)
            projector_probe = copy.deepcopy(starting_projector_cpu).to(DEVICE)
            residual_calibration = estimate_projector_residuals(
                projector_probe,
                original_ae_probe,
                provider.val,
                lookup,
                max_batches=MAX_EVAL_BATCHES,
            )
            residual_std_cpu = residual_calibration["residual_std"].cpu()
            torch.save(
                {
                    **residual_calibration,
                    "residual_std": residual_std_cpu,
                    "validation_split_fingerprint": (
                        provenance["split_fingerprints"]["val"]
                    ),
                    "original_ae_state_identity": sha256_state_dict(
                        original_ae_cpu
                    ),
                    "starting_projector_state_identity": sha256_state_dict(
                        starting_projector_cpu
                    ),
                },
                branch_dir / "latent_residual_calibration.pt",
            )
            atomic_write_json(
                branch_dir / "latent_residual_calibration.json",
                {
                    key: value
                    for key, value in residual_calibration.items()
                    if key != "residual_std"
                },
            )
            del original_ae_probe, projector_probe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            controls_dir = branch_dir / "controls"
            controls_dir.mkdir(exist_ok=True)
            original_probe = untouched_autoencoder(original_ae_cpu).to(DEVICE)
            initial_adapted_probe = copy.deepcopy(starting_ae_cpu).to(DEVICE)
            bypass = true_latent_bypass_control(
                original_probe,
                initial_adapted_probe,
                provider.val,
                lookup,
            )
            atomic_write_json(
                controls_dir / "initial_true_latent_bypass.json", bypass
            )
            if RUN_TINY_OVERFIT_CONTROL:
                run_tiny_overfit_control(
                    controls_dir,
                    original_ae_cpu,
                    starting_ae_cpu,
                    starting_projector_cpu,
                    provider.train,
                    lookup,
                )
            del original_probe, initial_adapted_probe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if RUN_FULL_TRAINING:
                for run_spec in expanded_run_specs():
                    record = train_run(
                        branch_dir,
                        run_spec,
                        original_ae_cpu,
                        starting_ae_cpu,
                        starting_projector_cpu,
                        semantic_model_cpu,
                        provider,
                        lookup,
                        provenance,
                        standardization_cpu,
                        residual_std_cpu,
                    )
                    record["branch"] = branch
                    run_records.append(record)

            del (
                original_ae_cpu,
                starting_ae_cpu,
                starting_projector_cpu,
                semantic_model_cpu,
                provider,
                train_latents,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        atomic_write_json(EXPERIMENT_ROOT / "provenance.json", root_provenance)
        atomic_write_json(
            EXPERIMENT_ROOT / "validation_selected_runs.json", run_records
        )
        print("Validation selection complete. Test has not been evaluated.")
        """
    ),
    markdown(
        """
        ## 10. Held-out test evaluation after validation selection

        Only checkpoints named in the validation-created manifest are evaluated.
        Test results are never fed back into checkpoint or variant selection.
        """
    ),
    code(
        """
        def save_generated_maps(run_dir, role, examples):
            examples_dir = Path(run_dir) / "generated_maps"
            examples_dir.mkdir(exist_ok=True)
            torch.save(
                {"role": role, "items": examples},
                examples_dir / f"{role}.pt",
            )
            if not examples:
                return
            fig, axes = plt.subplots(
                len(examples), 4,
                figsize=(12, 3 * len(examples)),
                squeeze=False,
            )
            for row, example in enumerate(examples):
                target = example["target"].squeeze()
                prediction = example["prediction"].squeeze()
                original = example["original_reconstruction"].squeeze()
                adapted = example["adapted_reconstruction"].squeeze()
                z = int(target.abs().sum(dim=(0, 1)).argmax())
                for column, (title, volume) in enumerate(
                    (
                        ("target", target),
                        ("generated", prediction),
                        ("original AE", original),
                        ("adapted AE", adapted),
                    )
                ):
                    axes[row, column].imshow(
                        volume[:, :, z], cmap="coolwarm"
                    )
                    axes[row, column].set_title(
                        f"{title} | {example['map_id']}"
                    )
                    axes[row, column].axis("off")
            fig.tight_layout()
            fig.savefig(
                examples_dir / f"{role}.png", dpi=160, bbox_inches="tight"
            )
            plt.close(fig)

        generation_rows = []
        ae_rows = []
        safety_rows_all = []
        noise_rows_all = []
        all_manifest_rows = []

        for record in run_records:
            branch = record["branch"]
            run_dir = Path(record["run_dir"])
            run_spec = record["run_spec"]
            (
                spec, ae_path, original_ae_cpu, starting_ae_cpu,
                starting_projector_cpu, provider, semantic_model_cpu,
            ) = load_branch_resources(branch)
            provenance = root_provenance["branches"][branch]
            residual_payload = torch.load(
                EXPERIMENT_ROOT
                / branch
                / "latent_residual_calibration.pt",
                map_location="cpu",
                weights_only=True,
            )
            residual_std = residual_payload["residual_std"].to(DEVICE)
            manifest_json = json.loads(
                (run_dir / "trainable_parameter_manifest.json").read_text()
            )
            run_config = json.loads(
                (run_dir / "effective_config.json").read_text()
            )

            for role, selected in record["checkpoint_manifest"][
                "checkpoints"
            ].items():
                if role == "last" or not selected.get("safe", False):
                    continue
                original_ae = untouched_autoencoder(original_ae_cpu).to(DEVICE)
                adapted_ae = copy.deepcopy(starting_ae_cpu).to(DEVICE)
                projector = copy.deepcopy(starting_projector_cpu).to(DEVICE)
                groups = configure_trainable_variant(
                    adapted_ae,
                    projector,
                    run_spec["variant"],
                    noise_decoder_components=NOISE_DECODER_COMPONENTS,
                )
                optimizer_groups, optimizer_settings = optimizer_group_settings(
                    groups,
                    projector_learning_rate=PROJECTOR_LEARNING_RATE,
                    decoder_learning_rate=DECODER_LEARNING_RATE,
                    encoder_head_learning_rate=ENCODER_HEAD_LEARNING_RATE,
                    weight_decay=WEIGHT_DECAY,
                )
                optimizer = torch.optim.AdamW(optimizer_groups)
                expected = checkpoint_binding(
                    original_ae_identity=provenance["original_ae_identity"],
                    starting_ae=starting_ae_cpu,
                    adapted_ae=adapted_ae,
                    projector=projector,
                    text_cache_identity=provenance["text_cache_identity"],
                    split_fingerprints=provenance["split_fingerprints"],
                    unfrozen_parameter_names=(
                        manifest_json["unfrozen_parameter_names"]
                    ),
                    loss_weights=LOSS_WEIGHTS,
                    optimizer_groups=optimizer_settings,
                )
                checkpoint_path = run_dir / selected["path"]
                payload = load_checkpoint(
                    checkpoint_path,
                    projector,
                    adapted_ae,
                    optimizer,
                    expected,
                    run_config,
                )
                semantic_model = copy.deepcopy(semantic_model_cpu).to(DEVICE)
                evaluation = evaluate_joint(
                    projector,
                    adapted_ae,
                    original_ae,
                    semantic_model,
                    provider.test,
                    lookup,
                    split="test",
                    max_batches=MAX_EVAL_BATCHES,
                )
                base = {
                    "branch": branch,
                    "run_name": run_spec["run_name"],
                    "variant": run_spec["variant"],
                    "control": run_spec["control"],
                    "noise_scale": run_spec["replay_noise_scale"],
                    "checkpoint_role": role,
                    "checkpoint_epoch": payload["epoch"],
                    "selection_split": "validation",
                    "test_used_for_selection": False,
                    "n": evaluation["n"],
                }
                generation_rows.append(
                    {
                        **base,
                        **{
                            name: value
                            for name, value in evaluation["summary"].items()
                            if (
                                name.startswith("generation_")
                                or name.startswith("adapted_target_")
                                or name
                                in {
                                    "latent_mse",
                                    "latent_variance_ratio",
                                    "latent_norm_ratio",
                                    "explained_variance",
                                    "decoded_mse",
                                    "foreground_mse",
                                    "spatial_correlation",
                                    "top5_dice",
                                    "semantic_normalized_recall_auc",
                                    "adapted_vs_original_latent_mse",
                                }
                            )
                        },
                    }
                )
                ae_rows.append(
                    {
                        **base,
                        **{
                            name: value
                            for name, value in evaluation["summary"].items()
                            if (
                                name.startswith("original_ae_")
                                or name.startswith("adapted_ae_")
                                or name.startswith(
                                    "adapted_decoder_true_latent_"
                                )
                            )
                        },
                        **evaluation["safety"],
                    }
                )
                safety_rows_all.append({**base, **evaluation["safety"]})
                all_manifest_rows.extend(
                    {
                        "branch": branch,
                        "run_name": run_spec["run_name"],
                        **row,
                    }
                    for row in manifest_json["trainable"]
                )
                if role == "best_generation_top5":
                    save_generated_maps(
                        run_dir, role, evaluation["examples"]
                    )
                    bypass = true_latent_bypass_control(
                        original_ae,
                        adapted_ae,
                        provider.test,
                        lookup,
                        max_batches=(
                            MAX_EVAL_BATCHES
                            if MAX_EVAL_BATCHES is not None
                            else math.inf
                        ),
                    )
                    atomic_write_json(
                        run_dir / "controls" / "selected_true_latent_bypass.json",
                        bypass,
                    )
                    noise_rows = latent_noise_reconstruction_test(
                        adapted_ae,
                        original_ae,
                        provider.test,
                        lookup,
                        residual_std,
                        [0.0, 0.25, 0.5, 1.0, 2.0],
                    )
                    noise_rows_all.extend(
                        {
                            "branch": branch,
                            "run_name": run_spec["run_name"],
                            "variant": run_spec["variant"],
                            "training_noise_scale": (
                                run_spec["replay_noise_scale"]
                            ),
                            **row,
                        }
                        for row in noise_rows
                    )
                    atomic_write_csv(
                        run_dir
                        / "controls"
                        / "latent_noise_reconstruction_test.csv",
                        noise_rows,
                    )
                del original_ae, adapted_ae, projector, semantic_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            del (
                original_ae_cpu,
                starting_ae_cpu,
                starting_projector_cpu,
                semantic_model_cpu,
                provider,
            )

        atomic_write_csv(
            EXPERIMENT_ROOT / "generation_metrics.csv", generation_rows
        )
        atomic_write_csv(
            EXPERIMENT_ROOT / "ae_retention_metrics.csv", ae_rows
        )
        atomic_write_csv(
            EXPERIMENT_ROOT / "safety_rule_decisions.csv", safety_rows_all
        )
        atomic_write_csv(
            EXPERIMENT_ROOT / "latent_noise_robustness.csv", noise_rows_all
        )
        atomic_write_csv(
            EXPERIMENT_ROOT / "trainable_parameter_manifest.csv",
            all_manifest_rows,
        )
        comparison = [
            {
                **generation,
                **{
                    key: value
                    for key, value in ae.items()
                    if key not in generation
                },
            }
            for generation in generation_rows
            for ae in ae_rows
            if (
                generation["branch"],
                generation["run_name"],
                generation["checkpoint_role"],
            )
            == (
                ae["branch"],
                ae["run_name"],
                ae["checkpoint_role"],
            )
        ]
        atomic_write_csv(
            EXPERIMENT_ROOT / "comparison.csv", comparison
        )
        atomic_write_json(
            EXPERIMENT_ROOT / "comparison.json", comparison
        )
        print("Held-out test evaluation complete:", EXPERIMENT_ROOT)
        """
    ),
    markdown("## 11. Plots and final report"),
    code(
        """
        def collect_history():
            rows = []
            safety = []
            for record in run_records:
                run_dir = Path(record["run_dir"])
                rows.extend(
                    {
                        "branch": record["branch"],
                        "run_name": record["run_spec"]["run_name"],
                        "variant": record["run_spec"]["variant"],
                        **row,
                    }
                    for row in read_csv_rows(run_dir / "history.csv")
                )
                safety.extend(
                    {
                        "branch": record["branch"],
                        "run_name": record["run_spec"]["run_name"],
                        "variant": record["run_spec"]["variant"],
                        **row,
                    }
                    for row in read_csv_rows(
                        run_dir / "safety_rule_decisions.csv"
                    )
                )
            return rows, safety

        history_rows, validation_safety_rows = collect_history()
        atomic_write_csv(EXPERIMENT_ROOT / "histories.csv", history_rows)
        atomic_write_csv(
            EXPERIMENT_ROOT / "validation_safety_history.csv",
            validation_safety_rows,
        )

        plots_dir = EXPERIMENT_ROOT / "plots"
        plots_dir.mkdir(exist_ok=True)
        if history_rows:
            history_frame = pd.DataFrame(history_rows)
            drift_columns = [
                column
                for column in history_frame.columns
                if column.startswith("parameter_drift_")
            ]
            fig, ax = plt.subplots(figsize=(12, 6))
            for (branch, run_name), frame in history_frame.groupby(
                ["branch", "run_name"]
            ):
                for column in drift_columns:
                    values = pd.to_numeric(frame[column], errors="coerce")
                    if values.notna().any():
                        ax.plot(
                            pd.to_numeric(frame["epoch"]),
                            values,
                            label=f"{branch}/{run_name}/{column}",
                            alpha=0.75,
                        )
            ax.set_yscale("symlog", linthresh=1e-8)
            ax.set_xlabel("epoch")
            ax.set_ylabel("L2 drift from initialization")
            ax.set_title("Parameter drift by optimizer group")
            ax.legend(fontsize=5, ncol=2)
            fig.tight_layout()
            fig.savefig(
                plots_dir / "parameter_drift_plots.png", dpi=180
            )
            plt.close(fig)

        if noise_rows_all:
            noise_frame = pd.DataFrame(noise_rows_all)
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            for (run_name, decoder), frame in noise_frame.groupby(
                ["run_name", "decoder"]
            ):
                frame = frame.sort_values("noise_scale")
                axes[0].plot(
                    frame["noise_scale"],
                    frame["top5_dice"],
                    marker="o",
                    label=f"{run_name}/{decoder}",
                )
                axes[1].plot(
                    frame["noise_scale"],
                    frame["mse"],
                    marker="o",
                    label=f"{run_name}/{decoder}",
                )
            axes[0].set_title("Latent-noise robustness: top-5 Dice")
            axes[1].set_title("Latent-noise robustness: reconstruction MSE")
            for ax in axes:
                ax.set_xlabel("residual standard-deviation scale")
                ax.legend(fontsize=5)
            fig.tight_layout()
            fig.savefig(
                plots_dir / "latent_noise_robustness_plots.png", dpi=180
            )
            plt.close(fig)

        def percent(value):
            return f"{float(value):.3f}%"

        selected = [
            row
            for row in comparison
            if row["checkpoint_role"] == "best_generation_top5"
            and row.get("control") in {None, "latent_noise_adaptation"}
        ]
        baseline_by_branch = {
            row["branch"]: row
            for row in selected
            if row["variant"] == "projector_only_baseline"
        }
        best = (
            max(selected, key=lambda row: float(row["top5_dice"]))
            if selected
            else None
        )
        improvements = []
        for row in selected:
            baseline = baseline_by_branch.get(row["branch"])
            if baseline is not None:
                improvements.append(
                    {
                        **row,
                        "top5_delta_vs_projector_only": (
                            float(row["top5_dice"])
                            - float(baseline["top5_dice"])
                        ),
                        "semantic_delta_vs_projector_only": (
                            float(row["semantic_normalized_recall_auc"])
                            - float(
                                baseline["semantic_normalized_recall_auc"]
                            )
                        ),
                    }
                )
        best_improvement = (
            max(
                improvements,
                key=lambda row: row["top5_delta_vs_projector_only"],
            )
            if improvements
            else None
        )
        threshold_counts = {
            threshold: sum(
                float(row["top5_dice_degradation_percent"]) <= threshold
                for row in improvements
                if row["top5_delta_vs_projector_only"] > 0
            )
            for threshold in (1, 2, 5)
        }
        noise_variants = [
            row
            for row in improvements
            if row["variant"] == "latent_noise_decoder_adaptation"
        ]
        best_noise = (
            max(
                noise_variants,
                key=lambda row: row["top5_delta_vs_projector_only"],
            )
            if noise_variants
            else None
        )
        noise_robustness_answer = None
        if noise_rows_all:
            robustness = pd.DataFrame(noise_rows_all)
            robustness = robustness[
                (robustness["variant"] == "latent_noise_decoder_adaptation")
                & (robustness["decoder"] == "adapted")
                & (pd.to_numeric(robustness["noise_scale"]) > 0)
            ].copy()
            if not robustness.empty:
                scores = (
                    robustness.groupby(
                        ["branch", "run_name", "training_noise_scale"],
                        as_index=False,
                    )
                    .agg(mean_top5_dice=("top5_dice", "mean"), mean_mse=("mse", "mean"))
                )
                zero_by_branch = {
                    row["branch"]: row
                    for _, row in scores[
                        pd.to_numeric(scores["training_noise_scale"]) == 0
                    ].iterrows()
                }
                comparisons = []
                for _, row in scores[
                    pd.to_numeric(scores["training_noise_scale"]) > 0
                ].iterrows():
                    zero = zero_by_branch.get(row["branch"])
                    if zero is not None:
                        comparisons.append(
                            {
                                **row.to_dict(),
                                "top5_delta_vs_zero_noise_training": (
                                    float(row["mean_top5_dice"])
                                    - float(zero["mean_top5_dice"])
                                ),
                                "mse_delta_vs_zero_noise_training": (
                                    float(row["mean_mse"])
                                    - float(zero["mean_mse"])
                                ),
                            }
                        )
                if comparisons:
                    robust_best = max(
                        comparisons,
                        key=lambda row: row[
                            "top5_delta_vs_zero_noise_training"
                        ],
                    )
                    robust = (
                        robust_best["top5_delta_vs_zero_noise_training"] > 0
                        and robust_best["mse_delta_vs_zero_noise_training"] < 0
                    )
                    noise_robustness_answer = (
                        f"{'Yes' if robust else 'No clear joint improvement'}: "
                        f"training at {float(robust_best['training_noise_scale']):g}× "
                        f"changed mean nonzero-noise top-5 Dice by "
                        f"{robust_best['top5_delta_vs_zero_noise_training']:+.4f} "
                        f"and MSE by "
                        f"{robust_best['mse_delta_vs_zero_noise_training']:+.6g} "
                        "versus zero-noise decoder training."
                    )

        if best_improvement is None:
            answer_1 = "No completed validation-selected test runs are available."
            answer_2 = "Not determined."
            answer_3 = "Not determined."
            answer_4 = "Not determined."
            answer_5 = "Not determined."
        else:
            delta = best_improvement["top5_delta_vs_projector_only"]
            answer_1 = (
                f"{'Yes' if delta > 0 else 'No'}: best test top-5 Dice delta "
                f"versus projector-only was {delta:+.4f} for "
                f"`{best_improvement['run_name']}`. Validation, not test, "
                "selected the checkpoint."
            )
            answer_2 = (
                f"The strongest tested layer selection was "
                f"`{best_improvement['variant']}` "
                f"(test top-5 Dice {float(best_improvement['top5_dice']):.4f})."
            )
            answer_3 = (
                f"Positive top-5 gains satisfying <1%, <2%, and <5% AE "
                f"degradation: {threshold_counts[1]}, {threshold_counts[2]}, "
                f"and {threshold_counts[5]} runs, respectively."
            )
            if noise_robustness_answer is not None:
                answer_4 = noise_robustness_answer
            elif best_noise is None:
                answer_4 = "No latent-noise adaptation result completed."
            else:
                answer_4 = (
                    f"The best noise-adapted run changed top-5 Dice by "
                    f"{best_noise['top5_delta_vs_projector_only']:+.4f}; see "
                    "`latent_noise_robustness.csv` for scale-wise decoder tests."
                )
            decoder_gain = (
                float(best_improvement["generation_top5_dice"])
                - float(
                    best_improvement[
                        "generation_original_decoder_top5_dice"
                    ]
                )
            )
            target_shift = float(
                best_improvement["adapted_vs_original_latent_mse"]
            )
            answer_5 = (
                f"Adapted-vs-original decoding at the same predicted latent "
                f"changed top-5 Dice by {decoder_gain:+.4f}; encoder target-space "
                f"shift MSE was {target_shift:.6g}. A decoder gain with negligible "
                "target shift supports better decoding rather than target redefinition."
            )

        report = f'''# Stage 4 joint AE/projector fine-tuning report

        Experiment directory: `{EXPERIMENT_ROOT}`

        Selection policy: checkpoints were selected on validation only. Held-out test
        evaluation ran afterward. The original Stage 1 checkpoint was never written.

        ## 1. Does adapting the decoder improve text-to-brain generation?

        {answer_1}

        ## 2. Which decoder layers are most useful?

        {answer_2}

        ## 3. Can improvements stay below 1%, 2%, or 5% AE degradation?

        {answer_3}

        ## 4. Does latent-noise training improve robustness?

        {answer_4}

        ## 5. Better decoding or a changed AE target space?

        {answer_5}

        ## Safety interpretation

        The hard acceptance rule is a relative top-5 Dice degradation no greater than
        {MAXIMUM_AE_TOP5_DEGRADATION_PERCENT:.1f}%. The report also records 1%, 2%, and
        5% flags, spatial-correlation degradation, and MSE degradation for every
        validation and test evaluation. Unsafe epochs only update `last.pt`; they
        cannot replace a safe best checkpoint.

        ## Main artifacts

        - `effective_config.json`, `environment.json`, `provenance.json`
        - `trainable_parameter_manifest.csv`
        - `histories.csv`, `validation_safety_history.csv`
        - `generation_metrics.csv`, `ae_retention_metrics.csv`
        - `safety_rule_decisions.csv`
        - `latent_noise_robustness.csv`
        - `comparison.csv`, `comparison.json`
        - `plots/parameter_drift_plots.png`
        - `plots/latent_noise_robustness_plots.png`
        - per-run `generated_maps/`, controls, histories, and identity-bound checkpoints
        '''
        report = textwrap.dedent(report)
        (EXPERIMENT_ROOT / "final_report.md").write_text(
            report, encoding="utf-8"
        )
        atomic_write_json(
            ACTIVE_POINTER,
            {
                "path": str(EXPERIMENT_ROOT),
                "state": "completed",
                "updated_at": utc_stamp(),
            },
        )
        print(report)
        """
    ),
    markdown(
        """
        ## Regression guarantees

        Repository tests cover exact variant freezing, replay-gradient behavior,
        immutable checkpoint bindings, and direction-aware 1%/2%/5% AE-retention
        decisions. The training loop repeats the frozen-state and untouched-original
        identity assertions after every epoch.

        ```bash
        pytest -q tests/test_stage4_joint_finetuning.py
        ```
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
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
