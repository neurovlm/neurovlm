"""Build the Stage 4 probabilistic latent-generation Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "stage4_probabilistic_latent_generation.ipynb"


def lines(text: str) -> list[str]:
    value = dedent(text).strip("\n") + "\n"
    return value.splitlines(keepends=True)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


cells = [
    markdown(
        """
        # Stage 4 probabilistic latent generation

        This experiment tests whether deterministic MSE Stage 4 prediction
        collapses toward a conditional mean because text-to-brain generation is
        one-to-many. It trains a conditional VAE in the **exact frozen Stage 1
        autoencoder latent space**, preserves the released deterministic Stage 4
        model as baseline A, and treats oracle best-of-K as diagnostic only.

        The notebook is fresh-Colab compatible, resume-safe, branch-configurable,
        and binds every checkpoint to the AE, normalized SPECTER2 cache, immutable
        train/validation/test splits, fitted train-only standardization, code
        revision, and run configuration. The held-out test split is opened only
        after validation selection is frozen.
        """
    ),
    markdown("## 1. Drive, checkout, install, and immutable revision"),
    code(
        """
        from pathlib import Path
        import os, re, subprocess, sys

        IN_COLAB = "google.colab" in sys.modules
        if IN_COLAB:
            from google.colab import drive
            drive.mount("/content/drive")

        REPO_URL = "https://github.com/neurovlm/neurovlm.git"
        REPO_REF = os.environ.get("NEUROVLM_REPO_REF", "neurovlm_experiments")
        PINNED_COMMIT = os.environ.get("NEUROVLM_PINNED_COMMIT", "").strip()
        REPO_DIR = Path("/content/neurovlm" if IN_COLAB else Path.cwd()).resolve()
        if not (REPO_DIR / ".git").is_dir():
            REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)
        if IN_COLAB:
            subprocess.run(["git", "fetch", "--all", "--tags", "--prune"], cwd=REPO_DIR, check=True)
            target = PINNED_COMMIT or REPO_REF
            subprocess.run(["git", "checkout", "--detach", target], cwd=REPO_DIR, check=True)
        RESOLVED_COMMIT = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_DIR, text=True
        ).strip()
        if PINNED_COMMIT and RESOLVED_COMMIT != PINNED_COMMIT:
            raise RuntimeError(f"Pinned commit mismatch: {PINNED_COMMIT} != {RESOLVED_COMMIT}")
        if not re.fullmatch(r"[0-9a-f]{40}", RESOLVED_COMMIT):
            raise RuntimeError("Could not resolve an immutable git commit")
        print("Repository:", REPO_DIR)
        print("Configured ref:", REPO_REF)
        print("Exact commit:", RESOLVED_COMMIT)
        """
    ),
    code(
        """
        import subprocess, sys
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-e", f"{REPO_DIR}[cnn,metrics]"],
            check=True,
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q",
             "pyarrow", "scikit-learn", "seaborn", "nibabel"],
            check=True,
        )
        """
    ),
    markdown("## 2. Effective configuration"),
    code(
        """
        # Execution mode: smoke validates the entire path; full performs the scientific sweep.
        SMOKE_MODE = True
        FULL_EXPERIMENT = False
        assert SMOKE_MODE ^ FULL_EXPERIMENT

        ALL_BRANCHES = [
            "mixed_to_pubmed",
            "mixed_to_nilearn",
            "mixed_to_neurovault",
        ]
        BRANCHES_TO_RUN = ["mixed_to_pubmed"] if SMOKE_MODE else ALL_BRANCHES
        FULL_DATA_LIMIT = 128 if SMOKE_MODE else None
        EPOCHS = 2 if SMOKE_MODE else 100
        BATCH_SIZE = 16 if SMOKE_MODE else 64
        # Training batch stays baseline-compatible. K-sample evaluation already
        # expands this batch by K, so it is raised conservatively.
        EVAL_BATCH_SIZE = 32 if SMOKE_MODE else 64
        NUM_WORKERS = 8 if IN_COLAB else 0
        PREFETCH_FACTOR = 4
        SEED = 42
        LEARNING_RATE = 3e-4
        WEIGHT_DECAY = 1e-4
        GRADIENT_CLIP = 1.0
        AMP_DTYPE = "auto"  # BF16 when supported, otherwise FP16 on CUDA.
        LATENT_EPSILON = 1e-4

        U_DIMS = [64] if SMOKE_MODE else [32, 64, 128]
        BETA_VALUES = [0.01] if SMOKE_MODE else [0.001, 0.01, 0.05, 0.1]
        KL_SCHEDULES = ["linear", "cyclical"] if FULL_EXPERIMENT else ["linear"]
        KL_WARMUP_FRACTION = 0.30
        KL_CYCLE_EPOCHS = 20
        FREE_BITS_VALUES = [0.0, 0.01] if FULL_EXPERIMENT else [0.0]
        CONDITION_DROPOUT_VALUES = [0.0, 0.10] if FULL_EXPERIMENT else [0.0]
        POSTERIOR_DROPOUT = 0.0  # Disabled: it adds noise to q and needs separate justification.

        W_LATENT = 1.0
        W_IMAGE = 1.0
        W_FG = 0.0
        W_COS = 0.0
        K_VALUES = [8, 16]
        INCLUDE_K32_FOR_SELECTED = False
        if INCLUDE_K32_FOR_SELECTED:
            K_VALUES.append(32)
        VALIDATION_K = 8
        ACTIVE_KL_THRESHOLD = 0.01
        CLUSTER_DISTANCE_THRESHOLD = 4.0
        SEMANTIC_MAX_EXAMPLES = 512
        SEMANTIC_NEIGHBORS = 10
        SAVE_EXAMPLES = 6
        MAX_TRAIN_BATCHES = 2 if SMOKE_MODE else None
        MAX_EVAL_BATCHES = 2 if SMOKE_MODE else None
        RUN_SHUFFLED_CONTROL = True
        RUN_POSTERIOR_RECONSTRUCTION_CONTROL = True
        CHECKPOINT_SELECTORS = [
            "mean_top5_dice", "expected_one_sample_top5_dice",
            "semantic_normalized_auc", "accuracy_diversity_pareto", "last",
        ]

        DRIVE_OUTPUT_BASE = Path(
            "/content/drive/MyDrive/neurovlm/stage4_probabilistic_latent_generation"
            if IN_COLAB else REPO_DIR / "runs" / "stage4_probabilistic_latent_generation"
        )
        RESUME_EXPERIMENT_DIR = None
        AUTO_RESUME_ACTIVE = True
        """
    ),
    markdown("## 3. Imports, deterministic seeds, precision, and environment"),
    code(
        """
        import copy, csv, hashlib, importlib.metadata, itertools, json, math
        import platform, random, tempfile, time
        from dataclasses import asdict
        from datetime import datetime, timezone

        import numpy as np
        import pandas as pd
        import torch
        import torch.nn.functional as F
        from matplotlib import pyplot as plt
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.decomposition import PCA
        from torch import nn
        from torch.utils.data import DataLoader, Dataset

        from neurovlm import retrieval_resources as rr
        from neurovlm.atlas_free_dataset import AtlasFreeCNNDataProvider
        from neurovlm.atlas_free_text import AtlasFreeContrastiveCollator, AtlasFreeTextEmbeddingLookup
        from neurovlm.cnn import GenerativeTextToAELatent, autoencoder_from_payload
        from neurovlm.evaluation.spatial import reconstruction_metrics
        from neurovlm.evaluation.text_to_brain_audit import audit_text_preprocessing, autoencoder_identity
        from neurovlm.experiments.stage4_latent_ablation import (
            encode_stage1_latents, resolve_amp_dtype, split_fingerprint, text_cache_identity,
        )
        from neurovlm.experiments.stage4_probabilistic import (
            ConditionalLatentVAE, ConditionalVAEConfig, LatentStandardization,
            annealed_beta, architecture_record, checkpoint_payload, compute_cvae_loss,
            gather_samples, load_checkpoint, medoid_indices, pairwise_sample_distances,
            posterior_diagnostics, sample_interval_coverage, validate_provenance,
        )
        from neurovlm.pipelines import (
            atomic_write_csv, atomic_write_json, environment_provenance,
            git_provenance, sha256_file, sha256_state_dict, sha256_value,
        )
        from neurovlm.semantic_evaluation import evaluate_semantic_neighbor_retrieval
        from neurovlm.training.text_to_brain import (
            _autoencoder_state_provenance, _text_cache_provenance,
            _validate_recorded_autoencoder_state, _validate_recorded_text_cache,
        )

        def seed_everything(seed):
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        seed_everything(SEED)
        torch.set_float32_matmul_precision("high")
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        MIXED_PRECISION_DTYPE = resolve_amp_dtype(DEVICE, AMP_DTYPE)
        AMP_ENABLED = DEVICE.type == "cuda" and MIXED_PRECISION_DTYPE != torch.float32
        ENVIRONMENT = {
            **environment_provenance(["torch", "numpy", "pandas", "scikit-learn", "pyarrow"]),
            "python": sys.version, "platform": platform.platform(),
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "device": str(DEVICE),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
            "mixed_precision_dtype": str(MIXED_PRECISION_DTYPE),
            "git": git_provenance(REPO_DIR), "resolved_commit": RESOLVED_COMMIT,
        }
        print(json.dumps(ENVIRONMENT, indent=2, default=str))
        """
    ),
    markdown("## 4. Branch resources, exact provenance, and immutable data"),
    code(
        """
        BRANCH_SPECS = {
            "mixed_to_pubmed": {"domain": "pubmed", "variant": "mixed_baseline", "stage1": "1A", "ae_variant": "mixed"},
            "mixed_to_nilearn": {"domain": "nilearn", "variant": "mixed_baseline", "stage1": "1A", "ae_variant": "mixed"},
            "mixed_to_neurovault": {"domain": "neurovault", "variant": "mixed_baseline", "stage1": "1A", "ae_variant": "mixed"},
        }
        for name, spec in BRANCH_SPECS.items(): spec["branch"] = name
        unknown = set(BRANCHES_TO_RUN) - set(BRANCH_SPECS)
        if unknown: raise ValueError(f"Unknown branches: {sorted(unknown)}")

        def utc_stamp():
            return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        def atomic_torch_save(path, payload):
            path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
            )
            os.close(descriptor); temporary = Path(temporary_name)
            try:
                torch.save(payload, temporary)
                os.replace(temporary, path)
            except BaseException:
                temporary.unlink(missing_ok=True); raise

        def experiment_root():
            DRIVE_OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
            pointer = DRIVE_OUTPUT_BASE / "ACTIVE_EXPERIMENT.json"
            if RESUME_EXPERIMENT_DIR:
                root = Path(RESUME_EXPERIMENT_DIR)
            elif AUTO_RESUME_ACTIVE and pointer.exists():
                saved = json.loads(pointer.read_text())
                candidate = Path(saved["path"])
                root = candidate if saved.get("state") != "completed" and candidate.exists() else DRIVE_OUTPUT_BASE / utc_stamp()
            else:
                root = DRIVE_OUTPUT_BASE / utc_stamp()
            root.mkdir(parents=True, exist_ok=True)
            atomic_write_json(pointer, {"path": str(root), "state": "running", "updated_at": utc_stamp()})
            return root, pointer

        def freeze(module):
            module.eval()
            for parameter in module.parameters(): parameter.requires_grad_(False)
            return module

        def loader(dataset, lookup, *, batch_size, shuffle, seed):
            lookup.validate_dataset(dataset.rows)
            return DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle, num_workers=NUM_WORKERS,
                collate_fn=AtlasFreeContrastiveCollator(lookup, (36, 45, 38)),
                pin_memory=DEVICE.type == "cuda", persistent_workers=NUM_WORKERS > 0,
                generator=torch.Generator().manual_seed(seed),
                **({"prefetch_factor": PREFETCH_FACTOR} if NUM_WORKERS > 0 else {}),
            )

        class ShuffledTargetDataset(Dataset):
            def __init__(self, dataset, seed):
                self.dataset = dataset
                self.rows = dataset.rows
                self.permutation = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed)).tolist()
            def __len__(self): return len(self.dataset)
            def __getitem__(self, index):
                condition_item = dict(self.dataset[index])
                target_item = self.dataset[self.permutation[index]]
                condition_item["volume"] = target_item["volume"]
                # The stock collator preserves dataset_index, so it carries the
                # shuffled target identity while positive_texts remains the
                # original condition.
                condition_item["dataset_index"] = self.permutation[index]
                return condition_item

        def load_resources(branch):
            spec = BRANCH_SPECS[branch]
            ae_name = rr.CNN_AUTOENCODER_FILENAMES[spec["ae_variant"]]
            ae_path = Path(rr._download_from_hf(rr.ATLAS_FREE_CNN_MODEL_REPO_ID, ae_name, repo_type="model"))
            ae = freeze(autoencoder_from_payload(torch.load(ae_path, map_location="cpu", weights_only=True)))
            provider = AtlasFreeCNNDataProvider(domain=spec["domain"], limit=FULL_DATA_LIMIT)
            released = freeze(rr._load_cnn_text_to_brain(branch))
            semantic = freeze(rr._load_cnn_contrastive(branch))
            stage4_name = rr.CNN_T2B_FILENAMES[branch]
            stage4_path = Path(rr._download_from_hf(rr.ATLAS_FREE_CNN_MODEL_REPO_ID, stage4_name, repo_type="model"))
            return spec, ae_path, ae, provider, released, semantic, stage4_path

        def build_provenance(spec, ae_path, ae, provider, lookup, stage4_path):
            ae_source = _autoencoder_state_provenance({
                "kind": "released", "path": str(ae_path.resolve()), "sha256": sha256_file(ae_path),
                "branch": spec["branch"], "domain": spec["domain"], "stage1": spec["stage1"],
                "variant": spec["variant"], "loader_variant": spec["ae_variant"],
            }, ae)
            cache = _text_cache_provenance(lookup)
            _validate_recorded_autoencoder_state(ae_source, ae)
            _validate_recorded_text_cache(cache, _text_cache_provenance(lookup))
            audit = audit_text_preprocessing(lookup)
            if not audit["passed"]: raise RuntimeError(f"Text-cache audit failed: {audit}")
            splits = {name: split_fingerprint(getattr(provider, name)) for name in ("train", "val", "test")}
            split_ids = [set(row["map_id"] for row in getattr(provider, name).rows) for name in ("train", "val", "test")]
            if any(split_ids[i] & split_ids[j] for i in range(3) for j in range(i + 1, 3)):
                raise RuntimeError("Immutable split map IDs overlap")
            return {
                "autoencoder": ae_source,
                "autoencoder_identity": autoencoder_identity(
                    ae, checkpoint=ae_path, domain=spec["domain"], branch=spec["branch"]
                ),
                "released_stage4": {"path": str(stage4_path), "sha256": sha256_file(stage4_path)},
                "text_cache": {**cache, **text_cache_identity(lookup)},
                "text_preprocessing_audit": audit, "splits": splits,
                "branch": dict(spec), "git_commit": RESOLVED_COMMIT,
                "test_used_for_selection": False,
            }

        EXPERIMENT_ROOT, ACTIVE_POINTER = experiment_root()
        EFFECTIVE_CONFIG = {
            name: value for name, value in globals().items()
            if name in {
                "SMOKE_MODE", "FULL_EXPERIMENT", "BRANCHES_TO_RUN", "FULL_DATA_LIMIT",
                "EPOCHS", "BATCH_SIZE", "EVAL_BATCH_SIZE", "NUM_WORKERS",
                "PREFETCH_FACTOR", "SEED", "LEARNING_RATE",
                "WEIGHT_DECAY", "GRADIENT_CLIP", "AMP_DTYPE", "U_DIMS", "BETA_VALUES",
                "KL_SCHEDULES", "FREE_BITS_VALUES", "CONDITION_DROPOUT_VALUES",
                "W_LATENT", "W_IMAGE", "W_FG", "W_COS", "K_VALUES",
                "CHECKPOINT_SELECTORS", "RUN_SHUFFLED_CONTROL",
            }
        }
        atomic_write_json(EXPERIMENT_ROOT / "effective_config.json", EFFECTIVE_CONFIG)
        atomic_write_json(EXPERIMENT_ROOT / "environment.json", ENVIRONMENT)
        print("Resume-safe output:", EXPERIMENT_ROOT)
        """
    ),
    markdown("## 5. Train-only latent targets, standardization, and run definitions"),
    code(
        """
        def load_or_encode_latents(branch_dir, provenance, ae, dataset, lookup, split):
            path = branch_dir / f"{split}_raw_stage1_latents.pt"
            fingerprint = provenance["splits"][split]["ordered_rows_sha256"]
            if path.exists():
                saved = torch.load(path, map_location="cpu", weights_only=True)
                if saved["split_sha256"] != fingerprint:
                    raise ValueError("Cached latent split mismatch")
                if saved["encoder_state_sha256"] != provenance["autoencoder"]["encoder_state_sha256"]:
                    raise ValueError("Cached latent encoder mismatch")
                return saved["latents"]
            values = encode_stage1_latents(
                ae, dataset, lookup, device=DEVICE, batch_size=EVAL_BATCH_SIZE,
                num_workers=NUM_WORKERS,
            )
            atomic_torch_save(path, {
                "latents": values, "split_sha256": fingerprint,
                "encoder_state_sha256": provenance["autoencoder"]["encoder_state_sha256"],
            })
            return values

        def cvae_run_specs():
            specs = []
            for u_dim, beta, schedule in itertools.product(U_DIMS, BETA_VALUES, KL_SCHEDULES):
                specs.append({
                    "name": f"paired_u{u_dim}_b{beta:g}_{schedule}",
                    "u_dim": u_dim, "beta_max": beta, "kl_schedule": schedule,
                    "free_bits": 0.0, "condition_dropout": 0.0, "pairing": "matched",
                })
            if FULL_EXPERIMENT:
                for free_bits in FREE_BITS_VALUES[1:]:
                    specs.append({
                        "name": f"freebits_{free_bits:g}", "u_dim": 64, "beta_max": 0.01,
                        "kl_schedule": "linear", "free_bits": free_bits,
                        "condition_dropout": 0.0, "pairing": "matched",
                    })
                for dropout in CONDITION_DROPOUT_VALUES[1:]:
                    specs.append({
                        "name": f"condition_dropout_{dropout:g}", "u_dim": 64,
                        "beta_max": 0.01, "kl_schedule": "linear", "free_bits": 0.0,
                        "condition_dropout": dropout, "pairing": "matched",
                    })
            if RUN_SHUFFLED_CONTROL:
                specs.append({
                    "name": "shuffled_control", "u_dim": 64, "beta_max": 0.01,
                    "kl_schedule": "linear", "free_bits": 0.0,
                    "condition_dropout": 0.0, "pairing": "shuffled",
                })
            return specs

        RUN_SPECS = cvae_run_specs()
        display(pd.DataFrame(RUN_SPECS))
        """
    ),
    markdown(
        """
        ## 6. Metrics, diversity, calibration, and semantic retrieval

        `expected_one_sample_*` is the mean across exchangeable prior samples.
        Oracle metrics select a potentially different sample for each metric and
        are always labeled diagnostic. They are excluded from checkpoint and
        model selection.
        """
    ),
    code(
        """
        def per_item_metrics(prediction, target, predicted_latent, target_latent):
            pred = torch.nan_to_num(prediction.float().flatten(1), nan=0, posinf=1, neginf=0).clamp(0, 1)
            truth = torch.nan_to_num(target.float().flatten(1), nan=0, posinf=1, neginf=0).clamp(0, 1)
            voxels = pred.shape[1]; k = max(1, math.ceil(0.05 * voxels))
            pm = torch.zeros_like(pred, dtype=torch.bool).scatter(1, pred.topk(k, dim=1).indices, True)
            tm = torch.zeros_like(truth, dtype=torch.bool).scatter(1, truth.topk(k, dim=1).indices, True)
            dice = 2 * (pm & tm).sum(1).float() / (pm.sum(1) + tm.sum(1)).clamp_min(1)
            pc, tc = pred - pred.mean(1, keepdim=True), truth - truth.mean(1, keepdim=True)
            corr = (pc * tc).sum(1) / (pc.norm(dim=1) * tc.norm(dim=1)).clamp_min(1e-8)
            fg = truth > 0
            fg_mse = ((pred - truth).square() * fg).sum(1) / fg.sum(1).clamp_min(1)
            return {
                "top5_dice": dice, "spatial_corr": corr, "foreground_mse": fg_mse,
                "latent_mse": (predicted_latent.float() - target_latent.float()).square().mean(1),
            }

        def top5_support(flat_volume):
            flat = flat_volume.float().flatten(-3)
            k = max(1, math.ceil(0.05 * flat.shape[-1]))
            return torch.zeros_like(flat, dtype=torch.bool).scatter(-1, flat.topk(k, dim=-1).indices, True)

        def diversity_statistics(latent_samples, volume_samples):
            latent_dist = pairwise_sample_distances(latent_samples)
            flat = volume_samples.float().flatten(2)
            voxel_dist = torch.cdist(flat, flat) / math.sqrt(flat.shape[-1])
            support = top5_support(volume_samples)
            inter = (support[:, :, None] & support[:, None, :]).sum(-1).float()
            support_dice = 2 * inter / (
                support.sum(-1)[:, :, None] + support.sum(-1)[:, None, :]
            ).clamp_min(1)
            mask = torch.triu(torch.ones(latent_samples.shape[1], latent_samples.shape[1], dtype=torch.bool, device=latent_samples.device), diagonal=1)
            counts = []
            for value in latent_samples.detach().float().cpu():
                labels = AgglomerativeClustering(
                    n_clusters=None, distance_threshold=CLUSTER_DISTANCE_THRESHOLD,
                    linkage="average",
                ).fit_predict(value.numpy())
                counts.append(len(np.unique(labels)))
            centered = latent_samples.float() - latent_samples.float().mean(1, keepdim=True)
            covariance_trace = centered.square().sum(-1).mean(1)
            return {
                "pairwise_latent_distance": latent_dist[:, mask].mean(1),
                "pairwise_voxel_distance": voxel_dist[:, mask].mean(1),
                "pairwise_top5_support_dice": support_dice[:, mask].mean(1),
                "voxelwise_predictive_variance": flat.var(1, unbiased=False).mean(1),
                "latent_covariance_trace": covariance_trace,
                "distinct_activation_modes": torch.tensor(counts, device=latent_samples.device),
                "latent_distance_matrix": latent_dist,
            }

        def semantic_auc(semantic_model, volumes, text, raw_text, ids):
            if len(ids) < 2: return float("nan")
            safe = torch.nan_to_num(volumes.float(), nan=0, posinf=1, neginf=0).clamp(0, 1)
            brain = semantic_model.encode_brain(safe)
            projected_text = semantic_model.encode_text(text)
            neighbors = max(0, min(SEMANTIC_NEIGHBORS, len(ids) - 2))
            metrics, _ = evaluate_semantic_neighbor_retrieval(
                brain.float().cpu(), projected_text.float().cpu(), ids,
                neighbor_text_embeddings=raw_text.float().cpu(), n_neighbors=neighbors,
            )
            return float(metrics["semantic_normalized_k_recall_curve_auc"])

        def uncertainty_error_correlation(uncertainty, error):
            x, y = uncertainty.float(), error.float()
            x, y = x - x.mean(), y - y.mean()
            return float((x * y).sum() / (x.norm() * y.norm()).clamp_min(1e-8))
        """
    ),
    markdown("## 7. Validation evaluator and baseline comparison strategies A–G"),
    code(
        """
        @torch.no_grad()
        def evaluate_generator(
            model, standardization, ae, released, retrained, semantic_model,
            dataset, lookup, *, split, k, sample_seed, save_dir=None,
        ):
            model.eval(); ae.eval(); released.eval(); retrained.eval(); semantic_model.eval()
            rows, posterior_rows, semantic_buffers, diagnostics = [], [], {}, []
            saved = 0
            for batch_index, batch in enumerate(loader(
                dataset, lookup, batch_size=EVAL_BATCH_SIZE, shuffle=False, seed=SEED
            )):
                if MAX_EVAL_BATCHES is not None and batch_index >= MAX_EVAL_BATCHES: break
                target = batch["volume"].to(DEVICE)
                text = F.normalize(batch["text_embedding"].to(DEVICE), dim=-1)
                target_raw = ae.encoder(target)
                target_std = standardization.transform(target_raw)
                released_raw = released.text_projection(text)
                retrained_std = retrained(text)
                retrained_raw = standardization.inverse(retrained_std)
                mean_std = model.mean_path(text, mode="zero")
                samples_std = model.sample_prior(text, k=k, seed=sample_seed + batch_index)
                samples_raw = standardization.inverse(samples_std)
                shape = target.shape[1:]
                sample_volumes = ae.decoder(samples_raw.flatten(0, 1)).reshape(len(target), k, *shape)
                mean_raw = standardization.inverse(mean_std)
                base_latents = {
                    "released_deterministic": released_raw,
                    "retrained_deterministic": retrained_raw,
                    "cvae_mean": mean_raw,
                    "one_random_sample": samples_raw[:, 0],
                    "average_of_k": samples_raw.mean(1),
                    "consensus_medoid": gather_samples(samples_raw, medoid_indices(samples_std)),
                }
                base_volumes = {
                    "released_deterministic": released(text),
                    "retrained_deterministic": ae.decoder(retrained_raw),
                    "cvae_mean": ae.decoder(mean_raw),
                    "one_random_sample": sample_volumes[:, 0],
                    "average_of_k": ae.decoder(base_latents["average_of_k"]),
                    "consensus_medoid": ae.decoder(base_latents["consensus_medoid"]),
                }
                metrics_by_strategy = {
                    name: per_item_metrics(base_volumes[name], target, latent, target_raw)
                    for name, latent in base_latents.items()
                }
                sample_metric = per_item_metrics(
                    sample_volumes.flatten(0, 1),
                    target[:, None].expand(-1, k, *shape).flatten(0, 1),
                    samples_raw.flatten(0, 1),
                    target_raw[:, None].expand(-1, k, -1).flatten(0, 1),
                )
                sample_metric = {name: value.reshape(len(target), k) for name, value in sample_metric.items()}
                diversity = diversity_statistics(samples_std, sample_volumes)
                coverage = sample_interval_coverage(samples_std, target_std)
                target_error = sample_metric["latent_mse"].mean(1)
                uncertainty = samples_std.var(1, unbiased=False).mean(-1)
                for i, map_id in enumerate(batch["map_id"]):
                    common = {"split": split, "map_id": str(map_id), "text_id": str(batch["text_id"][i]), "source": str(batch["source"][i]), "k": k}
                    for strategy, values in metrics_by_strategy.items():
                        rows.append({**common, "strategy": strategy, **{n: float(v[i]) for n, v in values.items()}})
                    for j in range(k):
                        rows.append({**common, "strategy": "cvae_sample", "sample_index": j, **{n: float(v[i, j]) for n, v in sample_metric.items()}})
                    rows.append({
                        **common, "strategy": "expected_one_sample",
                        **{n: float(v[i].mean()) for n, v in sample_metric.items()},
                        **{n: float(v[i]) for n, v in diversity.items() if n != "latent_distance_matrix"},
                        **coverage, "predictive_uncertainty": float(uncertainty[i]),
                        "expected_generation_error": float(target_error[i]),
                        "prob_any_sample_beats_released_top5": float(
                            (sample_metric["top5_dice"][i] > metrics_by_strategy["released_deterministic"]["top5_dice"][i]).any()
                        ),
                        "oracle_best_top5_dice_diagnostic_only": float(sample_metric["top5_dice"][i].max()),
                        "oracle_best_spatial_corr_diagnostic_only": float(sample_metric["spatial_corr"][i].max()),
                        "oracle_best_latent_mse_diagnostic_only": float(sample_metric["latent_mse"][i].min()),
                    })
                # Cohort semantic buffers are bounded and evaluated per strategy.
                take = min(len(target), max(0, SEMANTIC_MAX_EXAMPLES - len(semantic_buffers.get("ids", []))))
                if take:
                    semantic_buffers.setdefault("ids", []).extend(str(x) for x in batch["map_id"][:take])
                    semantic_buffers.setdefault("text", []).append(text[:take])
                    semantic_buffers.setdefault("raw_text", []).append(text[:take])
                    for name, volume in base_volumes.items():
                        semantic_buffers.setdefault(name, []).append(volume[:take])
                mu, logvar = model.encode_posterior(text, target_std)
                diagnostics.append((mu.cpu(), logvar.cpu()))
                if RUN_POSTERIOR_RECONSTRUCTION_CONTROL:
                    posterior = model.sample_posterior(text, target_std, k=k, seed=sample_seed + batch_index)
                    posterior_raw = standardization.inverse(posterior)
                    posterior_volume = ae.decoder(posterior_raw.flatten(0, 1)).reshape(len(target), k, *shape)
                    pm = per_item_metrics(
                        posterior_volume.flatten(0, 1),
                        target[:, None].expand(-1, k, *shape).flatten(0, 1),
                        posterior_raw.flatten(0, 1),
                        target_raw[:, None].expand(-1, k, -1).flatten(0, 1),
                    )
                    posterior_rows.extend({
                        "split": split, "map_id": str(batch["map_id"][i]), "k": k,
                        "strategy": "posterior_reconstruction_sample", "sample_index": j,
                        **{name: float(value.reshape(len(target), k)[i, j]) for name, value in pm.items()},
                    } for i in range(len(target)) for j in range(k))
                if save_dir is not None and saved < SAVE_EXAMPLES:
                    save_dir.mkdir(parents=True, exist_ok=True)
                    uncertainty_dir = save_dir.parents[1] / "uncertainty_volumes" / f"k{k}"
                    uncertainty_dir.mkdir(parents=True, exist_ok=True)
                    for i in range(min(len(target), SAVE_EXAMPLES - saved)):
                        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(batch["map_id"][i]))
                        uncertainty_volume = sample_volumes[i].float().var(0, unbiased=False).cpu()
                        atomic_torch_save(save_dir / f"{saved:03d}_{safe_id}.pt", {
                            "map_id": str(batch["map_id"][i]), "target": target[i].cpu(),
                            "released": base_volumes["released_deterministic"][i].cpu(),
                            "cvae_mean": base_volumes["cvae_mean"][i].cpu(),
                            "samples": sample_volumes[i].cpu(),
                            "consensus": base_volumes["consensus_medoid"][i].cpu(),
                            "uncertainty": uncertainty_volume,
                            "latent_samples": samples_raw[i].cpu(),
                            "latent_covariance": torch.cov(samples_raw[i].float().T).cpu(),
                        })
                        atomic_torch_save(
                            uncertainty_dir / f"{saved:03d}_{safe_id}_predictive_variance.pt",
                            uncertainty_volume,
                        )
                        saved += 1
            frame = pd.DataFrame(rows)
            summary = frame.groupby("strategy", dropna=False).mean(numeric_only=True).reset_index()
            ids = semantic_buffers.get("ids", [])
            if ids:
                text_all = torch.cat(semantic_buffers["text"]).to(DEVICE)
                raw_all = torch.cat(semantic_buffers["raw_text"])
                semantic_values = {}
                for name in base_volumes:
                    semantic_values[name] = semantic_auc(
                        semantic_model, torch.cat(semantic_buffers[name]).to(DEVICE),
                        text_all, raw_all, ids,
                    )
                summary["semantic_normalized_auc"] = summary["strategy"].map(semantic_values)
            mus, logvars = zip(*diagnostics, strict=True)
            posterior = posterior_diagnostics(torch.cat(mus), torch.cat(logvars), active_threshold=ACTIVE_KL_THRESHOLD)
            expected = frame[frame.strategy.eq("expected_one_sample")]
            posterior["uncertainty_error_correlation"] = uncertainty_error_correlation(
                torch.tensor(expected.predictive_uncertainty.to_numpy()),
                torch.tensor(expected.expected_generation_error.to_numpy()),
            )
            return frame, summary, posterior, pd.DataFrame(posterior_rows)
        """
    ),
    markdown("## 8. Resume-safe deterministic and CVAE training"),
    code(
        """
        def deterministic_binding(provenance, standardization):
            return sha256_value({
                "autoencoder": provenance["autoencoder"],
                "text_cache": provenance["text_cache"],
                "splits": provenance["splits"],
                "branch": provenance["branch"],
                "latent_standardization": standardization.metadata(),
                "model": "retrained_deterministic_standardized_target",
            })

        def train_retrained_deterministic(
            run_dir, provenance, ae, provider, lookup, train_latents, standardization
        ):
            checkpoint = run_dir / "retrained_deterministic.pt"
            model = GenerativeTextToAELatent(768, 512, 384).to(DEVICE)
            if checkpoint.exists():
                saved = torch.load(checkpoint, map_location=DEVICE, weights_only=True)
                if saved["binding_sha256"] != deterministic_binding(provenance, standardization):
                    raise ValueError("Re-trained deterministic checkpoint provenance mismatch")
                model.load_state_dict(saved["state_dict"], strict=True)
                if saved["state_sha256"] != sha256_state_dict(model):
                    raise ValueError("Re-trained deterministic checkpoint state mismatch")
                return model
            optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            for epoch in range(EPOCHS):
                model.train()
                for batch_index, batch in enumerate(loader(provider.train, lookup, batch_size=BATCH_SIZE, shuffle=True, seed=SEED + epoch)):
                    if MAX_TRAIN_BATCHES is not None and batch_index >= MAX_TRAIN_BATCHES: break
                    indices = torch.as_tensor(batch["dataset_index"], dtype=torch.long)
                    text = F.normalize(batch["text_embedding"].to(DEVICE), dim=-1)
                    target = batch["volume"].to(DEVICE)
                    target_raw = train_latents.index_select(0, indices).to(DEVICE)
                    target_std = standardization.transform(target_raw)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(DEVICE.type, dtype=MIXED_PRECISION_DTYPE, enabled=AMP_ENABLED):
                        pred_std = model(text)
                        pred_volume = ae.decoder(standardization.inverse(pred_std))
                        loss = W_LATENT * F.mse_loss(pred_std.float(), target_std.float()) + W_IMAGE * F.mse_loss(pred_volume.float(), target.float())
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                    optimizer.step()
            atomic_torch_save(checkpoint, {
                "state_dict": model.state_dict(),
                "state_sha256": sha256_state_dict(model),
                "binding_sha256": deterministic_binding(provenance, standardization),
            })
            return model

        def selector_values(summary, diagnostics):
            by_name = summary.set_index("strategy")
            return {
                "mean_top5_dice": float(by_name.loc["cvae_mean", "top5_dice"]),
                "expected_one_sample_top5_dice": float(by_name.loc["expected_one_sample", "top5_dice"]),
                "semantic_normalized_auc": float(by_name.loc["cvae_mean", "semantic_normalized_auc"]),
                "accuracy_diversity_pareto": (
                    float(by_name.loc["expected_one_sample", "top5_dice"]),
                    float(by_name.loc["expected_one_sample", "pairwise_latent_distance"]),
                ),
            }

        def dominates(left, right):
            return left[0] >= right[0] and left[1] >= right[1] and left != right

        def train_cvae(run_dir, run_spec, provenance, ae, provider, lookup, train_latents, standardization, released, retrained, semantic):
            run_dir.mkdir(parents=True, exist_ok=True)
            effective = {**run_spec, "epochs": EPOCHS, "weights": {
                "latent": W_LATENT, "image": W_IMAGE, "foreground": W_FG, "cosine": W_COS,
            }, "test_used_for_selection": False}
            bound_provenance = {
                **provenance,
                "run_identity": effective,
                "latent_standardization": standardization.metadata(),
            }
            atomic_write_json(run_dir / "effective_config.json", effective)
            atomic_write_json(run_dir / "provenance.json", bound_provenance)
            config = ConditionalVAEConfig(
                u_dim=run_spec["u_dim"], condition_dropout=run_spec["condition_dropout"],
                posterior_dropout=POSTERIOR_DROPOUT,
            )
            model = ConditionalLatentVAE(config).to(DEVICE)
            atomic_write_json(run_dir / "model_architecture.json", architecture_record(model))
            optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            history_path = run_dir / "training_history.csv"
            diagnostics_path = run_dir / "posterior_diagnostics.csv"
            history = pd.read_csv(history_path).to_dict("records") if history_path.exists() else []
            posterior_history = pd.read_csv(diagnostics_path).to_dict("records") if diagnostics_path.exists() else []
            manifest_path = run_dir / "checkpoint_manifest.json"
            manifest = (
                json.loads(manifest_path.read_text())
                if manifest_path.exists()
                else {"format_version": 1, "selectors": {}, "pareto_front": [], "test_used_for_selection": False}
            )
            start_epoch = 0; global_step = 0
            last_path = run_dir / "checkpoints" / "last.pt"
            if last_path.exists():
                model, loaded_standardization, payload = load_checkpoint(
                    last_path, expected_provenance=bound_provenance, device=DEVICE
                )
                if sha256_state_dict(loaded_standardization) != sha256_state_dict(standardization):
                    raise ValueError("Resume standardization mismatch")
                optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
                optimizer.load_state_dict(payload["optimizer_state_dict"])
                start_epoch, global_step = payload["epoch"] + 1, payload["global_step"]
            training_dataset = provider.train if run_spec["pairing"] == "matched" else ShuffledTargetDataset(provider.train, SEED)
            best = {
                selector: float(manifest["selectors"].get(selector, {}).get("value", -math.inf))
                for selector in (
                    "mean_top5_dice", "expected_one_sample_top5_dice",
                    "semantic_normalized_auc",
                )
            }
            pareto = manifest.get("pareto_front", [])
            total_steps = max(1, EPOCHS * math.ceil(len(training_dataset) / BATCH_SIZE))
            warmup_steps = int(KL_WARMUP_FRACTION * total_steps)
            for epoch in range(start_epoch, EPOCHS):
                model.train(); totals = {}; n = 0
                mus, logvars = [], []
                for batch_index, batch in enumerate(loader(training_dataset, lookup, batch_size=BATCH_SIZE, shuffle=True, seed=SEED + epoch)):
                    if MAX_TRAIN_BATCHES is not None and batch_index >= MAX_TRAIN_BATCHES: break
                    text = F.normalize(batch["text_embedding"].to(DEVICE), dim=-1)
                    target = batch["volume"].to(DEVICE)
                    indices = torch.as_tensor(batch["dataset_index"], dtype=torch.long)
                    target_raw = train_latents.index_select(0, indices).to(DEVICE)
                    target_std = standardization.transform(target_raw)
                    beta = annealed_beta(
                        global_step, beta_max=run_spec["beta_max"], warmup_steps=warmup_steps,
                        schedule=run_spec["kl_schedule"],
                        cycle_steps=KL_CYCLE_EPOCHS * max(1, math.ceil(len(training_dataset) / BATCH_SIZE)),
                    )
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(DEVICE.type, dtype=MIXED_PRECISION_DTYPE, enabled=AMP_ENABLED):
                        output = model(text, target_std)
                        loss = compute_cvae_loss(
                            output, target_std, target, standardization=standardization,
                            decoder=ae.decoder, beta=beta, free_bits_per_dim=run_spec["free_bits"],
                            w_latent=W_LATENT, w_image=W_IMAGE, w_fg=W_FG, w_cos=W_COS,
                        )
                    loss.total.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                    optimizer.step(); global_step += 1
                    values = {
                        "loss": loss.total, "standardized_latent_mse": loss.standardized_latent_mse,
                        "true_kl": loss.true_kl, "optimized_kl": loss.optimized_kl,
                        "decoded_volume_mse": loss.decoded_volume_mse,
                        "foreground_mse": loss.foreground_mse,
                        "latent_cosine_loss": loss.latent_cosine_loss,
                    }
                    for name, value in values.items(): totals[name] = totals.get(name, 0) + float(value.detach()) * len(text)
                    n += len(text); mus.append(output["mu"].detach().cpu()); logvars.append(output["logvar"].detach().cpu())
                train_row = {"epoch": epoch, "global_step": global_step, "beta": beta, **{f"train_{k}": v / n for k, v in totals.items()}}
                diag = {"epoch": epoch, **posterior_diagnostics(torch.cat(mus), torch.cat(logvars), active_threshold=ACTIVE_KL_THRESHOLD)}
                diag.pop("per_dimension_kl"); diag.pop("per_dimension_posterior_mean_variance")
                val_frame, val_summary, val_diag, _ = evaluate_generator(
                    model, standardization, ae, released, retrained, semantic,
                    provider.val, lookup, split="val", k=VALIDATION_K, sample_seed=SEED + 10000 + epoch,
                )
                values = selector_values(val_summary, val_diag)
                train_row.update({f"val_{k}": v for k, v in values.items() if not isinstance(v, tuple)})
                train_row.update({f"val_{k}": float(v) for k, v in val_diag.items() if isinstance(v, (int, float))})
                history.append(train_row); posterior_history.append(diag)
                atomic_write_csv(history_path, history); atomic_write_csv(diagnostics_path, posterior_history)
                metrics = {**train_row, **{f"posterior_{k}": v for k, v in diag.items()}}
                payload = checkpoint_payload(
                    model, standardization=standardization, provenance=bound_provenance,
                    epoch=epoch, global_step=global_step, metrics=metrics, optimizer=optimizer,
                )
                atomic_torch_save(last_path, payload)
                manifest["selectors"]["last"] = {"path": str(last_path), "epoch": epoch, "sha256": sha256_file(last_path)}
                for selector in ("mean_top5_dice", "expected_one_sample_top5_dice", "semantic_normalized_auc"):
                    value = values[selector]
                    if math.isfinite(value) and value > best[selector]:
                        best[selector] = value
                        path = run_dir / "checkpoints" / f"best_validation_{selector}.pt"
                        atomic_torch_save(path, payload)
                        manifest["selectors"][selector] = {"path": str(path), "epoch": epoch, "value": value, "sha256": sha256_file(path)}
                point = values["accuracy_diversity_pareto"]
                pareto = [item for item in pareto if not dominates(point, item["point"])]
                if not any(dominates(item["point"], point) for item in pareto):
                    path = run_dir / "checkpoints" / f"pareto_epoch_{epoch:03d}.pt"
                    atomic_torch_save(path, payload); pareto.append({"point": point, "path": str(path), "epoch": epoch})
                # Deterministic knee rule, validation only.
                selected = max(pareto, key=lambda item: item["point"][0] + 0.01 * item["point"][1])
                manifest["selectors"]["accuracy_diversity_pareto"] = {
                    **selected, "selected_by": "validation Pareto front; fixed accuracy+0.01*diversity knee",
                    "oracle_used": False,
                }
                manifest["pareto_front"] = pareto
                atomic_write_json(manifest_path, manifest)
                atomic_write_csv(run_dir / "validation_metrics.csv", val_summary.to_dict("records"))
            selected_path = manifest["selectors"]["mean_top5_dice"]["path"]
            selected, selected_standardization, payload = load_checkpoint(
                selected_path, expected_provenance=bound_provenance, device=DEVICE
            )
            return selected, selected_standardization, manifest
        """
    ),
    markdown("## 9. Execute validation-only training and freeze selection"),
    code(
        """
        lookup = AtlasFreeTextEmbeddingLookup.published()
        validation_leaderboard = []
        selected_models = {}
        all_provenance = {}

        for branch in BRANCHES_TO_RUN:
            print(f"===== {branch} =====")
            branch_dir = EXPERIMENT_ROOT / branch; branch_dir.mkdir(parents=True, exist_ok=True)
            spec, ae_path, ae, provider, released, semantic, stage4_path = load_resources(branch)
            ae, released, semantic = ae.to(DEVICE), released.to(DEVICE), semantic.to(DEVICE)
            provenance = build_provenance(spec, ae_path, ae, provider, lookup, stage4_path)
            all_provenance[branch] = provenance
            atomic_write_json(branch_dir / "provenance.json", provenance)
            train_latents = load_or_encode_latents(branch_dir, provenance, ae, provider.train, lookup, "train")
            standardization = LatentStandardization.fit(train_latents, epsilon=LATENT_EPSILON).to(DEVICE)
            atomic_torch_save(branch_dir / "latent_standardization.pt", standardization.to_payload())
            retrained = train_retrained_deterministic(
                branch_dir, provenance, ae, provider, lookup, train_latents, standardization
            ).to(DEVICE)
            for run_spec in RUN_SPECS:
                run_dir = branch_dir / run_spec["name"]
                model, fitted, manifest = train_cvae(
                    run_dir, run_spec, provenance, ae, provider, lookup,
                    train_latents, standardization, released, retrained, semantic,
                )
                val_frame, val_summary, val_diag, posterior_frame = evaluate_generator(
                    model, fitted, ae, released, retrained, semantic, provider.val, lookup,
                    split="val", k=max(K_VALUES), sample_seed=SEED + 20000,
                )
                val_summary.insert(0, "branch", branch); val_summary.insert(1, "run", run_spec["name"])
                atomic_write_csv(run_dir / "validation_metrics.csv", val_summary.to_dict("records"))
                validation_leaderboard.extend(val_summary.to_dict("records"))
                selected_models[(branch, run_spec["name"])] = {
                    "model": model.cpu(), "standardization": fitted.cpu(),
                    "run_spec": run_spec, "run_dir": run_dir,
                }
            del ae, provider, released, semantic, retrained, train_latents
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        atomic_write_json(EXPERIMENT_ROOT / "provenance.json", all_provenance)
        atomic_write_csv(EXPERIMENT_ROOT / "validation_metrics.csv", validation_leaderboard)
        leaderboard = pd.DataFrame(validation_leaderboard)
        # Shuffled runs are controls and cannot win deployment selection.
        candidates = leaderboard[
            leaderboard.strategy.eq("cvae_mean") & ~leaderboard.run.str.contains("shuffled")
        ].sort_values(["branch", "top5_dice"], ascending=[True, False])
        selected_runs = candidates.groupby("branch", as_index=False).head(1)
        atomic_write_csv(EXPERIMENT_ROOT / "validation_selected_models.csv", selected_runs.to_dict("records"))
        display(selected_runs)
        """
    ),
    markdown(
        """
        ## 10. Held-out test evaluation

        The following cell is the first test access after validation selection
        has been written. It evaluates every requested strategy at K=8 and K=16
        (and optionally K=32) while keeping oracle results diagnostic-only.
        """
    ),
    code(
        """
        test_summaries, sample_rows, posterior_control_rows = [], [], []
        for selected in selected_runs.to_dict("records"):
            branch, run_name = selected["branch"], selected["run"]
            spec, ae_path, ae, provider, released, semantic, stage4_path = load_resources(branch)
            ae, released, semantic = ae.to(DEVICE), released.to(DEVICE), semantic.to(DEVICE)
            provenance = build_provenance(spec, ae_path, ae, provider, lookup, stage4_path)
            validate_provenance(all_provenance[branch], provenance)
            branch_dir = EXPERIMENT_ROOT / branch
            standardization = LatentStandardization.from_payload(
                torch.load(branch_dir / "latent_standardization.pt", map_location="cpu", weights_only=False)
            ).to(DEVICE)
            retrained = GenerativeTextToAELatent(768, 512, 384).to(DEVICE)
            retrained_payload = torch.load(
                branch_dir / "retrained_deterministic.pt", map_location=DEVICE, weights_only=True
            )
            if retrained_payload["binding_sha256"] != deterministic_binding(provenance, standardization):
                raise ValueError("Re-trained deterministic checkpoint provenance mismatch")
            retrained.load_state_dict(retrained_payload["state_dict"], strict=True)
            if retrained_payload["state_sha256"] != sha256_state_dict(retrained):
                raise ValueError("Re-trained deterministic checkpoint state mismatch")
            run_dir = branch_dir / run_name
            run_provenance = json.loads((run_dir / "provenance.json").read_text())
            base_from_run = {name: run_provenance[name] for name in (
                "autoencoder", "text_cache", "splits", "branch"
            )}
            validate_provenance(provenance, base_from_run)
            manifest = json.loads((run_dir / "checkpoint_manifest.json").read_text())
            checkpoint = manifest["selectors"]["mean_top5_dice"]
            if sha256_file(checkpoint["path"]) != checkpoint["sha256"]:
                raise ValueError("Selected checkpoint checksum mismatch")
            model, fitted, payload = load_checkpoint(
                checkpoint["path"], expected_provenance=run_provenance, device=DEVICE
            )
            for k in K_VALUES:
                frame, summary, diag, posterior_frame = evaluate_generator(
                    model, fitted, ae, released, retrained, semantic, provider.test, lookup,
                    split="test", k=k, sample_seed=SEED + 30000 + k,
                    save_dir=run_dir / "generated_samples" / f"k{k}",
                )
                frame.insert(0, "branch", branch); frame.insert(1, "run", run_name)
                summary.insert(0, "branch", branch); summary.insert(1, "run", run_name)
                summary["k"] = k
                summary["oracle_is_diagnostic_only"] = True
                for key, value in diag.items():
                    if isinstance(value, (int, float)): summary[f"posterior_{key}"] = value
                sample_rows.extend(frame.to_dict("records"))
                test_summaries.extend(summary.to_dict("records"))
                posterior_control_rows.extend(posterior_frame.assign(branch=branch, run=run_name).to_dict("records"))
            del ae, provider, released, semantic, retrained, model
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        sample_frame = pd.DataFrame(sample_rows)
        try:
            sample_frame.to_parquet(EXPERIMENT_ROOT / "sample_level_metrics.parquet", index=False)
        except Exception:
            sample_frame.to_csv(EXPERIMENT_ROOT / "sample_level_metrics.csv", index=False)
        atomic_write_csv(EXPERIMENT_ROOT / "test_metrics.csv", test_summaries)
        atomic_write_csv(EXPERIMENT_ROOT / "posterior_reconstruction_metrics.csv", posterior_control_rows)
        """
    ),
    markdown("## 11. Diversity/calibration curves and required visualizations"),
    code(
        """
        plots_dir = EXPERIMENT_ROOT / "plots"; plots_dir.mkdir(exist_ok=True)
        test_frame = pd.DataFrame(test_summaries)
        curve = test_frame[test_frame.strategy.eq("expected_one_sample")][[
            "branch", "run", "k", "top5_dice", "pairwise_latent_distance",
            "pairwise_voxel_distance", "distinct_activation_modes",
            "voxelwise_predictive_variance", "posterior_uncertainty_error_correlation",
        ]]
        atomic_write_csv(EXPERIMENT_ROOT / "accuracy_diversity_curve.csv", curve.to_dict("records"))

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for (branch, run), group in curve.groupby(["branch", "run"]):
            axes[0].plot(group.k, group.top5_dice, marker="o", label=f"{branch}/{run}")
            axes[1].plot(group.pairwise_latent_distance, group.top5_dice, marker="o")
            axes[2].plot(group.voxelwise_predictive_variance, group.top5_dice, marker="o")
        axes[0].set(xlabel="K", ylabel="Expected one-sample top-5 Dice", title="Accuracy by K")
        axes[1].set(xlabel="Pairwise latent distance", ylabel="Top-5 Dice", title="Accuracy–diversity")
        axes[2].set(xlabel="Predictive variance", ylabel="Top-5 Dice", title="Uncertainty–accuracy")
        axes[0].legend(fontsize=6)
        fig.tight_layout(); fig.savefig(plots_dir / "accuracy_versus_diversity.png", dpi=180); plt.close(fig)

        histories = []
        for path in EXPERIMENT_ROOT.glob("*/*/training_history.csv"):
            frame = pd.read_csv(path); frame["run_dir"] = str(path.parent); histories.append(frame)
        if histories:
            history = pd.concat(histories, ignore_index=True)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            for name, group in history.groupby("run_dir"):
                axes[0].plot(group.epoch, group.train_true_kl, alpha=.7)
                axes[1].plot(group.epoch, group.val_active_latent_dimensions, alpha=.7)
            axes[0].set(title="KL curve", xlabel="Epoch", ylabel="Mean KL")
            axes[1].set(title="Active stochastic dimensions", xlabel="Epoch", ylabel="Count")
            fig.tight_layout(); fig.savefig(plots_dir / "kl_and_active_dimension_curves.png", dpi=180); plt.close(fig)

        expected = sample_frame[sample_frame.strategy.eq("expected_one_sample")]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(expected.predictive_uncertainty, expected.expected_generation_error, alpha=.35)
        ax.set(xlabel="Predictive latent variance", ylabel="Expected latent MSE", title="Uncertainty versus error")
        fig.tight_layout(); fig.savefig(plots_dir / "uncertainty_versus_error.png", dpi=180); plt.close(fig)

        def middle_slice(volume):
            value = torch.as_tensor(volume).squeeze().float()
            return value[value.shape[0] // 2].T

        for artifact in EXPERIMENT_ROOT.glob("*/*/generated_samples/k*/*.pt"):
            payload = torch.load(artifact, map_location="cpu", weights_only=False)
            fig, axes = plt.subplots(2, 5, figsize=(16, 7))
            panels = [
                ("Reference", payload["target"]), ("Released deterministic", payload["released"]),
                ("CVAE mean", payload["cvae_mean"]), ("Consensus", payload["consensus"]),
                ("Uncertainty", payload["uncertainty"]),
            ] + [(f"Sample {i + 1}", value) for i, value in enumerate(payload["samples"][:5])]
            for ax, (title, value) in zip(axes.flat, panels, strict=True):
                ax.imshow(middle_slice(value), cmap="magma"); ax.set_title(title); ax.axis("off")
            fig.tight_layout(); fig.savefig(plots_dir / f"maps_{payload['map_id']}.png", dpi=160); plt.close(fig)

            distances = torch.cdist(payload["latent_samples"].float(), payload["latent_samples"].float())
            projection = PCA(n_components=2).fit_transform(payload["latent_samples"].numpy())
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].imshow(distances, cmap="viridis"); axes[0].set_title("Sample diversity matrix")
            axes[1].scatter(projection[:, 0], projection[:, 1], c=np.arange(len(projection)), cmap="tab20")
            axes[1].set_title("Latent-space sample projection")
            fig.tight_layout(); fig.savefig(plots_dir / f"diversity_projection_{payload['map_id']}.png", dpi=160); plt.close(fig)
            break
        """
    ),
    markdown("## 12. Final comparison and eight-question report"),
    code(
        """
        final = pd.DataFrame(test_summaries)
        final.to_csv(EXPERIMENT_ROOT / "final_comparison.csv", index=False)
        expected = final[final.strategy.eq("expected_one_sample")]
        released = final[final.strategy.eq("released_deterministic")]
        means = final[final.strategy.eq("cvae_mean")]
        shuffled = pd.DataFrame(validation_leaderboard)
        shuffled = shuffled[shuffled.run.str.contains("shuffled", na=False)]

        def mean_or_nan(frame, column):
            return float(frame[column].mean()) if len(frame) and column in frame else float("nan")

        released_top5 = mean_or_nan(released, "top5_dice")
        expected_top5 = mean_or_nan(expected, "top5_dice")
        oracle_top5 = mean_or_nan(expected, "oracle_best_top5_dice_diagnostic_only")
        diversity = mean_or_nan(expected, "pairwise_latent_distance")
        modes = mean_or_nan(expected, "distinct_activation_modes")
        uncertainty_error = mean_or_nan(expected, "posterior_uncertainty_error_correlation")
        observed_coverage = np.array([mean_or_nan(expected, f"coverage_{level}") for level in (50, 80, 90, 95)])
        nominal_coverage = np.array([.50, .80, .90, .95])
        calibration_mae = float(np.nanmean(np.abs(observed_coverage - nominal_coverage)))
        branch_pivot = final[
            final.strategy.isin(["released_deterministic", "expected_one_sample"])
        ].pivot_table(index="branch", columns="strategy", values="top5_dice", aggfunc="mean")
        if {"released_deterministic", "expected_one_sample"} <= set(branch_pivot.columns):
            branch_pivot["expected_delta_vs_released"] = (
                branch_pivot["expected_one_sample"] - branch_pivot["released_deterministic"]
            )
        sparse_dense = branch_pivot.reset_index().to_dict("records")
        answers = {
            "conditional_mean_collapse": bool(
                diversity > 0 and modes > 1 and oracle_top5 > expected_top5
            ),
            "expected_single_sample_improves": mean_or_nan(expected, "top5_dice") > mean_or_nan(released, "top5_dice"),
            "best_of_k_represents_target": mean_or_nan(expected, "oracle_best_top5_dice_diagnostic_only") > mean_or_nan(released, "top5_dice"),
            "diversity_meaningful_not_noise": bool(
                diversity > 0 and modes > 1 and oracle_top5 > released_top5
                and uncertainty_error > 0
            ),
            "uncertainty_calibrated": {
                "coverage_50": mean_or_nan(expected, "coverage_50"),
                "coverage_80": mean_or_nan(expected, "coverage_80"),
                "coverage_90": mean_or_nan(expected, "coverage_90"),
                "coverage_95": mean_or_nan(expected, "coverage_95"),
                "uncertainty_error_correlation": mean_or_nan(expected, "posterior_uncertainty_error_correlation"),
                "coverage_mean_absolute_error": calibration_mae,
                "calibrated_within_0.10": calibration_mae <= 0.10,
            },
            "sparse_vs_dense": sparse_dense,
            "posterior_collapse": {
                "active_dimensions": mean_or_nan(expected, "posterior_active_latent_dimensions"),
                "mean_kl_per_dimension": mean_or_nan(expected, "posterior_mean_kl_per_dimension"),
                "posterior_mean_variance": mean_or_nan(expected, "posterior_posterior_mean_variance"),
            },
            "complexity_justified": (
                mean_or_nan(expected, "top5_dice") > mean_or_nan(released, "top5_dice")
                and mean_or_nan(expected, "posterior_active_latent_dimensions") > 0
            ),
        }
        report = f'''# Stage 4 probabilistic latent generation report

        Exact commit: `{RESOLVED_COMMIT}`

        ## Guardrails
        The released deterministic projector and Stage 1 AE were never modified.
        All selection used validation. Test was evaluation-only. Oracle best-of-K
        is diagnostic and is not a deployable single-sample metric.

        1. **Does probabilistic generation reduce conditional-mean collapse?**
           `{answers["conditional_mean_collapse"]}`. Mean pairwise latent distance
           was `{diversity:.4g}`, mean mode count `{modes:.3g}`, and diagnostic
           oracle-minus-expected top-5 Dice `{oracle_top5 - expected_top5:.4g}`.
        2. **Does expected single-sample performance improve?**
           `{answers["expected_single_sample_improves"]}`.
        3. **Does best-of-K show target representation?**
           `{answers["best_of_k_represents_target"]}` (diagnostic only).
        4. **Is diversity meaningful or noise?**
           `{answers["diversity_meaningful_not_noise"]}` under the predeclared joint
           evidence rule (multiple modes, oracle gain, and positive uncertainty–error
           association). The shuffled validation control remains the causal negative control.
        5. **Is uncertainty calibrated?**
           `{json.dumps(answers["uncertainty_calibrated"])}`.
        6. **Do PubMed/Nilearn benefit more than dense NeuroVault?**
           Compare branch deltas in `final_comparison.csv`: `{json.dumps(answers["sparse_vs_dense"])}`.
        7. **Does posterior collapse occur?**
           `{json.dumps(answers["posterior_collapse"])}`. A run ignores `u` when KL,
           active dimensions, posterior-mean variance, and sample diversity approach zero.
        8. **Is complexity justified?**
           `{answers["complexity_justified"]}` under the predeclared joint requirement
           of improved expected accuracy and non-collapsed stochastic dimensions.

        The shuffled control must not show meaningful paired improvement. Posterior
        reconstruction samples are an inference diagnostic because they use the target;
        text-only prior samples are the deployable stochastic path.
        '''
        (EXPERIMENT_ROOT / "final_report.md").write_text(report)
        atomic_write_json(ACTIVE_POINTER, {
            "path": str(EXPERIMENT_ROOT), "state": "completed", "updated_at": utc_stamp()
        })
        print(report)
        """
    ),
    markdown(
        """
        ## Artifact checklist

        The run writes `effective_config.json`, `provenance.json`,
        `latent_standardization.pt`, `model_architecture.json`,
        `training_history.csv`, `posterior_diagnostics.csv`,
        `validation_metrics.csv`, `test_metrics.csv`,
        `sample_level_metrics.parquet` (CSV fallback),
        `checkpoint_manifest.json`, generated sample tensors, uncertainty
        volumes, plots, `accuracy_diversity_curve.csv`, `final_comparison.csv`,
        and `final_report.md`. Per-run manifests explicitly record that test and
        oracle best-of-K were not used for selection.
        """
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "A100", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n")
print(OUTPUT)
