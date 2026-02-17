"""Multi-label classification models for term-level brain map prediction.

This module implements classifiers that predict specific CogAtlas terms
(~808 terms) from brain activation patterns, rather than high-level categories.

All models use MultiOutputClassifier for efficient multi-label training.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_fscore_support,
    label_ranking_average_precision_score,
    coverage_error,
    label_ranking_loss,
    f1_score,
    hamming_loss
)
from typing import Dict, List, Tuple, Optional, Any
from scipy.special import expit
import pickle
import json
from pathlib import Path

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


def load_and_align_term_data(
    data_source: str = "cogatlas",
    threshold: Optional[float] = None,
    latent_neuro_loader: Optional[callable] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load and align brain vectors with term-level labels from HuggingFace.

    Always uses projected brain vectors (InfoNCE projection head) as this is
    required for optimal classification performance.

    Parameters
    ----------
    data_source : str, default="cogatlas"
        Data source to load. Options:
        - "cogatlas": Standard CogAtlas terms (~808 terms)
        - "cogatlas_threshold": Filtered CogAtlas terms (requires threshold param)
        - Custom filenames for hierarchical data (e.g., "fine_hierarchical_term")
    threshold : float, optional
        Threshold value for loading threshold-filtered data (e.g., 0.6, 0.65).
        Required when data_source="cogatlas_threshold".
    latent_neuro_loader : callable, optional
        Function to load latent neuro vectors. If None, uses default from retrieval_resources.

    Returns
    -------
    brain_vectors : np.ndarray
        Brain activation vectors (N x 384 projected features)
    term_labels : np.ndarray
        Term binary matrix (N x num_terms)
    pmids : np.ndarray
        Aligned PMIDs
    term_names : list of str
        Term names in order
    """
    print("Loading brain vectors...")
    if latent_neuro_loader is None:
        from neurovlm.retrieval_resources import _load_latent_neuro
        latent_neuro_loader = _load_latent_neuro

    brain_vectors, brain_pmids = latent_neuro_loader()

    # Always apply projection head
    print("Applying InfoNCE projection head...")
    from neurovlm.retrieval_resources import _proj_head_image_infonce

    proj_head_img = _proj_head_image_infonce()
    proj_head_img.eval()

    with torch.no_grad():
        brain_projected = proj_head_img(brain_vectors)
        brain_projected = brain_projected.numpy()

    # Normalize (same as in search_cogatlas_from_brain)
    brain_projected = brain_projected / np.linalg.norm(
        brain_projected, axis=1, keepdims=True
    )

    print(f"Using projected features (dim: {brain_projected.shape[1]})...")
    brain_vectors_final = brain_projected

    print("Loading term labels from HuggingFace...")

    # Load data based on source
    if data_source == "cogatlas":
        from neurovlm.retrieval_resources import _load_cogatlas_term_data
        term_matrix, term_names_array, label_pmids, _ = _load_cogatlas_term_data()
        term_names = term_names_array.tolist() if hasattr(term_names_array, 'tolist') else list(term_names_array)
    elif data_source == "cogatlas_threshold":
        if threshold is None:
            raise ValueError("threshold parameter is required when data_source='cogatlas_threshold'")
        from neurovlm.retrieval_resources import _load_cogatlas_term_threshold_data
        term_matrix, term_names_array, label_pmids, _ = _load_cogatlas_term_threshold_data(threshold)
        term_names = term_names_array.tolist() if hasattr(term_names_array, 'tolist') else list(term_names_array)
    else:
        # Legacy support: load from local files for hierarchical data
        import os
        from neurovlm.retrieval_resources import _download_from_hf

        # Try to load from HuggingFace cognitive_atlas dataset
        try:
            matrix_file = f"{data_source}_matrix.npy"
            labels_file = f"{data_source}_labels.npy"
            pmids_file = f"{data_source}_pmids.npy"

            matrix_path = _download_from_hf("neurovlm/cognitive_atlas", matrix_file)
            labels_path = _download_from_hf("neurovlm/cognitive_atlas", labels_file)
            pmids_path = _download_from_hf("neurovlm/cognitive_atlas", pmids_file)

            term_matrix = np.load(matrix_path)
            term_names = np.load(labels_path, allow_pickle=True).tolist()
            label_pmids = np.load(pmids_path)
        except Exception:
            # Fallback to local files if not in HuggingFace
            def _find_file(filepath: str) -> str:
                """Find file by trying multiple path strategies."""
                if os.path.exists(filepath):
                    return filepath
                if filepath.startswith('docs/02_models/'):
                    relative_path = filepath.replace('docs/02_models/', '', 1)
                    if os.path.exists(relative_path):
                        return relative_path
                if not filepath.startswith('docs/02_models/'):
                    prefixed_path = f'docs/02_models/{filepath}'
                    if os.path.exists(prefixed_path):
                        return prefixed_path
                return filepath

            term_matrix = np.load(_find_file(f"docs/02_models/{data_source}_matrix.npy"))
            term_names = np.load(_find_file(f"docs/02_models/{data_source}_labels.npy"), allow_pickle=True).tolist()
            label_pmids = np.load(_find_file(f"docs/02_models/{data_source}_pmids.npy"))

    print("Aligning data...")
    brain_pmids_str = brain_pmids.astype(str)
    label_pmids_str = label_pmids.astype(str)

    common_pmids = np.intersect1d(brain_pmids_str, label_pmids_str)
    print(f"Brain vectors: {len(brain_pmids)}")
    print(f"Labeled papers: {len(label_pmids)}")
    print(f"Common PMIDs: {len(common_pmids)}")

    brain_indices = np.array([np.where(brain_pmids_str == pmid)[0][0] for pmid in common_pmids])
    label_indices = np.array([np.where(label_pmids_str == pmid)[0][0] for pmid in common_pmids])

    aligned_brain_vectors = brain_vectors_final[brain_indices]
    aligned_term_labels = term_matrix[label_indices]

    print(f"\nAligned dataset:")
    print(f"  Brain vectors: {aligned_brain_vectors.shape}")
    print(f"  Term labels: {aligned_term_labels.shape}")
    print(f"  Number of terms: {len(term_names)}")

    # Statistics
    terms_per_paper = aligned_term_labels.sum(axis=1)
    papers_per_term = aligned_term_labels.sum(axis=0)
    print(f"\nLabel statistics:")
    print(f"  Terms per paper: mean={terms_per_paper.mean():.1f}, std={terms_per_paper.std():.1f}")
    print(f"  Papers per term: mean={papers_per_term.mean():.1f}, std={papers_per_term.std():.1f}")

    return aligned_brain_vectors, aligned_term_labels, common_pmids, term_names


class TermPredictionModel:
    """Base class for term-level prediction models.

    Implements sklearn BaseEstimator interface for compatibility with
    GridSearchCV, RandomizedSearchCV, etc.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator.

        Required for sklearn compatibility (GridSearchCV, etc.)

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        # Get all __init__ parameters
        import inspect
        signature = inspect.signature(self.__init__)
        params = {}

        for param_name in signature.parameters:
            if param_name == 'self':
                continue
            if hasattr(self, param_name):
                params[param_name] = getattr(self, param_name)

        return params

    def set_params(self, **params) -> 'TermPredictionModel':
        """Set the parameters of this estimator.

        Required for sklearn compatibility (GridSearchCV, etc.)

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : object
            Estimator instance.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model."""
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict term probabilities."""
        raise NotImplementedError

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict term presence."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the mean F1 score on the given test data and labels.

        Required for sklearn compatibility (GridSearchCV, etc.)

        Parameters
        ----------
        X : np.ndarray
            Test samples
        y : np.ndarray
            True labels for X

        Returns
        -------
        score : float
            Mean F1 score (samples-averaged)
        """
        from sklearn.metrics import f1_score
        y_pred = self.predict(X, threshold=0.5)
        return f1_score(y, y_pred, average='samples', zero_division=0)

    def predict_top_k(self, X: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Predict top-k most likely terms.

        Returns
        -------
        indices : np.ndarray
            Indices of top-k terms for each sample (n_samples, k)
        scores : np.ndarray
            Scores for top-k terms (n_samples, k)
        """
        proba = self.predict_proba(X)
        top_k_indices = np.argsort(proba, axis=1)[:, -k:][:, ::-1]
        top_k_scores = np.take_along_axis(proba, top_k_indices, axis=1)
        return top_k_indices, top_k_scores

    def save(self, filepath: str):
        """Save model to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filepath}")

    @staticmethod
    def load(filepath: str):
        """Load model from file."""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filepath}")
        return model


class LogisticTermPredictor(TermPredictionModel):
    """Multi-output Logistic Regression for term prediction.

    Uses MultiOutputClassifier for efficient parallel training across all terms.
    """

    def __init__(
        self,
        C: float = 1.0,
        penalty: str = 'l2',
        solver: str = 'lbfgs',
        max_iter: int = 1000,
        class_weight: Optional[str] = 'balanced',
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """Initialize Logistic Regression model.

        Parameters
        ----------
        C : float
            Inverse regularization strength (default 1.0)
        penalty : str
            Regularization type: 'l1', 'l2' (default 'l2')
        solver : str
            Optimization algorithm (default 'lbfgs')
        max_iter : int
            Maximum iterations (default 1000)
        class_weight : str, optional
            'balanced' to adjust for class imbalance (default 'balanced')
        random_state : int
            Random seed
        n_jobs : int
            Number of parallel jobs (-1 = use all cores)
        """
        super().__init__(random_state)
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.n_jobs = n_jobs

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        if verbose:
            print("Scaling features...")
        X_scaled = self.scaler.fit_transform(X)

        if verbose:
            print(f"Training multi-output logistic regression...")
            print(f"  Input shape: {X_scaled.shape}")
            print(f"  Output shape: {y.shape}")

        base_model = LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            solver=self.solver,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state
        )
        self.model = MultiOutputClassifier(base_model, n_jobs=self.n_jobs)
        self.model.fit(X_scaled, y)

        if verbose:
            print("Training complete!")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        # Get probabilities for positive class from all estimators
        proba = np.column_stack([
            estimator.predict_proba(X_scaled)[:, 1]
            for estimator in self.model.estimators_
        ])
        return proba


class XGBoostTermPredictor(TermPredictionModel):
    """XGBoost for multi-label term prediction.

    Uses MultiOutputClassifier for efficient parallel training.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        tree_method: str = 'hist',
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """Initialize XGBoost model.

        Parameters
        ----------
        n_estimators : int
            Number of boosting rounds (default 100)
        max_depth : int
            Maximum tree depth (default 3, shallow to prevent overfitting)
        learning_rate : float
            Boosting learning rate (default 0.1)
        subsample : float
            Subsample ratio of training instances (default 0.8)
        colsample_bytree : float
            Subsample ratio of columns (default 0.8)
        reg_alpha : float
            L1 regularization (default 0.1)
        reg_lambda : float
            L2 regularization (default 1.0)
        tree_method : str
            Tree construction algorithm (default 'hist' for speed)
        random_state : int
            Random seed
        n_jobs : int
            Number of parallel jobs (-1 = use all cores)
        """
        super().__init__(random_state)
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.tree_method = tree_method
        self.n_jobs = n_jobs

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        if verbose:
            print("Scaling features...")
        X_scaled = self.scaler.fit_transform(X)

        if verbose:
            print(f"Training XGBoost multi-output classifier...")
            print(f"  Input shape: {X_scaled.shape}")
            print(f"  Output shape: {y.shape}")

        base_model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            tree_method=self.tree_method,
            random_state=self.random_state,
            eval_metric='logloss'
        )
        self.model = MultiOutputClassifier(base_model, n_jobs=self.n_jobs)
        self.model.fit(X_scaled, y)

        if verbose:
            print("Training complete!")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        proba = np.column_stack([
            estimator.predict_proba(X_scaled)[:, 1]
            for estimator in self.model.estimators_
        ])
        return proba


class LightGBMTermPredictor(TermPredictionModel):
    """LightGBM for multi-label term prediction.

    Uses MultiOutputClassifier for efficient parallel training.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """Initialize LightGBM model.

        Parameters
        ----------
        n_estimators : int
            Number of boosting rounds (default 100)
        max_depth : int
            Maximum tree depth (default 3)
        learning_rate : float
            Boosting learning rate (default 0.1)
        num_leaves : int
            Maximum tree leaves (default 31)
        subsample : float
            Subsample ratio of training instances (default 0.8)
        colsample_bytree : float
            Subsample ratio of columns (default 0.8)
        reg_alpha : float
            L1 regularization (default 0.1)
        reg_lambda : float
            L2 regularization (default 1.0)
        random_state : int
            Random seed
        n_jobs : int
            Number of parallel jobs (-1 = use all cores)
        """
        super().__init__(random_state)
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed. Install with: pip install lightgbm")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.n_jobs = n_jobs

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        if verbose:
            print("Scaling features...")
        X_scaled = self.scaler.fit_transform(X)

        if verbose:
            print(f"Training LightGBM multi-output classifier...")
            print(f"  Input shape: {X_scaled.shape}")
            print(f"  Output shape: {y.shape}")

        base_model = lgb.LGBMClassifier(
            objective='binary',
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state
        )
        self.model = MultiOutputClassifier(base_model, n_jobs=self.n_jobs)
        self.model.fit(X_scaled, y)

        if verbose:
            print("Training complete!")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        proba = np.column_stack([
            estimator.predict_proba(X_scaled)[:, 1]
            for estimator in self.model.estimators_
        ])
        return proba


class CatBoostTermPredictor(TermPredictionModel):
    """CatBoost for multi-label term prediction.

    Uses MultiOutputClassifier for efficient parallel training.
    """

    def __init__(
        self,
        iterations: int = 100,
        depth: int = 3,
        learning_rate: float = 0.1,
        l2_leaf_reg: float = 3.0,
        subsample: float = 0.8,
        random_state: int = 42,
        verbose: bool = False,
        n_jobs: int = -1
    ):
        """Initialize CatBoost model.

        Parameters
        ----------
        iterations : int
            Number of boosting rounds (default 100)
        depth : int
            Maximum tree depth (default 3)
        learning_rate : float
            Boosting learning rate (default 0.1)
        l2_leaf_reg : float
            L2 regularization (default 3.0)
        subsample : float
            Subsample ratio (default 0.8)
        random_state : int
            Random seed
        verbose : bool
            Whether to show training progress (default False)
        n_jobs : int
            Number of parallel jobs (-1 = use all cores)
        """
        super().__init__(random_state)
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost not installed. Install with: pip install catboost")

        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.subsample = subsample
        self.verbose = verbose
        self.n_jobs = n_jobs

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        if verbose:
            print("Scaling features...")
        X_scaled = self.scaler.fit_transform(X)

        if verbose:
            print(f"Training CatBoost multi-output classifier...")
            print(f"  Input shape: {X_scaled.shape}")
            print(f"  Output shape: {y.shape}")

        base_model = CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            subsample=self.subsample,
            random_seed=self.random_state,
            verbose=self.verbose,
            allow_writing_files=False
        )
        self.model = MultiOutputClassifier(base_model, n_jobs=self.n_jobs)
        self.model.fit(X_scaled, y)

        if verbose:
            print("Training complete!")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        proba = np.column_stack([
            estimator.predict_proba(X_scaled)[:, 1]
            for estimator in self.model.estimators_
        ])
        return proba


def evaluate_term_prediction(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    term_names: List[str],
    top_k: int = 10
) -> Dict[str, Any]:
    """Evaluate term-level prediction performance.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (n_samples, n_terms)
    y_pred : np.ndarray
        Predicted binary labels (n_samples, n_terms)
    y_proba : np.ndarray
        Predicted probabilities (n_samples, n_terms)
    term_names : list of str
        Term names
    top_k : int
        For top-k metrics

    Returns
    -------
    results : dict
        Evaluation metrics
    """
    results = {}

    # Overall metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )
    results['micro'] = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support
    }

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    results['macro'] = {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    # Ranking metrics (more appropriate for multi-label)
    results['ranking'] = {
        'label_ranking_avg_precision': label_ranking_average_precision_score(y_true, y_proba),
        'coverage_error': coverage_error(y_true, y_proba),
        'label_ranking_loss': label_ranking_loss(y_true, y_proba)
    }

    # Top-k accuracy (at least one hit)
    top_k_pred = np.argsort(y_proba, axis=1)[:, -top_k:]
    top_k_hits = 0
    for i in range(len(y_true)):
        true_terms = np.where(y_true[i] == 1)[0]
        if len(true_terms) > 0 and np.any(np.isin(top_k_pred[i], true_terms)):
            top_k_hits += 1

    results['top_k'] = {
        'k': top_k,
        'accuracy': top_k_hits / len(y_true),
        'description': f"Fraction of samples with at least one true term in top-{top_k}"
    }

    # Recall@K metrics (5, 7, 10)
    recall_at_k = {}
    for k in [5, 7, 10]:
        top_k_pred = np.argsort(y_proba, axis=1)[:, -k:]
        recalls = []
        for i in range(len(y_true)):
            true_terms = np.where(y_true[i] == 1)[0]
            if len(true_terms) > 0:
                hits = np.sum(np.isin(top_k_pred[i], true_terms))
                recalls.append(hits / len(true_terms))
        recall_at_k[f'recall@{k}'] = np.mean(recalls) if recalls else 0.0

    results['recall_at_k'] = recall_at_k

    # AUC metrics
    try:
        from sklearn.metrics import roc_auc_score

        auc_micro = roc_auc_score(y_true.ravel(), y_proba.ravel())

        auc_per_label = []
        for term_idx in range(y_true.shape[1]):
            n_classes = len(np.unique(y_true[:, term_idx]))
            if n_classes >= 2:
                try:
                    auc = roc_auc_score(y_true[:, term_idx], y_proba[:, term_idx])
                    auc_per_label.append(auc)
                except ValueError:
                    pass

        auc_macro = np.mean(auc_per_label) if auc_per_label else 0.0

        results['auc'] = {
            'micro': auc_micro,
            'macro': auc_macro,
            'description': 'AUC-ROC scores for multi-label classification'
        }
    except Exception as e:
        results['auc'] = {'error': str(e)}

    # Per-term metrics for most common terms
    papers_per_term = y_true.sum(axis=0)
    top_terms_indices = np.argsort(papers_per_term)[-20:][::-1]

    results['top_terms'] = {}
    for idx in top_terms_indices:
        term = term_names[idx]
        p, r, f, _ = precision_recall_fscore_support(
            y_true[:, idx], y_pred[:, idx], average='binary', zero_division=0
        )
        term_support = int(np.sum(y_true[:, idx] == 1))

        results['top_terms'][term] = {
            'precision': p,
            'recall': r,
            'f1': f,
            'support': term_support
        }

    return results


def print_term_evaluation_results(results: Dict[str, Any]):
    """Print evaluation results in formatted way."""
    print("\n=== Overall Performance ===")
    print(f"{'Metric':<30} {'Micro':>10} {'Macro':>10}")
    print("-" * 52)
    for metric in ['precision', 'recall', 'f1']:
        micro_val = results['micro'][metric]
        macro_val = results['macro'][metric]
        print(f"{metric.capitalize():<30} {micro_val:>10.3f} {macro_val:>10.3f}")

    print("\n=== Ranking Metrics ===")
    for key, value in results['ranking'].items():
        print(f"  {key}: {value:.4f}")

    print(f"\n=== Top-{results['top_k']['k']} Accuracy ===")
    print(f"  {results['top_k']['accuracy']:.3f} - {results['top_k']['description']}")

    if 'recall_at_k' in results:
        print("\n=== Recall@K Metrics ===")
        for key, value in results['recall_at_k'].items():
            k = key.split('@')[1]
            print(f"  {key}: {value:.4f} - Average fraction of true labels found in top-{k}")

    if 'auc' in results and 'error' not in results['auc']:
        print("\n=== AUC Metrics ===")
        print(f"  Micro-AUC: {results['auc']['micro']:.4f}")
        print(f"  Macro-AUC: {results['auc']['macro']:.4f}")

    print("\n=== Performance on Most Common Terms ===")
    print(f"{'Term':<40} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 82)
    for term, metrics in list(results['top_terms'].items())[:10]:
        print(f"{term:<40} {metrics['precision']:>10.3f} {metrics['recall']:>10.3f} {metrics['f1']:>10.3f} {metrics['support']:>10}")


def save_term_results(results: Dict[str, Any], output_file: str):
    """Save evaluation results to JSON."""
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_serializable = json.loads(json.dumps(results, default=convert))

    with open(output_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print(f"\nResults saved to {output_file}")


# ============================================================================
# PYTORCH NEURAL NETWORK MODELS
# ============================================================================

# Architecture 5: Flexible feedforward model (configurable depth/width)
class FlexibleModel(nn.Module):
    """Flexible feedforward network with configurable architecture.

    This is a cleaner alternative to WarmStartModel for exploring different
    architectures without the constraint of preserving a linear solution.

    Parameters
    ----------
    input_dim : int
    output_dim : int
    hidden_dims : list of int
        Hidden layer sizes, e.g. [1024, 512, 256]
    activation : str
        'gelu', 'relu', or 'silu'. Default 'gelu'.
    dropout : float
        Dropout rate. Default 0.3.
    use_batchnorm : bool
        Whether to use batch normalization. Default True.
    """

    def __init__(self, input_dim, output_dim, hidden_dims=None,
                 activation='gelu', dropout=0.3, use_batchnorm=True):
        super(FlexibleModel, self).__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256]

        # Activation function
        if activation == 'gelu':
            act_fn = nn.GELU
        elif activation == 'silu':
            act_fn = nn.SiLU
        else:
            act_fn = nn.ReLU

        # Build layers
        layers = []
        prev_dim = input_dim

        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(act_fn())
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hdim))
            layers.append(nn.Dropout(dropout))
            prev_dim = hdim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ============================================================================
# LOSS FUNCTIONS AND UTILITIES
# ============================================================================

def calculate_pos_weights(y_train: np.ndarray) -> torch.Tensor:
    """Calculate balanced class weights for BCEWithLogitsLoss.

    For each label, pos_weight = num_negatives / num_positives.
    This rebalances the loss so rare positives aren't drowned out.

    Parameters
    ----------
    y_train : np.ndarray, shape (n_samples, n_labels)
        Binary label matrix.

    Returns
    -------
    torch.Tensor, shape (n_labels,)
        Per-label positive weights.
    """
    n_samples = y_train.shape[0]
    pos_counts = y_train.sum(axis=0)  # per-label positive count
    neg_counts = n_samples - pos_counts

    # Avoid division by zero; labels with 0 positives get weight 1.0
    pos_weights = np.where(pos_counts > 0, neg_counts / pos_counts, 1.0)

    return torch.FloatTensor(pos_weights)


# Import FocalWithLogitsLoss from loss module
from neurovlm.loss import FocalWithLogitsLoss


def find_optimal_threshold(
    model: nn.Module,
    X_val: np.ndarray,
    y_val: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
) -> float:
    """Find the sigmoid threshold that maximizes micro-F1 on the validation set.

    Parameters
    ----------
    model : nn.Module
        Trained model that outputs raw logits.
    X_val : np.ndarray
    y_val : np.ndarray
    thresholds : np.ndarray, optional
        Candidate thresholds to search. Defaults to np.arange(0.1, 0.6, 0.02).
    device : torch.device, optional

    Returns
    -------
    float
        Best threshold value.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if thresholds is None:
        thresholds = np.arange(0.05, 0.55, 0.02)

    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(X_val).to(device)).cpu().numpy()
    probs = expit(logits)  # sigmoid in numpy

    best_f1, best_t = -1.0, 0.5
    for t in thresholds:
        preds = (probs > t).astype(int)
        f1 = f1_score(y_val, preds, average='micro', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    return best_t


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_model(model, X_train, y_train, X_val, y_val,
                epochs=100, batch_size=32, learning_rate=0.001,
                patience=10, verbose=1, criterion=None, optimizer=None,
                scheduler_type='cosine', grad_clip=1.0,
                label_smoothing=0.0, mixup_alpha=0.0):
    """
    Train a PyTorch model with early stopping.

    Models output raw logits (no sigmoid). The criterion should operate on
    logits (e.g. BCEWithLogitsLoss or FocalWithLogitsLoss).

    Args:
        model: PyTorch model (outputs raw logits)
        X_train, y_train: Training data (numpy arrays)
        X_val, y_val: Validation data (numpy arrays)
        epochs: Maximum number of epochs
        batch_size: Batch size
        learning_rate: Learning rate (ignored if optimizer is provided)
        patience: Early stopping patience
        verbose: Verbosity (0=silent, 1=progress bar, 2=one line per epoch)
        criterion: Loss function (default: BCEWithLogitsLoss)
        optimizer: Pre-configured optimizer. If None, creates AdamW.
            Pass a custom optimizer for differential learning rates.
        scheduler_type: 'cosine' for CosineAnnealingWarmRestarts,
            'plateau' for ReduceLROnPlateau. Default 'cosine'.
        grad_clip: Max gradient norm for clipping. 0 to disable. Default 1.0.
        label_smoothing: Smooth binary labels toward 0.5.
            e.g. 0.05 maps 0->0.05, 1->0.95. Default 0.0 (off).
        mixup_alpha: Alpha for Beta distribution in mixup augmentation.
            0 to disable. Default 0.0 (off).

    Returns:
        model: Trained model
        history: Dictionary with training history
    """

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Using device: {device}")

    model = model.to(device)

    # Apply label smoothing to training labels
    if label_smoothing > 0:
        y_train_smooth = y_train * (1 - label_smoothing) + (1 - y_train) * label_smoothing
        if verbose:
            print(f"Label smoothing: {label_smoothing} (0->{label_smoothing:.2f}, 1->{1-label_smoothing:.2f})")
    else:
        y_train_smooth = y_train

    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train_smooth)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)  # no smoothing on validation
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Loss function (default: plain BCEWithLogitsLoss)
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()
    criterion = criterion.to(device)

    # Optimizer (default: AdamW; pass custom optimizer for differential LR)
    if optimizer is None:
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                                weight_decay=1e-2)

    # Learning rate scheduler
    if scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-6
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_f1': [],
        'val_f1': [],
        'lr': [],
    }

    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # Training loop
    for epoch in range(epochs):
        # ========== Training ==========
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            # Mixup augmentation
            if mixup_alpha > 0:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                idx = torch.randperm(batch_X.size(0), device=device)
                batch_X = lam * batch_X + (1 - lam) * batch_X[idx]
                batch_y = lam * batch_y + (1 - lam) * batch_y[idx]

            # Forward pass
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)

            # Backward pass
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            # Statistics — apply sigmoid to logits before thresholding
            probs = torch.sigmoid(logits)
            train_loss += loss.item() * batch_X.size(0)
            train_preds.append((probs > 0.5).cpu().numpy())
            train_targets.append((batch_y > 0.5).cpu().numpy())  # threshold smoothed labels

        train_loss = train_loss / len(train_dataset)
        train_preds = np.vstack(train_preds)
        train_targets = np.vstack(train_targets)
        train_f1 = f1_score(train_targets, train_preds, average='micro', zero_division=0)

        # ========== Validation ==========
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                logits = model(batch_X)
                loss = criterion(logits, batch_y)

                probs = torch.sigmoid(logits)
                val_loss += loss.item() * batch_X.size(0)
                val_preds.append((probs > 0.5).cpu().numpy())
                val_targets.append(batch_y.cpu().numpy())

        val_loss = val_loss / len(val_dataset)
        val_preds = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)
        val_f1 = f1_score(val_targets, val_preds, average='micro', zero_division=0)

        # Update learning rate
        if scheduler_type == 'cosine':
            scheduler.step(epoch)
        else:
            scheduler.step(val_loss)

        # Save history
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        history['lr'].append(current_lr)

        # Print progress
        if verbose >= 1:
            if verbose == 1 and epoch % 5 == 0:  # Print every 5 epochs
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Loss: {train_loss:.4f} - F1: {train_f1:.4f} - "
                      f"Val Loss: {val_loss:.4f} - Val F1: {val_f1:.4f} - "
                      f"LR: {current_lr:.2e}")
            elif verbose == 2:  # Print every epoch
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Loss: {train_loss:.4f} - F1: {train_f1:.4f} - "
                      f"Val Loss: {val_loss:.4f} - Val F1: {val_f1:.4f} - "
                      f"LR: {current_lr:.2e}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    print(f"Best validation loss: {best_val_loss:.4f}")
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history


# ============================================================================
# COMPLETE TRAINING PIPELINE
# ============================================================================



def train_neural_classifier(
    X_train, y_train, X_test, y_test,
    hidden_dims=None,
    activation='gelu',
    dropout=0.3,
    use_batchnorm=True,
    learning_rate=1e-3,
    weight_decay=1e-2,
    pos_weight_scale=0.5,
    loss_type='bce',
    focal_gamma=2.0,
    epochs=200,
    batch_size=256,
    validation_split=0.15,
    patience=20,
    scheduler_type='cosine',
    grad_clip=1.0,
    verbose=1,
):
    """Train a neural network classifier from random initialization.

    This is a cleaner alternative to warm-start training that lets the model
    learn optimal representations without being constrained to preserve a
    linear solution.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training data (should be pre-scaled with StandardScaler)
    X_test, y_test : np.ndarray
        Test data (scaled with same scaler)
    hidden_dims : list of int, optional
        Hidden layer sizes. Default [1024, 512].
    activation : str
        'gelu', 'relu', or 'silu'. Default 'gelu'.
    dropout : float
        Dropout rate. Default 0.3.
    use_batchnorm : bool
        Whether to use batch normalization. Default True.
    learning_rate : float
        Learning rate. Default 1e-3.
    weight_decay : float
        AdamW weight decay. Default 1e-2.
    pos_weight_scale : float
        Pos-weight damping. Default 0.5 (sqrt).
    loss_type : str
        'bce' or 'focal'. Default 'bce'.
    focal_gamma : float
        Focal loss gamma. Default 2.0.
    epochs : int
        Max epochs. Default 200.
    batch_size : int
        Batch size. Default 256.
    validation_split : float
        Fraction for validation. Default 0.15.
    patience : int
        Early stopping patience. Default 20.
    scheduler_type : str
        'cosine' or 'plateau'. Default 'cosine'.
    grad_clip : float
        Max gradient norm. 0 to disable. Default 1.0.
    verbose : int
        Verbosity. Default 1.

    Returns
    -------
    model : FlexibleModel
    history : dict
    metrics : dict
    """
    if hidden_dims is None:
        hidden_dims = [1024, 512]

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    print(f"Input dimension: {input_dim}")
    print(f"Output dimension: {output_dim}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")

    # ========== Build model ==========
    print("\n" + "="*70)
    print("BUILDING MODEL")
    print("="*70)

    model = FlexibleModel(
        input_dim, output_dim,
        hidden_dims=hidden_dims,
        activation=activation,
        dropout=dropout,
        use_batchnorm=use_batchnorm
    )

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Architecture: {input_dim} -> {' -> '.join(map(str, hidden_dims))} -> {output_dim}")
        print(f"Activation: {activation}")
        print(f"Batch normalization: {use_batchnorm}")
        print(f"Dropout: {dropout}")
        print(f"Total parameters: {n_params:,}")

    # ========== Configure optimizer ==========
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                            weight_decay=weight_decay)

    # ========== Train ==========
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)

    # Split for validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=validation_split, random_state=42
    )

    print(f"Training set: {X_tr.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")

    # Compute pos_weight
    pos_weight_raw = calculate_pos_weights(y_tr)
    if pos_weight_scale == 0.0:
        pos_weight = None
    else:
        pos_weight = torch.pow(pos_weight_raw, pos_weight_scale)
        if verbose:
            print(f"Pos-weight scale: {pos_weight_scale}, "
                  f"range: {pos_weight.min():.2f} to {pos_weight.max():.2f}")

    if loss_type == 'focal':
        criterion = FocalWithLogitsLoss(gamma=focal_gamma, pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model, history = train_model(
        model, X_tr, y_tr, X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        verbose=verbose,
        criterion=criterion,
        optimizer=optimizer,
        scheduler_type=scheduler_type,
        grad_clip=grad_clip,
        label_smoothing=0.0,
        mixup_alpha=0.0,
    )

    # ========== Evaluate ==========
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_threshold = find_optimal_threshold(model, X_val, y_val, device=device)
    if verbose:
        print(f"\nOptimal threshold: {best_threshold:.3f}")

    print("\n" + "="*70)
    print("EVALUATION ON TEST SET")
    print("="*70)

    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(X_test).to(device)).cpu().numpy()
    y_pred_prob = expit(logits)
    y_pred = (y_pred_prob > best_threshold).astype(int)

    hamming = hamming_loss(y_test, y_pred)
    f1_micro = f1_score(y_test, y_pred, average='micro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_samples = f1_score(y_test, y_pred, average='samples', zero_division=0)
    lrap = label_ranking_average_precision_score(y_test, y_pred_prob)

    k = min(10, y_pred_prob.shape[1])
    top_k_indices = np.argsort(y_pred_prob, axis=1)[:, -k:]
    r10 = []
    for i in range(len(y_test)):
        true_labels = np.where(y_test[i] == 1)[0]
        if len(true_labels) > 0:
            r10.append(len(set(true_labels) & set(top_k_indices[i])) / len(true_labels))
        else:
            r10.append(1.0)
    recall_at_10 = np.mean(r10)

    metrics = {
        'hamming_loss': hamming,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_samples': f1_samples,
        'lrap': lrap,
        'recall_at_10': recall_at_10,
        'threshold': best_threshold,
    }

    print(f"Threshold:                   {best_threshold:.3f}")
    print(f"Hamming Loss:                {hamming:.4f}")
    print(f"F1 Score (micro):            {f1_micro:.4f}")
    print(f"F1 Score (macro):            {f1_macro:.4f}")
    print(f"F1 Score (samples):          {f1_samples:.4f}")
    print(f"Label Ranking Avg Precision: {lrap:.4f}")
    print(f"Recall@10:                   {recall_at_10:.4f}")

    labels_pred = y_pred.sum(axis=1)
    print(f"\nPrediction Statistics:")
    print(f"Avg labels per sample (true): {y_test.sum(axis=1).mean():.2f}")
    print(f"Avg labels per sample (pred): {labels_pred.mean():.2f}")
    print(f"Avg prediction confidence:    {y_pred_prob.mean():.4f}")

    return model, history, metrics


# ============================================================================
# PLOTTING FUNCTION
# ============================================================================

def plot_training_history(history):
    """Plot training curves"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # F1 Score
    axes[1].plot(history['train_f1'], label='Train F1', linewidth=2)
    axes[1].plot(history['val_f1'], label='Val F1', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('F1 Micro', fontsize=12)
    axes[1].set_title('Training and Validation F1 (Micro)', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ============================================================================
# REGRESSION TRAINING (CONTINUOUS PROBABILITY TARGETS)
# ============================================================================


def train_model_regression(model, X_train, y_train, X_val, y_val,
                           epochs=100, batch_size=32, learning_rate=0.001,
                           patience=10, verbose=1, criterion=None, optimizer=None,
                           scheduler_type='cosine', grad_clip=1.0):
    """Train a PyTorch model for regression with early stopping.

    Models output raw values (no activation). The criterion should be a
    regression loss (e.g. MSELoss, SmoothL1Loss).

    Parameters
    ----------
    model : nn.Module
        PyTorch model (outputs raw values, same shape as y)
    X_train, y_train : np.ndarray
        Training data. y_train contains continuous target values.
    X_val, y_val : np.ndarray
        Validation data.
    epochs : int
        Maximum number of epochs.
    batch_size : int
        Batch size.
    learning_rate : float
        Learning rate (ignored if optimizer provided).
    patience : int
        Early stopping patience.
    verbose : int
        Verbosity (0=silent, 1=every 5 epochs, 2=every epoch).
    criterion : nn.Module, optional
        Loss function. Default: SmoothL1Loss.
    optimizer : torch.optim.Optimizer, optional
        Pre-configured optimizer. Default: AdamW.
    scheduler_type : str
        'cosine' or 'plateau'.
    grad_clip : float
        Max gradient norm. 0 to disable.

    Returns
    -------
    model : nn.Module
        Trained model (best validation loss checkpoint).
    history : dict
        Training history with losses and metrics.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Using device: {device}")

    model = model.to(device)

    # Data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if criterion is None:
        criterion = nn.SmoothL1Loss()
    criterion = criterion.to(device)

    if optimizer is None:
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                                weight_decay=1e-2)

    if scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-6
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_r2': [],
        'val_r2': [],
        'lr': [],
    }

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_preds_all = []
        train_targets_all = []

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            train_loss += loss.item() * batch_X.size(0)
            train_preds_all.append(outputs.detach().cpu().numpy())
            train_targets_all.append(batch_y.cpu().numpy())

        train_loss = train_loss / len(train_dataset)
        train_preds_all = np.vstack(train_preds_all)
        train_targets_all = np.vstack(train_targets_all)

        # R² on training set
        ss_res = ((train_targets_all - train_preds_all) ** 2).sum()
        ss_tot = ((train_targets_all - train_targets_all.mean()) ** 2).sum()
        train_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds_all = []
        val_targets_all = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item() * batch_X.size(0)
                val_preds_all.append(outputs.cpu().numpy())
                val_targets_all.append(batch_y.cpu().numpy())

        val_loss = val_loss / len(val_dataset)
        val_preds_all = np.vstack(val_preds_all)
        val_targets_all = np.vstack(val_targets_all)

        ss_res = ((val_targets_all - val_preds_all) ** 2).sum()
        ss_tot = ((val_targets_all - val_targets_all.mean()) ** 2).sum()
        val_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # LR scheduler
        if scheduler_type == 'cosine':
            scheduler.step(epoch)
        else:
            scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_r2'].append(train_r2)
        history['val_r2'].append(val_r2)
        history['lr'].append(current_lr)

        if verbose >= 1:
            if verbose == 1 and epoch % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Loss: {train_loss:.6f} - R²: {train_r2:.4f} - "
                      f"Val Loss: {val_loss:.6f} - Val R²: {val_r2:.4f} - "
                      f"LR: {current_lr:.2e}")
            elif verbose == 2:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Loss: {train_loss:.6f} - R²: {train_r2:.4f} - "
                      f"Val Loss: {val_loss:.6f} - Val R²: {val_r2:.4f} - "
                      f"LR: {current_lr:.2e}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    print(f"Best validation loss: {best_val_loss:.6f}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history


def train_neural_regressor(
    X_train, y_train, X_test, y_test,
    hidden_dims=None,
    activation='gelu',
    dropout=0.3,
    use_batchnorm=True,
    learning_rate=1e-3,
    weight_decay=1e-2,
    loss_type='smooth_l1',
    epochs=200,
    batch_size=256,
    validation_split=0.15,
    patience=20,
    scheduler_type='cosine',
    grad_clip=1.0,
    verbose=1,
):
    """Train a neural network regressor for continuous probability targets.

    Parallel to train_neural_classifier but for regression (continuous P(T|A) targets).

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training data. y_train is float32 with continuous probability values.
    X_test, y_test : np.ndarray
        Test data.
    hidden_dims : list of int, optional
        Hidden layer sizes. Default [1024, 512].
    activation : str
        'gelu', 'relu', or 'silu'. Default 'gelu'.
    dropout : float
        Dropout rate. Default 0.3.
    use_batchnorm : bool
        Whether to use batch normalization. Default True.
    learning_rate : float
        Learning rate. Default 1e-3.
    weight_decay : float
        AdamW weight decay. Default 1e-2.
    loss_type : str
        'mse', 'smooth_l1', or 'kl_div'. Default 'smooth_l1'.
    epochs : int
        Max epochs. Default 200.
    batch_size : int
        Batch size. Default 256.
    validation_split : float
        Fraction for validation. Default 0.15.
    patience : int
        Early stopping patience. Default 20.
    scheduler_type : str
        'cosine' or 'plateau'. Default 'cosine'.
    grad_clip : float
        Max gradient norm. 0 to disable. Default 1.0.
    verbose : int
        Verbosity. Default 1.

    Returns
    -------
    model : FlexibleModel
    history : dict
    metrics : dict
    """
    if hidden_dims is None:
        hidden_dims = [1024, 512]

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    print(f"Input dimension: {input_dim}")
    print(f"Output dimension: {output_dim}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Target range: [{y_train.min():.6f}, {y_train.max():.6f}]")

    # Build model
    print("\n" + "="*70)
    print("BUILDING MODEL (Regression)")
    print("="*70)

    model = FlexibleModel(
        input_dim, output_dim,
        hidden_dims=hidden_dims,
        activation=activation,
        dropout=dropout,
        use_batchnorm=use_batchnorm
    )

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Architecture: {input_dim} -> {' -> '.join(map(str, hidden_dims))} -> {output_dim}")
        print(f"Activation: {activation}")
        print(f"Batch normalization: {use_batchnorm}")
        print(f"Dropout: {dropout}")
        print(f"Total parameters: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                            weight_decay=weight_decay)

    # Train
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=validation_split, random_state=42
    )

    print(f"Training set: {X_tr.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")

    if loss_type == 'mse':
        criterion = nn.MSELoss()
    elif loss_type == 'kl_div':
        criterion = nn.KLDivLoss(reduction='batchmean')
    else:
        criterion = nn.SmoothL1Loss()

    print(f"Loss function: {loss_type}")

    model, history = train_model_regression(
        model, X_tr, y_tr, X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        verbose=verbose,
        criterion=criterion,
        optimizer=optimizer,
        scheduler_type=scheduler_type,
        grad_clip=grad_clip,
    )

    # Evaluate
    print("\n" + "="*70)
    print("EVALUATION ON TEST SET")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.FloatTensor(X_test).to(device)).cpu().numpy()

    metrics = evaluate_regression(y_test, y_pred)

    print(f"MSE:                         {metrics['mse']:.6f}")
    print(f"MAE:                         {metrics['mae']:.6f}")
    print(f"R² (overall):                {metrics['r2']:.4f}")
    print(f"Mean Spearman corr (per sample): {metrics['spearman_per_sample']:.4f}")
    print(f"Label Ranking Avg Precision: {metrics['lrap']:.4f}")
    print(f"Recall@10:                   {metrics['recall_at_10']:.4f}")
    print(f"Recall@5:                    {metrics['recall_at_5']:.4f}")

    print(f"\nPrediction Statistics:")
    print(f"  Target mean:     {y_test.mean():.6f}")
    print(f"  Prediction mean: {y_pred.mean():.6f}")
    print(f"  Target std:      {y_test.std():.6f}")
    print(f"  Prediction std:  {y_pred.std():.6f}")

    return model, history, metrics


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Evaluate regression predictions for continuous probability targets.

    Parameters
    ----------
    y_true : np.ndarray
        True continuous values (n_samples, n_terms).
    y_pred : np.ndarray
        Predicted continuous values (n_samples, n_terms).

    Returns
    -------
    metrics : dict
        MSE, MAE, R², Spearman correlation, LRAP, Recall@K.
    """
    from scipy.stats import spearmanr

    # Regression metrics
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Per-sample Spearman correlation (how well does the model rank terms for each paper?)
    spearman_corrs = []
    for i in range(len(y_true)):
        if y_true[i].std() > 0 and y_pred[i].std() > 0:
            corr, _ = spearmanr(y_true[i], y_pred[i])
            if not np.isnan(corr):
                spearman_corrs.append(corr)
    spearman_per_sample = float(np.mean(spearman_corrs)) if spearman_corrs else 0.0

    # Ranking metrics — use continuous targets as relevance scores
    # LRAP: for each sample, how well does the predicted ranking match true ranking?
    # We need binary relevance for LRAP — use top-K of true values as "relevant"
    # Use papers with top 12 true probabilities as "relevant" (matching original top-k count)
    n_relevant = min(12, y_true.shape[1])
    y_true_binary = np.zeros_like(y_true, dtype=int)
    for i in range(len(y_true)):
        top_indices = np.argsort(y_true[i])[-n_relevant:]
        y_true_binary[i, top_indices] = 1

    lrap = float(label_ranking_average_precision_score(y_true_binary, y_pred))

    # Recall@K
    recall_at_k = {}
    for k in [5, 10]:
        top_k_pred = np.argsort(y_pred, axis=1)[:, -k:]
        recalls = []
        for i in range(len(y_true)):
            true_top = set(np.argsort(y_true[i])[-n_relevant:])
            pred_top = set(top_k_pred[i])
            if len(true_top) > 0:
                recalls.append(len(true_top & pred_top) / len(true_top))
        recall_at_k[k] = float(np.mean(recalls)) if recalls else 0.0

    return {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'spearman_per_sample': spearman_per_sample,
        'lrap': lrap,
        'recall_at_5': recall_at_k.get(5, 0.0),
        'recall_at_10': recall_at_k.get(10, 0.0),
    }


def plot_regression_history(history):
    """Plot training curves for regression models."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # R²
    axes[1].plot(history['train_r2'], label='Train R²', linewidth=2)
    axes[1].plot(history['val_r2'], label='Val R²', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('R²', fontsize=12)
    axes[1].set_title('Training and Validation R²', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
