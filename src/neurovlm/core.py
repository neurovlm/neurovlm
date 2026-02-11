"""Unified retrieval interface for NeuroVLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import nibabel as nib
from nilearn.image import resample_to_img

from neurovlm.data import load_dataset, load_latent, load_masker
from neurovlm.models import load_model

__all__ = ["NeuroVLM", "TextSearchResult", "BrainSearchResult", "BrainTopKResult"]

TEXT_EMBED_DIM = 768
LATENT_DIM = 384
BRAIN_FLAT_DIM = 28542

DEFAULT_TEXT_DATASETS = ("pubmed", "wiki", "cogatlas", "networks")
DATASET_ALIASES = {
    "publications": "pubmed",
    "neurowiki": "wiki",
    "concepts": "cogatlas",
    "tasks": "cogatlas_task",
    "disorders": "cogatlas_disorder",
    "network": "networks",
    "networks": "networks",
    "networks_canonical": "networks",
    "canonical_networks": "networks",
}
DATASET_ID_COLUMNS = {
    "pubmed": "pmid",
    "wiki": "id",
    "cogatlas": "term",
    "cogatlas_task": "term",
    "cogatlas_disorder": "term",
    "networks": "title",
}

TEXT_DATASET_LOAD_KEYS: Dict[str, Tuple[str, ...]] = {
    "pubmed": ("pubmed_text", "publications", "pubmed"),
    "wiki": ("wiki", "neurowiki"),
    "cogatlas": ("cogatlas",),
    "cogatlas_task": ("cogatlas_task",),
    "cogatlas_disorder": ("cogatlas_disorder",),
}
TEXT_LATENT_LOAD_KEYS: Dict[str, Tuple[str, ...]] = {
    "pubmed": ("pubmed_text", "publications", "pubmed"),
    "wiki": ("wiki", "neurowiki"),
    "cogatlas": ("cogatlas",),
    "cogatlas_task": ("cogatlas_task",),
    "cogatlas_disorder": ("cogatlas_disorder",),
}


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    """Row-wise L2 normalization with numerical guard."""
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


@dataclass
class BrainTopKResult:
    """Chainable wrapper around a top-k brain result table."""

    table: pd.DataFrame
    parent: "BrainSearchResult"

    def to_nifti(self) -> List[nib.Nifti1Image]:
        """Decode NIfTI images aligned with this top-k table."""
        return self.parent.niftis_for(self.table)

    def to_dataset_nifti(self) -> List[nib.Nifti1Image]:
        """Load dataset-native NIfTI images aligned with this top-k table."""
        return self.parent.dataset_niftis_for(self.table)

    def plot_row(
        self,
        index: int = 0,
        source: Literal["dataset", "decoded"] = "dataset",
        threshold: Union[str, float] = "auto",
        display_mode: str = "ortho",
        cut_coords: Optional[Sequence[float]] = None,
        colorbar: bool = True,
        title: Optional[str] = None,
    ) -> Any:
        """Plot one ranked row from this top-k output."""
        return self.parent.plot_topk_row(
            self.table,
            index=index,
            source=source,
            threshold=threshold,
            display_mode=display_mode,
            cut_coords=cut_coords,
            colorbar=colorbar,
            title=title,
        )

    def to_pandas(self) -> pd.DataFrame:
        """Return the underlying pandas dataframe."""
        return self.table

    def __getattr__(self, name: str) -> Any:
        return getattr(self.table, name)

    def __getitem__(self, key: Any) -> Any:
        return self.table.__getitem__(key)

    def __len__(self) -> int:
        return len(self.table)

    def __repr__(self) -> str:
        return repr(self.table)


@dataclass
class TextSearchResult:
    """Container for text retrieval scores and metadata."""

    scores_by_dataset: Dict[str, torch.Tensor]
    metadata_by_dataset: Dict[str, pd.DataFrame]
    query_embeddings: torch.Tensor
    retrieval_space: Literal["raw_text", "shared"]

    def top_k(
        self,
        k: int = 5,
        query_index: Optional[int] = None,
        dataset: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return top-k text matches as one merged dataframe.

        Parameters
        ----------
        k : int, optional
            Number of matches to return per query.
        query_index : int, optional
            Query index when running multi-query retrieval. If None, returns
            top-k per query for all queries.
        dataset : str, optional
            Restrict ranking to one dataset. If None, return top-k per dataset.

        Returns
        -------
        pandas.DataFrame
            Columns include:
            - ``dataset``
            - ``title``
            - ``description``
            - ``cosine_similarity``
            and ``query_index`` when multiple queries are present.
            When ``dataset=None``, each dataset contributes up to ``k`` rows.
        """
        if k <= 0:
            raise ValueError("k must be > 0")

        datasets = list(self.scores_by_dataset.keys())
        if dataset is not None:
            if dataset not in self.scores_by_dataset:
                raise ValueError(f"Unknown dataset '{dataset}'. Available: {datasets}")
            datasets = [dataset]

        n_queries = int(self.query_embeddings.shape[0])
        query_ids = self._resolve_query_indices(query_index, n_queries)

        out_blocks: List[pd.DataFrame] = []
        for q_idx in query_ids:
            candidates: List[pd.DataFrame] = []
            for ds in datasets:
                scores = self.scores_by_dataset[ds][:, q_idx]
                n = min(int(k), int(scores.shape[0]))
                if n == 0:
                    continue
                top_idx = torch.topk(scores, k=n, largest=True, sorted=True).indices
                meta = self.metadata_by_dataset[ds].iloc[top_idx.cpu().numpy()]
                part = pd.DataFrame(
                    {
                        "dataset": ds,
                        "title": meta["title"].astype(str).to_numpy(),
                        "description": meta["description"].astype(str).to_numpy(),
                        "cosine_similarity": scores[top_idx].cpu().numpy(),
                    }
                )
                part = part.sort_values("cosine_similarity", ascending=False).reset_index(drop=True)
                candidates.append(part)

            if not candidates:
                continue

            merged = pd.concat(candidates, ignore_index=True)
            if n_queries > 1:
                merged.insert(0, "query_index", q_idx)
            out_blocks.append(merged)

        if not out_blocks:
            columns = ["dataset", "title", "description", "cosine_similarity"]
            if n_queries > 1:
                columns = ["query_index"] + columns
            return pd.DataFrame(columns=columns)

        out = pd.concat(out_blocks, ignore_index=True)
        if n_queries > 1:
            out = out.sort_values(["query_index", "dataset", "cosine_similarity"], ascending=[True, True, False]).reset_index(drop=True)
        else:
            out = out.sort_values(["dataset", "cosine_similarity"], ascending=[True, False]).reset_index(drop=True)
        return out

    def format(
        self,
        k: int = 5,
        query_index: Optional[int] = None,
        dataset: Optional[str] = None,
    ) -> str:
        """Build a printable summary for top-k text results."""
        table = self.top_k(k=k, query_index=query_index, dataset=dataset)
        return table.to_string(index=False)

    def print(
        self,
        k: int = 5,
        query_index: Optional[int] = None,
        dataset: Optional[str] = None,
    ) -> None:
        """Print top-k text results."""
        print(self.format(k=k, query_index=query_index, dataset=dataset))

    def plot(self, *args: Any, **kwargs: Any) -> Any:
        """Text results are not directly plottable as brain maps."""
        raise ValueError("plot is only available for brain outputs.")

    @property
    def df(self) -> pd.DataFrame:
        """Return a dataframe view of ranked text retrieval results."""
        if not self.scores_by_dataset:
            return pd.DataFrame(columns=["dataset", "title", "description", "cosine_similarity"])
        k = max(int(scores.shape[0]) for scores in self.scores_by_dataset.values())
        return self.top_k(k=k)

    @staticmethod
    def _resolve_query_indices(query_index: Optional[int], n_queries: int) -> List[int]:
        """Validate and resolve query index selection."""
        if query_index is None:
            return list(range(n_queries))
        if query_index < 0 or query_index >= n_queries:
            raise IndexError(f"query_index={query_index} out of range for {n_queries} queries.")
        return [query_index]


@dataclass
class BrainSearchResult:
    """Container for brain retrieval or generation outputs."""

    scores: Optional[torch.Tensor]
    metadata: Optional[pd.DataFrame]
    latents: Optional[torch.Tensor]
    query_embeddings: torch.Tensor
    retrieval_space: Literal["mse", "infonce"]
    generated_flatmaps: Optional[torch.Tensor] = None
    masker: Any = None
    decoder: Any = None
    dataset_image_getter: Optional[Callable[[str, int], nib.Nifti1Image]] = None

    def top_k(self, k: int = 5, query_index: Optional[int] = None) -> BrainTopKResult:
        """Return top-k brain matches.

        Parameters
        ----------
        k : int, optional
            Number of matches to return per query.
        query_index : int, optional
            Query index when running multi-query retrieval. If None, returns
            top-k for every query.

        Returns
        -------
        BrainTopKResult
            Chainable table-like result with ranked rows.
            Rows include ``dataset_index`` so they can be plotted via
            ``plot_topk_item`` or ``BrainTopKResult.plot_row``.
        """
        if self.scores is None or self.metadata is None:
            raise ValueError("top_k is only available for retrieval outputs (e.g., head='infonce').")

        if k <= 0:
            raise ValueError("k must be > 0")

        n_queries = int(self.query_embeddings.shape[0])
        query_ids = TextSearchResult._resolve_query_indices(query_index, n_queries)

        out_blocks: List[pd.DataFrame] = []
        for q_idx in query_ids:
            score_vec = self.scores[:, q_idx]
            if "dataset" in self.metadata.columns:
                per_dataset: List[pd.DataFrame] = []
                for ds in sorted(self.metadata["dataset"].astype(str).unique().tolist()):
                    ds_idx_np = np.where(self.metadata["dataset"].astype(str).to_numpy() == ds)[0]
                    if ds_idx_np.size == 0:
                        continue
                    ds_idx = torch.as_tensor(ds_idx_np, dtype=torch.long)
                    ds_scores = score_vec[ds_idx]
                    n = min(int(k), int(ds_scores.shape[0]))
                    if n == 0:
                        continue
                    local_top = torch.topk(ds_scores, k=n, largest=True, sorted=True).indices
                    top_idx = ds_idx[local_top]
                    top_meta = self.metadata.iloc[top_idx.cpu().numpy()].copy()
                    top_meta["cosine_similarity"] = score_vec[top_idx].cpu().numpy()
                    top_meta["_brain_index"] = top_idx.cpu().numpy()
                    per_dataset.append(top_meta)

                if not per_dataset:
                    continue
                top_meta = pd.concat(per_dataset, ignore_index=True)
            else:
                n = min(int(k), int(score_vec.shape[0]))
                top_idx = torch.topk(score_vec, k=n, largest=True, sorted=True).indices
                top_meta = self.metadata.iloc[top_idx.cpu().numpy()].copy()
                top_meta["cosine_similarity"] = score_vec[top_idx].cpu().numpy()
                top_meta["_brain_index"] = top_idx.cpu().numpy()

            if "dataset" not in top_meta.columns:
                top_meta["dataset"] = "pubmed"
            if "_dataset_index" not in top_meta.columns:
                top_meta["_dataset_index"] = np.arange(len(top_meta), dtype=np.int64)
            if "title" not in top_meta.columns:
                top_meta["title"] = ""
            if "description" not in top_meta.columns:
                top_meta["description"] = ""
            if n_queries > 1:
                top_meta.insert(0, "query_index", q_idx)
            keep = ["dataset", "_dataset_index", "title", "description", "cosine_similarity", "_brain_index"]
            if n_queries > 1:
                keep = ["query_index"] + keep
            top_meta = top_meta.loc[:, keep]
            out_blocks.append(top_meta)

        if not out_blocks:
            columns = ["dataset", "dataset_index", "title", "description", "cosine_similarity"]
            if n_queries > 1:
                columns = ["query_index"] + columns
            out = pd.DataFrame(columns=columns)
            out.attrs["brain_indices"] = []
            out.attrs["dataset_names"] = []
            out.attrs["dataset_indices"] = []
            return BrainTopKResult(table=out, parent=self)

        out = pd.concat(out_blocks, ignore_index=True)
        if n_queries > 1:
            out = out.sort_values(["query_index", "dataset", "cosine_similarity"], ascending=[True, True, False]).reset_index(drop=True)
        else:
            out = out.sort_values(["dataset", "cosine_similarity"], ascending=[True, False]).reset_index(drop=True)
        brain_indices = out["_brain_index"].astype(int).tolist()
        dataset_names = out["dataset"].astype(str).tolist()
        dataset_indices = out["_dataset_index"].astype(int).tolist()
        out = out.drop(columns=["_brain_index"]).rename(columns={"_dataset_index": "dataset_index"})
        out.attrs["brain_indices"] = brain_indices
        out.attrs["dataset_names"] = dataset_names
        out.attrs["dataset_indices"] = dataset_indices
        return BrainTopKResult(table=out, parent=self)

    def print(self, k: int = 5, query_index: Optional[int] = None) -> None:
        """Print top-k brain results."""
        print(self.top_k(k=k, query_index=query_index).to_string(index=False))

    @property
    def df(self) -> pd.DataFrame:
        """Return a dataframe view of ranked brain retrieval results."""
        if self.scores is None or self.metadata is None:
            raise ValueError("df is only available for retrieval outputs (e.g., head='infonce').")
        return self.top_k(k=int(self.scores.shape[0])).to_pandas()

    @property
    def images(self) -> List[nib.Nifti1Image]:
        """Return generated NIfTI images."""
        if self.generated_flatmaps is None:
            raise ValueError("images is only available for generated outputs (head='mse').")
        return self.to_nifti()

    def niftis_for(self, table: Union[pd.DataFrame, BrainTopKResult]) -> List[nib.Nifti1Image]:
        """Decode NIfTI images aligned with rows from ``top_k`` output."""
        if isinstance(table, BrainTopKResult):
            table = table.table
        if self.masker is None:
            raise ValueError("Masker is not attached to this result.")
        brain_indices = table.attrs.get("brain_indices")
        if brain_indices is None:
            raise ValueError("Expected a DataFrame returned by `top_k` (missing row-to-brain index mapping).")
        return self._decode_nifti_indices(brain_indices)

    def dataset_niftis_for(self, table: Union[pd.DataFrame, BrainTopKResult]) -> List[nib.Nifti1Image]:
        """Load dataset-native NIfTI images aligned with rows from ``top_k`` output."""
        if isinstance(table, BrainTopKResult):
            table = table.table
        if self.dataset_image_getter is None:
            raise ValueError("Dataset image loading is not attached to this result.")

        dataset_names = table.attrs.get("dataset_names")
        dataset_indices = table.attrs.get("dataset_indices")
        if dataset_names is None or dataset_indices is None:
            if "dataset" in table.columns and "dataset_index" in table.columns:
                dataset_names = table["dataset"].astype(str).tolist()
                dataset_indices = table["dataset_index"].astype(int).tolist()
            else:
                raise ValueError("Expected a DataFrame returned by `top_k` with dataset row mapping.")

        out: List[nib.Nifti1Image] = []
        for ds_name, ds_index in zip(dataset_names, dataset_indices):
            out.append(self.dataset_image_getter(str(ds_name), int(ds_index)))
        return out

    @staticmethod
    def _dataset_row_ref(table: pd.DataFrame, index: int) -> Tuple[str, int]:
        """Resolve (dataset, dataset_index) tuple for one top-k row."""
        dataset_names = table.attrs.get("dataset_names")
        dataset_indices = table.attrs.get("dataset_indices")
        if dataset_names is not None and dataset_indices is not None:
            return str(dataset_names[index]), int(dataset_indices[index])
        if "dataset" in table.columns and "dataset_index" in table.columns:
            return str(table.iloc[index]["dataset"]), int(table.iloc[index]["dataset_index"])
        raise ValueError("Expected a DataFrame returned by `top_k` with dataset row mapping.")

    def plot_topk_row(
        self,
        table: Union[pd.DataFrame, BrainTopKResult],
        index: int = 0,
        source: Literal["dataset", "decoded"] = "dataset",
        threshold: Union[str, float] = "auto",
        display_mode: str = "ortho",
        cut_coords: Optional[Sequence[float]] = None,
        colorbar: bool = True,
        title: Optional[str] = None,
    ) -> Any:
        """Plot one row from a ``top_k`` table using dataset-native or decoded image."""
        if isinstance(table, BrainTopKResult):
            table = table.table
        if index < 0 or index >= len(table):
            raise IndexError(f"index={index} out of range for {len(table)} rows.")

        row = table.iloc[index]
        if source == "dataset":
            if self.dataset_image_getter is None:
                raise ValueError("Dataset image loading is not attached to this result.")
            ds_name, ds_index = self._dataset_row_ref(table, index)
            image = self.dataset_image_getter(ds_name, ds_index)
        else:
            brain_indices = table.attrs.get("brain_indices")
            if brain_indices is None:
                raise ValueError("Decoded plotting expects a DataFrame returned by `top_k`.")
            image = self.to_nifti(index=int(brain_indices[index]))

        if title is None and "title" in row and pd.notna(row["title"]):
            title = str(row["title"])
            if len(title) > 55:
                title = title[:55] + "..."
        return NeuroVLM.plot(
            image=image,
            threshold=threshold,
            display_mode=display_mode,
            cut_coords=cut_coords,
            colorbar=colorbar,
            title=title,
        )

    def plot_topk_item(
        self,
        row: pd.Series,
        threshold: Union[str, float] = "auto",
        display_mode: str = "ortho",
        cut_coords: Optional[Sequence[float]] = None,
        colorbar: bool = True,
        title: Optional[str] = None,
    ) -> Any:
        """Plot one row selected from ``results.top_k(...).iloc[...]``."""
        if not isinstance(row, pd.Series):
            raise TypeError("plot_topk_item expects a pandas.Series from `top_k(...).iloc[...]`.")
        if self.dataset_image_getter is None:
            raise ValueError("Dataset image loading is not attached to this result.")
        if "dataset" not in row.index or "dataset_index" not in row.index:
            raise ValueError("Row must include `dataset` and `dataset_index` columns.")

        image = self.dataset_image_getter(str(row["dataset"]), int(row["dataset_index"]))
        if title is None and "title" in row.index and pd.notna(row["title"]):
            title = str(row["title"])
        return NeuroVLM.plot(
            image=image,
            threshold=threshold,
            display_mode=display_mode,
            cut_coords=cut_coords,
            colorbar=colorbar,
            title=title,
        )

    def top_k_niftis(
        self,
        k: int = 5,
        query_index: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, List[nib.Nifti1Image]]:
        """Return ``top_k`` table and its row-aligned NIfTI images."""
        top = self.top_k(k=k, query_index=query_index)
        return top.table, top.to_nifti()

    def to_nifti(
        self,
        index: Optional[int] = None,
    ) -> Union[nib.Nifti1Image, List[nib.Nifti1Image]]:
        """Convert brain outputs to NIfTI images.

        Parameters
        ----------
        index : int, optional
            If provided, return one image. Otherwise return one image per query.

        Returns
        -------
        nibabel.nifti1.Nifti1Image or list[nibabel.nifti1.Nifti1Image]
            Brain map image(s).
        """
        if self.masker is None:
            raise ValueError("Masker is not attached to this result.")

        if self.generated_flatmaps is not None:
            total = int(self.generated_flatmaps.shape[0])
        elif self.latents is not None:
            total = int(self.latents.shape[0])
        else:
            raise ValueError("No generated maps or latent vectors available on this result.")

        if index is not None:
            if index < 0 or index >= total:
                raise IndexError(f"index={index} out of range for {total} maps.")
            images = self._decode_nifti_indices([index])
            return images[0]

        return self._decode_nifti_indices(list(range(total)))

    def _decode_nifti_indices(self, indices: Sequence[int]) -> List[nib.Nifti1Image]:
        """Decode selected row indices into NIfTI images."""
        if self.masker is None:
            raise ValueError("Masker is not attached to this result.")

        images: List[nib.Nifti1Image] = []
        if self.generated_flatmaps is not None:
            maps = self.generated_flatmaps.detach().cpu()
            for idx in indices:
                images.append(self.masker.inverse_transform(maps[idx:idx + 1].numpy()))
            return images

        if self.latents is None:
            raise ValueError("No latent vectors attached to this result.")
        if self.decoder is None:
            raise ValueError("Decoder is not attached to this result.")

        with torch.no_grad():
            for idx in indices:
                flat = torch.sigmoid(self.decoder(self.latents[idx:idx + 1])).detach().cpu().numpy()
                images.append(self.masker.inverse_transform(flat))
        return images

    def plot(
        self,
        index: int = 0,
        query_index: int = 0,
        threshold: Union[str, float] = "auto",
        display_mode: str = "ortho",
        cut_coords: Optional[Sequence[float]] = None,
        colorbar: bool = True,
        title: Optional[str] = None,
    ) -> Any:
        """Plot one brain image from this result."""
        if index < 0:
            raise ValueError("index must be >= 0")

        if self.generated_flatmaps is not None:
            total = int(self.generated_flatmaps.shape[0])
            if index >= total:
                raise IndexError(f"index={index} out of range for {total} generated maps.")
            image = self.to_nifti(index=index)
        else:
            if self.scores is None:
                raise ValueError("plot is only available when brain images can be decoded from this result.")
            n_queries = int(self.scores.shape[1])
            if query_index < 0 or query_index >= n_queries:
                raise IndexError(f"query_index={query_index} out of range for {n_queries} queries.")
            score_vec = self.scores[:, query_index]
            if index >= int(score_vec.shape[0]):
                raise IndexError(f"index={index} out of range for {int(score_vec.shape[0])} candidates.")
            ordered = torch.topk(score_vec, k=index + 1, largest=True, sorted=True).indices
            corpus_index = int(ordered[index].item())
            image = self.to_nifti(index=corpus_index)
            if title is None and self.metadata is not None:
                row = self.metadata.iloc[corpus_index]
                if "title" in row and pd.notna(row["title"]):
                    title = str(row["title"])
                elif "pmid" in row and pd.notna(row["pmid"]):
                    title = f"PMID {row['pmid']}"

        return NeuroVLM.plot(
            image=image,
            threshold=threshold,
            display_mode=display_mode,
            cut_coords=cut_coords,
            colorbar=colorbar,
            title=title,
        )


@dataclass
class _QueryBuilder:
    """Chainable query wrapper used by ``NeuroVLM.text(...)`` and ``NeuroVLM.brain(...)``."""

    parent: "NeuroVLM"
    payload: Any
    source: Literal["text", "brain"]

    def to_brain(
        self,
        head: Literal["mse", "infonce"] = "mse",
        project: bool = True,
        dataset: Optional[Union[str, Sequence[str]]] = None,
    ) -> BrainSearchResult:
        """Run text/brain query against the brain target space."""
        return self.parent.to_brain(self.payload, head=head, project=project, dataset=dataset)

    def to_text(
        self,
        head: Literal["mse", "infonce"] = "infonce",
        datasets: Optional[Sequence[str]] = None,
    ) -> TextSearchResult:
        """Run text/brain query against the text target space."""
        if head != "infonce":
            raise ValueError("to_text only supports head='infonce'.")
        return self.parent.to_text(self.payload, datasets=datasets, project=True)


class NeuroVLM:
    """Unified interface for text-to-brain and brain-to-text retrieval.

    Parameters
    ----------
    datasets : sequence of str, optional
        Text corpora to include for ``to_text``.
    device : str, optional
        Torch device for inference.
    """

    def __init__(
        self,
        datasets: Optional[Sequence[str]] = None,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)

        self.datasets = tuple(self._canonicalize_datasets(datasets or DEFAULT_TEXT_DATASETS))

        self._proj_head_image = None
        self._proj_head_text_infonce = None
        self._proj_head_text_mse = None
        self._specter = None
        self._autoencoder = None
        self._masker = None

        self._text_raw_embeddings: Dict[str, torch.Tensor] = {}
        self._text_shared_embeddings: Dict[str, torch.Tensor] = {}
        self._text_metadata: Dict[str, pd.DataFrame] = {}

        self._brain_latent: Optional[torch.Tensor] = None
        self._brain_latent_cpu: Optional[torch.Tensor] = None
        self._brain_embeddings_mse: Optional[torch.Tensor] = None
        self._brain_embeddings_infonce: Optional[torch.Tensor] = None
        self._brain_metadata: Optional[pd.DataFrame] = None
        self._brain_pmids: Optional[np.ndarray] = None
        self._network_brain_latent: Optional[torch.Tensor] = None
        self._network_brain_latent_cpu: Optional[torch.Tensor] = None
        self._network_brain_embeddings_infonce: Optional[torch.Tensor] = None
        self._network_brain_metadata: Optional[pd.DataFrame] = None
        self._neurovault_brain_latent: Optional[torch.Tensor] = None
        self._neurovault_brain_latent_cpu: Optional[torch.Tensor] = None
        self._neurovault_brain_embeddings_infonce: Optional[torch.Tensor] = None
        self._neurovault_brain_metadata: Optional[pd.DataFrame] = None

        self._pubmed_images_flat: Optional[torch.Tensor] = None
        self._pubmed_image_pmids: Optional[np.ndarray] = None
        self._pubmed_pmid_to_image_index: Optional[Dict[Any, int]] = None
        self._neurovault_images_flat: Optional[torch.Tensor] = None

        self._pub_df: Optional[pd.DataFrame] = None
        self._pub_lookup: Optional[pd.DataFrame] = None

        self.last_text_result: Optional[TextSearchResult] = None
        self.last_brain_result: Optional[BrainSearchResult] = None
        self.last_generated_brain_flat: Optional[torch.Tensor] = None
        self._last_result: Optional[Union[TextSearchResult, BrainSearchResult]] = None

    @property
    def results(self) -> Union[TextSearchResult, BrainSearchResult]:
        """Return the latest retrieval/generation result object."""
        if self._last_result is None:
            raise ValueError("No results available yet. Run a query first.")
        return self._last_result

    def text(self, X: Any) -> _QueryBuilder:
        """Start a text-driven chain."""
        return _QueryBuilder(parent=self, payload=X, source="text")

    def brain(self, X: Any) -> _QueryBuilder:
        """Start a brain-driven chain."""
        return _QueryBuilder(parent=self, payload=X, source="brain")

    def to_text(
        self,
        X: Any,
        datasets: Optional[Sequence[str]] = None,
        project: bool = True,
    ) -> TextSearchResult:
        """Retrieve text results for one or many queries.

        Parameters
        ----------
        X : Any
            Query payload. Supported:
            - Raw text payloads (`str`, `dict`, `pandas.DataFrame`, list of strings/dicts)
            - Text embeddings of shape ``(768,)`` or ``(N, 768)``
            - Brain latent embeddings of shape ``(384,)`` or ``(N, 384)``
            - Flattened brain vectors of shape ``(28542,)`` or ``(N, 28542)``
            - ``nibabel.Nifti1Image``
        datasets : sequence of str, optional
            Optional subset of initialized text datasets.
        project : bool, optional
            Whether to use projection heads. Required for brain-to-text retrieval.

        Returns
        -------
        TextSearchResult
            Retrieval object exposing ``top_k``/``print``.
        """
        query, retrieval_space = self._prepare_text_query(X, project=project)
        dataset_names = self._resolve_active_datasets(datasets)
        self._ensure_text_indices(dataset_names, require_shared=(retrieval_space == "shared"))

        scores_by_dataset: Dict[str, torch.Tensor] = {}
        metadata_by_dataset: Dict[str, pd.DataFrame] = {}
        for ds in dataset_names:
            corpus = self._text_raw_embeddings[ds] if retrieval_space == "raw_text" else self._text_shared_embeddings[ds]
            scores = corpus @ query.T
            scores_by_dataset[ds] = scores.detach().cpu()
            metadata_by_dataset[ds] = self._text_metadata[ds]

        result = TextSearchResult(
            scores_by_dataset=scores_by_dataset,
            metadata_by_dataset=metadata_by_dataset,
            query_embeddings=query.detach().cpu(),
            retrieval_space=retrieval_space,
        )
        self.last_text_result = result
        self._last_result = result
        return result

    def to_brain(
        self,
        X: Any,
        head: Literal["mse", "infonce"] = "mse",
        project: bool = True,
        dataset: Optional[Union[str, Sequence[str]]] = None,
    ) -> BrainSearchResult:
        """Retrieve brain results for one or many queries.

        Parameters
        ----------
        X : Any
            Query payload with same supported types as ``to_text``.
        head : {"mse", "infonce"}, optional
            Projection mode for text-to-brain retrieval.
        project : bool, optional
            Whether to use projection heads. Required for text-to-brain retrieval.
        dataset : str or sequence of str, optional
            Retrieval corpus/corpora for InfoNCE text-to-brain search.
            Defaults to ``("pubmed", "networks", "neurovault")`` for InfoNCE.
            Ignored when ``head="mse"`` because MSE directly generates
            brain maps from projected text.

        Returns
        -------
        BrainSearchResult
            Retrieval object exposing ``top_k``/``print``.
        """
        self._validate_head(head)
        query, retrieval_space = self._prepare_brain_query(X, head=head, project=project)

        # MSE path: decode projected text latents directly to brain flatmaps.
        if retrieval_space == "mse":
            self._ensure_autoencoder()
            self._ensure_masker()
            with torch.no_grad():
                flatmaps = torch.sigmoid(self._autoencoder.decoder(query.to(self.device))).detach().cpu()

            self.last_generated_brain_flat = flatmaps
            result = BrainSearchResult(
                scores=None,
                metadata=None,
                latents=query.detach().cpu(),
                query_embeddings=query.detach().cpu(),
                retrieval_space="mse",
                generated_flatmaps=flatmaps,
                masker=self._masker,
                decoder=self._autoencoder.decoder,
                dataset_image_getter=self._get_dataset_nifti,
            )
            self.last_brain_result = result
            self._last_result = result
            return result

        # InfoNCE path: retrieve over one or more brain corpora.
        corpora = self._resolve_brain_datasets(dataset)
        embeddings_list: List[torch.Tensor] = []
        metadata_list: List[pd.DataFrame] = []
        latents_list: List[torch.Tensor] = []

        for corpus_name in corpora:
            if corpus_name == "neuro":
                self._ensure_brain_index(require_infonce=True)
                assert self._brain_embeddings_infonce is not None
                assert self._brain_metadata is not None
                assert self._brain_latent_cpu is not None
                emb = self._brain_embeddings_infonce
                meta = self._brain_metadata.copy()
                lat = self._brain_latent_cpu
                meta["dataset"] = "pubmed"
            elif corpus_name == "networks":
                self._ensure_network_brain_index()
                assert self._network_brain_embeddings_infonce is not None
                assert self._network_brain_metadata is not None
                assert self._network_brain_latent_cpu is not None
                emb = self._network_brain_embeddings_infonce
                meta = self._network_brain_metadata.copy()
                lat = self._network_brain_latent_cpu
                meta["dataset"] = "networks"
            elif corpus_name == "neurovault":
                self._ensure_neurovault_brain_index()
                assert self._neurovault_brain_embeddings_infonce is not None
                assert self._neurovault_brain_metadata is not None
                assert self._neurovault_brain_latent_cpu is not None
                emb = self._neurovault_brain_embeddings_infonce
                meta = self._neurovault_brain_metadata.copy()
                lat = self._neurovault_brain_latent_cpu
                meta["dataset"] = "neurovault"
            else:
                raise ValueError(f"Unsupported corpus '{corpus_name}'.")

            if "title" not in meta.columns:
                meta["title"] = ""
            if "description" not in meta.columns:
                meta["description"] = ""
            if "_dataset_index" not in meta.columns:
                meta["_dataset_index"] = np.arange(len(meta), dtype=np.int64)
            embeddings_list.append(emb)
            metadata_list.append(meta[["dataset", "_dataset_index", "title", "description"]].reset_index(drop=True))
            latents_list.append(lat)

        corpus = torch.vstack(embeddings_list)
        metadata = pd.concat(metadata_list, ignore_index=True)
        latents = torch.vstack(latents_list)
        scores = corpus @ query.T

        self._ensure_autoencoder()
        self._ensure_masker()
        result = BrainSearchResult(
            scores=scores.detach().cpu(),
            metadata=metadata,
            latents=latents,
            query_embeddings=query.detach().cpu(),
            retrieval_space="infonce",
            generated_flatmaps=None,
            masker=self._masker,
            decoder=self._autoencoder.decoder,
            dataset_image_getter=self._get_dataset_nifti,
        )
        self.last_brain_result = result
        self.last_generated_brain_flat = None
        self._last_result = result
        return result

    def top_k(
        self,
        k: int = 5,
        target: Literal["text", "brain"] = "text",
        query_index: Optional[int] = None,
        dataset: Optional[str] = None,
    ) -> Union[pd.DataFrame, BrainTopKResult]:
        """Return top-k rows from the latest retrieval."""
        if target == "text":
            if self.last_text_result is None:
                raise ValueError("No text retrieval available. Call `to_text` first.")
            return self.last_text_result.top_k(k=k, query_index=query_index, dataset=dataset)

        if dataset is not None:
            raise ValueError("`dataset` is only valid for text retrieval.")
        if self.last_brain_result is None:
            raise ValueError("No brain retrieval available. Call `to_brain` first.")
        return self.last_brain_result.top_k(k=k, query_index=query_index)

    @staticmethod
    def plot(
        image: nib.Nifti1Image,
        threshold: Union[str, float] = "auto",
        display_mode: str = "ortho",
        cut_coords: Optional[Sequence[float]] = None,
        colorbar: bool = True,
        title: Optional[str] = None,
    ) -> Any:
        """Plot a provided NIfTI image."""
        if not isinstance(image, nib.Nifti1Image):
            raise TypeError("plot expects a nib.Nifti1Image.")
        from nilearn.plotting import plot_stat_map

        return plot_stat_map(
            image,
            threshold=threshold,
            display_mode=display_mode,
            cut_coords=cut_coords,
            colorbar=colorbar,
            title=title,
            draw_cross=False,
            cmap="hot"
        )

    def get_niftis(
        self,
        index: Optional[int] = None,
    ) -> Union[nib.Nifti1Image, List[nib.Nifti1Image]]:
        """Return NIfTI image(s) from the latest generated (MSE) brain maps."""
        if self.last_brain_result is None:
            raise ValueError("No brain result available. Call `to_brain` first.")
        return self.last_brain_result.to_nifti(index=index)

    def decode_brain(self, latent: Union[torch.Tensor, np.ndarray]) -> nib.Nifti1Image:
        """Decode one latent brain vector into NIfTI."""
        self._ensure_autoencoder()
        self._ensure_masker()
        latent_tensor = self._as_2d_tensor(latent)
        if latent_tensor.shape[1] != LATENT_DIM:
            raise ValueError(f"Expected latent dim {LATENT_DIM}, got {latent_tensor.shape[1]}.")

        with torch.no_grad():
            decoded = self._autoencoder.decoder(latent_tensor.to(self.device))
            decoded = torch.sigmoid(decoded)
        return self._masker.inverse_transform(decoded.detach().cpu().numpy())

    def decode_brains(self, latents: Union[torch.Tensor, np.ndarray]) -> List[nib.Nifti1Image]:
        """Decode multiple latent brain vectors into NIfTI images."""
        latent_tensor = self._as_2d_tensor(latents)
        return [self.decode_brain(latent_tensor[i]) for i in range(latent_tensor.shape[0])]

    def _prepare_text_query(
        self,
        X: Any,
        project: bool,
    ) -> Tuple[torch.Tensor, Literal["raw_text", "shared"]]:
        """Prepare query vectors for text retrieval."""
        if self._is_text_payload(X):
            text_emb = self._encode_text(X)
            if project:
                self._ensure_projection_heads()
                with torch.no_grad():
                    text_emb = self._proj_head_text_infonce(text_emb.to(self.device))
                return _l2_normalize(text_emb), "shared"
            return _l2_normalize(text_emb.to(self.device)), "raw_text"

        if isinstance(X, nib.Nifti1Image):
            if not project:
                raise ValueError("brain-to-text retrieval requires `project=True`.")
            latent = self._encode_brain_image(X)
            self._ensure_projection_heads()
            with torch.no_grad():
                latent = self._proj_head_image(latent.to(self.device))
            return _l2_normalize(latent), "shared"

        tensor = self._coerce_numeric_query(X)
        if tensor is None:
            raise TypeError(
                "Unsupported query type for to_text. Use text, tensor/array embeddings, or NIfTI image."
            )

        dim = int(tensor.shape[1])
        if dim == TEXT_EMBED_DIM:
            if project:
                self._ensure_projection_heads()
                with torch.no_grad():
                    tensor = self._proj_head_text_infonce(tensor.to(self.device))
                return _l2_normalize(tensor), "shared"
            return _l2_normalize(tensor.to(self.device)), "raw_text"

        if dim == BRAIN_FLAT_DIM:
            if not project:
                raise ValueError("brain-to-text retrieval requires `project=True`.")
            latent = self._encode_brain_flat(tensor)
            self._ensure_projection_heads()
            with torch.no_grad():
                latent = self._proj_head_image(latent.to(self.device))
            return _l2_normalize(latent), "shared"

        if dim == LATENT_DIM:
            if not project:
                raise ValueError("brain-to-text retrieval requires `project=True`.")
            self._ensure_projection_heads()
            with torch.no_grad():
                latent = self._proj_head_image(tensor.to(self.device))
            return _l2_normalize(latent), "shared"

        raise ValueError(
            f"Unsupported embedding dim {dim}. Expected {TEXT_EMBED_DIM}, {LATENT_DIM}, or {BRAIN_FLAT_DIM}."
        )

    def _prepare_brain_query(
        self,
        X: Any,
        head: Literal["mse", "infonce"],
        project: bool,
    ) -> Tuple[torch.Tensor, Literal["mse", "infonce"]]:
        """Prepare query vectors for brain retrieval."""
        if self._is_text_payload(X):
            if not project:
                raise ValueError("text-to-brain retrieval requires `project=True`.")
            text_emb = self._encode_text(X)
            return self._project_text_to_brain_space(text_emb, head=head)

        if isinstance(X, nib.Nifti1Image):
            latent = self._encode_brain_image(X)
            return self._project_brain_latent_for_brain_search(latent, head=head, project=project)

        tensor = self._coerce_numeric_query(X)
        if tensor is None:
            raise TypeError(
                "Unsupported query type for to_brain. Use text, tensor/array embeddings, or NIfTI image."
            )

        dim = int(tensor.shape[1])
        if dim == TEXT_EMBED_DIM:
            if not project:
                raise ValueError("text-to-brain retrieval requires `project=True`.")
            return self._project_text_to_brain_space(tensor.to(self.device), head=head)

        if dim == BRAIN_FLAT_DIM:
            latent = self._encode_brain_flat(tensor)
            return self._project_brain_latent_for_brain_search(latent, head=head, project=project)

        if dim == LATENT_DIM:
            return self._project_brain_latent_for_brain_search(tensor.to(self.device), head=head, project=project)

        raise ValueError(
            f"Unsupported embedding dim {dim}. Expected {TEXT_EMBED_DIM}, {LATENT_DIM}, or {BRAIN_FLAT_DIM}."
        )

    def _project_text_to_brain_space(
        self,
        text_embedding: torch.Tensor,
        head: Literal["mse", "infonce"],
    ) -> Tuple[torch.Tensor, Literal["mse", "infonce"]]:
        """Project text embedding into the requested brain retrieval space."""
        self._ensure_projection_heads()
        text_embedding = _l2_normalize(text_embedding.to(self.device))
        with torch.no_grad():
            if head == "mse":
                out = self._proj_head_text_mse(text_embedding.to(self.device))
                return out, "mse"
            out = self._proj_head_text_infonce(text_embedding.to(self.device))
            return _l2_normalize(out), "infonce"

    def _project_brain_latent_for_brain_search(
        self,
        latent: torch.Tensor,
        head: Literal["mse", "infonce"],
        project: bool,
    ) -> Tuple[torch.Tensor, Literal["mse", "infonce"]]:
        """Prepare latent brain embeddings for brain retrieval."""
        if head == "infonce" and project:
            self._ensure_projection_heads()
            with torch.no_grad():
                out = self._proj_head_image(latent.to(self.device))
            return _l2_normalize(out), "infonce"
        return latent.to(self.device), "mse"

    def _ensure_text_indices(self, datasets: Sequence[str], require_shared: bool) -> None:
        """Load and cache text corpora embeddings and aligned metadata."""
        for dataset in datasets:
            if dataset not in self._text_raw_embeddings:
                if dataset == "networks":
                    latent_payload = self._load_latent_compat("networks_text")
                    latent = self._extract_networks_canonical_text_latent(latent_payload)
                    latent = self._as_2d_tensor(latent).to(self.device)
                    table = self._load_dataset_compat("networks_canonical").copy()
                    metadata = self._build_networks_canonical_metadata(table, n_rows=int(latent.shape[0]))
                else:
                    latent, ids = self._load_text_latent_with_ids(dataset)
                    latent = self._as_2d_tensor(latent).to(self.device)
                    table = self._load_text_dataset_table(dataset).copy()
                    lookup = self._build_lookup(dataset, table)
                    metadata = self._build_aligned_text_metadata(dataset, np.asarray(ids), table, lookup)

                self._text_raw_embeddings[dataset] = _l2_normalize(latent)
                self._text_metadata[dataset] = metadata

            if require_shared and dataset not in self._text_shared_embeddings:
                self._ensure_projection_heads()
                with torch.no_grad():
                    shared = self._proj_head_text_infonce(self._text_raw_embeddings[dataset].to(self.device))
                self._text_shared_embeddings[dataset] = _l2_normalize(shared)

    def _ensure_brain_index(self, require_infonce: bool) -> None:
        """Load and cache brain corpus embeddings and aligned metadata."""
        if self._brain_latent is None:
            latent, pmids = self._unpack_latent_with_ids(
                self._load_latent_compat("pubmed_images", "neuro"),
                default_id_key="pmid",
            )
            latent = self._as_2d_tensor(latent).to(self.device)
            pmids = np.asarray(pmids)

            self._brain_latent = latent
            self._brain_latent_cpu = latent.detach().cpu()
            self._brain_embeddings_mse = _l2_normalize(latent)
            self._brain_pmids = pmids
            self._brain_metadata = self._build_aligned_brain_metadata(pmids)
            self._brain_metadata["_dataset_index"] = self._resolve_pubmed_dataset_indices(pmids)

        if require_infonce and self._brain_embeddings_infonce is None:
            self._ensure_projection_heads()
            with torch.no_grad():
                shared = self._proj_head_image(self._brain_latent.to(self.device))
            self._brain_embeddings_infonce = _l2_normalize(shared)

    def _ensure_network_brain_index(self) -> None:
        """Load and cache network-atlas latent brains for InfoNCE retrieval."""
        if self._network_brain_embeddings_infonce is not None:
            return

        raw_payload = self._load_latent_compat("networks_neuro")
        latent_tensor, metadata = self._flatten_networks_neuro_payload(raw_payload)
        metadata["_dataset_index"] = np.arange(len(metadata), dtype=np.int64)

        self._ensure_projection_heads()
        with torch.no_grad():
            shared = self._proj_head_image(latent_tensor.to(self.device))

        self._network_brain_latent = latent_tensor
        self._network_brain_latent_cpu = latent_tensor.detach().cpu()
        self._network_brain_embeddings_infonce = _l2_normalize(shared)
        self._network_brain_metadata = metadata

    def _ensure_neurovault_brain_index(self) -> None:
        """Load and cache NeuroVault latent brains for InfoNCE retrieval."""
        if self._neurovault_brain_embeddings_infonce is not None:
            return

        latent_payload = self._load_latent_compat("neurovault_images", "latent_neurovault_images")
        latent = self._extract_latent_tensor(latent_payload)
        latent_tensor = self._as_2d_tensor(latent).to(self.device)

        self._ensure_projection_heads()
        with torch.no_grad():
            shared = self._proj_head_image(latent_tensor.to(self.device))

        self._neurovault_brain_latent = latent_tensor
        self._neurovault_brain_latent_cpu = latent_tensor.detach().cpu()
        self._neurovault_brain_embeddings_infonce = _l2_normalize(shared)
        self._neurovault_brain_metadata = self._build_neurovault_brain_metadata(n_rows=int(latent_tensor.shape[0]))

    def _build_neurovault_brain_metadata(self, n_rows: int) -> pd.DataFrame:
        """Build NeuroVault metadata aligned with latent/image order."""
        images_meta = self._load_dataset_compat("neurovault_images_meta", "images_neurovault")
        if not isinstance(images_meta, pd.DataFrame):
            raise TypeError("Expected NeuroVault image metadata to be a pandas.DataFrame.")

        frame = images_meta.copy().reset_index(drop=True)
        text_meta = self._load_dataset_compat("neurovault_text", "publications_neurovault")
        if isinstance(text_meta, pd.DataFrame) and "doi" in frame.columns and "doi" in text_meta.columns:
            pubs = text_meta.copy().drop_duplicates("doi").reset_index(drop=True)
            frame = frame.merge(
                pubs,
                on="doi",
                how="left",
                suffixes=("", "_pub"),
            )

        rows: List[Dict[str, Any]] = []
        for idx in range(n_rows):
            row = frame.iloc[idx] if idx < len(frame) else None
            fallback = f"neurovault_{idx}"
            if row is not None:
                for key in ("name", "image_name", "id", "image_id"):
                    if key in row and pd.notna(row[key]):
                        candidate = self._clean_text(str(row[key]))
                        if candidate:
                            fallback = candidate
                            break
            title, description = self._extract_title_description(row, fallback_title=fallback)
            rows.append(
                {
                    "title": title,
                    "description": description,
                    "_dataset_index": idx,
                }
            )
        return pd.DataFrame(rows)

    def _build_aligned_text_metadata(
        self,
        dataset: str,
        ids: np.ndarray,
        table: pd.DataFrame,
        lookup: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Align dataset metadata rows with latent row order."""
        rows: List[Dict[str, str]] = []
        for i, item_id in enumerate(ids):
            row = None
            if lookup is not None:
                key = self._normalize_lookup_key(dataset, item_id)
                if key in lookup.index:
                    row = lookup.loc[key]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
            if row is None and i < len(table):
                row = table.iloc[i]
            title, description = self._extract_title_description(row, fallback_title=str(item_id))
            rows.append({"title": title, "description": description})
        return pd.DataFrame(rows)

    def _build_aligned_brain_metadata(self, pmids: np.ndarray) -> pd.DataFrame:
        """Align publication metadata rows with latent neuro row order."""
        rows: List[Dict[str, Any]] = []
        for i, pmid in enumerate(pmids):
            row = self._lookup_publication_row(pmid, i)
            title, description = self._extract_title_description(row, fallback_title=f"PMID {pmid}")
            rows.append({"pmid": pmid, "title": title, "description": description})
        return pd.DataFrame(rows)

    def _build_networks_canonical_metadata(self, table: pd.DataFrame, n_rows: int) -> pd.DataFrame:
        """Build canonical network text metadata aligned to latent row order."""
        frame = table.copy()
        title_col = "title" if "title" in frame.columns else ("network_standard_name" if "network_standard_name" in frame.columns else None)
        desc_col = "description" if "description" in frame.columns else None

        if title_col is None:
            raise ValueError("Network canonical dataframe must include a `title` column.")

        if desc_col is None:
            frame["description"] = ""
            desc_col = "description"

        frame = frame.loc[:, [title_col, desc_col]].rename(columns={title_col: "title", desc_col: "description"})
        frame["title"] = frame["title"].astype(str).map(self._clean_text)
        frame["description"] = frame["description"].astype(str).map(self._clean_text)

        if len(frame) < n_rows:
            missing = n_rows - len(frame)
            pad = pd.DataFrame({"title": [f"network_{i}" for i in range(len(frame), len(frame) + missing)], "description": [""] * missing})
            frame = pd.concat([frame, pad], ignore_index=True)
        elif len(frame) > n_rows:
            frame = frame.iloc[:n_rows].reset_index(drop=True)
        else:
            frame = frame.reset_index(drop=True)
        return frame

    def _extract_networks_canonical_text_latent(self, payload: Any) -> torch.Tensor:
        """Extract canonical network text latent tensor from loader payload."""
        latent = payload
        if isinstance(payload, tuple) and len(payload) > 0:
            latent = payload[0]
        elif isinstance(payload, dict):
            if "latent" in payload:
                latent = payload["latent"]
            elif "embeddings" in payload:
                latent = payload["embeddings"]

        if not isinstance(latent, (torch.Tensor, np.ndarray)):
            raise ValueError("Unsupported `networks_text` latent payload format.")
        return self._as_2d_tensor(latent)

    def _flatten_networks_neuro_payload(self, payload: Any) -> Tuple[torch.Tensor, pd.DataFrame]:
        """Flatten nested network neuro latent payload into matrix + metadata."""
        if not isinstance(payload, dict):
            raise ValueError("Expected `networks_neuro` latent payload to be a nested dict.")

        latents: List[torch.Tensor] = []
        rows: List[Dict[str, str]] = []

        for atlas_name, atlas_payload in payload.items():
            if not isinstance(atlas_payload, dict):
                continue
            for network_name, vec in atlas_payload.items():
                vec_tensor = self._as_2d_tensor(vec)
                if vec_tensor.shape[0] != 1:
                    raise ValueError("Each networks_neuro entry must be a single 384-d vector.")
                latents.append(vec_tensor.squeeze(0))
                rows.append(
                    {
                        "title": str(atlas_name),
                        "description": str(network_name),
                    }
                )

        if not latents:
            raise ValueError("No vectors found in `networks_neuro` payload.")

        matrix = torch.stack(latents, dim=0).to(dtype=torch.float32, device=self.device)
        metadata = pd.DataFrame(rows)
        return matrix, metadata

    @staticmethod
    def _normalize_record_id(value: Any) -> Any:
        """Normalize identifier values used for lookup joins."""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            if np.isnan(value):
                return None
            if float(value).is_integer():
                return int(value)
        text = str(value).strip()
        if not text:
            return None
        if text.isdigit():
            return int(text)
        return text

    def _load_dataset_compat(self, *names: str) -> Any:
        """Try multiple dataset loader keys for backward compatibility."""
        if not names:
            raise ValueError("At least one dataset key must be provided.")
        last_error: Optional[Exception] = None
        for name in names:
            try:
                return load_dataset(name)
            except ValueError as exc:
                last_error = exc
            except Exception:
                raise
        message = ", ".join(names)
        raise ValueError(f"Unable to load dataset using keys: {message}") from last_error

    def _load_latent_compat(self, *names: str) -> Any:
        """Try multiple latent loader keys for backward compatibility."""
        if not names:
            raise ValueError("At least one latent key must be provided.")
        last_error: Optional[Exception] = None
        for name in names:
            try:
                return load_latent(name)
            except ValueError as exc:
                last_error = exc
            except Exception:
                raise
        message = ", ".join(names)
        raise ValueError(f"Unable to load latent using keys: {message}") from last_error

    def _load_text_dataset_table(self, dataset: str) -> pd.DataFrame:
        """Load one text metadata table using canonical dataset name."""
        keys = TEXT_DATASET_LOAD_KEYS.get(dataset, (dataset,))
        payload = self._load_dataset_compat(*keys)
        if not isinstance(payload, pd.DataFrame):
            raise TypeError(f"Expected dataset '{dataset}' to resolve to pandas.DataFrame.")
        return payload

    def _load_text_latent_with_ids(self, dataset: str) -> Tuple[torch.Tensor, np.ndarray]:
        """Load one text latent matrix and aligned IDs."""
        keys = TEXT_LATENT_LOAD_KEYS.get(dataset, (dataset,))
        payload = self._load_latent_compat(*keys)
        default_id_key = DATASET_ID_COLUMNS.get(dataset, "id")
        latent, ids = self._unpack_latent_with_ids(payload, default_id_key=default_id_key)
        return self._as_2d_tensor(latent), np.asarray(ids)

    def _extract_latent_tensor(self, payload: Any) -> torch.Tensor:
        """Extract latent tensor from flexible payload shapes."""
        if isinstance(payload, torch.Tensor):
            return payload
        if isinstance(payload, np.ndarray):
            return torch.from_numpy(payload)
        if isinstance(payload, tuple) and len(payload) > 0:
            first = payload[0]
            if isinstance(first, (torch.Tensor, np.ndarray)):
                return self._as_2d_tensor(first)
        if isinstance(payload, dict):
            for key in ("latent", "embeddings", "tensor", "images", "X"):
                if key in payload and isinstance(payload[key], (torch.Tensor, np.ndarray)):
                    return self._as_2d_tensor(payload[key])
        raise ValueError("Unable to extract tensor from latent payload.")

    def _unpack_latent_with_ids(
        self,
        payload: Any,
        default_id_key: str,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """Extract (latent, ids) from tuple/dict payloads, with row-index fallback IDs."""
        latent: Optional[torch.Tensor] = None
        ids: Optional[np.ndarray] = None

        if isinstance(payload, tuple):
            if len(payload) >= 1:
                latent = self._as_2d_tensor(payload[0])
            if len(payload) >= 2:
                ids = np.asarray(payload[1])
        elif isinstance(payload, dict):
            for key in ("latent", "embeddings", "tensor"):
                value = payload.get(key)
                if isinstance(value, (torch.Tensor, np.ndarray)):
                    latent = self._as_2d_tensor(value)
                    break
            for key in (default_id_key, "pmid", "id", "term", "title", "name"):
                if key in payload:
                    ids = np.asarray(payload[key])
                    break
        elif isinstance(payload, (torch.Tensor, np.ndarray)):
            latent = self._as_2d_tensor(payload)
        else:
            raise ValueError("Unsupported latent payload format.")

        if latent is None:
            raise ValueError("Could not extract latent tensor from payload.")
        if ids is None:
            ids = np.arange(int(latent.shape[0]))
        return latent, ids

    def _ensure_pubmed_images_flat(self) -> None:
        """Lazy-load flattened PubMed images and PMID lookup."""
        if self._pubmed_images_flat is not None:
            return

        payload = self._load_dataset_compat("pubmed_images")
        images: Optional[torch.Tensor] = None
        pmids: Optional[np.ndarray] = None

        if isinstance(payload, tuple):
            if len(payload) >= 1 and isinstance(payload[0], (torch.Tensor, np.ndarray)):
                images = self._as_2d_tensor(payload[0]).detach().cpu()
            if len(payload) >= 2:
                pmids = np.asarray(payload[1])
        elif isinstance(payload, dict):
            for key in ("images", "image", "X", "data"):
                if key in payload and isinstance(payload[key], (torch.Tensor, np.ndarray)):
                    images = self._as_2d_tensor(payload[key]).detach().cpu()
                    break
            for key in ("pmid", "pmids", "id"):
                if key in payload:
                    pmids = np.asarray(payload[key])
                    break
        elif isinstance(payload, (torch.Tensor, np.ndarray)):
            images = self._as_2d_tensor(payload).detach().cpu()

        if images is None:
            raise ValueError("Unable to load flattened PubMed images.")

        self._pubmed_images_flat = images
        self._pubmed_image_pmids = pmids
        if pmids is not None:
            lookup: Dict[Any, int] = {}
            for i, pmid in enumerate(pmids):
                key = self._normalize_record_id(pmid)
                if key is not None and key not in lookup:
                    lookup[key] = i
            self._pubmed_pmid_to_image_index = lookup
        else:
            self._pubmed_pmid_to_image_index = None

    def _ensure_neurovault_images_flat(self) -> None:
        """Lazy-load flattened NeuroVault images."""
        if self._neurovault_images_flat is not None:
            return
        payload = self._load_dataset_compat("neurovault_images")
        if not isinstance(payload, (torch.Tensor, np.ndarray)):
            raise TypeError("Expected NeuroVault images to load as a tensor.")
        self._neurovault_images_flat = self._as_2d_tensor(payload).detach().cpu()

    def _resolve_pubmed_dataset_indices(self, pmids: np.ndarray) -> np.ndarray:
        """Map latent PubMed rows to dataset image indices."""
        self._ensure_pubmed_images_flat()
        if self._pubmed_pmid_to_image_index is None:
            return np.arange(len(pmids), dtype=np.int64)

        idxs: List[int] = []
        fallback_cap = int(self._pubmed_images_flat.shape[0]) if self._pubmed_images_flat is not None else len(pmids)
        if fallback_cap <= 0:
            raise ValueError("PubMed image dataset is empty.")
        for i, pmid in enumerate(pmids):
            key = self._normalize_record_id(pmid)
            mapped = self._pubmed_pmid_to_image_index.get(key) if key is not None else None
            if mapped is None:
                mapped = i if i < fallback_cap else fallback_cap - 1
            idxs.append(int(mapped))
        return np.asarray(idxs, dtype=np.int64)

    def _get_dataset_nifti(self, dataset: str, dataset_index: int) -> nib.Nifti1Image:
        """Load one dataset-native image and return it as NIfTI."""
        if dataset_index < 0:
            raise IndexError("dataset_index must be >= 0.")

        self._ensure_masker()
        key = str(dataset).strip().lower()

        if key in {"pubmed", "neuro", "publications", "papers"}:
            self._ensure_pubmed_images_flat()
            assert self._pubmed_images_flat is not None
            if dataset_index >= int(self._pubmed_images_flat.shape[0]):
                raise IndexError(f"dataset_index={dataset_index} out of range for pubmed images.")
            flat = self._pubmed_images_flat[dataset_index:dataset_index + 1].numpy()
            return self._masker.inverse_transform(flat)

        if key in {"neurovault", "neurovault_images"}:
            self._ensure_neurovault_images_flat()
            assert self._neurovault_images_flat is not None
            if dataset_index >= int(self._neurovault_images_flat.shape[0]):
                raise IndexError(f"dataset_index={dataset_index} out of range for neurovault images.")
            flat = self._neurovault_images_flat[dataset_index:dataset_index + 1].numpy()
            return self._masker.inverse_transform(flat)

        if key in {"network", "networks"}:
            self._ensure_network_brain_index()
            if self._network_brain_latent_cpu is None:
                raise ValueError("Network latent corpus is not available.")
            if dataset_index >= int(self._network_brain_latent_cpu.shape[0]):
                raise IndexError(f"dataset_index={dataset_index} out of range for network corpus.")
            self._ensure_autoencoder()
            with torch.no_grad():
                flat = torch.sigmoid(self._autoencoder.decoder(self._network_brain_latent_cpu[dataset_index:dataset_index + 1].to(self.device)))
            return self._masker.inverse_transform(flat.detach().cpu().numpy())

        raise ValueError(f"Unsupported dataset '{dataset}' for image plotting.")

    def _lookup_publication_row(self, pmid: Any, fallback_index: int) -> Optional[pd.Series]:
        """Resolve publication row by PMID, fallback to row index when needed."""
        self._ensure_publication_lookup()
        if self._pub_lookup is not None and pmid in self._pub_lookup.index:
            row = self._pub_lookup.loc[pmid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            return row
        if self._pub_df is not None and 0 <= fallback_index < len(self._pub_df):
            return self._pub_df.iloc[fallback_index]
        return None

    def _ensure_publication_lookup(self) -> None:
        """Lazy-load publication metadata lookup."""
        if self._pub_df is None:
            self._pub_df = self._load_dataset_compat("pubmed_text", "publications", "pubmed").copy()
        if self._pub_lookup is None and "pmid" in self._pub_df.columns:
            self._pub_lookup = self._pub_df.drop_duplicates("pmid").set_index("pmid", drop=False)

    def _extract_title_description(
        self,
        row: Optional[pd.Series],
        fallback_title: str = "",
    ) -> Tuple[str, str]:
        """Extract canonical title/description from a metadata row."""
        if row is None:
            return fallback_title, ""

        title = fallback_title
        for key in ("title", "name", "term"):
            if key in row and pd.notna(row[key]):
                candidate = self._clean_text(str(row[key]))
                if candidate:
                    title = candidate
                    break

        description = ""
        for key in ("description", "summary", "definition", "abstract"):
            if key in row and pd.notna(row[key]):
                candidate = self._clean_text(str(row[key]))
                if candidate:
                    description = candidate
                    break

        return title, description

    @staticmethod
    def _clean_text(text: str) -> str:
        """Collapse whitespace and strip."""
        return " ".join(text.split()).strip()

    def _ensure_projection_heads(self) -> None:
        """Lazy-load projection heads."""
        if self._proj_head_image is None:
            self._proj_head_image = load_model("proj_head_image_infonce").to(self.device).eval()
        if self._proj_head_text_infonce is None:
            self._proj_head_text_infonce = load_model("proj_head_text_infonce").to(self.device).eval()
        if self._proj_head_text_mse is None:
            self._proj_head_text_mse = load_model("proj_head_text_mse").to(self.device).eval()

    def _ensure_specter(self) -> None:
        """Lazy-load SPECTER."""
        if self._specter is None:
            self._specter = load_model("specter")
            self._specter.to(self.device)

    def _ensure_autoencoder(self) -> None:
        """Lazy-load autoencoder."""
        if self._autoencoder is None:
            self._autoencoder = load_model("autoencoder").to(self.device).eval()

    def _ensure_masker(self) -> None:
        """Lazy-load NIfTI masker."""
        if self._masker is None:
            self._masker = load_masker()

    def _encode_text(self, X: Any) -> torch.Tensor:
        """Encode raw text payload(s) into 768-d embeddings."""
        self._ensure_specter()
        with torch.no_grad():
            emb = self._specter(X)
        emb = self._as_2d_tensor(emb).to(self.device)
        if emb.shape[1] != TEXT_EMBED_DIM:
            raise ValueError(f"SPECTER output dim mismatch. Expected {TEXT_EMBED_DIM}, got {emb.shape[1]}.")
        return emb

    def _encode_brain_image(self, image: nib.Nifti1Image) -> torch.Tensor:
        """Resample, flatten, and encode NIfTI image(s) into latent space."""
        self._ensure_masker()
        self._ensure_autoencoder()
        mask_img = self._masker.mask_img_

        arr = np.asarray(image.get_fdata())
        unique_values = np.unique(np.round(arr, 6))
        is_binary = unique_values.size <= 2 and set(unique_values.tolist()) <= {0.0, 1.0}
        interpolation = "nearest" if is_binary else "continuous"

        resampled = resample_to_img(image, mask_img, interpolation=interpolation)
        flattened = self._masker.transform(resampled).astype(np.float32)
        if is_binary:
            flattened = (flattened > 0).astype(np.float32)
        else:
            flattened[flattened < 0] = 0.0
            threshold = np.percentile(flattened, 95)
            if threshold > 0:
                flattened = (flattened >= threshold).astype(np.float32)

        return self._encode_brain_flat(torch.from_numpy(flattened))

    def _encode_brain_flat(self, flat: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Encode flattened brain vectors into latent 384-d vectors."""
        self._ensure_autoencoder()
        flat_tensor = self._as_2d_tensor(flat).to(self.device)
        if flat_tensor.shape[1] != BRAIN_FLAT_DIM:
            raise ValueError(f"Expected flattened brain dim {BRAIN_FLAT_DIM}, got {flat_tensor.shape[1]}.")
        with torch.no_grad():
            latent = self._autoencoder.encoder(flat_tensor)
        return latent

    def _coerce_numeric_query(self, X: Any) -> Optional[torch.Tensor]:
        """Coerce numeric inputs into 2D tensors."""
        tensor: Optional[torch.Tensor] = None
        if isinstance(X, torch.Tensor):
            tensor = X.detach()
        elif isinstance(X, np.ndarray):
            tensor = torch.from_numpy(X)
        elif isinstance(X, (list, tuple)) and len(X) > 0:
            first = X[0]
            if isinstance(first, (int, float, np.number, list, tuple, np.ndarray)):
                tensor = torch.as_tensor(X)
        if tensor is None:
            return None
        return self._as_2d_tensor(tensor)

    @staticmethod
    def _as_2d_tensor(X: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Ensure tensor shape is (N, D)."""
        tensor = X if isinstance(X, torch.Tensor) else torch.as_tensor(X)
        tensor = tensor.detach().to(dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 2:
            raise ValueError(f"Expected 1D/2D input, got shape {tuple(tensor.shape)}.")
        return tensor

    @staticmethod
    def _is_text_payload(X: Any) -> bool:
        """Return True for raw text payload types."""
        if isinstance(X, (str, dict, pd.DataFrame)):
            return True
        if isinstance(X, (list, tuple)) and len(X) > 0:
            return isinstance(X[0], (str, dict))
        return False

    def _build_lookup(self, dataset: str, table: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Build id-based metadata lookup for one dataset."""
        id_col = DATASET_ID_COLUMNS.get(dataset)
        if id_col is None or id_col not in table.columns:
            return None
        temp = table.copy()
        if id_col == "term":
            temp[id_col] = temp[id_col].astype(str).str.lower()
        return temp.drop_duplicates(id_col).set_index(id_col, drop=False)

    def _normalize_lookup_key(self, dataset: str, value: Any) -> Any:
        """Normalize lookup key by dataset id type."""
        if DATASET_ID_COLUMNS.get(dataset) == "term":
            return str(value).lower()
        return value

    def _canonicalize_datasets(self, datasets: Sequence[str]) -> List[str]:
        """Normalize dataset aliases and validate names."""
        canonical: List[str] = []
        for name in datasets:
            key = DATASET_ALIASES.get(str(name).lower(), str(name).lower())
            if key not in DATASET_ID_COLUMNS:
                valid = sorted(DATASET_ID_COLUMNS.keys()) + sorted(DATASET_ALIASES.keys())
                raise ValueError(f"Unknown dataset '{name}'. Valid values include: {valid}")
            if key not in canonical:
                canonical.append(key)
        return canonical

    @staticmethod
    def _canonicalize_brain_dataset(dataset: str) -> str:
        """Normalize/validate brain retrieval dataset selection."""
        key = str(dataset).strip().lower()
        aliases = {
            "pubmed": "neuro",
            "pubmed_images": "neuro",
            "publications": "neuro",
            "papers": "neuro",
            "neuro": "neuro",
            "network": "networks",
            "networks": "networks",
            "neurovault": "neurovault",
            "neurovault_images": "neurovault",
            "nv": "neurovault",
        }
        key = aliases.get(key, key)
        if key not in {"neuro", "networks", "neurovault"}:
            raise ValueError("dataset must be one of: 'neuro', 'pubmed', 'networks', 'neurovault'.")
        return key

    def _resolve_brain_datasets(
        self,
        dataset: Optional[Union[str, Sequence[str]]],
    ) -> List[str]:
        """Resolve one or many brain corpora for InfoNCE retrieval."""
        if dataset is None:
            requested: List[str] = ["neuro", "networks", "neurovault"]
        elif isinstance(dataset, str):
            requested = [self._canonicalize_brain_dataset(dataset)]
        else:
            requested = [self._canonicalize_brain_dataset(name) for name in dataset]

        out: List[str] = []
        for name in requested:
            if name not in out:
                out.append(name)
        return out

    def _resolve_active_datasets(self, datasets: Optional[Sequence[str]]) -> List[str]:
        """Resolve datasets for one text retrieval call."""
        if datasets is None:
            return list(self.datasets)
        return self._canonicalize_datasets(datasets)

    @staticmethod
    def _validate_head(head: str) -> None:
        """Validate retrieval head mode."""
        if head not in {"mse", "infonce"}:
            raise ValueError("head must be either 'mse' or 'infonce'.")
