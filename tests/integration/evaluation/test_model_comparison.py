from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

import neurovlm.evaluation.comparison as comparison
from neurovlm.data.atlas_free_text import AtlasFreeTextEmbeddingLookup
from neurovlm.evaluation import (
    ComparisonSelection,
    default_comparison_matrix,
    evaluate_contrastive_comparison,
    evaluate_reconstruction_comparison,
    evaluate_text_to_brain_comparison,
    resolve_comparison_manifest,
)
from neurovlm.core.runtime import RuntimeMetadata


class _Maps(Dataset):
    def __init__(self):
        self.rows = []
        for index, source in enumerate(("pubmed", "nilearn", "neurovault")):
            self.rows.append({
                "volume": torch.rand(1, 36, 45, 38, generator=torch.Generator().manual_seed(index)),
                "map_id": f"map-{index}",
                "positive_texts": [{
                    "text_id": f"text-{index}",
                    "text": f"raw positive text {index}",
                }],
                "metadata": {
                    "source": source,
                    # These historical fields must be ignored by comparison.
                    "tensor_path": "experiments/3dcnn/never-read.pt",
                },
            })

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def _lookup():
    embeddings = torch.zeros(3, 768)
    embeddings[0, 0] = 1
    embeddings[1, 1] = 1
    embeddings[2, 2] = 1
    return AtlasFreeTextEmbeddingLookup(embeddings, ["text-0", "text-1", "text-2"])


class _Runtime:
    def __init__(self, selection: ComparisonSelection):
        family = selection.family
        variant = selection.variant or ("mixed_baseline" if family == "cnn" else "default")
        self.metadata = RuntimeMetadata(
            canonical_name=f"{family}:{selection.task}:{variant}", family=family,
            task=selection.task, domain=selection.domain, variant=variant,
            loader_variant=(f"mixed_to_{selection.domain}" if selection.domain and variant == "mixed_baseline" else selection.domain),
            source="released", checkpoint=None, device="cpu",
            brain_space="test", text_space="test",
        )

    def reconstruct(self, value):
        return torch.as_tensor(value) * 0.9

    def encode_brain(self, value):
        value = torch.as_tensor(value).reshape(len(value), -1)
        features = torch.stack((value.mean(1), value.std(1), value[:, 0], value[:, 1]), 1)
        return F.normalize(features, dim=1)

    def encode_text(self, value):
        return F.normalize(torch.as_tensor(value)[:, :4] + 0.01, dim=1)

    def generate(self, value):
        return torch.zeros(len(value), 1, 36, 45, 38)


@pytest.fixture
def _offline(monkeypatch):
    def load(**kwargs):
        selection = ComparisonSelection(
            kwargs["family"], kwargs["task"], kwargs.get("domain"), kwargs.get("variant")
        )
        return _Runtime(selection)

    monkeypatch.setattr(comparison, "load_pipeline", load)
    monkeypatch.setattr(
        comparison,
        "atlas_free_volume_to_mlp_flat",
        lambda value, binarize=True: torch.as_tensor(value).reshape(len(value), -1),
    )


def test_default_matrix_is_mlp_plus_all_mixed_domains_and_finetuned_is_opt_in():
    default = default_comparison_matrix("contrastive")
    assert len(default) == 6
    assert [(item.family, item.domain, item.variant, item.evaluation_domain) for item in default] == [
        ("mlp", None, None, "pubmed"),
        ("cnn", "pubmed", "mixed_baseline", "pubmed"),
        ("mlp", None, None, "nilearn"),
        ("cnn", "nilearn", "mixed_baseline", "nilearn"),
        ("mlp", None, None, "neurovault"),
        ("cnn", "neurovault", "mixed_baseline", "neurovault"),
    ]
    expanded = default_comparison_matrix("text_to_brain", include_finetuned=True)
    assert sum(item.variant == "finetuned" for item in expanded) == 3
    ae = default_comparison_matrix("autoencoder")
    assert len(ae) == 6 and all(item.evaluation_domain for item in ae)
    assert all(item.domain is None for item in ae if item.family == "cnn")
    with pytest.raises(ValueError, match="not supported"):
        default_comparison_matrix("brain_to_text_generation")


def test_manifest_is_failure_tolerant_and_exposes_resolution_metadata():
    selections = (
        ComparisonSelection("mlp", "autoencoder"),
        ComparisonSelection("cnn", "autoencoder", variant="mixed_baseline"),
    )

    def loader(**kwargs):
        if kwargs["family"] == "cnn":
            raise FileNotFoundError("released checkpoint unavailable")
        return _Runtime(selections[0])

    manifest = resolve_comparison_manifest(selections, loader=loader)
    assert manifest[0]["status"] == "resolved"
    assert manifest[0]["canonical_name"] == "mlp:autoencoder:default"
    assert manifest[1]["status"] == "missing_checkpoint"
    assert "unavailable" in manifest[1]["error"]


def test_reconstruction_returns_standard_summary_source_and_sample_rows(_offline):
    result = evaluate_reconstruction_comparison(
        selections=(ComparisonSelection("mlp", "autoencoder"),
                    ComparisonSelection("cnn", "autoencoder", variant="mixed_baseline")),
        data=_Maps(), batch_size=2,
    )
    assert len(result.summary) == 2
    assert len(result.by_sample) == 6
    assert len(result.by_source) == 6
    assert {row["comparison_space"] for row in result.summary} == {
        "mlp_masker_flatmap", "native_atlas_free_volume"
    }
    assert {row["comparison_protocol"] for row in result.summary} == {"paired_atlas_free"}
    assert all("reconstruction_mse" in row and "top5_dice" in row for row in result.summary)
    assert all(row["status"] == "resolved" for row in result.manifest)


def test_default_rows_are_paired_within_each_evaluation_domain(_offline):
    result = evaluate_reconstruction_comparison(
        selections=default_comparison_matrix("autoencoder", domains=("pubmed", "nilearn")),
        data=_Maps(), batch_size=2,
    )
    assert [(row["evaluation_domain"], row["family"], row["n"]) for row in result.summary] == [
        ("pubmed", "mlp", 1),
        ("pubmed", "cnn", 1),
        ("nilearn", "mlp", 1),
        ("nilearn", "cnn", 1),
    ]
    assert len({row["model_id"] for row in result.summary}) == 4


def test_contrastive_reuses_bidirectional_metrics_and_curves(_offline):
    result = evaluate_contrastive_comparison(
        selections=(ComparisonSelection("cnn", "contrastive", domain="pubmed",
                                        variant="mixed_baseline"),),
        data=_Maps(), lookup=_lookup(), batch_size=2,
    )
    assert result.summary[0]["n"] == 3
    assert "t2i_normalized_k_recall_curve_auc" in result.summary[0]
    assert "i2t_mrr" in result.summary[0]
    assert len(result.recall_curves) == 3
    assert len(result.by_sample) == 3


def test_mlp_comparison_reencodes_raw_text_with_family_native_preprocessing(_offline):
    encoded_texts = []
    encoded_batch_sizes = []

    def encode(texts):
        encoded_texts.extend(texts)
        encoded_batch_sizes.append(len(texts))
        embeddings = torch.zeros(len(texts), 768)
        embeddings[:, 0] = 1
        return embeddings

    result = evaluate_contrastive_comparison(
        selections=(ComparisonSelection("mlp", "contrastive"),),
        data=_Maps(),
        lookup=_lookup(),
        mlp_text_encoder=encode,
        batch_size=2,
    )

    assert encoded_texts == [
        "raw positive text 0",
        "raw positive text 1",
        "raw positive text 2",
    ]
    assert encoded_batch_sizes == [2, 1]
    assert result.summary[0]["comparison_protocol"] == "paired_atlas_free"
    assert (
        result.summary[0]["text_preprocessing"]
        == "specter2_adhoc_query_orthogonalized_then_l2"
    )


def test_text_to_brain_returns_generation_metrics_without_reading_local_paths(_offline):
    result = evaluate_text_to_brain_comparison(
        selections=(ComparisonSelection("cnn", "text_to_brain", domain="neurovault",
                                        variant="finetuned"),),
        data=_Maps(), lookup=_lookup(), batch_size=2,
    )
    assert result.summary[0]["variant"] == "finetuned"
    assert result.summary[0]["comparison_space"] == "native_atlas_free_volume"
    assert result.summary[0]["n"] == 3
    assert "top5_dice" in result.summary[0]
    assert {row["source"] for row in result.by_source} == {"pubmed", "nilearn", "neurovault"}


def test_task_mismatch_is_rejected_before_loading(_offline):
    with pytest.raises(ValueError, match="reconstruction selections"):
        evaluate_reconstruction_comparison(
            selections=(ComparisonSelection("mlp", "contrastive"),), data=_Maps()
        )
