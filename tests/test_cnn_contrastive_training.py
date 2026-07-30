from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from neurovlm.ale_cnn import ALE3DCNNAutoEncoder
from neurovlm.atlas_free_text import (
    AtlasFreeContrastiveCollator,
    AtlasFreeTextEmbeddingLookup,
    primary_positive_text,
    primary_positive_text_id,
)
from neurovlm.cnn import CNNContrastiveModel
from neurovlm.evaluation import evaluate_contrastive
from neurovlm.models import ProjHead
from neurovlm.training import (
    ContrastiveTrainConfig,
    build_contrastive,
    contrastive_from_checkpoint,
    train_contrastive,
)
from neurovlm.training.contrastive import _make_loader


class _Pairs(Dataset):
    def __init__(self, split: str, n: int = 3):
        self.rows = []
        for index in range(n):
            self.rows.append(
                {
                    "volume": torch.rand(1, 4, 4, 4, generator=torch.Generator().manual_seed(index)),
                    "map_id": f"{split}-{index}",
                    "positive_texts": [{"text_id": f"text-{index}", "text": f"text {index}"}],
                    "metadata": {"source": "pubmed"},
                }
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class _Provider:
    def __init__(self):
        self.train = _Pairs("train")
        self.val = _Pairs("val")
        self.test = _Pairs("test")


def _lookup() -> AtlasFreeTextEmbeddingLookup:
    vectors = torch.zeros(3, 768)
    vectors[0, 0] = 1
    vectors[1, 1] = 1
    vectors[2, 2] = 1
    return AtlasFreeTextEmbeddingLookup(vectors, ["text-0", "text-1", "text-2"])


def _config(tmp_path: Path, **overrides) -> ContrastiveTrainConfig:
    values = {
        "domain": "pubmed",
        "output_root": tmp_path,
        "run_id": "tiny-contrastive",
        "epochs": 1,
        "batch_size": 2,
        "eval_batch_size": 2,
        "device": "cpu",
        "amp": False,
        "early_stopping_patience": None,
        "preset": "custom",
        "target_shape": (4, 4, 4),
        "base_channels": 2,
        "num_blocks": 1,
    }
    values.update(overrides)
    return ContrastiveTrainConfig(**values)


def _model() -> CNNContrastiveModel:
    ae = ALE3DCNNAutoEncoder(
        output_shape=(4, 4, 4), base_channels=2, num_blocks=1, latent_dim=384
    )
    return CNNContrastiveModel(ae.encoder, ProjHead())


def test_lookup_is_id_strict_linear_safe_and_numpy_payload_compatible() -> None:
    lookup = _lookup()
    item = _Pairs("train")[0]
    assert primary_positive_text_id(item) == "text-0"
    assert primary_positive_text(item) == "text 0"
    assert torch.equal(lookup["text-0"], torch.nn.functional.one_hot(torch.tensor(0), 768).float())
    payload_lookup = AtlasFreeTextEmbeddingLookup.from_payload(
        {"embeddings": lookup.embeddings, "text_ids": np.asarray(lookup.text_ids)}
    )
    assert len(payload_lookup) == 3
    with pytest.raises(ValueError, match="Duplicate text IDs"):
        AtlasFreeTextEmbeddingLookup(torch.eye(2, 768), ["same", "same"])
    with pytest.raises(KeyError, match="missing"):
        lookup["absent"]


def test_collator_uses_first_positive_and_train_loader_drops_singleton() -> None:
    dataset = _Pairs("train")
    dataset.rows[0]["positive_texts"].append({"text_id": "text-2"})
    batch = AtlasFreeContrastiveCollator(_lookup(), (4, 4, 4))([dataset[0], dataset[1]])
    assert batch["text_id"] == ["text-0", "text-1"]
    loader = _make_loader(
        dataset, _lookup(), _config(Path("unused")), batch_size=2, shuffle=True, seed=1
    )
    assert len(loader) == 1
    assert all(len(item["volume"]) == 2 for item in loader)


def test_full_split_evaluation_emits_bidirectional_metrics_and_curves() -> None:
    result = evaluate_contrastive(
        _model(), _Pairs("val"), lookup=_lookup(), target_shape=(4, 4, 4), batch_size=2
    )
    assert result.n == 3
    assert len(result.recall_curves) == 3
    for name in (
        "t2i_recall@1",
        "i2t_recall@5",
        "mean_recall@10",
        "t2i_mrr",
        "i2t_mean_rank",
        "mean_normalized_k_recall_curve_auc",
    ):
        assert name in result.summary


def test_tiny_offline_train_artifacts_reload_and_internal_variant(tmp_path: Path) -> None:
    result = train_contrastive(
        _config(tmp_path), provider=_Provider(), lookup=_lookup(), model=_model()
    )
    assert result.best_checkpoint.is_file()
    assert (result.run_dir / "checkpoints" / "best_val_normalized_recall_auc.pt").is_file()
    assert (result.run_dir / "metrics" / "history.csv").is_file()
    assert (result.run_dir / "metrics" / "recall_curves.csv").is_file()
    assert (result.run_dir / "metrics" / "test_summary.csv").is_file()
    effective = json.loads((result.run_dir / "config" / "effective.json").read_text())
    assert effective["values"]["internal_variant"] == "mixed_to_pubmed"
    assert effective["values"]["text_preprocessing"] == "empty_string_centered_l2_unit_normalized"

    reloaded = contrastive_from_checkpoint(result.best_checkpoint)
    volume = torch.rand(1, 1, 4, 4, 4)
    text = _lookup().embeddings[:1]
    with torch.no_grad():
        expected = result.model(volume, text)
        actual = reloaded(volume, text)
    assert all(torch.allclose(left, right) for left, right in zip(expected, actual))


def test_default_and_finetuned_initialize_exact_autoencoder(monkeypatch, tmp_path: Path) -> None:
    import neurovlm.models as models

    calls = []

    def fake_load_model(**kwargs):
        calls.append(kwargs)
        return ALE3DCNNAutoEncoder(
            output_shape=(36, 45, 38), base_channels=64, num_blocks=4, latent_dim=384
        )

    monkeypatch.setattr(models, "load_model", fake_load_model)
    projection = ProjHead()
    build_contrastive(ContrastiveTrainConfig(domain="nilearn"), text_projection=projection)
    build_contrastive(
        ContrastiveTrainConfig(domain="neurovault", variant="finetuned"),
        text_projection=projection,
    )
    assert calls == [
        {"family": "cnn", "task": "autoencoder", "variant": "mixed_baseline"},
        {
            "family": "cnn",
            "task": "autoencoder",
            "variant": "finetuned",
            "domain": "neurovault",
        },
    ]


@pytest.mark.parametrize(
    ("domain", "variant", "internal"),
    [
        ("pubmed", "mixed_baseline", "mixed_to_pubmed"),
        ("nilearn", "mixed_baseline", "mixed_to_nilearn"),
        ("neurovault", "mixed_baseline", "mixed_to_neurovault"),
        ("pubmed", "finetuned", "pubmed"),
        ("nilearn", "finetuned", "nilearn"),
        ("neurovault", "finetuned", "neurovault"),
    ],
)
def test_all_six_branch_names_are_exact(domain: str, variant: str, internal: str) -> None:
    assert ContrastiveTrainConfig(domain=domain, variant=variant).internal_variant == internal


def test_resume_restores_composite_optimizer_and_history(monkeypatch, tmp_path: Path) -> None:
    import neurovlm.training.contrastive as module

    original = module._train_epoch
    calls = 0

    def interrupt_second_epoch(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return original(*args, **kwargs)

    config = _config(tmp_path, epochs=2)
    monkeypatch.setattr(module, "_train_epoch", interrupt_second_epoch)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train_contrastive(config, provider=_Provider(), lookup=_lookup(), model=_model())
    history_path = tmp_path / "tiny-contrastive" / "metrics" / "history.csv"
    before = list(csv.DictReader(history_path.open()))
    assert {row["epoch"] for row in before if row["epoch"]} == {"1"}

    monkeypatch.setattr(module, "_train_epoch", original)
    resumed = train_contrastive(
        _config(tmp_path, epochs=2, resume="last.pt"),
        provider=_Provider(),
        lookup=_lookup(),
        model=_model(),
    )
    assert resumed.epochs_completed == 2
    after = list(csv.DictReader(history_path.open()))
    assert {row["epoch"] for row in after if row["epoch"]} == {"1", "2"}
    assert len(after) > len(before)
