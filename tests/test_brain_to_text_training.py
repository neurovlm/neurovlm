from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from neurovlm.evaluation.brain_to_text import (
    brain_to_text_lm_forward,
    parse_brain_to_text_batch,
)
from neurovlm.models import ProjHead
from neurovlm.qformer import NeuroQFormer
from neurovlm.training.brain_to_text import (
    BrainToTextCollator,
    BrainToTextGenerationTrainConfig,
    brain_to_text_generation_from_checkpoint,
    build_brain_to_text_generation,
    train_brain_to_text_generation,
)
from neurovlm.training.mlp import (
    MLPBrainToTextRetrievalTrainConfig,
    build_mlp_brain_to_text_retrieval,
    mlp_brain_to_text_retrieval_from_checkpoint,
    train_mlp_brain_to_text_retrieval,
)


class _FakeCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 11, hidden_size: int = 6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, *, inputs_embeds, attention_mask, labels):
        del attention_mask
        logits = self.head(inputs_embeds)
        loss = F.cross_entropy(
            logits[:, :-1].transpose(1, 2), labels[:, 1:], ignore_index=-100
        )
        return SimpleNamespace(loss=loss, logits=logits)


def _qformer() -> NeuroQFormer:
    return NeuroQFormer(
        image_dim=4,
        semantic_dim=4,
        lm_dim=6,
        num_queries=2,
        hidden_dim=8,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
        use_canonical_projection=False,
    )


class _GenerationRows(Dataset):
    def __init__(self, offset: int = 0):
        generator = torch.Generator().manual_seed(100 + offset)
        self.brain = torch.randn(4, 4, generator=generator)
        self.ids = torch.tensor(
            [[1, 2, 3, 0], [2, 3, 4, 0], [3, 4, 5, 0], [4, 5, 6, 0]]
        )

    def __len__(self):
        return len(self.brain)

    def __getitem__(self, index):
        return {
            "brain_embedding": self.brain[index],
            "input_ids": self.ids[index],
            "attention_mask": self.ids[index].ne(0).long(),
            "source": "pubmed" if index % 2 else "nilearn",
            "sample_id": f"generation-{index}",
            "reference_text": f"reference {index}",
        }


class _RetrievalRows(Dataset):
    def __init__(self, offset: int = 0):
        generator = torch.Generator().manual_seed(200 + offset)
        self.brain = torch.randn(6, 4, generator=generator)
        matrix = torch.randn(4, 6, generator=generator)
        self.text = self.brain @ matrix

    def __len__(self):
        return len(self.brain)

    def __getitem__(self, index):
        return {
            "brain_embedding": self.brain[index],
            "text_embedding": self.text[index],
            "source": "pubmed",
            "sample_id": f"retrieval-{index}",
        }


def _generation_config(tmp_path: Path, **overrides):
    values = dict(
        output_root=tmp_path,
        run_id="b2t-generation",
        device="cpu",
        epochs=1,
        batch_size=2,
        eval_batch_size=2,
        initialization="scratch",
        preset="custom",
        image_dim=4,
        semantic_dim=4,
        lm_dim=6,
        num_queries=2,
        hidden_dim=8,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
        use_canonical_projection=False,
        pad_token_id=0,
    )
    values.update(overrides)
    return BrainToTextGenerationTrainConfig(**values)


def _retrieval_config(tmp_path: Path, **overrides):
    values = dict(
        output_root=tmp_path,
        run_id="b2t-retrieval",
        device="cpu",
        epochs=2,
        batch_size=3,
        eval_batch_size=3,
        preset="custom",
        text_dim=6,
        text_hidden_dim=5,
        brain_dim=4,
        brain_hidden_dim=4,
        shared_dim=4,
        initialize_text_from_mse=False,
    )
    values.update(overrides)
    return MLPBrainToTextRetrievalTrainConfig(**values)


def test_visual_and_padding_labels_are_ignored_and_only_qformer_gets_gradients(tmp_path: Path):
    config = _generation_config(tmp_path)
    qformer, lm = build_brain_to_text_generation(
        config, qformer=_qformer(), causal_lm=_FakeCausalLM()
    )
    batch = parse_brain_to_text_batch(
        {
            "brain_embedding": torch.randn(2, 4),
            "input_ids": torch.tensor([[1, 2, 0], [2, 3, 4]]),
            "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
        }
    )
    output = brain_to_text_lm_forward(qformer, lm, batch)
    assert torch.all(output.labels[:, :2] == -100)
    assert output.labels[0, -1].item() == -100
    assert output.labels[1, -1].item() == 4
    output.loss.backward()
    assert any(parameter.grad is not None for parameter in qformer.qformer.parameters())
    assert all(parameter.grad is None for parameter in lm.parameters())
    assert all(not parameter.requires_grad for parameter in lm.parameters())

    collated = BrainToTextCollator(0)(
        [
            {"brain_embedding": torch.ones(4), "input_ids": torch.tensor([1, 2])},
            {"brain_embedding": torch.zeros(4), "input_ids": torch.tensor([3])},
        ]
    )
    assert collated["input_ids"].tolist() == [[1, 2], [3, 0]]
    assert collated["attention_mask"].tolist() == [[1, 1], [1, 0]]


def test_generation_artifacts_callbacks_and_standalone_reload(tmp_path: Path):
    provider = {
        "train": _GenerationRows(0),
        "val": _GenerationRows(1),
        "test": _GenerationRows(2),
    }

    def generate(qformer, lm, batch):
        del qformer, lm
        return [f"generated {value}" for value in batch.sample_ids]

    def semantic(predictions, references, rows):
        assert len(predictions) == len(references) == len(rows) == 2
        return {"semantic_score": 0.75}

    config = _generation_config(tmp_path, generated_samples_limit=2)
    result = train_brain_to_text_generation(
        config,
        provider=provider,
        qformer=_qformer(),
        causal_lm=_FakeCausalLM(),
        generation_callback=generate,
        semantic_metric_callback=semantic,
    )
    assert result.best_checkpoint.is_file() and result.last_checkpoint.is_file()
    assert result.test_metrics["semantic_score"] == pytest.approx(0.75)
    assert (result.run_dir / "metrics/generated_text.csv").is_file()
    assert json.loads((result.run_dir / "status.json").read_text())["state"] == "completed"
    payload = torch.load(result.best_checkpoint, weights_only=True)
    assert not any("causal_lm" in key or "embedding" in key for key in payload["model_state_dict"])
    reloaded = brain_to_text_generation_from_checkpoint(result.best_checkpoint)
    sample = provider["test"].brain[:2]
    with torch.no_grad():
        assert torch.allclose(result.qformer(sample), reloaded(sample))
    legacy = NeuroQFormer.from_state_dict_payload(
        {
            "state_dict": result.qformer.state_dict(),
            "config": result.qformer.architecture_config(),
        }
    )
    with torch.no_grad():
        assert torch.allclose(result.qformer(sample), legacy(sample))

    malformed = dict(payload)
    malformed["architecture"] = dict(payload["architecture"])
    malformed["architecture"]["hidden_dim"] = 10
    with pytest.raises(ValueError, match="architecture"):
        brain_to_text_generation_from_checkpoint(malformed)


def test_generation_resume_preserves_history(tmp_path: Path, monkeypatch):
    import neurovlm.training.brain_to_text as module

    provider = {
        "train": _GenerationRows(0),
        "val": _GenerationRows(1),
        "test": _GenerationRows(2),
    }
    original = module._train_epoch
    calls = 0

    def interrupt_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted")
        return original(*args, **kwargs)

    config = _generation_config(tmp_path, epochs=2)
    monkeypatch.setattr(module, "_train_epoch", interrupt_second)
    with pytest.raises(RuntimeError, match="interrupted"):
        train_brain_to_text_generation(
            config, provider=provider, qformer=_qformer(), causal_lm=_FakeCausalLM()
        )
    monkeypatch.setattr(module, "_train_epoch", original)
    result = train_brain_to_text_generation(
        _generation_config(tmp_path, epochs=2, resume="last.pt"),
        provider=provider,
        qformer=_qformer(),
        causal_lm=_FakeCausalLM(),
    )
    rows = list(csv.DictReader((result.run_dir / "metrics/history.csv").open()))
    assert {row["epoch"] for row in rows if row["epoch"]} == {"1", "2"}
    assert json.loads((result.run_dir / "status.json").read_text())["resume_count"] == 1


def test_retrieval_reuses_contrastive_model_and_selects_i2t_auc(tmp_path: Path):
    provider = {
        "train": _RetrievalRows(0),
        "val": _RetrievalRows(1),
        "test": _RetrievalRows(2),
    }
    config = _retrieval_config(tmp_path)
    model = build_mlp_brain_to_text_retrieval(
        config,
        brain_projection=ProjHead(4, 4, 4, seed=1),
        text_projection=ProjHead(6, 5, 4, seed=2),
    )
    result = train_mlp_brain_to_text_retrieval(config, provider=provider, model=model)
    assert "i2t_normalized_k_recall_curve_auc" in result.test_metrics
    assert "t2i_normalized_k_recall_curve_auc" in result.test_metrics
    rows = list(csv.DictReader((result.run_dir / "metrics/history.csv").open()))
    values = [
        float(row["value"])
        for row in rows
        if row["split"] == "val"
        and row["epoch"]
        and row["metric"] == "i2t_normalized_k_recall_curve_auc"
    ]
    assert result.best_metric == pytest.approx(max(values))
    manifest = json.loads(
        (result.run_dir / "checkpoints/checkpoint_manifest.json").read_text()
    )
    assert manifest["primary_metric"] == "val_i2t_normalized_k_recall_curve_auc"
    assert manifest["metric_direction"] == "max"
    payload = torch.load(result.best_checkpoint, weights_only=True)
    assert payload["model_spec"]["task"] == "brain_to_text_retrieval"
    reloaded = mlp_brain_to_text_retrieval_from_checkpoint(result.best_checkpoint)
    with torch.no_grad():
        expected = result.model(provider["test"].brain, provider["test"].text)
        actual = reloaded(provider["test"].brain, provider["test"].text)
    assert all(torch.allclose(left, right) for left, right in zip(expected, actual))
