"""SPECTER wrapper and the packaged-model loaders without downloads.

``transformers.AutoTokenizer``, ``transformers.AutoModel`` and ``adapters.AutoAdapterModel`` are replaced by tiny
stand-ins so ``Specter`` can be constructed, pooled and queried offline; the resource loaders behind
``load_model`` are replaced to check the dispatch of every MLP alias.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch import nn

from neurovlm.models.base import ConceptClf, NeuroAutoEncoder, ProjHead, Specter, load_model

HIDDEN = 6


class FakeTokenizer:
    sep_token = "[SEP]"

    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, text, **kwargs):
        texts = [text] if isinstance(text, str) else list(text)
        self.calls.append(texts)
        n = len(texts)
        return {"input_ids": torch.ones(n, 4, dtype=torch.long), "attention_mask": torch.ones(n, 4, dtype=torch.long)}


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.loaded: list[tuple[str, dict]] = []

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        n, t = input_ids.shape
        hidden = torch.arange(n * t * HIDDEN, dtype=torch.float32).reshape(n, t, HIDDEN) / 10 + 0.1
        return SimpleNamespace(last_hidden_state=hidden)

    def load_adapter(self, adapter_id, **kwargs):
        self.loaded.append((adapter_id, kwargs))


@pytest.fixture
def fake_hf(monkeypatch):
    import adapters
    import transformers

    state = {"tokenizer": FakeTokenizer(), "model": FakeEncoder(), "adapter_model": FakeEncoder(),
             "local_fails": False, "calls": []}

    def tokenizer(name, **kwargs):
        state["calls"].append(("tokenizer", name, kwargs))
        if state["local_fails"] and kwargs.get("local_files_only"):
            raise OSError("not cached")
        return state["tokenizer"]

    def model(name, **kwargs):
        state["calls"].append(("model", name, kwargs))
        return state["model"]

    def adapter_model(name, **kwargs):
        state["calls"].append(("adapter_model", name, kwargs))
        if state["local_fails"] and kwargs.get("local_files_only"):
            raise OSError("not cached")
        return state["adapter_model"]

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer)
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", model)
    monkeypatch.setattr(adapters.AutoAdapterModel, "from_pretrained", adapter_model)
    return state


def test_specter_default_adapter_prefers_the_local_cache(fake_hf):
    specter = Specter()
    assert specter.specter is fake_hf["adapter_model"]
    assert fake_hf["calls"][0] == ("tokenizer", "allenai/specter2_aug2023refresh_base", {"local_files_only": True})
    adapter_id, kwargs = fake_hf["adapter_model"].loaded[0]
    assert adapter_id == "allenai/specter2_aug2023refresh_adhoc_query" and kwargs["local_files_only"] is True
    assert specter.ref.shape == (1, HIDDEN) and torch.isclose(specter.ref.norm(), torch.tensor(1.0))


def test_specter_falls_back_to_a_download(fake_hf):
    fake_hf["local_fails"] = True
    specter = Specter(adapter="proximity")
    assert fake_hf["adapter_model"].loaded == [("allenai/specter2", {"source": "hf", "load_as": "specter2",
                                                                      "set_active": True})]
    assert ("tokenizer", "allenai/specter2_aug2023refresh_base", {}) in fake_hf["calls"]
    assert specter.sep_token == "[SEP]"


def test_specter_without_an_adapter_or_with_a_custom_one(fake_hf):
    plain = Specter(adapter=None, orthgonalize=False)
    assert plain.specter is fake_hf["model"] and plain.f_transform(torch.ones(2)).tolist() == [1.0, 1.0]
    assert not hasattr(plain, "ref")
    custom = Specter(adapter="org/custom-adapter")
    assert custom.specter is fake_hf["model"] and fake_hf["model"].loaded == [("org/custom-adapter", {})]


def test_specter_encodes_every_payload_type(fake_hf):
    specter = Specter()
    calls = fake_hf["tokenizer"].calls
    assert specter("a text").shape == (1, HIDDEN) and calls[-1] == ["a text"]
    assert specter(["x", "y"]).shape == (2, HIDDEN) and calls[-1] == ["x", "y"]
    specter({"title": "T", "abstract": "A"})
    assert calls[-1] == ["T[SEP]A"]
    specter({"title": "T", "summary": "S"})
    assert calls[-1] == ["T[SEP]S"]
    specter([{"title": "T1"}, {"title": "T2", "abstract": "A2"}])
    assert calls[-1] == ["T1[SEP]", "T2[SEP]A2"]
    specter(pd.DataFrame({"title": ["T"], "abstract": ["A"]}))
    assert calls[-1] == ["T[SEP]A"]
    specter(pd.DataFrame({"title": ["T", None], "summary": ["S", "S2"]}))
    assert calls[-1] == ["T[SEP]S", "[SEP]S2"]
    specter(pd.DataFrame({"other": [1]}))
    assert calls[-1] == ["[SEP]"]
    specter(42)
    assert calls[-1] == ["42"]
    embedding = specter("a text")
    assert abs(float(embedding @ specter.ref.T)) < 1e-4  # orthogonalized against the empty-string reference


def test_specter_pooling_methods(fake_hf):
    specter = Specter(orthgonalize=False)
    hidden = torch.arange(24.0).reshape(1, 4, HIDDEN)
    mask = torch.tensor([[1, 1, 1, 0]])
    assert torch.equal(specter.pool(hidden, mask), hidden[:, 0])
    assert torch.allclose(specter.pool(hidden, mask, method="mean"), hidden[:, :3].mean(1))
    assert torch.allclose(specter.pool(hidden, mask, method="max"), hidden[:, :3].max(1).values)
    assert specter.pool(hidden, mask, method="mean_max").shape == (1, 2 * HIDDEN)
    attention = specter.pool(hidden, mask, method="attention")
    assert attention.shape == (1, HIDDEN)
    masked = hidden.clone()
    masked[:, 3] = 1e3
    assert torch.allclose(attention, specter.pool(masked, mask, method="attention"))


def test_specter_to_and_from_pretrained(fake_hf):
    specter = Specter.from_pretrained()
    assert isinstance(specter, Specter)
    assert specter.to("cpu") is specter and specter.device == "cpu"


def test_load_model_dispatches_every_mlp_alias(fake_hf, monkeypatch):
    import neurovlm.resources.loaders as resources

    monkeypatch.setattr(resources, "_proj_head_text_infonce", lambda: "text-infonce")
    monkeypatch.setattr(resources, "_proj_head_image_infonce", lambda: "image-infonce")
    monkeypatch.setattr(resources, "_proj_head_text_mse", lambda: "text-mse")
    monkeypatch.setattr(resources, "_load_autoencoder", lambda: "autoencoder")
    monkeypatch.setattr(resources, "_load_neuro_qformer", lambda **kwargs: ("qformer", kwargs))
    monkeypatch.setattr(resources, "_load_neuro_adapter", lambda: "adapter")

    assert ProjHead.from_pretrained("text_infonce") == "text-infonce"
    assert ProjHead.from_pretrained("image_infonce") == "image-infonce"
    assert ProjHead.from_pretrained("text_mse") == "text-mse"
    assert ProjHead.from_pretrained("other") is None
    assert NeuroAutoEncoder.from_pretrained() == "autoencoder"
    assert load_model("autoencoder") == "autoencoder"
    assert load_model("proj_head_text_infonce") == "text-infonce"
    assert load_model("proj_head_image_infonce") == "image-infonce"
    assert load_model(family="mlp", task="brain_to_text_retrieval") == "image-infonce"
    assert load_model("proj_head_text_mse") == "text-mse"
    assert load_model("neuro_adapter") == "adapter"
    assert load_model("neuro_qformer") == ("qformer", {})
    assert load_model("neuro_qformer_pubmed") == ("qformer", {"qformer_variant": "pubmed"})
    assert isinstance(load_model("specter"), Specter)
    with pytest.raises(NotImplementedError):
        ConceptClf.from_pretrained()
