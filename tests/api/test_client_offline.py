"""Offline tests of the NeuroVLM inference client.

Everything the client loads lazily (SPECTER, the autoencoder, the projection heads, the masker, the NeuroQFormer and
NeuroQwen pair, the text and brain corpora) is replaced by a small deterministic stand-in, injected through the
``load_model`` / ``load_dataset`` / ``load_latent`` / ``load_masker`` names of ``neurovlm.core.client`` and the private
attributes scripts/pipeline/validate_models.py also sets. The retrieval, generation, decoding and plotting paths then
run end to end without a download.
"""

from __future__ import annotations

import types
import zlib

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from neurovlm.core import client as client_module
from neurovlm.core.client import (
    BRAIN_FLAT_DIM,
    LATENT_DIM,
    TEXT_EMBED_DIM,
    BrainSearchResult,
    BrainTopKResult,
    NeuroVLM,
    TextSearchResult,
)
from neurovlm.models.base import ProjHead

SHAPE = (4, 4, 4)
PMIDS = np.array([101, 102, 103, 104])


# --------------------------------------------------------------------------------------------------------------------
# stand-ins
# --------------------------------------------------------------------------------------------------------------------
def _flatten_text(X) -> list[str]:
    if isinstance(X, str):
        return [X]
    if isinstance(X, dict):
        return [f"{X.get('title', '')} {X.get('abstract', X.get('summary', ''))}"]
    if isinstance(X, pd.DataFrame):
        return [" ".join(map(str, row)) for row in X.astype(str).itertuples(index=False)]
    return [item if isinstance(item, str) else _flatten_text(item)[0] for item in X]


def _embed(text: str, dim: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(zlib.crc32(text.encode()) % (2**31))
    return torch.randn(dim, generator=generator)


def _latent(n: int, dim: int, seed: int) -> torch.Tensor:
    return torch.randn(n, dim, generator=torch.Generator().manual_seed(seed))


class FakeSpecter:
    """Deterministic 768-d text encoder."""

    def __init__(self):
        self.device = torch.device("cpu")

    def to(self, device):
        self.device = torch.device(device)
        return self

    def __call__(self, X):
        return torch.stack([_embed(text, TEXT_EMBED_DIM) for text in _flatten_text(X)])


class _Encoder(nn.Module):
    def forward(self, x):
        return x[:, :LATENT_DIM]


class _Decoder(nn.Module):
    def forward(self, z):
        return z.repeat(1, BRAIN_FLAT_DIM // LATENT_DIM + 1)[:, :BRAIN_FLAT_DIM]


class FakeAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _Encoder()
        self.decoder = _Decoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))


class FakeAdapter(nn.Module):
    def forward(self, text):
        return text.repeat(1, BRAIN_FLAT_DIM // TEXT_EMBED_DIM + 1)[:, :BRAIN_FLAT_DIM]


class FakeMasker:
    """A 4x4x4 stand-in for the fitted NiftiMasker: 64 voxels tiled to the flat dimension."""

    def __init__(self):
        self.mask_img_ = nib.Nifti1Image(np.ones(SHAPE, dtype=np.float32), np.eye(4))

    def transform(self, image):
        data = np.asarray(image.get_fdata(), dtype=np.float32).ravel()
        return np.resize(data, (1, BRAIN_FLAT_DIM))

    def inverse_transform(self, flat):
        values = np.asarray(flat, dtype=np.float32).reshape(-1)[: int(np.prod(SHAPE))]
        return nib.Nifti1Image(values.reshape(SHAPE), np.eye(4))


class FakeQFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection_temp = 0.05
        self.canonical_basis = "network"
        self.use_canonical_projection = True
        self.calls: list = []

    def forward(self, latent, *, basis=None, projection_temp=None, use_canonical_projection=None):
        self.calls.append(basis)
        return latent[:, :16].unsqueeze(1).repeat(1, 4, 1)


class FakeTokenizer:
    eos_token_id = 0
    pad_token_id = None
    pad_token = None
    eos_token = "<eos>"

    def __call__(self, text, add_special_tokens=False, return_tensors="pt"):
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    def decode(self, row, skip_special_tokens=True):
        return " ".join(str(int(token)) for token in row)


class FakeLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(50, 16)
        self.config = types.SimpleNamespace(pad_token_id=None)
        self.generate_kwargs = None

    def get_input_embeddings(self):
        return self.embeddings

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return torch.arange(1, 4).repeat(kwargs["inputs_embeds"].shape[0], 1)


MODELS = {
    "proj_head_image_infonce": lambda: ProjHead(384, 384, 384, seed=0),
    "proj_head_text_infonce": lambda: ProjHead(768, 512, 384, seed=1),
    "proj_head_text_mse": lambda: ProjHead(768, 512, 384, seed=2),
    "specter": FakeSpecter,
    "autoencoder": FakeAutoencoder,
    "neuro_adapter": FakeAdapter,
}

DATASETS = {
    "pubmed_text": pd.DataFrame({"pmid": PMIDS, "title": [f"Paper {i}" for i in PMIDS],
                                 "abstract": [f"Abstract {i}" for i in PMIDS]}),
    "pubmed_summaries": pd.DataFrame({"pmid": PMIDS, "summary": [f"Summary {i}" for i in PMIDS]}),
    "wiki": pd.DataFrame({"id": [1, 2, 3], "title": ["Amygdala", "Cortex", "Thalamus"],
                          "description": ["fear", "sheet", "relay"]}),
    "cogatlas": pd.DataFrame({"term": ["Working Memory", "Attention"], "definition": ["hold", "select"]}),
    "pubmed_mesh": pd.DataFrame({"term": ["Hippocampus", "Dopamine"], "definition": ["memory", "reward"]}),
    "llm_neuro_terms": pd.DataFrame({"term": ["novel term"], "definition": ["new"]}),
    "networks_canonical": pd.DataFrame({"title": ["Default mode", "Visual"], "description": ["rest", "sight"]}),
    "pubmed_images": ((torch.rand(4, BRAIN_FLAT_DIM, generator=torch.Generator().manual_seed(0)) > 0.5).float(), PMIDS),
    "neurovault_images": (torch.rand(3, BRAIN_FLAT_DIM, generator=torch.Generator().manual_seed(1)) > 0.5).float(),
    "neurovault_images_meta": pd.DataFrame({"name": ["map a", "map b", None], "doi": ["d1", "d2", "d3"]}),
    "neurovault_text": pd.DataFrame({"doi": ["d1", "d2"], "title": ["NV paper 1", "NV paper 2"], "abstract": ["x", "y"]}),
}

LATENTS = {
    "pubmed_text": (_latent(4, TEXT_EMBED_DIM, 1), PMIDS),
    "pubmed_summaries": (_latent(4, TEXT_EMBED_DIM, 2), PMIDS),
    "wiki": (_latent(3, TEXT_EMBED_DIM, 3), np.array([1, 2, 3])),
    "cogatlas": (_latent(2, TEXT_EMBED_DIM, 4), np.array(["working memory", "attention"])),
    "pubmed_mesh": (_latent(2, TEXT_EMBED_DIM, 5), np.array(["Hippocampus", "Dopamine"])),
    "llm_neuro_terms": (_latent(1, TEXT_EMBED_DIM, 6), np.array(["novel term"])),
    "pubmed_images": (_latent(4, LATENT_DIM, 7), PMIDS),
    "networks_text": {"latent": _latent(2, TEXT_EMBED_DIM, 8)},
    "networks_neuro": {"Du": {"AUD": _latent(1, LATENT_DIM, 9)[0], "VIS": _latent(1, LATENT_DIM, 10)[0]},
                       "Yeo": {"DMN": _latent(1, LATENT_DIM, 11)[0]}, "version": 2},
    "neurovault_images": _latent(3, LATENT_DIM, 12),
}
NGRAM_LABELS = np.array(["n-gram a", "n-gram b", "n-gram c"])


def _fake_load_dataset(name):
    if name in DATASETS:
        return DATASETS[name]
    raise ValueError(f"{name} is not a fixture dataset")


def _fake_load_latent(name):
    if name in LATENTS:
        return LATENTS[name]
    raise ValueError(f"{name} is not a fixture latent")


@pytest.fixture
def nvlm(monkeypatch):
    import neurovlm.resources.loaders as resources

    monkeypatch.setattr(client_module, "load_model", lambda name: MODELS[name]())
    monkeypatch.setattr(client_module, "load_masker", FakeMasker)
    monkeypatch.setattr(client_module, "load_dataset", _fake_load_dataset)
    monkeypatch.setattr(client_module, "load_latent", _fake_load_latent)
    monkeypatch.setattr(resources, "_load_latent_ngram", lambda: _latent(3, LATENT_DIM, 13))
    monkeypatch.setattr(resources, "_load_ngram", lambda: NGRAM_LABELS)
    return NeuroVLM(device="cpu")


@pytest.fixture
def plotted(monkeypatch):
    """Replace nilearn's plot_stat_map and record its keyword arguments."""
    import nilearn.plotting

    calls: list[dict] = []

    def fake_plot_stat_map(image, **kwargs):
        assert isinstance(image, nib.Nifti1Image)
        calls.append(kwargs)
        return "plotted"

    monkeypatch.setattr(nilearn.plotting, "plot_stat_map", fake_plot_stat_map)
    return calls


def _nifti(binary: bool = False) -> nib.Nifti1Image:
    data = np.random.rand(*SHAPE).astype(np.float32)
    if binary:
        data = (data > 0.5).astype(np.float32)
    return nib.Nifti1Image(data, np.eye(4))


# --------------------------------------------------------------------------------------------------------------------
# text-to-brain
# --------------------------------------------------------------------------------------------------------------------
def test_text_to_brain_mse_generates_maps(nvlm, plotted):
    result = nvlm.text("auditory processing").to_brain(head="mse")
    assert result.retrieval_space == "mse"
    assert result.generated_flatmaps.shape == (1, BRAIN_FLAT_DIM)
    assert nvlm.last_generated_brain_flat is result.generated_flatmaps
    assert nvlm.results is result and nvlm._last_text_query == "auditory processing"
    images = result.images
    assert len(images) == 1 and isinstance(images[0], nib.Nifti1Image)
    assert isinstance(result.to_nifti(index=0), nib.Nifti1Image)
    assert isinstance(nvlm.get_niftis(0), nib.Nifti1Image)
    assert result.plot(0, threshold=0.1) == "plotted" and plotted[-1]["threshold"] == 0.1
    with pytest.raises(IndexError):
        result.to_nifti(index=5)
    with pytest.raises(IndexError):
        result.plot(3)
    with pytest.raises(ValueError):
        result.plot(-1)
    with pytest.raises(ValueError):
        result.top_k()
    with pytest.raises(ValueError):
        result.df


def test_text_to_brain_accepts_every_text_payload(nvlm):
    assert nvlm.to_brain(["a", "b"], head="mse").generated_flatmaps.shape == (2, BRAIN_FLAT_DIM)
    assert nvlm._last_text_query is None
    nvlm.to_brain(["only"], head="mse")
    assert nvlm._last_text_query == "only"
    nvlm.to_brain({"title": "t", "abstract": "a"}, head="mse")
    assert nvlm._last_text_query is None
    nvlm.to_brain(pd.DataFrame({"title": ["t"], "abstract": ["a"]}), head="mse")
    nvlm.to_brain(torch.randn(TEXT_EMBED_DIM), head="mse")
    nvlm.to_brain(np.random.rand(2, TEXT_EMBED_DIM).astype(np.float32), head="mse")
    with pytest.raises(ValueError):
        nvlm.to_brain(torch.randn(10), head="mse")
    with pytest.raises(TypeError):
        nvlm.to_brain(object(), head="mse")
    with pytest.raises(ValueError):
        nvlm.to_brain("x", head="bogus")


def test_to_brain_from_brain_inputs(nvlm):
    latent = torch.randn(LATENT_DIM)
    result = nvlm.to_brain(latent, head="mse")
    assert torch.allclose(result.latents, latent.unsqueeze(0))
    flat = (torch.rand(BRAIN_FLAT_DIM) > 0.5).float()
    assert nvlm.to_brain(flat, head="mse").latents.shape == (1, LATENT_DIM)
    assert nvlm.to_brain(_nifti(), head="infonce", dataset="pubmed").retrieval_space == "infonce"
    assert nvlm.to_brain(_nifti(binary=True), head="mse").generated_flatmaps.shape == (1, BRAIN_FLAT_DIM)
    assert nvlm.to_brain(latent, head="infonce", dataset="networks").scores.shape == (3, 1)
    with pytest.raises(ValueError):
        nvlm._encode_brain_flat(torch.zeros(1, 10))


def test_text_to_brain_infonce_retrieves_across_corpora(nvlm, plotted):
    result = nvlm.text("visual cortex").to_brain(head="infonce")
    assert result.retrieval_space == "infonce"
    assert result.scores.shape == (4 + 3 + 3, 1)
    meta = result.metadata
    assert meta.loc[meta["dataset"] == "pubmed", "title"].tolist() == [f"Paper {i}" for i in PMIDS]
    assert meta.loc[meta["dataset"] == "networks", "title"].tolist() == ["Du", "Du", "Yeo"]
    assert meta.loc[meta["dataset"] == "networks", "description"].tolist() == ["AUD", "VIS", "DMN"]
    assert meta.loc[meta["dataset"] == "neurovault", "title"].tolist() == ["NV paper 1", "NV paper 2", "neurovault_2"]
    assert meta.loc[meta["dataset"] == "neurovault", "description"].tolist() == ["x", "y", ""]

    top = result.top_k(2)
    assert isinstance(top, BrainTopKResult) and len(top) == 6
    assert list(top.table.columns) == ["dataset", "dataset_index", "title", "description", "cosine_similarity"]
    assert top["dataset"].tolist() == ["networks", "networks", "neurovault", "neurovault", "pubmed", "pubmed"]
    assert "cosine_similarity" in repr(top) and top.to_pandas() is top.table
    assert len(top.to_nifti()) == 6 and len(top.to_dataset_nifti()) == 6
    assert top.plot_row(0) == "plotted" and plotted[-1]["title"] == "Du"
    assert top.plot_row(1, source="decoded") == "plotted"
    with pytest.raises(IndexError):
        top.plot_row(99)
    assert result.plot_topk_item(top.table.iloc[4]) == "plotted" and plotted[-1]["title"].startswith("Paper")
    with pytest.raises(TypeError):
        result.plot_topk_item("row")

    table, niftis = result.top_k_niftis(k=1)
    assert len(table) == 3 and len(niftis) == 3
    assert len(result.df) == 10
    result.print(k=1)
    assert result.plot(index=0, query_index=0) == "plotted"
    with pytest.raises(IndexError):
        result.plot(index=0, query_index=3)
    with pytest.raises(IndexError):
        result.plot(index=50)
    with pytest.raises(ValueError):
        result.top_k(k=0)
    assert isinstance(nvlm.top_k(k=1, target="brain"), BrainTopKResult)
    with pytest.raises(ValueError):
        nvlm.top_k(k=1, target="brain", dataset="pubmed")


def test_multi_query_brain_retrieval_and_dataset_selection(nvlm):
    result = nvlm.to_brain(["a", "b"], head="infonce", dataset=["pubmed", "nv"])
    assert result.scores.shape == (7, 2)
    top = result.top_k(1)
    assert top.table["query_index"].tolist() == [0, 0, 1, 1]
    assert set(result.top_k(2, query_index=1)["query_index"]) == {1}
    with pytest.raises(IndexError):
        result.top_k(1, query_index=2)

    assert nvlm.to_brain("q", head="infonce", dataset="pubmed").scores.shape[0] == 4
    assert nvlm.to_brain("q", head="infonce", dataset=("network", "networks")).scores.shape[0] == 3
    assert nvlm.retrieve_brain("q", dataset="neurovault").scores.shape[0] == 3
    assert nvlm.text("q").retrieve_brain(dataset="neuro").scores.shape[0] == 4
    assert nvlm.generate_brain("q").retrieval_space == "mse"
    with pytest.raises(ValueError):
        nvlm.to_brain("q", head="infonce", dataset="moon")
    with pytest.raises(ValueError):
        nvlm.brain(torch.randn(LATENT_DIM)).generate_brain()


def test_adapter_text_to_brain(nvlm):
    result = nvlm.text("motor").to_brain(head="mse", adapter=True)
    assert result.generated_flatmaps.shape == (1, BRAIN_FLAT_DIM) and result.latents is None
    assert result.query_embeddings.shape == (1, TEXT_EMBED_DIM)
    assert nvlm.generate_brain(torch.randn(2, TEXT_EMBED_DIM), adapter=True).generated_flatmaps.shape == (2, BRAIN_FLAT_DIM)
    assert nvlm.text("motor").generate_brain(adapter=True).retrieval_space == "mse"
    with pytest.raises(ValueError):
        nvlm.to_brain("motor", head="infonce", adapter=True)
    with pytest.raises(ValueError):
        nvlm.to_brain(torch.randn(LATENT_DIM), head="mse", adapter=True)
    with pytest.raises(TypeError):
        nvlm.to_brain(object(), head="mse", adapter=True)


# --------------------------------------------------------------------------------------------------------------------
# brain-to-text and text-to-text retrieval
# --------------------------------------------------------------------------------------------------------------------
def test_brain_to_text_retrieval(nvlm):
    latent = torch.randn(LATENT_DIM)
    result = nvlm.brain(latent).to_text(head="infonce")
    assert isinstance(result, TextSearchResult) and result.retrieval_space == "shared"
    assert set(result.scores_by_dataset) == {"pubmed", "wiki", "cogatlas", "ngrams", "pubmed_mesh", "llm_neuro_terms"}
    assert result.metadata_by_dataset["pubmed"]["title"].tolist() == [f"Paper {i}" for i in PMIDS]
    # term lookups are case-insensitive, so the aligned title is the lower-cased lookup key
    assert result.metadata_by_dataset["cogatlas"]["title"].tolist() == ["working memory", "attention"]
    assert result.metadata_by_dataset["pubmed_mesh"]["description"].tolist() == ["memory", "reward"]
    assert result.metadata_by_dataset["ngrams"]["title"].tolist() == NGRAM_LABELS.tolist()

    top = result.top_k(3)
    assert len(top) == 3 and top["cosine_similarity"].is_monotonic_decreasing
    wiki = result.top_k(2, dataset="wiki")
    assert set(wiki["dataset"]) == {"wiki"} and len(wiki) == 2
    assert len(result.df) == 4
    result.print(k=1)
    assert "cosine_similarity" in result.format(k=1)
    assert isinstance(nvlm.top_k(k=2), pd.DataFrame)
    assert nvlm.last_text_result is result and nvlm.results is result
    with pytest.raises(ValueError):
        result.plot()
    with pytest.raises(ValueError):
        result.top_k(dataset="moon")
    with pytest.raises(ValueError):
        result.top_k(k=0)

    flat = (torch.rand(BRAIN_FLAT_DIM) > 0.5).float()
    image = _nifti()
    assert nvlm.to_text(flat, head="infonce", datasets=["wiki"]).retrieval_space == "shared"
    assert nvlm.to_text(image, head="infonce", datasets=["wiki"]).retrieval_space == "shared"
    pair = nvlm.to_text([torch.randn(LATENT_DIM), torch.randn(LATENT_DIM)], head="infonce", datasets=["wiki"])
    assert pair.query_embeddings.shape == (2, LATENT_DIM)
    for brain_input in (latent, flat, image):
        with pytest.raises(ValueError):
            nvlm.to_text(brain_input, head="infonce", project=False)
    with pytest.raises(ValueError):
        nvlm.to_text(torch.randn(9), head="infonce")
    with pytest.raises(TypeError):
        nvlm.to_text(object(), head="infonce")


def test_multi_query_text_retrieval(nvlm):
    result = nvlm.to_text(torch.randn(2, LATENT_DIM), head="infonce", datasets=["wiki", "cogatlas"])
    top = result.top_k(2)
    assert top["query_index"].tolist() == [0, 0, 1, 1]
    assert set(result.top_k(1, query_index=1)["query_index"]) == {1}
    assert result.top_k(1, dataset="wiki")["dataset"].tolist() == ["wiki", "wiki"]
    assert len(result.df) == 2 * 3
    assert TextSearchResult({}, {}, torch.zeros(1, 3), "shared").df.empty


def test_text_to_text_retrieval_raw_and_shared(nvlm):
    raw = nvlm.to_text("memory", head="infonce", project=False, datasets=["pubmed", "wiki"])
    assert raw.retrieval_space == "raw_text" and raw.query_embeddings.shape == (1, TEXT_EMBED_DIM)
    shared = nvlm.text("memory").to_text(head="infonce", datasets=["networks", "summaries", "concepts"])
    assert shared.retrieval_space == "shared"
    assert set(shared.scores_by_dataset) == {"networks", "pubmed_summaries", "cogatlas"}
    assert shared.metadata_by_dataset["networks"]["title"].tolist() == ["Default mode", "Visual"]
    assert shared.metadata_by_dataset["pubmed_summaries"]["description"].tolist() == [f"Summary {i}" for i in PMIDS]
    embedding = torch.randn(TEXT_EMBED_DIM)
    assert nvlm.to_text(embedding, head="infonce", datasets=["wiki"]).retrieval_space == "shared"
    assert nvlm.to_text(embedding, head="infonce", project=False, datasets=["wiki"]).retrieval_space == "raw_text"
    assert nvlm.retrieve_text("memory", datasets=["wiki"]).retrieval_space == "shared"
    assert nvlm.text("memory").retrieve_text(datasets=["wiki"]).retrieval_space == "shared"
    with pytest.raises(ValueError):
        nvlm.to_text("x", head="infonce", datasets=["moon"])
    with pytest.raises(ValueError):
        nvlm.to_text("x", head="qformer", datasets=["wiki"])
    with pytest.raises(ValueError):
        nvlm.to_text("x", head="bogus")
    with pytest.raises(TypeError):
        nvlm.to_text("x", head="infonce", num_beams=2)
    with pytest.raises(TypeError):
        nvlm.to_text("x", method="infonce")
    with pytest.raises(ValueError):
        nvlm.text("x").to_text()
    with pytest.raises(ValueError):
        nvlm.text("x").generate_text()


def test_constructor_and_errors_before_any_query(nvlm):
    with pytest.raises(ValueError):
        nvlm.results
    with pytest.raises(ValueError):
        nvlm.top_k()
    with pytest.raises(ValueError):
        nvlm.top_k(target="brain")
    with pytest.raises(ValueError):
        nvlm.get_niftis()
    with pytest.raises(ValueError):
        NeuroVLM(datasets=["moon"])
    assert NeuroVLM(datasets=["publications", "wiki", "pubmed"]).datasets == ("pubmed", "wiki")


# --------------------------------------------------------------------------------------------------------------------
# brain-to-text generation
# --------------------------------------------------------------------------------------------------------------------
def test_generate_text_with_fake_qformer_and_language_model(nvlm, monkeypatch):
    import transformers
    import neurovlm.resources.loaders as resources

    tokenizer, language_model = FakeTokenizer(), FakeLanguageModel()
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda *args, **kwargs: language_model)
    loaded: list[FakeQFormer] = []

    def fake_load_qformer(**kwargs):
        qformer = FakeQFormer()
        qformer.kwargs = kwargs
        loaded.append(qformer)
        return qformer

    monkeypatch.setattr(resources, "_load_neuro_qformer", fake_load_qformer)

    latent = torch.randn(LATENT_DIM)
    text = nvlm.brain(latent).to_text()
    assert text == "[NETWORK]1 2 3"
    assert nvlm._neuro_qwen_token_norm is not None and len(loaded) == 1
    assert loaded[0].kwargs["qformer_variant"] == "canonical" and loaded[0].kwargs["canonical_basis"] == "network"
    kwargs = language_model.generate_kwargs
    assert kwargs["num_beams"] == 3 and kwargs["inputs_embeds"].shape == (1, 4 + 3, 16)
    assert kwargs["temperature"] == 1.0 and kwargs["top_k"] == 50 and kwargs["pad_token_id"] == 0

    # the loaded pair is reused; the projector is re-tuned in place
    texts = nvlm.generate_text(torch.randn(2, LATENT_DIM), basis="region", projection_temp=0.1)
    assert texts == ["[REGION]1 2 3", "[REGION]1 2 3"]
    assert len(loaded) == 1 and loaded[0].projection_temp == 0.1 and loaded[0].calls[-1] == "region"
    nvlm._neuro_qwen_token_norm = None
    assert nvlm.generate_text(latent, basis=None, use_canonical_projection=False) == "1 2 3"
    assert nvlm._neuro_qwen_token_norm is not None and loaded[0].use_canonical_projection is False
    assert nvlm.generate_text(latent, prefix_text="") == "1 2 3"
    assert nvlm.generate_text(latent, basis="cognitive", prefix_text="Prefix: ") == "Prefix: 1 2 3"

    # the pubmed variant loads the other package and uses no prefix
    assert nvlm.generate_text(latent, qformer_variant="pubmed") == "1 2 3"
    assert len(loaded) == 2 and loaded[1].kwargs["qformer_variant"] == "pubmed"
    assert language_model.generate_kwargs["inputs_embeds"].shape == (1, 4, 16)

    nvlm.generate_text(latent, do_sample=True, temperature=0.7, top_p=0.9, seed=None)
    assert language_model.generate_kwargs["temperature"] == 0.7 and language_model.generate_kwargs["top_p"] == 0.9
    assert isinstance(nvlm.generate_text((torch.rand(BRAIN_FLAT_DIM) > 0.5).float()), str)
    assert isinstance(nvlm.brain(_nifti()).generate_text(), str)
    with pytest.raises(TypeError):
        nvlm.generate_text("text")
    with pytest.raises(TypeError):
        nvlm.generate_text(object())
    with pytest.raises(ValueError):
        nvlm.generate_text(torch.randn(9))
    with pytest.raises(ValueError):
        nvlm.generate_text(latent, qformer_variant="other")
    assert NeuroVLM._generation_prefix_for_basis("functions") == "[FUNCTION]"
    assert NeuroVLM._generation_prefix_for_basis("all") is None


# --------------------------------------------------------------------------------------------------------------------
# decoding, plotting, loaders and payload helpers
# --------------------------------------------------------------------------------------------------------------------
def test_decode_brain_and_static_plot(nvlm, plotted):
    assert isinstance(nvlm.decode_brain(torch.randn(LATENT_DIM)), nib.Nifti1Image)
    assert len(nvlm.decode_brains(np.random.rand(3, LATENT_DIM).astype(np.float32))) == 3
    with pytest.raises(ValueError):
        nvlm.decode_brain(torch.randn(10))
    assert NeuroVLM.plot(torch.rand(BRAIN_FLAT_DIM)) == "plotted"
    assert NeuroVLM.plot(_nifti(), title="t") == "plotted" and plotted[-1]["title"] == "t"
    with pytest.raises(TypeError):
        NeuroVLM.plot("nope")


def test_text_encoder_output_is_checked(nvlm):
    nvlm._specter = lambda X: torch.zeros(1, 10)
    with pytest.raises(ValueError):
        nvlm.to_brain("x", head="mse")


def test_pubmed_image_payloads_and_dataset_images(nvlm, monkeypatch):
    images = (torch.rand(3, BRAIN_FLAT_DIM) > 0.5).float()
    monkeypatch.setitem(DATASETS, "pubmed_images", {"images": images, "pmid": [101, 102, 999]})
    nvlm._ensure_pubmed_images_flat()
    assert nvlm._pubmed_pmid_to_image_index == {101: 0, 102: 1, 999: 2}
    assert nvlm._resolve_pubmed_dataset_indices(np.array([102, 555, 101, 777, 888])).tolist() == [1, 1, 0, 2, 2]
    assert isinstance(nvlm._get_dataset_nifti("papers", 2), nib.Nifti1Image)
    assert isinstance(nvlm._get_dataset_nifti("neurovault_images", 0), nib.Nifti1Image)
    assert isinstance(nvlm._get_dataset_nifti("network", 1), nib.Nifti1Image)
    with pytest.raises(IndexError):
        nvlm._get_dataset_nifti("pubmed", -1)
    with pytest.raises(IndexError):
        nvlm._get_dataset_nifti("pubmed", 3)
    with pytest.raises(IndexError):
        nvlm._get_dataset_nifti("neurovault", 3)
    with pytest.raises(IndexError):
        nvlm._get_dataset_nifti("networks", 3)
    with pytest.raises(ValueError):
        nvlm._get_dataset_nifti("moon", 0)

    plain = NeuroVLM()
    monkeypatch.setitem(DATASETS, "pubmed_images", images)
    assert plain._resolve_pubmed_dataset_indices(np.array([5, 6])).tolist() == [0, 1]
    monkeypatch.setitem(DATASETS, "pubmed_images", "bad")
    with pytest.raises(ValueError):
        NeuroVLM()._ensure_pubmed_images_flat()
    monkeypatch.setitem(DATASETS, "neurovault_images", "bad")
    with pytest.raises(TypeError):
        NeuroVLM()._ensure_neurovault_images_flat()
    monkeypatch.setitem(DATASETS, "neurovault_images_meta", "bad")
    with pytest.raises(TypeError):
        NeuroVLM()._build_neurovault_brain_metadata(n_rows=1)


def test_metadata_alignment_fallbacks(nvlm):
    brain_meta = nvlm._build_aligned_brain_metadata(np.array([999, 101]))
    assert brain_meta["title"].tolist() == ["Paper 101", "Paper 101"]
    assert nvlm._build_aligned_brain_metadata(np.array([999] * 6))["title"].tolist()[-1] == "PMID 999"
    table = pd.DataFrame({"title": ["t0", "t1"], "description": ["d0", "d1"]})
    assert nvlm._build_lookup("wiki", table) is None
    aligned = nvlm._build_aligned_text_metadata("wiki", np.array([7, 8, 9]), table, None)
    assert aligned["title"].tolist() == ["t0", "t1", "9"] and aligned["description"].tolist() == ["d0", "d1", ""]
    with pytest.raises(TypeError):
        nvlm._load_text_dataset_table("pubmed_images")


def test_payload_helpers(nvlm):
    latent = torch.randn(2, LATENT_DIM)
    assert nvlm._unpack_latent_with_ids((latent,), default_id_key="pmid")[1].tolist() == [0, 1]
    unpacked = nvlm._unpack_latent_with_ids({"embeddings": latent.numpy(), "term": ["a", "b"]}, default_id_key="term")
    assert unpacked[1].tolist() == ["a", "b"]
    assert nvlm._unpack_latent_with_ids(latent.numpy(), default_id_key="id")[0].shape == (2, LATENT_DIM)
    with pytest.raises(ValueError):
        nvlm._unpack_latent_with_ids("x", default_id_key="id")
    with pytest.raises(ValueError):
        nvlm._unpack_latent_with_ids({"term": ["a"]}, default_id_key="term")

    assert nvlm._extract_latent_tensor(latent) is latent
    assert nvlm._extract_latent_tensor(latent.numpy()).shape == (2, LATENT_DIM)
    assert nvlm._extract_latent_tensor((latent.numpy(), None)).shape == (2, LATENT_DIM)
    assert nvlm._extract_latent_tensor({"images": latent}).shape == (2, LATENT_DIM)
    with pytest.raises(ValueError):
        nvlm._extract_latent_tensor({"other": 1})

    assert nvlm._extract_networks_canonical_text_latent((latent,)).shape == (2, LATENT_DIM)
    assert nvlm._extract_networks_canonical_text_latent({"embeddings": latent}).shape == (2, LATENT_DIM)
    assert nvlm._extract_networks_canonical_text_latent(latent[0]).shape == (1, LATENT_DIM)
    with pytest.raises(ValueError):
        nvlm._extract_networks_canonical_text_latent({"x": 1})
    with pytest.raises(ValueError):
        nvlm._flatten_networks_neuro_payload([1])
    with pytest.raises(ValueError):
        nvlm._flatten_networks_neuro_payload({"atlas": {"net": torch.randn(2, LATENT_DIM)}})
    with pytest.raises(ValueError):
        nvlm._flatten_networks_neuro_payload({"atlas": 3})

    frame = pd.DataFrame({"network_standard_name": [" Default  mode "]})
    meta = nvlm._build_networks_canonical_metadata(frame, n_rows=3)
    assert meta["title"].tolist() == ["Default mode", "network_1", "network_2"]
    assert meta["description"].tolist() == ["", "", ""]
    three = pd.DataFrame({"title": ["a", "b", "c"], "description": ["1", "2", "3"]})
    assert len(nvlm._build_networks_canonical_metadata(three, n_rows=2)) == 2
    with pytest.raises(ValueError):
        nvlm._build_networks_canonical_metadata(pd.DataFrame({"x": [1]}), n_rows=1)

    assert nvlm._coerce_numeric_query([torch.randn(3), torch.randn(3)]).shape == (2, 3)
    assert nvlm._coerce_numeric_query([1.0, 2.0]).shape == (1, 2)
    assert nvlm._coerce_numeric_query([[1.0, 2.0]]).shape == (1, 2)
    assert nvlm._coerce_numeric_query(["a"]) is None and nvlm._coerce_numeric_query([]) is None
    with pytest.raises(ValueError):
        nvlm._as_2d_tensor(torch.zeros(2, 2, 2))
    assert nvlm._resolve_brain_datasets(None) == ["neuro", "networks", "neurovault"]
    with pytest.raises(ValueError):
        nvlm._load_dataset_compat()
    with pytest.raises(ValueError):
        nvlm._load_latent_compat()
    with pytest.raises(ValueError):
        nvlm._load_dataset_compat("moon", "sun")
    with pytest.raises(ValueError):
        nvlm._load_latent_compat("moon")


def test_result_guards_without_attached_models():
    empty = BrainSearchResult(scores=None, metadata=None, latents=None, query_embeddings=torch.zeros(1, 3),
                              retrieval_space="mse")
    for call in (empty.to_nifti, lambda: empty.images, lambda: empty.niftis_for(pd.DataFrame()),
                 lambda: empty.dataset_niftis_for(pd.DataFrame()), empty.plot, lambda: empty.plot_topk_row(pd.DataFrame([{}])),
                 lambda: empty.plot_topk_item(pd.Series({"dataset": "pubmed", "dataset_index": 0}))):
        with pytest.raises(ValueError):
            call()

    masker = FakeMasker()
    latent_only = BrainSearchResult(scores=None, metadata=None, latents=torch.zeros(1, 3),
                                    query_embeddings=torch.zeros(1, 3), retrieval_space="mse", masker=masker)
    with pytest.raises(ValueError):
        latent_only.to_nifti()
    with pytest.raises(ValueError):
        latent_only.niftis_for(pd.DataFrame())
    with pytest.raises(ValueError):
        BrainSearchResult(scores=None, metadata=None, latents=None, query_embeddings=torch.zeros(1, 3),
                          retrieval_space="mse", masker=masker).to_nifti()

    getter = lambda dataset, index: nib.Nifti1Image(np.zeros(SHAPE, dtype=np.float32), np.eye(4))  # noqa: E731
    with_getter = BrainSearchResult(scores=None, metadata=None, latents=None, query_embeddings=torch.zeros(1, 3),
                                    retrieval_space="mse", dataset_image_getter=getter)
    table = pd.DataFrame({"dataset": ["pubmed"], "dataset_index": [0], "title": ["x"]})
    assert len(with_getter.dataset_niftis_for(table)) == 1
    assert with_getter._dataset_row_ref(table, 0) == ("pubmed", 0)
    with pytest.raises(ValueError):
        with_getter.dataset_niftis_for(pd.DataFrame({"title": ["x"]}))
    with pytest.raises(ValueError):
        with_getter._dataset_row_ref(pd.DataFrame({"title": ["x"]}), 0)
    with pytest.raises(ValueError):
        with_getter.plot_topk_row(table, source="decoded")
    with pytest.raises(ValueError):
        with_getter.plot_topk_item(pd.Series({"title": "x"}))
