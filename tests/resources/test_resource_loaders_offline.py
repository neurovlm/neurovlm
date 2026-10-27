"""Offline tests of the pinned-resource loaders.

``_download_from_hf`` is replaced by a stand-in that writes a small fixture file for each (repository, filename) the
loaders ask for, so every loader of ``neurovlm.resources.loaders`` runs against real parquet, torch, npz, safetensors,
json and pickle files without touching the Hub. The autoencoder is built at toy dimensions through the module's
``NeuroAutoEncoder`` name; the released dimensions would need a 100 MB fixture.
"""

from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from safetensors.torch import save_file

import neurovlm.resources.loaders as rr
from neurovlm.models.adapter import InterleavedDecoderAdapter
from neurovlm.models.base import NeuroAutoEncoder, ProjHead
from neurovlm.models.qformer import NeuroQFormer

TINY_AUTOENCODER = {"dim_neuro": 32, "dim_h0": 12, "dim_h1": 10, "dim_latent": 8}
MESH_TERMS = ["Hippocampus", "Dopamine", "Working memory/physiology", "Caffeine"]


def _tiny_autoencoder(**kwargs) -> NeuroAutoEncoder:
    return NeuroAutoEncoder(**{**TINY_AUTOENCODER, **kwargs})


def _tiny_qformer() -> NeuroQFormer:
    torch.manual_seed(0)
    banks = {"all": torch.randn(3, 8), "network": torch.randn(2, 8)}
    return NeuroQFormer(image_dim=8, semantic_dim=8, lm_dim=16, num_queries=2, width=16, depth=1, align_hidden=8,
                        canonical_banks=banks).eval()


def _frame(filename: str) -> pd.DataFrame:
    if filename.startswith("cogatlas"):
        return pd.DataFrame({"term": ["Working Memory", "Attention"], "definition": ["hold", "select"],
                             "file": [filename, filename]})
    tables = {
        "publications.parquet": {"pmid": [1, 2, 3, 4], "title": ["p1", "p2", "p3", "p4"],
                                 "abstract": ["a1", "a2", "a3", "a4"]},
        "neuro_summaries.parquet": {"pmid": [1, 2, 3, 4], "summary": ["s1", "s2", "s3", "s4"]},
        "coordinates.parquet": {"pmid": [1, 1, 2], "x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0], "z": [0.0, 1.0, 2.0]},
        "neurowiki_with_ids.parquet": {"id": [1, 2, 3], "title": ["w1", "w2", "w3"], "description": ["d", "d", "d"]},
        "neurowiki_graph.parquet": {"source": [1, 2], "target": [2, 3]},
        "images_neurovault.parquet": {"name": ["a", "b", "c"], "doi": ["d1", "d2", "d3"]},
        "publications_neurovault.parquet": {"doi": ["d1", "d2"], "title": ["nv1", "nv2"]},
        "networks_labels.parquet": {"title": ["Du"], "label": ["AUD"]},
        "network_text.parquet": {"title": ["Default mode", "Visual"], "description": ["rest", "sight"]},
        "kg_mesh.parquet": {"term": MESH_TERMS, "definition": ["memory", "reward", "hold", "drug"]},
        "mesh_kg_nodes.parquet": {"name": ["Hippocampus", "Dopamine", "Working memory", "Caffeine"],
                                  "node_type": ["anatomical_region", "molecular", "cognitive_construct", "chemical"]},
        "mesh_descriptors.parquet": {"descriptor": ["D1"], "name": ["Hippocampus"]},
        "llm_neuro_terms.parquet": {"term": ["novel term"], "definition": ["new"]},
    }
    return pd.DataFrame(tables[filename])


def _tensor_payload(filename: str):
    generator = torch.Generator().manual_seed(len(filename))
    if filename in {"latent_neurovault_images.pt", "latent_neurovault_text.pt", "neurovault_images.pt"}:
        return torch.randn(3, 8, generator=generator)
    if filename == "pubmed_images.pt":
        return {"images": torch.rand(4, 16, generator=generator), "pmid": [1, 2, 3, 4]}
    if filename == "ngram_embeddings.pt":
        return torch.randn(3, 8, generator=generator)
    if filename == "latent_kg_mesh.pt":
        return {"latent": torch.randn(4, 8, generator=generator), "term": MESH_TERMS}
    if filename == "latent_specter_wiki.pt":
        return {"latent": torch.randn(3, 8, generator=generator), "id": [1, 2, 3]}
    if filename == "latent_networks_text.pt":
        return {"latent": torch.randn(2, 8, generator=generator)}
    if filename == "latent_networks_image.pt":
        return {"Du": {"AUD": torch.randn(8, generator=generator)}}
    if "cogatlas" in filename or filename == "latent_llm_neuro_terms.pt":
        return {"latent": torch.randn(2, 8, generator=generator), "term": ["a", "b"]}
    return {"latent": torch.randn(4, 8, generator=generator), "pmid": [1, 2, 3, 4]}


def _state_dict(repo_id: str, filename: str) -> dict:
    if repo_id == rr.NEURO_AUTOENCODER_REPO_ID:
        return _tiny_autoencoder(seed=0, out="logit").state_dict()
    if repo_id == rr.PROJECTION_HEADS_REPO_ID:
        dims = (384, 384, 384) if "image" in filename else (768, 512, 384)
        return ProjHead(*dims, seed=0).state_dict()
    if repo_id in (rr.NEURO_QFORMER_REPO_ID, rr.NEURO_QFORMER_PUBMED_REPO_ID):
        return _tiny_qformer().state_dict()
    if repo_id == rr.NEURO_ADAPTER_REPO_ID:
        adapter = InterleavedDecoderAdapter(_tiny_autoencoder(seed=0, out="logit"), ProjHead(768, 512, 384, seed=0),
                                            hidden_dim=16)
        return adapter.state_dict()
    raise KeyError(repo_id)


def _config(repo_id: str) -> dict:
    if repo_id in (rr.NEURO_QFORMER_REPO_ID, rr.NEURO_QFORMER_PUBMED_REPO_ID):
        return {"qformer": _tiny_qformer().architecture_config(), "canonical_metadata": {"all": ["x", "y", "z"]}}
    if repo_id == rr.NEURO_ADAPTER_REPO_ID:
        return {"adapter": {"hidden_dim": 16}}
    raise KeyError(repo_id)


def _write_fixture(target: Path, repo_id: str, filename: str) -> None:
    if filename.endswith(".parquet"):
        _frame(filename).to_parquet(target)
    elif filename.endswith(".pt"):
        torch.save(_tensor_payload(filename), target)
    elif filename == "ngram_labels.npy":
        np.save(target, np.array(["alpha", "beta", "gamma"]))
    elif filename == "mask.npz":
        np.savez(target, mask=np.ones((4, 4, 4)), affine=np.eye(4))
    elif filename == "networks_arrays.pkl.gz":
        with gzip.open(target, "wb") as handle:
            pickle.dump({"Du": {"AUD": [0, 1]}}, handle)
    elif filename == "network_test_set_labels.csv":
        target.write_text("title,label\nDu,AUD\nYeo,DMN\n")
    elif filename == "mesh_annotations.json":
        target.write_text(json.dumps({"1": ["Hippocampus"]}))
    elif filename.endswith(".safetensors"):
        save_file(_state_dict(repo_id, filename), str(target))
    elif filename == "config.json":
        target.write_text(json.dumps(_config(repo_id)))
    else:
        raise KeyError(f"no fixture for {repo_id}/{filename}")


class FakeHub:
    def __init__(self, root: Path):
        self.root = root
        self.requests: list[tuple[str, str, str]] = []

    def path(self, repo_id: str, filename: str, repo_type: str = "dataset") -> str:
        self.requests.append((repo_id, filename, repo_type))
        target = self.root / repo_id.replace("/", "--") / filename
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            _write_fixture(target, repo_id, filename)
        return str(target)


def _clear_caches() -> None:
    for name in dir(rr):
        if hasattr(getattr(rr, name), "cache_clear"):
            getattr(rr, name).cache_clear()


@pytest.fixture
def hub(tmp_path, monkeypatch):
    fake = FakeHub(tmp_path)
    monkeypatch.setattr(rr, "_download_from_hf", fake.path)
    monkeypatch.chdir(tmp_path)  # the MeSH and LLM-term loaders look for local overrides relative to the cwd
    _clear_caches()
    yield fake
    _clear_caches()


# --------------------------------------------------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------------------------------------------------
def test_dataset_loaders_read_the_fixture_tables(hub):
    assert list(rr._load_pubmed_dataframe().columns) == ["pmid", "title", "abstract"]
    assert "summary" in rr._load_pubmed_summaries_dataframe().columns
    assert set(rr._load_pubmed_coordinates().columns) >= {"x", "y", "z"}
    assert len(rr._load_neuro_wiki()) == 3 and len(rr._load_neuro_wiki_graph()) == 2
    assert len(rr._load_cogatlas_dataset()) == 2
    rr._load_cogatlas_task_dataset(filtered=True)
    assert hub.requests[-1] == ("neurovlm/cognitive_atlas", "cogatlas_task_filtered.parquet", "dataset")
    rr._load_cogatlas_task_dataset()
    assert hub.requests[-1][1] == "cogatlas_task.parquet"
    rr._load_cogatlas_disorder_dataset(filtered=True)
    assert hub.requests[-1][1] == "cogatlas_disorder_filtered.parquet"
    rr._load_cogatlas_disorder_dataset()
    assert hub.requests[-1][1] == "cogatlas_disorder.parquet"
    assert len(rr._load_cogatlas_graph_dataset()) == 2
    assert rr._load_networks() == {"Du": {"AUD": [0, 1]}}
    assert rr._load_networks_labels()["label"].tolist() == ["AUD"]
    assert rr._load_networks_canonical()["title"].tolist() == ["Default mode", "Visual"]
    assert rr._load_network_test_set_labels()["title"].tolist() == ["Du", "Yeo"]
    assert rr._load_pubmed_mesh_annotations() == {"1": ["Hippocampus"]}
    assert rr._load_images_neurovault_dataframe()["doi"].tolist() == ["d1", "d2", "d3"]
    assert rr._load_publications_neurovault_dataframe()["title"].tolist() == ["nv1", "nv2"]
    assert rr._load_llm_neuro_terms_dataset()["term"].tolist() == ["novel term"]
    assert rr._load_ngram().tolist() == ["alpha", "beta", "gamma"]
    # the lru-cached loaders do not hit the hub twice
    before = len(hub.requests)
    rr._load_pubmed_summaries_dataframe()
    rr._load_network_test_set_labels()
    assert len(hub.requests) == before


def test_latent_loaders(hub):
    latent, pmids = rr._load_latent_neuro()
    assert latent.shape == (4, 8) and pmids.tolist() == [1, 2, 3, 4]
    assert rr._load_latent_text()[0].shape == (4, 8)
    assert rr._load_latent_neuro_summaries()[1].tolist() == [1, 2, 3, 4]
    assert rr._load_latent_wiki()[1].tolist() == [1, 2, 3]
    for loader in (rr._load_latent_cogatlas, rr._load_latent_cogatlas_disorder, rr._load_latent_cogatlas_task):
        latent, terms = loader()
        assert latent.shape == (2, 8) and terms.tolist() == ["a", "b"]
    assert rr._load_latent_neurovault_images().shape == (3, 8)
    assert rr._load_latent_neurovault_text().shape == (3, 8)
    assert rr._load_neurovault_images().shape == (3, 8)
    images, pmids = rr._load_pubmed_images()
    assert images.shape == (4, 16) and list(pmids) == [1, 2, 3, 4]
    assert rr._load_latent_ngram().shape == (3, 8)
    assert rr._load_latent_llm_neuro_terms()[1].tolist() == ["a", "b"]
    assert rr._load_latent_networks_canonical_text()["latent"].shape == (2, 8)
    assert rr._load_latent_networks_neuro()["Du"]["AUD"].shape == (8,)


def test_tensor_loaders_reject_other_payloads(hub):
    for filename, loader in (("latent_neurovault_images.pt", rr._load_latent_neurovault_images),
                             ("latent_neurovault_text.pt", rr._load_latent_neurovault_text),
                             ("neurovault_images.pt", rr._load_neurovault_images)):
        torch.save({"not": "a tensor"}, hub.path("neurovlm/embedded_text", filename))
        with pytest.raises(TypeError):
            loader()


# --------------------------------------------------------------------------------------------------------------------
# MeSH
# --------------------------------------------------------------------------------------------------------------------
def test_mesh_loaders_filter_brain_rankable_node_types(hub):
    assert rr._normalize_mesh_term("Working memory/physiology") == "working memory"
    assert rr._normalize_mesh_term(None) == ""
    assert rr._brain_rankable_mesh_terms() == {"hippocampus", "working memory"}
    assert rr._brain_rankable_mesh_terms(include_molecular=True) == {"hippocampus", "working memory", "dopamine"}
    assert len(rr._load_kg_mesh_dataset()) == 4 and len(rr._load_mesh_kg_descriptors()) == 1
    assert rr._load_kg_mesh_brain_rankable_dataset()["term"].tolist() == ["Hippocampus", "Working memory/physiology"]
    assert len(rr._load_kg_mesh_brain_rankable_dataset(include_molecular=True)) == 3
    latent, terms = rr._load_latent_kg_mesh()
    assert latent.shape == (4, 8) and terms.tolist() == MESH_TERMS
    latent, terms = rr._load_latent_kg_mesh_brain_rankable(include_molecular=True)
    assert latent.shape == (3, 8) and "Dopamine" in terms.tolist()
    assert rr._load_latent_kg_mesh_brain_rankable()[0].shape == (2, 8)


def test_mesh_nodes_need_a_node_type_column(hub):
    pd.DataFrame({"name": ["x"]}).to_parquet(hub.path("neurovlm/mesh_kg", "mesh_kg_nodes.parquet"))
    with pytest.raises(ValueError):
        rr._brain_rankable_mesh_terms()


def test_local_overrides_take_precedence_over_the_hub(hub, tmp_path):
    local = tmp_path / "experiments/data/mesh_kg/mesh_kg_nodes.parquet"
    local.parent.mkdir(parents=True)
    pd.DataFrame({"name": ["Only local"], "node_type": ["disorder"]}).to_parquet(local)
    assert rr._load_mesh_kg_nodes()["name"].tolist() == ["Only local"]
    assert ("neurovlm/mesh_kg", "mesh_kg_nodes.parquet", "dataset") not in hub.requests


# --------------------------------------------------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------------------------------------------------
def test_model_loaders_rebuild_the_packaged_models(hub, monkeypatch):
    monkeypatch.setattr(rr, "NeuroAutoEncoder", _tiny_autoencoder)
    autoencoder = rr._load_autoencoder()
    assert isinstance(autoencoder, NeuroAutoEncoder) and not autoencoder.training
    assert hub.requests[-1] == (rr.NEURO_AUTOENCODER_REPO_ID, "autoencoder.safetensors", "model")

    masker = rr._load_masker()
    assert masker.mask_img_.shape == (4, 4, 4)
    assert masker.transform(masker.mask_img_).size == 64

    for loader, shape in ((rr._proj_head_image_infonce, (384, 384)), (rr._proj_head_text_mse, (512, 768)),
                          (rr._proj_head_text_infonce, (512, 768))):
        head = loader()
        assert isinstance(head, ProjHead) and head.aligner[0].weight.shape == shape and not head.training

    qformer = rr._load_neuro_qformer(projection_temp=0.2, canonical_basis="network", use_canonical_projection=False)
    assert isinstance(qformer, NeuroQFormer) and not qformer.training
    assert qformer.projection_temp == 0.2 and qformer.canonical_basis == "network"
    assert qformer.use_canonical_projection is False
    assert hub.requests[-1][0] == rr.NEURO_QFORMER_REPO_ID
    with torch.no_grad():
        assert qformer(torch.randn(3, 8)).shape == (3, 2, 16)
    rr._load_neuro_qformer(qformer_variant="pubmed")
    assert hub.requests[-1][0] == rr.NEURO_QFORMER_PUBMED_REPO_ID
    assert rr._neuro_qformer_repo_id(None) == rr.NEURO_QFORMER_REPO_ID
    with pytest.raises(ValueError):
        rr._neuro_qformer_repo_id("other")

    adapter = rr._load_neuro_adapter()
    assert isinstance(adapter, InterleavedDecoderAdapter) and not adapter.training
    assert adapter.extra_h1.fc0.out_features == 16


# --------------------------------------------------------------------------------------------------------------------
# the Hub download helper
# --------------------------------------------------------------------------------------------------------------------
def test_download_from_hf_prefers_the_local_cache(monkeypatch):
    calls = []

    def fake_download(repo_id, filename, repo_type, local_files_only=False):
        calls.append((repo_id, filename, repo_type, local_files_only))
        return "/cache/file"

    monkeypatch.setattr(rr, "hf_hub_download", fake_download)
    assert rr._download_from_hf("org/repo", "f.pt", repo_type="model") == "/cache/file"
    assert calls == [("org/repo", "f.pt", "model", True)]


def test_download_from_hf_downloads_when_online(monkeypatch):
    def fake_download(repo_id, filename, repo_type, local_files_only=False):
        if local_files_only:
            raise OSError("not cached")
        return "/downloaded"

    monkeypatch.setattr(rr, "hf_hub_download", fake_download)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    assert rr._download_from_hf("org/repo", "f.pt") == "/downloaded"


def test_download_from_hf_scans_the_snapshots_when_offline(monkeypatch, tmp_path):
    import huggingface_hub.constants as constants

    def fake_download(**kwargs):
        raise OSError("offline")

    monkeypatch.setattr(rr, "hf_hub_download", fake_download)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        rr._download_from_hf("org/repo", "f.pt", repo_type="model")
    model_file = tmp_path / "models--org--repo" / "snapshots" / "abc" / "f.pt"
    model_file.parent.mkdir(parents=True)
    model_file.write_text("x")
    assert rr._download_from_hf("org/repo", "f.pt", repo_type="model") == str(model_file)
    (tmp_path / "datasets--org--data" / "snapshots" / "s1").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        rr._download_from_hf("org/data", "missing.pt")
    data_file = tmp_path / "datasets--org--data" / "snapshots" / "s1" / "g.parquet"
    data_file.write_text("x")
    assert rr._download_from_hf("org/data", "g.parquet") == str(data_file)
