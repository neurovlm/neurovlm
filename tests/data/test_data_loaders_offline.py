"""Offline tests of ``neurovlm.data.loaders``: the Hub snapshot download and every resource loader are replaced."""

from __future__ import annotations

import pytest
import torch

import neurovlm.data.loaders as dl

DATASET_LOADERS = {
    "pubmed_text": "_load_pubmed_dataframe",
    "pubmed_summaries": "_load_pubmed_summaries_dataframe",
    "pubmed_coordinates": "_load_pubmed_coordinates",
    "pubmed_images": "_load_pubmed_images",
    "wiki": "_load_neuro_wiki",
    "neurowiki_graph": "_load_neuro_wiki_graph",
    "cogatlas": "_load_cogatlas_dataset",
    "cogatlas_task": "_load_cogatlas_task_dataset",
    "cogatlas_disorder": "_load_cogatlas_disorder_dataset",
    "cogatlas_graph": "_load_cogatlas_graph_dataset",
    "networks": "_load_networks",
    "networks_canonical": "_load_networks_canonical",
    "network_test_set_labels": "_load_network_test_set_labels",
    "pubmed_mesh_annotations": "_load_pubmed_mesh_annotations",
    "neurovault_text": "_load_publications_neurovault_dataframe",
    "neurovault_images_meta": "_load_images_neurovault_dataframe",
    "neurovault_images": "_load_neurovault_images",
    "pubmed_mesh": "_load_kg_mesh_dataset",
    "pubmed_mesh_nodes": "_load_mesh_kg_nodes",
    "pubmed_mesh_descriptors": "_load_mesh_kg_descriptors",
    "pubmed_mesh_brain_rankable": "_load_kg_mesh_brain_rankable_dataset",
    "pubmed_mesh_brain_rankable_plus_molecular": "_load_kg_mesh_brain_rankable_dataset",
    "llm_neuro_terms": "_load_llm_neuro_terms_dataset",
    "ngrams": "_load_ngram",
}

LATENT_LOADERS = {
    "pubmed_text": "_load_latent_text",
    "pubmed_summaries": "_load_latent_neuro_summaries",
    "pubmed_images": "_load_latent_neuro",
    "wiki": "_load_latent_wiki",
    "cogatlas": "_load_latent_cogatlas",
    "cogatlas_task": "_load_latent_cogatlas_task",
    "cogatlas_disorder": "_load_latent_cogatlas_disorder",
    "networks_text": "_load_latent_networks_canonical_text",
    "networks_neuro": "_load_latent_networks_neuro",
    "neurovault_images": "_load_latent_neurovault_images",
    "neurovault_text": "_load_latent_neurovault_text",
    "ngrams": "_load_latent_ngram",
    "pubmed_mesh": "_load_latent_kg_mesh",
    "pubmed_mesh_brain_rankable": "_load_latent_kg_mesh_brain_rankable",
    "pubmed_mesh_brain_rankable_plus_molecular": "_load_latent_kg_mesh_brain_rankable",
    "llm_neuro_terms": "_load_latent_llm_neuro_terms",
}


def test_every_public_name_has_a_loader():
    assert set(DATASET_LOADERS) == set(dl.DATASET_NAMES)
    assert set(LATENT_LOADERS) == set(dl.LATENT_NAMES)


def test_load_dataset_dispatches_to_the_resource_loaders(monkeypatch):
    for name, loader in DATASET_LOADERS.items():
        calls = []
        monkeypatch.setattr(dl, loader, lambda *args, _calls=calls, **kwargs: _calls.append(kwargs) or object())
        if name == "ngrams":
            continue
        result = dl.load_dataset(name)
        assert calls, name
        if name.startswith("pubmed_mesh_brain_rankable"):
            assert calls[0] == {"include_molecular": name.endswith("molecular")}
        assert result is not None
    monkeypatch.setattr(dl, "_load_ngram", lambda: ["alpha", "beta"])
    assert dl.load_dataset("ngrams")["term"].tolist() == ["alpha", "beta"]


def test_load_latent_dispatches_and_drops_gradients(monkeypatch):
    for name, loader in LATENT_LOADERS.items():
        tensor = torch.ones(2, 3, requires_grad=True)
        payload = (tensor, ["a", "b"]) if "mesh" in name else {"latent": tensor}
        calls = []
        monkeypatch.setattr(dl, loader, lambda *args, _calls=calls, _payload=payload, **kwargs:
                            _calls.append(kwargs) or _payload)
        result = dl.load_latent(name)
        assert calls, name
        latent = result[0] if isinstance(result, tuple) else result["latent"]
        assert not latent.requires_grad
        if name.startswith("pubmed_mesh_brain_rankable"):
            assert calls[0] == {"include_molecular": name.endswith("molecular")}


def test_load_masker_aliases_the_resource_loader(monkeypatch):
    monkeypatch.setattr(dl, "_load_masker", lambda: "masker")
    assert dl.load_masker() == "masker"


def test_fetch_data_downloads_every_repository(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(dl, "snapshot_download", lambda **kwargs: calls.append(kwargs))
    assert dl.fetch_data(cache_dir="/tmp/neurovlm-cache") == "/tmp/neurovlm-cache"
    repo_ids = [call["repo_id"] for call in calls]
    assert repo_ids == (list(dl.REPO_DATASETS.values()) + list(dl.REPO_MODELS.values())
                        + [dl.SPECTER_BASE_REPO, dl.SPECTER_ADAPTER_REPO])
    assert all(call["cache_dir"] == "/tmp/neurovlm-cache" for call in calls)
    assert calls[0]["repo_type"] == "dataset" and calls[0]["ignore_patterns"]
    assert calls[-1]["repo_type"] == "model"
    assert "Data fetch complete" in capsys.readouterr().out


def test_fetch_data_skips_unknown_keys_and_reports_failures(monkeypatch, capsys):
    def failing(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(dl, "snapshot_download", failing)
    assert dl.fetch_data(datasets=["nope"], models=["ProjectionHeads", "nah"]) == dl.HUGGINGFACE_HUB_CACHE
    out = capsys.readouterr().out
    assert "Unknown dataset key 'nope'" in out and "Unknown model key 'nah'" in out and "boom" in out


def test_preload_all_data_calls_every_loader(monkeypatch, capsys):
    loaders = [
        "_load_pubmed_dataframe", "_load_neuro_wiki", "_load_neuro_wiki_graph", "_load_cogatlas_dataset",
        "_load_cogatlas_task_dataset", "_load_cogatlas_disorder_dataset", "_load_cogatlas_graph_dataset",
        "_load_latent_text", "_load_latent_wiki", "_load_latent_cogatlas", "_load_latent_cogatlas_disorder",
        "_load_latent_cogatlas_task", "_load_autoencoder", "_load_masker", "_load_networks",
        "_proj_head_image_infonce", "_proj_head_text_mse", "_proj_head_text_infonce", "_load_neuro_qformer",
        "_load_neuro_adapter", "_load_specter",
    ]
    called = []
    for name in loaders:
        monkeypatch.setattr(dl, name, lambda *args, _name=name, **kwargs: called.append(_name))

    def broken():
        raise RuntimeError("wiki down")

    monkeypatch.setattr(dl, "_load_neuro_wiki", broken)
    dl.preload_all_data()
    out = capsys.readouterr().out
    assert "Error: wiki down" in out and out.count("✓") == len(loaders) - 1
    assert set(called) == set(loaders) - {"_load_neuro_wiki"}
    dl.preload_all_data(verbose=False)
    assert capsys.readouterr().out == ""


def test_get_data_dir_is_the_user_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(dl.Path, "home", classmethod(lambda cls: tmp_path))
    assert dl.get_data_dir() == tmp_path / ".cache" / "neurovlm"
    assert dl.get_data_dir().is_dir()
