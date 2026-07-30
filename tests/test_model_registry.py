"""No-network tests for canonical packaged-model resolution."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import neurovlm.models as models
import neurovlm.retrieval_resources as resources
from neurovlm.model_registry import (
    MODEL_ALIASES,
    MODEL_REGISTRY,
    ModelDomain,
    ModelFamily,
    ModelLoader,
    ModelTask,
    ModelVariant,
    resolve_model_spec,
)


def test_registry_has_unique_immutable_canonical_names_and_aliases() -> None:
    assert len(MODEL_REGISTRY) == len(set(MODEL_REGISTRY))
    assert len(MODEL_ALIASES) == len(set(MODEL_ALIASES))
    with pytest.raises(TypeError):
        MODEL_REGISTRY["new"] = MODEL_REGISTRY["mlp:autoencoder:default"]
    with pytest.raises(TypeError):
        MODEL_ALIASES["new"] = MODEL_REGISTRY["mlp:autoencoder:default"]


def test_mlp_is_the_structured_global_default() -> None:
    spec = resolve_model_spec(task="autoencoder")
    assert spec.family is ModelFamily.MLP
    assert spec.task is ModelTask.AUTOENCODER
    assert spec.variant is ModelVariant.DEFAULT
    assert spec.loader is ModelLoader.MLP_AUTOENCODER


@pytest.mark.parametrize("task", ["contrastive", "text_to_brain"])
@pytest.mark.parametrize("domain", list(ModelDomain))
def test_cnn_domain_tasks_default_to_mixed_baseline(task, domain) -> None:
    spec = resolve_model_spec(family="cnn", task=task, domain=domain)
    assert spec.family is ModelFamily.CNN
    assert spec.domain is domain
    assert spec.variant is ModelVariant.MIXED_BASELINE
    assert spec.loader_variant == f"mixed_to_{domain.value}"


@pytest.mark.parametrize("task", ["contrastive", "text_to_brain"])
@pytest.mark.parametrize("domain", list(ModelDomain))
def test_cnn_domain_tasks_require_explicit_finetuned_variant(task, domain) -> None:
    spec = resolve_model_spec(
        family=ModelFamily.CNN,
        task=task,
        domain=domain.value,
        variant=ModelVariant.FINETUNED,
    )
    assert spec.variant is ModelVariant.FINETUNED
    assert spec.loader_variant == domain.value


def test_cnn_autoencoder_defaults_to_domain_independent_mixed_model() -> None:
    spec = resolve_model_spec(family="cnn", task="autoencoder")
    assert spec.canonical_name == "cnn:autoencoder:mixed_baseline"
    assert spec.domain is None
    assert spec.loader_variant == "mixed"


@pytest.mark.parametrize("domain", list(ModelDomain))
def test_cnn_autoencoder_finetuning_is_explicit(domain) -> None:
    spec = resolve_model_spec(
        family="cnn",
        task="autoencoder",
        domain=domain,
        variant="finetuned",
    )
    assert spec.domain is domain
    assert spec.loader_variant == domain.value


def test_separate_brain_to_text_tasks_are_canonical_registry_entries() -> None:
    retrieval = resolve_model_spec(task="brain_to_text_retrieval")
    generation = resolve_model_spec(task="brain_to_text_generation")
    pubmed_generation = resolve_model_spec("neuro_qformer_pubmed")
    assert retrieval.task is ModelTask.BRAIN_TO_TEXT_RETRIEVAL
    assert retrieval.loader is ModelLoader.MLP_IMAGE_INFONCE
    assert generation.task is ModelTask.BRAIN_TO_TEXT_GENERATION
    assert generation.loader is ModelLoader.MLP_NEURO_QFORMER
    assert pubmed_generation.task is ModelTask.BRAIN_TO_TEXT_GENERATION
    assert pubmed_generation.variant is ModelVariant.PUBMED
    assert pubmed_generation.loader is ModelLoader.MLP_NEURO_QFORMER
    assert pubmed_generation.loader_variant == "pubmed"


@pytest.mark.parametrize("alias", sorted(MODEL_ALIASES))
def test_every_legacy_alias_resolves_to_its_registered_spec(alias) -> None:
    assert resolve_model_spec(alias) is MODEL_ALIASES[alias]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "task is required"),
        ({"family": "cnn", "task": "contrastive"}, "domain is required"),
        ({"family": "cnn", "task": "text_to_brain"}, "domain is required"),
        (
            {"family": "cnn", "task": "autoencoder", "domain": "pubmed"},
            "omit domain",
        ),
        (
            {"family": "cnn", "task": "autoencoder", "variant": "finetuned"},
            "domain is required",
        ),
        ({"task": "autoencoder", "domain": "pubmed"}, "do not accept a domain"),
        ({"family": "cnn", "task": "brain_to_text_generation"}, "No released model"),
        ({"family": "transformer", "task": "autoencoder"}, "Unknown model family"),
        ({"task": "classification"}, "Unknown model task"),
        ({"task": "autoencoder", "variant": "experimental"}, "Unknown model variant"),
    ],
)
def test_invalid_structured_combinations_fail_before_loading(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_model_spec(**kwargs)


def test_legacy_name_cannot_be_mixed_with_structured_fields() -> None:
    with pytest.raises(ValueError, match="either a legacy model name or structured"):
        resolve_model_spec("autoencoder", task="autoencoder")


def test_structured_load_model_dispatches_exact_cnn_resource_variants(monkeypatch) -> None:
    calls = []
    sentinels = [object(), object(), object()]
    monkeypatch.setattr(
        resources,
        "_load_cnn_autoencoder",
        lambda variant: calls.append(("ae", variant)) or sentinels[0],
    )
    monkeypatch.setattr(
        resources,
        "_load_cnn_contrastive",
        lambda variant: calls.append(("contrastive", variant)) or sentinels[1],
    )
    monkeypatch.setattr(
        resources,
        "_load_cnn_text_to_brain",
        lambda variant: calls.append(("t2b", variant)) or sentinels[2],
    )

    assert models.load_model(family="cnn", task="autoencoder") is sentinels[0]
    assert (
        models.load_model(family="cnn", task="contrastive", domain="pubmed")
        is sentinels[1]
    )
    assert (
        models.load_model(
            family="cnn",
            task="text_to_brain",
            domain="nilearn",
            variant="finetuned",
        )
        is sentinels[2]
    )
    assert calls == [
        ("ae", "mixed"),
        ("contrastive", "mixed_to_pubmed"),
        ("t2b", "nilearn"),
    ]


def test_legacy_mlp_names_keep_exact_loader_dispatch(monkeypatch) -> None:
    calls = []
    sentinels = {name: object() for name in ("text_infonce", "image_infonce", "text_mse")}

    monkeypatch.setattr(
        models.ProjHead,
        "from_pretrained",
        staticmethod(lambda name: calls.append(name) or sentinels[name]),
    )
    autoencoder = object()
    monkeypatch.setattr(
        models.NeuroAutoEncoder,
        "from_pretrained",
        staticmethod(lambda: calls.append("autoencoder") or autoencoder),
    )

    assert models.load_model("proj_head_text_infonce") is sentinels["text_infonce"]
    assert models.load_model("proj_head_image_infonce") is sentinels["image_infonce"]
    assert models.load_model("proj_head_text_mse") is sentinels["text_mse"]
    assert models.load_model("autoencoder") is autoencoder
    assert calls == ["text_infonce", "image_infonce", "text_mse", "autoencoder"]


def test_structured_special_mlp_tasks_dispatch_existing_resources(monkeypatch) -> None:
    qformer = object()
    pubmed_qformer = object()
    adapter = object()

    def load_qformer(*, qformer_variant=None):
        if qformer_variant == "pubmed":
            return pubmed_qformer
        return qformer

    monkeypatch.setattr(resources, "_load_neuro_qformer", load_qformer)
    monkeypatch.setattr(resources, "_load_neuro_adapter", lambda: adapter)

    assert models.load_model(task="brain_to_text_generation") is qformer
    assert models.load_model("neuro_qformer_pubmed") is pubmed_qformer
    assert (
        models.load_model(task="brain_to_text_generation", variant="pubmed")
        is pubmed_qformer
    )
    assert models.load_model(task="text_to_brain", variant="adapter") is adapter


def test_cnn_resource_loaders_return_distinct_mutable_modules(monkeypatch) -> None:
    payloads = []

    def fake_from_payload(payload):
        payloads.append(payload)
        return nn.Linear(2, 2)

    monkeypatch.setattr(resources, "_load_trusted_cnn_checkpoint", lambda _: {"weights": 1})
    monkeypatch.setattr("neurovlm.cnn.autoencoder_from_payload", fake_from_payload)

    first = resources._load_cnn_autoencoder("mixed")
    second = resources._load_cnn_autoencoder("mixed")
    assert first is not second
    assert len(payloads) == 2


def test_mlp_architecture_public_invariants_remain_unchanged() -> None:
    autoencoder = models.NeuroAutoEncoder(
        seed=3,
        dim_neuro=12,
        dim_h0=8,
        dim_h1=6,
        dim_latent=4,
        normalize_latent=True,
    )
    inputs = torch.rand(5, 12)
    latent = autoencoder.encoder(inputs)
    reconstruction = autoencoder(inputs)
    assert latent.shape == (5, 4)
    assert torch.allclose(latent.norm(dim=1), torch.ones(5), atol=1e-6)
    assert reconstruction.shape == inputs.shape
    assert torch.all((0 <= reconstruction) & (reconstruction <= 1))

    projection = models.ProjHead(latent_in_dim=7, hidden_dim=5, latent_out_dim=3, seed=3)
    assert projection(torch.randn(2, 7)).shape == (2, 3)
