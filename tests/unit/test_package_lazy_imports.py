"""The package entry point resolves its public names lazily."""

import pytest


def test_lazy_public_attributes():
    import neurovlm
    from neurovlm.core import NeuroVLM

    assert neurovlm.NeuroVLM is NeuroVLM
    assert neurovlm.AtlasFreeCNNDataset.__name__ == "AtlasFreeCNNDataset"
    assert callable(neurovlm.atlas_free_cnn_splits)
    assert neurovlm.NeuroVLMRuntime.__name__ == "NeuroVLMRuntime"
    assert callable(neurovlm.load_pipeline) and neurovlm.RuntimeMetadata.__name__ == "RuntimeMetadata"
    with pytest.raises(AttributeError):
        neurovlm.nothing_here


def test_select_tqdm_returns_a_progress_class(monkeypatch):
    from neurovlm.utils.progress import select_tqdm

    bar = select_tqdm()
    assert list(bar(range(3), disable=True)) == [0, 1, 2]
