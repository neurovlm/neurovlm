"""Compatibility checks for documented pre-refactor import paths."""

from neurovlm import AtlasFreeCNNDataset, NeuroVLM, NeuroVLMRuntime
from neurovlm.ale_cnn import ALE3DCNNAutoEncoder as LegacyCNN
from neurovlm.atlas_free_dataset import AtlasFreeCNNDataset as LegacyDataset
from neurovlm.cnn.architectures import ALE3DCNNAutoEncoder
from neurovlm.core import NeuroVLM as CoreNeuroVLM
from neurovlm.core.runtime import NeuroVLMRuntime as CoreRuntime
from neurovlm.io import load_model as legacy_load_weights
from neurovlm.loss import InfoNCELoss as LegacyInfoNCELoss
from neurovlm.model_registry import ModelFamily as LegacyModelFamily
from neurovlm.models.losses import InfoNCELoss
from neurovlm.models.registry import ModelFamily
from neurovlm.models.serialization import load_model as load_weights


def test_package_level_api_uses_reorganized_implementations() -> None:
    assert NeuroVLM is CoreNeuroVLM
    assert NeuroVLMRuntime is CoreRuntime
    assert AtlasFreeCNNDataset is LegacyDataset


def test_compatibility_modules_reexport_identical_objects() -> None:
    assert LegacyCNN is ALE3DCNNAutoEncoder
    assert LegacyInfoNCELoss is InfoNCELoss
    assert LegacyModelFamily is ModelFamily
    assert legacy_load_weights is load_weights
