import torch

from atlas_free_multipositive.evaluation.generation_metrics import generation_metrics
from atlas_free_multipositive.training.generation_baselines import (
    global_mean_map,
    nearest_neighbor_text_maps,
    random_training_maps,
)


def test_generation_baselines_run_on_four_examples():
    volumes = torch.rand(4, 1, 3, 3, 3)
    text = torch.eye(4)

    mean = global_mean_map(volumes)
    rand = random_training_maps(volumes, 4, seed=1)
    nn_maps, idx = nearest_neighbor_text_maps(text, text, volumes)
    metrics = generation_metrics(nn_maps, volumes)

    assert mean.shape == (1, 3, 3, 3)
    assert rand.shape == volumes.shape
    assert nn_maps.shape == volumes.shape
    assert idx.tolist() == [0, 1, 2, 3]
    assert "top5_dice" in metrics

