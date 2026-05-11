import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from atlas_free_multipositive.training.model_wrappers import build_cnn_autoencoder
from atlas_free_multipositive.training.train_text_to_brain import train_from_config


def test_text_to_brain_training_loop_runs_on_four_examples(tmp_path: Path):
    rows = []
    text_cache = {}
    for i in range(4):
        arr = np.zeros((4, 4, 4), dtype=np.float32)
        arr[i % 4, i % 4, i % 4] = 1.0
        nii = tmp_path / f"map_{i}.nii.gz"
        nib.save(nib.Nifti1Image(arr, np.eye(4)), nii)
        text = f"term {i} [SEP] definition {i}"
        text_cache[text] = torch.randn(768)
        rows.append(
            {
                "map_id": f"m{i}",
                "source": "test",
                "map_type": "coordinate_ale",
                "nifti_path": str(nii),
                "tensor_path": None,
                "positive_texts": [{"text": text, "term": f"term {i}", "category": "test", "source": "test", "weight": 1.0}],
                "positive_terms": [f"term {i}"],
                "positive_categories": ["test"],
                "pmid": str(i),
            }
        )
    train_jsonl = tmp_path / "train.jsonl"
    val_jsonl = tmp_path / "val.jsonl"
    for path in (train_jsonl, val_jsonl):
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    cache_path = tmp_path / "text_cache.pt"
    torch.save(text_cache, cache_path)
    ae = build_cnn_autoencoder((4, 4, 4), base_channels=2, num_blocks=1)
    ae_ckpt = tmp_path / "ae.pt"
    torch.save({"model": ae.state_dict()}, ae_ckpt)

    result = train_from_config(
        {
            "train_jsonl": str(train_jsonl),
            "val_jsonl": str(val_jsonl),
            "text_embedding_cache": str(cache_path),
            "autoencoder_checkpoint": str(ae_ckpt),
            "checkpoint_dir": str(tmp_path / "ckpt"),
            "output_dir": str(tmp_path / "out"),
            "device": "cpu",
            "target_shape": [4, 4, 4],
            "batch_size": 2,
            "epochs": 1,
            "max_train_batches": 1,
            "max_val_batches": 1,
            "text_projection_init": "random",
            "model": {"latent_dim": 384, "base_channels": 2, "num_blocks": 1},
            "loss": {"lambda_recon": 1.0, "lambda_latent": 1.0, "lambda_dice": 0.1, "lambda_topk": 0.1, "lambda_corr": 0.1},
        }
    )

    assert result["history"]

