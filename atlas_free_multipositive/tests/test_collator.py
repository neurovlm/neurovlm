import torch

from atlas_free_multipositive.training.collators import MultiPositiveCollator


def test_collator_builds_flat_texts_and_masks():
    batch = [
        {
            "volume": torch.zeros(1, 3, 3, 3),
            "map_id": "a",
            "metadata": {"source": "pubmed"},
            "positive_texts": [
                {"text": "t1", "source": "mesh", "category": "anatomical_region", "weight": 1.0},
                {"text": "t2", "source": "wiki_style_summary", "category": "paper_summary", "weight": 0.7},
            ],
        },
        {
            "volume": torch.ones(1, 3, 3, 3),
            "map_id": "b",
            "metadata": {"source": "nilearn:yeo_2011"},
            "positive_texts": [
                {"text": "t3", "source": "nilearn_atlas_label", "category": "network", "weight": 1.0},
            ],
        },
    ]

    out = MultiPositiveCollator(positives_per_map=2, seed=1)(batch)

    assert out["volume"].shape == (2, 1, 3, 3, 3)
    assert out["pos_mask"].shape[0] == 2
    assert out["pos_mask"].shape[1] == len(out["texts"])
    assert out["pos_mask"].any(dim=1).all()
    assert out["pos_weights"][out["pos_mask"]].min() > 0

