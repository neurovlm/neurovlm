import json

from atlas_free_multipositive.data_building.text_registry import attach_text_ids


def test_attach_text_ids_preserves_schema():
    rows = [
        {
            "map_id": "m",
            "positive_texts": [
                {
                    "text": "Hippocampus [SEP] A medial temporal lobe structure.",
                    "term": "Hippocampus",
                    "category": "anatomical_region",
                    "source": "mesh",
                    "weight": 1.0,
                    "reliability": "strong",
                }
            ],
        }
    ]
    out, registry = attach_text_ids(rows)

    assert out[0]["positive_texts"][0]["text_id"] in registry
    json.dumps(out[0])

