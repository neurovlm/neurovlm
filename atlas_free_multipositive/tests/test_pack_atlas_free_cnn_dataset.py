import json

import pandas as pd
import torch

from atlas_free_multipositive.data_building.pack_atlas_free_cnn_dataset import (
    PackConfig,
    load_neurovault_rows,
    pack_rows,
    text_pairs_for_row,
)


def test_text_pairs_choose_metadata_primary_before_paper_title():
    row = {
        "map_id": "m1",
        "split": "test",
        "positive_texts": [
            {"text": "Paper title", "category": "paper_title", "source": "pubmed_title", "weight": 1.0},
            {"text": "Task label", "category": "cognitive_task_or_contrast", "source": "neurovault_task_label", "weight": 0.5},
        ],
    }

    pairs = text_pairs_for_row(row, PackConfig())

    assert pairs[0]["text"] == "Task label"
    assert pairs[0]["is_primary_text"] is True
    assert pairs[1]["is_primary_text"] is False


def test_neurovault_primary_prefers_image_description_before_task_label():
    row = {
        "map_id": "neurovault_1",
        "split": "train",
        "positive_texts": [
            {
                "text": "working memory task",
                "category": "cognitive_task_or_contrast",
                "source": "neurovault_task_label",
                "weight": 0.9,
            },
            {
                "text": "N-back contrast [SEP] Activation for two-back greater than zero-back.",
                "category": "image_description",
                "source": "neurovault_image",
                "weight": 0.85,
            },
        ],
    }

    pairs = text_pairs_for_row(row, PackConfig())

    assert pairs[0]["source"] == "neurovault_image"
    assert pairs[0]["is_primary_text"] is True


def test_pack_rows_writes_volumes_rows_and_text_pairs(tmp_path):
    source_pt = tmp_path / "source.pt"
    torch.save({"volumes": torch.ones(2, 1, 3, 4, 5), "map_ids": ["m1", "m2"]}, source_pt)
    rows = [
        {
            "map_id": "m1",
            "split": "train",
            "source": "unit",
            "map_type": "test_map",
            "tensor_path": str(source_pt),
            "tensor_index": 0,
            "positive_texts": [{"text": "A [SEP] B", "category": "image_description", "source": "unit", "weight": 1.0}],
        },
        {
            "map_id": "m2",
            "split": "test",
            "source": "unit",
            "map_type": "test_map",
            "tensor_path": str(source_pt),
            "tensor_index": 1,
            "positive_texts": [{"text": "C [SEP] D", "category": "paper_title", "source": "unit", "weight": 0.5}],
        },
    ]

    outputs = pack_rows(rows, tmp_path / "packed", config=PackConfig(cache_dtype="float32"))
    payload = torch.load(outputs["volumes"], map_location="cpu", weights_only=False)
    map_rows = pd.read_parquet(outputs["rows"])
    text_pairs = pd.read_parquet(outputs["text_pairs"])
    manifest = json.loads(outputs["manifest"].read_text())

    assert tuple(payload["volumes"].shape) == (2, 1, 3, 4, 5)
    assert map_rows["map_id"].tolist() == ["m1", "m2"]
    assert text_pairs["is_primary_text"].tolist() == [True, True]
    assert manifest["counts"]["splits"] == {"train": 1, "test": 1}


def test_pack_rows_can_write_primary_text_only(tmp_path):
    source_pt = tmp_path / "source.pt"
    torch.save({"volumes": torch.ones(1, 1, 3, 4, 5), "map_ids": ["m1"]}, source_pt)
    rows = [
        {
            "map_id": "m1",
            "split": "train",
            "source": "unit",
            "map_type": "test_map",
            "tensor_path": str(source_pt),
            "tensor_index": 0,
            "positive_texts": [
                {"text": "Primary", "category": "paper_summary", "source": "wiki_style_summary", "weight": 1.0},
                {"text": "Secondary", "category": "paper_title", "source": "pubmed_title", "weight": 1.0},
            ],
        }
    ]

    outputs = pack_rows(rows, tmp_path / "packed_primary", config=PackConfig(cache_dtype="float32", primary_text_only=True))
    text_pairs = pd.read_parquet(outputs["text_pairs"])
    map_rows = pd.read_parquet(outputs["rows"])

    assert text_pairs["text"].tolist() == ["Primary"]
    assert text_pairs["is_primary_text"].tolist() == [True]
    assert map_rows["n_text_pairs"].tolist() == [1]


def test_load_neurovault_rows_from_staged_outputs(tmp_path):
    nv = tmp_path / "neurovault"
    nv.mkdir()
    pd.DataFrame(
        [
            {
                "map_id": "neurovault_1",
                "quality_tier": "strong",
                "tensor_index": 0,
                "quality_score": 7,
                "pmid": "123",
                "doi": "",
                "collection_id": 5,
                "missing_metadata": False,
                "no_task_label": False,
                "no_doi_or_pmid": False,
                "weird_shape": False,
                "mostly_empty": False,
                "negative_values_present": True,
                "thresholded_map_possible": False,
                "failed_resample": False,
                "low_quality_text": False,
            }
        ]
    ).to_csv(nv / "neurovault_manifest.csv", index=False)
    (nv / "neurovault_text_positives.jsonl").write_text(
        json.dumps(
            {
                "map_id": "neurovault_1",
                "text": "Task",
                "term": "Task",
                "category": "cognitive_task_or_contrast",
                "source": "neurovault_task_label",
                "weight": 0.9,
            }
        )
        + "\n"
    )

    rows = load_neurovault_rows(nv, split="train")

    assert len(rows) == 1
    assert rows[0]["tensor_index"] == 0
    assert rows[0]["positive_texts"][0]["text"] == "Task"


def test_load_neurovault_rows_caps_each_collection_and_prefers_strong(tmp_path):
    nv = tmp_path / "neurovault"
    nv.mkdir()
    records = []
    positives = []
    for i in range(5):
        map_id = f"neurovault_{i}"
        records.append(
            {
                "map_id": map_id,
                "quality_tier": "strong" if i != 4 else "weak",
                "tensor_index": i,
                "quality_score": 7 if i != 4 else 3,
                "pmid": "",
                "doi": "",
                "collection_id": 9,
                "collection_name": "Dominant collection",
                "missing_metadata": False,
                "no_task_label": False,
                "no_doi_or_pmid": False,
                "weird_shape": False,
                "mostly_empty": False,
                "negative_values_present": True,
                "thresholded_map_possible": False,
                "failed_resample": False,
                "low_quality_text": False,
            }
        )
        positives.append(
            {
                "map_id": map_id,
                "text": f"Task {i}",
                "term": f"Task {i}",
                "category": "cognitive_task_or_contrast",
                "source": "neurovault_task_label",
                "weight": 0.9,
            }
        )
    pd.DataFrame(records).to_csv(nv / "neurovault_manifest.csv", index=False)
    (nv / "neurovault_text_positives.jsonl").write_text("\n".join(json.dumps(p) for p in positives) + "\n")

    rows = load_neurovault_rows(nv, include_weak=True, split="train", max_per_collection=2)

    assert len(rows) == 2
    assert all(row["quality_tier"] == "strong" for row in rows)
    assert all(row["selection"]["collection_cap"] == 2 for row in rows)
