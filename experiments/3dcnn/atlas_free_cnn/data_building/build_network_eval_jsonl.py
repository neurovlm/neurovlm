"""Build the held-out network-map JSONL used for text-to-brain generation eval.

This converts the same network resources used by brain-to-text semantic
evaluation into packed atlas-free CNN tensors plus a test-only JSONL.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import nibabel as nib
import pandas as pd

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[4]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from atlas_free_cnn.data_building.definitions import slugify
from atlas_free_cnn.data_building.export_hf_pack_jsonl import export_jsonl
from atlas_free_cnn.data_building.pack_atlas_free_cnn_dataset import PackConfig, pack_rows
from neurovlm.semantic_evaluation import load_network_maps


def _load_label_lookup(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not Path(path).exists():
        return {}
    labels = pd.read_csv(path)
    if "raw_network_label" not in labels.columns:
        return {}
    return {str(row["raw_network_label"]): row for row in labels.to_dict("records")}


def network_rows(
    *,
    networks_path: str | Path | None,
    labels_csv: str | Path | None,
    output_dir: str | Path,
) -> list[dict[str, Any]]:
    output_dir = Path(output_dir)
    nifti_dir = output_dir / "niftis"
    nifti_dir.mkdir(parents=True, exist_ok=True)
    labels = _load_label_lookup(labels_csv)
    rows = []
    for i, rec in enumerate(load_network_maps(networks_path)):
        raw_label = str(rec["network_label"])
        atlas = str(rec["atlas"])
        label_row = labels.get(raw_label, {})
        network_name = str(label_row.get("network_name") or raw_label)
        definition_parts = []
        for key in ["definition", "cognitive_terms", "region_terms"]:
            value = str(label_row.get(key) or "").strip()
            if value:
                definition_parts.append(value)
        definition = "; ".join(definition_parts) or network_name
        map_id = f"networks_{slugify(atlas)}_{slugify(raw_label)}_{i:04d}"
        nifti_path = nifti_dir / f"{map_id}.nii.gz"
        nib.save(rec["image"], str(nifti_path))
        text = f"{network_name} network [SEP] {definition}"
        rows.append(
            {
                "map_id": map_id,
                "source": "networks",
                "source_detail": f"networks:{atlas}",
                "map_type": "network_map",
                "split": "test",
                "nifti_path": str(nifti_path),
                "tensor_path": None,
                "positive_texts": [
                    {
                        "category": "network_label",
                        "source": "network_test_set",
                        "term": network_name,
                        "text": text,
                        "weight": 1.0,
                        "reliability": "strong",
                    }
                ],
                "positive_terms": [network_name],
                "positive_categories": ["network_label"],
                "atlas": atlas,
                "raw_network_label": raw_label,
                "network_key": str(label_row.get("network_key") or ""),
                "network_name": network_name,
                "negative_sampling_groups": {"source": "networks", "atlas": atlas},
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--networks-path", default=None, help="Optional local networks_arrays.pkl.gz. If omitted, use NeuroVLM resource loader.")
    parser.add_argument(
        "--labels-csv",
        default="experiments/evaluation_resources/networks_labels/network_test_set_labels.csv",
    )
    parser.add_argument("--output-dir", default="experiments/3dcnn/atlas_free_cnn/cache/network_eval")
    parser.add_argument("--cache-dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    rows = network_rows(networks_path=args.networks_path, labels_csv=args.labels_csv, output_dir=output_dir)
    pack_dir = output_dir / "hf_pack"
    pack_rows(rows, pack_dir, config=PackConfig(cache_dtype=args.cache_dtype, show_progress=True))
    counts = export_jsonl(pack_dir, output_dir)
    split_test = output_dir / "splits" / "test.jsonl"
    if split_test.exists():
        shutil.copyfile(split_test, output_dir / "network_test.jsonl")
    with (output_dir / "network_eval_build_summary.json").open("w") as f:
        json.dump({"n_input_rows": len(rows), "counts": counts}, f, indent=2)
    print(json.dumps({"n_input_rows": len(rows), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
