"""PyTorch Dataset for unified map-text JSONL rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class UnifiedMapTextDataset(Dataset):
    """Read unified JSONL rows and return brain volume plus all positives."""

    def __init__(self, jsonl_path: str | Path, *, load_volumes: bool = True):
        self.path = Path(jsonl_path)
        with self.path.open() as f:
            self.rows = [json.loads(line) for line in f if line.strip()]
        self.load_volumes = load_volumes
        self._tensor_cache: dict[str, Any] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _load_from_tensor_cache(self, row: dict) -> torch.Tensor:
        tensor_path = row.get("tensor_path")
        if tensor_path not in self._tensor_cache:
            self._tensor_cache[tensor_path] = torch.load(tensor_path, map_location="cpu", weights_only=False)
        payload = self._tensor_cache[tensor_path]
        if isinstance(payload, dict) and "volumes" in payload:
            idx = int(row["tensor_index"])
            tensor = payload["volumes"][idx].float()
        else:
            tensor = payload[int(row.get("tensor_index", 0))].float()
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _load_nifti(self, row: dict) -> torch.Tensor:
        import nibabel as nib
        import numpy as np

        data = np.nan_to_num(nib.load(row["nifti_path"]).get_fdata().astype("float32"))
        return torch.from_numpy(data).unsqueeze(0)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        volume = None
        if self.load_volumes:
            if row.get("tensor_path"):
                volume = self._load_from_tensor_cache(row)
            elif row.get("nifti_path"):
                volume = self._load_nifti(row)
            else:
                raise ValueError(f"Row {row.get('map_id')} has no tensor_path or nifti_path")
        return {
            "volume": volume,
            "map_id": row["map_id"],
            "positive_texts": row.get("positive_texts", []),
            "metadata": row,
        }


def load_text_registry(path: str | Path) -> dict[str, dict]:
    with Path(path).open() as f:
        return {row["text_id"]: row for row in (json.loads(line) for line in f if line.strip())}

