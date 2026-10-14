"""Text2Brain text -> map baseline used by run_text_to_brain.py (Ngo et al., MICCAI 2021; github.com/ngohgia/text2brain, MIT).

The released checkpoint, mirrored with the upstream source on neurovlm/text2brain-mirror (pinned in scripts/config/hf_manifest.json),
runs through the upstream model code in text2brain/ with two fixes for transformers 4.x: the encoder is called with
return_dict=False (text2brain/text2brain_model.py), and the checkpoint key encoder.embeddings.position_ids, a buffer that
transformers no longer saves (it equals arange(512)), is dropped before the strict load (below). As upstream predict.py,
each text runs alone (no padding) after removing "/", and the 40x48x40 output fills the 46x55x46 MNI152 4 mm grid,
the grid of the NeuroVLM mask.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

N_VOXELS = 28_542
MIRROR, WEIGHTS = "neurovlm/text2brain-mirror", "text2brain.safetensors"
SCIBERT = "allenai/scibert_scivocab_uncased"
AFFINE = np.array([[4.0, 0, 0, -90], [0, 4, 0, -126], [0, 0, 4, -72], [0, 0, 0, 1]])  # predict.py


class Text2BrainTextToMap:
    def __init__(self, masker: Any, device: torch.device = torch.device("cpu"), weights: Path | None = None) -> None:
        self.masker = masker
        self.device = device
        self.weights = weights
        self.model = None

    def build(self) -> bool:
        from safetensors.torch import load_file

        from _hf_pins import hf_file, hf_snapshot
        from text2brain.text2brain_model import Text2BrainModel

        model = Text2BrainModel(out_channels=1, fc_channels=64, decoder_filters=32, pretrained_bert_dir=str(hf_snapshot(SCIBERT)),
                                drop_p=0.55)  # predict.py settings
        sd = load_file(self.weights or hf_file(MIRROR, WEIGHTS))
        assert torch.equal(sd.pop("encoder.embeddings.position_ids"), torch.arange(512)[None])  # fix 2
        model.load_state_dict(sd)
        self.model = model.to(self.device).eval()
        return True

    def predict(self, texts: list[str]) -> np.ndarray:
        import nibabel as nib
        from nilearn.image import resample_to_img

        assert self.model is not None
        out = np.zeros((len(texts), N_VOXELS), dtype=np.float32)
        mask_img = self.masker.mask_img_
        vol = np.zeros((46, 55, 46))
        for i, text in enumerate(tqdm(texts, desc="text2brain queries", leave=False)):
            with torch.inference_mode():
                vol[3:-3, 3:-4, :-6] = self.model((text.replace("/", ""),)).cpu().numpy()[0, 0]
            img = nib.Nifti1Image(vol, AFFINE)
            if img.shape != mask_img.shape or not np.allclose(img.affine, mask_img.affine):
                img = resample_to_img(img, mask_img, interpolation="continuous", force_resample=True, copy_header=True)
            out[i] = self.masker.transform(img).ravel().astype(np.float32)
        return out
