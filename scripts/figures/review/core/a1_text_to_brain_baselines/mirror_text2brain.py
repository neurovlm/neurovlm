#!/usr/bin/env python
"""Build the files of the Hugging Face mirror neurovlm/text2brain-mirror from the original Text2Brain release (run once).

Downloads the released checkpoint (the Google Drive link in the upstream README) and checks its sha256, saves its
state_dict as safetensors (all 230 tensors unchanged, without the optimizer state and the pickle), and archives the
upstream source at the pinned commit. --upload creates the repo and pushes the files with a model card. The mirror is
pinned in scripts/hf_manifest.json; run_text_to_brain.py reads text2brain.safetensors from there.

    .venv/bin/python scripts/figures/review/core/a1_text_to_brain_baselines/mirror_text2brain.py --out DIR [--upload]
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

DRIVE_URL = "https://drive.usercontent.google.com/download?id=1mBO1frmlSMgXGN-F1WKQKlCO0gIk_MHq&export=download&confirm=t"
CKPT, CKPT_SHA256 = "text2brain_checkpoint.tar.gz", "08e8af9af54b5e38e743be56b47c0c987197bc60dc91e3e56aa7feaf406ec6c8"
GIT_URL, COMMIT = "https://github.com/ngohgia/text2brain", "6f29389658e23b89e69248f3d3421dd2bd79b269"
SOURCE = f"text2brain-{COMMIT[:7]}.tar.gz"
REPO = "neurovlm/text2brain-mirror"

CARD = """---
license: mit
library_name: pytorch
tags: [neuroimaging, text-to-brain, scibert]
---
# Text2Brain (mirror)

Unmodified mirror of the released Text2Brain model: Ngo, Nguyen, Chen and Sabuncu, *Text2Brain: Synthesis of Brain
Activation Maps from Free-form Text Query*, MICCAI 2021 ([arXiv:2109.13814](https://arxiv.org/abs/2109.13814)).
Code: {git_url} (MIT, Copyright (c) 2021 ngo.h.gia, see LICENSE). We are not the authors. It is mirrored so the NeuroVLM
reviewer analyses can pin it: the checkpoint link of the paper's repository is dead, and the current link is a Google Drive file.

| file | origin | sha256 |
|---|---|---|
| `{ckpt}` | {drive}, downloaded 2026-10-13, byte-identical | `{ckpt_sha}` |
| `text2brain.safetensors` | `state_dict` of `text2brain_checkpoint.pth` in the tar (epoch {epoch}); every tensor unchanged; optimizer and scheduler state dropped | `{st_sha}` |
| `{source}` | `git archive` of {git_url} at commit `{commit}` | `{src_sha}` |

The model: fine-tuned SciBERT (`allenai/scibert_scivocab_uncased` vocabulary and config) -> Linear(768, 64x4x5x4) -> 3D
transposed-convolution decoder (32 filters), output 40x48x40 placed into the 46x55x46 MNI152 4 mm grid as upstream `predict.py`.

## Running it with current libraries

The upstream code (Python 3.6, torch 1.5.1, transformers 3.3.1) needs two fixes under transformers 4.x:

1. `models/text2brain_model.py`: call the encoder with `return_dict=False`. Otherwise `_, embedding = self.encoder(...)`
   unpacks the keys of the returned `ModelOutput` (`"last_hidden_state"`, `"pooler_output"`) and dropout fails on a string.
2. Drop the checkpoint key `encoder.embeddings.position_ids` (it equals `arange(512)`; transformers no longer saves this
   buffer) before the strict `load_state_dict`. Otherwise the load fails with "Unexpected key(s)" (upstream issue #6).

The upstream code also moves the encoder and the inputs to CUDA whenever a GPU is visible; the NeuroVLM copy keeps them on
the model's device. The patched code is in the NeuroVLM repository under
`scripts/figures/review/core/a1_text_to_brain_baselines/text2brain/`.
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True, help="Work dir for the downloads and the mirror files.")
    p.add_argument("--upload", action="store_true", help=f"Create {REPO} and upload the files.")
    args = p.parse_args()
    import torch
    from safetensors.torch import save_file

    out = args.out
    mirror = out / "mirror"
    mirror.mkdir(parents=True, exist_ok=True)
    tgz = out / CKPT
    if not tgz.exists():
        with urllib.request.urlopen(DRIVE_URL) as r, open(tgz, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 24)
    assert sha256(tgz) == CKPT_SHA256, f"{tgz} is not the released checkpoint"
    with tarfile.open(tgz) as t:
        t.extract("text2brain_checkpoint.pth", out, filter="data")
    ck = torch.load(out / "text2brain_checkpoint.pth", map_location="cpu", weights_only=True)
    save_file({k: v.contiguous() for k, v in ck["state_dict"].items()}, mirror / "text2brain.safetensors",
              metadata={"source": f"{CKPT} sha256 {CKPT_SHA256}, epoch {ck['epoch']}"})  # one key: a fixed header
    shutil.copyfile(tgz, mirror / CKPT)

    src = out / "text2brain"
    if not src.exists():
        subprocess.run(["git", "clone", "-q", GIT_URL, str(src)], check=True)
    subprocess.run(["git", "-C", str(src), "archive", "--format=tar.gz", f"--prefix=text2brain-{COMMIT[:7]}/",
                    "-o", str(mirror / SOURCE), COMMIT], check=True)
    subprocess.run(["git", "-C", str(src), "show", f"{COMMIT}:LICENSE"], check=True, stdout=open(mirror / "LICENSE", "wb"))
    sums = {f.name: sha256(f) for f in sorted(mirror.iterdir()) if f.name != "README.md"}
    (mirror / "README.md").write_text(CARD.format(git_url=GIT_URL, commit=COMMIT, drive=DRIVE_URL.split("&confirm")[0],
                                                  ckpt=CKPT, ckpt_sha=CKPT_SHA256, epoch=ck["epoch"], source=SOURCE,
                                                  st_sha=sums["text2brain.safetensors"], src_sha=sums[SOURCE]))
    for name, digest in sums.items():
        print(f"{digest}  {name}")
    if args.upload:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(REPO, repo_type="model", exist_ok=True)
        info = api.upload_folder(repo_id=REPO, repo_type="model", folder_path=str(mirror),
                                 commit_message=f"Mirror of the Text2Brain release ({GIT_URL} @ {COMMIT[:7]})")
        print(f"uploaded {info.commit_url}")


if __name__ == "__main__":
    main()
