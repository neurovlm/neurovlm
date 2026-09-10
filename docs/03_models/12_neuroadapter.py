# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Train NeuroAdapter on anatomical synthetic maps
#
# This is the reproducible producer of `data_dir / "adapter_anat_full.pt"`, recovered from
# the archived anatomy_v8 notebook.
#
# **Direct inputs from `11_anatomical_atlases.ipynb`:**
# `data_dir / "images_synth.pt"` and `data_dir / "synth.parquet"`.
#
# **Base models:** `autoencoder` and `proj_head_text_mse` loaded through
# `neurovlm.models.load_model`. The notebook records hashes for both inputs and
# base-model states in `data_dir / "adapter_anat_full.lineage.json"` so drift is visible.
#
# **Primary output:** `data_dir / "adapter_anat_full.pt"`, subsequently packaged as
# `neurovlm/NeuroAdapter` and loaded by the canonical Q-Former notebooks.
#

# %%
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from neurovlm.models.adapter import InterleavedDecoderAdapter
from neurovlm.data import data_dir
from neurovlm.models import load_model


def find_project_root(start: Path = Path.cwd()) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "neurovlm").is_dir():
            return candidate
    raise RuntimeError("Run this notebook from within the neurovlm repository.")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_state_dict(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


PROJECT_ROOT = find_project_root()
MODEL_DIR = PROJECT_ROOT / "docs" / "03_models"
MODEL_DATA_DIR = data_dir
MODEL_DATA_DIR.mkdir(parents=True, exist_ok=True)

IMAGES_SYNTH_PATH = MODEL_DATA_DIR / "images_synth.pt"
SYNTH_TABLE_PATH = MODEL_DATA_DIR / "synth.parquet"
ADAPTER_PATH = MODEL_DATA_DIR / "adapter_anat_full.pt"
LINEAGE_PATH = MODEL_DATA_DIR / "adapter_anat_full.lineage.json"


def path_label(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type != "cuda":
    print("WARNING: CUDA is unavailable; the recovered 2,000-epoch recipe will be very slow on CPU.")

for required_path in (IMAGES_SYNTH_PATH, SYNTH_TABLE_PATH):
    if not required_path.is_file():
        raise FileNotFoundError(required_path)


# %% [markdown]
# ## Load the aligned synthetic dataset
#

# %%
images_gen = torch.load(IMAGES_SYNTH_PATH, map_location="cpu", weights_only=True).float()
df_synth = pd.read_parquet(SYNTH_TABLE_PATH)

required_columns = {"title", "description", "text"}
missing_columns = required_columns.difference(df_synth.columns)
if missing_columns:
    raise ValueError(f"{SYNTH_TABLE_PATH} is missing columns: {sorted(missing_columns)}")

text = df_synth["text"].str.replace(" [sep] ", " [SEP] ", regex=False).tolist()
if len(images_gen) != len(text):
    raise ValueError(f"unaligned inputs: {len(images_gen)} images != {len(text)} texts")
if not torch.isfinite(images_gen).all():
    raise ValueError("images_synth.pt contains non-finite values")

print(images_gen.shape, df_synth.shape)


# %%
specter = load_model("specter")
specter = specter.to(DEVICE)
specter.specter.eval()

batch_size = 512
text_emb = torch.empty((len(text), 768), dtype=torch.float32)
for start in range(0, len(text), batch_size):
    stop = min(start + batch_size, len(text))
    with torch.inference_mode():
        embeddings = F.normalize(specter(text[start:stop]), dim=1)
    text_emb[start:stop] = embeddings.detach().cpu()

print(text_emb.shape, images_gen.shape)


# %%
# Preserve the recovered run's split and subsequent initialization RNG state.
SEED = 123
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
inds = torch.randperm(len(images_gen))

n_train_rows = int(len(inds) * 0.99)
if n_train_rows == 0 or n_train_rows == len(inds):
    raise ValueError("The 99/1 train-validation split produced an empty partition.")

images_train = images_gen[inds[:n_train_rows]].to(DEVICE)
images_val = images_gen[inds[n_train_rows:]].to(DEVICE)
text_train_emb = text_emb[inds[:n_train_rows]].to(DEVICE)
text_val_emb = text_emb[inds[n_train_rows:]].to(DEVICE)


# %% [markdown]
# ## Train and select by validation loss
#

# %%
# These are the exact base model types used by the recovered run.
autoencoder = load_model("autoencoder").to(DEVICE).eval()
proj_head = load_model("proj_head_text_mse").to(DEVICE).eval()

# The decoder is frozen by InterleavedDecoderAdapter; the MSE projection head
# and the inserted residual/calibration layers remain trainable, matching the
# checkpoint recovered from the archived anatomy_v8 notebook.
adapter = InterleavedDecoderAdapter(
    autoencoder,
    proj_head,
    hidden_dim=1024,
    freeze_pretrained_decoder=True,
    freeze_proj_head=False,
).to(DEVICE)

n_epochs = 2000
lr = 2e-4
train_batch_size = 2048
val_batch_size = 2048
eval_every = 20

lineage = {
    "producer": "docs/03_models/12_neuroadapter.ipynb",
    "recovered_from": "archived anatomy_v8 notebook",
    "inputs": {
        path_label(IMAGES_SYNTH_PATH): sha256_file(IMAGES_SYNTH_PATH),
        path_label(SYNTH_TABLE_PATH): sha256_file(SYNTH_TABLE_PATH),
    },
    "base_models": {
        "autoencoder_state_sha256": sha256_state_dict(autoencoder),
        "proj_head_text_mse_state_sha256": sha256_state_dict(proj_head),
    },
    "training": {
        "seed": SEED,
        "split": "torch.randperm; first 99% train, final 1% validation",
        "hidden_dim": 1024,
        "epochs": n_epochs,
        "learning_rate": lr,
        "train_batch_size": train_batch_size,
        "validation_batch_size": val_batch_size,
        "evaluate_every": eval_every,
        "optimizer": "AdamW",
        "weight_decay": 0.0,
        "loss": "binary_cross_entropy_with_logits",
    },
}

opt = AdamW(adapter.trainable_parameters(), lr=lr, weight_decay=0.0)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt,
    mode="min",
    factor=0.5,
    patience=4,
    threshold=1e-5,
    threshold_mode="abs",
    cooldown=0,
    min_lr=1e-5,
)

best_val_loss = math.inf
best_epoch = -1
best_adapter_state = None
n_train = images_train.shape[0]
n_val = images_val.shape[0]

for epoch in range(n_epochs):
    adapter.train()
    order = torch.randperm(n_train, device=images_train.device)
    total_loss = 0.0
    n_seen = 0

    for start in range(0, n_train, train_batch_size):
        batch_idx = order[start:start + train_batch_size]
        x = images_train[batch_idx]
        y = text_train_emb[batch_idx]

        opt.zero_grad(set_to_none=True)
        logits = adapter(y)
        loss = F.binary_cross_entropy_with_logits(logits, x)
        loss.backward()
        opt.step()

        batch_n = x.shape[0]
        total_loss += loss.detach().item() * batch_n
        n_seen += batch_n

    train_loss = total_loss / n_seen

    if epoch % eval_every == 0 or epoch == n_epochs - 1:
        adapter.eval()
        val_total = 0.0
        val_seen = 0

        with torch.inference_mode():
            for start in range(0, n_val, val_batch_size):
                x_val = images_val[start:start + val_batch_size]
                y_val = text_val_emb[start:start + val_batch_size]
                val_logits = adapter(y_val)
                val_loss = F.binary_cross_entropy_with_logits(val_logits, x_val)

                batch_n = x_val.shape[0]
                val_total += val_loss.detach().item() * batch_n
                val_seen += batch_n

        val_loss = val_total / val_seen
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_adapter_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in adapter.state_dict().items()
            }

        sched.step(val_loss)
        print(
            f"epoch {epoch}: train={train_loss:.6f} val={val_loss:.6f} "
            f"best_val={best_val_loss:.6f}@{best_epoch} "
            f"lr={opt.param_groups[0]['lr']:.2e}"
        )


# %%
if best_adapter_state is None:
    raise RuntimeError("Training completed without a validation checkpoint.")

adapter.load_state_dict(best_adapter_state)
adapter.eval()
torch.save(adapter, ADAPTER_PATH)

lineage["result"] = {
    "best_epoch": best_epoch,
    "best_validation_loss": best_val_loss,
    "adapter_state_sha256": sha256_state_dict(adapter),
    "checkpoint": path_label(ADAPTER_PATH),
}
LINEAGE_PATH.write_text(json.dumps(lineage, indent=2) + "\n")

print(f"loaded best validation adapter from epoch {best_epoch}: val={best_val_loss:.6f}")
print(f"saved adapter checkpoint: {ADAPTER_PATH}")
print(f"saved lineage manifest: {LINEAGE_PATH}")


# %% [markdown]
# ## Export aligned derivative artifacts
#

# %%
with torch.inference_mode():
    image_gen_f = torch.sigmoid(adapter(text_emb.to(DEVICE)))
    latent_image_synth = autoencoder.encoder(image_gen_f)

if image_gen_f.shape != images_gen.shape:
    raise RuntimeError(f"unexpected generated image shape: {image_gen_f.shape}")

torch.save(image_gen_f, MODEL_DATA_DIR / "images_synth_trained_full.pt")
df_synth.to_parquet(MODEL_DATA_DIR / "text_synth_full.parquet")
torch.save(text_emb, MODEL_DATA_DIR / "latent_text_synth_full.pt")
torch.save(latent_image_synth, MODEL_DATA_DIR / "latent_image_synth_full.pt")

# These preserve the exact components embedded in adapter_anat_full.pt.
torch.save(autoencoder, MODEL_DATA_DIR / "autoencoder_full.pt")
torch.save(proj_head, MODEL_DATA_DIR / "proj_head_full.pt")

