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
#     display_name: .venv (3.12.12)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## NeuroVault Text-Brain Decoding Examples

# %%
import matplotlib.pyplot as plt
import json

import numpy as np
import torch
from torch.nn import functional as F


import torch.nn.functional as F
import torch
import pandas as pd
from pathlib import Path

def select_device() -> str:
    """Prefer Apple MPS, then CUDA, and otherwise use the CPU."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

from tqdm import tqdm
from neurovlm.models import load_model
from neurovlm.data import load_dataset
from neurovlm.data import load_masker
from neurovlm.evaluation.notebook_utils import resolve_evaluation_output_dir

evaluation_output_dir = resolve_evaluation_output_dir()
evaluation_output_dir.mkdir(parents=True, exist_ok=True)

from nilearn.plotting import plot_stat_map
from nilearn.datasets import load_mni152_template

# Load specter
device = select_device()
specter = load_model("specter").to(device)
adapter = load_model("neuro_adapter").to(device)
proj_head_text = load_model("proj_head_text_infonce").to(device)
proj_head_image = load_model("proj_head_image_infonce").to(device)
enc = load_model("autoencoder").encoder.to(device)
masker = load_masker()
mni = load_mni152_template(1)

# %%
# load
df_text = load_dataset('neurovault_text')
images = load_dataset('neurovault_images')
images = images / images.quantile(0.99, dim=1).unsqueeze(1)
images = images.clamp(0, 1)
meta = load_dataset('neurovault_images_meta')

m = torch.where(~images.isnan().any(dim=1))[0]
images = images[m]
meta = meta.iloc[m]

# contrastive text-image similarities
text = (df_text["title"] + " [SEP] " + df_text["abstract"]).tolist()
with torch.no_grad():
    e = F.normalize(specter(text), dim=1)
    im_pred = torch.sigmoid(adapter(e))
    text_contrastive = F.normalize(proj_head_text(e), dim=1)
    generated_image_contrastive = F.normalize(
        proj_head_image(enc(im_pred)),
        dim=1,
    )
    true_image_contrastive = F.normalize(
        proj_head_image(enc(images.to(device))),
        dim=1,
    )

true_pair_sim_matrix = text_contrastive @ true_image_contrastive.T
generated_pair_sim = (
    text_contrastive * generated_image_contrastive
).sum(dim=1).detach().cpu().numpy()
cos_sim = true_pair_sim_matrix


# %%
# best contrastively aligned true image per text DOI
true_pair_sim_np = true_pair_sim_matrix.detach().cpu().numpy()
meta_doi = meta["doi"].to_numpy()

idxs = []
true_pair_sim = []

for i, doi in enumerate(df_text["doi"].to_numpy()):
    candidate_idxs = np.flatnonzero(meta_doi == doi)

    if len(candidate_idxs) == 0:
        idxs.append(-1)
        true_pair_sim.append(np.nan)
        continue

    best_local = int(np.argmax(true_pair_sim_np[i, candidate_idxs]))
    best_idx = int(candidate_idxs[best_local])

    idxs.append(best_idx)
    true_pair_sim.append(float(true_pair_sim_np[i, best_idx]))

idxs = np.array(idxs)
true_pair_sim = np.array(true_pair_sim)
text_to_brain_pair_sim = generated_pair_sim
pair_sim = np.minimum(true_pair_sim, text_to_brain_pair_sim)
image_sim = text_to_brain_pair_sim
sim = pair_sim


# %%
# manually selected df_text indices for the final figure
keep_text_idxs = [
    189, # (0.389065, 0.345614)
    149, # (0.340806, 0.279071)
    #289, # (0.321985, 0.276818)

    #129, # (0.212434, 0.177845)
    124, # (0.227416, 0.22498)
    152, # (0.203613, 0.226288)

    147, # (0.14427, 0.081241)
    167, # (0.116343, 0.10695)
    # 97,  # (0.124199, 0.067719)
]


# %%
# brain-to-text generation, cache rebuilt on every run
from neurovlm import NeuroVLM

batch_size = 64
results_path = evaluation_output_dir / "neurovault_decoding_examples.json"

if results_path.exists():
    results_path.unlink()

payload = {
    "cache_created_at": pd.Timestamp.now().isoformat(),
    "similarity_model": "proj_head_text_infonce + proj_head_image_infonce",
}
generated_text_by_image_idx = {}

generation_text_idxs = np.array(
    [int(i) for i in keep_text_idxs]
    if "keep_text_idxs" in globals()
    else np.arange(len(df_text)),
    dtype=int,
)
valid_text_idxs = generation_text_idxs[
    (idxs[generation_text_idxs] >= 0)
    & np.isfinite(true_pair_sim[generation_text_idxs])
    & np.isfinite(text_to_brain_pair_sim[generation_text_idxs])
    & np.isfinite(pair_sim[generation_text_idxs])
]
valid_image_idxs = idxs[valid_text_idxs].astype(int)
unique_image_idxs = np.unique(valid_image_idxs)
missing_image_idxs = unique_image_idxs

payload["brain_to_text_generation_scope"] = {
    "source": "keep_text_idxs" if "keep_text_idxs" in globals() else "all_valid_text_idxs",
    "requested_text_idxs": [int(i) for i in generation_text_idxs],
    "valid_text_idxs": [int(i) for i in valid_text_idxs],
    "image_idxs": [int(i) for i in valid_image_idxs],
}
print(f"Generating brain-to-text for {len(valid_text_idxs)} text rows / {len(missing_image_idxs)} images")

def as_text_list(output, expected):
    if isinstance(output, str):
        texts = [output]
    elif isinstance(output, (list, tuple)):
        texts = [str(x) for x in output]
    elif hasattr(output, "tolist"):
        value = output.tolist()
        texts = [value] if isinstance(value, str) else [str(x) for x in value]
    else:
        raise TypeError(f"Unexpected generate_text output type: {type(output)!r}")

    if len(texts) != expected:
        raise ValueError(f"Expected {expected} generated texts, got {len(texts)}")
    return texts

def save_generation_cache():
    payload["brain_to_text"] = {
        "basis": "all",
        "num_beams": 5,
        "batch_size": batch_size,
        "device": str(generation_device),
        "generated_text_by_image_idx": generated_text_by_image_idx,
    }
    results_path.write_text(json.dumps(payload, indent=2))

generation_device = torch.device(device)
if generation_device.type == "cuda":
    print(f"Generating brain-to-text on {generation_device}: {torch.cuda.get_device_name(generation_device)}")
else:
    print(f"Generating brain-to-text on {generation_device}; CUDA available: {torch.cuda.is_available()}")

def get_nvlm():
    global nvlm
    if "nvlm" not in globals():
        nvlm = NeuroVLM(device=str(generation_device))
        if hasattr(nvlm, "to"):
            moved = nvlm.to(generation_device)
            if moved is not None:
                nvlm = moved
    return nvlm

def generate_brain_text_for_image_idxs(
    image_idxs,
    *,
    basis="all",
    num_beams=5,
    batch_size=batch_size,
    desc=None,
    **generate_kwargs,
):
    model = get_nvlm()
    image_idxs = np.array([int(i) for i in image_idxs])
    generated_texts = []
    if desc is None:
        desc = f"Generating brain-to-text ({basis}, beams={num_beams})"

    for start in tqdm(range(0, len(image_idxs), batch_size), desc=desc):
        batch_image_idxs = image_idxs[start:start + batch_size]
        batch_image_idxs_t = torch.as_tensor(batch_image_idxs, dtype=torch.long)
        batch_images = images[batch_image_idxs_t].to(generation_device)
        generated = model.brain(batch_images).generate_text(
            basis=basis,
            num_beams=num_beams,
            seed=start,
            **generate_kwargs,
        )
        generated_texts.extend(as_text_list(generated, len(batch_image_idxs)))

    return generated_texts

def generate_brain_text_for_text_idxs(text_idxs, update_cache=True, **generate_kwargs):
    text_idxs = [int(i) for i in text_idxs]
    image_idxs = [int(idxs[i]) for i in text_idxs]
    generated_texts = generate_brain_text_for_image_idxs(image_idxs, **generate_kwargs)
    generated_by_text_idx = {
        str(int(text_idx)): generated_text
        for text_idx, generated_text in zip(text_idxs, generated_texts)
    }

    if update_cache:
        for image_idx, generated_text in zip(image_idxs, generated_texts):
            generated_text_by_image_idx[str(int(image_idx))] = generated_text
        save_generation_cache()
        if "generated_text_by_text_idx" in globals():
            generated_text_by_text_idx.update(generated_by_text_idx)
            payload["brain_to_text"]["generated_text_by_text_idx"] = generated_text_by_text_idx
        payload["brain_to_text"]["last_manual_text_idxs"] = text_idxs
        payload["brain_to_text"]["last_manual_generation_kwargs"] = generate_kwargs
        results_path.write_text(json.dumps(payload, indent=2))

    return generated_by_text_idx

if len(missing_image_idxs) > 0:
    nvlm = get_nvlm()
    for start in tqdm(
        range(0, len(missing_image_idxs), batch_size),
        desc="Generating brain-to-text",
    ):
        batch_image_idxs = missing_image_idxs[start:start + batch_size]
        batch_image_idxs_t = torch.as_tensor(batch_image_idxs, dtype=torch.long)
        batch_images = images[batch_image_idxs_t].to(generation_device)
        batch_images[batch_images < 0.25] = 0.0
        generated = nvlm.brain(batch_images).generate_text(
            basis="all",
            num_beams=3,
            projection_temp=0.035,
            repetition_penalty=1.5

        )
        generated = as_text_list(generated, len(batch_image_idxs))

        for image_idx, generated_text in zip(batch_image_idxs, generated):
            generated_text_by_image_idx[str(int(image_idx))] = generated_text
        save_generation_cache()

save_generation_cache()

generated_text_by_text_idx = {
    str(int(text_idx)): generated_text_by_image_idx[str(int(image_idx))]
    for text_idx, image_idx in zip(valid_text_idxs, valid_image_idxs)
}
payload["brain_to_text"]["generated_text_by_text_idx"] = generated_text_by_text_idx
results_path.write_text(json.dumps(payload, indent=2))


# %%
# brain-to-text similarity in SPECTER space
generated_text = [generated_text_by_text_idx[str(int(i))] for i in valid_text_idxs]
generated_text_enc_chunks = []

with torch.no_grad():
    for start in tqdm(
        range(0, len(generated_text), batch_size),
        desc="Embedding generated text",
    ):
        generated_text_batch = generated_text[start:start + batch_size]
        generated_text_enc_chunks.append(
            F.normalize(specter(generated_text_batch), dim=1).detach().cpu()
        )

    generated_text_enc = torch.cat(generated_text_enc_chunks, dim=0)
    true_text_enc = e[
        torch.as_tensor(valid_text_idxs, dtype=torch.long, device=e.device)
    ].detach().cpu()

text_sim = np.full(len(df_text), np.nan)
text_sim[valid_text_idxs] = (generated_text_enc * true_text_enc).sum(dim=1).numpy()

valid_image_idxs = idxs[valid_text_idxs].astype(int)
if "text_to_brain_pair_sim" not in globals():
    text_to_brain_pair_sim = generated_pair_sim

selected_image_idxs_t = torch.as_tensor(
    valid_image_idxs,
    dtype=torch.long,
    device=true_image_contrastive.device,
)
with torch.no_grad():
    generated_text_contrastive = F.normalize(
        proj_head_text(generated_text_enc.to(device)),
        dim=1,
    ).detach().cpu()
    selected_true_image_contrastive = true_image_contrastive[
        selected_image_idxs_t
    ].detach().cpu()

brain_to_text_pair_sim = np.full(len(df_text), np.nan)
brain_to_text_pair_sim[valid_text_idxs] = (
    generated_text_contrastive * selected_true_image_contrastive
).sum(dim=1).numpy()
direction_pair_sim = {
    "text_to_brain": text_to_brain_pair_sim,
    "brain_to_text": brain_to_text_pair_sim,
}

payload["true_pair_similarity_by_text_idx"] = {
    str(int(i)): float(true_pair_sim[int(i)])
    for i in valid_text_idxs
}
payload["text_to_brain_pair_similarity_by_text_idx"] = {
    str(int(i)): float(text_to_brain_pair_sim[int(i)])
    for i in valid_text_idxs
}
payload["brain_to_text_pair_similarity_by_text_idx"] = {
    str(int(i)): float(brain_to_text_pair_sim[int(i)])
    for i in valid_text_idxs
}
payload["generated_pair_similarity_by_text_idx"] = payload["text_to_brain_pair_similarity_by_text_idx"]
payload["brain_to_text_specter_similarity_by_text_idx"] = {
    str(int(i)): float(text_sim[int(i)])
    for i in valid_text_idxs
}
payload["brain_to_text_similarity_by_text_idx"] = payload["brain_to_text_pair_similarity_by_text_idx"]
payload["image_similarity_by_text_idx"] = payload["text_to_brain_pair_similarity_by_text_idx"]
payload["text_similarity_by_text_idx"] = payload["brain_to_text_pair_similarity_by_text_idx"]
results_path.write_text(json.dumps(payload, indent=2))


# %%
# selected examples for final figure
def percentile_rank(values, value):
    values = values[np.isfinite(values)]
    return float(100 * np.mean(values <= value))

def value_at(series, idx):
    value = series.iloc[int(idx)]
    return None if pd.isna(value) else str(value)

def best_mse_cut_coords(true_vec, pred_vec, threshold=0.5):
    true_img = masker.inverse_transform(true_vec)
    pred_img = masker.inverse_transform(pred_vec)
    true_data = true_img.get_fdata()
    pred_data = pred_img.get_fdata()
    mse_data = (true_data - pred_data) ** 2
    support = np.isfinite(mse_data) & ((true_data >= threshold) | (pred_data >= threshold))
    if not np.any(support):
        support = np.isfinite(mse_data)
    ijks = np.argwhere(support)
    ijk = ijks[int(np.argmin(mse_data[support]))]
    return true_img.affine.dot(np.r_[ijk, 1])[:3].astype(float).tolist()

finite_text_idxs = valid_text_idxs[
    np.isfinite(true_pair_sim[valid_text_idxs])
    & np.isfinite(text_to_brain_pair_sim[valid_text_idxs])
    & np.isfinite(brain_to_text_pair_sim[valid_text_idxs])
    & np.isfinite(text_sim[valid_text_idxs])
]
true_pair_pct = np.full(len(df_text), np.nan)
image_similarity_pct = np.full(len(df_text), np.nan)
text_similarity_pct = np.full(len(df_text), np.nan)
for text_idx in finite_text_idxs:
    true_pair_pct[int(text_idx)] = percentile_rank(
        true_pair_sim[finite_text_idxs],
        true_pair_sim[int(text_idx)],
    )
    image_similarity_pct[int(text_idx)] = percentile_rank(
        text_to_brain_pair_sim[finite_text_idxs],
        text_to_brain_pair_sim[int(text_idx)],
    )
    text_similarity_pct[int(text_idx)] = percentile_rank(
        brain_to_text_pair_sim[finite_text_idxs],
        brain_to_text_pair_sim[int(text_idx)],
    )

joint_similarity_pct = np.minimum(image_similarity_pct, text_similarity_pct)
direction_percentile_gap = np.abs(image_similarity_pct - text_similarity_pct)

# To try different generation settings, run for example:
# generate_brain_text_for_text_idxs(keep_text_idxs, num_beams=10)
# Then rerun the brain-to-text similarity cell and this cell.
examples = []

for rank, text_idx in enumerate(keep_text_idxs, start=1):
    text_idx = int(text_idx)
    image_idx = int(idxs[text_idx])

    true_image = images[image_idx].detach().cpu().numpy()
    generated_image = im_pred[text_idx].detach().cpu().numpy()

    examples.append({
        "label": "selected",
        "rank": rank,
        "text_idx": text_idx,
        "image_idx": image_idx,
        "doi": value_at(df_text["doi"], text_idx),
        "image_doi": value_at(meta["doi"], image_idx),
        "title": value_at(df_text["title"], text_idx),
        "abstract": value_at(df_text["abstract"], text_idx),
        "true_text": text[text_idx],
        "generated_text": generated_text_by_text_idx[str(text_idx)],
        "image_similarity": float(text_to_brain_pair_sim[text_idx]),
        "text_similarity": float(brain_to_text_pair_sim[text_idx]),
        "image_similarity_percentile": float(image_similarity_pct[text_idx]),
        "text_similarity_percentile": float(text_similarity_pct[text_idx]),
        "joint_similarity_percentile": float(joint_similarity_pct[text_idx]),
        "direction_percentile_gap": float(direction_percentile_gap[text_idx]),
        "true_pair_similarity": float(true_pair_sim[text_idx]),
        "true_pair_similarity_percentile": float(true_pair_pct[text_idx]),
        "text_to_brain_pair_similarity": float(text_to_brain_pair_sim[text_idx]),
        "brain_to_text_pair_similarity": float(brain_to_text_pair_sim[text_idx]),
        "brain_to_text_specter_similarity": float(text_sim[text_idx]),
        "brain_to_text_specter_similarity_percentile": percentile_rank(
            text_sim[finite_text_idxs],
            text_sim[text_idx],
        ),
        "cut_coords": best_mse_cut_coords(true_image, generated_image, threshold=0.5),
    })

payload["selection_mode"] = "manual_text_idx"
payload["keep_text_idxs"] = [int(i) for i in keep_text_idxs]
payload.pop("percentile_targets", None)
payload.pop("selection_constraints", None)
payload.pop("examples_by_direction", None)
payload.pop("percentile_targets_by_direction", None)
payload["examples"] = examples
results_path.write_text(json.dumps(payload, indent=2))


# %%
_df = pd.DataFrame(payload["examples"])

for _, row in _df.sort_values(["label", "rank"]).iterrows():
    idx = int(row["text_idx"])
    print(row["generated_text"])
    print()


# %%
# text_idx = keep_text_idxs[-1]
# text_idx = int(text_idx)
# image_idx = int(idxs[text_idx])

# true_image = images[image_idx].detach().cpu().numpy()
# generated_image = im_pred[text_idx].detach().cpu().numpy()

# from nilearn.plotting import view_img, plot_stat_map
# view_img(
#     masker.inverse_transform(
#         true_image
#     ),
#     threshold=0.5
# )

# %%
# save each brain image as a separate figure
import joblib
from joblib import Parallel, delayed
from contextlib import contextmanager

figure_dir = evaluation_output_dir / "neurovault_decoding_figures"
figure_dir.mkdir(exist_ok=True)
plot_jobs = []

for example in tqdm(examples, desc="Preparing brain figures"):
    text_idx = example["text_idx"]
    image_idx = example["image_idx"]
    cc = example["cut_coords"]
    stem = f"{example['label']}_{example['rank']}_text{text_idx}_image{image_idx}"

    image_specs = [
        ("generated_image", im_pred[text_idx].detach().cpu().numpy()),
        ("true_image", images[image_idx].detach().cpu().numpy()),
    ]

    for image_type, stat_vec in image_specs:
        output_file = figure_dir / f"{stem}_{image_type}.png"
        plot_jobs.append({
            "label": example["label"],
            "rank": example["rank"],
            "text_idx": text_idx,
            "image_idx": image_idx,
            "image_type": image_type,
            "stat_vec": stat_vec,
            "cut_coords": cc,
            "path": str(output_file),
        })

@contextmanager
def tqdm_joblib(tqdm_object):
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()

def save_brain_figure(job):
    stat_img = masker.inverse_transform(job["stat_vec"])
    plot_stat_map(
        stat_img,
        colorbar=False,
        draw_cross=False,
        annotate=False,
        bg_img=mni,
        black_bg=False,
        cmap="Reds",
        threshold=0.5,
        #cut_coords=job["cut_coords"],
        output_file=job["path"],
    )
    return {
        "label": job["label"],
        "rank": job["rank"],
        "text_idx": job["text_idx"],
        "image_idx": job["image_idx"],
        "image_type": job["image_type"],
        "path": job["path"],
    }

n_jobs = min(8, len(plot_jobs)) or 1
with tqdm_joblib(tqdm(total=len(plot_jobs), desc="Saving brain figures")):
    plot_paths = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(save_brain_figure)(job) for job in plot_jobs
    )

payload["plots"] = {
    "figure_dir": str(figure_dir),
    "n_jobs": n_jobs,
    "images": plot_paths,
}
results_path.write_text(json.dumps(payload, indent=2))
plot_paths


# %%
_df = pd.DataFrame(payload["examples"])

for _, row in _df.sort_values(["label", "rank"]).iterrows():
    idx = int(row["text_idx"])

    print("IDX", idx, row["label"], "rank", row["rank"])
    print(row[[
        "text_similarity",
        "image_similarity",
        "text_similarity_percentile",
        "image_similarity_percentile",
        "joint_similarity_percentile",
        "direction_percentile_gap",
        "true_pair_similarity_percentile",
    ]].to_string())
    print()

    print("GENERATED")
    print(row["generated_text"])
    print()

    print("TRUE")
    print(df_text.iloc[idx]["title"])
    print(str(df_text.iloc[idx]["abstract"])[:220])
    print()
    print()
