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
# # Grounded Q-Former pretraining
#
# Trains the grounded Q-Former on aligned PubMed image/text examples using the
# latest projection heads refreshed from Hugging Face `main`. The stable handoff is
# `data_dir / "qformer_grounded_permissive_neuro_best.pt"`; downstream notebooks must never
# select the final epoch merely because it has the largest epoch number.
#

# %%
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path
import math
import re
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset as load_dataset_hf
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm
from IPython.display import display

from neurovlm.models import ProjHead, load_model
from neurovlm.models.serialization import load_model as load_safetensors_model
from neurovlm.data import data_dir, load_dataset, load_latent
from neurovlm.retrieval.summarization import load_huggingface_model
from neurovlm.resources.loaders import NEURO_QWEN_REPO_ID

SEED = 17
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cuda.matmul.allow_tf32 = True

MODEL_DATA_DIR = data_dir
MODEL_DATA_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_ROOT = next(
    candidate for candidate in (Path.cwd(), *Path.cwd().parents)
    if (candidate / "pyproject.toml").is_file()
)
SUMMARY_PATH = PROJECT_ROOT / "docs" / "02_data" / "neuro_summaries.parquet"


# %%
# Load high-level summary targets and align every row by PMID.
_df_pubs = load_dataset("pubmed_text").sort_values(by="pmid").reset_index(drop=True)

summary = load_dataset_hf("neurovlm/pubmed_summary_qa")["train"].to_pandas()
summary = summary.rename(columns={"title": "name", "summary": "description"})
summary = summary.sort_values(by="pmid").reset_index(drop=True)

latent_images_raw, pmids_images_raw = load_latent("pubmed_images")
latent_images_raw = torch.as_tensor(latent_images_raw, dtype=torch.float32).cpu().clone()
pmids_images_raw = np.asarray(pmids_images_raw)

shared_pmids = np.intersect1d(summary["pmid"].to_numpy(), pmids_images_raw)
summary = summary[summary["pmid"].isin(shared_pmids)].sort_values("pmid").reset_index(drop=True)
image_lookup = {pmid: i for i, pmid in enumerate(pmids_images_raw)}
image_idx = np.array([image_lookup[pmid] for pmid in summary["pmid"].to_numpy()])
latent_images = latent_images_raw[torch.as_tensor(image_idx, dtype=torch.long)]
pmids_images = pmids_images_raw[image_idx]
assert (summary["pmid"].to_numpy() == pmids_images).all()

split_meta = _df_pubs[_df_pubs["pmid"].isin(summary["pmid"])].sort_values("pmid").reset_index(drop=True)
assert (summary["pmid"].to_numpy() == split_meta["pmid"].to_numpy()).all()
summary[["train", "test", "val"]] = split_meta[["train", "test", "val"]]

df_pubs = summary
train_mask = df_pubs["train"].to_numpy(dtype=bool)
val_mask = df_pubs["val"].to_numpy(dtype=bool)
test_mask = df_pubs["test"].to_numpy(dtype=bool)

print(f"Aligned dataset: {len(df_pubs):,}")
print(f"Train: {int(train_mask.sum()):,} | Val: {int(val_mask.sum()):,} | Test: {int(test_mask.sum()):,}")
print("First target title:", df_pubs.loc[0, "name"])


# %%
df_summaries = pd.read_parquet(SUMMARY_PATH)
df_summaries = df_summaries[df_summaries["pmid"].isin(df_pubs["pmid"])]
df_summaries = df_summaries.sort_values(by="pmid", ignore_index=True)
assert (df_summaries["pmid"].values == df_pubs["pmid"].values).all()
df_pubs["summary"] = df_summaries["summary"]

# %%
# Permissive all-data scoring.
# We compute the same canonical-quality signals as the strict notebook, but we DO NOT drop rows.
# Instead, every row receives LM/alignment weights that reduce the effect of likely noisy targets.
CANONICAL_TOP_K_TEXT = 32
CANONICAL_TOP_K_IMAGE = 16
CANONICAL_KEEP_FRAC_FOR_REPORTING = 0.35
CANONICAL_PROJ_BATCH = 1024
CANONICAL_SIM_BATCH = 64
MIN_LM_WEIGHT = 0.12
MIN_ALIGN_WEIGHT = 0.20
CLINICAL_LM_MULT = 0.35
CLINICAL_ALIGN_MULT = 0.60
ARTIFACT_LM_MULT = 0.25
ARTIFACT_ALIGN_MULT = 0.50
CLINICAL_PATTERN = re.compile(
    r"\b("
    r"patient|patients|clinical|disease|disorder|syndrome|symptom|diagnosis|treatment|therapy|medication|drug|"
    r"schizophrenia|hallucination|psychosis|autism|adhd|depression|depressive|bipolar|anxiety|ptsd|"
    r"alzheimer|dementia|parkinson|stroke|tumou?r|lesion|addiction|substance|smoking|"
    r"mdd|asd|ocd|tbi"
    r")\b",
    flags=re.IGNORECASE,
)

filter_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECTION_HEADS_REPO = "neurovlm/ProjectionHeads"

def load_latest_projection_head(filename, dims):
    path = hf_hub_download(
        repo_id=PROJECTION_HEADS_REPO,
        filename=filename,
        revision="main",
        repo_type="model",
        force_download=True,
    )
    return load_safetensors_model(ProjHead(*dims), path, device=filter_device)

proj_head_text = load_latest_projection_head(
    "proj_head_text_infonce.safetensors", (768, 512, 384)
)
proj_head_image = load_latest_projection_head(
    "proj_head_image_infonce.safetensors", (384, 384, 384)
)
print(f"Projection heads refreshed from {PROJECTION_HEADS_REPO}@main")

latent_text_raw, pmids_text_raw = load_latent("pubmed_text")
latent_text_raw = torch.as_tensor(latent_text_raw, dtype=torch.float32).cpu().clone()
pmids_text_raw = np.asarray(pmids_text_raw)
assert np.all(np.sort(pmids_text_raw) == pmids_text_raw)
text_lookup = {pmid: i for i, pmid in enumerate(pmids_text_raw)}
missing = [pmid for pmid in pmids_images if pmid not in text_lookup]
assert not missing, f"Missing {len(missing)} PMIDs from pubmed_text latents; first few: {missing[:5]}"
text_idx = np.array([text_lookup[pmid] for pmid in pmids_images])
latent_text = latent_text_raw[torch.as_tensor(text_idx, dtype=torch.long)]
pmids_text = pmids_text_raw[text_idx]
assert (pmids_text == pmids_images).all()
print(f"PMID alignment OK: {len(pmids_text):,} rows")

@torch.no_grad()
def project_batched(head, x, batch_size=CANONICAL_PROJ_BATCH):
    outs = []
    for start in range(0, len(x), batch_size):
        batch = x[start:start + batch_size].to(filter_device, non_blocking=True)
        outs.append(head(batch).detach().float().cpu())
    return F.normalize(torch.cat(outs, dim=0), dim=1)

@torch.no_grad()
def topk_cross(query, key, k, batch_size=CANONICAL_SIM_BATCH):
    key_gpu = key.to(filter_device)
    vals_all, idx_all = [], []
    for start in range(0, len(query), batch_size):
        end = min(start + batch_size, len(query))
        q = query[start:end].to(filter_device, non_blocking=True)
        sim = q @ key_gpu.T
        vals, idx = sim.topk(k, dim=1)
        vals_all.append(vals.cpu())
        idx_all.append(idx.cpu())
        del q, sim, vals, idx
    del key_gpu
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(vals_all, dim=0), torch.cat(idx_all, dim=0)

@torch.no_grad()
def topk_self_without_self(x, k, batch_size=CANONICAL_SIM_BATCH):
    x_gpu = x.to(filter_device)
    vals_all, idx_all = [], []
    for start in range(0, len(x), batch_size):
        end = min(start + batch_size, len(x))
        q = x[start:end].to(filter_device, non_blocking=True)
        sim = q @ x_gpu.T
        row = torch.arange(end - start, device=filter_device)
        sim[row, torch.arange(start, end, device=filter_device)] = -float("inf")
        vals, idx = sim.topk(k, dim=1)
        vals_all.append(vals.cpu())
        idx_all.append(idx.cpu())
        del q, sim, vals, idx
    del x_gpu
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(vals_all, dim=0), torch.cat(idx_all, dim=0)

text_emb_all = project_batched(proj_head_text, latent_text)
image_emb_all = project_batched(proj_head_image, latent_images)

retrieved_text_sim, retrieved_text_idx = topk_cross(image_emb_all, text_emb_all, CANONICAL_TOP_K_TEXT)
image_neighbor_sim, image_neighbor_idx = topk_self_without_self(image_emb_all, CANONICAL_TOP_K_IMAGE)

true_text = text_emb_all.unsqueeze(1)
retrieved_text_consistency = (true_text * text_emb_all[retrieved_text_idx]).sum(dim=-1).mean(dim=1)
image_neighbor_text_consistency = (true_text * text_emb_all[image_neighbor_idx]).sum(dim=-1).mean(dim=1)
paired_image_text_score = (image_emb_all * text_emb_all).sum(dim=1)
top_retrieved_score = retrieved_text_sim[:, 0]

canonical_score = (
    0.35 * retrieved_text_consistency
    + 0.35 * image_neighbor_text_consistency
    + 0.20 * paired_image_text_score
    + 0.10 * top_retrieved_score
)

filter_text = (df_pubs["name"].fillna("") + " " + df_pubs["description"].fillna(""))
clinical_mask = filter_text.str.contains(CLINICAL_PATTERN, regex=True, na=False).to_numpy()
artifact_mask = filter_text.str.contains(r"\[[0-9JjIi]+\]|\bPMID\b|\bdoi\b", regex=True, na=False).to_numpy()

train_scores = canonical_score[torch.as_tensor(train_mask)]
score_lo = torch.quantile(train_scores, 0.05)
score_hi = torch.quantile(train_scores, 0.95)
quality = ((canonical_score - score_lo) / (score_hi - score_lo).clamp_min(1e-6)).clamp(0, 1).numpy()

report_threshold = torch.quantile(train_scores, 1.0 - CANONICAL_KEEP_FRAC_FOR_REPORTING)
strict_like_mask = canonical_score.numpy() >= float(report_threshold)

lm_weights = MIN_LM_WEIGHT + (1.0 - MIN_LM_WEIGHT) * quality
align_weights = MIN_ALIGN_WEIGHT + (1.0 - MIN_ALIGN_WEIGHT) * quality
lm_weights[clinical_mask] *= CLINICAL_LM_MULT
align_weights[clinical_mask] *= CLINICAL_ALIGN_MULT
lm_weights[artifact_mask] *= ARTIFACT_LM_MULT
align_weights[artifact_mask] *= ARTIFACT_ALIGN_MULT
lm_weights = np.clip(lm_weights, MIN_LM_WEIGHT, 1.0).astype("float32")
align_weights = np.clip(align_weights, MIN_ALIGN_WEIGHT, 1.0).astype("float32")

# Attach scores and weights. No rows are removed in this permissive notebook.
df_pubs["canonical_score"] = canonical_score.numpy()
df_pubs["canonical_quality"] = quality
df_pubs["strict_like"] = strict_like_mask
df_pubs["retrieved_text_consistency"] = retrieved_text_consistency.numpy()
df_pubs["image_neighbor_text_consistency"] = image_neighbor_text_consistency.numpy()
df_pubs["paired_image_text_score"] = paired_image_text_score.numpy()
df_pubs["clinical_like"] = clinical_mask
df_pubs["artifact_like"] = artifact_mask
df_pubs["lm_weight"] = lm_weights
df_pubs["align_weight"] = align_weights

image_semantic = image_emb_all.cpu().clone()
text_semantic = text_emb_all.cpu().clone()
latent_images = torch.as_tensor(latent_images, dtype=torch.float32).cpu().clone()
assert (df_pubs["pmid"].to_numpy() == pmids_images).all()

print(f"Permissive scoring: kept all {len(df_pubs):,} rows")
print(f"Strict-like reporting threshold={float(report_threshold):.4f} top_frac={CANONICAL_KEEP_FRAC_FOR_REPORTING:.2f}")
for split_name, split_mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
    total = int(split_mask.sum())
    strict_like = int((strict_like_mask & split_mask).sum())
    eff_lm = float(lm_weights[split_mask].sum())
    eff_align = float(align_weights[split_mask].sum())
    print(f"{split_name}: rows {total:,} | strict-like {strict_like:,} ({strict_like / max(total, 1):.1%}) | effective lm {eff_lm:,.1f} | effective align {eff_align:,.1f}")
print(f"Clinical-like rows: {int(clinical_mask.sum()):,}; artifact-like rows: {int(artifact_mask.sum()):,}")
print(f"LM weight mean/range: {lm_weights.mean():.3f} [{lm_weights.min():.3f}, {lm_weights.max():.3f}]")
print(f"Align weight mean/range: {align_weights.mean():.3f} [{align_weights.min():.3f}, {align_weights.max():.3f}]")

print("Lowest-weight examples:")
display(df_pubs.nsmallest(8, "lm_weight")[["pmid", "name", "canonical_score", "clinical_like", "artifact_like", "lm_weight", "align_weight"]])
print("Highest-weight examples:")
display(df_pubs.nlargest(8, "lm_weight")[["pmid", "name", "canonical_score", "clinical_like", "artifact_like", "lm_weight", "align_weight"]])


# %%
# Natural-language targets. Do not use literal [SEP]; Qwen treats it as bracket text.
def clean_space(x):
    x = str(x).replace("\n", " ").strip()
    x = re.sub(r"\s+", " ", x)
    return x

text_name = df_pubs["name"].map(clean_space).tolist()
text_full = [f"{clean_space(n)}\n{clean_space(d)}" for n, d in zip(df_pubs["name"], df_pubs["description"])]

print("Target example:")
print(text_full[0][:500])
print("Contains literal [SEP]?", any("[SEP]" in x for x in text_full))


# %%
autoencoder = load_model("autoencoder")
model, tokenizer = load_huggingface_model(
    NEURO_QWEN_REPO_ID, device="cuda", dtype=torch.bfloat16
)

device = next(model.parameters()).device
lm_dim = model.config.hidden_size
vocab_rows = model.get_input_embeddings().weight.shape[0]
print(f"Qwen hidden size: {lm_dim}")
print(f"Embedding matrix rows/vocab rows: {vocab_rows:,}")
print(f"Tokenizer length: {len(tokenizer):,}")
print(f"EOS: {tokenizer.eos_token_id} {tokenizer.decode([tokenizer.eos_token_id])}")


# %%
# Atlas probes and contrastive retrieval baseline.
from nilearn.image import resample_to_img
from neurovlm.data import load_masker
import nibabel as nib

masker = load_masker()
networks = load_dataset("networks")
network_imgs = []
for k in networks:
    for a in networks[k]:
        network_imgs.append((k, a, nib.Nifti1Image(networks[k][a]["array"], affine=networks[k][a]["affine"])))
network_imgs = [x for x in network_imgs if x[0] not in ("UKBICA", "HCPICA")]
network_imgs = [x for i, x in enumerate(network_imgs) if i in (69, 23, 127, 107, 137, 21, 94, 17)]

latent_neuro = []
for k, a, img in tqdm(network_imgs, desc="atlas latents"):
    img_re = resample_to_img(img, masker.mask_img, interpolation="nearest", force_resample=False, copy_header=True)
    v = masker.transform(img_re)
    with torch.no_grad():
        z = autoencoder.encoder(torch.from_numpy(v).float()).detach().cpu()
    latent_neuro.append(z.reshape(-1))
latent_neuro = torch.stack(latent_neuro).float()
_labels = [f"{k}_{a}" for k, a, _ in network_imgs]
print("Atlas shape:", tuple(latent_neuro.shape))
print("Labels:", _labels)

@torch.no_grad()
def atlas_contrastive_retrieval(n=8):
    proj_head_image.eval()
    atlas_sem = []
    for start in range(0, len(latent_neuro), 8):
        batch = latent_neuro[start:start + 8].to(filter_device)
        atlas_sem.append(proj_head_image(batch).float().cpu())
    atlas_sem = F.normalize(torch.cat(atlas_sem, dim=0), dim=1)
    sim = atlas_sem @ image_semantic.T
    for i, lab in enumerate(_labels):
        idx = sim[i].topk(n).indices.tolist()
        print("\n" + lab)
        for j in idx:
            print(f"  {float(sim[i, j]):+.3f} | {df_pubs.loc[j, 'name']}")

atlas_contrastive_retrieval(n=8)


# %%
PAD_ID = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
EOS_ID = tokenizer.eos_token_id
MAX_TOKENS = 192
TRAIN_BATCH = 32
EVAL_BATCH = 32
NUM_WORKERS = 32

class BrainGroundedTextDataset(Dataset):
    def __init__(self, raw_images, semantic_images, semantic_texts, texts, lm_weights, align_weights, tokenizer, max_tokens=MAX_TOKENS):
        self.raw_images = torch.as_tensor(raw_images, dtype=torch.float32).cpu().clone()
        self.semantic_images = torch.as_tensor(semantic_images, dtype=torch.float32).cpu().clone()
        self.semantic_texts = torch.as_tensor(semantic_texts, dtype=torch.float32).cpu().clone()
        self.lm_weights = torch.as_tensor(lm_weights, dtype=torch.float32).cpu().clone()
        self.align_weights = torch.as_tensor(align_weights, dtype=torch.float32).cpu().clone()
        self.tokenizer = tokenizer
        self.set_texts(texts, max_tokens=max_tokens)

    def set_texts(self, texts, max_tokens=MAX_TOKENS):
        enc = self.tokenizer(list(texts), truncation=True, max_length=max_tokens - 1, add_special_tokens=False)
        self.input_ids = []
        for ids in enc["input_ids"]:
            row = list(ids)
            if len(row) == 0 or row[-1] != EOS_ID:
                row.append(EOS_ID)
            self.input_ids.append(torch.tensor(row[:max_tokens], dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return (
            self.raw_images[idx],
            self.semantic_images[idx],
            self.semantic_texts[idx],
            self.lm_weights[idx],
            self.align_weights[idx],
            self.input_ids[idx],
        )


def collate_fn(batch):
    raw, sem_img, sem_txt, lm_w, align_w, ids = zip(*batch)
    raw = torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in raw])
    sem_img = torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in sem_img])
    sem_txt = torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in sem_txt])
    lm_w = torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in lm_w])
    align_w = torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in align_w])
    lengths = torch.tensor([len(x) for x in ids], dtype=torch.long)
    input_ids = pad_sequence(ids, batch_first=True, padding_value=PAD_ID)
    attn_mask = (torch.arange(input_ids.size(1))[None, :] < lengths[:, None]).long()
    return raw, sem_img, sem_txt, lm_w, align_w, input_ids, attn_mask

train_idx = np.flatnonzero(train_mask)
val_idx = np.flatnonzero(val_mask)
test_idx = np.flatnonzero(test_mask)
texts = np.array(text_full)
lm_weights_all = df_pubs["lm_weight"].to_numpy(dtype="float32")
align_weights_all = df_pubs["align_weight"].to_numpy(dtype="float32")

train_ds = BrainGroundedTextDataset(latent_images[train_idx], image_semantic[train_idx], text_semantic[train_idx], texts[train_idx], lm_weights_all[train_idx], align_weights_all[train_idx], tokenizer)
val_ds = BrainGroundedTextDataset(latent_images[val_idx], image_semantic[val_idx], text_semantic[val_idx], texts[val_idx], lm_weights_all[val_idx], align_weights_all[val_idx], tokenizer)
test_ds = BrainGroundedTextDataset(latent_images[test_idx], image_semantic[test_idx], text_semantic[test_idx], texts[test_idx], lm_weights_all[test_idx], align_weights_all[test_idx], tokenizer)

train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH, shuffle=True, drop_last=True, collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS > 0)
val_loader = DataLoader(val_ds, batch_size=EVAL_BATCH, shuffle=False, drop_last=False, collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS > 0)
test_loader = DataLoader(test_ds, batch_size=EVAL_BATCH, shuffle=False, drop_last=False, collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS > 0)

sample_ids = train_ds[0][-1]
print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
print(f"Batches/epoch: {len(train_loader)} | max tokens: {MAX_TOKENS} | train batch: {TRAIN_BATCH}")
print(f"Effective train LM weight: {float(train_ds.lm_weights.sum()):,.1f}; align weight: {float(train_ds.align_weights.sum()):,.1f}")
print("Sample target:", tokenizer.decode(sample_ids, skip_special_tokens=True)[:500])
print("Sample weights:", float(train_ds.lm_weights[0]), float(train_ds.align_weights[0]))
print("EOS check:", int(sample_ids[-1]) == EOS_ID or EOS_ID in sample_ids.tolist(), EOS_ID, tokenizer.decode([EOS_ID]))


# %%
class GroundedQFormer(nn.Module):
    def __init__(self, image_dim, semantic_dim, lm_dim, num_queries=32, hidden_dim=512, num_heads=8, num_layers=6, dropout=0.05):
        super().__init__()
        self.num_queries = num_queries
        self.raw_proj = nn.Sequential(
            nn.LayerNorm(image_dim),
            nn.Linear(image_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.semantic_proj = nn.Sequential(
            nn.LayerNorm(semantic_dim),
            nn.Linear(semantic_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.to_lm = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, lm_dim),
        )
        self.align_head = nn.Sequential(
            nn.LayerNorm(lm_dim),
            nn.Linear(lm_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, semantic_dim),
        )

    def forward(self, raw_images, semantic_images):
        batch = raw_images.size(0)
        dtype = self.query_tokens.dtype
        raw_images = raw_images.to(dtype=dtype)
        semantic_images = semantic_images.to(dtype=dtype)
        raw_mem = self.raw_proj(raw_images)
        sem_mem = self.semantic_proj(semantic_images)
        mem = torch.stack([raw_mem, sem_mem], dim=1)
        q = self.query_tokens.expand(batch, -1, -1)
        out = self.transformer(q, mem)
        return self.to_lm(out)

    def semantic_from_visual(self, visual_tokens):
        x = visual_tokens.to(dtype=self.query_tokens.dtype).mean(dim=1)
        return F.normalize(self.align_head(x).float(), dim=1)



# %%
for p in model.parameters():
    p.requires_grad = False
# Backward through frozen LM activations is needed for gradients to flow into the visual prompt.
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.train()

semantic_dim = image_semantic.shape[1]
qformer = GroundedQFormer(
    image_dim=latent_images.shape[1],
    semantic_dim=semantic_dim,
    lm_dim=lm_dim,
    num_queries=32,
    hidden_dim=512,
    num_heads=8,
    num_layers=6,
).to(device).to(torch.bfloat16)

with torch.no_grad():
    emb = model.get_input_embeddings().weight.detach().float()
    LM_EMB_MEAN = emb.mean().item()
    LM_EMB_STD = emb.std().item()
    LM_EMB_NORM = emb.norm(dim=1).mean().item()
    LM_EMB_NORM_STD = emb.norm(dim=1).std().item()
print(f"LM embedding stats: mean={LM_EMB_MEAN:.4f} std={LM_EMB_STD:.4f} norm={LM_EMB_NORM:.2f}±{LM_EMB_NORM_STD:.2f}")
print(f"QFormer params: {sum(p.numel() for p in qformer.parameters()):,}")

EPOCHS = 30
PEAK_LR = 3e-4
MIN_LR = 3e-5
WARMUP_FRAC = 0.05
ALIGN_WEIGHT = 0.35
TOKEN_SPACE_WEIGHT = 0.03
WEIGHT_DECAY = 0.03

total_steps = EPOCHS * len(train_loader)
warmup_steps = max(1, int(WARMUP_FRAC * total_steps))
optimizer = torch.optim.AdamW(qformer.parameters(), lr=PEAK_LR, weight_decay=WEIGHT_DECAY, fused=True)

def lr_lambda(step):
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_scale = MIN_LR / PEAK_LR
    return min_scale + (1.0 - min_scale) * cosine

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
print(f"Training: epochs={EPOCHS}, total_steps={total_steps:,}, warmup={warmup_steps:,}, lr={PEAK_LR:g}->{MIN_LR:g}")


# %%
dict(
    image_dim=latent_images.shape[1],
semantic_dim=semantic_dim,
lm_dim=lm_dim,
)


# %%
def match_lm_token_norm(vis):
    target_norm = torch.tensor(LM_EMB_NORM, device=vis.device, dtype=vis.dtype)
    vis_norm = vis.float().norm(dim=-1, keepdim=True).clamp_min(1e-6).to(vis.dtype)
    return vis * (target_norm / vis_norm)

def visual_tokens(qformer, raw_images, semantic_images, norm_match=True):
    vis = qformer(raw_images, semantic_images)
    if not norm_match:
        return vis
    return match_lm_token_norm(vis)


def token_space_reg(vis):
    vf = vis.float()
    norm_mean = vf.norm(dim=-1).mean()
    std = vf.std()
    reg_norm = ((norm_mean - LM_EMB_NORM) / max(LM_EMB_NORM, 1e-6)).pow(2)
    reg_std = ((std - LM_EMB_STD) / max(LM_EMB_STD, 1e-6)).pow(2)
    return reg_norm + 0.25 * reg_std


def weighted_mean(values, weights):
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def grounded_loss(batch, image_mode="correct"):
    raw, sem_img, sem_txt, lm_w, align_w, input_ids, attn_mask = batch
    raw = raw.to(device, non_blocking=True)
    sem_img = sem_img.to(device, non_blocking=True)
    sem_txt = F.normalize(sem_txt.to(device, non_blocking=True).float(), dim=1)
    lm_w = lm_w.to(device, non_blocking=True).float()
    align_w = align_w.to(device, non_blocking=True).float()
    input_ids = input_ids.to(device, non_blocking=True)
    attn_mask = attn_mask.to(device, non_blocking=True)

    if image_mode == "shuffled":
        perm = torch.randperm(raw.size(0), device=device)
        raw = raw[perm]
        sem_img = sem_img[perm]
    elif image_mode == "zero":
        raw = torch.zeros_like(raw)
        sem_img = torch.zeros_like(sem_img)

    batch_size = raw.size(0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        vis_raw = qformer(raw.to(torch.bfloat16), sem_img.to(torch.bfloat16))
        vis = match_lm_token_norm(vis_raw)
        num_q = vis.size(1)
        with torch.no_grad():
            txt = model.get_input_embeddings()(input_ids)
        embeds = torch.cat([vis.to(model.dtype), txt.to(model.dtype)], dim=1)
        full_mask = torch.cat([torch.ones(batch_size, num_q, device=device, dtype=torch.long), attn_mask], dim=1)
        labels = torch.cat([
            torch.full((batch_size, num_q), -100, dtype=torch.long, device=device),
            input_ids.masked_fill(attn_mask == 0, -100),
        ], dim=1)
        out = model(inputs_embeds=embeds, attention_mask=full_mask, use_cache=False)
        shift_logits = out.logits[:, :-1].float().contiguous()
        shift_labels = labels[:, 1:].contiguous()
        valid = shift_labels.ne(-100)
        safe_labels = shift_labels.masked_fill(~valid, 0)
        token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            safe_labels.view(-1),
            reduction="none",
        ).view_as(safe_labels)
        per_sample_lm = (token_loss * valid.float()).sum(dim=1) / valid.float().sum(dim=1).clamp_min(1.0)
        lm = weighted_mean(per_sample_lm, lm_w)

        pred_sem = qformer.semantic_from_visual(vis_raw)
        per_sample_align = 1.0 - (pred_sem * sem_txt).sum(dim=1)
        align = weighted_mean(per_sample_align, align_w)
        reg = token_space_reg(vis_raw)
        loss = lm + ALIGN_WEIGHT * align + TOKEN_SPACE_WEIGHT * reg

    return loss, {
        "lm": lm.detach(),
        "align": align.detach(),
        "reg": reg.detach(),
        "lm_unweighted": per_sample_lm.detach().mean(),
        "align_unweighted": per_sample_align.detach().mean(),
        "lm_weight": lm_w.detach().mean(),
        "align_weight": align_w.detach().mean(),
    }


@torch.no_grad()
def eval_loader(loader, image_mode="correct", max_batches=None):
    qformer.eval()
    model.eval()
    total = {"loss": 0.0, "lm": 0.0, "align": 0.0, "reg": 0.0, "lm_unweighted": 0.0, "align_unweighted": 0.0, "lm_weight": 0.0, "align_weight": 0.0}
    n = 0
    for batch in loader:
        loss, parts = grounded_loss(batch, image_mode=image_mode)
        total["loss"] += float(loss.item())
        for key in parts:
            total[key] += float(parts[key].item())
        n += 1
        if max_batches is not None and n >= max_batches:
            break
    return {k: v / max(n, 1) for k, v in total.items()}



# %%
CLINICAL_BAD_WORDS = [
    "schizophrenia", "hallucination", "hallucinations", "Alzheimer", "Parkinson", "depression", "bipolar",
    "autism", "ADHD", "patients", "patient", "disorder", "disease", "syndrome", "clinical", "treatment",
]

def bad_words_ids(tokenizer, terms):
    bad = []
    for term in terms:
        ids = tokenizer(term, add_special_tokens=False).input_ids
        if ids:
            bad.append(ids)
        ids_space = tokenizer(" " + term, add_special_tokens=False).input_ids
        if ids_space:
            bad.append(ids_space)
    return bad

BAD_WORDS_IDS = bad_words_ids(tokenizer, CLINICAL_BAD_WORDS)

@torch.no_grad()
def generate_caption(raw_img, sem_img=None, *, max_new_tokens=128, num_beams=3, ban_clinical=True):
    qformer.eval()
    model.eval()
    if not torch.is_tensor(raw_img):
        raw_img = torch.tensor(raw_img, dtype=torch.float32)
    raw_img = raw_img.reshape(1, -1).to(device=device, dtype=torch.bfloat16)
    if sem_img is None:
        with torch.no_grad():
            sem_img = F.normalize(proj_head_image(raw_img.float().to(filter_device)).float(), dim=1).to(device)
    elif not torch.is_tensor(sem_img):
        sem_img = torch.tensor(sem_img, dtype=torch.float32)
    sem_img = sem_img.reshape(1, -1).to(device=device, dtype=torch.bfloat16)
    vis = visual_tokens(qformer, raw_img, sem_img, norm_match=True).to(model.dtype)
    attn = torch.ones(vis.shape[:2], dtype=torch.long, device=device)
    out_ids = model.generate(
        inputs_embeds=vis,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=False,
        repetition_penalty=1.18,
        no_repeat_ngram_size=4,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        bad_words_ids=BAD_WORDS_IDS if ban_clinical else None,
    )
    return tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

@torch.no_grad()
def atlas_probe(max_new_tokens=128):
    proj_head_image.eval()
    atlas_sem = []
    for start in range(0, len(latent_neuro), 8):
        batch = latent_neuro[start:start + 8].to(filter_device)
        atlas_sem.append(F.normalize(proj_head_image(batch).float().cpu(), dim=1))
    atlas_sem = torch.cat(atlas_sem, dim=0)
    for i, lab in enumerate(_labels):
        pred = generate_caption(latent_neuro[i], atlas_sem[i], max_new_tokens=max_new_tokens)
        print(f"{lab}\n{pred}\n")

# @torch.no_grad()
# def visual_nearest_token_audit(n_images=4, n_tokens=4):
#     qformer.eval()
#     emb = F.normalize(model.get_input_embeddings().weight.detach().float(), dim=1)
#     raw = latent_images[:n_images].to(device=device, dtype=torch.bfloat16)
#     sem = image_semantic[:n_images].to(device=device, dtype=torch.bfloat16)
#     vis = visual_tokens(qformer, raw, sem, norm_match=True).float().cpu()
#     sims = F.normalize(vis.reshape(-1, vis.shape[-1]), dim=1) @ emb.cpu().T
#     vals, idx = sims.topk(n_tokens, dim=1)
#     for row in range(min(8, idx.size(0))):
#         toks = tokenizer.convert_ids_to_tokens(idx[row].tolist())
#         print(row, [f"{t}:{float(v):.3f}" for t, v in zip(toks, vals[row])])



# %%
history = {"train": [], "val": [], "shuffle": [], "zero": [], "align": [], "lr": []}
best_val = float("inf")
best_epoch = -1
BEST_GROUNDED_CKPT = MODEL_DATA_DIR / "qformer_grounded_permissive_neuro_best.pt"

print("Initial diagnostics")
val = eval_loader(val_loader, "correct", max_batches=50)
shuf = eval_loader(val_loader, "shuffled", max_batches=50)
zero = eval_loader(val_loader, "zero", max_batches=50)
print(f"initial | val {val['loss']:.4f} lm {val['lm']:.4f} align {val['align']:.4f} | shuffled {shuf['loss']:.4f} gap {shuf['loss'] - val['loss']:+.4f} | zero {zero['loss']:.4f} gap {zero['loss'] - val['loss']:+.4f}")
atlas_probe(max_new_tokens=96)

for epoch in range(EPOCHS):
    qformer.train()
    model.train()
    train_loss = train_lm = train_align = train_reg = 0.0
    n_train = 0
    for batch in tqdm(train_loader, desc=f"epoch {epoch + 1}/{EPOCHS}"):
        optimizer.zero_grad(set_to_none=True)
        loss, parts = grounded_loss(batch, image_mode="correct")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(qformer.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        train_loss += float(loss.item())
        train_lm += float(parts["lm"].item())
        train_align += float(parts["align"].item())
        train_reg += float(parts["reg"].item())
        n_train += 1

    train_loss /= max(n_train, 1)
    train_lm /= max(n_train, 1)
    train_align /= max(n_train, 1)
    train_reg /= max(n_train, 1)
    val = eval_loader(val_loader, "correct")
    shuf = eval_loader(val_loader, "shuffled", max_batches=100)
    zero = eval_loader(val_loader, "zero", max_batches=100)
    lr = scheduler.get_last_lr()[0]

    history["train"].append(train_loss)
    history["val"].append(val["loss"])
    history["shuffle"].append(shuf["loss"])
    history["zero"].append(zero["loss"])
    history["align"].append(val["align"])
    history["lr"].append(lr)

    epoch_ckpt = MODEL_DATA_DIR / f"qformer_grounded_permissive_neuro_epoch-{epoch}.pt"
    torch.save(qformer.state_dict(), epoch_ckpt)
    if val["loss"] < best_val:
        best_val = val["loss"]
        best_epoch = epoch
        torch.save(qformer.state_dict(), BEST_GROUNDED_CKPT)
        torch.save(qformer.state_dict(), MODEL_DATA_DIR / f"qformer_grounded_permissive_neuro_best_epoch-{epoch}.pt")

    print(
        f"Epoch {epoch + 1:3d} | train {train_loss:.4f} lm {train_lm:.4f} align {train_align:.4f} "
        f"| val {val['loss']:.4f} lm {val['lm']:.4f} align {val['align']:.4f} "
        f"| shuffle gap {shuf['loss'] - val['loss']:+.4f} zero gap {zero['loss'] - val['loss']:+.4f} | lr {lr:.2e}"
    )
    if epoch % 5 == 0:
        atlas_probe(max_new_tokens=128)

if best_epoch < 0 or not BEST_GROUNDED_CKPT.exists():
    raise RuntimeError("No grounded best checkpoint was produced.")
qformer.load_state_dict(torch.load(BEST_GROUNDED_CKPT, map_location=device, weights_only=True), strict=True)
qformer.eval()
print(f"Stable grounded handoff: {BEST_GROUNDED_CKPT} | best epoch={best_epoch} | val={best_val:.4f}")
atlas_probe(max_new_tokens=128)


# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(history["train"], label="train")
axes[0].plot(history["val"], label="val")
axes[0].set(title="Total Loss", xlabel="Epoch")
axes[0].legend()
axes[1].plot(np.array(history["shuffle"]) - np.array(history["val"]), label="shuffled gap")
axes[1].plot(np.array(history["zero"]) - np.array(history["val"]), label="zero gap")
axes[1].axhline(0, color="k", linewidth=0.8)
axes[1].set(title="Image Dependence", xlabel="Epoch")
axes[1].legend()
axes[2].plot(history["align"], label="contrastive align")
axes[2].plot(history["lr"], label="lr")
axes[2].set(title="Alignment / LR", xlabel="Epoch")
axes[2].legend()
plt.tight_layout()
plt.show()


# %%

@torch.no_grad()
def _generate_caption(raw_img, sem_img=None, *, max_new_tokens=128, do_sample=False, temperature=None, top_p=None, num_beams=3, ban_clinical=True):
    qformer.eval()
    model.eval()
    if not torch.is_tensor(raw_img):
        raw_img = torch.tensor(raw_img, dtype=torch.float32)
    raw_img = raw_img.reshape(1, -1).to(device=device, dtype=torch.bfloat16)
    if sem_img is None:
        with torch.no_grad():
            sem_img = F.normalize(proj_head_image(raw_img.float().to(filter_device)).float(), dim=1).to(device)
    elif not torch.is_tensor(sem_img):
        sem_img = torch.tensor(sem_img, dtype=torch.float32)
    sem_img = sem_img.reshape(1, -1).to(device=device, dtype=torch.bfloat16)
    vis = visual_tokens(qformer, raw_img, sem_img, norm_match=True).to(model.dtype)
    attn = torch.ones(vis.shape[:2], dtype=torch.long, device=device)
    out_ids = model.generate(
        inputs_embeds=vis,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.18,
        no_repeat_ngram_size=4,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        bad_words_ids=BAD_WORDS_IDS if ban_clinical else None,
    )
    return tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()


# %%
proj_head_image.eval()
atlas_sem = []
for start in range(0, len(latent_neuro), 8):
    batch = latent_neuro[start:start + 8].to(filter_device)
    atlas_sem.append(F.normalize(proj_head_image(batch).float().cpu(), dim=1))
atlas_sem = torch.cat(atlas_sem, dim=0)

# %%
gens= {i: [] for i in _labels}
for i, lab in enumerate(_labels):
    pred = _generate_caption(latent_neuro[i], atlas_sem[i], ban_clinical=False)
    print(f"{lab}\n{pred}\n")
    gens[lab].append(pred)
