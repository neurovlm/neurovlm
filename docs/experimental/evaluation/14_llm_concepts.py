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
#     display_name: .env
#     language: python
#     name: python3
# ---

# %%
import pickle, gzip
from tqdm.notebook import tqdm
from math import ceil
import pandas as pd
import numpy as np
import torch

from nilearn.image import resample_to_img
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from neurovlm.resources.loaders import (
    _load_masker, _load_autoencoder, _load_networks
)
from neurovlm.data import data_dir
from neurovlm.models import ConceptClf
def select_device() -> str:
    """Prefer Apple MPS, then CUDA, and otherwise use the CPU."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

device = select_device()

# %% [markdown]
# # Interpreting Brain Maps
#
# ## Corpus Extraction
# Extract n-grams for the training corpus. N-grams are weighted by cosine similarity to article embeddings, e.g. if n-gram is highly similar to the articles it gets a value near 1, otherwise it gets a value near 0.
#
# # Concept Classifier
# The concept classifier predicts which concepts are present given a latent neuro embeddings. The top-10 related concepts are passed to an LLM to summarize the brain map. Here, Llama-3.1-8B-Instruct is used to generated interpretations. Any language model may be used. Larger models or models trained one neuroscience literature may provided better brain map interpretations.

# %%
# Load from 10_n_grams.ipynb
concept_clf = torch.load(data_dir / "concept_clf.pt", weights_only=False)

# %%
X = np.load(data_dir / "ngram_matrix.npy")
features = np.load(data_dir / "ngram_labels.npy")
X.shape

# %% [markdown]
# ## Network Correspondence
#
# Test geneartion on the network dataset. Predicted concepts are passed to the LLM to summarizes.

# %%
# Load
masker = _load_masker()
autoencoder = _load_autoencoder()
networks = _load_networks()

# %%
# Load LLM
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model.config.pad_token_id = tokenizer.pad_token_id

# Eval mode and disable gradients
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
torch.set_grad_enabled(False)

# %%
system_prompt = """
You are a neuroscience editor writing a short wiki-style article from a list of terms.

INPUT: a list of neuroscience terms (networks, brain regions, cognitive functions, disorders).
OUTPUT: ONE article that uses the terms to form a coherent theme.

Rules:
1) Title (required): 6–12 words. Make it specific and content-based.
   - Use 1–2 of the most informative terms (prefer: network/circuit + region + cognition; add disorder only if strongly supported).
   - DO NOT use generic titles like: "Summary", "Overview", "Brain Network Analysis", "A Summary of Terms".

2) Lead paragraph (2–3 sentences):
   - State the unifying theme directly (what the terms collectively describe).
   - Name 3–5 “anchor” terms that drive the theme.
   - Do NOT say “the provided list”, “top-ranked”, “these terms appear”, or anything about scoring/ranking.

3) Body sections:
   - Networks
   - Key Regions
   - Cognitive Functions
   - Clinical Relevance

4) Be concrete:
   - Prefer specific mechanisms, pathways, and canonical associations over vague statements.
   - If a term is too vague/ambiguous/unrelated, ignore it in the main text.

No references. Do not mention this prompt.
""".strip()

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

# %%
import nibabel as nib

with gzip.open(data_dir / "networks_arrays.pkl.gz", "rb") as f:
    networks = pickle.load(f)

network_imgs = []
for k in networks.keys():
    for a in networks[k].keys():
        network_imgs.append((k, a, nib.Nifti1Image(networks[k][a]["array"], affine=networks[k][a]["affine"])))

networks = [i for i in network_imgs if i[0] not in ["UKBICA", "HCPICA"]]

# %%
if not (data_dir / "networks_emb.pt").exists():
    # Encode networks
    networks_emb = torch.zeros((len(networks), 384))
    for i, row in tqdm(enumerate(networks), total=len(networks)):

        # Encode network image
        with torch.no_grad():
            x = masker.transform(
                resample_to_img(row[2], masker.mask_img, interpolation="nearest", force_resample=True, copy_header=True)
            )
            x = autoencoder.encoder(torch.from_numpy(x))
            networks_emb[i] = x

    torch.save(networks_emb, data_dir / "networks_emb.pt")
else:
    networks_emb = torch.load(data_dir / "networks_emb.pt")

# %%
# Load from 06_n_grams.ipynb
features = np.load(data_dir / "ngram_labels.npy")

# Compute llm response
messages = []

for i, row in tqdm(enumerate(networks), total=len(networks)):

    # Concept classifier
    scores = torch.sigmoid(concept_clf(networks_emb[i].to(device)).cpu().detach())

    # Pass concepts to LLM
    user_prompt = ", ".join(features[scores.argsort().flip(0).numpy()[:20]])

    messages.append([
        {"role": "system", "content": system_prompt.strip("\n")},
        {"role": "user", "content": user_prompt},
    ])

# %%
out_dir = data_dir / "networks_text_gen"
out_dir.mkdir(exist_ok=True)

generated_summaries = []
batch_size = 5
for i in tqdm(range(0, len(networks), batch_size), total=ceil(len(networks)/batch_size)):
    with torch.inference_mode():
        outputs = pipe(
            messages[i:i+batch_size],
            max_new_tokens=1000,
            do_sample=True,
            temperature=0.2,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.1,
            return_full_text=True,
            padding=True,
            truncation=True
        )
    for idx in range(len(outputs)):
        generated_summary = outputs[idx][0]["generated_text"][-1]["content"].strip()
        row = networks[i:i+batch_size][idx]
        with open(out_dir / f"generated_summaries_{row[0].lower().replace("/", "-")}_{row[1].lower().replace("/", "-")}.txt", "w") as f:
            f.write(generated_summary)

# %%
files = list(out_dir.glob("*.txt"))
files.sort()
rows = []
for file in files:
    with open(file, "r") as f:
        txt = f.read().strip()
    atlas, name = str(file).split("generated_summaries_")[-1].split(".txt")[0].split("_")
    rows.append((atlas, name, txt))

# %%
df_results = pd.DataFrame({
    "atlas": [i[0] for i in rows],
    "name": [i[1] for i in rows],
    "text": [i[2] for i in rows]
})

df_results.to_csv(data_dir / "networks_text_gen.csv")
