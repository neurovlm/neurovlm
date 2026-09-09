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
#     display_name: .conda
#     language: python
#     name: python3
# ---

# %%
from tqdm.notebook import tqdm
import re
from math import ceil
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import CountVectorizer
from neurovlm.data import data_dir, load_dataset, load_latent
from neurovlm.models import Specter


# %% [markdown]
# # Corpus Extraction
# Extract n-grams for the training corpus. N-grams are weighted by cosine similarity to article embeddings, e.g. if n-gram is highly similar to the articles it gets a value near 1, otherwise it gets a value near 0.

# %%
def extract_ngrams(docs, ngram_range):
    counts = CountVectorizer(
        ngram_range=ngram_range,
        stop_words="english",
        min_df=1
    ).fit(docs)

    X = counts.transform(docs)  # shape: (n_docs, n_features)

    feature_names = counts.get_feature_names_out()

    mask = np.array(X.sum(axis=0) >= 100)[0]

    X = np.array(X[:, mask].todense())
    feature_names = feature_names[mask]

    return X, feature_names

# manual cleaning
DROP_SUBSTRINGS = [
    # study-like language
    "study", "studies", "result", "indicate", "show", "related",
    "differences", "significant", "effect", "role", "measure",
    "displayed", "involved", "examined", "associated", "altered",
    "performed", "demonstrated", "conclus", "correlate", "individuals",
    "common", "prior",
    # too general
    "brain", "neural", "neuroimaging", "mri", "fmri", "connectivity",
    "diagnosed", "patients", "little", "known", "activation", "blood",
    "alterations", "neuroscience", "people",
]

DROP_REGEXES = [
    r"^cortex",
    # general single terms
    r"^ventral$", r"^frontal$", r"^neuronal$", r"^cognitive$",
    r"^cerebral$", r"^resting_state$",  r"^disorder$",
    r"^neuropsychological$", r"^cognition$", r"^stimulus$",
    r"^dysfunction$", r"^imaging$", r"^functional$",
    r"^functional imaging$", r"^task performance$", r"^impairments$",
    r"^traits$", r"^dysfunction$",  r"^cognitive abilities$", r"^imaging dti$",
    # [SEP] token
    r"\bsep\b",
]

pattern = "|".join(
    [re.escape(s) for s in DROP_SUBSTRINGS] +  # plain terms
    DROP_REGEXES                               # regex terms
)


# %%
def clean_vocab_and_remap_X(
    X: np.ndarray,
    terms,
    *,
    force_keep=(),
    drop_any_tokens=(
        "magnetic","resonance","imaging","mri","fmri","functional","scan","scanner","image","data",
    ),
    drop_first_tokens=(
        "use","using","undergo","undergoing","show","showing","change","changes",
        "investigate","investigating","examine","examining","perform","performed",
    ),
):
    """
    Dense binary X (n_docs, n_terms) + term list -> cleaned vocab + remapped dense X via OR-merge.

    Steps:
      1) normalize
      2) keep only "nouny" terms (heuristic) + force_keep
      3) collapse contiguous subphrases into longest phrase
      4) OR-merge columns into new X

    Returns:
      X_new: (n_docs, n_new) bool
      new_terms: list[str]
      old_to_new_term: dict[old_raw -> new_term or None]
      old_to_new_idx: np.int32 (len=n_terms; -1 dropped)
    """
    assert X.shape[1] == len(terms)

    _acronym_re = re.compile(r"^[a-z]{2,6}$")
    _vowel_re = re.compile(r"[aeiou]")

    def norm(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r"[^a-z0-9\s\-]", " ", s)
        s = re.sub(r"\s+", " ", s)
        return s

    def tok(s: str):
        return [w for w in norm(s).split() if w.isalpha()]

    def is_acronym_like(s: str) -> bool:
        s = norm(s).replace(" ", "")
        return s.isalpha() and (2 <= len(s) <= 6) and (_vowel_re.search(s) is None)  # dmn, pcc

    def is_acronym(s: str) -> bool:
        s = norm(s).replace(" ", "")
        return bool(_acronym_re.match(s))

    drop_any_tokens = {norm(w) for w in drop_any_tokens}
    drop_first_tokens = {norm(w) for w in drop_first_tokens}
    force_keep = {norm(w) for w in force_keep}

    terms = list(terms)
    n_docs, n_terms = X.shape

    # normalize + tokens
    norm_terms = [norm(t) for t in terms]
    toks = [tok(t) for t in norm_terms]

    # noun keep mask (heuristic) + force_keep
    keep = np.zeros(n_terms, dtype=bool)
    canon_terms = [None] * n_terms

    for j, (raw, nt, tt) in enumerate(zip(terms, norm_terms, toks)):
        if nt in force_keep:
            keep[j] = True
            canon_terms[j] = nt
            continue

        if not tt:
            continue

        if len(tt) == 1:
            # keep only acronym-ish unigrams (dmn/pcc) OR explicit force_keep handled above
            if is_acronym_like(tt[0]) or is_acronym(tt[0]):
                keep[j] = True
                canon_terms[j] = tt[0]
            continue

        if tt[0] in drop_first_tokens:
            continue
        if any(w in drop_any_tokens for w in tt):
            continue
        if any(w.endswith(("ing", "ed")) for w in tt):
            continue
        if any(w in {"of","to","in","on","with","by","for","from","as","at","and","or","the","a","an"} for w in tt):
            continue

        keep[j] = True
        canon_terms[j] = " ".join(tt)

    # contiguous-subphrase, collapse on token tuples
    tok_tuples = [tuple(ct.split()) if ct else tuple() for ct in canon_terms]
    keep_toks = {tok_tuples[j] for j in range(n_terms) if keep[j] and tok_tuples[j]}

    supers = sorted(keep_toks, key=len, reverse=True)
    sub_to_super = {}
    dropped = set()

    for super_tt in supers:
        L = len(super_tt)
        if L < 2:
            continue
        for i in range(L):
            for k in range(i + 1, L + 1):
                sub_tt = super_tt[i:k]
                if len(sub_tt) < 2 or sub_tt == super_tt:
                    continue
                if sub_tt in keep_toks and sub_tt not in dropped:
                    dropped.add(sub_tt)
                    sub_to_super[sub_tt] = super_tt

    final_toks = keep_toks - dropped
    new_terms = sorted((" ".join(tt) for tt in final_toks), key=lambda s: (len(s.split()), s))
    new_index = {t: i for i, t in enumerate(new_terms)}

    old_to_new_idx = np.full(n_terms, -1, dtype=np.int32)
    old_to_new_term = {}

    for j, raw in enumerate(terms):
        if not keep[j] or not canon_terms[j]:
            old_to_new_term[raw] = None
            continue
        tt = tok_tuples[j]
        canon_tt = sub_to_super.get(tt, tt)
        canon = " ".join(canon_tt)
        old_to_new_idx[j] = new_index[canon]
        old_to_new_term[raw] = canon

    # --- 4) OR-merge dense X
    Xb = X.astype(bool, copy=False)
    X_new = np.zeros((n_docs, len(new_terms)), dtype=bool)
    for old_j, new_j in enumerate(old_to_new_idx):
        if new_j >= 0:
            X_new[:, new_j] |= Xb[:, old_j]

    return X_new, new_terms



# %%
# load latent text
latent, pmids = load_latent("pubmed_text")

# load text
df = load_dataset("pubmed_text")
text = df["name"] + " [SEP] " + df["description"]

# extract
X_uni, features_uni = extract_ngrams(text, (1, 1))
X_bi, features_bi = extract_ngrams(text, (2, 2))
X_tri, features_tri = extract_ngrams(text, (3, 3))

# %%
X = np.hstack((X_uni, X_bi, X_tri))
features = np.concat((features_uni, features_bi, features_tri))

# drop
mask = ~pd.Series(features).str.contains(pattern, case=False, na=False, regex=True).to_numpy()
features = features[mask]
X = X[:, mask]
X = X[df["pmid"].argsort().to_numpy()]

# clean
X, features = clean_vocab_and_remap_X(X, features, force_keep=["precuneus", "working memory", "putamen"])
features = np.array(features)

# %%
# manually edited
# corpus = pd.read_csv("corpus.txt")
# pd.concat((corpus, pd.DataFrame({"terms": features[mask], "counts": X.sum(axis=0)[mask]}))).sort_values(by="counts", ascending=False).to_csv("corpus.txt", index=False)
# pd.DataFrame({"terms": features, "counts": X.sum(axis=0)}).sort_values(by="counts", ascending=False).to_csv("corpus.txt", index=False)

# %%
# apply manual filter
corpus = pd.read_csv("corpus.txt")
mask = pd.Series(features).isin(corpus["terms"])
features = features[mask]
X = X[:, mask]
features.shape, X.shape

# %%
np.save(data_dir / "ngram_matrix.npy", X)
np.save(data_dir / "ngram_labels.npy", features)

# %%
# specter embeddings for ngrams
specter = Specter()
specter.specter = specter.specter.eval()

# if not (data_dir / "ngram_emb.pt").exists():
ngram_emb = []
batch_size = 512
for i in tqdm(range(0, len(features), batch_size), total=ceil(len(features)//batch_size)):
    with torch.no_grad():
        ngram_emb.append(specter(features[i:i+batch_size].tolist()))
ngram_emb = torch.vstack(ngram_emb)
ngram_emb = ngram_emb / ngram_emb.norm(dim=1)[:, None] # unit vector
torch.save(ngram_emb, data_dir / "ngram_emb.pt")

# %%
