#!/usr/bin/env python
"""Publish the NeuroVLM revision (2026-10) on the Hugging Face org neurovlm. Dry run unless --execute.

    .venv/bin/python scripts/pipeline/push_hf_release.py                      # dry run of everything: uploads, checks, manifest diff
    .venv/bin/python scripts/pipeline/push_hf_release.py --only models --execute --update-manifest
    .venv/bin/python scripts/pipeline/push_hf_release.py --only neuroqwen --neuroqwen-dir scripts/outputs/runs/<run>/neuroqwen \\
        --execute --update-manifest
    .venv/bin/python scripts/pipeline/push_hf_release.py --only corpus --execute --yes --update-manifest

models     runs/final to the existing repos, under the names the library loaders read: NeuroAutoEncoder, ProjectionHeads
           (+ map_calibration.json), NeuroAdapter (anatomical head, + map_calibration.json), NeuroQformer (stage 14,
           + canonical_semantic_banks.pt, title_vocabulary.json), NeuroQformer-PubMed (stage 13). The 20 fold checkpoints
           of stage 20 go to the new repo neurovlm/NeuroVLM-cv-folds. One commit per repo.
neuroqwen  the export of models/00_neuroqwen_dapt.py (--neuroqwen-dir: neuroqwen/ or neuroqwen/NeuroQwen3-0.6B/) to
           neurovlm/NeuroQwen3-0.6B. The export must say tie_word_embeddings false and pass its export_check.json.
corpus     outputs/data/pubmed_neuroimaging_redacted.parquet to the new dataset repo neurovlm/pubmed_neuroimaging_corpus.
           Publishes 1.23 M PubMed records, so --execute also needs --yes.

Each README.md gets a "Revision 2026-10" section (replaced when present, the rest of the card is kept), new repos a whole
card. config.json files are uploaded with absolute local paths cut to scripts/... The dry run only reads the Hub.

--execute refuses when git status is dirty, a run file's sha256 differs from EXPECTED_SHA256 (the validated runs/final,
cross-checked with scripts/README.md, title_vocabulary.json and map_calibration.json), a parquet/csv/json/md file holds an
e-mail address, the installed neurovlm loaders cannot load a weight file (--allow-loader-mismatch overrides), or a
NeuroQwen export is a smoke run or fails its export check. After the commits it reads every file's sha256 back from the
Hub and, with --update-manifest, re-pins those repos in scripts/config/hf_manifest.json (diff printed; files new to a repo
inherit its groups, files of new repos get the group "release"). --pin-only does only that last step, for the current
Hub heads. If runs/final changes after validation, check it and paste the output of --print-lock into EXPECTED_SHA256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/pipeline/
from common import HF_MANIFEST, REPO_ROOT, SCRIPTS_DIR, read_json, sha256_file  # noqa: E402

TAG = "Revision 2026-10"
RUN_DIR = SCRIPTS_DIR / "outputs" / "runs" / "final"
CORPUS = SCRIPTS_DIR / "outputs" / "data" / "pubmed_neuroimaging_redacted.parquet"
CV_REPO = "neurovlm/NeuroVLM-cv-folds"
CORPUS_REPO = "neurovlm/pubmed_neuroimaging_corpus"
QWEN_REPO = "neurovlm/NeuroQwen3-0.6B"
NEW_GROUP = "release"
HEADS = ("proj_head_image_infonce", "proj_head_text_infonce", "proj_head_text_mse")

# repo -> {path in the repo: path in the run dir}
MODEL_FILES = {
    "neurovlm/NeuroAutoEncoder": {"autoencoder.safetensors": "autoencoder.safetensors"},
    "neurovlm/ProjectionHeads": {**{f"{h}.safetensors": f"{h}.safetensors" for h in HEADS},
                                 "map_calibration.json": "map_calibration.json"},
    "neurovlm/NeuroAdapter": {"model.safetensors": "adapter_anat_full.safetensors", "map_calibration.json": "map_calibration.json"},
    "neurovlm/NeuroQformer": {"model.safetensors": "neuro_qformer_canonical/model.safetensors",
                              "config.json": "neuro_qformer_canonical/config.json",
                              "canonical_semantic_banks.pt": "canonical_semantic_banks.pt",
                              "title_vocabulary.json": "title_vocabulary.json"},
    "neurovlm/NeuroQformer-PubMed": {"model.safetensors": "neuro_qformer_pubmed/model.safetensors",
                                     "config.json": "neuro_qformer_pubmed/config.json"},
    CV_REPO: {f"{m}_kfold-{k}.safetensors": f"{m}_kfold-{k}.safetensors" for m in ("autoencoder", *HEADS) for k in range(5)},
}

# revisions before this release (runs/final/provenance.json hf_revisions), named in the cards as the earlier weights
EARLIER = {
    "neurovlm/NeuroAutoEncoder": "b7c288b34d68a71dd303a1a9de02c8aadf6dd1ae",
    "neurovlm/ProjectionHeads": "bcb7e8ce6664d364359d3614eb82dca4eb9a72d7",
    "neurovlm/NeuroAdapter": "1db5f18c512dad7af14c35a548593557a9399f31",
    "neurovlm/NeuroQformer": "29b45da49a50edb4a130c5134d600e3aad66c668",
    "neurovlm/NeuroQformer-PubMed": "2493bafd8f41b9130d8d76832870c18bdcd8ef4c",
    QWEN_REPO: "23f50c9cb35635d473b709e074fb5efb53c7ba30",
}
LABEL = {
    "neurovlm/NeuroAutoEncoder": "final autoencoder",
    "neurovlm/ProjectionHeads": "final projection heads, map calibration",
    "neurovlm/NeuroAdapter": "final anatomical head, map calibration",
    "neurovlm/NeuroQformer": "final stage-14 MLP projector, banks, title vocabulary",
    "neurovlm/NeuroQformer-PubMed": "final stage-13 MLP projector",
    CV_REPO: "5-fold PubMed CV checkpoints",
}

# sha256 of the validated runs/final (2026-10-13; equal to the scripts/README.md table) and of the redacted corpus
EXPECTED_SHA256 = {
    "adapter_anat_full.safetensors": "60d8ae78086462f1c261419dea0ce4409583c8e9cc8766195635abc80fd78372",
    "autoencoder.safetensors": "151d62eebc805ea79007415355c8b26069af7d9c8ee545b7fddb6a5922731f55",
    "autoencoder_kfold-0.safetensors": "759157278ec88416a76127249a8fcb98fe2090ada209df64746e5fb8a4cf751a",
    "autoencoder_kfold-1.safetensors": "f39e8be41530ed8d40cc0b8f30180aa9bb5a77a8818b04cda06afe3d5c1056bb",
    "autoencoder_kfold-2.safetensors": "e688e26b86f52e45402a8231b03463eccd7bde6394fc324fcbef0a76273471b4",
    "autoencoder_kfold-3.safetensors": "764818ee46cae91a85c60d8c0262cc1818defa5730c4e755a5e260e8bb6065e0",
    "autoencoder_kfold-4.safetensors": "c07c28c515fdd3d29466323bfdcb8dcb219ad5277527f67219e65392b9ca36e1",
    "canonical_semantic_banks.pt": "f28293a0595afab0a137ba8cc96b8023b5562e7b464f7071a3d5230da7f20771",
    "map_calibration.json": "6f9daa11efb35446e32ad5604c36bdeae19ea3bc305cd5c3f289aea28154be83",
    "neuro_qformer_canonical/config.json": "e11b001a523b696102a6145b4b0a9d96ced6bd24614374c340d7e75ec11b1e17",
    "neuro_qformer_canonical/model.safetensors": "eaf5ee494e96e40f9c75f2b9bd4aed0edcbe4589b8fafe9d0365402a5515a5a9",
    "neuro_qformer_pubmed/config.json": "1779627e3070f0268d674f00da6e1393f54f8f64ad2710293d5692b90b137738",
    "neuro_qformer_pubmed/model.safetensors": "4fc17243c5514021ca8d8543112826107da3c87899531c279698531bbeb5cb37",
    "proj_head_image_infonce.safetensors": "917f14964043ce9f021990cb758d7dacf463ed40a31c6f9ac990e53fe31647c9",
    "proj_head_image_infonce_kfold-0.safetensors": "29851eeaeeada4d3b03769e6586454f8476289896d74d4129f9a4396ea78f88e",
    "proj_head_image_infonce_kfold-1.safetensors": "081ac3b3f62cdd8db0576e3b048a943b752bffd5658a9296b6bf8441efb7b5b5",
    "proj_head_image_infonce_kfold-2.safetensors": "fbbb5fc5f3a88f47d2c948625c78442ffd6c64d6fc183f6be1e2149fa751b58b",
    "proj_head_image_infonce_kfold-3.safetensors": "2e9213f0af4b3fee97896aa83191e5ea768cd84a3ff578f6f8af50627b86b2cb",
    "proj_head_image_infonce_kfold-4.safetensors": "f6e304d98992fcf3c4395a8d7c6489b0689a04a7c8822009a46448218a22fdb6",
    "proj_head_text_infonce.safetensors": "7a079ac851c627be8fb8804ad23ec227a03fbfe35548b5cc284069f7069ff5fa",
    "proj_head_text_infonce_kfold-0.safetensors": "4b4adec81052ed87141157624be9943297ee4a4ffa1f681416db1d59c074ecc5",
    "proj_head_text_infonce_kfold-1.safetensors": "681161296c151b58dd341fe1c4ac331a4d416ac8b7fe5b385853fc725fe90ee7",
    "proj_head_text_infonce_kfold-2.safetensors": "67af327d926fd991fb3695218d8cd0d777dd401328f2af19a58ce34d98ab6e11",
    "proj_head_text_infonce_kfold-3.safetensors": "7aebacf3eaca7e5b126cb36f94c23b7e1a5b00c27e22b8724968346786ea5618",
    "proj_head_text_infonce_kfold-4.safetensors": "fd9e0878e6fa14a5d23a2d01772280d541f510e5269f6f9dbc99a84c1be176d1",
    "proj_head_text_mse.safetensors": "ed00cccbf66f5082464c3cd0d1ab0643a7f6ba21ea66ce0617eabfef13c615c0",
    "proj_head_text_mse_kfold-0.safetensors": "6cc88269b2e6be4cdc5bae00594a953b5fd6ddeb8d3253629ffd9f328ba83ff4",
    "proj_head_text_mse_kfold-1.safetensors": "2d97b40afa4013329bdbd21f84660eaad98518e6911842f1194c06653a9ae758",
    "proj_head_text_mse_kfold-2.safetensors": "affcdc1231292c3c946256ef1d196c75e01a9672202056522e953a23c9a49e4d",
    "proj_head_text_mse_kfold-3.safetensors": "07c0462b9b145eeb0579bfe3a740ebc043ff009422ed775e76416486032245c1",
    "proj_head_text_mse_kfold-4.safetensors": "427a3bfbef4caca6dbb7fe18beed7314fe448e6cdeb5b1580219cc694153aaae",
    "title_vocabulary.json": "31acb78f7f5ef304cd6e69d037d4f91af069883859b2919b2c5c1a79056bdf85",
    "corpus/pubmed_neuroimaging_redacted.parquet": "ec74c2479ae452d19bf4b22f2ff32e2ef17959a3e96ac31f9a30de618e0ebf95",
}

HERE = {
    "neurovlm/NeuroAutoEncoder": "`autoencoder.safetensors` is the retrained autoencoder (stage 08). `mask.npz` and "
                                 "`config.json` are unchanged.",
    "neurovlm/ProjectionHeads": "The three heads are retrained (stage 09). `proj_head_text_mse` is the text-to-brain head "
                                "(v2 loss), and its calibration factor is `head` in `map_calibration.json`.",
    "neurovlm/NeuroAdapter": "`model.safetensors` is the retrained anatomically centered projection head (stage 12, "
                             "anchored recipe), and `config.json` is unchanged. Its calibration factor is `adapter` in "
                             "`map_calibration.json`.",
    "neurovlm/NeuroQformer": "`model.safetensors` and `config.json` hold the stage-14 MLP projector with the frozen image head "
                             "and the canonical banks, which are also in `canonical_semantic_banks.pt`. `title_vocabulary.json` "
                             "lists the titles the decoder may write. Loading needs a loader that reads "
                             "`config.json[\"projector\"]` (`load_projector` in NeuroVLM's `scripts/models/_common_models.py`).",
    "neurovlm/NeuroQformer-PubMed": "`model.safetensors` and `config.json` hold the stage-13 MLP projector (best validation "
                                    "epoch), with canonical projection off. Loading needs a loader that reads "
                                    "`config.json[\"projector\"]` (`load_projector` in NeuroVLM's `scripts/models/_common_models.py`).",
    CV_REPO: "First release of these checkpoints.",
}

CV_CARD = """---
library_name: neurovlm
tags:
- neuroimaging
- brain-maps
- cross-validation
---

# NeuroVLM-cv-folds

Checkpoints of the 5-fold PubMed cross-validation of NeuroVLM (stage 20, `scripts/models/20_pubmed_cv.py`).

The {rows} PubMed studies with at most 100 coordinates are split into five folds
(`KFold(n_splits=5, shuffle=True, random_state=0)`). Fold k trains the autoencoder, the text-to-brain head and the two
InfoNCE heads on the other four folds and holds out fold k. `scripts/pipeline/validate_models.py --components cv` scores them.

| files | module |
|---|---|
| `autoencoder_kfold-{{k}}.safetensors` | `NeuroAutoEncoder(seed=0, out="logit")` |
| `proj_head_text_mse_kfold-{{k}}.safetensors` | `ProjHead(768, 512, 384)` |
| `proj_head_text_infonce_kfold-{{k}}.safetensors` | `ProjHead(768, 512, 384)` |
| `proj_head_image_infonce_kfold-{{k}}.safetensors` | `ProjHead(384, 384, 384)` |

```python
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from neurovlm.models.base import NeuroAutoEncoder

ae = NeuroAutoEncoder(seed=0, out="logit")
ae.load_state_dict(load_file(hf_hub_download("{repo}", "autoencoder_kfold-0.safetensors")))
```
"""

QUERY = ('("neuroimaging"[MeSH Terms] OR "brain mapping"[MeSH Terms] OR "magnetic resonance imaging"[MeSH Terms] '
         'OR "functional magnetic resonance imaging"[MeSH Terms] OR "diffusion magnetic resonance imaging"[MeSH Terms] '
         'OR "diffusion tensor imaging"[Title/Abstract] OR DTI[Title/Abstract] OR tractography[Title/Abstract] '
         'OR fMRI[Title/Abstract] OR "functional MRI"[Title/Abstract] OR "resting state"[Title/Abstract] '
         'OR MRI[Title/Abstract] OR "magnetic resonance"[Title/Abstract] OR PET[Title/Abstract] '
         'OR "positron emission tomography"[Title/Abstract] OR EEG[Title/Abstract] OR electroencephalograph*[Title/Abstract] '
         'OR MEG[Title/Abstract] OR magnetoencephalograph*[Title/Abstract] OR "voxel-based morphometry"[Title/Abstract] '
         'OR VBM[Title/Abstract] OR "brain connectivity"[Title/Abstract] OR connectome[Title/Abstract] '
         'OR connectomics[Title/Abstract])')

CORPUS_CARD = """---
license: other
license_name: pubmed-terms
license_link: https://www.nlm.nih.gov/databases/download/terms_and_conditions.html
language:
- en
tags:
- neuroimaging
- neuroscience
- pubmed
task_categories:
- text-generation
size_categories:
- 1M<n<10M
configs:
- config_name: default
  data_files: {path}
---

# PubMed neuroimaging corpus

Titles and abstracts of {rows:,} PubMed records ({y0}-{y1}, {journals:,} journals) returned by a broad neuroimaging
query. NeuroQwen3-0.6B, the text decoder of NeuroVLM, was trained on it (continued pretraining of Qwen3-0.6B).

| column | content |
|---|---|
| `pmid` | PubMed ID (unique) |
| `doi` | DOI |
| `year` | publication year |
| `journal` | journal |
| `title` | title |
| `abstract` | abstract (empty in {empty:,} records) |
| `authors` | author names |

## How it was built

`project/scrape.py` of efficient_ai queried NCBI E-utilities with the query below, from 1990/01/01 on. ESearch is split
over publication-date ranges to get past its 10,000-record limit, the PMIDs are deduplicated in SQLite, and EFetch
downloads the records in batches of 200. The output was saved as parquet on 2026-02-27.

```text
{query}
```

## Redaction

E-mail addresses were replaced with `[email removed]`, 99 addresses in 82 records (5 titles, 72 abstracts, 7 author
lists). Everything else is identical to the file NeuroQwen3-0.6B was trained on (sha256 `{source}`, pinned under
`not_on_hub` in NeuroVLM's `scripts/config/hf_manifest.json`), which is not published.

## Evaluation papers

The corpus still holds the validation and test papers of NeuroVLM and its NeuroVault evaluation papers.
`scripts/models/00_neuroqwen_dapt.py` removes them (6,144 records) before training.

## License

The titles and abstracts come from PubMed (U.S. National Library of Medicine). NLM's terms apply to the records, and
abstracts can be under the copyright of their publishers or authors. The corpus is shared for non-commercial research
and text mining. Check those terms before any other use.
"""

EMAIL = r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}"
TOKENIZER_FILES = {"tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json", "added_tokens.json",
                   "special_tokens_map.json", "chat_template.jinja"}  # vocabularies, not scanned for e-mail addresses


@dataclass
class Release:
    repo: str
    repo_type: str  # model | dataset
    title: str
    body: str
    head: str | None  # Hub head when read (None: new repo)
    files: dict[str, Path] = field(default_factory=dict)  # path in the repo -> local file (generated ones in the staging dir)
    sources: dict[str, Path] = field(default_factory=dict)  # path in the repo -> run / corpus file checked against EXPECTED_SHA256
    notes: dict[str, str] = field(default_factory=dict)  # path in the repo -> how a generated file was made
    sha: dict[str, str] = field(default_factory=dict)  # path in the repo -> sha256 of the local file

    def add(self, path: str, local: Path, source: Path | None = None, note: str = "") -> None:
        self.files[path], self.sha[path] = Path(local), sha256_file(local)
        if source is not None:
            self.sources[path] = Path(source)
        if note:
            self.notes[path] = note


def size_str(n: int) -> str:
    return f"{n / 1e6:.1f} MB" if n >= 1e5 else f"{n / 1e3:.1f} kB"


def git(*cmd: str) -> str:
    return subprocess.run(["git", *cmd], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def git_blob_sha1(path: Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def lock_key(path: Path, run_dir: Path) -> str:
    path = Path(path).resolve()
    return str(path.relative_to(run_dir.resolve())) if path.is_relative_to(run_dir.resolve()) else f"corpus/{path.name}"


# ---------------------------------------------------------------------------------------------------- Hub reads
class Hub:
    def __init__(self):
        from huggingface_hub import HfApi

        self.api, self.errors = HfApi(), []

    def head(self, repo: str, repo_type: str) -> str | None:
        from huggingface_hub.utils import RepositoryNotFoundError

        try:
            return self.api.repo_info(repo, repo_type=repo_type).sha
        except RepositoryNotFoundError:
            return None
        except Exception as exc:  # offline, ...
            self.errors.append(f"{repo}: {type(exc).__name__}: {exc}")
            return "?"

    def readme(self, repo: str, repo_type: str, head: str | None, manifest: dict) -> str:
        """README.md at the Hub head, else at the revision pinned in the manifest (HF cache), else ''."""
        from huggingface_hub import hf_hub_download

        pinned = next((e["revision"] for e in manifest["files"] if e["repo"] == repo and e["path"] == "README.md"), None)
        for rev, offline in ((head, False), (pinned, True)):
            if rev in (None, "?"):
                continue
            try:
                return Path(hf_hub_download(repo, "README.md", repo_type=repo_type, revision=rev, local_files_only=offline)).read_text()
            except Exception:
                continue
        return ""


# ---------------------------------------------------------------------------------------------------- cards
def put_section(card: str, section: str) -> str:
    """Replace the '## Revision 2026-10' section of a card (up to the next '## ' heading), or append it."""
    lines = card.rstrip("\n").split("\n") if card.strip() else []
    sec = section.rstrip("\n").split("\n")
    if f"## {TAG}" in lines:
        i = lines.index(f"## {TAG}")
        j = next((k for k in range(i + 1, len(lines)) if lines[k].startswith("## ")), len(lines))
        lines = lines[:i] + sec + ([""] + lines[j:] if j < len(lines) else [])
    else:
        lines = lines + ([""] if lines else []) + sec
    return "\n".join(lines) + "\n"


def files_table(rel: Release) -> list[str]:
    rows = ["| file | size | sha256 |", "|---|---|---|"]
    rows += [f"| `{p}` | {size_str(f.stat().st_size)} | `{rel.sha[p]}` |" for p, f in rel.files.items() if p != "README.md"]
    return rows


def model_section(rel: Release, facts: dict, earlier: str | None) -> str:
    f = facts
    bullets = [
             "- **Deterministic training.** Every stage runs with seeded data loaders and deterministic kernels. Two runs from "
             "scratch gave byte-identical weights on the same GPU model, driver and library versions.",
             f"- **MLP projector.** The brain-to-text projectors (stages 13 and 14) are residual MLPs ({f['mlp_params'] / 1e6:.1f} M "
             "parameters) instead of Q-Formers. A package names its projector in `config.json[\"projector\"]`.",
             "- **Stage-14 recipe.** Each training text is projected onto the canonical bank of its own level, every training map "
             f"is re-binarized per batch, and title tokens weigh 5 in the loss. Five seeds were trained and the one with the lowest "
             f"validation loss (seed {f['seed']}) is packaged.",
             "- **Calibration.** Text-to-brain maps are multiplied by one global factor per model, fitted on the validation split "
             f"and clipped at 1 (`map_calibration.json`, `head` {f['head']:.2f} and `adapter` {f['adapter']:.2f}).",
             "- **Title decoding.** After a [NETWORK] or [REGION] tag the decoder writes one of the canonical names in "
             f"`title_vocabulary.json` ({f['n_network']} network and {f['n_region']} region titles), so it never invents a title."]
    if rel.repo == CV_REPO:  # only the first point concerns the fold checkpoints
        bullets = bullets[:1]
    lines = [f"## {TAG}", "", "Weights of the revised NeuroVLM paper. What changed:", "", *bullets,
             "", HERE[rel.repo], "", *files_table(rel), "",
             f"Provenance: `scripts/outputs/runs/final` of NeuroVLM (`scripts/pipeline/train_pipeline.py`, finished {f['finished']}), "
             f"training code at git `{f['commit'][:7]}` with {f['modified']} modified files, inputs pinned by "
             f"`scripts/config/hf_manifest.json` (sha256 `{f['manifest'][:12]}`). `scripts/README.md` records the later runs "
             "from scratch that gave the same bytes."]
    if earlier:
        lines += ["", f"Earlier weights: revision `{earlier}`."]
    return "\n".join(lines) + "\n"


def neuroqwen_section(rel: Release, s: dict, earlier: str | None) -> str:
    ef, prov = s["eval_filter"], s["provenance"]
    gpu = ", ".join(g.get("name", "?") for g in prov.get("gpu") or []) if str(s.get("device")).startswith("cuda") else "the CPU"
    lines = [f"## {TAG}", "", "Retrained for the revised NeuroVLM paper (`scripts/models/00_neuroqwen_dapt.py`). What changed:", "",
             f"- **Evaluation papers removed.** The validation and test papers of NeuroVLM and its NeuroVault evaluation papers "
             f"are left out of training ({ef['removed']:,} of {ef['corpus_documents']:,} records). {s['train_documents']:,} "
             "documents were trained on.",
             "- **Training loop.** The recipe above (one epoch, lr 5e-6, cosine schedule, 16 sequences of 1,024 tokens per step) "
             "runs in a plain loop instead of TRL `SFTTrainer`, with fp32 master weights under bf16 autocast (bf16 export) and "
             f"deterministic kernels. Final validation loss {s['final_val_loss']:.4f}, {s['train_seconds'] / 3600:.1f} h on {gpu}.",
             "- **Output head.** `config.json` now says `tie_word_embeddings: false`. The earlier revision said `true` but stored a "
             "separately trained `lm_head`, which transformers 4.57 replaced with the input embeddings on load. The trained head "
             "is now kept in every transformers version.",
             "", *files_table(rel), "",
             f"Provenance: `00_neuroqwen_dapt.summary.json` of the run ({s['ts']}), code at git `{(prov['git'].get('commit') or '?')[:7]}`, "
             f"inputs pinned by `scripts/config/hf_manifest.json` (sha256 `{prov['hf_manifest']['sha256'][:12]}`), corpus sha256 "
             f"`{s['corpus']['sha256'][:12]}`."]
    if earlier:
        lines += ["", f"Earlier weights: revision `{earlier}`."]
    return "\n".join(lines) + "\n"


def corpus_section(rel: Release) -> str:
    return "\n".join([f"## {TAG}", "", "First release, for the revised NeuroVLM paper.", "", *files_table(rel)]) + "\n"


def scrub_paths(obj):
    """Absolute local paths in a config -> the part from scripts/ on (else the file name)."""
    if isinstance(obj, dict):
        return {k: scrub_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub_paths(v) for v in obj]
    if isinstance(obj, str) and obj.startswith("/"):
        return obj[obj.index("scripts/"):] if "/scripts/" in obj else Path(obj).name
    return obj


def card_note(base: str, head: str | None) -> str:
    if head is None:
        return "new card"
    return f"'{TAG}' section {'replaced' if f'## {TAG}' in base.splitlines() else 'appended'}, rest of the card kept"


def write_staged(stage: Path, repo: str, path: str, text: str) -> Path:
    out = stage / repo.split("/")[1] / path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return out


# ---------------------------------------------------------------------------------------------------- plans
def run_facts(run_dir: Path) -> dict:
    prov, cal, vocab = (read_json(run_dir / n) for n in ("provenance.json", "map_calibration.json", "title_vocabulary.json"))
    cfg = read_json(run_dir / "neuro_qformer_canonical" / "config.json")
    return {"commit": prov["git"]["commit"] or "?", "modified": prov["git"]["modified_files"], "manifest": prov["hf_manifest"]["sha256"],
            "finished": prov["finished"], "seed": read_json(run_dir / "stage14_seed_selection.json")["selected_seed"], "head": cal["head"], "adapter": cal["adapter"],
            "n_network": len(vocab["titles"]["network"]), "n_region": len(vocab["titles"]["region"]),
            "mlp_params": cfg["projector_config"]["params"]}


def commit_text(label: str, facts: dict, rel: Release, release_commit: str) -> tuple[str, str]:
    title = f"{TAG}: {label} (git {facts['commit'][:7]}, hf_manifest {facts['manifest'][:12]}, {facts['finished'][:10]})"
    body = [f"Run: scripts/outputs/runs/final (scripts/pipeline/train_pipeline.py), finished {facts['finished']}.",
            f"Training code: git {facts['commit']} with {facts['modified']} modified files (runs/final/provenance.json).",
            f"Inputs: scripts/config/hf_manifest.json sha256 {facts['manifest']}.",
            f"Published by scripts/pipeline/push_hf_release.py at git {release_commit}.", ""]
    body += [f"{p}  {rel.sha[p]}" for p in rel.files if p != "README.md"]
    return title, "\n".join(body)


def model_releases(hub: Hub, run_dir: Path, stage: Path, manifest: dict, release_commit: str) -> list[Release]:
    facts = run_facts(run_dir)
    out = []
    for repo, files in MODEL_FILES.items():
        head = hub.head(repo, "model")
        rel = Release(repo, "model", "", "", head)
        for path, src in files.items():
            src = run_dir / src
            if path == "config.json":
                cfg = read_json(src)
                clean = scrub_paths(cfg)
                if clean != cfg:
                    rel.add(path, write_staged(stage, repo, path, json.dumps(clean, indent=2) + "\n"), src, "local paths cut to scripts/...")
                    continue
            rel.add(path, src, src)
        if repo == CV_REPO:
            rows = re.search(r"rows: (\d+)", (run_dir / "logs" / "20.log").read_text())
            base = CV_CARD.format(rows=f"{int(rows.group(1)):,}" if rows else "selected", repo=repo) if head is None else hub.readme(repo, "model", head, manifest)
        else:
            base = hub.readme(repo, "model", head, manifest)
        card = put_section(base, model_section(rel, facts, EARLIER.get(repo)))
        rel.add("README.md", write_staged(stage, repo, "README.md", card), note=card_note(base, head))
        rel.title, rel.body = commit_text(LABEL[repo], facts, rel, release_commit)
        out.append(rel)
    return out


def neuroqwen_release(hub: Hub, qdir: Path, stage: Path, manifest: dict, release_commit: str, blockers: list[str]) -> Release:
    export = qdir if (qdir / "model.safetensors").exists() else qdir / "NeuroQwen3-0.6B"
    summary_path, check_path = export.parent / "00_neuroqwen_dapt.summary.json", export.parent / "export_check.json"
    for p in (export / "model.safetensors", export / "config.json", summary_path, check_path):
        if not p.exists():
            raise SystemExit(f"--neuroqwen-dir: {p} missing (expected the neuroqwen/ folder of 00_neuroqwen_dapt.py)")
    s, check, cfg = read_json(summary_path), read_json(check_path), read_json(export / "config.json")
    from safetensors import safe_open

    with safe_open(str(export / "model.safetensors"), "pt") as f:
        keys = set(f.keys())
    problems = [msg for bad, msg in [
        (s.get("smoke"), "is a smoke run"),
        (cfg.get("tie_word_embeddings") is not False, "config.json does not say tie_word_embeddings false"),
        ("lm_head.weight" not in keys, "model.safetensors holds no separate lm_head.weight"),
        (check.get("lm_head_shares_embedding") or not check.get("lm_head_equals_trained_head"), "export_check.json: lm_head not the trained untied head"),
        (check.get("generated_tokens") != 10 or not check.get("tokenizer_matches_base"), "export_check.json: generation or tokenizer check failed"),
        (not s.get("eval_filter", {}).get("applied"), "trained without the evaluation filter"),
        (not s.get("corpus", {}).get("pinned"), "trained on an unpinned corpus"),
        ((export / "model.safetensors").stat().st_mtime > check_path.stat().st_mtime, "model.safetensors is newer than export_check.json"),
    ] if bad]
    blockers += [f"NeuroQwen export {export}: {m}" for m in problems]
    head = hub.head(QWEN_REPO, "model")
    rel = Release(QWEN_REPO, "model", "", "", head)
    for p in sorted(export.iterdir()):
        if p.is_file() and not p.name.startswith(".") and p.name != "README.md":
            rel.add(p.name, p)
    base = hub.readme(QWEN_REPO, "model", head, manifest)
    rel.add("README.md", write_staged(stage, QWEN_REPO, "README.md", put_section(base, neuroqwen_section(rel, s, EARLIER[QWEN_REPO]))),
            note=card_note(base, head))
    prov = s["provenance"]
    rel.title = (f"{TAG}: NeuroQwen3-0.6B retrained without the evaluation papers (git {(prov['git'].get('commit') or '?')[:7]}, "
                 f"hf_manifest {prov['hf_manifest']['sha256'][:12]}, {s['ts'][:10]})")
    rel.body = "\n".join([f"Run: {summary_path.parent.relative_to(REPO_ROOT) if summary_path.is_relative_to(REPO_ROOT) else summary_path.parent}"
                          f" (scripts/models/00_neuroqwen_dapt.py), finished {s['ts']}.",
                          f"Code: git {prov['git'].get('commit')} with {prov['git'].get('modified_files')} modified files.",
                          f"Inputs: scripts/config/hf_manifest.json sha256 {prov['hf_manifest']['sha256']}, corpus sha256 {s['corpus']['sha256']}.",
                          "tie_word_embeddings false: the trained lm_head is stored and loaded untied (export_check.json passed).",
                          f"Published by scripts/pipeline/push_hf_release.py at git {release_commit}.", ""]
                         + [f"{p}  {rel.sha[p]}" for p in rel.files if p != "README.md"])
    return rel


def corpus_release(hub: Hub, corpus: Path, stage: Path, manifest: dict, release_commit: str) -> Release:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    t = pq.read_table(corpus, columns=["year", "journal", "abstract"])
    facts = {"rows": t.num_rows, "y0": pc.min(t["year"]).as_py(), "y1": pc.max(t["year"]).as_py(),
             "journals": pc.count_distinct(t["journal"]).as_py(),
             "empty": t.num_rows - pc.sum(pc.fill_null(pc.greater(pc.utf8_length(t["abstract"]), 0), False)).as_py()}
    source = next(e for e in manifest["not_on_hub"] if e["name"] == "pubmed_neuroimaging.parquet")
    head = hub.head(CORPUS_REPO, "dataset")
    rel = Release(CORPUS_REPO, "dataset", "", "", head)
    rel.add(corpus.name, corpus, corpus)
    base = CORPUS_CARD.format(path=corpus.name, query=QUERY, source=source["sha256"], **facts) if head is None \
        else hub.readme(CORPUS_REPO, "dataset", head, manifest)
    rel.add("README.md", write_staged(stage, CORPUS_REPO, "README.md", put_section(base, corpus_section(rel))),
            note=card_note(base, head))
    msha = sha256_file(HF_MANIFEST)
    rel.title = f"{TAG}: PubMed neuroimaging corpus, {facts['rows']:,} records, e-mail addresses removed (git {release_commit[:7]}, hf_manifest {msha[:12]})"
    rel.body = "\n".join([f"Source: pubmed_neuroimaging.parquet (sha256 {source['sha256']}, hf_manifest.json not_on_hub), the "
                          "NeuroQwen3-0.6B training corpus from efficient_ai project/scrape.py.",
                          "Redaction: e-mail addresses replaced with [email removed]. No other change.",
                          f"Published by scripts/pipeline/push_hf_release.py at git {release_commit}, hf_manifest.json sha256 {msha}.", "",
                          f"{corpus.name}  {rel.sha[corpus.name]}"])
    return rel


# ---------------------------------------------------------------------------------------------------- checks
def lock_check(releases: list[Release], run_dir: Path) -> list[str]:
    bad = []
    for rel in releases:
        for path, src in rel.sources.items():
            key, digest = lock_key(src, run_dir), sha256_file(src)
            if EXPECTED_SHA256.get(key) != digest:
                bad.append(f"{key}: sha256 {digest[:12]} is not the recorded {str(EXPECTED_SHA256.get(key))[:12]}")
    return bad


def provenance_check(run_dir: Path) -> list[str]:
    """EXPECTED_SHA256 against the hashes the run itself recorded: the scripts/README.md table, banks and calibration sources."""
    bad = []
    for path, digest in re.findall(r"^\| `([^`]+)`[^|]*\| `([0-9a-f]{64})` \|", (SCRIPTS_DIR / "README.md").read_text(), re.M):
        if path in EXPECTED_SHA256 and EXPECTED_SHA256[path] != digest:
            bad.append(f"{path}: EXPECTED_SHA256 differs from scripts/README.md ({digest[:12]})")
    banks = read_json(run_dir / "title_vocabulary.json")["source"]["banks_sha256"]
    if EXPECTED_SHA256.get("canonical_semantic_banks.pt") != banks:
        bad.append(f"canonical_semantic_banks.pt: title_vocabulary.json was built from banks {banks[:12]}")
    for name, rec in read_json(run_dir / "map_calibration.json")["source"]["models"].items():
        fname, digest = rec.rsplit(":", 1)
        if sha256_file(run_dir / fname) != digest:
            bad.append(f"{fname}: not the model map_calibration.json was fitted on ({name})")
    return bad


def email_scan(releases: list[Release]) -> tuple[list[str], int]:
    """E-mail addresses in every parquet / csv / tsv / json / jsonl / md / txt upload (tokenizer vocabularies skipped)."""
    hits, n = [], 0
    rx = re.compile(EMAIL)
    for rel in releases:
        for path, f in rel.files.items():
            suffix = Path(path).suffix
            if Path(path).name in TOKENIZER_FILES or suffix not in (".parquet", ".csv", ".tsv", ".json", ".jsonl", ".md", ".txt"):
                continue
            n += 1
            if suffix == ".parquet":
                import pyarrow as pa
                import pyarrow.compute as pc
                import pyarrow.parquet as pq

                pf = pq.ParquetFile(f)
                for col in pf.schema_arrow:
                    if not (pa.types.is_string(col.type) or pa.types.is_large_string(col.type)
                            or (pa.types.is_list(col.type) and pa.types.is_string(col.type.value_type))):
                        continue
                    for g in range(pf.num_row_groups):
                        arr = pf.read_row_group(g, columns=[col.name]).column(0)
                        arr = pc.list_flatten(arr) if pa.types.is_list(col.type) else arr
                        k = pc.sum(pc.fill_null(pc.match_substring_regex(arr, EMAIL), False)).as_py() or 0
                        if k:
                            hits.append(f"{rel.repo}/{path}: {k} values of column {col.name} (row group {g})")
            else:
                k = len(rx.findall(f.read_text(errors="replace")))
                if k:
                    hits.append(f"{rel.repo}/{path}: {k} e-mail addresses")
    return hits, n


def loader_check(releases: list[Release]) -> tuple[list[str], list[str]]:
    """Load every planned weight file with the installed library's loaders (neurovlm.resources.loaders), reading the
    planned files instead of the Hub."""
    import neurovlm.resources.loaders as L
    from safetensors.torch import load_file

    local = {(r.repo, p): str(f) for r in releases for p, f in r.files.items()}
    orig = L._download_from_hf
    L._download_from_hf = lambda repo_id, filename, repo_type="dataset": local.get((repo_id, filename)) or orig(repo_id, filename, repo_type=repo_type)
    calls = {"neurovlm/NeuroAutoEncoder": [("_load_autoencoder", L._load_autoencoder)],
             "neurovlm/ProjectionHeads": [(f"_{h}", getattr(L, f"_{h}")) for h in HEADS],
             "neurovlm/NeuroAdapter": [("_load_neuro_adapter", L._load_neuro_adapter)],
             "neurovlm/NeuroQformer": [("_load_neuro_qformer(qformer_variant='canonical')", lambda: L._load_neuro_qformer(qformer_variant="canonical"))],
             "neurovlm/NeuroQformer-PubMed": [("_load_neuro_qformer(qformer_variant='pubmed')", lambda: L._load_neuro_qformer(qformer_variant="pubmed"))]}
    ok, bad = [], []
    try:
        for rel in releases:
            for name, fn in calls.get(rel.repo, []):
                try:
                    fn()
                    ok.append(f"{rel.repo}: {name}")
                except Exception as exc:
                    bad.append(f"{rel.repo}: {name} fails: {type(exc).__name__}: {str(exc).splitlines()[0][:160]}")
            if rel.repo == CV_REPO:
                for path, f in rel.files.items():
                    if path.endswith(".safetensors"):
                        module = (L.NeuroAutoEncoder(seed=0, out="logit") if path.startswith("autoencoder") else
                                  L.ProjHead(384, 384, 384) if path.startswith("proj_head_image") else L.ProjHead(768, 512, 384))
                        try:
                            module.load_state_dict(load_file(str(f)), strict=True)
                        except Exception as exc:
                            bad.append(f"{rel.repo}/{path}: {type(exc).__name__}: {str(exc).splitlines()[0][:160]}")
                ok.append(f"{rel.repo}: 20 fold checkpoints into NeuroAutoEncoder / ProjHead")
    finally:
        L._download_from_hf = orig
    return ok, bad


# ---------------------------------------------------------------------------------------------------- manifest
def dump_manifest(m: dict) -> str:
    """hf_manifest.json in its layout: one line per entry."""
    out, keys = ["{"], list(m)
    for i, k in enumerate(keys):
        end = "," if i < len(keys) - 1 else ""
        if isinstance(m[k], list):
            out += [f"  {json.dumps(k)}: [", ",\n".join("    " + json.dumps(e, ensure_ascii=False) for e in m[k]), f"  ]{end}"]
        else:
            out.append(f"  {json.dumps(k)}: {json.dumps(m[k], ensure_ascii=False)}{end}")
    return "\n".join(out) + "\n}\n"


def repin(m: dict, rel: Release, revision: str, files: dict[str, tuple[str, int]]) -> None:
    """Point every manifest entry of rel.repo at revision with the given (sha256, size); add entries for new files."""
    entries = [e for e in m["files"] if e["repo"] == rel.repo]
    groups = next((e["groups"] for e in entries if e["path"].endswith(".safetensors")), entries[0]["groups"] if entries else [NEW_GROUP])
    for e in entries:
        e["revision"] = revision
        if e["path"] in files:
            e["sha256"], e["size"] = files[e["path"]]
    have = {e["path"] for e in entries}
    new = [{"repo": rel.repo, "repo_type": rel.repo_type, "path": p, "revision": revision, "ref": "main", "sha256": files[p][0],
            "size": files[p][1], "local_name": None, "hf_cache": True, "groups": list(groups), "consumers": []}
           for p in rel.files if p not in have and p in files]
    if new:
        same = [i for i, e in enumerate(m["files"]) if e["repo"] == rel.repo]
        kin = [i for i, e in enumerate(m["files"]) if e["repo"].startswith("neurovlm/") and e["repo_type"] == rel.repo_type]
        at = (same or kin or [len(m["files"]) - 1])[-1] + 1
        m["files"][at:at] = new
    note = " Revision 2026-10 repos re-pinned by scripts/pipeline/push_hf_release.py."
    if note not in m["about"]:
        m["about"] += note


def show_diff(old: str, new: str) -> None:
    """Field-level diff of two manifests: changed and added entries of files."""
    a, b = json.loads(old), json.loads(new)
    before = {(e["repo"], e["path"]): e for e in a["files"]}
    lines = [f"  ~ about: + {b['about'][len(a['about']):].strip()!r}"] if a["about"] != b["about"] else []
    for e in b["files"]:
        o = before.get((e["repo"], e["path"]))
        if o is None:
            lines.append(f"  + {e['repo']}/{e['path']}  revision {e['revision'][:12]}  sha256 {e['sha256'][:12]}  size {e['size']}  groups {','.join(e['groups'])}")
            continue
        ch = [f"{k} {str(o[k])[:12]} -> {str(e[k])[:12]}" for k in ("revision", "sha256", "size") if o[k] != e[k]]
        if ch:
            lines.append(f"  ~ {e['repo']}/{e['path']}  " + ", ".join(ch))
    print("\n".join(lines) if lines else "  (no change)")
    print(f"  {sum(l.startswith('  ~ neurovlm') for l in lines)} entries changed, {sum(l.startswith('  +') for l in lines)} added")


def remote_files(hub: Hub, rel: Release, revision: str, manifest: dict) -> tuple[dict[str, tuple[str, int]], list[str]]:
    """(sha256, size) on the Hub at revision of every planned file and every manifest entry of the repo; errors when a
    planned file differs from the local one or an unchanged file differs from its pin."""
    from huggingface_hub import hf_hub_download

    info = (hub.api.model_info if rel.repo_type == "model" else hub.api.dataset_info)(rel.repo, revision=revision, files_metadata=True)
    sib = {s.rfilename: s for s in info.siblings}
    pinned = {e["path"]: e for e in manifest["files"] if e["repo"] == rel.repo}
    out, errors = {}, []
    for path in list(rel.files) + [p for p in pinned if p not in rel.files]:
        s = sib.get(path)
        if s is None:
            errors.append(f"{rel.repo}/{path}: not at {revision[:8]}")
            continue
        if s.lfs is not None:
            digest, size = s.lfs.sha256, s.lfs.size
        elif path in rel.files and s.blob_id == git_blob_sha1(rel.files[path]):
            digest, size = rel.sha[path], rel.files[path].stat().st_size
        else:
            p = Path(hf_hub_download(rel.repo, path, repo_type=rel.repo_type, revision=revision))
            digest, size = sha256_file(p), p.stat().st_size
        want = rel.sha[path] if path in rel.files else pinned[path]["sha256"]
        if digest != want:
            errors.append(f"{rel.repo}/{path}@{revision[:8]}: sha256 {digest[:12]}, expected {want[:12]}")
        out[path] = (digest, size)
    return out, errors


# ---------------------------------------------------------------------------------------------------- main
def print_plan(releases: list[Release]) -> None:
    total = 0
    for rel in releases:
        state = "new repo" if rel.head is None else f"head {rel.head[:8]}"
        print(f"\n== {rel.repo} ({rel.repo_type}, {state})")
        print(f"   commit: {rel.title}")
        for line in rel.body.splitlines()[:4]:
            print(f"           {line}")
        for path, f in rel.files.items():
            n = f.stat().st_size
            total += n
            local = rel.sources.get(path, f)
            local = local.relative_to(REPO_ROOT) if local.is_relative_to(REPO_ROOT) else local
            note = f"  [generated: {rel.notes[path]}]" if path in rel.notes else ""
            print(f"   {path:<44} {size_str(n):>10}  {rel.sha[path]}  {local}{note}")
    n_files = sum(len(r.files) for r in releases)
    print(f"\nSummary: {len(releases)} repos, {n_files} files, {total / 1e6:,.1f} MB")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", choices=["models", "corpus", "neuroqwen", "all"], default="all")
    p.add_argument("--execute", action="store_true", help="Upload (default: dry run, reads the Hub only).")
    p.add_argument("--yes", action="store_true", help="Confirm publishing the PubMed corpus (needed with --execute).")
    p.add_argument("--update-manifest", action="store_true", help="After --execute, re-pin the pushed repos in hf_manifest.json.")
    p.add_argument("--pin-only", action="store_true", help="No upload: check the current Hub heads hold the planned files and re-pin them.")
    p.add_argument("--neuroqwen-dir", type=Path, help="neuroqwen/ (or neuroqwen/NeuroQwen3-0.6B/) of 00_neuroqwen_dapt.py.")
    p.add_argument("--run-dir", type=Path, default=RUN_DIR)
    p.add_argument("--corpus", type=Path, default=CORPUS)
    p.add_argument("--manifest", type=Path, default=HF_MANIFEST)
    p.add_argument("--private", action="store_true", help="Create the new repos as private.")
    p.add_argument("--allow-loader-mismatch", action="store_true", help="Push files the installed neurovlm loaders cannot load.")
    p.add_argument("--keep-staging", type=Path, help="Write the generated cards and configs here and keep them.")
    p.add_argument("--print-lock", action="store_true", help="Print EXPECTED_SHA256 for the current run dir and corpus, then exit.")
    args = p.parse_args(argv)
    run_dir = args.run_dir.resolve()

    if args.print_lock:
        paths = sorted({run_dir / s for files in MODEL_FILES.values() for s in files.values()})
        print("EXPECTED_SHA256 = {")
        for f in [*paths, args.corpus.resolve()]:
            print(f'    "{lock_key(f, run_dir)}": "{sha256_file(f)}",')
        print("}")
        return 0
    if args.only == "neuroqwen" and args.neuroqwen_dir is None:
        p.error("--only neuroqwen needs --neuroqwen-dir")

    stage = args.keep_staging or Path(tempfile.mkdtemp(prefix="push_hf_release_"))
    stage.mkdir(parents=True, exist_ok=True)
    try:
        return run(args, run_dir, stage)
    finally:
        if not args.keep_staging:
            shutil.rmtree(stage, ignore_errors=True)


def run(args, run_dir: Path, stage: Path) -> int:
    manifest_text = args.manifest.read_text()
    manifest = json.loads(manifest_text)
    hub = Hub()
    blockers: list[str] = []
    dirty = git("status", "--porcelain").splitlines()
    release_commit = git("rev-parse", "HEAD") + ("-dirty" if dirty else "")

    releases: list[Release] = []
    if args.only in ("models", "all"):
        releases += model_releases(hub, run_dir, stage, manifest, release_commit)
    if args.only in ("neuroqwen", "all"):
        if args.neuroqwen_dir is not None:
            releases.append(neuroqwen_release(hub, args.neuroqwen_dir.resolve(), stage, manifest, release_commit, blockers))
        else:
            print("neuroqwen: skipped (no --neuroqwen-dir; run it after the retrain)")
    if args.only in ("corpus", "all"):
        releases.append(corpus_release(hub, args.corpus.resolve(), stage, manifest, release_commit))
    print_plan(releases)
    if args.keep_staging:
        print(f"generated files kept in {stage}")

    # checks
    print("\nChecks")
    if dirty:
        blockers.append(f"git status is dirty ({len(dirty)} entries, e.g. {dirty[0].strip()})")
    print(f"  git: HEAD {release_commit[:12]}, {'clean' if not dirty else f'{len(dirty)} uncommitted entries'}")
    models = any(r.repo in MODEL_FILES for r in releases)
    bad = lock_check(releases, run_dir) + (provenance_check(run_dir) if models else [])
    n_src = sum(len(r.sources) for r in releases)
    if n_src:
        print(f"  sha256 vs provenance: {'all' if not bad else 'FAIL,'} {n_src} run/corpus files {'match' if not bad else 'checked against'} EXPECTED_SHA256"
              + (", which agrees with scripts/README.md, title_vocabulary.json and map_calibration.json" if models and not bad else ""))
    for b in bad:
        print(f"    {b}")
    blockers += bad
    hits, n_scanned = email_scan(releases)
    print(f"  e-mail scan: {len(hits)} hits in {n_scanned} parquet/csv/json/md files")
    blockers += [f"e-mail address found: {h}" for h in hits]
    if models:
        ok, fails = loader_check(releases)
        print(f"  installed neurovlm loaders: {len(ok)} ok, {len(fails)} failing")
        for f in fails:
            print(f"    {f}")
        if fails and not args.allow_loader_mismatch:
            blockers.append(f"{len(fails)} planned weight files do not load with the installed neurovlm loaders "
                            "(update src/neurovlm/resources/loaders.py or pass --allow-loader-mismatch)")
    if any(r.repo == CORPUS_REPO for r in releases) and not args.yes:
        blockers.append("the corpus upload publishes 1.23 M PubMed records: pass --yes")
    try:
        who = hub.api.whoami()
        orgs, role = [o.get("name") for o in who.get("orgs", [])], who.get("auth", {}).get("accessToken", {}).get("role")
        print(f"  Hub token: {who.get('name')} (orgs {', '.join(orgs) or '-'}, role {role})")
        if "neurovlm" not in orgs or role == "read":
            blockers.append("the Hub token cannot write to the neurovlm org")
    except Exception as exc:
        blockers.append(f"no Hub token ({type(exc).__name__})")
    blockers += [f"Hub not readable: {e}" for e in hub.errors]
    for b in blockers:
        print(f"  BLOCKS --execute: {b}")

    if args.pin_only:
        return pin(hub, releases, args, manifest_text, manifest)

    preview = json.loads(manifest_text)
    for rel in releases:
        files = {pth: (rel.sha[pth], f.stat().st_size) for pth, f in rel.files.items()}
        repin(preview, rel, "<new>", files)
    if not args.execute:
        print(f"\nhf_manifest.json after --execute --update-manifest (new commit shas filled in then):")
        show_diff(manifest_text, dump_manifest(preview))
        print(f"\nDry run: nothing uploaded. {'--execute would refuse (see BLOCKS).' if blockers else '--execute would upload the files above.'}")
        return 1 if blockers else 0
    if blockers:
        print("\nRefusing to upload.")
        return 1

    from huggingface_hub import CommitOperationAdd

    pushed: dict[str, str] = {}
    try:
        for rel in releases:
            parent = rel.head
            if rel.head is None:
                hub.api.create_repo(rel.repo, repo_type=rel.repo_type, private=args.private)
                parent = hub.head(rel.repo, rel.repo_type)
            ops = [CommitOperationAdd(path_in_repo=pth, path_or_fileobj=str(f)) for pth, f in rel.files.items()]
            print(f"\npushing {len(ops)} files to {rel.repo} ...", flush=True)
            info = hub.api.create_commit(rel.repo, ops, commit_message=rel.title, commit_description=rel.body,
                                         repo_type=rel.repo_type, parent_commit=parent)
            pushed[rel.repo] = info.oid
            print(f"  {rel.repo}: commit {info.oid} {info.commit_url}")
    finally:
        done = [r for r in releases if r.repo in pushed]
        status = finish(hub, done, pushed, args, manifest_text, manifest)
    return 1 if len(done) < len(releases) else status


def finish(hub: Hub, releases: list[Release], revisions: dict[str, str], args, manifest_text: str, manifest: dict) -> int:
    """Read every file back at its new revision, then re-pin (with --update-manifest / --pin-only) or print the diff."""
    if not releases:
        print("\nnothing pushed or pinned, hf_manifest.json unchanged")
        return 1
    new, errors = json.loads(manifest_text), []
    for rel in releases:
        files, err = remote_files(hub, rel, revisions[rel.repo], manifest)
        errors += err
        repin(new, rel, revisions[rel.repo], files)
    print("\nHub check: " + ("every file has the local sha256" if not errors else f"{len(errors)} problems"))
    for e in errors:
        print(f"  {e}")
    text = dump_manifest(new)
    print("hf_manifest.json changes:")
    show_diff(manifest_text, text)
    if errors:
        print("hf_manifest.json not written.")
        return 1
    if args.update_manifest or args.pin_only:
        args.manifest.write_text(text)
        print(f"wrote {args.manifest}. Next: commit it, then scripts/data/fetch_hf.py re-pins the HF cache.")
    else:
        print("hf_manifest.json not written (pass --update-manifest, or rerun with --pin-only).")
    return 0


def pin(hub: Hub, releases: list[Release], args, manifest_text: str, manifest: dict) -> int:
    """--pin-only: re-pin the planned repos that are on the Hub, at their current heads."""
    heads = {r.repo: hub.head(r.repo, r.repo_type) for r in releases}
    missing = [r for r, h in heads.items() if h in (None, "?")]
    if missing:
        print(f"\n--pin-only: not on the Hub, skipped: {', '.join(missing)}")
    found = [r for r in releases if r.repo not in missing]
    print("--pin-only: " + (", ".join(f"{r.repo} @ {heads[r.repo][:8]}" for r in found) or "nothing to pin"))
    return finish(hub, found, heads, args, manifest_text, manifest)


if __name__ == "__main__":
    sys.exit(main())
