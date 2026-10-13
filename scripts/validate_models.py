#!/usr/bin/env python
"""Score retrained NeuroVLM components with the paper's metrics and compare them with the released checkpoints.

    python scripts/validate_models.py --released                        # reference numbers for the HF checkpoints
    python scripts/validate_models.py --run scripts/outputs/runs/final  # retrained run vs the reference
    python scripts/validate_models.py --run ... --components ae,heads --max-samples 64   # smoke

Components: ae (autoencoder), heads (InfoNCE retrieval), networks (network classification), text_to_brain
(proj_head_text_mse + decoder), adapter (NeuroAdapter), qformer / qformer_pubmed (stage 2 / stage 1 brain-to-text
projector, MLP or Q-Former), cv (5-fold CV models). The leakage-audit items in scripts/eval_exclusions/ are dropped
unless --no-exclusions. Writes scripts/outputs/validation/<name>/{metrics.json,report.md}; the released reference is
computed once into validation/released[_excl][_n<N>]/ and reused. Released weights, data and the scoring models (BERTScore,
MiniLM) are read at the revisions pinned in scripts/hf_manifest.json; metrics.json["provenance"] records them with the
software, GPU, determinism settings and git commit. Captions are generated with Latin-script tokens only
(_common_models.latin_script_suppress_tokens) and, after a [NETWORK] / [REGION] steering token, with a title from the
run's stage-14 canonical names (_common_models.title_vocabulary / install_title_constraint; --free-titles turns that off);
validity_frac_non_latin and validity_frac_noncanonical_title (stage 2) must be 0.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))             # scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))  # scripts/models/
from common import (DEFAULT_OUT_ROOT, EVAL_EXCLUSIONS_DIR, apply_neurovault_exclusions, apply_pubmed_exclusions, data_dir,  # noqa: E402
                    get_logger, hf_pinned, hf_pinned_dir, hf_split_pmids, provenance, read_json, select_device, setup_torch, timed,
                    write_json)
from _common_models import (MAP_CALIBRATION_DEFAULT, MAP_CALIBRATION_RULES, RUNS_ROOT, TITLE_LEVELS, calibrate_maps,  # noqa: E402
                            caption_char_ok, encode_in_batches, install_title_constraint, latin_script_suppress_tokens,
                            load_module_or_state, load_projector, load_torch, map_calibration, noncanonical_titles, package_projector_kind,
                            released_model, released_qformer, strip_steering_tags, title_constraint_note, title_vocabulary)

log = get_logger("validate_models")
VALIDATION_ROOT = DEFAULT_OUT_ROOT / "validation"
ALL_COMPONENTS = ("ae", "heads", "networks", "text_to_brain", "adapter", "qformer", "qformer_pubmed", "cv")

# values reported in paper/sn-article.tex
PAPER = {
    "ae": {"pubmed_voxel_auroc": 0.975, "pubmed_voxel_auroc_figure": 0.94},
    "heads": {"pubmed_mean_auc": 0.81, "pubmed_cv_mean_auc": 0.82, "neurovault_auc_a": 0.78, "neurovault_auc_b": 0.85},
    "qformer": {"bertscore_f1_matched": 0.56, "bertscore_f1_null": 0.52},
    "text_to_brain": {"dice_matched": 0.47, "dice_null": 0.40},
    "adapter": {"dice_matched": 0.47, "dice_null": 0.40},
}

# brain-to-text metrics of both projectors
B2T_PRIMARY = [("recall_auc", "auc", True), ("contrastive_sim_matched", "cos", True), ("bertscore_f1_matched", "cos", True),
               ("semantic_sim_matched", "cos", True), ("validity_frac_degenerate", "frac", False),
               ("validity_frac_non_latin", "frac", False)]
# stage 2: share of [NETWORK] / [REGION] captions whose title is not a stage-14 canonical name (0 unless --free-titles)
B2T_CANONICAL_PRIMARY = B2T_PRIMARY + [("validity_frac_noncanonical_title", "frac", False)]
MUST_BE_ZERO = {"validity_frac_non_latin", "validity_frac_noncanonical_title"}  # FAIL whenever > 0, whatever the reference
# (metric, tolerance kind, higher is better) that decide PASS/FAIL
PRIMARY = {
    "ae": [("pubmed_voxel_auroc", "auc", True), ("pubmed_bpp_gain_pct_median", "bpp", True),
           ("neurovault_voxel_auroc", "auc", True), ("networks_voxel_auroc", "auc", True)],
    "heads": [("pubmed_t2i_auc", "auc", True), ("pubmed_i2t_auc", "auc", True), ("pubmed_mean_recall@10", "acc", True),
              ("pubmed_mean_mrr", "acc", True), ("neurovault_t2i_auc", "auc", True), ("neurovault_i2t_auc", "auc", True)],
    "networks": [("accuracy", "acc", True), ("macro_auc", "auc", True)],
    "text_to_brain": [("dice_matched", "cos", True), ("pearson_matched", "cos", True), ("dice90_vs_raw_matched", "cos", True),
                      ("recall_auc", "auc", True), ("contrastive_sim_matched", "cos", True)],
    "adapter": [("recall_auc", "auc", True), ("contrastive_sim_matched", "cos", True), ("dice_matched", "cos", True),
                ("pearson_matched", "cos", True)],
    "qformer": B2T_CANONICAL_PRIMARY,
    "qformer_pubmed": B2T_PRIMARY,
    "cv": [("contrastive_mean_auc", "auc", True), ("t2b_generated_mean_auc", "auc", True)],
}

NETWORK_TEXT_LABELS = {  # figures/13 cell 4
    "Language": ("Language network (LAN; perisylvian language network; frontotemporal language system) [SEP] "
                 "Primary regions: left inferior frontal gyrus (Broca’s complex), posterior superior/middle temporal gyrus, "
                 "temporoparietal junction/angular gyrus. "
                 "Function: speech/text comprehension and production—semantics, syntax, phonology, and sentence-level integration."),
    "Auditory": ("Auditory network (AUD; auditory cortex network) [SEP] "
                 "Primary regions: Heschl’s gyrus (primary auditory cortex), planum temporale, superior temporal gyrus. "
                 "Function: acoustic feature analysis (pitch/timbre/timing), auditory scene analysis, early speech-sound encoding, "
                 "and detection of salient sounds."),
    "Default Mode": ("Default mode network (DMN; default network; default state network) [SEP] "
                     "Primary regions: medial prefrontal cortex, posterior cingulate/precuneus, angular gyrus (lateral parietal). "
                     "Function: internally oriented cognition—autobiographical/episodic memory, self-referential thought, future simulation, "
                     "social inference, and narrative/semantic integration."),
    "Frontoparietal Control": ("Frontoparietal control network (FPCN; frontoparietal network/FPN; central executive network/CEN) [SEP] "
                               "Primary regions: dorsolateral/rostrolateral prefrontal cortex (middle frontal), posterior parietal cortex "
                               "(inferior parietal/around intraparietal sulcus). "
                               "Function: goal maintenance and flexible executive control—working memory, rule implementation, planning, and rapid task switching."),
    "Attention": ("Dorsal attention network [SEP] "
                  "Primary regions: intraparietal sulcus/superior parietal lobule and frontal eye fields. "
                  "Function: goal-directed, voluntary control of visuospatial attention."),
    "Visual": ("Visual network (VIS; occipital visual network) [SEP] "
               "Primary regions: calcarine cortex/V1 (cuneus/lingual), extrastriate occipital cortex, ventral occipitotemporal visual areas. "
               "Function: visual feature processing (form/color/motion), object/scene representations, and visuospatial analysis."),
    "Motor": ("Motor network (motor/sensorimotor network; SMN) [SEP] "
              "Primary regions: primary motor cortex (precentral gyrus), supplementary motor area, premotor cortex; "
              "with primary somatosensory cortex (postcentral gyrus) for sensorimotor integration. "
              "Function: movement planning, initiation, and execution, plus proprioceptive/tactile feedback used to control actions "
              "and maintain body-state representations."),
    "Cingulo-Opercular": ("Salience network (SN; cingulo-opercular network/CON; midcingulo-insular network) [SEP] "
                          "Primary regions: anterior insula/frontal operculum, dorsal anterior cingulate/medial frontal cortex. "
                          "Function: detects behaviorally relevant internal/external events (including interoceptive/affective signals), "
                          "prioritizes attention and arousal, and drives performance monitoring and control adjustments (error/conflict/uncertainty)."),
}
NETWORK_KEYS = ["language", "auditory", "default_mode", "frontoparietal_control", "attention", "visual", "motor", "cingulo_opercular"]

FABRICATION_PATTERNS = {
    "sample_size": re.compile(r"\b[nN]\s*=\s*\d+|\b\d+\s+(?:participants|subjects|patients|healthy|volunteers|adults|children)\b"),
    "p_value": re.compile(r"\bp\s*[<=>]\s*0?\.\d+|\bp\s*[<=>]\s*\d\.\d+e-\d+", re.IGNORECASE),
    "percentage": re.compile(r"\d+(?:\.\d+)?\s*%"),
    "year": re.compile(r"\b(?:19|20)\d{2}\b"),
    "citation": re.compile(r"\bet al\.?|\[\d+\]"),
}


def deranged_indices(n, rng):
    """Permutation without fixed points."""
    if n < 2:
        return np.arange(n)
    perm = rng.permutation(n)
    fixed = np.flatnonzero(perm == np.arange(n))
    if len(fixed) > 1:
        perm[fixed] = np.roll(perm[fixed], 1)
    elif len(fixed) == 1:
        i = fixed[0]
        j = (i + 1) % n
        perm[i], perm[j] = perm[j], perm[i]
    return perm


@contextlib.contextmanager
def log_phase(name: str, **info):
    """Log the start of a phase and its duration."""
    extra = ", ".join(f"{k}={v}" for k, v in info.items())
    log.info("start %s%s", name, f" ({extra})" if extra else "")
    t0 = time.perf_counter()
    yield
    log.info("%s done in %.1fs", name, time.perf_counter() - t0)


def release_cuda_cache() -> str:
    """Hand cached CUDA blocks back to the driver and describe GPU memory ("cpu" without a GPU).

    Without this, a large model loaded after generation (deberta-xlarge-mnli) can spill into shared system memory on
    WSL2 and crawl.
    """
    import torch

    gc.collect()
    if not torch.cuda.is_available():
        return "cpu"
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    return (f"GPU free {free / 2**30:.1f}/{total / 2**30:.1f} GiB, torch allocated {torch.cuda.memory_allocated() / 2**30:.1f} GiB, "
            f"reserved {torch.cuda.memory_reserved() / 2**30:.1f} GiB")


BERTSCORE_MODELS = {"deberta": "microsoft/deberta-xlarge-mnli", "roberta": "roberta-large"}
BERTSCORE_REPOS = {"roberta-large": "FacebookAI/roberta-large"}
_BERTSCORERS: dict = {}  # model_type -> BERTScorer, kept on the CPU between calls


def cached_bertscorer(model_type: str):
    """One ``bert_score.BERTScorer`` per model type, built on the CPU from the pinned snapshot (BERTScorer takes no revision)."""
    if model_type not in _BERTSCORERS:
        from bert_score import BERTScorer
        from bert_score.utils import model2layers

        path = hf_pinned_dir(BERTSCORE_REPOS.get(model_type, model_type))
        _BERTSCORERS[model_type] = BERTScorer(model_type=str(path), num_layers=model2layers[model_type], lang="en", batch_size=32,
                                              device="cpu")
    return _BERTSCORERS[model_type]


def bertscore_f1_many(cands: list[str], refs: dict, *, model_type: str, device, desc: str = "BERTScore") -> dict:
    """``{name: F1(cands[i], refs[name][i])}`` for every named reference list, embedding each unique text once.

    Same as ``bert_score.score(cands, refs[name])[2]`` (idf off, length-sorted batches of 32, ``greedy_cos_idf``), but the
    matched and shuffled references share one embedding pass. The model sits on ``device`` only during the call.
    """
    import torch
    from torch.nn.utils.rnn import pad_sequence
    from bert_score.utils import get_bert_embedding, greedy_cos_idf

    sc = cached_bertscorer(model_type)
    tok, model, bs, all_layers = sc._tokenizer, sc._model, sc.batch_size, sc.all_layers
    device = torch.device(device)
    cands = [str(c) for c in cands]
    refs = {k: [str(r) for r in v] for k, v in refs.items()}
    try:
        mem = release_cuda_cache()
        model.to(device)
        idf = defaultdict(lambda: 1.0)
        idf[tok.sep_token_id] = 0
        idf[tok.cls_token_id] = 0
        # as bert_score's dedup_and_sort(refs + hyps), so the embedding batches match bert_score.score
        pool = [r for v in refs.values() for r in v] + cands
        sents = sorted(set(pool), key=lambda x: len(x.split(" ")), reverse=True)
        log.info("%s: %s on %s, %d unique texts for %d x %d pairs (%s)", desc, model_type, device, len(sents), len(refs), len(cands), mem)
        stats = {}
        with torch.no_grad():
            for s in range(0, len(sents), bs):
                batch = sents[s:s + bs]
                emb, mask, pidf = get_bert_embedding(batch, model, tok, idf, device=device, all_layers=all_layers)
                emb, mask, pidf = emb.cpu(), mask.cpu(), pidf.cpu()
                for i, sen in enumerate(batch):
                    n_tok = int(mask[i].sum().item())
                    stats[sen] = (emb[i, :n_tok], pidf[i, :n_tok])

            def pad_batch_stats(sen_batch):  # as in bert_score.utils.bert_cos_score_idf
                emb = [stats[x][0].to(device) for x in sen_batch]
                w = [stats[x][1].to(device) for x in sen_batch]
                lens = torch.tensor([e.size(0) for e in emb], dtype=torch.long)
                pad_mask = (torch.arange(int(lens.max()), dtype=torch.long).expand(len(lens), int(lens.max())) < lens.unsqueeze(1)).to(device)
                return pad_sequence(emb, batch_first=True, padding_value=2.0), pad_mask, pad_sequence(w, batch_first=True)

            out = {}
            for name, rlist in refs.items():
                f1 = [greedy_cos_idf(*pad_batch_stats(rlist[s:s + bs]), *pad_batch_stats(cands[s:s + bs]), all_layers)[2].cpu()
                      for s in range(0, len(cands), bs)]
                out[name] = torch.cat(f1, dim=1 if all_layers else 0)
        return out
    finally:
        model.to("cpu")
        release_cuda_cache()


class Bundle:
    """The models of a run directory (falling back to the released checkpoints), or the released ones."""

    def __init__(self, run_dir: Path | None, device, *, released: bool, free_titles: bool = False):
        self.run_dir, self.device, self.released, self.free_titles = run_dir, device, released, free_titles
        self.title_vocab: dict | None = None  # stage-14 canonical names, set with the generator
        self.sources: dict[str, str] = {}
        self.projectors: dict[str, str] = {}  # qformer_<variant> -> projector kind (qformer / mlp)
        self._cache: dict[str, object] = {}

    def _pick(self, name, filename, released_name, builder=None):
        """The run's ``filename`` if present, else the released ``released_name``."""
        if name not in self._cache:
            path = self.run_dir / filename if (self.run_dir is not None and not self.released) else None
            if path is not None and path.exists():
                model, source = load_module_or_state(path, builder=builder, device=self.device), str(path)
            else:
                model, source = released_model(released_name, self.device), f"released:{released_name}"
            self.sources[name], self._cache[name] = source, model
        return self._cache[name]

    @property
    def autoencoder(self):
        from neurovlm.models.base import NeuroAutoEncoder

        return self._pick("autoencoder", "autoencoder.pt", "autoencoder", lambda: NeuroAutoEncoder(seed=0, out="logit"))

    @property
    def text_head(self):
        from neurovlm.models import ProjHead

        return self._pick("proj_head_text_infonce", "proj_head_text_infonce.pt", "proj_head_text_infonce", lambda: ProjHead(768, 512, 384))

    @property
    def image_head(self):
        from neurovlm.models import ProjHead

        return self._pick("proj_head_image_infonce", "proj_head_image_infonce.pt", "proj_head_image_infonce", lambda: ProjHead(384, 384, 384))

    @property
    def mse_head(self):
        from neurovlm.models import ProjHead

        return self._pick("proj_head_text_mse", "proj_head_text_mse.pt", "proj_head_text_mse", lambda: ProjHead(768, 512, 384))

    @property
    def adapter(self):
        return self._pick("adapter", "adapter_anat_full.pt", "neuro_adapter")

    @property
    def specter(self):
        if "specter" not in self._cache:
            self._cache["specter"] = released_model("specter", self.device)
        return self._cache["specter"]

    def map_factor(self, kind: str) -> float:
        """Global map-calibration factor of the bundle's text-to-brain head ("head") or anatomical head ("adapter")."""
        if "map_calibration" not in self._cache:
            self._cache["map_calibration"] = map_calibration(None if self.released else self.run_dir, device=self.device, logger=log,
                                                             specter=self.specter)
        return self._cache["map_calibration"][kind]

    def qformer(self, variant: str):
        """Brain-to-text projector for ``variant`` (canonical / pubmed): the run's package (Q-Former or MLP), else the release."""
        key = f"qformer_{variant}"
        if key in self._cache:
            return self._cache[key]
        model, source, kind = None, None, "qformer"
        if not self.released and self.run_dir is not None:
            pkg = self.run_dir / ("neuro_qformer_canonical" if variant == "canonical" else "neuro_qformer_pubmed")
            if (pkg / "model.safetensors").exists():
                kind = package_projector_kind(pkg)
                model = load_projector(pkg, device=self.device).eval()
                source = str(pkg)
                log.info("%s: %s projector from %s", key, kind, pkg)
        if model is None:
            model = released_qformer(variant, self.device)
            source = f"released:neurovlm/{'NeuroQformer' if variant == 'canonical' else 'NeuroQformer-PubMed'}"
        self.sources[key] = source
        self.projectors[key] = kind
        self._cache[key] = model
        return model

    def fold(self, idx: int):
        """Fold checkpoints of the 5-fold CV (run dir, else data_dir)."""
        roots = ([self.run_dir] if (self.run_dir and not self.released) else []) + [data_dir()]
        for root in roots:
            paths = {stem: root / f"{stem}_kfold-{idx}.pt" for stem in ("autoencoder", "proj_head_text_mse", "proj_head_text_infonce", "proj_head_image_infonce")}
            if all(p.exists() for p in paths.values()):
                self.sources[f"fold{idx}"] = str(root)
                return {k: load_module_or_state(v, device=self.device) for k, v in paths.items()}
        return None

    def nvlm(self, *, generator: bool = False):
        """``NeuroVLM`` client with this bundle's modules plugged in; ``generator`` also sets NeuroQwen3 (pinned revision,
        the client's dtype and pad settings) restricted to Latin-script tokens, whose [NETWORK] / [REGION] titles are
        the canonical names of the bundle's stage-14 banks (free with --free-titles)."""
        if "nvlm" not in self._cache:
            from neurovlm import NeuroVLM

            nvlm = NeuroVLM(device=str(self.device))
            nvlm._autoencoder = self.autoencoder.to(self.device).eval()
            nvlm._proj_head_image_infonce = self.image_head.to(self.device).eval()
            nvlm._proj_head_text_infonce = self.text_head.to(self.device).eval()
            nvlm._proj_head_text_mse = self.mse_head.to(self.device).eval()
            nvlm._specter = self.specter
            self._cache["nvlm"] = nvlm
        nvlm = self._cache["nvlm"]
        if generator and nvlm._neuro_qwen_model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from neurovlm.resources.loaders import NEURO_QWEN_REPO_ID

            rev = hf_pinned(NEURO_QWEN_REPO_ID)
            tok = AutoTokenizer.from_pretrained(NEURO_QWEN_REPO_ID, revision=rev, use_fast=True)
            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token
            dtype = torch.float32 if nvlm.device.type == "cpu" else torch.bfloat16
            lm = AutoModelForCausalLM.from_pretrained(NEURO_QWEN_REPO_ID, revision=rev, torch_dtype=dtype, device_map=None).to(nvlm.device).eval()
            if getattr(lm.config, "pad_token_id", None) is None:
                lm.config.pad_token_id = tok.pad_token_id
            lm.generation_config.suppress_tokens = list(latin_script_suppress_tokens(tok))
            self.title_vocab = title_vocabulary(None if self.released else self.run_dir, device=self.device, logger=log,
                                                specter=self.specter, text_head=self.text_head)
            install_title_constraint(lm, tok, None if self.free_titles else self.title_vocab["titles"])
            nvlm._neuro_qwen_tokenizer, nvlm._neuro_qwen_model = tok, lm
            self.sources["neuroqwen"] = (f"released:{NEURO_QWEN_REPO_ID}@{rev[:8]} (frozen LM, Latin-script tokens only; "
                                         f"{title_constraint_note(lm)})")
            self.sources["title_vocabulary"] = self.title_vocab["source"]["banks"]
        return nvlm


def load_neuro_vectors():
    """``(maps, pmids)`` of every PubMed paper: the pinned HF pubmed_images.pt (= data_dir/neuro_vectors.pt)."""
    payload = load_torch(hf_pinned("neurovlm/embedded_text", "pubmed_images.pt"))
    return payload["neuro_vectors"], np.asarray(payload["pmid"])


def load_text_latents():
    """``(SPECTER2 latents, pmids)``: the pinned HF latent_specter2_adhoc.pt."""
    payload = load_torch(hf_pinned("neurovlm/embedded_text", "latent_specter2_adhoc.pt"))
    return payload["latent"].float(), np.asarray(payload["pmid"])


class Data:
    """Lazy access to the test material (PubMed test split, NeuroVault, networks)."""

    def __init__(self, run_dir: Path | None, max_samples: int | None, cache_dir: Path, *, exclusions: bool = True):
        self.run_dir, self.max_samples, self.cache_dir = run_dir, max_samples, cache_dir
        self.exclusions = exclusions
        self.exclusion_counts: dict[str, object] = {"enabled": exclusions, "source": str(EVAL_EXCLUSIONS_DIR) if exclusions else None}
        self._c: dict[str, object] = {}
        cache_dir.mkdir(parents=True, exist_ok=True)

    def _count(self, key: str, keep: np.ndarray) -> None:
        self.exclusion_counts[key] = {"before": int(len(keep)), "dropped": int((~keep).sum()), "kept": int(keep.sum())}

    def test_pmids(self) -> np.ndarray:
        if "test_pmids" not in self._c:
            # the run's pmids_split.pt (08), else the pinned HF split flags (the same sets); never data_dir
            split_path = self.run_dir / "pmids_split.pt" if self.run_dir else None
            if split_path is not None and split_path.exists():
                test = np.asarray(load_torch(split_path)["test"])
                src = str(split_path)
            else:
                test = hf_split_pmids()["test"]
                src = f"neurovlm/neuro_image_papers/publications.parquet@{hf_pinned('neurovlm/neuro_image_papers')[:8]} test flags"
            test = np.sort(test)
            keep = apply_pubmed_exclusions(test, enabled=self.exclusions, logger=log, label="PubMed test")
            self._count("pubmed_test", keep)
            test = test[keep]
            if self.max_samples:
                test = test[: self.max_samples]
            self._c["test_pmids"] = test
            log.info("PubMed test split: %d pmids (%s)", len(test), src)
        return self._c["test_pmids"]

    def pubmed_maps(self):
        """``(X_test [N, 28542] float32, pmids)``."""
        if "maps" not in self._c:
            import torch

            test = self.test_pmids()
            vectors, pmids = load_neuro_vectors()
            pos = {p: i for i, p in enumerate(pmids)}
            idx = np.array([pos[p] for p in test if p in pos])
            self._c["maps"] = (vectors[torch.from_numpy(idx)].float().clone(), pmids[idx])
            del vectors
        return self._c["maps"]

    def pubmed_text_latents(self):
        if "text_lat" not in self._c:
            import torch

            _, pmids = self.pubmed_maps()
            lat, lp = load_text_latents()
            pos = {p: i for i, p in enumerate(lp)}
            self._c["text_lat"] = lat[torch.as_tensor([pos[p] for p in pmids])]
        return self._c["text_lat"]

    def pubmed_texts(self):
        """``(name [SEP] description, name, summary)`` lists for the test pmids."""
        if "texts" not in self._c:
            from neurovlm.data import load_dataset

            _, pmids = self.pubmed_maps()
            df = load_dataset("pubmed_text").drop_duplicates("pmid").set_index("pmid")
            rows = df.loc[pmids]
            texts = (rows["name"].fillna("").astype(str) + " [SEP] " + rows["description"].fillna("").astype(str)).tolist()
            try:
                summ = load_dataset("pubmed_summaries")
                summ["pmid"] = summ["pmid"].astype(df.index.dtype)
                summ = summ.drop_duplicates("pmid").set_index("pmid")
                summaries = summ.reindex(pmids)["summary"].fillna("").astype(str).str.strip("\n").str.strip(" \n").tolist()
            except Exception as exc:
                log.warning("pubmed_summaries unavailable (%s); brain-to-text truth falls back to abstracts", exc)
                summaries = rows["description"].fillna("").astype(str).tolist()
            self._c["texts"] = (texts, rows["name"].fillna("").astype(str).tolist(), summaries)
        return self._c["texts"]

    def neurovault_hf(self):
        if "nv_hf" not in self._c:
            import torch
            from neurovlm.data import load_dataset

            df_text = load_dataset("neurovault_text")
            images = torch.as_tensor(load_dataset("neurovault_images")).float().cpu()
            images = (images / images.quantile(0.99, dim=1).unsqueeze(1)).clamp(0, 1)
            meta = load_dataset("neurovault_images_meta")
            m = torch.where(~images.isnan().any(dim=1))[0]
            images, meta = images[m], meta.iloc[m.numpy()].reset_index(drop=True)
            # drop the papers that are also PubMed training papers, with all their images
            keep_t = apply_neurovault_exclusions(df_text["doi"].tolist(), enabled=self.exclusions, logger=log, label="NeuroVault HF publications")
            keep_i = apply_neurovault_exclusions(meta["doi"].tolist(), image_ids=meta["id"].to_numpy(), enabled=self.exclusions, logger=log,
                                                 label="NeuroVault HF images")
            self._count("neurovault_hf_publications", keep_t)
            self._count("neurovault_hf_images", keep_i)
            df_text = df_text[keep_t].reset_index(drop=True)
            images, meta = images[torch.from_numpy(keep_i)], meta[keep_i].reset_index(drop=True)
            texts = (df_text["title"] + " [SEP] " + df_text["abstract"]).tolist()
            self._c["nv_hf"] = (df_text, texts, images, meta)
        return self._c["nv_hf"]

    def neurovault_local(self):
        """data_dir/neurovault.pt (clustered maps + text embeddings) if present, minus the excluded papers and images."""
        if "nv_local" not in self._c:
            import torch

            nv = load_torch(hf_pinned("neurovlm/neurovault_maps", "derived/neurovault.pt"))
            if nv is not None and "df_neuro" in nv and "df_pubs" in nv:
                keep_p = apply_neurovault_exclusions(nv["df_pubs"]["doi"].tolist(), enabled=self.exclusions, logger=log, label="neurovault.pt publications")
                keep_i = apply_neurovault_exclusions(nv["df_neuro"]["doi"].tolist(), image_ids=nv["df_neuro"]["id"].to_numpy(), enabled=self.exclusions,
                                                     logger=log, label="neurovault.pt images")
                self._count("neurovault_local_publications", keep_p)
                self._count("neurovault_local_images", keep_i)
                if not keep_p.all() or not keep_i.all():
                    n_p, n_i = len(keep_p), len(keep_i)
                    out = {}
                    for k, v in nv.items():
                        if k == "df_pubs":
                            out[k] = v[keep_p].reset_index(drop=True)
                        elif k == "df_neuro":
                            out[k] = v[keep_i].reset_index(drop=True)
                        elif hasattr(v, "shape") and len(v.shape) and v.shape[0] == n_p and k.startswith("text_emb"):
                            out[k] = v[torch.from_numpy(keep_p)] if torch.is_tensor(v) else v[keep_p]
                        elif hasattr(v, "shape") and len(v.shape) and v.shape[0] == n_i:
                            out[k] = v[torch.from_numpy(keep_i)] if torch.is_tensor(v) else v[keep_i]
                        else:
                            out[k] = v
                    nv = out
            self._c["nv_local"] = nv
        return self._c["nv_local"]

    def network_maps(self):
        """``(flat binarised network maps [145, 28542], names, atlases)``, figures/13 recipe, cached."""
        if "networks" not in self._c:
            import torch

            cache = self.cache_dir / "networks_flat.pt"
            if cache.exists():
                payload = torch.load(cache, weights_only=False)
            else:
                import nibabel as nib
                from nilearn.image import resample_img
                from neurovlm.data import load_dataset, load_masker

                masker = load_masker()
                networks = load_dataset("networks")
                items = [(k, a, nib.Nifti1Image(networks[k][a]["array"], affine=networks[k][a]["affine"]))
                         for k in networks for a in networks[k] if k not in ("UKBICA", "HCPICA")]
                flats = []
                for _k, _a, img in items:
                    arr = img.get_fdata()
                    if len(np.unique(arr)) == 2:
                        res = resample_img(img, masker.affine_, interpolation="nearest", force_resample=True, copy_header=True)
                    else:
                        res = resample_img(img, masker.affine_, force_resample=True, copy_header=True)
                        a = res.get_fdata()
                        a[a < 0] = 0.0
                        thresh = np.percentile(a.flatten(), 95)
                        a[a < thresh] = 0.0
                        a[a >= thresh] = 1.0
                        res = nib.Nifti1Image(a, affine=masker.affine_)
                    flats.append(torch.from_numpy(masker.transform(res)).float().reshape(-1))
                payload = {"flat": torch.stack(flats), "names": [a for _, a, _ in items], "atlases": [k for k, _, _ in items]}
                torch.save(payload, cache)
            self._c["networks"] = (payload["flat"], payload["names"], payload["atlases"])
        return self._c["networks"]

    def network_truth(self):
        """network_key per map from the HF label table ("unknown" when missing)."""
        from neurovlm.data import load_dataset

        _, names, _ = self.network_maps()
        labels = load_dataset("network_test_set_labels").drop_duplicates("raw_network_label").set_index("raw_network_label")
        return [str(labels["network_key"].get(n, "unknown")) for n in names]


def eval_ae(bundle: Bundle, data: Data, device) -> dict:
    import torch
    from neurovlm.metrics.text_to_brain import compute_ae_performance

    ae = bundle.autoencoder.to(device).eval()
    out = {"source": bundle.sources.get("autoencoder")}

    def run(X, prefix):
        logits = encode_in_batches(ae, X, batch_size=1024, device=device)
        _fpr, _tpr, pct, roc_auc = compute_ae_performance(X, logits)
        out[f"{prefix}_voxel_auroc"] = float(roc_auc)
        out[f"{prefix}_bpp_gain_pct_median"] = float(np.median(pct))
        out[f"{prefix}_bpp_gain_pct_mean"] = float(np.mean(pct))
        out[f"{prefix}_bpp_gain_frac_positive"] = float(np.mean(pct > 0))
        out[f"{prefix}_n"] = int(len(X))

    X, _ = data.pubmed_maps()
    run(X, "pubmed")
    nv = data.neurovault_local()
    if nv is not None:
        X_nv = torch.from_numpy(np.asarray(nv["neuro_clustered"])) if not torch.is_tensor(nv["neuro_clustered"]) else nv["neuro_clustered"]
        X_nv = (X_nv > 0).float()
        if data.max_samples:
            X_nv = X_nv[: data.max_samples]
        run(X_nv, "neurovault")
    else:
        out["neurovault_note"] = "data_dir/neurovault.pt (02_data/03) not found; NeuroVault AUROC skipped"
    flat, _, _ = data.network_maps()
    run(flat, "networks")
    return out


def _retrieval_block(z_text, z_img, prefix) -> dict:
    from neurovlm.metrics.retrieval import bidirectional_retrieval_metrics, recall_curve

    m = bidirectional_retrieval_metrics(z_text, z_img)
    out = {f"{prefix}_{k}": float(v) for k, v in m.items() if not k.startswith("mean_random")}
    t2i, i2t = recall_curve(z_text, z_img, step=1)
    n = len(t2i)
    idx = np.unique(np.linspace(0, n - 1, min(n, 101)).astype(int))
    out[f"{prefix}_curve"] = {"normalized_k": ((idx + 1) / n).tolist(), "t2i": t2i[idx].tolist(), "i2t": i2t[idx].tolist()}
    out[f"{prefix}_n"] = int(n)
    return out


def _best_image_per_doi(sim, text_dois, image_dois):
    """Texts that have images, and for each its most similar image (figures/12 recipe)."""
    keep, best = [], []
    for i, doi in enumerate(text_dois):
        cand = np.flatnonzero(image_dois == doi)
        if len(cand):
            keep.append(i)
            best.append(int(cand[int(np.argmax(sim[i, cand]))]))
    return keep, best


def eval_heads(bundle: Bundle, data: Data, device) -> dict:
    import torch
    import torch.nn.functional as F

    ae, th, ih = bundle.autoencoder.to(device).eval(), bundle.text_head.to(device).eval(), bundle.image_head.to(device).eval()
    out = {"source": {k: bundle.sources.get(k) for k in ("autoencoder", "proj_head_text_infonce", "proj_head_image_infonce")}}
    # PubMed test (figures/20, figures/23)
    X, _ = data.pubmed_maps()
    txt = data.pubmed_text_latents()
    with torch.no_grad():
        lat = encode_in_batches(ae.encoder, X, batch_size=2048, device=device, out_device=device)
        z_img = F.normalize(ih(lat), dim=1).cpu()
        z_txt = F.normalize(th(F.normalize(txt.to(device), dim=1)), dim=1).cpu()
    out.update(_retrieval_block(z_txt, z_img, "pubmed"))
    # NeuroVault HF texts and images (figures/12)
    try:
        df_text, texts, images, meta = data.neurovault_hf()
        if data.max_samples:
            df_text, texts = df_text.iloc[: data.max_samples], texts[: data.max_samples]
        with torch.no_grad():
            e = F.normalize(bundle.specter(texts), dim=1).to(device)
            text_c = F.normalize(th(e), dim=1)
            img_lat = encode_in_batches(ae.encoder, images, batch_size=1024, device=device, out_device=device)
            true_c = F.normalize(ih(img_lat), dim=1)
            sim = (text_c @ true_c.T).cpu().numpy()
        keep, best = _best_image_per_doi(sim, df_text["doi"].to_numpy(), meta["doi"].to_numpy())
        out.update(_retrieval_block(text_c[keep].cpu(), true_c[best].cpu(), "neurovault"))
        out["neurovault_n_texts_with_images"] = len(keep)
    except Exception as exc:
        out["neurovault_note"] = f"NeuroVault evaluation skipped: {type(exc).__name__}: {exc}"
    # local neurovault.pt: clustered binary maps + SPECTER title [SEP] abstract embeddings
    nv = data.neurovault_local()
    if nv is not None and "df_neuro" in nv and "doi" in getattr(nv["df_neuro"], "columns", []):
        try:
            t_emb = torch.as_tensor(nv["text_emb_titles_abstracts"]).float()
            X_nv = torch.as_tensor(np.asarray(nv["neuro_clustered"])).float()
            X_nv = (X_nv > 0).float()
            dois_text = nv["df_pubs"]["doi"].to_numpy()
            dois_img = nv["df_neuro"]["doi"].to_numpy()
            if data.max_samples:
                t_emb, dois_text = t_emb[: data.max_samples], dois_text[: data.max_samples]
            with torch.no_grad():
                text_c = F.normalize(th(F.normalize(t_emb.to(device), dim=1)), dim=1)
                true_c = F.normalize(ih(encode_in_batches(ae.encoder, X_nv, batch_size=1024, device=device, out_device=device)), dim=1)
                sim = (text_c @ true_c.T).cpu().numpy()
            keep, best = _best_image_per_doi(sim, dois_text, dois_img)
            out.update(_retrieval_block(text_c[keep].cpu(), true_c[best].cpu(), "neurovault_local"))
        except Exception as exc:
            out["neurovault_local_note"] = f"skipped: {type(exc).__name__}: {exc}"
    return out


def eval_networks(bundle: Bundle, data: Data, device) -> dict:
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import confusion_matrix, roc_auc_score
    from sklearn.preprocessing import label_binarize

    ae, th, ih = bundle.autoencoder.to(device).eval(), bundle.text_head.to(device).eval(), bundle.image_head.to(device).eval()
    flat, names, atlases = data.network_maps()
    keys = data.network_truth()
    with torch.no_grad():
        emb = encode_in_batches(ae.encoder, flat, batch_size=256, device=device, out_device=device)
        net_norm = F.normalize(ih(emb), dim=1)
        text_embed = F.normalize(th(bundle.specter(list(NETWORK_TEXT_LABELS.values())).to(device)), dim=1)
        sim = (net_norm @ text_embed.T).cpu()
    pred = sim.argmax(dim=1).numpy()
    class_names = list(NETWORK_TEXT_LABELS.keys())
    key_to_idx = {k: i for i, k in enumerate(NETWORK_KEYS)}
    label_inds = np.array([key_to_idx.get(k, -1) for k in keys])
    mask = label_inds != -1
    y_true, y_pred = label_inds[mask], pred[mask]
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    row = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row, out=np.zeros_like(cm, dtype=float), where=row != 0)
    y_bin = label_binarize(y_true, classes=np.arange(len(class_names)))
    scores = sim.numpy()[mask]
    per_class_auc = {class_names[i]: float(roc_auc_score(y_bin[:, i], scores[:, i])) for i in range(len(class_names))
                     if 0 < y_bin[:, i].sum() < len(y_bin)}
    return {
        "source": {k: bundle.sources.get(k) for k in ("autoencoder", "proj_head_text_infonce", "proj_head_image_infonce")},
        "accuracy": float((y_pred == y_true).mean()), "n_evaluated": int(mask.sum()), "n_maps": int(len(names)),
        "macro_auc": float(np.mean(list(per_class_auc.values()))) if per_class_auc else float("nan"),
        "one_vs_rest_auc": per_class_auc, "classes": class_names,
        "per_class_recall": {class_names[i]: float(cm_norm[i, i]) for i in range(len(class_names)) if row[i, 0] > 0},
        "confusion_counts": cm.tolist(), "confusion_row_normalized": np.round(cm_norm, 4).tolist(),
        "predictions": [{"atlas": a, "network": n, "truth": k, "pred": class_names[p]} for a, n, k, p in zip(atlases, names, keys, pred)],
    }


def _score_text_to_brain(pred, true, X, texts, text_lat, nvlm, *, n_jobs: int) -> dict:
    """Metrics of generated maps ``pred`` (22_text_to_brain_metrics recipe, no spin test).

    ``true`` is the autoencoder reconstruction sigmoid(decoder(encoder(X))) of the raw maps ``X``.
    """
    from joblib import Parallel, delayed
    from neurovlm.metrics.text_to_brain import dice_percentile, generated_image_retrieval_curve, generated_image_text_retrieval_curve

    out = {}
    auc, _curve, ranks = generated_image_text_retrieval_curve(nvlm, pred, texts, text_latents=text_lat, image_batch_size=256, seed=13)
    out["recall_auc"] = float(auc)
    out["contrastive_sim_matched"] = float(ranks["matched_contrastive_sim"].mean())
    out["contrastive_sim_null"] = float(ranks["null_contrastive_sim"].mean())
    out["contrastive_frac_matched_gt_null"] = float((ranks["matched_contrastive_sim"] > ranks["null_contrastive_sim"]).mean())
    out["contrastive_median_rank"] = float(ranks["rank"].median())
    out["contrastive_recall@10"] = float((ranks["rank"] <= 10).mean())
    auc_img, _, _ = generated_image_retrieval_curve(nvlm, pred, true, batch_size=256, seed=13)
    out["recall_auc_generated_vs_true_image"] = float(auc_img)
    pred_np, true_np = pred.numpy(), true.numpy()
    rng = np.random.default_rng(13)
    null = deranged_indices(len(pred_np), rng)

    def dice_rows(truth, pct):  # mean Dice, matched and shuffled
        m = Parallel(n_jobs=n_jobs)(delayed(dice_percentile)(pred_np[i], truth[i], pct=pct) for i in range(len(pred_np)))
        nl = Parallel(n_jobs=n_jobs)(delayed(dice_percentile)(pred_np[i], truth[null[i]], pct=pct) for i in range(len(pred_np)))
        return float(np.mean(m)), float(np.mean(nl))

    dice_by_pct = {pct: dice_rows(true_np, pct) for pct in (70, 75, 80, 85, 90, 95)}
    out["dice_by_pct"] = {str(k): {"matched": v[0], "null": v[1]} for k, v in dice_by_pct.items()}
    best_pct = max(dice_by_pct, key=lambda k: dice_by_pct[k][0])
    out["dice_pct"] = int(best_pct)
    out["dice_matched"], out["dice_null"] = dice_by_pct[best_pct]
    out["dice90_matched"], out["dice90_null"] = dice_by_pct[90]

    def zrows(x):
        x = x - x.mean(axis=1, keepdims=True)
        return x / np.clip(x.std(axis=1, keepdims=True), 1e-6, None)

    def mean_corr(a, b, n):
        return float(np.mean(np.einsum("ij,ij->i", a, b) / n))

    pz, tz = zrows(pred_np), zrows(true_np)
    d = float(pz.shape[1])
    out["pearson_matched"] = mean_corr(pz, tz, d)
    out["pearson_null"] = mean_corr(pz, tz[null], d)
    vox = np.random.default_rng(13).choice(pred_np.shape[1], size=min(5000, pred_np.shape[1]), replace=False)
    pzs, tzs = zrows(pred_np[:, vox]), zrows(true_np[:, vox])
    out["pearson5000_matched"] = mean_corr(pzs, tzs, len(vox))
    out["pearson5000_null"] = mean_corr(pzs, tzs[null], len(vox))
    # the raw maps as truth (independent of the autoencoder)
    raw_np = X.numpy()
    rz = zrows(raw_np)
    out["pearson_vs_raw_matched"] = mean_corr(pz, rz, d)
    out["pearson_vs_raw_null"] = mean_corr(pz, rz[null], d)
    out["dice90_vs_raw_matched"], out["dice90_vs_raw_null"] = dice_rows(raw_np, 90)
    out["spin_test"] = "not run (surface spin permutations are slow; enable in the notebook recipe if required)"
    return out


def _text_to_brain_inputs(bundle: Bundle, data: Data, device):
    """``(X, texts, SPECTER2 latents, sigmoid(decoder(encoder(X))))`` of the PubMed test split, on the CPU."""
    import torch

    ae = bundle.autoencoder.to(device).eval()
    X, _ = data.pubmed_maps()
    texts, _, _ = data.pubmed_texts()
    with torch.no_grad():
        text_lat = torch.cat([bundle.specter(texts[s:s + 32]).detach().float().cpu() for s in range(0, len(texts), 32)])
        true = torch.cat([torch.sigmoid(ae.decoder(ae.encoder(X[s:s + 512].to(device)))).float().cpu() for s in range(0, len(X), 512)])
    return X, texts, text_lat, true


def eval_generated_maps(bundle: Bundle, data: Data, device, *, adapter: bool, n_jobs: int, calibration: str) -> dict:
    """text_to_brain: sigmoid(decoder(proj_head_text_mse(SPECTER2))); adapter: sigmoid(NeuroAdapter(SPECTER2))."""
    import torch
    import torch.nn.functional as F

    nvlm = bundle.nvlm()
    if adapter:
        model, key = bundle.adapter.to(device).eval(), "adapter"
    else:
        head, ae = bundle.mse_head.to(device).eval(), bundle.autoencoder.to(device).eval()
        key = "proj_head_text_mse"

        def model(z):
            return ae.decoder(head(z))
    X, texts, text_lat, true = _text_to_brain_inputs(bundle, data, device)
    with torch.no_grad():
        pred = torch.cat([torch.sigmoid(model(F.normalize(text_lat[s:s + 512].to(device), dim=1))).float().cpu()
                          for s in range(0, len(text_lat), 512)])
    raw_peak = float(pred.amax(dim=1).mean())
    factor = bundle.map_factor("adapter" if adapter else "head") if calibration == "global" else None
    pred = calibrate_maps(pred, calibration, factor=factor)
    out = {"source": {k: bundle.sources.get(k) for k in (key, "autoencoder", "proj_head_text_infonce", "proj_head_image_infonce")},
           "n": len(texts), "map_calibration": calibration, "map_calibration_factor": factor,
           "generated_peak_mean_before_calibration": raw_peak, "generated_frac_at_1": float((pred.amax(dim=1) >= 1).float().mean())}
    out.update(_score_text_to_brain(pred, true, X, texts, text_lat, nvlm, n_jobs=n_jobs))
    return out


def validity_checks(texts: list[str]) -> dict:
    """Empty, degenerate (< 3 words or > 30 % repeated 4-grams), duplicated and non-Latin-script outputs (a character
    outside _common_models.CAPTION_CHAR_RANGES), length, fabricated details."""
    stripped = [re.sub(r"\s+", " ", re.sub(r"^\s*\[[A-Z]+\]\s*", "", t)).strip() for t in texts]
    n = max(len(stripped), 1)
    empty = sum(len(t) < 3 for t in stripped)
    lens = [len(t.split()) for t in stripped]

    def repeat_rate(t):
        toks = t.split()
        if len(toks) < 8:
            return 0.0
        grams = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
        return 1.0 - len(set(grams)) / len(grams)

    rep = [repeat_rate(t) for t in stripped]
    counts = Counter(stripped)
    duplicated = sum(c for c in counts.values() if c > 1)
    degenerate = sum(r > 0.3 or k < 3 for r, k in zip(rep, lens))
    fab = {k: float(np.mean([bool(p.search(t)) for t in stripped])) for k, p in FABRICATION_PATTERNS.items()}
    return {
        "validity_n": len(stripped), "validity_frac_empty": empty / n, "validity_frac_degenerate": degenerate / n,
        "validity_frac_duplicated_outputs": duplicated / n, "validity_n_unique": len(counts),
        "validity_frac_non_latin": sum(not all(map(caption_char_ok, t)) for t in texts) / n,
        "validity_mean_words": float(np.mean(lens)), "validity_median_words": float(np.median(lens)),
        "validity_mean_4gram_repeat_rate": float(np.mean(rep)),
        "validity_fabrication_rate": fab, "validity_fabrication_any": float(np.mean([any(p.search(t) for p in FABRICATION_PATTERNS.values()) for t in stripped])),
        "examples": stripped[:5],
    }


def eval_qformer(bundle: Bundle, data: Data, device, *, variant: str, out_dir: Path, gen_batch: int, bert_model: str, skip_generation: bool) -> dict:
    """Brain-to-text metrics (figures/24 recipe) of the canonical or the PubMed projector."""
    import pandas as pd
    import torch
    from neurovlm.metrics.common import project_brain_latents_to_shared, project_text_latents_to_shared
    from neurovlm.metrics.semantic import exact_pmid_retrieval_metrics

    nvlm = bundle.nvlm(generator=True)
    qf = bundle.qformer(variant)
    nvlm._neuro_qformer, nvlm._neuro_qformer_variant = qf.to(device).eval(), variant
    X, pmids = data.pubmed_maps()
    _, names, summaries = data.pubmed_texts()
    truth = [f"{n} [SEP] {s}" for n, s in zip(names, summaries)]
    out = {"source": {"qformer": bundle.sources.get(f"qformer_{variant}"), **{k: bundle.sources.get(k) for k in ("autoencoder", "proj_head_text_infonce", "proj_head_image_infonce")}},
           "variant": variant, "projector": bundle.projectors.get(f"qformer_{variant}", "qformer"), "n": len(truth)}
    gen_path = out_dir / f"df_gen_{variant}.parquet"
    bases = ["network", "region", "function"] if variant == "canonical" else ["all"]
    titles = "free" if bundle.free_titles else "vocab"  # stage-2 [NETWORK] / [REGION] titles
    df_gen = pd.read_parquet(gen_path) if gen_path.exists() else None  # reused when its pmids (and title mode) match
    if df_gen is not None and (len(df_gen) != len(pmids) or list(df_gen["pmid"].astype(str)) != [str(p) for p in pmids]):
        df_gen = None
    if df_gen is not None and variant == "canonical" and not skip_generation and \
            (df_gen["title_constraint"].iloc[0] if "title_constraint" in df_gen else "free") != titles:
        log.info("%s holds %s titles; regenerating with %s titles", gen_path.name, "free" if "title_constraint" not in df_gen else df_gen["title_constraint"].iloc[0], titles)
        df_gen = None
    if df_gen is None:
        if skip_generation:
            raise FileNotFoundError(f"{gen_path} missing (or not matching the evaluated pmids) and --skip-generation set")
        outputs = {b: [] for b in bases}
        t0 = time.perf_counter()
        with log_phase(f"{variant} generation", n=len(X), gen_batch=gen_batch, bases=len(bases)):
            for start in range(0, len(X), gen_batch):
                batch = X[start:start + gen_batch]
                kwargs = dict(seed=123 + start, do_sample=False, num_beams=1, max_new_tokens=128, projection_temp=0.035, repetition_penalty=1.25)
                for basis in bases:
                    if variant == "canonical":
                        gen = nvlm.generate_text(batch, basis=basis, **kwargs)
                    else:
                        gen = nvlm.generate_text(batch, basis=None, prefix_text="", qformer_variant="pubmed", use_canonical_projection=False, **kwargs)
                    outputs[basis].extend([gen] if isinstance(gen, str) else gen)
                log.info("generated %d/%d (%s) %.0fs", min(start + gen_batch, len(X)), len(X), variant, time.perf_counter() - t0)
        df_gen = pd.DataFrame({"pmid": [str(p) for p in pmids], **outputs})
        if variant == "canonical":
            for k in bases:
                df_gen[k] = df_gen[k].map(strip_steering_tags).str.replace("\n\n", "\n", regex=True)
            df_gen["text"] = df_gen["network"] + " [SEP] " + df_gen["region"] + " [SEP] " + df_gen["function"]
            df_gen["title_constraint"] = titles
        else:
            df_gen["text"] = df_gen["all"].str.replace("\n\n", "\n", regex=True)
        df_gen.to_parquet(gen_path)
        out["generation_seconds"] = time.perf_counter() - t0
    generated = df_gen["text"].fillna("").astype(str).tolist()
    # generation leaves the CUDA cache full; hand it back before loading the metric encoders
    log.info("%s: released CUDA cache before metrics (%s)", variant, release_cuda_cache())
    out.update(validity_checks(generated))
    if variant == "canonical":  # [NETWORK] / [REGION] titles that are no canonical name of the bundle's stage-14 banks
        bad = {lv: int(sum(noncanonical_titles(df_gen[lv].fillna("").astype(str).tolist(), bundle.title_vocab["titles"][lv])))
               for lv in TITLE_LEVELS}
        out["title_constraint"] = df_gen["title_constraint"].iloc[0] if "title_constraint" in df_gen else "free"
        out["validity_frac_noncanonical_title"] = sum(bad.values()) / max(len(TITLE_LEVELS) * len(df_gen), 1)
        out["validity_n_noncanonical_title"] = bad
    rng = np.random.default_rng(0)
    null = deranged_indices(len(generated), rng)

    # contrastive similarity and retrieval: generated text -> source map in the shared InfoNCE space
    with log_phase(f"{variant} SPECTER2 + shared space", n=len(generated)), torch.no_grad():
        text_latents = torch.cat([nvlm._encode_text(generated[s:s + 32]).detach().cpu() for s in range(0, len(generated), 32)])
        brain_latents = encode_in_batches(nvlm._autoencoder.encoder, X, batch_size=512, device=device)
        z_text = project_text_latents_to_shared(nvlm, text_latents, batch_size=512)
        z_brain = project_brain_latents_to_shared(nvlm, brain_latents, batch_size=512)
    scores = z_text @ z_brain.T
    nvlm_sim = scores.diag()
    out["contrastive_sim_matched"] = float(nvlm_sim.mean())
    out["contrastive_sim_null"] = float(scores[torch.arange(len(generated)), torch.as_tensor(null)].mean())
    out["contrastive_null_test_frac_p_lt_0.05"] = float(((scores > nvlm_sim[:, None]).float().mean(dim=1) < 0.05).float().mean())
    ret = exact_pmid_retrieval_metrics(scores)
    out["recall_auc"] = float(ret["normalized_k_recall_curve_auc"])
    out.update({f"retrieval_{k}": float(v) for k, v in ret.items() if not k.startswith("random")})

    # text-vs-text: BERTScore F1 and MiniLM cosine, matched vs shuffled
    if bert_model != "none":
        refs = {"matched": truth, "null": [truth[i] for i in null]}
        model_type = BERTSCORE_MODELS[bert_model]
        try:
            with log_phase(f"{variant} BERTScore matched + null", n=len(generated)):
                try:
                    f1 = bertscore_f1_many(generated, refs, model_type=model_type, device=device, desc=f"{variant} BERTScore")
                except Exception as exc:
                    if bert_model != "deberta":
                        raise
                    log.warning("deberta-xlarge-mnli unavailable (%s); falling back to roberta-large", exc)
                    model_type = "roberta-large"
                    f1 = bertscore_f1_many(generated, refs, model_type=model_type, device=device, desc=f"{variant} BERTScore")
            out["bertscore_model"] = model_type
            out["bertscore_f1_matched"], out["bertscore_f1_null"] = float(f1["matched"].mean()), float(f1["null"].mean())
        except Exception as exc:
            out["bertscore_note"] = f"skipped: {type(exc).__name__}: {exc}"
    try:
        from sentence_transformers import SentenceTransformer

        with log_phase(f"{variant} MiniLM", texts=2 * len(generated)):
            st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", revision=hf_pinned("sentence-transformers/all-MiniLM-L6-v2"),
                                     device=str(device))
            e = st.encode(generated + truth, batch_size=32, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            g, t = e[:len(generated)], e[len(generated):]
            out["semantic_sim_matched"] = float(np.mean(np.einsum("ij,ij->i", g, t)))
            out["semantic_sim_null"] = float(np.mean(np.einsum("ij,ij->i", g, t[null])))
            del st
    except Exception as exc:
        out["semantic_note"] = f"skipped: {type(exc).__name__}: {exc}"
    release_cuda_cache()
    return out


def eval_cv(bundle: Bundle, data: Data, device, *, folds, cv_b2t: bool) -> dict:
    """Per-fold retrieval and generative recall of the 5-fold CV models (figures/20 cells 8-10)."""
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import auc as sk_auc
    from sklearn.model_selection import KFold
    from neurovlm.metrics.retrieval import bidirectional_retrieval_metrics, recall_curve
    from _common_models import coordinate_filter

    vectors, pmids = load_neuro_vectors()
    mask = coordinate_filter(pmids, max_coords=100)
    pmids, vectors = pmids[mask], vectors[torch.from_numpy(mask)]
    lat, lpm = load_text_latents()
    m = pd.Series(lpm).isin(pmids).to_numpy()
    lat = lat[torch.from_numpy(m)]
    assert (lpm[m] == pmids).all()
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    splits = list(kf.split(np.arange(len(pmids))))
    # the folds stay as trained; the excluded PMIDs are dropped from each fold's held-out part
    keep_cv = apply_pubmed_exclusions(pmids, enabled=data.exclusions, logger=log, label="CV pool (all folds)")
    data._count("cv_pool", keep_cv)
    features = None
    if cv_b2t:
        features = np.load(hf_pinned("neurovlm/cognitive_atlas", "ngram_labels.npy"), allow_pickle=True)
    out = {"folds": {}, "sources": {}}
    per = {"t2i": [], "i2t": [], "t2b_tg": [], "t2b_gt": [], "b2t_tg": [], "b2t_gt": []}
    for idx in folds:
        models = bundle.fold(idx)
        if models is None:
            out["folds"][idx] = "missing"
            continue
        out["sources"][idx] = bundle.sources.get(f"fold{idx}")
        ae, ph_t, ph_i, ph_mse = (models[k].to(device).eval() for k in ("autoencoder", "proj_head_text_infonce", "proj_head_image_infonce", "proj_head_text_mse"))
        test = splits[idx][1]
        n_fold_before = len(test)
        test = test[keep_cv[test]]
        if data.max_samples:
            test = test[: data.max_samples]
        Xt = vectors[torch.from_numpy(test)].float()
        tt = lat[torch.from_numpy(test)].to(device)
        with torch.no_grad():
            latent_img = encode_in_batches(ae.encoder, Xt, batch_size=2048, device=device, out_device=device)
            latent_img_proj = F.normalize(ph_i(latent_img), dim=1)
            latent_text = F.normalize(ph_t(F.normalize(tt, dim=1)), dim=1)
        t_to_i, i_to_t = recall_curve(latent_text.cpu(), latent_img_proj.cpu(), step=10)
        x = np.arange(0, len(latent_img), 10) / len(latent_img)
        fold = {"n_test": int(len(test)), "n_test_excluded": int(n_fold_before - (keep_cv[splits[idx][1]]).sum()),
                "contrastive_t2i_auc_step10": float(sk_auc(x, t_to_i.numpy())), "contrastive_i2t_auc_step10": float(sk_auc(x, i_to_t.numpy()))}
        metrics = bidirectional_retrieval_metrics(latent_text.cpu(), latent_img_proj.cpu())
        fold.update({f"contrastive_{k}": float(v) for k, v in metrics.items() if not k.startswith("mean_random")})
        per["t2i"].append(fold["contrastive_t2i_auc"])
        per["i2t"].append(fold["contrastive_i2t_auc"])
        # text-to-brain generative recall
        with torch.no_grad():
            latent_pred = ph_mse(tt)
            imgs_gen = torch.sigmoid(ae.decoder(latent_pred))
            imgs_gen = imgs_gen / imgs_gen.max(dim=1).values[:, None]  # figures/20 cell 9 (the folds have no validation split)
            latent_img_gen = F.normalize(ph_i(ae.encoder(imgs_gen)), dim=1)
        t_to_g, g_to_t = recall_curve(latent_img_proj.cpu(), latent_img_gen.cpu(), step=1)
        fold["t2b_true_to_generated_auc"], fold["t2b_generated_to_true_auc"] = float(t_to_g.mean()), float(g_to_t.mean())
        per["t2b_tg"].append(fold["t2b_true_to_generated_auc"])
        per["t2b_gt"].append(fold["t2b_generated_to_true_auc"])
        # brain-to-text n-gram recall
        if features is not None:
            with torch.no_grad():
                feat_lat = torch.cat([bundle.specter(features[s:s + 512].tolist()).detach().float().cpu() for s in range(0, len(features), 512)]).to(device)
                feature_emb = F.normalize(ph_t(F.normalize(feat_lat, dim=1)), dim=1)
                sim = latent_img_proj @ feature_emb.T
                inds = torch.argsort(sim, descending=True, dim=1)
                f = [", ".join(features[i.cpu().numpy()].tolist()) for i in inds[:, :10]]
                gen_lat = torch.cat([bundle.specter(f[s:s + 64]).detach().float().cpu() for s in range(0, len(f), 64)]).to(device)
                gen_proj = F.normalize(ph_t(F.normalize(gen_lat, dim=1)), dim=1)
            t_to_g, g_to_t = recall_curve(latent_text.cpu(), gen_proj.cpu(), step=1)
            fold["b2t_true_to_generated_auc"], fold["b2t_generated_to_true_auc"] = float(t_to_g.mean()), float(g_to_t.mean())
            per["b2t_tg"].append(fold["b2t_true_to_generated_auc"])
            per["b2t_gt"].append(fold["b2t_generated_to_true_auc"])
        out["folds"][idx] = fold
        del models
    done = [k for k, v in out["folds"].items() if isinstance(v, dict)]
    out["n_folds"] = len(done)
    if done:
        out["contrastive_t2i_auc"], out["contrastive_i2t_auc"] = float(np.mean(per["t2i"])), float(np.mean(per["i2t"]))
        out["contrastive_mean_auc"] = float((out["contrastive_t2i_auc"] + out["contrastive_i2t_auc"]) / 2)
        out["t2b_generated_mean_auc"] = float(np.mean(per["t2b_tg"] + per["t2b_gt"]))
        if per["b2t_tg"]:
            out["b2t_generated_mean_auc"] = float(np.mean(per["b2t_tg"] + per["b2t_gt"]))
    return out


def compare(component: str, retrained: dict, reference: dict | None, tol: dict) -> dict:
    rows, status = [], "PASS"
    paper = PAPER.get(component, {})
    if not isinstance(retrained, dict) or "error" in retrained:
        return {"status": "ERROR", "rows": [], "error": (retrained or {}).get("error") if isinstance(retrained, dict) else str(retrained)}
    for key, kind, higher in PRIMARY[component]:
        r = retrained.get(key)
        ref = reference.get(key) if isinstance(reference, dict) else None
        row = {"metric": key, "paper": paper.get(key), "released": ref, "retrained": r}
        if isinstance(r, (int, float)) and isinstance(ref, (int, float)) and not (math.isnan(r) or math.isnan(ref)):
            delta = (r - ref) if higher else (ref - r)
            row["delta"] = r - ref
            row["pass"] = bool(delta >= -tol[kind])
            if not row["pass"]:
                status = "FAIL"
        else:
            row["delta"], row["pass"] = None, None
            if r is None and ref is not None:
                status = "FAIL" if status == "PASS" else status
        free = key == "validity_frac_noncanonical_title" and retrained.get("title_constraint") == "free"  # --free-titles
        if key in MUST_BE_ZERO and not free and isinstance(r, (int, float)) and r != 0:
            row["pass"], status = False, "FAIL"
        rows.append(row)
    if all(r["pass"] is None for r in rows):
        status = "N/A"
    return {"status": status, "rows": rows}


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def exclusion_line(ex: dict | None) -> str:
    if not ex or not ex.get("enabled"):
        return "off (full test sets; --no-exclusions)"
    parts = []
    for key, label in (("pubmed_test", "PubMed test"), ("cv_pool", "CV pool"), ("neurovault_hf_publications", "NeuroVault HF publications"),
                       ("neurovault_hf_images", "NeuroVault HF images"), ("neurovault_local_publications", "neurovault.pt publications"),
                       ("neurovault_local_images", "neurovault.pt images")):
        c = ex.get(key)
        if isinstance(c, dict):
            parts.append(f"{label} -{c['dropped']} of {c['before']}")
    return "on (scripts/eval_exclusions/: " + ("; ".join(parts) or "no item set loaded") + ")"


def write_report(path: Path, *, name: str, reference_name: str, results: dict, comparisons: dict, tol: dict, sources: dict, meta: dict):
    lines = [f"# Validation report: `{name}`", "",
             f"Reference (released checkpoints): `{reference_name}` | tolerances: AUC {tol['auc']}, cosine-type {tol['cos']}, accuracy/recall {tol['acc']}, bpp-gain {tol['bpp']} pct-points | "
             f"max_samples {meta.get('max_samples')} | device {meta.get('device')} | {meta.get('ts')}", "",
             "PASS = retrained >= released - tolerance on every primary metric (lower-is-better metrics inverted).", "",
             f"Evaluation exclusions: {exclusion_line(meta.get('exclusions'))}", "",
             "Components: `text_to_brain` = NeuroVLM's text-to-brain model (`proj_head_text_mse` + decoder; head v2 in runs trained "
             "since 2026-10-08, the released head as reference); `adapter` = the anatomical head (NeuroAdapter, stage 12; internal, "
             "it synthesises the stage-14 training latents); "
             "`qformer_pubmed` = stage 1 projector (grounded, PubMed), `qformer` = stage 2 projector (canonical); "
             "the names keep the historical `qformer` prefix whichever projector the package holds"
             + "".join(f" (`{c}`: {results[c]['projector']})" for c in ("qformer_pubmed", "qformer")
                       if isinstance(results.get(c), dict) and results[c].get("projector")) + ".", ""]
    lines += ["| component | status | metric | paper | released | retrained | delta |", "|---|---|---|---|---|---|---|"]
    for comp, cmp in comparisons.items():
        if cmp.get("error"):
            lines.append(f"| {comp} | ERROR | {cmp['error'][:80]} | | | | |")
            continue
        for i, row in enumerate(cmp["rows"]):
            flag = "" if row["pass"] is None else (" ok" if row["pass"] else " **FAIL**")
            lines.append(f"| {comp if i == 0 else ''} | {cmp['status'] if i == 0 else ''} | {row['metric']} | {fmt(row['paper'])} | {fmt(row['released'])} | {fmt(row['retrained'])} | {fmt(row['delta'])}{flag} |")
    lines += ["", "## Artifact sources", ""]
    for k, v in sorted(sources.items()):
        lines.append(f"- `{k}`: {v}")
    lines += ["", "## Secondary metrics", ""]
    for comp, res in results.items():
        if not isinstance(res, dict):
            continue
        lines.append(f"### {comp}")
        skip = {"source", "sources", "predictions", "examples", "folds", "confusion_counts", "confusion_row_normalized", "one_vs_rest_auc", "per_class_recall", "dice_by_pct", "validity_fabrication_rate"}
        for k, v in res.items():
            if k in skip or k.endswith("_curve"):
                continue
            if isinstance(v, (int, float)):
                lines.append(f"- {k}: {fmt(float(v)) if isinstance(v, float) else v}")
            elif isinstance(v, str) or k == "validity_n_noncanonical_title":
                lines.append(f"- {k}: {v}")
        if comp == "networks" and "confusion_counts" in res:
            lines.append("")
            lines.append("Confusion (rows = truth, cols = predicted): " + ", ".join(res["classes"]))
            for cls, row in zip(res["classes"], res["confusion_counts"]):
                lines.append(f"- {cls}: {row}")
        if comp in ("qformer", "qformer_pubmed") and "examples" in res:
            lines.append("")
            lines.append("Example generations:")
            for ex in res["examples"][:3]:
                lines.append(f"- {ex[:300]}")
            lines.append(f"- fabrication rates: {res.get('validity_fabrication_rate')}")
        if comp in ("text_to_brain", "adapter") and "dice_by_pct" in res:
            lines.append(f"- dice by percentile (matched/null): {res['dice_by_pct']}")
        if comp == "cv" and "folds" in res:
            for fi, fold in res["folds"].items():
                lines.append(f"- fold {fi}: {fold if not isinstance(fold, dict) else {k: round(v, 4) for k, v in fold.items() if isinstance(v, float)}}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def run_components(bundle, data, device, components, args, out_dir, meta: dict | None = None) -> dict:
    """Evaluate each component (a failure is recorded, not raised); metrics.json is saved after each one."""
    results = {}
    for comp in components:
        try:
            with timed(f"eval_{comp}", log, out=out_dir, component=comp):
                if comp == "ae":
                    results[comp] = eval_ae(bundle, data, device)
                elif comp == "heads":
                    results[comp] = eval_heads(bundle, data, device)
                elif comp == "networks":
                    results[comp] = eval_networks(bundle, data, device)
                elif comp in ("text_to_brain", "adapter"):
                    results[comp] = eval_generated_maps(bundle, data, device, adapter=comp == "adapter", n_jobs=args.n_jobs,
                                                        calibration=args.map_calibration)
                elif comp in ("qformer", "qformer_pubmed"):
                    variant = "canonical" if comp == "qformer" else "pubmed"
                    results[comp] = eval_qformer(bundle, data, device, variant=variant, out_dir=out_dir, gen_batch=args.generation_batch,
                                                 bert_model=args.bertscore_model, skip_generation=args.skip_generation)
                elif comp == "cv":
                    results[comp] = eval_cv(bundle, data, device, folds=[int(f) for f in args.folds.split(",")], cv_b2t=not args.no_cv_b2t)
        except Exception as exc:
            log.exception("component %s failed", comp)
            results[comp] = {"error": f"{type(exc).__name__}: {exc}"}
        write_json(out_dir / "metrics.json", {"components": results, "sources": bundle.sources, "partial": True,
                                              **({"meta": meta} if meta is not None else {"exclusions": data.exclusion_counts})})
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", type=Path, help="Run directory with retrained artifacts (scripts/outputs/runs/<name>).")
    g.add_argument("--released", action="store_true", help="Evaluate the released checkpoints.")
    p.add_argument("--components", default=",".join(ALL_COMPONENTS), help="Comma list of components.")
    p.add_argument("--max-samples", type=int, default=None, help="Cap the test items per component.")
    p.add_argument("--name", default=None, help="Output dir name under scripts/outputs/validation/.")
    p.add_argument("--reference", type=Path, default=None,
                   help="Reference metrics.json (default: validation/released[_excl][_n<N>]/metrics.json, computed if missing).")
    p.add_argument("--no-reference", action="store_true", help="Do not compare with the released reference.")
    p.add_argument("--auc-tol", type=float, default=0.01)
    p.add_argument("--cos-tol", type=float, default=0.02)
    p.add_argument("--acc-tol", type=float, default=0.02)
    p.add_argument("--bpp-tol", type=float, default=2.0, help="Tolerance for the bits-per-pixel gain (percentage points).")
    p.add_argument("--frac-tol", type=float, default=0.05, help="Tolerance for the fraction of degenerate outputs.")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-jobs", type=int, default=16)
    p.add_argument("--generation-batch", type=int, default=512)
    p.add_argument("--bertscore-model", choices=["deberta", "roberta", "none"], default="deberta")
    p.add_argument("--skip-generation", action="store_true", help="Only reuse the cached df_gen_*.parquet.")
    p.add_argument("--folds", default="0,1,2,3,4")
    p.add_argument("--no-cv-b2t", action="store_true", help="Skip the n-gram brain-to-text recall of the CV folds.")
    p.add_argument("--no-exclusions", action="store_true", help="Keep the leakage-audit items (for comparison only).")
    p.add_argument("--free-titles", action="store_true",
                   help="Decode free [NETWORK] / [REGION] titles (default: only the canonical names of the stage-14 banks).")
    p.add_argument("--map-calibration", choices=MAP_CALIBRATION_RULES, default=MAP_CALIBRATION_DEFAULT,
                   help="text_to_brain/adapter maps: global = one factor per model fitted on the validation split, clipped at 1; "
                        "max = divide each map by its maximum; none = raw sigmoid maps.")
    p.add_argument("--rescale-max", dest="map_calibration", action="store_const", const="max",
                   help="Divide each generated map by its own maximum (= --map-calibration max, the previous rule).")
    args = p.parse_args(argv)
    exclusions = not args.no_exclusions
    setup_torch(args.seed)
    device = select_device(args.device)
    components = [c.strip() for c in args.components.split(",") if c.strip()]
    unknown = set(components) - set(ALL_COMPONENTS)
    if unknown:
        p.error(f"unknown components {sorted(unknown)}")
    tol = {"auc": args.auc_tol, "cos": args.cos_tol, "acc": args.acc_tol, "bpp": args.bpp_tol, "frac": args.frac_tol}
    suffix = f"_n{args.max_samples}" if args.max_samples else ""
    excl_tag = "_excl" if exclusions else ""
    name = args.name or ("released" + excl_tag + suffix if args.released else Path(args.run).name + excl_tag)
    out_dir = VALIDATION_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = VALIDATION_ROOT / "cache"
    run_dir = None if args.released else Path(args.run)
    if run_dir is not None and not run_dir.is_absolute() and not run_dir.exists():
        run_dir = RUNS_ROOT / run_dir
    if run_dir is not None and not run_dir.exists():
        p.error(f"run directory {run_dir} does not exist")
    meta = {"name": name, "run_dir": str(run_dir) if run_dir else None, "released": args.released, "max_samples": args.max_samples,
            "device": str(device), "components": components, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "tolerances": tol,
            "map_calibration": args.map_calibration, "titles": "free" if args.free_titles else "vocab"}

    data = Data(run_dir, args.max_samples, cache_dir, exclusions=exclusions)
    meta["exclusions"] = data.exclusion_counts  # filled in place while the components load their data
    bundle = Bundle(run_dir, device, released=args.released, free_titles=args.free_titles)
    results = run_components(bundle, data, device, components, args, out_dir, meta=meta)
    payload = {"meta": meta, "components": results, "sources": bundle.sources}

    if not args.released and not args.no_reference:
        ref_path = args.reference or (VALIDATION_ROOT / f"released{excl_tag}{suffix}" / "metrics.json")
        ref = read_json(ref_path) if Path(ref_path).exists() else {"components": {}}
        ref_excl = (ref.get("meta", {}).get("exclusions") or {}).get("enabled", False) if ref.get("components") else exclusions
        if bool(ref_excl) != exclusions:
            log.warning("reference %s was computed with exclusions=%s but this run uses exclusions=%s -- the comparison mixes item sets",
                        ref_path, bool(ref_excl), exclusions)
            meta["reference_exclusions_mismatch"] = True
        missing = [c for c in components if c not in ref.get("components", {}) or "error" in ref["components"].get(c, {})]
        # a text_to_brain / adapter reference scored with another map calibration (no key = none) is recomputed;
        # the old entry is kept under ref["superseded"]
        for c in ("text_to_brain", "adapter"):
            old = ref.get("components", {}).get(c)
            if c in components and c not in missing and isinstance(old, dict) and old.get("map_calibration", "none") != args.map_calibration:
                log.info("reference %s was scored with map calibration %r; recomputing with %r", c, old.get("map_calibration", "none"), args.map_calibration)
                ref.setdefault("superseded", {})[f"{c}@{old.get('map_calibration', 'none')}"] = old
                missing.append(c)
        if missing:
            log.info("computing released reference for %s -> %s", missing, ref_path)
            ref_dir = Path(ref_path).parent
            ref_dir.mkdir(parents=True, exist_ok=True)
            ref_bundle = Bundle(None, device, released=True, free_titles=args.free_titles)
            ref_data = Data(None, args.max_samples, cache_dir, exclusions=exclusions)
            ref_results = run_components(ref_bundle, ref_data, device, missing, args, ref_dir)
            ref.setdefault("components", {}).update(ref_results)
            ref["sources"] = {**ref.get("sources", {}), **ref_bundle.sources}
            ref["meta"] = {**meta, "name": ref_dir.name, "released": True, "run_dir": None,
                           "exclusions": {**(ref.get("meta", {}).get("exclusions") or {}), **ref_data.exclusion_counts}}
            write_json(ref_path, ref)
            write_report(ref_dir / "report.md", name=ref_dir.name, reference_name="(self)", results=ref["components"],
                         comparisons={c: compare(c, ref["components"][c], None, tol) for c in ref["components"]}, tol=tol,
                         sources=ref.get("sources", {}), meta=ref["meta"])
        comparisons = {c: compare(c, results.get(c, {}), ref["components"].get(c), tol) for c in components}
        payload["comparisons"] = comparisons
        payload["reference"] = str(ref_path)
        reference_name = str(ref_path)
    else:
        comparisons = {c: compare(c, results.get(c, {}), None, tol) for c in components}
        payload["comparisons"] = comparisons
        reference_name = "(self)"
    payload["provenance"] = provenance()
    write_json(out_dir / "metrics.json", payload)
    write_report(out_dir / "report.md", name=name, reference_name=reference_name, results=results, comparisons=comparisons, tol=tol, sources=bundle.sources, meta=meta)
    log.info("wrote %s and report.md", out_dir / "metrics.json")
    for comp, cmp in comparisons.items():
        log.info("%-15s %s", comp, cmp["status"] + ("" if not cmp.get("error") else f" ({cmp['error']})"))
        for row in cmp.get("rows", []):
            log.info("   %-36s paper=%s released=%s retrained=%s delta=%s%s", row["metric"], fmt(row["paper"]), fmt(row["released"]), fmt(row["retrained"]), fmt(row["delta"]),
                     "" if row["pass"] is None else ("" if row["pass"] else "  FAIL"))
    return 0 if all(c["status"] in ("PASS", "N/A") for c in comparisons.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
