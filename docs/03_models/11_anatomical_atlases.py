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
# # Anatomical-atlas synthetic training data
#
# This notebook reconstructs the data stage recovered from
# the archived anatomical extraction notebook.
#
# **Inputs:** PubMed text/images loaded by `neurovlm.data.load_dataset`, atlas
# resources loaded by `load_masker`, and `data_dir / "synth.parquet"` containing
# the generated `title` and `description` rows.
#
# **Outputs:** `data_dir / "images_synth.pt"` and the aligned
# `data_dir / "synth.parquet"`. These are the direct inputs to
# `12_neuroadapter.ipynb`.
#

# %%
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from contextlib import redirect_stdout, redirect_stderr, contextmanager
import hashlib
import json
import math
import random
from pathlib import Path
import re

from IPython.display import display
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from nilearn import datasets
from nilearn.image import load_img, resample_to_img, smooth_img
from nilearn.plotting import view_img
from scipy import ndimage
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from tqdm.auto import tqdm

from neurovlm.data import load_dataset, load_masker, data_dir


def find_project_root(start: Path = Path.cwd()) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "neurovlm").is_dir():
            return candidate
    raise RuntimeError("Run this notebook from within the neurovlm repository.")


PROJECT_ROOT = find_project_root()
MODEL_DIR = PROJECT_ROOT / "docs" / "03_models"
MODEL_DATA_DIR = data_dir
MODEL_DATA_DIR.mkdir(parents=True, exist_ok=True)

SYNTH_TABLE_PATH = MODEL_DATA_DIR / "synth.parquet"
IMAGES_SYNTH_PATH = MODEL_DATA_DIR / "images_synth.pt"
REGION_TERMS_PATH = MODEL_DATA_DIR / "df_region_terms.parquet"
IMAGES_AGG_PATH = MODEL_DATA_DIR / "images_agg.pt"
ATLAS_LINEAGE_PATH = MODEL_DATA_DIR / "anatomical_atlases.lineage.json"


def path_label(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()

SEED = 17
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
masker = load_masker()

NOTEBOOK_DIR = MODEL_DIR
LOAD_EXISTING = False
CACHE_VERSION = "anatomy_v5_single_canonical_v12"
PUBMED_CACHE_VERSION = "pubmed_canonical_text_pairs_v7"
CACHE_DIR = MODEL_DATA_DIR / "anatomy_v5_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SEP = " [SEP] "

ATLAS_FWHM = 9.0
DIFUMO_FWHM = 5.0

TARGET_FLOOR = 0.04
TARGET_MIN_VOXELS = 8
TARGET_MAX_VOXELS = 2500
TARGET_CORE_MAX_VOXELS = 120
TARGET_FINAL_MAX_FACTOR = 3
TARGET_SMOOTH_STEPS = 3
TARGET_SMOOTH_ALPHA = 0.35
TARGET_SMOOTH_FLOOR = 0.06
TARGET_GAMMA = 0.85

USE_ATLAS_DESCRIPTION_TEXT = False
USE_RANDOM_COMBOS = True

TRAIN_MIN_SUPPORT = 8
TRAIN_MAX_SUPPORT = 1800
TRAIN_MIN_MID_MASS_FRAC = 0.15

LOSS_BCE_WEIGHT = 0.00
LOSS_MSE_WEIGHT = 1.00
LOSS_DICE_WEIGHT = 0.10
LOSS_RECALL_WEIGHT = 0.00
LOSS_SMOOTH_WEIGHT = 0.50
LOSS_MASS_WEIGHT = 0.20
SMOOTH_MAX_EDGES = 100000

PUBMED_MAX_TERMS = 10
PUBMED_MAX_PAIRS = 10_000
PUBMED_MIN_INSIDE_FRAC = 0.0
PUBMED_MIN_REGION_MAP_FRAC = 0.08
PUBMED_MIN_REGION_COVERAGE = 0.10
PUBMED_MAX_REGION_MATCHES = 5
PUBMED_MAX_SCAN_SENTENCES = 30000


# %%
def cache_file(name):
    return CACHE_DIR / name


def data_file(name):
    path = Path(name)
    if path.exists():
        return path
    path = NOTEBOOK_DIR / name
    if path.exists():
        return path
    raise FileNotFoundError(name)


def stable_hash(value):
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def _clean_text(value):
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).replace("\u00a0", " ")
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*\[\s*sep\s*\]\s*", SEP, text, flags=re.IGNORECASE)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return "" if text.casefold() in {"", "nan", "none", "null"} else text


def _join_sep(parts):
    out = []
    seen = set()
    for part in parts:
        part = _clean_text(part)
        if not part:
            continue
        key = part.casefold()
        if key not in seen:
            out.append(part)
            seen.add(key)
    return SEP.join(out)


def _text_key(value):
    return _clean_text(value).casefold()


def _join_names(names):
    names = [_clean_text(name).lower() for name in names if _clean_text(name)]
    if len(names) <= 1:
        return names[0] if names else ""
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def _side_and_base(name):
    name = _clean_text(name).lower()
    for side in ("left", "right"):
        prefix = f"{side} "
        if name.startswith(prefix):
            return side, name[len(prefix):]
    return "", name


def _opposite_side(side):
    return "right" if side == "left" else "left"


def _has_bilateral_directive(text):
    text = _clean_text(text).lower()
    return bool(re.search(r"(?<![a-z0-9])(bilateral|both|left and right|right and left)(?![a-z0-9])", text))


def _side_from_text(text):
    text = _clean_text(text).lower()
    if _has_bilateral_directive(text):
        return ""
    has_left = bool(re.search(r"(?<![a-z0-9])left(?![a-z0-9])", text))
    has_right = bool(re.search(r"(?<![a-z0-9])right(?![a-z0-9])", text))
    if has_left and not has_right:
        return "left"
    if has_right and not has_left:
        return "right"
    return ""


def _strip_side_text(text):
    text = _clean_text(text).lower()
    text = re.sub(r"^(left|right) hemisphere ", "", text)
    text = re.sub(r"^(left|right) hemispheric ", "", text)
    text = re.sub(r"^(left|right) ", "", text)
    text = re.sub(r" in the (left|right) hemisphere$", "", text)
    text = re.sub(r" of the (left|right) hemisphere$", "", text)
    text = re.sub(r" with (left|right) hemisphere dominance$", "", text)
    text = re.sub(r" with (left|right) hemispheric dominance$", "", text)
    text = re.sub(r" with (left|right) lateralization$", "", text)
    text = re.sub(r" with (left|right) lateralized activity$", "", text)
    return _clean_text(text).lower()


MODIFIER_WORDS = {
    "anterior", "posterior", "dorsal", "ventral", "superior", "inferior",
    "medial", "lateral", "rostral", "caudal", "orbital", "opercular",
    "triangular", "triangularis", "orbitalis", "central", "short", "long",
}


def _modifiers(text):
    words = set(re.findall(r"[a-z0-9]+", _clean_text(text).lower()))
    return words & MODIFIER_WORDS


def _phrase_match(a, b):
    a = _clean_text(a).lower()
    b = _clean_text(b).lower()
    if not a or not b:
        return False
    if a == b:
        return True
    if len(a.split()) < 2 and len(a) < 5:
        return False
    mods_a = _modifiers(a)
    mods_b = _modifiers(b)
    if mods_a and not mods_a.issubset(mods_b):
        return False
    pa = r"(?<![a-z0-9])" + re.escape(a) + r"(?![a-z0-9])"
    pb = r"(?<![a-z0-9])" + re.escape(b) + r"(?![a-z0-9])"
    return bool(re.search(pa, b) or re.search(pb, a))


ALIASES_BY_BASE = {
    "insula": ["insular cortex", "insular", "insular lobe"],
    "insular cortex": ["insula", "insular", "insular lobe"],
    "anterior insula": ["anterior insular cortex", "frontoinsular cortex", "fronto insular cortex", "a insula", "a ins"],
    "posterior insula": ["posterior insular cortex", "p insula", "p ins"],
    "medial frontal cortex": ["medial prefrontal cortex", "medial pfc", "mpfc", "m pfc", "frontal medial cortex", "prefrontal medial cortex", "anterior medial prefrontal cortex", "ventromedial prefrontal cortex", "vm pfc", "vmpfc"],
    "medial prefrontal cortex": ["medial frontal cortex", "medial pfc", "mpfc", "m pfc", "frontal medial cortex", "prefrontal medial cortex", "anterior medial prefrontal cortex", "ventromedial prefrontal cortex", "vm pfc", "vmpfc"],
    "frontal default mode network": ["front dmn", "anterior default mode network", "anterior dmn", "medial prefrontal cortex", "medial pfc", "mpfc", "ventromedial prefrontal cortex", "vmpfc"],
    "anterior cingulate cortex": ["anterior cingulate", "acc", "dorsal anterior cingulate cortex", "dacc", "da cingulate"],
    "dorsal anterior cingulate cortex": ["dorsal anterior cingulate", "dacc", "acc", "anterior cingulate cortex"],
    "posterior cingulate cortex": ["posterior cingulate", "pcc", "retrosplenial posterior cingulate"],
    "inferior frontal gyrus": ["ifg", "broca area", "broca's area", "brocas area", "ventrolateral prefrontal cortex", "vlpfc", "vl pfc"],
    "pars opercularis of the inferior frontal gyrus": ["pars opercularis", "opercular inferior frontal gyrus", "inferior frontal gyrus pars opercularis", "broca pars opercularis"],
    "pars triangularis of the inferior frontal gyrus": ["pars triangularis", "triangular inferior frontal gyrus", "inferior frontal gyrus pars triangularis", "broca pars triangularis"],
    "pars orbitalis of the inferior frontal gyrus": ["pars orbitalis", "orbital inferior frontal gyrus", "inferior frontal gyrus pars orbitalis"],
    "middle frontal gyrus": ["mfg", "lateral prefrontal cortex", "dlpfc", "dorsolateral prefrontal cortex", "dorsolateral pfc"],
    "dorsolateral prefrontal cortex": ["dlpfc", "dorsolateral pfc", "middle frontal gyrus", "mfg"],
    "superior frontal gyrus": ["sfg", "frontal eye field", "frontal eye fields", "fef"],
    "frontal pole": ["frontopolar cortex", "anterior prefrontal cortex", "rostral prefrontal cortex", "ba10", "brodmann area 10"],
    "precentral gyrus": ["motor cortex", "primary motor cortex", "m1"],
    "postcentral gyrus": ["somatosensory cortex", "primary somatosensory cortex", "s1"],
    "superior parietal lobule": ["spl", "superior parietal cortex"],
    "inferior parietal lobule": ["ipl", "inferior parietal cortex"],
    "angular gyrus": ["angular", "posterior inferior parietal cortex"],
    "supramarginal gyrus": ["supramarginal", "smg", "temporoparietal junction", "temporo parietal junction", "tpj"],
    "intraparietal sulcus": ["ips", "intraparietal cortex", "intra parietal sulcus"],
    "middle temporal gyrus": ["mtg", "middle temporal cortex"],
    "superior temporal gyrus": ["stg", "posterior superior temporal cortex", "posterior superior temporal gyrus", "wernicke area", "wernicke's area", "wernickes area"],
    "posterior superior temporal gyrus": ["posterior superior temporal cortex", "wernicke area", "wernicke's area", "wernickes area"],
    "inferior temporal gyrus": ["itg", "inferior temporal cortex"],
    "temporal pole": ["anterior temporal pole", "anterior temporal lobe", "atl"],
    "hippocampus": ["hippocampal formation", "hippocampal"],
    "parahippocampal gyrus": ["parahippocampal cortex", "parahippocampal", "phg"],
    "occipitotemporal fusiform cortex": ["visual word form area", "vwfa", "ventral occipitotemporal cortex", "occipitotemporal cortex"],
    "fusiform gyrus": ["fusiform cortex", "visual word form area", "vwfa", "ventral occipitotemporal cortex"],
    "supplementary motor area": ["sma", "pre sma", "presupplementary motor area", "pre supplementary motor area"],
    "caudate": ["caudate nucleus"],
    "putamen": ["lentiform nucleus", "lentiform"],
    "amygdala": ["amygdalar complex", "amygdaloid complex"],
    "thalamus": ["thalamic nucleus", "thalamic nuclei", "thalamic"],
    "cerebellum": ["cerebellar", "cerebellar cortex", "cerebellar hemisphere", "cerebellar hemispheres"],
}


def base_aliases(base):
    base = _clean_text(base).lower()
    aliases = {base, base.replace("-", " ")}
    for suffix in [" cortex", " gyrus", " sulcus", " lobule", " lobe", " area", " nucleus", " network"]:
        if base.endswith(suffix):
            aliases.add(base[: -len(suffix)])
    if base.endswith("gyri"):
        aliases.add(base[:-4] + "gyrus")
    if base.endswith("gyrus"):
        aliases.add(base[:-5] + "gyri")
    if base == "cerebellum":
        aliases.add("cerebellar")
    if base == "hippocampus":
        aliases.add("hippocampal")
    if base == "thalamus":
        aliases.add("thalamic")
    if base == "amygdala":
        aliases.add("amygdalar")
    if base == "insula":
        aliases.add("insular")
    aliases.update(a.lower() for a in ALIASES_BY_BASE.get(base, []))
    return sorted(_clean_text(a) for a in aliases if _clean_text(a))


def name_aliases(name):
    side, base = _side_and_base(name)
    aliases = []
    for alias in base_aliases(base):
        if side:
            aliases.extend([
                f"{side} {alias}",
                f"{side} hemisphere {alias}",
                f"{alias} in the {side} hemisphere",
            ])
        else:
            aliases.append(alias)
    return sorted(dict.fromkeys(a.lower() for a in aliases if a))


def build_region_lookup(names):
    names = pd.Series(names).astype(str).map(_clean_text).str.lower().to_numpy()
    lookup = {}
    by_base = {}
    for i, name in enumerate(names):
        side, base = _side_and_base(name)
        for alias in name_aliases(name):
            lookup.setdefault(alias, set()).add(i)
        if side:
            by_base.setdefault(base, {}).setdefault(side, set()).add(i)

    for base, sides in by_base.items():
        if "left" in sides and "right" in sides:
            bilateral = sides["left"] | sides["right"]
            for alias in base_aliases(base):
                lookup.setdefault(alias, set()).update(bilateral)
                # lookup.setdefault(f"bilateral {alias}", set()).update(bilateral)
                # lookup.setdefault(f"left and right {alias}", set()).update(bilateral)
                # lookup.setdefault(f"both {alias}", set()).update(bilateral)
    return {k: sorted(v) for k, v in lookup.items()}, names


def resolve_region_indices(term, lookup, names):
    term = _clean_text(term).lower().strip(".,;:")
    if not term:
        return []
    if term in lookup:
        return lookup[term]
    exact = np.flatnonzero(names == term).tolist()
    if exact:
        return exact

    side = _side_from_text(term)
    base_term = _strip_side_text(term) if side else term
    if side:
        for alias in base_aliases(base_term):
            side_term = f"{side} {alias}"
            if side_term in lookup:
                return lookup[side_term]
        out = []
        for i, name in enumerate(names):
            name_side, name_base = _side_and_base(name)
            if name_side == side and any(_phrase_match(alias, name_base) for alias in base_aliases(base_term)):
                out.append(i)
        return sorted(set(out))

    for alias in base_aliases(term):
        if alias in lookup:
            return lookup[alias]
    return sorted({i for i, name in enumerate(names) if any(_phrase_match(alias, name) for alias in base_aliases(term))})


_TARGET_EDGE_CACHE = None


def _target_edges_cpu():
    global _TARGET_EDGE_CACHE
    if _TARGET_EDGE_CACHE is None:
        mask = masker.mask_img_.get_fdata().astype(bool)
        index = -np.ones(mask.shape, dtype=np.int64)
        index[mask] = np.arange(int(mask.sum()), dtype=np.int64)
        edge_a = []
        edge_b = []
        for axis in range(3):
            lo = [slice(None)] * 3
            hi = [slice(None)] * 3
            lo[axis] = slice(0, -1)
            hi[axis] = slice(1, None)
            both = mask[tuple(lo)] & mask[tuple(hi)]
            edge_a.append(index[tuple(lo)][both])
            edge_b.append(index[tuple(hi)][both])
        a = torch.as_tensor(np.concatenate(edge_a), dtype=torch.long)
        b = torch.as_tensor(np.concatenate(edge_b), dtype=torch.long)
        degree = torch.bincount(torch.cat([a, b]), minlength=int(mask.sum())).float().clamp_min(1.0)
        _TARGET_EDGE_CACHE = (a, b, degree)
    return _TARGET_EDGE_CACHE


def smooth_target_maps(x, steps=TARGET_SMOOTH_STEPS, alpha=TARGET_SMOOTH_ALPHA, batch_size=256):
    if steps <= 0 or alpha <= 0:
        return x
    edge_a, edge_b, degree = _target_edges_cpu()
    edge_a = edge_a.to(x.device)
    edge_b = edge_b.to(x.device)
    degree = degree.to(x.device)
    out_batches = []
    for start in range(0, x.shape[0], batch_size):
        xb = x[start:start + batch_size].clone()
        for _ in range(steps):
            neigh = torch.zeros_like(xb)
            neigh.index_add_(1, edge_a, xb[:, edge_b])
            neigh.index_add_(1, edge_b, xb[:, edge_a])
            xb = (1 - alpha) * xb + alpha * (neigh / degree)
        out_batches.append(xb)
    return torch.cat(out_batches, dim=0)


def normalize_maps(
    x,
    floor=TARGET_FLOOR,
    min_voxels=TARGET_MIN_VOXELS,
    max_voxels=TARGET_MAX_VOXELS,
    gamma=TARGET_GAMMA,
    core_max_voxels=TARGET_CORE_MAX_VOXELS,
    smooth_steps=TARGET_SMOOTH_STEPS,
):
    x = torch.as_tensor(x).float()
    was_1d = x.ndim == 1
    if was_1d:
        x = x[None]
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0)
    x = x / x.amax(dim=1, keepdim=True).clamp_min(1e-6)
    if floor:
        x = x.clone()
        x[x < floor] = 0
    return x.squeeze(0) if was_1d else x


def _normalize_targets(x, floor=None, **kwargs):
    return normalize_maps(x, floor=TARGET_FLOOR if floor is None else floor, **kwargs)


def renormalize_targets(x):
    return normalize_maps(x, floor=0.0, core_max_voxels=None, smooth_steps=0)


def combine_region_maps(maps, *, max_voxels=TARGET_MAX_VOXELS):
    maps = torch.as_tensor(maps).float()
    if maps.ndim == 1:
        combined = maps
    else:
        combined = maps.sum(dim=0).clamp(0, 1)
    return normalize_maps(combined, floor=0.0, max_voxels=max_voxels, core_max_voxels=None, smooth_steps=0)


def _hemi_masks(masker):
    mask = masker.mask_img_.get_fdata().astype(bool)
    ijk = np.argwhere(mask)
    xyz = nib.affines.apply_affine(masker.mask_img_.affine, ijk)
    return xyz[:, 0] < -2, xyz[:, 0] > 2


def _mask_xyz(masker):
    mask = masker.mask_img_.get_fdata().astype(bool)
    ijk = np.argwhere(mask)
    return nib.affines.apply_affine(masker.mask_img_.affine, ijk).astype(np.float32)


DIRECTIONAL_AXES = {
    "anterior": (1, "high"),
    "posterior": (1, "low"),
    "dorsal": (2, "high"),
    "superior": (2, "high"),
    "ventral": (2, "low"),
    "inferior": (2, "low"),
}


def _direction_words(text):
    words = set(re.findall(r"[a-z0-9]+", _clean_text(text).lower()))
    return words & set(DIRECTIONAL_AXES)


def _remove_words(text, words):
    tokens = re.findall(r"[a-z0-9]+", _clean_text(text).lower())
    return _clean_text(" ".join(t for t in tokens if t not in words)).lower()


def _target_stats(x):
    x = torch.as_tensor(x).float()
    support = int((x > 0).sum().item())
    support_02 = int((x > 0.2).sum().item())
    mass = float(x.sum().item())
    if mass <= 0:
        return {"support": 0, "support_02": 0, "mass": 0.0, "mid_mass_frac": 0.0}
    mid = (x > 0.05) & (x < 0.95)
    mid_mass_frac = float(x[mid].sum().item() / mass)
    return {"support": support, "support_02": support_02, "mass": mass, "mid_mass_frac": mid_mass_frac}


def _pick_canonical_index(idxs, atlas_maps):
    idxs = [int(i) for i in idxs]
    if len(idxs) <= 1:
        return idxs[0]

    def key(i):
        stats = _target_stats(atlas_maps[i])
        binary_penalty = stats["mid_mass_frac"] < TRAIN_MIN_MID_MASS_FRAC
        over_penalty = max(0, stats["support"] - TRAIN_MAX_SUPPORT)
        under_penalty = max(0, TRAIN_MIN_SUPPORT - stats["support_02"])
        return (binary_penalty, over_penalty, under_penalty, i, stats["support_02"], stats["support"])

    return sorted(idxs, key=key)[0]


def _select_compact_region_indices(term, idxs, names, atlas_maps):
    groups = {"": [], "left": [], "right": []}
    for idx in sorted(set(int(i) for i in idxs)):
        side, _ = _side_and_base(names[idx])
        groups.setdefault(side, []).append(idx)

    requested_side = _side_from_text(term)
    if requested_side:
        candidates = groups.get(requested_side) or sorted(set(int(i) for i in idxs))
        return [_pick_canonical_index(candidates, atlas_maps)]

    if groups.get(""):
        # A no-side atlas row usually represents the full bilateral parcel.
        return [_pick_canonical_index(groups[""], atlas_maps)]

    selected = []
    for side in ["left", "right"]:
        if groups.get(side):
            selected.append(_pick_canonical_index(groups[side], atlas_maps))
    return selected or [_pick_canonical_index(idxs, atlas_maps)]


def _directional_gate(target, term, matched_names, masker):
    missing = _direction_words(term) - _direction_words(" ".join(matched_names))
    if not missing:
        return target
    xyz = torch.as_tensor(_mask_xyz(masker), dtype=target.dtype, device=target.device)
    out = target.clone()
    for word in sorted(missing):
        axis, polarity = DIRECTIONAL_AXES[word]
        pos = out > 0
        if int(pos.sum().item()) < TARGET_MIN_VOXELS:
            continue
        coords = xyz[:, axis]
        vals = coords[pos].detach().cpu().numpy()
        center_q = 0.55 if polarity == "high" else 0.45
        center = float(np.quantile(vals, center_q))
        scale = max(2.5, float(np.std(vals)) * 0.20)
        z = (coords - center) / scale
        gate = torch.sigmoid(z if polarity == "high" else -z)
        out = out * gate
    return out


def neighbor_edges_for_mask(masker):
    mask = masker.mask_img_.get_fdata().astype(bool)
    index = -np.ones(mask.shape, dtype=np.int64)
    index[mask] = np.arange(int(mask.sum()), dtype=np.int64)
    edge_a = []
    edge_b = []
    for axis in range(3):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis] = slice(0, -1)
        hi[axis] = slice(1, None)
        both = mask[tuple(lo)] & mask[tuple(hi)]
        edge_a.append(index[tuple(lo)][both])
        edge_b.append(index[tuple(hi)][both])
    return np.concatenate(edge_a), np.concatenate(edge_b)


def lateralize_map(x, side, left_mask, right_mask):
    y = torch.as_tensor(x).float().clone()
    if side == "left":
        y[torch.as_tensor(right_mask, dtype=torch.bool)] = 0
    elif side == "right":
        y[torch.as_tensor(left_mask, dtype=torch.bool)] = 0
    return renormalize_targets(y)


def region_target_for_text(term, atlas_maps, lookup, names, left_mask, right_mask):
    side = _side_from_text(term)
    idxs = resolve_region_indices(term, lookup, names)
    if not idxs and side:
        idxs = resolve_region_indices(_strip_side_text(term), lookup, names)
    if not idxs:
        return None, [], side
    matched_names = [names[i] for i in idxs]
    missing_direction = _direction_words(term) - _direction_words(" ".join(matched_names))
    if missing_direction:
        base_term = _remove_words(_strip_side_text(term), missing_direction)
        base_idxs = resolve_region_indices(base_term, lookup, names) if base_term else []
        if base_idxs:
            idxs = base_idxs
    idxs = _select_compact_region_indices(term, idxs, names, atlas_maps)
    matched_names = [names[i] for i in idxs]
    target = combine_region_maps(atlas_maps[torch.as_tensor(idxs, dtype=torch.long)])
    target = _directional_gate(target, term, matched_names, masker)
    if side:
        target = lateralize_map(target, side, left_mask, right_mask)
    else:
        target = renormalize_targets(target)
    if float(target.sum()) <= 0:
        return None, [], side
    return target, sorted(set(idxs)), side


def region_target_for_terms(terms, atlas_maps, lookup, names, left_mask, right_mask):
    targets = []
    idxs = []
    missing = []
    for term in terms:
        target, term_idxs, _ = region_target_for_text(term, atlas_maps, lookup, names, left_mask, right_mask)
        if target is None:
            missing.append(term)
        else:
            targets.append(target)
            idxs.extend(term_idxs)
    if not targets:
        return None, [], missing
    max_voxels = min(targets[0].numel(), TARGET_MAX_VOXELS * max(1, len(targets)))
    return combine_region_maps(torch.stack(targets), max_voxels=max_voxels), sorted(set(idxs)), missing


def side_code_from_meta(pair_meta):
    side = pair_meta.get("side", pd.Series([""] * len(pair_meta))).fillna("").astype(str).str.lower()
    return torch.as_tensor(side.map({"left": -1, "right": 1}).fillna(0).to_numpy(copy=True), dtype=torch.long)


def audit_training_pairs(text, maps, *, pair_meta=None, name="pairs"):
    text_s = pd.Series([_clean_text(t) for t in text])
    x = torch.as_tensor(maps).float()
    support = (x > 0).sum(dim=1).float()
    support_02 = (x > 0.2).sum(dim=1).float()
    mass = x.sum(dim=1).clamp_min(1e-6)
    mid_mass_frac = (x * ((x > 0.05) & (x < 0.95)).float()).sum(dim=1) / mass
    row = {
        "name": name,
        "n": len(text_s),
        "blank_text": int(text_s.eq("").sum()),
        "duplicate_text": int(text_s.map(_text_key).duplicated().sum()),
        "empty_maps": int((support == 0).sum().item()),
        "support_median": float(support.median().item()),
        "support_p95": float(torch.quantile(support, 0.95).item()),
        "support_02_median": float(support_02.median().item()),
        "support_02_p95": float(torch.quantile(support_02, 0.95).item()),
        "mid_mass_frac_median": float(mid_mass_frac.median().item()),
        "low_mid_mass": int((mid_mass_frac < TRAIN_MIN_MID_MASS_FRAC).sum().item()),
        "density_median": float((support / x.shape[1]).median().item()),
    }
    if pair_meta is not None and "source" in pair_meta:
        row["sources"] = ", ".join(f"{k}:{v}" for k, v in pair_meta["source"].value_counts().sort_index().items())
    report = pd.DataFrame([row])
    display(report)
    return report


def clean_training_pairs(text, maps, *, pair_meta=None, min_support=4, max_support_quantile=0.995, **_ignored):
    text_s = pd.Series([_clean_text(t) for t in text])
    meta = pair_meta.reset_index(drop=True).copy() if pair_meta is not None else pd.DataFrame(index=np.arange(len(text_s)))
    if "source" not in meta:
        meta["source"] = "unknown"
    if "side" not in meta:
        meta["side"] = ""
    if "group_key" not in meta:
        meta["group_key"] = text_s.map(_text_key)

    x = renormalize_targets(maps)
    support = (x > 0).sum(dim=1)
    support_02 = (x > 0.2).sum(dim=1)
    mass = x.sum(dim=1).clamp_min(1e-6)
    mid_mass_frac = (x * ((x > 0.05) & (x < 0.95)).float()).sum(dim=1) / mass
    min_support = max(int(min_support), TRAIN_MIN_SUPPORT)
    max_support = min(float(torch.quantile(support.float(), max_support_quantile).item()), float(TRAIN_MAX_SUPPORT))
    keep = (
        text_s.ne("").to_numpy()
        & (support.numpy() >= min_support)
        & (support.numpy() <= max_support)
        & (support_02.numpy() >= 1)
        & (mid_mass_frac.numpy() >= TRAIN_MIN_MID_MASS_FRAC)
    )
    drop_t = torch.as_tensor(~keep, dtype=torch.bool)
    removed = meta.loc[~keep].copy()
    removed["text"] = text_s.loc[~keep].to_numpy()
    removed["support_voxels"] = support[drop_t].numpy().astype(int)
    removed["support_02_voxels"] = support_02[drop_t].numpy().astype(int)
    removed["mid_mass_frac"] = mid_mass_frac[drop_t].numpy()

    keep_t = torch.as_tensor(keep, dtype=torch.bool)
    text_s = text_s.loc[keep].reset_index(drop=True)
    meta = meta.loc[keep].reset_index(drop=True)
    x = x[keep_t]
    support = support[keep_t]
    support_02 = support_02[keep_t]
    mid_mass_frac = mid_mass_frac[keep_t]

    priority = {
        "atlas": 0,
        "atlas_alias": 0,
        "atlas_lateralized_alias": 1,
        "bilateral_atlas": 2,
        "network_seed": 3,
        "network_component_seed": 4,
        "pubmed_text_canonical": 5,
    }
    text_key = text_s.map(_text_key)
    source_priority = meta["source"].astype(str).map(lambda s: priority.get(s, 99)).to_numpy()
    atlas_index = pd.to_numeric(meta.get("atlas_index", pd.Series(np.inf, index=meta.index)), errors="coerce").fillna(np.inf).to_numpy()
    quality = (
        (mid_mass_frac.numpy() < TRAIN_MIN_MID_MASS_FRAC).astype(float) * 1_000_000.0
        + np.maximum(0.0, support.numpy().astype(float) - TRAIN_MAX_SUPPORT) * 1_000.0
        + np.maximum(0.0, TRAIN_MIN_SUPPORT - support_02.numpy().astype(float)) * 1_000.0
        + np.minimum(atlas_index, 1_000_000.0) * 10.0
        + support_02.numpy().astype(float)
    )
    order = np.lexsort((np.arange(len(text_s)), quality, source_priority, text_key.to_numpy()))
    keep_first = pd.Series(text_key.iloc[order]).duplicated().to_numpy() == False
    chosen = order[keep_first]
    text_s = text_s.iloc[chosen].reset_index(drop=True)
    x = x[torch.as_tensor(chosen, dtype=torch.long)]
    meta = meta.iloc[chosen].reset_index(drop=True)
    meta["text"] = text_s.to_numpy()
    meta["n_merged"] = 1
    return text_s.tolist(), renormalize_targets(x), meta, removed


def group_train_val_indices(text, pair_meta, val_frac=0.1, seed=SEED):
    meta = pair_meta.reset_index(drop=True)
    force_train = meta["source"].astype(str).str.contains("atlas|network_seed|lateralized", regex=True).to_numpy()
    groups = meta.get("group_key", pd.Series(text)).astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    is_val = np.zeros(len(meta), dtype=bool)
    candidates = np.flatnonzero(~force_train)
    if len(candidates):
        val_groups = np.array(pd.unique(groups[candidates]), dtype=object)
        rng.shuffle(val_groups)
        val_groups = set(val_groups[:max(1, int(len(val_groups) * val_frac))])
        is_val[candidates] = np.array([g in val_groups for g in groups[candidates]])
    else:
        order = rng.permutation(len(meta))
        is_val[order[:max(1, int(len(meta) * val_frac))]] = True
    print(f"split: {(~is_val).sum():,} train / {is_val.sum():,} val; forced train={force_train.sum():,}")
    return torch.as_tensor(np.flatnonzero(~is_val)), torch.as_tensor(np.flatnonzero(is_val))


def soft_dice_loss(prob, target, eps=1e-6):
    numerator = 2 * (prob * target).sum(dim=1)
    denominator = prob.sum(dim=1) + target.sum(dim=1)
    return (1 - (numerator + eps) / (denominator + eps)).mean()


def positive_recall_loss(prob, target, eps=1e-6):
    pos = target > 0
    missed = ((1 - prob) * pos.float()).sum(dim=1)
    denom = pos.float().sum(dim=1).clamp_min(eps)
    return (missed / denom).mean()


def spatial_smoothness_loss(prob, target=None, edge_a=None, edge_b=None, max_edges=SMOOTH_MAX_EDGES):
    if edge_a is None or edge_b is None or LOSS_SMOOTH_WEIGHT <= 0:
        return prob.new_tensor(0.0)
    n_edges = edge_a.numel()
    if n_edges > max_edges:
        step = math.ceil(n_edges / max_edges)
        edge_a = edge_a[::step][:max_edges]
        edge_b = edge_b[::step][:max_edges]
    pred_grad = prob[:, edge_a] - prob[:, edge_b]
    if target is None:
        active = (prob[:, edge_a] > 0.01) | (prob[:, edge_b] > 0.01)
        return pred_grad.square()[active].mean() if active.any() else pred_grad.square().mean()
    target_grad = target[:, edge_a] - target[:, edge_b]
    active = (target[:, edge_a] > 0) | (target[:, edge_b] > 0) | (prob[:, edge_a] > 0.01) | (prob[:, edge_b] > 0.01)
    grad_match = (pred_grad - target_grad).square()
    return grad_match[active].mean() if active.any() else grad_match.mean()


def mass_loss(prob, target, eps=1e-6):
    target_mass = target.sum(dim=1).clamp_min(eps)
    return ((prob.sum(dim=1) - target_mass) / target_mass).square().mean()


def map_loss(logits, target, side_code=None, left_mask=None, right_mask=None, edge_a=None, edge_b=None):
    prob = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target)
    mse = F.mse_loss(prob, target)
    dice = soft_dice_loss(prob, target)
    recall = positive_recall_loss(prob, target)
    smooth = spatial_smoothness_loss(prob, target=target, edge_a=edge_a, edge_b=edge_b)
    mass = mass_loss(prob, target)
    return LOSS_BCE_WEIGHT * bce + LOSS_MSE_WEIGHT * mse + LOSS_DICE_WEIGHT * dice + LOSS_RECALL_WEIGHT * recall + LOSS_SMOOTH_WEIGHT * smooth + LOSS_MASS_WEIGHT * mass


def laterality_accuracy(prob, side_code, left_mask, right_mask):
    lateral = side_code != 0
    if not lateral.any():
        return float("nan")
    p = prob[lateral]
    s = side_code[lateral].to(prob.device)
    lm = p[:, left_mask].sum(dim=1)
    rm = p[:, right_mask].sum(dim=1)
    correct = torch.where(s < 0, lm > rm, rm > lm)
    return float(correct.float().mean().item())


def spot_check_pairs(text, maps, pair_meta, n=8, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(text), size=min(n, len(text)), replace=False)
    x = torch.as_tensor(maps).float()
    rows = []
    for i in idx:
        rows.append({
            "i": int(i),
            "source": pair_meta.iloc[i].get("source", ""),
            "text": _clean_text(text[i])[:160],
            "support": int((x[i] > 0).sum().item()),
            "mass": float(x[i].sum().item()),
            "pubmed_inside_frac": pair_meta.iloc[i].get("pubmed_inside_frac", np.nan),
        })
    display(pd.DataFrame(rows))



# %% [markdown]
# ## Load the aligned PubMed corpus
#

# %%
# ae = load_model("autoencoder").to("cuda").eval()
df_text = load_dataset("pubmed_text")
images, pmids = load_dataset("pubmed_images")
df_text = df_text[df_text["pmid"].isin(pmids)]
df_text = df_text.sort_values("pmid").reset_index(drop=True)
assert (df_text['pmid'] == pmids).all()

# %% [markdown]
# ## Mine spatially coherent anatomical terms
#

# %%

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import blake2b
from typing import Iterable, Iterator, Sequence
import math
import re
import unicodedata

import numpy as np
from joblib import Parallel, delayed

import xxhash
from scipy.sparse import coo_matrix, csr_matrix

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", flags=re.IGNORECASE)


DEFAULT_BOUNDARY_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
    "for", "from", "has", "have", "having", "in", "into", "is", "it",
    "its", "of", "on", "or", "our", "that", "the", "their", "these",
    "this", "those", "to", "was", "were", "with", "without",
})


@dataclass(frozen=True)
class NgramHit:
    hash: int
    phrase: str
    n: int
    start_token: int
    end_token: int


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("‐", " ")
    text = text.replace("-", " ")
    text = text.replace("–", " ")
    text = text.replace("—", " ")
    text = text.replace("-", " ")
    text = text.replace("/", " ")
    text = text.replace("\\", " ")
    return text.casefold()


def _stable_token_hash(token: str) -> int:
    """
    Stable uint64 token hash.

    xxhash is much faster if installed. blake2b fallback is stable but slower.
    """
    if xxhash is not None:
        return int(xxhash.xxh3_64_intdigest(token))

    digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _splitmix64(x: int) -> np.uint64:
    z = np.uint64(x) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def _iter_chunks(seq: Sequence, chunk_size: int) -> Iterator[tuple[int, Sequence]]:
    for start in range(0, len(seq), chunk_size):
        yield start, seq[start:start + chunk_size]


def _iter_record_chunks(
    records: Sequence[tuple[str | int, str]],
    chunk_size: int,
) -> Iterator[Sequence[tuple[str | int, str]]]:
    for start in range(0, len(records), chunk_size):
        yield records[start:start + chunk_size]


class FastNgramMiner:
    def __init__(
        self,
        *,
        n_min: int = 1,
        n_max: int = 7,
        min_token_len: int = 2,
        boundary_stopwords: Iterable[str] = DEFAULT_BOUNDARY_STOPWORDS,
        drop_numeric_boundary: bool = True,
    ) -> None:
        if n_min < 1:
            raise ValueError("n_min must be >= 1.")
        if n_max < n_min:
            raise ValueError("n_max must be >= n_min.")

        self.n_min = int(n_min)
        self.n_max = int(n_max)
        self.min_token_len = int(min_token_len)
        self.boundary_stopwords = frozenset(x.casefold() for x in boundary_stopwords)
        self.drop_numeric_boundary = bool(drop_numeric_boundary)

        self._position_salts = np.array(
            [_splitmix64(i + 1) for i in range(self.n_max)],
            dtype=np.uint64,
        )
        self._n_salts = np.array(
            [_splitmix64(10_000 + i) for i in range(self.n_max + 1)],
            dtype=np.uint64,
        )

    def tokenize(self, text: str) -> list[str]:
        norm = _normalize_text(text)
        toks = _TOKEN_RE.findall(norm)

        if self.min_token_len <= 1:
            return toks

        return [t for t in toks if len(t) >= self.min_token_len]

    def _valid_boundary(self, token: str) -> bool:
        if token in self.boundary_stopwords:
            return False
        if self.drop_numeric_boundary and token.isdigit():
            return False
        return True

    def _token_hash_array(self, tokens: Sequence[str]) -> np.ndarray:
        return np.fromiter(
            (_stable_token_hash(t) for t in tokens),
            dtype=np.uint64,
            count=len(tokens),
        )

    def ngram_hashes(
        self,
        text: str,
        *,
        unique: bool = True,
    ) -> np.ndarray:
        """
        Fast hash-only n-gram extraction for one document.

        unique=True is what you want for document frequency and doc-level
        phrase-map co-occurrence.
        """
        tokens = self.tokenize(text)
        length = len(tokens)

        if length == 0:
            return np.empty(0, dtype=np.uint64)

        token_hashes = self._token_hash_array(tokens)

        boundary_ok = np.fromiter(
            (self._valid_boundary(t) for t in tokens),
            dtype=bool,
            count=length,
        )

        all_hashes: list[np.ndarray] = []

        for n in range(self.n_min, self.n_max + 1):
            if length < n:
                continue

            n_windows = length - n + 1
            mask = boundary_ok[:n_windows] & boundary_ok[n - 1:]

            if not np.any(mask):
                continue

            windows = np.lib.stride_tricks.sliding_window_view(token_hashes, n)
            hashes = (windows * self._position_salts[:n]).sum(
                axis=1,
                dtype=np.uint64,
            )
            hashes = hashes ^ self._n_salts[n]
            hashes = hashes[mask]

            if hashes.size:
                all_hashes.append(hashes)

        if not all_hashes:
            return np.empty(0, dtype=np.uint64)

        out = np.concatenate(all_hashes)

        if unique:
            out = np.unique(out)

        return out

    def iter_ngram_hits(self, text: str) -> Iterator[NgramHit]:
        """
        Slower string-materializing extraction.

        Use this only after hash candidates have already been filtered.
        """
        tokens = self.tokenize(text)
        length = len(tokens)

        if length == 0:
            return

        token_hashes = self._token_hash_array(tokens)

        boundary_ok = np.fromiter(
            (self._valid_boundary(t) for t in tokens),
            dtype=bool,
            count=length,
        )

        for n in range(self.n_min, self.n_max + 1):
            if length < n:
                continue

            n_windows = length - n + 1
            windows = np.lib.stride_tricks.sliding_window_view(token_hashes, n)
            hashes = (windows * self._position_salts[:n]).sum(
                axis=1,
                dtype=np.uint64,
            )
            hashes = hashes ^ self._n_salts[n]

            valid = boundary_ok[:n_windows] & boundary_ok[n - 1:]

            for start in np.flatnonzero(valid):
                end = start + n
                yield NgramHit(
                    hash=int(hashes[start]),
                    phrase=" ".join(tokens[start:end]),
                    n=n,
                    start_token=int(start),
                    end_token=int(end),
                )

    def count_document_frequency_parallel(
        self,
        records: Sequence[tuple[str | int, str]],
        *,
        n_jobs: int = -1,
        chunk_docs: int = 20_000,
        backend: str = "loky",
        verbose: int = 0,
    ) -> list[dict[str, int]]:
        """
        Parallel document-frequency count over all n-gram hashes.

        Each document contributes at most 1 count per n-gram hash.

        records:
            Sequence of (doc_id, text)

        Returns:
            [{"hash": int, "df": int}, ...]
        """
        chunks = list(_iter_record_chunks(records, chunk_docs))

        results = Parallel(
            n_jobs=n_jobs,
            backend=backend,
            verbose=verbose,
            batch_size=1,
            prefer="processes" if backend == "loky" else None,
        )(
            delayed(_count_df_chunk)(self, chunk)
            for chunk in chunks
        )

        counts: Counter[int] = Counter()

        for values, freqs in results:
            if len(values) == 0:
                continue
            counts.update({
                int(h): int(c)
                for h, c in zip(values, freqs, strict=False)
            })

        return [
            {"hash": int(h), "df": int(c)}
            for h, c in counts.most_common()
        ]

    def collect_phrase_examples_parallel(
        self,
        records: Sequence[tuple[str | int, str]],
        candidate_hashes: Iterable[int],
        *,
        n_jobs: int = -1,
        chunk_docs: int = 20_000,
        max_examples_per_hash: int = 3,
        backend: str = "loky",
        verbose: int = 0,
    ) -> list[dict[str, object]]:
        """
        Recover representative phrase strings for candidate hashes.

        This is intentionally separate from DF counting so you do not store
        every possible n-gram string in RAM.
        """
        candidate_set = frozenset(int(h) for h in candidate_hashes)
        chunks = list(_iter_record_chunks(records, chunk_docs))

        results = Parallel(
            n_jobs=n_jobs,
            backend=backend,
            verbose=verbose,
            batch_size=1,
            prefer="processes" if backend == "loky" else None,
        )(
            delayed(_collect_examples_chunk)(
                self,
                chunk,
                candidate_set,
                max_examples_per_hash,
            )
            for chunk in chunks
        )

        merged: dict[int, list[dict[str, object]]] = defaultdict(list)

        for chunk_examples in results:
            for h, vals in chunk_examples.items():
                remaining = max_examples_per_hash - len(merged[h])
                if remaining > 0:
                    merged[h].extend(vals[:remaining])

        out: list[dict[str, object]] = []
        for vals in merged.values():
            out.extend(vals)

        return out

    def build_candidate_incidence_parallel(
        self,
        texts: Sequence[str],
        candidate_hashes: Sequence[int],
        *,
        n_jobs: int = -1,
        chunk_docs: int = 5_000,
        backend: str = "loky",
        verbose: int = 0,
        dtype: type = np.float32,
    ) -> csr_matrix:
        """
        Build sparse document × candidate-ngram incidence matrix.

        Shape:
            n_docs × n_candidate_hashes

        Entry:
            1 if candidate n-gram occurs in document, else 0.

        This is the clean bridge between text mining and brain-map scoring.
        """
        candidate_hashes_arr = np.asarray(candidate_hashes, dtype=np.uint64)

        if candidate_hashes_arr.ndim != 1:
            raise ValueError("candidate_hashes must be 1D.")

        n_docs = len(texts)
        n_candidates = len(candidate_hashes_arr)

        if n_candidates == 0:
            return csr_matrix((n_docs, 0), dtype=dtype)

        order = np.argsort(candidate_hashes_arr)
        sorted_hashes = candidate_hashes_arr[order]
        sorted_cols = np.arange(n_candidates, dtype=np.int64)[order]

        chunks = list(_iter_chunks(texts, chunk_docs))

        results = Parallel(
            n_jobs=n_jobs,
            backend=backend,
            verbose=verbose,
            batch_size=1,
            prefer="processes" if backend == "loky" else None,
        )(
            delayed(_candidate_incidence_chunk)(
                self,
                chunk_texts,
                offset,
                sorted_hashes,
                sorted_cols,
            )
            for offset, chunk_texts in chunks
        )

        row_parts: list[np.ndarray] = []
        col_parts: list[np.ndarray] = []

        for rows, cols in results:
            if rows.size:
                row_parts.append(rows)
                col_parts.append(cols)

        if not row_parts:
            return csr_matrix((n_docs, n_candidates), dtype=dtype)

        row_idx = np.concatenate(row_parts).astype(np.int64, copy=False)
        col_idx = np.concatenate(col_parts).astype(np.int64, copy=False)
        data = np.ones(row_idx.shape[0], dtype=dtype)

        mat = coo_matrix(
            (data, (row_idx, col_idx)),
            shape=(n_docs, n_candidates),
            dtype=dtype,
        ).tocsr()

        mat.sum_duplicates()
        mat.data[:] = 1

        return mat


def _count_df_chunk(
    miner: FastNgramMiner,
    chunk: Sequence[tuple[str | int, str]],
) -> tuple[np.ndarray, np.ndarray]:
    parts: list[np.ndarray] = []

    for _, text in chunk:
        hashes = miner.ngram_hashes(text, unique=True)
        if hashes.size:
            parts.append(hashes)

    if not parts:
        return (
            np.empty(0, dtype=np.uint64),
            np.empty(0, dtype=np.int64),
        )

    arr = np.concatenate(parts)
    values, freqs = np.unique(arr, return_counts=True)

    return values.astype(np.uint64, copy=False), freqs.astype(np.int64, copy=False)


def _collect_examples_chunk(
    miner: FastNgramMiner,
    chunk: Sequence[tuple[str | int, str]],
    candidate_hashes: frozenset[int],
    max_examples_per_hash: int,
) -> dict[int, list[dict[str, object]]]:
    examples: dict[int, list[dict[str, object]]] = defaultdict(list)

    for doc_id, text in chunk:
        seen_in_doc: set[int] = set()

        for hit in miner.iter_ngram_hits(text):
            h = int(hit.hash)

            if h not in candidate_hashes:
                continue
            if h in seen_in_doc:
                continue
            if len(examples[h]) >= max_examples_per_hash:
                continue

            examples[h].append({
                "hash": h,
                "phrase": hit.phrase,
                "n": hit.n,
                "doc_id": doc_id,
                "start_token": hit.start_token,
                "end_token": hit.end_token,
            })
            seen_in_doc.add(h)

    return dict(examples)


def _candidate_incidence_chunk(
    miner: FastNgramMiner,
    texts: Sequence[str],
    offset: int,
    sorted_hashes: np.ndarray,
    sorted_cols: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []

    n_candidates = len(sorted_hashes)

    for local_i, text in enumerate(texts):
        hashes = miner.ngram_hashes(text, unique=True)

        if hashes.size == 0:
            continue

        pos = np.searchsorted(sorted_hashes, hashes)
        valid = pos < n_candidates

        if not np.any(valid):
            continue

        valid_pos = pos[valid]
        valid_hashes = hashes[valid]

        matched = sorted_hashes[valid_pos] == valid_hashes

        if not np.any(matched):
            continue

        cols = sorted_cols[valid_pos[matched]]

        if cols.size == 0:
            continue

        rows = np.full(cols.shape[0], offset + local_i, dtype=np.int64)

        row_parts.append(rows)
        col_parts.append(cols.astype(np.int64, copy=False))

    if not row_parts:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    return np.concatenate(row_parts), np.concatenate(col_parts)


def row_l2_normalize(
    maps: np.ndarray,
    *,
    eps: float = 1e-8,
    copy: bool = True,
) -> np.ndarray:
    maps = np.asarray(maps, dtype=np.float32)

    if copy:
        maps = maps.copy()

    norms = np.linalg.norm(maps, axis=1, keepdims=True)
    maps /= np.maximum(norms, eps)

    return maps


def score_spatial_coherence_from_incidence(
    *,
    incidence: csr_matrix,
    maps: np.ndarray,
    candidate_hashes: Sequence[int],
    candidate_block_size: int = 2048,
    maps_are_normalized: bool = False,
    eps: float = 1e-8,
) -> list[dict[str, int | float]]:
    """
    Score candidate n-grams by spatial coherence.

    Parameters
    ----------
    incidence:
        Sparse document × candidate matrix.
    maps:
        Dense array of shape n_docs × n_features.
        Rows are brain maps aligned to documents.
    candidate_hashes:
        Candidate hash list aligned to incidence columns.
    candidate_block_size:
        Controls memory use of the dense block:
            candidate_block_size × n_features
    maps_are_normalized:
        If False, maps are row-L2-normalized before scoring.
        If True, assumes you already normalized rows.

    Score
    -----
    For candidate g:

        sum_map[g] = sum of normalized brain maps for docs containing g
        df[g] = number of docs containing g
        coherence[g] = ||sum_map[g]|| / df[g]

    High coherence means documents containing the phrase have spatially similar
    maps. This is what should pull out anatomical phrases from raw n-grams.
    """
    if incidence.ndim != 2:
        raise ValueError("incidence must be a 2D sparse matrix.")

    maps = np.asarray(maps, dtype=np.float32)

    if maps.ndim != 2:
        raise ValueError("maps must have shape n_docs × n_features.")

    if incidence.shape[0] != maps.shape[0]:
        raise ValueError("incidence rows must match maps rows.")

    if incidence.shape[1] != len(candidate_hashes):
        raise ValueError("incidence columns must match len(candidate_hashes).")

    if maps_are_normalized:
        maps_norm = maps
    else:
        maps_norm = row_l2_normalize(maps, eps=eps, copy=True)

    n_candidates = incidence.shape[1]
    candidate_hashes_arr = np.asarray(candidate_hashes, dtype=np.uint64)

    out: list[dict[str, int | float]] = []

    for start in range(0, n_candidates, candidate_block_size):
        end = min(start + candidate_block_size, n_candidates)
        block = incidence[:, start:end]

        dfs = np.asarray(block.sum(axis=0)).ravel().astype(np.int64)

        if np.all(dfs == 0):
            for j in range(start, end):
                out.append({
                    "hash": int(candidate_hashes_arr[j]),
                    "df": 0,
                    "sum_norm": 0.0,
                    "coherence": 0.0,
                    "score": 0.0,
                })
            continue

        sum_maps = block.T @ maps_norm
        sum_maps = np.asarray(sum_maps, dtype=np.float32)

        sum_norms = np.linalg.norm(sum_maps, axis=1)

        coherence = np.divide(
            sum_norms,
            np.maximum(dfs, 1),
            out=np.zeros_like(sum_norms, dtype=np.float32),
            where=dfs > 0,
        )

        score = coherence * np.log1p(dfs)

        for local_j, df, norm, coh, s in zip(
            range(end - start),
            dfs,
            sum_norms,
            coherence,
            score,
            strict=False,
        ):
            global_j = start + local_j
            out.append({
                "hash": int(candidate_hashes_arr[global_j]),
                "df": int(df),
                "sum_norm": float(norm),
                "coherence": float(coh),
                "score": float(s),
            })

    out.sort(key=lambda r: (r["score"], r["coherence"], r["df"]), reverse=True)
    return out


def merge_scores_with_examples(
    scores: Sequence[dict[str, int | float]],
    examples: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """
    Convenience join: attach phrase examples to scored hashes.
    """
    hash_to_phrases: dict[int, list[str]] = defaultdict(list)

    for ex in examples:
        h = int(ex["hash"])
        phrase = str(ex["phrase"])

        if phrase not in hash_to_phrases[h]:
            hash_to_phrases[h].append(phrase)

    merged: list[dict[str, object]] = []

    for row in scores:
        h = int(row["hash"])
        merged.append({
            **row,
            "phrases": hash_to_phrases.get(h, []),
            "phrase": hash_to_phrases.get(h, [""])[0],
        })

    return merged


# %%

from collections import defaultdict
from typing import Sequence
import re

import numpy as np
from scipy.sparse import csr_matrix


_BAD_LABEL_RE = re.compile(
    r"^(?:"
    r"study|studies|results|result|method|methods|analysis|analyses|"
    r"significant|significantly|activation|activations|activated|"
    r"increase|increased|decrease|decreased|effect|effects|"
    r"task|tasks|control|controls|condition|conditions|"
    r"subjects|patients|participants|group|groups|"
    r"brain|cerebral|regional|local"
    r")$"
)


def _example_lookup(
    examples: Sequence[dict[str, object]],
) -> dict[int, list[dict[str, object]]]:
    out: dict[int, list[dict[str, object]]] = defaultdict(list)

    for ex in examples:
        h = int(ex["hash"])
        out[h].append(ex)

    return dict(out)


def _best_label_for_hash(
    h: int,
    examples_by_hash: dict[int, list[dict[str, object]]],
) -> tuple[str, int]:
    """
    Returns:
        label, n_tokens

    If multiple examples exist for a hash, prefer the most common phrase string.
    Hash collisions are possible in principle but should be extremely rare with
    uint64 hashes. If you see multiple unrelated phrase strings for one hash,
    inspect that hash manually.
    """
    examples = examples_by_hash.get(h, [])

    if not examples:
        return "", 0

    phrase_counts: dict[str, int] = defaultdict(int)
    phrase_n: dict[str, int] = {}

    for ex in examples:
        phrase = str(ex.get("phrase", "")).strip()
        n = int(ex.get("n", len(phrase.split())))

        if not phrase:
            continue

        phrase_counts[phrase] += 1
        phrase_n[phrase] = n

    if not phrase_counts:
        return "", 0

    label = max(
        phrase_counts,
        key=lambda p: (
            phrase_counts[p],
            len(p.split()),
            len(p),
        ),
    )

    return label, phrase_n[label]


def _valid_label(
    label: str,
    *,
    min_chars: int,
    max_chars: int,
    drop_bad_unigrams: bool,
) -> bool:
    label = label.strip()

    if len(label) < min_chars:
        return False

    if len(label) > max_chars:
        return False

    toks = label.split()

    if not toks:
        return False

    if drop_bad_unigrams and len(toks) == 1 and _BAD_LABEL_RE.match(label):
        return False

    return True


def make_filtered_ngram_matrix(
    *,
    scores: Sequence[dict[str, int | float]],
    examples: Sequence[dict[str, object]],
    incidence: csr_matrix,
    candidate_hashes: Sequence[int],
    min_df: int = 20,
    max_df_frac: float | None = 0.20,
    min_coherence: float | None = None,
    min_score: float | None = None,
    top_k: int | None = 5_000,
    min_n: int = 1,
    max_n: int = 7,
    min_chars: int = 3,
    max_chars: int = 120,
    drop_bad_unigrams: bool = True,
    dedupe_labels: bool = True,
) -> tuple[np.ndarray, csr_matrix, list[dict[str, object]]]:
    """
    Filter scored n-grams and return aligned labels + sparse boolean matrix.

    Parameters
    ----------
    scores:
        Output from score_spatial_coherence_from_incidence(...).
    examples:
        Output from miner.collect_phrase_examples_parallel(...).
        Used to recover human-readable phrase labels.
    incidence:
        Sparse document × candidate matrix from:
            miner.build_candidate_incidence_parallel(...)
    candidate_hashes:
        Candidate hashes aligned to columns of incidence.
    min_df:
        Minimum document frequency.
    max_df_frac:
        Optional maximum document-frequency fraction.
        Example: 0.20 drops phrases appearing in >20% of papers.
    min_coherence:
        Optional minimum spatial coherence.
    min_score:
        Optional minimum combined score.
    top_k:
        Optional maximum number of retained n-grams after filtering/ranking.
    min_n, max_n:
        Keep only labels with this token-length range.
    min_chars, max_chars:
        Keep labels within this character-length range.
    drop_bad_unigrams:
        Drop generic one-word labels like "study", "activation", "task".
    dedupe_labels:
        If True, keep only the highest-scoring row for duplicate phrase labels.

    Returns
    -------
    labels:
        np.ndarray of shape (n_selected,), dtype=str.
        labels[j] names column j of X_bool.
    X_bool:
        sparse csr_matrix of shape (n_docs, n_selected), dtype=bool.
        X_bool[i, j] is True if labels[j] occurs in paper i.
    rows:
        List of metadata rows aligned to labels / X_bool columns.
        rows[j]["label"] == labels[j].
    """
    if not isinstance(incidence, csr_matrix):
        incidence = incidence.tocsr()

    n_docs, n_candidates = incidence.shape

    if len(candidate_hashes) != n_candidates:
        raise ValueError("len(candidate_hashes) must match incidence.shape[1].")

    candidate_hashes_arr = np.asarray(candidate_hashes, dtype=np.uint64)
    hash_to_col = {
        int(h): int(i)
        for i, h in enumerate(candidate_hashes_arr)
    }

    examples_by_hash = _example_lookup(examples)

    max_df = None
    if max_df_frac is not None:
        if not (0 < max_df_frac <= 1):
            raise ValueError("max_df_frac must be in (0, 1].")
        max_df = int(np.floor(max_df_frac * n_docs))

    candidate_rows: list[dict[str, object]] = []
    seen_labels: set[str] = set()

    # scores are usually already sorted high-to-low, but sort again defensively.
    sorted_scores = sorted(
        scores,
        key=lambda r: (
            float(r.get("score", 0.0)),
            float(r.get("coherence", 0.0)),
            int(r.get("df", 0)),
        ),
        reverse=True,
    )

    for row in sorted_scores:
        h = int(row["hash"])

        col = hash_to_col.get(h)
        if col is None:
            continue

        df = int(row.get("df", 0))
        coherence = float(row.get("coherence", 0.0))
        score = float(row.get("score", 0.0))

        if df < min_df:
            continue

        if max_df is not None and df > max_df:
            continue

        if min_coherence is not None and coherence < min_coherence:
            continue

        if min_score is not None and score < min_score:
            continue

        label, n = _best_label_for_hash(h, examples_by_hash)

        if not label:
            continue

        if n < min_n or n > max_n:
            continue

        if not _valid_label(
            label,
            min_chars=min_chars,
            max_chars=max_chars,
            drop_bad_unigrams=drop_bad_unigrams,
        ):
            continue

        if dedupe_labels:
            label_key = label.casefold()
            if label_key in seen_labels:
                continue
            seen_labels.add(label_key)

        candidate_rows.append({
            "hash": h,
            "label": label,
            "n": n,
            "df": df,
            "coherence": coherence,
            "score": score,
            "original_col": col,
        })

        if top_k is not None and len(candidate_rows) >= top_k:
            break

    selected_cols = np.array(
        [int(r["original_col"]) for r in candidate_rows],
        dtype=np.int64,
    )

    labels = np.array(
        [str(r["label"]) for r in candidate_rows],
        dtype=object,
    )

    if selected_cols.size == 0:
        X_bool = csr_matrix((n_docs, 0), dtype=bool)
        return labels, X_bool, candidate_rows

    X_bool = incidence[:, selected_cols].astype(bool)
    X_bool.eliminate_zeros()

    # Add final aligned column index after filtering.
    for j, row in enumerate(candidate_rows):
        row["column"] = j

    return labels, X_bool, candidate_rows


def sparse_bool_to_dense_if_small(
    X_bool: csr_matrix,
    *,
    max_cells: int = 50_000_000,
) -> np.ndarray:
    """
    Convert sparse bool matrix to dense only if it is small enough.

    50M bool cells is about 50 MB. For large corpora, keep CSR sparse.
    """
    n_cells = X_bool.shape[0] * X_bool.shape[1]

    if n_cells > max_cells:
        raise ValueError(
            f"Dense matrix would have {n_cells:,} cells. "
            f"Keep it sparse or raise max_cells explicitly."
        )

    return X_bool.toarray().astype(bool, copy=False)


# %%
miner = FastNgramMiner(n_min=1, n_max=7)

abstracts = (df_text["name"] + "." + df_text["description"]).tolist()#[:1000]
pmids = df_text["pmid"].tolist()#[:1000]
records = list(zip(pmids, abstracts))

df_rows = miner.count_document_frequency_parallel(
    records,
    n_jobs=16,
    chunk_docs=20_000,
)

candidate_hashes = [
    row["hash"]
    for row in df_rows
    if row["df"] >= 25
]

examples = miner.collect_phrase_examples_parallel(
    records,
    candidate_hashes,
    n_jobs=16,
    chunk_docs=20_000,
    max_examples_per_hash=3,
)

incidence = miner.build_candidate_incidence_parallel(
    texts=abstracts,
    candidate_hashes=candidate_hashes,
    n_jobs=16,
    chunk_docs=5_000,
)

# %%
scores = score_spatial_coherence_from_incidence(
    incidence=incidence,
    maps=images,
    candidate_hashes=candidate_hashes,
    candidate_block_size=2048,
    maps_are_normalized=False,
)

# %%
labels, X_bool, feature_rows = make_filtered_ngram_matrix(
    scores=scores,
    examples=examples,
    incidence=incidence,
    candidate_hashes=candidate_hashes,
    min_df=20,
    max_df_frac=0.2,
    min_coherence=None,
    min_score=None,
    top_k=50_000,
    min_n=1,
    max_n=7,
)

# %% [markdown]
# ## Construct atlas parcels and parcel-term associations
#

# %%

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, csc_matrix


OverlapMode = Literal["mass_fraction", "mean_in_parcel", "cosine"]


@dataclass(frozen=True)
class ParcelNgramConfig:
    overlap_mode: OverlapMode = "mass_fraction"
    min_overlap: float = 0.02
    top_k_parcels_per_paper: int | None = None
    top_k_ngrams_per_parcel: int = 50
    max_pmids_per_parcel: int | None = None
    positive_only: bool = True
    rank_by: Literal["count", "weighted_count", "lift", "count_lift"] = "count_lift"
    min_parcel_docs: int = 5
    min_ngram_count_in_parcel: int = 2
    lift_alpha: float = 1.0


def _to_numpy(x) -> np.ndarray:
    """
    Convert NumPy / pandas / torch-like arrays to NumPy without silently
    producing object arrays.
    """
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def _as_float32_array(x, *, name: str) -> np.ndarray:
    arr = _to_numpy(x)

    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}.")

    return arr.astype(np.float32, copy=False)


def _as_name_list(names: Sequence) -> list[str]:
    if hasattr(names, "tolist"):
        names = names.tolist()
    return [str(x) for x in names]


def _prepare_maps(
    brain_maps,
    *,
    positive_only: bool,
) -> np.ndarray:
    maps = _as_float32_array(brain_maps, name="brain_maps")

    if positive_only:
        maps = np.maximum(maps, 0.0)

    return maps


def _prepare_masks(atlas_masks) -> np.ndarray:
    masks = _as_float32_array(atlas_masks, name="atlas_masks")
    return (masks > 0).astype(np.float32, copy=False)


def build_paper_parcel_overlap_matrix(
    *,
    brain_maps,
    atlas_masks,
    config: ParcelNgramConfig = ParcelNgramConfig(),
    chunk_docs: int = 2048,
    eps: float = 1e-8,
) -> csr_matrix:
    """
    Build sparse paper × atlas-parcel overlap matrix.

    Important:
    This preserves empty atlas parcels instead of dropping them. Empty parcels
    get zero overlap with every paper, which keeps columns aligned with
    atlas_parcel_names.
    """
    maps = _prepare_maps(brain_maps, positive_only=config.positive_only)
    masks = _prepare_masks(atlas_masks)

    n_docs, n_voxels = maps.shape
    n_parcels, n_mask_voxels = masks.shape

    if n_voxels != n_mask_voxels:
        raise ValueError(
            f"brain_maps.shape[1]={n_voxels} does not match "
            f"atlas_masks.shape[1]={n_mask_voxels}."
        )

    mask_sizes = masks.sum(axis=1).astype(np.float32)

    if config.overlap_mode == "cosine":
        mask_norms = np.sqrt(mask_sizes).clip(min=eps)
        masks_for_dot = masks / mask_norms[:, None]
    else:
        masks_for_dot = masks

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []

    masks_t = np.ascontiguousarray(masks_for_dot.T)

    for start in range(0, n_docs, chunk_docs):
        end = min(start + chunk_docs, n_docs)
        block = maps[start:end]

        raw_overlap = block @ masks_t

        if config.overlap_mode == "mass_fraction":
            denom = block.sum(axis=1, keepdims=True).clip(min=eps)
            scores = raw_overlap / denom

        elif config.overlap_mode == "mean_in_parcel":
            scores = raw_overlap / mask_sizes[None, :].clip(min=eps)

        elif config.overlap_mode == "cosine":
            map_norms = np.linalg.norm(block, axis=1, keepdims=True).clip(min=eps)
            scores = raw_overlap / map_norms

        else:
            raise ValueError(f"Unknown overlap_mode: {config.overlap_mode}")

        scores = np.asarray(scores, dtype=np.float32)

        if config.top_k_parcels_per_paper is not None:
            k = min(config.top_k_parcels_per_paper, scores.shape[1])

            if k > 0:
                keep_idx = np.argpartition(scores, kth=scores.shape[1] - k, axis=1)[:, -k:]
                keep_mask = np.zeros(scores.shape, dtype=bool)
                local_rows = np.arange(scores.shape[0])[:, None]
                keep_mask[local_rows, keep_idx] = True
                scores = np.where(keep_mask, scores, 0.0)

        hit_r, hit_c = np.nonzero(scores >= config.min_overlap)

        if hit_r.size == 0:
            continue

        rows.append(hit_r.astype(np.int64) + start)
        cols.append(hit_c.astype(np.int64))
        vals.append(scores[hit_r, hit_c].astype(np.float32))

    if not rows:
        return csr_matrix((n_docs, n_parcels), dtype=np.float32)

    row_idx = np.concatenate(rows)
    col_idx = np.concatenate(cols)
    data = np.concatenate(vals)

    out = csr_matrix(
        (data, (row_idx, col_idx)),
        shape=(n_docs, n_parcels),
        dtype=np.float32,
    )
    out.sum_duplicates()

    return out


def _make_bool_copy(x: csr_matrix) -> csr_matrix:
    y = x.copy().astype(bool)
    y.eliminate_zeros()
    return y


def _top_pmids_for_parcel(
    paper_parcel_scores_csc: csc_matrix,
    parcel_col: int,
    pmids: Sequence[str | int],
    *,
    max_pmids: int | None,
) -> list[str | int]:
    col = paper_parcel_scores_csc.getcol(parcel_col)

    if col.nnz == 0:
        return []

    doc_idx = col.indices
    scores = col.data
    order = np.argsort(scores)[::-1]

    if max_pmids is not None:
        order = order[:max_pmids]

    return [pmids[int(doc_idx[i])] for i in order]


def _top_ngrams_for_parcel(
    *,
    parcel_idx: int,
    count_matrix: csr_matrix,
    weighted_matrix: csr_matrix,
    labels: Sequence[str],
    global_ngram_df: np.ndarray,
    n_docs_total: int,
    n_docs_parcel: int,
    config: ParcelNgramConfig,
) -> tuple[list[str], list[dict[str, object]]]:
    row_counts = count_matrix.getrow(parcel_idx)

    if row_counts.nnz == 0:
        return [], []

    term_cols = row_counts.indices
    counts = row_counts.data.astype(np.float32)

    keep = counts >= config.min_ngram_count_in_parcel
    term_cols = term_cols[keep]
    counts = counts[keep]

    if term_cols.size == 0:
        return [], []

    weighted_row = weighted_matrix.getrow(parcel_idx)
    weighted_lookup = dict(zip(weighted_row.indices.tolist(), weighted_row.data.tolist()))

    weighted_counts = np.array(
        [weighted_lookup.get(int(c), 0.0) for c in term_cols],
        dtype=np.float32,
    )

    alpha = float(config.lift_alpha)

    p_region = (counts + alpha) / (float(n_docs_parcel) + 2.0 * alpha)
    p_global = (global_ngram_df[term_cols].astype(np.float32) + alpha) / (
        float(n_docs_total) + 2.0 * alpha
    )

    lift = p_region / np.maximum(p_global, 1e-12)
    log_lift = np.log(lift)

    if config.rank_by == "count":
        rank_score = counts
    elif config.rank_by == "weighted_count":
        rank_score = weighted_counts
    elif config.rank_by == "lift":
        rank_score = log_lift
    elif config.rank_by == "count_lift":
        rank_score = counts * np.maximum(log_lift, 0.0)
    else:
        raise ValueError(f"Unknown rank_by: {config.rank_by}")

    order = np.argsort(rank_score)[::-1]
    order = order[: config.top_k_ngrams_per_parcel]

    rows: list[dict[str, object]] = []
    ngrams: list[str] = []

    for idx in order:
        col = int(term_cols[idx])
        label = str(labels[col])
        ngrams.append(label)

        rows.append({
            "ngram": label,
            "ngram_col": col,
            "count": int(counts[idx]),
            "weighted_count": float(weighted_counts[idx]),
            "global_df": int(global_ngram_df[col]),
            "lift": float(lift[idx]),
            "log_lift": float(log_lift[idx]),
            "rank_score": float(rank_score[idx]),
        })

    return ngrams, rows

def make_parcel_ngram_dataframe(
    *,
    brain_maps,
    atlas_masks,
    atlas_parcel_names: Sequence[str],
    labels: Sequence[str],
    X_bool: csr_matrix,
    pmids: Sequence[str | int],
    config: ParcelNgramConfig = ParcelNgramConfig(),
    chunk_docs: int = 2048,
) -> pd.DataFrame:
    """
    Construct a parcel-level n-gram dataframe.

    Output columns:
        atlas_parcel_names
        list_of_ngrams
        list_of_pmids
        n_parcel_pmids
        n_ngrams
        n_candidate_ngram_hits
        max_ngram_count
        ngram_stats
        empty_mask

    Important fix:
        Do NOT compute count_matrix as bool @ bool. Cast both matrices to
        numeric first, otherwise sparse multiplication may return boolean
        presence instead of integer-like counts.
    """
    if not isinstance(X_bool, csr_matrix):
        X_bool = X_bool.tocsr()

    X_bool = X_bool.astype(bool)
    X_bool.eliminate_zeros()

    maps = _as_float32_array(brain_maps, name="brain_maps")
    masks = _prepare_masks(atlas_masks)
    atlas_parcel_names = _as_name_list(atlas_parcel_names)
    labels = _as_name_list(labels)

    if maps.shape[0] != X_bool.shape[0]:
        raise ValueError(
            f"brain_maps rows ({maps.shape[0]}) must match "
            f"X_bool rows ({X_bool.shape[0]})."
        )

    if len(pmids) != maps.shape[0]:
        raise ValueError(
            f"len(pmids) ({len(pmids)}) must match brain_maps.shape[0] ({maps.shape[0]})."
        )

    if len(labels) != X_bool.shape[1]:
        raise ValueError(
            f"len(labels) ({len(labels)}) must match X_bool.shape[1] ({X_bool.shape[1]})."
        )

    if len(atlas_parcel_names) != masks.shape[0]:
        raise ValueError(
            f"len(atlas_parcel_names) ({len(atlas_parcel_names)}) must match "
            f"atlas_masks.shape[0] ({masks.shape[0]})."
        )

    paper_parcel_scores = build_paper_parcel_overlap_matrix(
        brain_maps=maps,
        atlas_masks=masks,
        config=config,
        chunk_docs=chunk_docs,
    )

    if paper_parcel_scores.shape[1] != len(atlas_parcel_names):
        raise RuntimeError(
            "Internal alignment error: paper_parcel_scores columns do not match "
            "atlas_parcel_names."
        )

    # Boolean document membership for each parcel.
    paper_parcel_bool = _make_bool_copy(paper_parcel_scores)

    # Critical fix: convert bool matrices to numeric before matrix multiplication.
    # Otherwise bool @ bool can produce boolean presence rather than counts.
    paper_parcel_count = paper_parcel_bool.astype(np.float32)
    X_count = X_bool.astype(np.float32)

    # parcel × ngram: number of papers overlapping parcel AND containing ngram.
    count_matrix = (paper_parcel_count.T @ X_count).tocsr()
    count_matrix.sum_duplicates()

    # parcel × ngram: overlap-weighted phrase count.
    weighted_matrix = (paper_parcel_scores.T @ X_count).tocsr()
    weighted_matrix.sum_duplicates()

    parcel_doc_counts = np.asarray(paper_parcel_bool.sum(axis=0)).ravel().astype(np.int64)
    global_ngram_df = np.asarray(X_bool.sum(axis=0)).ravel().astype(np.int64)
    mask_sizes = masks.sum(axis=1).astype(np.int64)

    paper_parcel_scores_csc = paper_parcel_scores.tocsc()

    rows: list[dict[str, object]] = []

    for parcel_idx, parcel_name in enumerate(atlas_parcel_names):
        n_parcel_docs = int(parcel_doc_counts[parcel_idx])
        empty_mask = bool(mask_sizes[parcel_idx] == 0)

        row_counts = count_matrix.getrow(parcel_idx)
        n_candidate_ngram_hits = int(row_counts.nnz)
        max_ngram_count = float(row_counts.data.max()) if row_counts.nnz else 0.0

        if n_parcel_docs < config.min_parcel_docs:
            rows.append({
                "atlas_parcel_names": parcel_name,
                "list_of_ngrams": [],
                "list_of_pmids": [],
                "n_parcel_pmids": n_parcel_docs,
                "n_ngrams": 0,
                "n_candidate_ngram_hits": n_candidate_ngram_hits,
                "max_ngram_count": max_ngram_count,
                "ngram_stats": [],
                "empty_mask": empty_mask,
            })
            continue

        list_of_ngrams, ngram_stats = _top_ngrams_for_parcel(
            parcel_idx=parcel_idx,
            count_matrix=count_matrix,
            weighted_matrix=weighted_matrix,
            labels=labels,
            global_ngram_df=global_ngram_df,
            n_docs_total=maps.shape[0],
            n_docs_parcel=n_parcel_docs,
            config=config,
        )

        list_of_pmids = _top_pmids_for_parcel(
            paper_parcel_scores_csc,
            parcel_idx,
            pmids,
            max_pmids=config.max_pmids_per_parcel,
        )

        rows.append({
            "atlas_parcel_names": parcel_name,
            "list_of_ngrams": list_of_ngrams,
            "list_of_pmids": list_of_pmids,
            "n_parcel_pmids": n_parcel_docs,
            "n_ngrams": len(list_of_ngrams),
            "n_candidate_ngram_hits": n_candidate_ngram_hits,
            "max_ngram_count": max_ngram_count,
            "ngram_stats": ngram_stats,
            "empty_mask": empty_mask,
        })

    return pd.DataFrame(rows)

def concatenate_atlases(
    atlas_dict: dict[str, tuple[object, Sequence[str]]],
    *,
    name_sep: str = "::",
) -> tuple[np.ndarray, list[str]]:
    """
    Concatenate many atlases while preserving parcel/name alignment.

    atlas_dict:
        {
            "AtlasName": (atlas_masks, parcel_names),
            ...
        }

    Each atlas_masks must be shape:
        n_parcels × n_voxels
    """
    masks_out: list[np.ndarray] = []
    names_out: list[str] = []

    n_voxels_expected: int | None = None

    for atlas_name, (masks, names) in atlas_dict.items():
        masks_arr = _prepare_masks(masks)
        names_list = _as_name_list(names)

        if len(names_list) != masks_arr.shape[0]:
            raise ValueError(
                f"{atlas_name}: len(names)={len(names_list)} must match "
                f"masks.shape[0]={masks_arr.shape[0]}."
            )

        if n_voxels_expected is None:
            n_voxels_expected = masks_arr.shape[1]
        elif masks_arr.shape[1] != n_voxels_expected:
            raise ValueError(
                f"{atlas_name}: voxel dimension {masks_arr.shape[1]} does not "
                f"match expected {n_voxels_expected}."
            )

        masks_out.append(masks_arr)
        names_out.extend([
            f"{atlas_name}{name_sep}{name}"
            for name in names_list
        ])

    if not masks_out:
        raise ValueError("atlas_dict is empty.")

    all_masks = np.vstack(masks_out).astype(np.float32, copy=False)

    if all_masks.shape[0] != len(names_out):
        raise RuntimeError("Internal alignment error after atlas concatenation.")

    return all_masks, names_out


from dataclasses import dataclass
from typing import Sequence
import re

import numpy as np
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class SyntheticMapConfig:
    weight_col: str = "rank_score"
    min_term_weight: float = 0.0

    expansion_weight: float = 0.25
    top_k_expansion_terms: int = 16

    top_k_parcels: int = 12
    top_k_per_direct_term: int = 2
    top_k_per_unlateralized_direct_term: int = 6
    top_k_per_expansion_term: int = 1

    prefer_bilateral_for_unlateralized_terms: bool = True
    max_parcels_per_hemisphere_per_term: int = 2

    min_term_parcel_score_frac: float = 0.10
    min_final_parcel_score_frac: float = 0.05
    max_spatial_jaccard: float = 0.65

    normalize_image: bool = True
    binarize_image: bool = False
    max_terms_to_match: int = 80
    n_jobs: int = 1


_HEMI_LEFT_PATTERNS = (
    r"\bleft\b",
    r"\bleft[-_\s]",
    r"\blh\b",
    r"\bl[-_\s]",
    r"\bhemisphere left\b",
)

_HEMI_RIGHT_PATTERNS = (
    r"\bright\b",
    r"\bright[-_\s]",
    r"\brh\b",
    r"\br[-_\s]",
    r"\bhemisphere right\b",
)


def term_has_explicit_hemisphere(term: str) -> bool:
    term_norm = normalize_term(term)
    return (
        any(re.search(p, term_norm) for p in _HEMI_LEFT_PATTERNS)
        or any(re.search(p, term_norm) for p in _HEMI_RIGHT_PATTERNS)
        or "bilateral" in term_norm
        or "bilaterally" in term_norm
    )


def infer_parcel_hemisphere(parcel_name: str) -> str:
    """
    Infer a coarse hemisphere label from a parcel name.

    Returns:
        "left", "right", or "midline_unknown"

    This is intentionally general and atlas-name based. It does not special-case
    any one region or network.
    """
    name = normalize_term(parcel_name)

    has_left = any(re.search(p, name) for p in _HEMI_LEFT_PATTERNS)
    has_right = any(re.search(p, name) for p in _HEMI_RIGHT_PATTERNS)

    # Common atlas shorthand after separators, e.g. "L G_precuneus", "R insula".
    if re.search(r"(?:^|\s|::)l\s+", name):
        has_left = True
    if re.search(r"(?:^|\s|::)r\s+", name):
        has_right = True

    if has_left and not has_right:
        return "left"
    if has_right and not has_left:
        return "right"

    return "midline_unknown"


def _binary_mask_jaccard(atlas_masks: np.ndarray, a: int, b: int) -> float:
    ma = atlas_masks[a] > 0
    mb = atlas_masks[b] > 0
    union = np.logical_or(ma, mb).sum()
    if union == 0:
        return 0.0
    inter = np.logical_and(ma, mb).sum()
    return float(inter / union)


def _term_column_scores(
    term: str,
    *,
    W: csr_matrix,
    term_to_col: dict[str, int],
) -> np.ndarray:
    col = term_to_col.get(normalize_term(term))
    if col is None:
        return np.zeros(W.shape[0], dtype=np.float32)

    scores = np.asarray(W[:, col].todense()).ravel().astype(np.float32)

    if scores.max() > 0:
        scores = scores / scores.max()

    return scores


def _candidate_rows_for_term(
    term: str,
    *,
    W: csr_matrix,
    term_to_col: dict[str, int],
    parcel_names: Sequence[str],
    term_weight: float,
    source: str,
    top_k: int,
    min_score_frac: float,
    prefer_bilateral_if_unlateralized: bool,
    max_per_hemi: int,
) -> list[dict[str, object]]:
    term_norm = normalize_term(term)
    scores = _term_column_scores(term_norm, W=W, term_to_col=term_to_col)

    if scores.max() <= 0:
        return []

    threshold = float(min_score_frac) * float(scores.max())
    keep = np.flatnonzero(scores >= threshold)

    if keep.size == 0:
        return []

    explicit_hemi = term_has_explicit_hemisphere(term_norm)
    prefer_hemi_balance = prefer_bilateral_if_unlateralized and not explicit_hemi

    if not prefer_hemi_balance:
        order = keep[np.argsort(scores[keep])[::-1]]
        order = order[:top_k]

        return [
            {
                "parcel_idx": int(parcel_idx),
                "term": term_norm,
                "source": source,
                "hemisphere": infer_parcel_hemisphere(str(parcel_names[parcel_idx])),
                "term_score": float(scores[parcel_idx]),
                "weighted_score": float(term_weight * scores[parcel_idx]),
            }
            for parcel_idx in order
        ]

    # Hemisphere-balanced candidate selection for unlateralized terms.
    hemi_to_indices: dict[str, list[int]] = {
        "left": [],
        "right": [],
        "midline_unknown": [],
    }

    for parcel_idx in keep:
        hemi = infer_parcel_hemisphere(str(parcel_names[int(parcel_idx)]))
        hemi_to_indices.setdefault(hemi, []).append(int(parcel_idx))

    selected: list[int] = []

    # Prefer one or more from each side if present.
    for hemi in ("left", "right", "midline_unknown"):
        idxs = hemi_to_indices.get(hemi, [])
        if not idxs:
            continue

        idxs_sorted = sorted(idxs, key=lambda i: float(scores[i]), reverse=True)
        selected.extend(idxs_sorted[:max_per_hemi])

    # Fill remaining slots by score.
    if len(selected) < top_k:
        all_sorted = keep[np.argsort(scores[keep])[::-1]].tolist()
        for parcel_idx in all_sorted:
            parcel_idx = int(parcel_idx)
            if parcel_idx not in selected:
                selected.append(parcel_idx)
            if len(selected) >= top_k:
                break

    selected = selected[:top_k]

    return [
        {
            "parcel_idx": int(parcel_idx),
            "term": term_norm,
            "source": source,
            "hemisphere": infer_parcel_hemisphere(str(parcel_names[parcel_idx])),
            "term_score": float(scores[parcel_idx]),
            "weighted_score": float(term_weight * scores[parcel_idx]),
        }
        for parcel_idx in selected
    ]


def select_term_balanced_parcels(
    *,
    direct_terms: Sequence[str],
    expansion_terms: Sequence[str],
    W: csr_matrix,
    term_to_col: dict[str, int],
    parcel_names: Sequence[str],
    atlas_masks,
    config: SyntheticMapConfig,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    """
    Select parcels in a term-balanced and hemisphere-aware way.

    Main behavior:
        - Each direct term gets its own parcel candidates.
        - Unlateralized terms prefer bilateral/hemisphere-balanced candidates.
        - Explicitly lateralized terms respect their learned parcel ranking.
        - Spatially redundant duplicate parcels are suppressed.
    """
    atlas_masks_np = np.asarray(atlas_masks, dtype=np.float32)
    n_parcels = W.shape[0]

    direct_terms = [normalize_term(t) for t in direct_terms if normalize_term(t)]
    expansion_terms = [normalize_term(t) for t in expansion_terms if normalize_term(t)]

    candidate_rows: list[dict[str, object]] = []

    for term in direct_terms:
        explicit_hemi = term_has_explicit_hemisphere(term)
        top_k = (
            config.top_k_per_direct_term
            if explicit_hemi
            else config.top_k_per_unlateralized_direct_term
        )

        candidate_rows.extend(
            _candidate_rows_for_term(
                term,
                W=W,
                term_to_col=term_to_col,
                parcel_names=parcel_names,
                term_weight=1.0,
                source="direct",
                top_k=top_k,
                min_score_frac=config.min_term_parcel_score_frac,
                prefer_bilateral_if_unlateralized=config.prefer_bilateral_for_unlateralized_terms,
                max_per_hemi=config.max_parcels_per_hemisphere_per_term,
            )
        )

    for term in expansion_terms:
        candidate_rows.extend(
            _candidate_rows_for_term(
                term,
                W=W,
                term_to_col=term_to_col,
                parcel_names=parcel_names,
                term_weight=config.expansion_weight,
                source="expansion",
                top_k=config.top_k_per_expansion_term,
                min_score_frac=config.min_term_parcel_score_frac,
                prefer_bilateral_if_unlateralized=False,
                max_per_hemi=1,
            )
        )

    if not candidate_rows:
        return (
            np.empty(0, dtype=np.int64),
            np.zeros(n_parcels, dtype=np.float32),
            [],
        )

    source_priority = {"direct": 1, "expansion": 0}
    candidate_rows = sorted(
        candidate_rows,
        key=lambda r: (
            source_priority.get(str(r["source"]), 0),
            float(r["weighted_score"]),
            float(r["term_score"]),
        ),
        reverse=True,
    )

    selected_rows: list[dict[str, object]] = []
    selected_indices: list[int] = []

    for row in candidate_rows:
        parcel_idx = int(row["parcel_idx"])

        if parcel_idx in selected_indices:
            continue

        too_redundant = False
        for old_idx in selected_indices:
            jacc = _binary_mask_jaccard(atlas_masks_np, parcel_idx, old_idx)
            if jacc >= config.max_spatial_jaccard:
                too_redundant = True
                break

        if too_redundant:
            continue

        selected_indices.append(parcel_idx)
        selected_rows.append(row)

        if len(selected_indices) >= config.top_k_parcels:
            break

    if len(selected_indices) < config.top_k_parcels:
        for row in candidate_rows:
            parcel_idx = int(row["parcel_idx"])

            if parcel_idx in selected_indices:
                continue

            selected_indices.append(parcel_idx)
            selected_rows.append(row)

            if len(selected_indices) >= config.top_k_parcels:
                break

    parcel_scores = np.zeros(n_parcels, dtype=np.float32)

    for row in selected_rows:
        parcel_idx = int(row["parcel_idx"])
        parcel_scores[parcel_idx] = max(
            parcel_scores[parcel_idx],
            float(row["weighted_score"]),
        )

    if parcel_scores.max() > 0:
        parcel_scores = parcel_scores / parcel_scores.max()

    keep_set = {
        int(i)
        for i in selected_indices
        if parcel_scores[int(i)] >= config.min_final_parcel_score_frac
    }

    selected_rows = [
        row for row in selected_rows
        if int(row["parcel_idx"]) in keep_set
    ]

    selected_indices_final = np.asarray(
        [int(row["parcel_idx"]) for row in selected_rows],
        dtype=np.int64,
    )

    return selected_indices_final, parcel_scores, selected_rows



# %%

from dataclasses import dataclass
from typing import Sequence
import re

import numpy as np
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class SyntheticMapConfig:
    weight_col: str = "rank_score"
    min_term_weight: float = 0.0

    expansion_weight: float = 0.25
    top_k_expansion_terms: int = 16

    top_k_parcels: int = 12
    top_k_per_direct_term: int = 2
    top_k_per_unlateralized_direct_term: int = 6
    top_k_per_expansion_term: int = 1

    prefer_bilateral_for_unlateralized_terms: bool = True
    max_parcels_per_hemisphere_per_term: int = 2

    min_term_parcel_score_frac: float = 0.10
    min_final_parcel_score_frac: float = 0.05
    max_spatial_jaccard: float = 0.65

    normalize_image: bool = True
    binarize_image: bool = False
    max_terms_to_match: int = 80
    n_jobs: int = 1


_HEMI_LEFT_PATTERNS = (
    r"\bleft\b",
    r"\bleft[-_\s]",
    r"\blh\b",
    r"\bl[-_\s]",
    r"\bhemisphere left\b",
)

_HEMI_RIGHT_PATTERNS = (
    r"\bright\b",
    r"\bright[-_\s]",
    r"\brh\b",
    r"\br[-_\s]",
    r"\bhemisphere right\b",
)


def term_has_explicit_hemisphere(term: str) -> bool:
    term_norm = normalize_term(term)
    return (
        any(re.search(p, term_norm) for p in _HEMI_LEFT_PATTERNS)
        or any(re.search(p, term_norm) for p in _HEMI_RIGHT_PATTERNS)
        or "bilateral" in term_norm
        or "bilaterally" in term_norm
    )


def infer_parcel_hemisphere(parcel_name: str) -> str:
    """
    Infer a coarse hemisphere label from a parcel name.

    Returns:
        "left", "right", or "midline_unknown"

    This is intentionally general and atlas-name based. It does not special-case
    any one region or network.
    """
    name = normalize_term(parcel_name)

    has_left = any(re.search(p, name) for p in _HEMI_LEFT_PATTERNS)
    has_right = any(re.search(p, name) for p in _HEMI_RIGHT_PATTERNS)

    # Common atlas shorthand after separators, e.g. "L G_precuneus", "R insula".
    if re.search(r"(?:^|\s|::)l\s+", name):
        has_left = True
    if re.search(r"(?:^|\s|::)r\s+", name):
        has_right = True

    if has_left and not has_right:
        return "left"
    if has_right and not has_left:
        return "right"

    return "midline_unknown"


def _binary_mask_jaccard(atlas_masks: np.ndarray, a: int, b: int) -> float:
    ma = atlas_masks[a] > 0
    mb = atlas_masks[b] > 0
    union = np.logical_or(ma, mb).sum()
    if union == 0:
        return 0.0
    inter = np.logical_and(ma, mb).sum()
    return float(inter / union)


def _term_column_scores(
    term: str,
    *,
    W: csr_matrix,
    term_to_col: dict[str, int],
) -> np.ndarray:
    col = term_to_col.get(normalize_term(term))
    if col is None:
        return np.zeros(W.shape[0], dtype=np.float32)

    scores = np.asarray(W[:, col].todense()).ravel().astype(np.float32)

    if scores.max() > 0:
        scores = scores / scores.max()

    return scores


def _candidate_rows_for_term(
    term: str,
    *,
    W: csr_matrix,
    term_to_col: dict[str, int],
    parcel_names: Sequence[str],
    term_weight: float,
    source: str,
    top_k: int,
    min_score_frac: float,
    prefer_bilateral_if_unlateralized: bool,
    max_per_hemi: int,
) -> list[dict[str, object]]:
    term_norm = normalize_term(term)
    scores = _term_column_scores(term_norm, W=W, term_to_col=term_to_col)

    if scores.max() <= 0:
        return []

    threshold = float(min_score_frac) * float(scores.max())
    keep = np.flatnonzero(scores >= threshold)

    if keep.size == 0:
        return []

    explicit_hemi = term_has_explicit_hemisphere(term_norm)
    prefer_hemi_balance = prefer_bilateral_if_unlateralized and not explicit_hemi

    if not prefer_hemi_balance:
        order = keep[np.argsort(scores[keep])[::-1]]
        order = order[:top_k]

        return [
            {
                "parcel_idx": int(parcel_idx),
                "term": term_norm,
                "source": source,
                "hemisphere": infer_parcel_hemisphere(str(parcel_names[parcel_idx])),
                "term_score": float(scores[parcel_idx]),
                "weighted_score": float(term_weight * scores[parcel_idx]),
            }
            for parcel_idx in order
        ]

    # Hemisphere-balanced candidate selection for unlateralized terms.
    hemi_to_indices: dict[str, list[int]] = {
        "left": [],
        "right": [],
        "midline_unknown": [],
    }

    for parcel_idx in keep:
        hemi = infer_parcel_hemisphere(str(parcel_names[int(parcel_idx)]))
        hemi_to_indices.setdefault(hemi, []).append(int(parcel_idx))

    selected: list[int] = []

    # Prefer one or more from each side if present.
    for hemi in ("left", "right", "midline_unknown"):
        idxs = hemi_to_indices.get(hemi, [])
        if not idxs:
            continue

        idxs_sorted = sorted(idxs, key=lambda i: float(scores[i]), reverse=True)
        selected.extend(idxs_sorted[:max_per_hemi])

    # Fill remaining slots by score.
    if len(selected) < top_k:
        all_sorted = keep[np.argsort(scores[keep])[::-1]].tolist()
        for parcel_idx in all_sorted:
            parcel_idx = int(parcel_idx)
            if parcel_idx not in selected:
                selected.append(parcel_idx)
            if len(selected) >= top_k:
                break

    selected = selected[:top_k]

    return [
        {
            "parcel_idx": int(parcel_idx),
            "term": term_norm,
            "source": source,
            "hemisphere": infer_parcel_hemisphere(str(parcel_names[parcel_idx])),
            "term_score": float(scores[parcel_idx]),
            "weighted_score": float(term_weight * scores[parcel_idx]),
        }
        for parcel_idx in selected
    ]


def select_term_balanced_parcels(
    *,
    direct_terms: Sequence[str],
    expansion_terms: Sequence[str],
    W: csr_matrix,
    term_to_col: dict[str, int],
    parcel_names: Sequence[str],
    atlas_masks,
    config: SyntheticMapConfig,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    """
    Select parcels in a term-balanced and hemisphere-aware way.

    Main behavior:
        - Each direct term gets its own parcel candidates.
        - Unlateralized terms prefer bilateral/hemisphere-balanced candidates.
        - Explicitly lateralized terms respect their learned parcel ranking.
        - Spatially redundant duplicate parcels are suppressed.
    """
    atlas_masks_np = np.asarray(atlas_masks, dtype=np.float32)
    n_parcels = W.shape[0]

    direct_terms = [normalize_term(t) for t in direct_terms if normalize_term(t)]
    expansion_terms = [normalize_term(t) for t in expansion_terms if normalize_term(t)]

    candidate_rows: list[dict[str, object]] = []

    for term in direct_terms:
        explicit_hemi = term_has_explicit_hemisphere(term)
        top_k = (
            config.top_k_per_direct_term
            if explicit_hemi
            else config.top_k_per_unlateralized_direct_term
        )

        candidate_rows.extend(
            _candidate_rows_for_term(
                term,
                W=W,
                term_to_col=term_to_col,
                parcel_names=parcel_names,
                term_weight=1.0,
                source="direct",
                top_k=top_k,
                min_score_frac=config.min_term_parcel_score_frac,
                prefer_bilateral_if_unlateralized=config.prefer_bilateral_for_unlateralized_terms,
                max_per_hemi=config.max_parcels_per_hemisphere_per_term,
            )
        )

    for term in expansion_terms:
        candidate_rows.extend(
            _candidate_rows_for_term(
                term,
                W=W,
                term_to_col=term_to_col,
                parcel_names=parcel_names,
                term_weight=config.expansion_weight,
                source="expansion",
                top_k=config.top_k_per_expansion_term,
                min_score_frac=config.min_term_parcel_score_frac,
                prefer_bilateral_if_unlateralized=False,
                max_per_hemi=1,
            )
        )

    if not candidate_rows:
        return (
            np.empty(0, dtype=np.int64),
            np.zeros(n_parcels, dtype=np.float32),
            [],
        )

    source_priority = {"direct": 1, "expansion": 0}
    candidate_rows = sorted(
        candidate_rows,
        key=lambda r: (
            source_priority.get(str(r["source"]), 0),
            float(r["weighted_score"]),
            float(r["term_score"]),
        ),
        reverse=True,
    )

    selected_rows: list[dict[str, object]] = []
    selected_indices: list[int] = []

    for row in candidate_rows:
        parcel_idx = int(row["parcel_idx"])

        if parcel_idx in selected_indices:
            continue

        too_redundant = False
        for old_idx in selected_indices:
            jacc = _binary_mask_jaccard(atlas_masks_np, parcel_idx, old_idx)
            if jacc >= config.max_spatial_jaccard:
                too_redundant = True
                break

        if too_redundant:
            continue

        selected_indices.append(parcel_idx)
        selected_rows.append(row)

        if len(selected_indices) >= config.top_k_parcels:
            break

    if len(selected_indices) < config.top_k_parcels:
        for row in candidate_rows:
            parcel_idx = int(row["parcel_idx"])

            if parcel_idx in selected_indices:
                continue

            selected_indices.append(parcel_idx)
            selected_rows.append(row)

            if len(selected_indices) >= config.top_k_parcels:
                break

    parcel_scores = np.zeros(n_parcels, dtype=np.float32)

    for row in selected_rows:
        parcel_idx = int(row["parcel_idx"])
        parcel_scores[parcel_idx] = max(
            parcel_scores[parcel_idx],
            float(row["weighted_score"]),
        )

    if parcel_scores.max() > 0:
        parcel_scores = parcel_scores / parcel_scores.max()

    keep_set = {
        int(i)
        for i in selected_indices
        if parcel_scores[int(i)] >= config.min_final_parcel_score_frac
    }

    selected_rows = [
        row for row in selected_rows
        if int(row["parcel_idx"]) in keep_set
    ]

    selected_indices_final = np.asarray(
        [int(row["parcel_idx"]) for row in selected_rows],
        dtype=np.int64,
    )

    return selected_indices_final, parcel_scores, selected_rows



# %%

from dataclasses import dataclass
from typing import Literal, Sequence
import re

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, csc_matrix


OverlapMode = Literal["mass_fraction", "mean_in_parcel", "cosine"]


@dataclass(frozen=True)
class ParcelNgramConfig:
    overlap_mode: OverlapMode = "mass_fraction"
    min_overlap: float = 0.02
    top_k_parcels_per_paper: int | None = None
    top_k_ngrams_per_parcel: int = 50
    max_pmids_per_parcel: int | None = None
    positive_only: bool = True
    rank_by: Literal["count", "weighted_count", "lift", "count_lift"] = "count_lift"
    min_parcel_docs: int = 5
    min_ngram_count_in_parcel: int = 2
    lift_alpha: float = 1.0


@dataclass(frozen=True)
class SyntheticMapConfig:
    weight_col: str = "rank_score"
    min_term_weight: float = 0.0

    expansion_weight: float = 0.25
    top_k_expansion_terms: int = 16

    top_k_parcels: int = 12
    top_k_per_direct_term: int = 2
    top_k_per_unlateralized_direct_term: int = 6
    top_k_per_expansion_term: int = 1

    prefer_bilateral_for_unlateralized_terms: bool = True
    max_parcels_per_hemisphere_per_term: int = 2

    min_term_parcel_score_frac: float = 0.10
    min_final_parcel_score_frac: float = 0.05
    max_spatial_jaccard: float = 0.65

    normalize_image: bool = True
    binarize_image: bool = False
    max_terms_to_match: int = 80
    n_jobs: int = 1


_HEMI_LEFT_PATTERNS = (
    r"\bleft\b",
    r"\bleft[-_\s]",
    r"\blh\b",
    r"\bl[-_\s]",
    r"\bhemisphere left\b",
)

_HEMI_RIGHT_PATTERNS = (
    r"\bright\b",
    r"\bright[-_\s]",
    r"\brh\b",
    r"\br[-_\s]",
    r"\bhemisphere right\b",
)


def normalize_term(term: str) -> str:
    term = str(term).strip().casefold()
    term = re.sub(r"[^a-z0-9]+", " ", term)
    term = re.sub(r"\s+", " ", term).strip()
    return term


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def _as_float32_array(x, *, name: str) -> np.ndarray:
    arr = _to_numpy(x)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}.")
    return arr.astype(np.float32, copy=False)


def _as_name_list(names: Sequence) -> list[str]:
    if hasattr(names, "tolist"):
        names = names.tolist()
    return [str(x) for x in names]


def _prepare_maps(brain_maps, *, positive_only: bool) -> np.ndarray:
    maps = _as_float32_array(brain_maps, name="brain_maps")
    if positive_only:
        maps = np.maximum(maps, 0.0)
    return maps


def _prepare_masks(atlas_masks) -> np.ndarray:
    masks = _as_float32_array(atlas_masks, name="atlas_masks")
    return (masks > 0).astype(np.float32, copy=False)


def build_paper_parcel_overlap_matrix(
    *,
    brain_maps,
    atlas_masks,
    config: ParcelNgramConfig = ParcelNgramConfig(),
    chunk_docs: int = 2048,
    eps: float = 1e-8,
) -> csr_matrix:
    maps = _prepare_maps(brain_maps, positive_only=config.positive_only)
    masks = _prepare_masks(atlas_masks)

    n_docs, n_voxels = maps.shape
    n_parcels, n_mask_voxels = masks.shape

    if n_voxels != n_mask_voxels:
        raise ValueError(
            f"brain_maps.shape[1]={n_voxels} does not match "
            f"atlas_masks.shape[1]={n_mask_voxels}."
        )

    mask_sizes = masks.sum(axis=1).astype(np.float32)

    if config.overlap_mode == "cosine":
        mask_norms = np.sqrt(mask_sizes).clip(min=eps)
        masks_for_dot = masks / mask_norms[:, None]
    else:
        masks_for_dot = masks

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []

    masks_t = np.ascontiguousarray(masks_for_dot.T)

    for start in range(0, n_docs, chunk_docs):
        end = min(start + chunk_docs, n_docs)
        block = maps[start:end]

        raw_overlap = block @ masks_t

        if config.overlap_mode == "mass_fraction":
            denom = block.sum(axis=1, keepdims=True).clip(min=eps)
            scores = raw_overlap / denom

        elif config.overlap_mode == "mean_in_parcel":
            scores = raw_overlap / mask_sizes[None, :].clip(min=eps)

        elif config.overlap_mode == "cosine":
            map_norms = np.linalg.norm(block, axis=1, keepdims=True).clip(min=eps)
            scores = raw_overlap / map_norms

        else:
            raise ValueError(f"Unknown overlap_mode: {config.overlap_mode}")

        scores = np.asarray(scores, dtype=np.float32)

        if config.top_k_parcels_per_paper is not None:
            k = min(config.top_k_parcels_per_paper, scores.shape[1])
            if k > 0:
                keep_idx = np.argpartition(scores, kth=scores.shape[1] - k, axis=1)[:, -k:]
                keep_mask = np.zeros(scores.shape, dtype=bool)
                local_rows = np.arange(scores.shape[0])[:, None]
                keep_mask[local_rows, keep_idx] = True
                scores = np.where(keep_mask, scores, 0.0)

        hit_r, hit_c = np.nonzero(scores >= config.min_overlap)

        if hit_r.size == 0:
            continue

        rows.append(hit_r.astype(np.int64) + start)
        cols.append(hit_c.astype(np.int64))
        vals.append(scores[hit_r, hit_c].astype(np.float32))

    if not rows:
        return csr_matrix((n_docs, n_parcels), dtype=np.float32)

    out = csr_matrix(
        (
            np.concatenate(vals),
            (np.concatenate(rows), np.concatenate(cols)),
        ),
        shape=(n_docs, n_parcels),
        dtype=np.float32,
    )
    out.sum_duplicates()
    return out


def _make_bool_copy(x: csr_matrix) -> csr_matrix:
    y = x.copy().astype(bool)
    y.eliminate_zeros()
    return y


def _top_pmids_for_parcel(
    paper_parcel_scores_csc: csc_matrix,
    parcel_col: int,
    pmids: Sequence[str | int],
    *,
    max_pmids: int | None,
) -> list[str | int]:
    col = paper_parcel_scores_csc.getcol(parcel_col)

    if col.nnz == 0:
        return []

    order = np.argsort(col.data)[::-1]
    if max_pmids is not None:
        order = order[:max_pmids]

    return [pmids[int(col.indices[i])] for i in order]


def _top_ngrams_for_parcel(
    *,
    parcel_idx: int,
    count_matrix: csr_matrix,
    weighted_matrix: csr_matrix,
    labels: Sequence[str],
    global_ngram_df: np.ndarray,
    n_docs_total: int,
    n_docs_parcel: int,
    config: ParcelNgramConfig,
) -> tuple[list[str], list[dict[str, object]]]:
    row_counts = count_matrix.getrow(parcel_idx)

    if row_counts.nnz == 0:
        return [], []

    term_cols = row_counts.indices
    counts = row_counts.data.astype(np.float32)

    keep = counts >= config.min_ngram_count_in_parcel
    term_cols = term_cols[keep]
    counts = counts[keep]

    if term_cols.size == 0:
        return [], []

    weighted_row = weighted_matrix.getrow(parcel_idx)
    weighted_lookup = dict(zip(weighted_row.indices.tolist(), weighted_row.data.tolist()))

    weighted_counts = np.array(
        [weighted_lookup.get(int(c), 0.0) for c in term_cols],
        dtype=np.float32,
    )

    alpha = float(config.lift_alpha)
    p_region = (counts + alpha) / (float(n_docs_parcel) + 2.0 * alpha)
    p_global = (global_ngram_df[term_cols].astype(np.float32) + alpha) / (
        float(n_docs_total) + 2.0 * alpha
    )

    lift = p_region / np.maximum(p_global, 1e-12)
    log_lift = np.log(lift)

    if config.rank_by == "count":
        rank_score = counts
    elif config.rank_by == "weighted_count":
        rank_score = weighted_counts
    elif config.rank_by == "lift":
        rank_score = log_lift
    elif config.rank_by == "count_lift":
        rank_score = counts * np.maximum(log_lift, 0.0)
    else:
        raise ValueError(f"Unknown rank_by: {config.rank_by}")

    order = np.argsort(rank_score)[::-1]
    order = order[: config.top_k_ngrams_per_parcel]

    ngrams: list[str] = []
    rows: list[dict[str, object]] = []

    for idx in order:
        col = int(term_cols[idx])
        label = str(labels[col])
        ngrams.append(label)
        rows.append({
            "ngram": label,
            "ngram_col": col,
            "count": int(counts[idx]),
            "weighted_count": float(weighted_counts[idx]),
            "global_df": int(global_ngram_df[col]),
            "lift": float(lift[idx]),
            "log_lift": float(log_lift[idx]),
            "rank_score": float(rank_score[idx]),
        })

    return ngrams, rows


def make_parcel_ngram_dataframe(
    *,
    brain_maps,
    atlas_masks,
    atlas_parcel_names: Sequence[str],
    labels: Sequence[str],
    X_bool: csr_matrix,
    pmids: Sequence[str | int],
    config: ParcelNgramConfig = ParcelNgramConfig(),
    chunk_docs: int = 2048,
) -> pd.DataFrame:
    """
    Construct parcel-level n-gram summaries.

    This version is intentionally parcel-specific:
        - list_of_pmids comes from the current atlas parcel only.
        - list_of_ngrams comes from the current atlas parcel only.
        - no same-name parcel pooling is performed.

    Laterality for synthetic maps should be handled downstream by
    select_term_balanced_parcels, not by changing this dataframe construction.
    """
    if not isinstance(X_bool, csr_matrix):
        X_bool = X_bool.tocsr()

    X_bool = X_bool.astype(bool)
    X_bool.eliminate_zeros()

    maps = _as_float32_array(brain_maps, name="brain_maps")
    masks = _prepare_masks(atlas_masks)
    atlas_parcel_names = _as_name_list(atlas_parcel_names)
    labels = _as_name_list(labels)

    if maps.shape[0] != X_bool.shape[0]:
        raise ValueError(
            f"brain_maps rows ({maps.shape[0]}) must match X_bool rows ({X_bool.shape[0]})."
        )

    if len(pmids) != maps.shape[0]:
        raise ValueError(
            f"len(pmids) ({len(pmids)}) must match brain_maps.shape[0] ({maps.shape[0]})."
        )

    if len(labels) != X_bool.shape[1]:
        raise ValueError(
            f"len(labels) ({len(labels)}) must match X_bool.shape[1] ({X_bool.shape[1]})."
        )

    if len(atlas_parcel_names) != masks.shape[0]:
        raise ValueError(
            f"len(atlas_parcel_names) ({len(atlas_parcel_names)}) must match "
            f"atlas_masks.shape[0] ({masks.shape[0]})."
        )

    paper_parcel_scores = build_paper_parcel_overlap_matrix(
        brain_maps=maps,
        atlas_masks=masks,
        config=config,
        chunk_docs=chunk_docs,
    )

    if paper_parcel_scores.shape[1] != len(atlas_parcel_names):
        raise RuntimeError(
            "Internal alignment error: paper_parcel_scores columns do not match atlas_parcel_names."
        )

    paper_parcel_bool = _make_bool_copy(paper_parcel_scores)

    # Critical: numeric sparse multiplication, not bool @ bool.
    paper_parcel_count = paper_parcel_bool.astype(np.float32)
    X_count = X_bool.astype(np.float32)

    count_matrix = (paper_parcel_count.T @ X_count).tocsr()
    count_matrix.sum_duplicates()

    weighted_matrix = (paper_parcel_scores.T @ X_count).tocsr()
    weighted_matrix.sum_duplicates()

    parcel_doc_counts = np.asarray(paper_parcel_bool.sum(axis=0)).ravel().astype(np.int64)
    global_ngram_df = np.asarray(X_bool.sum(axis=0)).ravel().astype(np.int64)
    mask_sizes = masks.sum(axis=1).astype(np.int64)

    paper_parcel_scores_csc = paper_parcel_scores.tocsc()

    rows: list[dict[str, object]] = []

    for parcel_idx, parcel_name in enumerate(atlas_parcel_names):
        n_parcel_docs = int(parcel_doc_counts[parcel_idx])
        empty_mask = bool(mask_sizes[parcel_idx] == 0)

        row_counts = count_matrix.getrow(parcel_idx)
        n_candidate_ngram_hits = int(row_counts.nnz)
        max_ngram_count = float(row_counts.data.max()) if row_counts.nnz else 0.0

        if n_parcel_docs < config.min_parcel_docs:
            rows.append({
                "atlas_parcel_names": parcel_name,
                "list_of_ngrams": [],
                "list_of_pmids": [],
                "n_parcel_pmids": n_parcel_docs,
                "n_ngrams": 0,
                "n_candidate_ngram_hits": n_candidate_ngram_hits,
                "max_ngram_count": max_ngram_count,
                "ngram_stats": [],
                "empty_mask": empty_mask,
            })
            continue

        list_of_ngrams, ngram_stats = _top_ngrams_for_parcel(
            parcel_idx=parcel_idx,
            count_matrix=count_matrix,
            weighted_matrix=weighted_matrix,
            labels=labels,
            global_ngram_df=global_ngram_df,
            n_docs_total=maps.shape[0],
            n_docs_parcel=n_parcel_docs,
            config=config,
        )

        list_of_pmids = _top_pmids_for_parcel(
            paper_parcel_scores_csc,
            parcel_idx,
            pmids,
            max_pmids=config.max_pmids_per_parcel,
        )

        rows.append({
            "atlas_parcel_names": parcel_name,
            "list_of_ngrams": list_of_ngrams,
            "list_of_pmids": list_of_pmids,
            "n_parcel_pmids": n_parcel_docs,
            "n_ngrams": len(list_of_ngrams),
            "n_candidate_ngram_hits": n_candidate_ngram_hits,
            "max_ngram_count": max_ngram_count,
            "ngram_stats": ngram_stats,
            "empty_mask": empty_mask,
        })

    return pd.DataFrame(rows)


def concatenate_atlases(
    atlas_dict: dict[str, tuple[object, Sequence[str]]],
    *,
    name_sep: str = "::",
) -> tuple[np.ndarray, list[str]]:
    masks_out: list[np.ndarray] = []
    names_out: list[str] = []
    n_voxels_expected: int | None = None

    for atlas_name, (masks, names) in atlas_dict.items():
        masks_arr = _prepare_masks(masks)
        names_list = _as_name_list(names)

        if len(names_list) != masks_arr.shape[0]:
            raise ValueError(
                f"{atlas_name}: len(names)={len(names_list)} must match "
                f"masks.shape[0]={masks_arr.shape[0]}."
            )

        if n_voxels_expected is None:
            n_voxels_expected = masks_arr.shape[1]
        elif masks_arr.shape[1] != n_voxels_expected:
            raise ValueError(
                f"{atlas_name}: voxel dimension {masks_arr.shape[1]} does not "
                f"match expected {n_voxels_expected}."
            )

        masks_out.append(masks_arr)
        names_out.extend([f"{atlas_name}{name_sep}{name}" for name in names_list])

    if not masks_out:
        raise ValueError("atlas_dict is empty.")

    all_masks = np.vstack(masks_out).astype(np.float32, copy=False)

    if all_masks.shape[0] != len(names_out):
        raise RuntimeError("Internal alignment error after atlas concatenation.")

    return all_masks, names_out


def term_has_explicit_hemisphere(term: str) -> bool:
    term_norm = normalize_term(term)
    return (
        any(re.search(p, term_norm) for p in _HEMI_LEFT_PATTERNS)
        or any(re.search(p, term_norm) for p in _HEMI_RIGHT_PATTERNS)
        or "bilateral" in term_norm
        or "bilaterally" in term_norm
    )


def infer_parcel_hemisphere(parcel_name: str) -> str:
    name = normalize_term(parcel_name)

    has_left = any(re.search(p, name) for p in _HEMI_LEFT_PATTERNS)
    has_right = any(re.search(p, name) for p in _HEMI_RIGHT_PATTERNS)

    if re.search(r"(?:^|\s|::)l\s+", name):
        has_left = True
    if re.search(r"(?:^|\s|::)r\s+", name):
        has_right = True

    if has_left and not has_right:
        return "left"
    if has_right and not has_left:
        return "right"

    return "midline_unknown"


def _binary_mask_jaccard(atlas_masks: np.ndarray, a: int, b: int) -> float:
    ma = atlas_masks[a] > 0
    mb = atlas_masks[b] > 0
    union = np.logical_or(ma, mb).sum()

    if union == 0:
        return 0.0

    inter = np.logical_and(ma, mb).sum()
    return float(inter / union)


def _term_column_scores(
    term: str,
    *,
    W: csr_matrix,
    term_to_col: dict[str, int],
) -> np.ndarray:
    col = term_to_col.get(normalize_term(term))

    if col is None:
        return np.zeros(W.shape[0], dtype=np.float32)

    scores = np.asarray(W[:, col].todense()).ravel().astype(np.float32)

    if scores.max() > 0:
        scores = scores / scores.max()

    return scores


def _candidate_rows_for_term(
    term: str,
    *,
    W: csr_matrix,
    term_to_col: dict[str, int],
    parcel_names: Sequence[str],
    term_weight: float,
    source: str,
    top_k: int,
    min_score_frac: float,
    prefer_bilateral_if_unlateralized: bool,
    max_per_hemi: int,
) -> list[dict[str, object]]:
    term_norm = normalize_term(term)
    scores = _term_column_scores(term_norm, W=W, term_to_col=term_to_col)

    if scores.max() <= 0:
        return []

    threshold = float(min_score_frac) * float(scores.max())
    keep = np.flatnonzero(scores >= threshold)

    if keep.size == 0:
        return []

    explicit_hemi = term_has_explicit_hemisphere(term_norm)
    prefer_hemi_balance = prefer_bilateral_if_unlateralized and not explicit_hemi

    if not prefer_hemi_balance:
        order = keep[np.argsort(scores[keep])[::-1]]
        order = order[:top_k]

        return [
            {
                "parcel_idx": int(parcel_idx),
                "term": term_norm,
                "source": source,
                "hemisphere": infer_parcel_hemisphere(str(parcel_names[parcel_idx])),
                "term_score": float(scores[parcel_idx]),
                "weighted_score": float(term_weight * scores[parcel_idx]),
            }
            for parcel_idx in order
        ]

    hemi_to_indices: dict[str, list[int]] = {
        "left": [],
        "right": [],
        "midline_unknown": [],
    }

    for parcel_idx in keep:
        hemi = infer_parcel_hemisphere(str(parcel_names[int(parcel_idx)]))
        hemi_to_indices.setdefault(hemi, []).append(int(parcel_idx))

    selected: list[int] = []

    for hemi in ("left", "right", "midline_unknown"):
        idxs = hemi_to_indices.get(hemi, [])
        if not idxs:
            continue

        idxs_sorted = sorted(idxs, key=lambda i: float(scores[i]), reverse=True)
        selected.extend(idxs_sorted[:max_per_hemi])

    if len(selected) < top_k:
        all_sorted = keep[np.argsort(scores[keep])[::-1]].tolist()
        for parcel_idx in all_sorted:
            parcel_idx = int(parcel_idx)
            if parcel_idx not in selected:
                selected.append(parcel_idx)
            if len(selected) >= top_k:
                break

    selected = selected[:top_k]

    return [
        {
            "parcel_idx": int(parcel_idx),
            "term": term_norm,
            "source": source,
            "hemisphere": infer_parcel_hemisphere(str(parcel_names[parcel_idx])),
            "term_score": float(scores[parcel_idx]),
            "weighted_score": float(term_weight * scores[parcel_idx]),
        }
        for parcel_idx in selected
    ]


def select_term_balanced_parcels(
    *,
    direct_terms: Sequence[str],
    expansion_terms: Sequence[str],
    W: csr_matrix,
    term_to_col: dict[str, int],
    parcel_names: Sequence[str],
    atlas_masks,
    config: SyntheticMapConfig,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    atlas_masks_np = np.asarray(atlas_masks, dtype=np.float32)
    n_parcels = W.shape[0]

    direct_terms = [normalize_term(t) for t in direct_terms if normalize_term(t)]
    expansion_terms = [normalize_term(t) for t in expansion_terms if normalize_term(t)]

    candidate_rows: list[dict[str, object]] = []

    for term in direct_terms:
        explicit_hemi = term_has_explicit_hemisphere(term)
        top_k = (
            config.top_k_per_direct_term
            if explicit_hemi
            else config.top_k_per_unlateralized_direct_term
        )

        candidate_rows.extend(
            _candidate_rows_for_term(
                term,
                W=W,
                term_to_col=term_to_col,
                parcel_names=parcel_names,
                term_weight=1.0,
                source="direct",
                top_k=top_k,
                min_score_frac=config.min_term_parcel_score_frac,
                prefer_bilateral_if_unlateralized=config.prefer_bilateral_for_unlateralized_terms,
                max_per_hemi=config.max_parcels_per_hemisphere_per_term,
            )
        )

    for term in expansion_terms:
        candidate_rows.extend(
            _candidate_rows_for_term(
                term,
                W=W,
                term_to_col=term_to_col,
                parcel_names=parcel_names,
                term_weight=config.expansion_weight,
                source="expansion",
                top_k=config.top_k_per_expansion_term,
                min_score_frac=config.min_term_parcel_score_frac,
                prefer_bilateral_if_unlateralized=False,
                max_per_hemi=1,
            )
        )

    if not candidate_rows:
        return (
            np.empty(0, dtype=np.int64),
            np.zeros(n_parcels, dtype=np.float32),
            [],
        )

    source_priority = {"direct": 1, "expansion": 0}
    candidate_rows = sorted(
        candidate_rows,
        key=lambda r: (
            source_priority.get(str(r["source"]), 0),
            float(r["weighted_score"]),
            float(r["term_score"]),
        ),
        reverse=True,
    )

    selected_rows: list[dict[str, object]] = []
    selected_indices: list[int] = []

    for row in candidate_rows:
        parcel_idx = int(row["parcel_idx"])

        if parcel_idx in selected_indices:
            continue

        too_redundant = False
        for old_idx in selected_indices:
            jacc = _binary_mask_jaccard(atlas_masks_np, parcel_idx, old_idx)
            if jacc >= config.max_spatial_jaccard:
                too_redundant = True
                break

        if too_redundant:
            continue

        selected_indices.append(parcel_idx)
        selected_rows.append(row)

        if len(selected_indices) >= config.top_k_parcels:
            break

    if len(selected_indices) < config.top_k_parcels:
        for row in candidate_rows:
            parcel_idx = int(row["parcel_idx"])

            if parcel_idx in selected_indices:
                continue

            selected_indices.append(parcel_idx)
            selected_rows.append(row)

            if len(selected_indices) >= config.top_k_parcels:
                break

    parcel_scores = np.zeros(n_parcels, dtype=np.float32)

    for row in selected_rows:
        parcel_idx = int(row["parcel_idx"])
        parcel_scores[parcel_idx] = max(
            parcel_scores[parcel_idx],
            float(row["weighted_score"]),
        )

    if parcel_scores.max() > 0:
        parcel_scores = parcel_scores / parcel_scores.max()

    keep_set = {
        int(i)
        for i in selected_indices
        if parcel_scores[int(i)] >= config.min_final_parcel_score_frac
    }

    selected_rows = [
        row for row in selected_rows
        if int(row["parcel_idx"]) in keep_set
    ]

    selected_indices_final = np.asarray(
        [int(row["parcel_idx"]) for row in selected_rows],
        dtype=np.int64,
    )

    return selected_indices_final, parcel_scores, selected_rows


# %%
def build_atlas_matrix(masker_loader, *, fwhm=8.0):
    """Convert the AAL-SPM12 atlas into normalized vectors in masker space."""
    atlas = datasets.fetch_atlas_aal(version="SPM12")
    atlas_img = load_img(atlas.maps)
    masker = masker_loader()
    atlas_img = resample_to_img(
        atlas_img,
        masker.mask_img_,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )
    atlas_data = atlas_img.get_fdata()

    vectors = []
    labels = []
    for index, label in zip(atlas.indices, atlas.labels, strict=True):
        region_data = (atlas_data == float(index)).astype(np.float32)
        if not region_data.any():
            continue
        region_img = nib.Nifti1Image(region_data, atlas_img.affine, atlas_img.header)
        if fwhm:
            region_img = smooth_img(region_img, fwhm=fwhm)
        vector = np.asarray(masker.transform(region_img)).reshape(-1).astype(np.float32)
        peak = float(vector.max(initial=0.0))
        if peak <= 0:
            continue
        vectors.append(vector / peak)
        labels.append(str(label))

    if not vectors:
        raise RuntimeError("AAL atlas produced no parcels inside the NeuroVLM mask.")
    return torch.from_numpy(np.stack(vectors)), np.asarray(labels)


ATLAS_FWHM = 8
image_t, text_t = build_atlas_matrix(load_masker, fwhm=ATLAS_FWHM)
image_b = (image_t > 0.1).long()
m = image_b.sum(dim=1) != 0
image_b = image_b[m]
text_b = text_t[m]

# %%
all_masks, all_parcel_names = concatenate_atlases({
    "atlas": (image_b, text_b),
})

# all_masks = [
#     all_masks[i] for i in range(len(all_parcel_names))
#     if "Cerebral" not in all_parcel_names[i]
# ]

# all_parcel_names = [
#     all_parcel_names[i] for i in range(len(all_parcel_names))
#     if "Cerebral" not in all_parcel_names[i]
# ]

# %%
df_region_terms = make_parcel_ngram_dataframe(
    brain_maps=images,          # n_papers × 28542
    atlas_masks=all_masks,          # n_parcels × 28542
    atlas_parcel_names=all_parcel_names,
    labels=labels,                  # ngram labels from previous step
    X_bool=X_bool,                  # paper × ngram sparse bool matrix
    pmids=pmids,
    config=ParcelNgramConfig(
        overlap_mode="mass_fraction",
        min_overlap=0.005,
        top_k_parcels_per_paper=50,
        top_k_ngrams_per_parcel=50,
        max_pmids_per_parcel=500,
        rank_by="count_lift",
        min_parcel_docs=1,
        min_ngram_count_in_parcel=1,
    ),
)
df_region_terms.to_parquet(REGION_TERMS_PATH)

image_b.shape, text_t.shape


# %%
images_agg = torch.zeros((len(df_region_terms), images.shape[1]), dtype=torch.float32)
for i in tqdm(range(len(df_region_terms)), total=len(df_region_terms)):
    x = images[df_text["pmid"].isin(df_region_terms.iloc[i]["list_of_pmids"])].sum(dim=0)
    peak = x.max()
    if peak > 0:
        images_agg[i] = x / peak

torch.save(images_agg, IMAGES_AGG_PATH)


# %% [markdown]
# ## Build and save the aligned synthetic maps
#

# %%
xx = [j for i in df_region_terms["list_of_ngrams"] for j in i]
xx = np.unique(np.array(xx))
print(xx.shape)

xx = pd.Series(xx)
xx = xx[~xx.str.contains("the")]
xx = xx[~xx.str.contains("during")]
xx = xx[~xx.str.contains("appied")]
xx = xx[~xx.str.contains("with")]
xx = xx[~xx.str.contains("activity")]
xx = xx[~xx.str.contains("between")]
xx = xx[~xx.str.contains(" of ")]
xx = xx[~xx.str.contains("but")]
xx = xx[~xx.str.contains("can")]
xx = xx[~xx.str.contains("change")]
xx = xx[~xx.str.contains("suggest")]
xx = xx[~xx.str.contains("exhibited")]
xx = xx[~xx.str.contains("finally")]
xx = xx[~xx.str.contains("functional mri")]

# Drop a bad terms, that are spatially specific
# there was only ~1500 terms to go through
for i in [
    'functional connectivity fc',
    'functional connectivity rsfc',
    #'functional mri',
    'functional near',
    'functional near infrared',
    #'functional near infrared spectroscopy',
    'functional near infrared spectroscopy fnirs',
    'functionally defined',
    'functioning',
    'functions'
]:
    xx = xx[~xx.str.contains(i)]

for i in [
    'indexed', 'induced', 'making', 'mean', 'maps', 'loss',
    'aim', "using", 'examined', 'present', "use", "total", "size",
    "similar", "which", "who", "will"
]:
    xx = xx[~xx.str.contains(i)]

for i in [
    'still', 'static','states', 'shown', 'sample', 'measure', 'measured', 'measures', '^lobes$', 'hemisphere', 'five', 'follow',
    'evaulate', '^even$', 'derived', 'creates', 'dataset', 'cross', '^central$', 'centers', 'categories', 'assess', 'any', 'approach',
    'revealed', 'among', "after", 'additionally$', 'assessment',  'ability', 'accurate', 'studies', "have", "participated", '^based',
    'estimates', 'bodies', '^brain activation$', '^brain areas$', '^brain functional$', '^brain networks$', '^brain structure$', '^spontaneous brain$',
    "when", 'unclear', 'characteristics', 'stonger', 'years', 'under',
    'baseline', '^cessation$', 'cohort', 'comparison', 'compelling',
    '^delay mid$', "^monetary incentive delay mid$", "influence", "obtaining", 'pathway',
    "resolution", "^second$", "^secondary$", 'modelled', '^red$', 'performs', 'stronger', "thus", '^triangular$',
    'systems', 'ultra', 'we', "three", "systems", '^clinical cognitive$', '^neurocognitive$', "consistent", 'interaction', 'included', 'remains',
    "^affect$"
]:
    xx = xx[~xx.str.contains(i)]


for i in [
    'attended', 'before', 'arising', 'determine', 'factors', 'dots', 'matched healthy', 'fifteen healthy',
    'first', 'generic', 'fmri data', 'rs fmri', '^state fmri$', 'increases', 'indicated', 'order'
]:
    xx = xx[~xx.str.contains(i)]

for i in [
    "^lobe$", "^deep$", "^form$", "^lateral$", "^rat", "^structure$", "^sulcus$", "^gyrus$", "^gyri$", "common",
    'structures', '^cognition$', '^object$', '^objective$', '^objects$'
]:
    xx = xx[~xx.str.contains(i)]


for i in [
    'nuclei', 'several', 'behavior', 'complex', 'actions', 'actions', 'pre', 'anatomical', 'component',
    'cluster', 'clusters', 'signal', 'regulation', 'multi', 'post', 'body', 'value', 'post', 'working', 'lobule', 'lobules', 'error',
]:
    s = f"^{i}$"
    xx = xx[~xx.str.contains(s)]

# xx = xx.tolist()
print(xx.shape)

# %%
source_synth_sha256 = sha256_file(SYNTH_TABLE_PATH)
df_synth = pd.read_parquet(SYNTH_TABLE_PATH)

t = np.unique([j for i in df_region_terms["list_of_ngrams"] for j in i])
t = t[pd.Series(t).isin(xx)]

w = np.zeros((len(t), len(df_region_terms)))
ngrams = [np.array([i['ngram'] for i in df_region_terms.iloc[j]["ngram_stats"]]) for j in range(len(df_region_terms)) ]

for i in range(len(t)):
    for j in range(len(df_region_terms)):
        if t[i] in ngrams[j]:
            idx = int(np.where(ngrams[j] == t[i])[0][0])
            w[i, j] = df_region_terms.iloc[j]['ngram_stats'][idx]["rank_score"]
            #w[i, j] = df_region_terms.iloc[j]['ngram_stats'][idx]["count"]

asort = np.argsort(w)[:, ::-1]

# %%

t_s = (df_synth['title'] + " [SEP] " + df_synth["description"]).str.lower()

# mask
mm = t_s.str.contains("network")
t_s = t_s[mm]

t = pd.Series(t).fillna("").astype(str).str.lower().to_numpy()
t_s = pd.Series(t_s).fillna("").astype(str).str.lower()

isin = np.column_stack([
    t_s.str.contains(term, regex=False, na=False).to_numpy(dtype=bool)
    for term in t
])

# Remove all zero rows
_m = isin.sum(axis=1) > 0
_df_synth = df_synth[mm][_m].reset_index(drop=True)
isin = isin[_m]
text = np.array(t_s)[_m].tolist()

# Stack images
images_gen = torch.zeros((len(_df_synth), 28_542))
top_k = 10
for idx in tqdm(range(len(_df_synth)), total=len(_df_synth)):
    matched_t_inds = np.flatnonzero(isin[idx])
    matched_image_inds = np.array([asort[i][:top_k] for i in matched_t_inds])
    ws = torch.from_numpy(np.array([w[i][asort[i][:top_k]] for i in matched_t_inds]))
    xx = (images_agg[matched_image_inds] * ws.unsqueeze(2)).sum(dim=(0, 1))
    images_gen[idx] = (xx / xx.quantile(0.99)).clamp(0, 1)

# Remove all zero rows
_m = ~images_gen.isnan().any(dim=1)
images_gen = images_gen[_m]
text = np.array(t_s)[_m].tolist()

# %%
t_t = df_synth["title"].str.lower()

# mask
mm = ~t_t.str.contains("network")
t_t = t_t[mm]

t_t = pd.Series(t_t).fillna("").astype(str).str.lower()
isin = np.column_stack([
    t_t.str.contains(term, regex=False, na=False).to_numpy(dtype=bool)
    for term in t
])

# Remove all zero rows
_m = isin.sum(axis=1) > 0
_df_synth_t = df_synth[mm][_m].reset_index(drop=True)
isin = isin[_m]
_text = np.array(t_t)[_m].tolist()


# Merge top matches
_images_gen = torch.zeros((len(_df_synth_t), 28_542))
top_k = 10
for idx in tqdm(range(len(_df_synth_t)), total=len(_df_synth_t)):
    matched_t_inds = np.flatnonzero(isin[idx])
    matched_image_inds = np.array([asort[i][:top_k] for i in matched_t_inds])
    ws = torch.from_numpy(np.array([w[i][asort[i][:top_k]] for i in matched_t_inds]))
    xx = (images_agg[matched_image_inds] * ws.unsqueeze(2)).sum(dim=(0, 1))
    _images_gen[idx] = (xx / xx.quantile(0.99)).clamp(0, 1)

# Remove all zero rows
_m = ~_images_gen.isnan().any(dim=1)
_images_gen = _images_gen[_m]
_text = np.array(_text)[_m].tolist()
isin = isin[_m.numpy()]

# %%
text = text + _text
images_gen = torch.vstack((images_gen, _images_gen))

df_synth_f = pd.concat((_df_synth, _df_synth_t))
df_synth_f["text"] = text


# %%
if len(images_gen) != len(df_synth_f) or len(text) != len(df_synth_f):
    raise RuntimeError("Synthetic images, text, and metadata are not aligned.")
if not torch.isfinite(images_gen).all():
    raise RuntimeError("Synthetic images contain non-finite values.")

torch.save(images_gen, IMAGES_SYNTH_PATH)
df_synth_f.to_parquet(SYNTH_TABLE_PATH)

atlas_lineage = {
    "producer": "docs/03_models/11_anatomical_atlases.ipynb",
    "recovered_from": "archived anatomical extraction notebook",
    "seed": SEED,
    "source_synth_sha256": source_synth_sha256,
    "pubmed_pmids_sha256": hashlib.sha256(
        "\n".join(map(str, pmids)).encode("utf-8")
    ).hexdigest(),
    "outputs": {
        path_label(IMAGES_SYNTH_PATH): sha256_file(IMAGES_SYNTH_PATH),
        path_label(SYNTH_TABLE_PATH): sha256_file(SYNTH_TABLE_PATH),
    },
}
ATLAS_LINEAGE_PATH.write_text(json.dumps(atlas_lineage, indent=2) + "\n")

print(f"saved {len(df_synth_f):,} aligned examples")
print(IMAGES_SYNTH_PATH)
print(SYNTH_TABLE_PATH)
print(ATLAS_LINEAGE_PATH)

