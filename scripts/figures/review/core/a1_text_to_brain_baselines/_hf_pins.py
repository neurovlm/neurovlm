"""Pinned Hugging Face inputs of this directory: revisions and sha256 from scripts/config/hf_manifest.json, read as a file (nothing
is imported from scripts/). The same file sits in every review directory; keep the copies identical.

Files the scripts read themselves come from ``hf_file`` (pinned revision, sha256-checked). bert_score takes no revision, so
it gets the pinned snapshot dir (``bertscore_model``). The neurovlm loaders take no revision either but read the HF cache
first: the wrappers below (``load_dataset``, ``load_latent``, ``load_masker``) first check that scripts/data/fetch_hf.py
pointed the cache at the pinned revisions (``hf_cache_pinned``). Released weights are built from the pinned safetensors
(``load_model``, ``released_qformer``, as scripts/models/_common_models.py) and NeuroQwen3 is set at its pinned revision
(``pin_neuroqwen``), so nothing lazy-loads the HF main branch. It generates Latin-script tokens only
(``latin_script_suppress_tokens``) and, after a [NETWORK] / [REGION] steering token, a title from the final run's
stage-14 canonical names, also after a tag the model writes itself (``install_title_constraint`` with
runs/final/title_vocabulary.json, read as a file; ``pin_neuroqwen(nvlm, free_titles=True)`` decodes free titles), both
as scripts/models/_common_models.py.
"""
from __future__ import annotations

import contextlib
import functools
import hashlib
import json
from pathlib import Path

import numpy as np

MANIFEST = Path(__file__).resolve().parents[4] / "config" / "hf_manifest.json"
# canonical [NETWORK] / [REGION] / [FUNCTION] names of the final run's stage-14 banks (scripts/models/_common_models.title_vocabulary)
TITLE_VOCABULARY = MANIFEST.parent / "outputs" / "runs" / "final" / "title_vocabulary.json"
NEURO_QWEN = "neurovlm/NeuroQwen3-0.6B"
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
SPECTER_REPOS = ("allenai/specter2_aug2023refresh_base", "allenai/specter2_aug2023refresh_adhoc_query")
# what the neurovlm loaders (load_dataset, load_latent, load_masker, load_model) and the NeuroVLM client's lazy loaders read
# through the HF cache
LIBRARY_REPOS = ("neurovlm/neuro_image_papers", "neurovlm/embedded_text", "neurovlm/cognitive_atlas", "neurovlm/NeuroAutoEncoder",
                 "neurovlm/ProjectionHeads", "neurovlm/NeuroAdapter", "neurovlm/NeuroQformer", "neurovlm/NeuroQformer-PubMed",
                 NEURO_QWEN, *SPECTER_REPOS)
BERTSCORE_REPOS = {"roberta-large": "FacebookAI/roberta-large"}  # bert_score model_type -> HF repo
# released weights by name: (HF repo, file), as scripts/models/_common_models.RELEASED
RELEASED = {
    "autoencoder": ("neurovlm/NeuroAutoEncoder", "autoencoder.safetensors"),
    "proj_head_text_infonce": ("neurovlm/ProjectionHeads", "proj_head_text_infonce.safetensors"),
    "proj_head_image_infonce": ("neurovlm/ProjectionHeads", "proj_head_image_infonce.safetensors"),
    "proj_head_text_mse": ("neurovlm/ProjectionHeads", "proj_head_text_mse.safetensors"),
    "neuro_adapter": ("neurovlm/NeuroAdapter", "model.safetensors"),
}
QFORMER_REPOS = {"canonical": "neurovlm/NeuroQformer", "pubmed": "neurovlm/NeuroQformer-PubMed"}


@functools.cache
def _manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def _entries(repo_id: str, *, files_only: bool = False) -> list[dict]:
    m = _manifest()
    return [e for e in m["files"] + ([] if files_only else m.get("pins", [])) if e["repo"] == repo_id]


def hf_revision(repo_id: str) -> str:
    """Main-branch revision of ``repo_id`` pinned in scripts/config/hf_manifest.json."""
    if "/" not in repo_id and not _entries(repo_id):  # short sentence-transformers names ("all-MiniLM-L6-v2")
        repo_id = f"sentence-transformers/{repo_id}"
    revs = {e["revision"] for e in _entries(repo_id) if e.get("ref", "main") == "main"}
    if len(revs) != 1:
        raise KeyError(f"{repo_id} is not pinned in scripts/config/hf_manifest.json")
    return revs.pop()


@functools.cache
def _sha256(path: str, size: int, mtime: float) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def hf_file(repo_id: str, path: str) -> Path:
    """``repo_id/path`` at its pinned revision (HF cache, else downloaded), sha256-checked."""
    rows = [e for e in _entries(repo_id, files_only=True) if e["path"] == path]
    if len(rows) != 1:
        raise KeyError(f"{repo_id}/{path} is not pinned in scripts/config/hf_manifest.json")
    e = rows[0]
    from huggingface_hub import hf_hub_download

    p = Path(hf_hub_download(repo_id, path, repo_type=e["repo_type"], revision=e["revision"]))
    st = p.stat()
    if _sha256(str(p), st.st_size, st.st_mtime) != e["sha256"]:
        raise RuntimeError(f"{p} ({repo_id}/{path}@{e['revision'][:8]}) does not match the pinned sha256")
    return p


def hf_snapshot(repo_id: str) -> Path:
    """Snapshot dir of ``repo_id`` at its pinned main revision, every manifest file of it checked. Repos pinned by
    revision only get their config, tokenizer and safetensors files, unchecked (as scripts/common.hf_pinned_dir)."""
    rev = hf_revision(repo_id)
    files = [e for e in _entries(repo_id, files_only=True) if e["revision"] == rev]
    if not files:
        from huggingface_hub import snapshot_download

        e = _entries(repo_id)[0]
        return Path(snapshot_download(repo_id, repo_type=e["repo_type"], revision=rev,
                                      allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors"]))
    for e in files:
        p = hf_file(repo_id, e["path"])
    return p.parents[len(Path(e["path"]).parts) - 1]  # snapshots/<revision>/


def bertscore_model(model_type: str) -> dict:
    """``BERTScorer`` / ``bert_score.score`` kwargs for ``model_type`` at its pinned revision: the snapshot dir and the
    layer bert_score uses for that model (it takes no revision)."""
    from bert_score.utils import model2layers

    return {"model_type": str(hf_snapshot(BERTSCORE_REPOS.get(model_type, model_type))), "num_layers": model2layers[model_type]}


def sentence_transformer(name: str = MINILM, device="cpu"):
    """``SentenceTransformer(name)`` at its pinned revision."""
    from sentence_transformers import SentenceTransformer

    repo = name if "/" in name else f"sentence-transformers/{name}"
    return SentenceTransformer(repo, revision=hf_revision(repo), device=str(device))


@functools.cache
def hf_cache_pinned(*repos: str, group: str = "figures") -> None:
    """Raise unless the HF cache serves every manifest file of ``repos`` (default LIBRARY_REPOS) in ``group`` at its
    pinned revision: refs/<ref> set to it (scripts/data/fetch_hf.py does that) and the file present. The neurovlm loaders
    take no revision and try the cache first, so they then read exactly the pinned files."""
    from huggingface_hub import try_to_load_from_cache
    from huggingface_hub.constants import HF_HUB_CACHE

    bad = []
    for repo in repos or LIBRARY_REPOS:
        rows = [e for e in _entries(repo, files_only=True) if e.get("hf_cache") and group in e.get("groups", [group])]
        if not rows:
            bad.append(f"{repo}: no {group} entry in scripts/config/hf_manifest.json")
        for e in rows:
            ref = e.get("ref", "main")
            p = Path(HF_HUB_CACHE) / f"{e['repo_type']}s--{repo.replace('/', '--')}" / "refs" / ref
            cur = p.read_text().strip() if p.exists() else None
            if cur != e["revision"]:
                bad.append(f"{repo}: HF cache refs/{ref} is {cur}, pinned {e['revision'][:8]}")
            elif not isinstance(try_to_load_from_cache(repo, e["path"], revision=e["revision"], repo_type=e["repo_type"]), str):
                bad.append(f"{repo}/{e['path']}@{e['revision'][:8]} is not in the HF cache")
    if bad:
        raise RuntimeError(f"the HF cache is not at the pinned revisions (run scripts/data/fetch_hf.py --only {group}):\n  "
                           + "\n  ".join(sorted(set(bad))))


def load_dataset(name: str):
    """``neurovlm.data.load_dataset`` after ``hf_cache_pinned()``."""
    from neurovlm.data import load_dataset as _load

    hf_cache_pinned()
    return _load(name)


def load_latent(name: str):
    """``neurovlm.data.load_latent`` after ``hf_cache_pinned()``."""
    from neurovlm.data import load_latent as _load

    hf_cache_pinned()
    return _load(name)


def load_masker():
    """``neurovlm.data.load_masker`` (NeuroAutoEncoder mask.npz) after ``hf_cache_pinned()``."""
    from neurovlm.data import load_masker as _load

    hf_cache_pinned()
    return _load()


def _released_module(name: str):
    from safetensors.torch import load_file
    from neurovlm.models import ProjHead
    from neurovlm.models.base import NeuroAutoEncoder

    repo, fname = RELEASED[name]
    if name == "neuro_adapter":  # as neurovlm.resources.loaders._load_neuro_adapter
        from neurovlm.models.adapter import InterleavedDecoderAdapter

        cfg = json.loads(hf_file(repo, "config.json").read_text()).get("adapter", {})
        module = InterleavedDecoderAdapter(_released_module("autoencoder"), _released_module("proj_head_text_mse"), **cfg)
    elif name == "autoencoder":
        module = NeuroAutoEncoder(seed=0, out="logit")
    else:
        module = ProjHead(384, 384, 384) if name == "proj_head_image_infonce" else ProjHead(768, 512, 384)
    module.load_state_dict(load_file(str(hf_file(repo, fname))), strict=True)
    return module


def load_model(name: str):
    """Drop-in for ``neurovlm.models.load_model(name)`` (CPU, eval mode): the released safetensors at the pinned revision,
    or SPECTER2 + adhoc_query from the library after ``hf_cache_pinned`` (its loader takes no revision)."""
    if name == "specter":
        from neurovlm.models import load_model as _load

        hf_cache_pinned(*SPECTER_REPOS)
        return _load(name)
    if name not in RELEASED:
        raise KeyError(f"{name}: no pinned release (add it to RELEASED in _hf_pins.py)")
    return _released_module(name).eval()


def released_qformer(variant: str = "canonical", device="cpu", *, projection_temp: float | None = 0.05,
                     canonical_basis: str | None = "all", use_canonical_projection: bool | None = None):
    """Released NeuroQFormer package (canonical: neurovlm/NeuroQformer, pubmed: neurovlm/NeuroQformer-PubMed) at the pinned
    revision; same construction and defaults as neurovlm.resources.loaders._load_neuro_qformer."""
    from safetensors.torch import load_file
    from neurovlm.models.qformer import NeuroQFormer

    repo = QFORMER_REPOS[variant]
    cfg = json.loads(hf_file(repo, "config.json").read_text())
    payload = {"state_dict": load_file(str(hf_file(repo, "model.safetensors")), device="cpu"), "config": cfg.get("qformer", {}),
               "canonical_metadata": cfg.get("canonical_metadata", {})}
    model = NeuroQFormer.from_state_dict_payload(payload, map_location=device).eval()
    model.projection_temp, model.canonical_basis = projection_temp, canonical_basis
    if use_canonical_projection is not None:
        model.use_canonical_projection = use_canonical_projection
    return model


# characters a caption may hold: ASCII, Latin-1, Greek, punctuation and symbols (U+2000-2BFF)
CAPTION_CHAR_RANGES = ((0x0000, 0x00FF), (0x0370, 0x03FF), (0x2000, 0x2BFF))


@functools.cache
def latin_script_suppress_tokens(tokenizer) -> tuple[int, ...]:
    """Sorted ids of every non-special token whose text has a character outside CAPTION_CHAR_RANGES (CJK, Cyrillic,
    Thai, ...) or is not valid UTF-8 alone (byte fragments); the frozen multilingual LM drifts into them on soft prefixes."""
    from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

    byte_of = {c: b for b, c in bytes_to_unicode().items()}  # byte-level BPE alphabet -> byte
    special = set(tokenizer.all_special_ids) | set(tokenizer.added_tokens_decoder)
    ids = []
    for i in range(len(tokenizer)):
        tok = tokenizer.convert_ids_to_tokens(i)
        if i in special or tok is None or any(c not in byte_of for c in tok):
            continue
        try:
            text = bytes(byte_of[c] for c in tok).decode("utf-8")
        except UnicodeDecodeError:
            ids.append(i)
            continue
        if not all(any(a <= ord(c) <= b for a, b in CAPTION_CHAR_RANGES) for c in text):
            ids.append(i)
    return tuple(ids)


# Caption titles, as scripts/models/_common_models.py: after a prompt that ends with the [NETWORK] or [REGION] steering
# token the title (up to its newline) can only be a canonical name of that level, and so can the title after such a tag
# the model writes itself when the prompt ends with no tag (basis "all"); the body and [FUNCTION] titles stay free. A
# trie of the names' token ids and a logits processor; generate of the LM instance is wrapped because
# NeuroVLM.generate_text passes no logits processor.
TITLE_LEVELS = ("network", "region")
STEERING_LEVELS = ("network", "region", "function")  # steering tags; a written [FUNCTION] tag ends a constrained title
TITLE_END_TEXTS = ("\n", "\n\n")


def read_title_vocabulary(path: Path = TITLE_VOCABULARY) -> dict:
    """{"titles": {level: names}, "bank_rows": {level: name of each bank row}, "source": ...} of a run's stage-14 banks,
    checked against the canonical_semantic_banks.pt next to it."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: scripts/pipeline/validate_models.py (or a6_steering_tokens/build_title_vocabulary.py) "
                                "builds it from the run's stage-14 banks")
    vocab = json.loads(path.read_text())
    banks = path.parent / str(vocab.get("source", {}).get("banks", ""))
    want = vocab.get("source", {}).get("banks_sha256")
    if want and banks.is_file() and _sha256(str(banks), banks.stat().st_size, banks.stat().st_mtime) != want:
        raise RuntimeError(f"{path} was built from other banks than {banks}; rebuild it (scripts/pipeline/validate_models.py)")
    return vocab


def title_trie(tokenizer, names) -> dict:
    """Nested {token id: subtrie} of ``names``, tokenised alone (as after the [LEVEL] tag); key -1 ends a name."""
    root: dict = {}
    for name in names:
        node = root
        for t in tokenizer(str(name), add_special_tokens=False)["input_ids"]:
            node = node.setdefault(int(t), {})
        node[-1] = {}
    return root


@functools.cache
def newline_token_ids(tokenizer) -> frozenset:
    """Ids of every token whose text holds a newline (any of them ends a free title)."""
    return frozenset(i for i in range(len(tokenizer)) if "\n" in tokenizer.decode([i]))


@functools.cache
def open_bracket_token_ids(tokenizer) -> frozenset:
    """Ids of every token whose text ends with '[' (the first token of a steering tag the model writes itself)."""
    return frozenset(i for i in range(len(tokenizer)) if tokenizer.decode([i]).endswith("["))


class TitleConstraint:
    """Logits processor for ``model.generate``: row r (beam-major, as HF orders beams) keeps its title on
    ``tries[r // num_beams]`` until a newline ends it; ``None`` leaves that map's title free. With ``written`` ({level:
    (tag token ids after its '[', trie or None)}), a row whose prompt ends with no steering tag (``tagged[r // num_beams]``
    False, e.g. basis "all") also keeps the title after the last tag it wrote itself on that level's trie, until a
    newline ends it (None: [FUNCTION], free), and closes such a tag with its own ']' token (not a merged ']]', '][' or
    ']\n'). ``offset``: prompt tokens at the start of input_ids (0 with inputs_embeds). Text after a title is never
    changed."""

    def __init__(self, tokenizer, tries, num_beams: int = 1, *, written=None, tagged=None, offset: int = 0):
        self.tries, self.num_beams = list(tries), int(num_beams)
        self.end_ids = [tokenizer(t, add_special_tokens=False)["input_ids"][0] for t in TITLE_END_TEXTS]
        self.newline = newline_token_ids(tokenizer)
        self.written, self.offset = written or {}, int(offset)
        self.tagged = [t is not None for t in self.tries] if tagged is None else [bool(t) for t in tagged]
        if self.written:
            self.open, self._ids = open_bracket_token_ids(tokenizer), {}

    def _ids_on(self, name, ids, device):
        import torch

        if name not in self._ids or self._ids[name].device != device:
            self._ids[name] = torch.as_tensor(sorted(ids), dtype=torch.long, device=device)
        return self._ids[name]

    def _written_titles(self, input_ids, rows) -> dict:
        """{row: trie node} for the rows (of ``rows``) inside a title after a tag they wrote (the last tag of the
        generated tokens is a [NETWORK] / [REGION] tag and no newline follows it) or inside such a tag (the tokens end
        with '[' and the level's name: the node allows only the tag's closing token)."""
        import torch

        x = input_ids[rows][:, self.offset:]
        n, L = x.shape
        pos = torch.arange(L, device=x.device).expand(n, L)
        nl = torch.where(torch.isin(x, self._ids_on("newline", self.newline, x.device)), pos, -1).amax(1)
        best, level = torch.full((n,), -1, device=x.device), [None] * n
        for lv, (tag, _) in self.written.items():
            k = len(tag)
            if L <= k:
                continue
            hit = torch.isin(x[:, :L - k], self._ids_on("open", self.open, x.device))  # '[' at s, tag ids at s+1..s+k
            for j, t in enumerate(tag):
                hit &= x[:, 1 + j:L - k + 1 + j] == t
            end = torch.where(hit, pos[:, :L - k] + k, -1).amax(1)
            for i in torch.nonzero(end > best).flatten().tolist():
                level[i] = lv
            best = torch.maximum(best, end)
        out = {}
        for i in torch.nonzero(best > nl).flatten().tolist():
            node = self.written[level[i]][1]
            for t in x[i, int(best[i]) + 1:].tolist():
                if node is None:
                    break
                node = node.get(t)
            if node is not None:
                out[rows[i]] = node
        for lv, (tag, trie) in self.written.items():
            k = len(tag)
            if trie is None or L < k:
                continue
            open_tag = torch.isin(x[:, L - k], self._ids_on("open", self.open, x.device))
            for j, t in enumerate(tag[:-1]):
                open_tag &= x[:, L - k + 1 + j] == t
            for i in torch.nonzero(open_tag).flatten().tolist():
                out[rows[i]] = {tag[-1]: {}}
        return out

    def __call__(self, input_ids, scores):
        import torch

        nodes = {}
        for r, seq in enumerate(input_ids.tolist()):
            if not self.tagged[r // self.num_beams]:
                continue
            node = self.tries[r // self.num_beams]
            for t in seq:
                if node is None or t in self.newline:
                    node = None
                    break
                node = node.get(t)
            if node is not None:
                nodes[r] = node
        if self.written:
            free = [r for r in range(input_ids.shape[0]) if not self.tagged[r // self.num_beams]]
            if free and input_ids.shape[1] > self.offset:
                nodes.update(self._written_titles(input_ids, free))
        for r, node in nodes.items():
            allowed = [t for t in node if t >= 0] + (self.end_ids if -1 in node else [])
            keep = scores[r, allowed]
            scores[r] = float("-inf")
            # a name token already banned by another processor (repetition) stays possible, at a large cost
            scores[r, allowed] = torch.where(torch.isinf(keep), torch.full_like(keep, -1e4), keep)
        return scores


def _prompt_levels(lm, tags, args, kwargs) -> list:
    """Per prompt row: the level whose steering tag ends the prompt (its token embeddings as the last inputs_embeds rows,
    or its token ids as the last input_ids), else None."""
    import torch

    emb = kwargs.get("inputs_embeds")
    ids = kwargs.get("input_ids", args[0] if args else None)
    x = emb if emb is not None else ids
    if x is None:
        return []
    levels = [None] * x.shape[0]
    for lv, tag in tags.items():
        k = len(tag)
        if x.shape[1] < k:
            continue
        if emb is not None:
            table = lm.get_input_embeddings().weight
            ref = table[torch.as_tensor(tag, device=table.device)].to(device=emb.device, dtype=emb.dtype)
            hit = (emb[:, -k:, :] == ref).flatten(1).all(1)
        else:
            hit = (ids[:, -k:] == torch.as_tensor(tag, device=ids.device)).all(1)
        for r in torch.nonzero(hit).flatten().tolist():
            levels[r] = lv
    return levels


class TitleConstrainedGenerate:
    """``lm.generate`` plus a TitleConstraint (set by install_title_constraint) for the rows whose prompt ends with a
    constrained steering tag and for the tags written after a prompt without one. A caller's own TitleConstraint
    (constrained_titles) replaces the default."""

    def __init__(self, lm, generate):
        self.lm, self.generate = lm, generate

    def __call__(self, *args, **kwargs):
        spec = self.lm.__dict__.get("_title_spec")
        procs = list(kwargs.get("logits_processor") or [])
        if spec is not None and not any(type(p).__name__ == "TitleConstraint" for p in procs):
            levels = _prompt_levels(self.lm, spec["tags"], args, kwargs)  # [FUNCTION] prompts: tagged, free title
            tries = [spec["tries"].get(lv) for lv in levels]
            if any(t is not None for t in tries) or not all(lv is not None for lv in levels):
                from transformers import LogitsProcessorList

                ids = kwargs.get("input_ids", args[0] if args else None)
                beams = int(kwargs.get("num_beams") or getattr(self.lm.generation_config, "num_beams", 1) or 1)
                kwargs["logits_processor"] = LogitsProcessorList([*procs, TitleConstraint(
                    spec["tokenizer"], tries, beams, written=spec["written"], tagged=[lv is not None for lv in levels],
                    offset=0 if ids is None else ids.shape[1])])
        return self.generate(*args, **kwargs)


def install_title_constraint(lm, tokenizer, titles) -> None:
    """The title after a [NETWORK] / [REGION] steering token, in the prompt or written by the model, is one of
    ``titles[level]``; None = free titles. Wraps ``generate`` of this LM instance once; a second call only changes the
    names."""
    spec = None
    if titles is not None:
        names = {lv: sorted(set(map(str, titles[lv]))) for lv in TITLE_LEVELS}
        spec = {"tokenizer": tokenizer, "n": {lv: len(v) for lv, v in names.items()},
                "tags": {lv: list(tokenizer(f"[{lv.upper()}]", add_special_tokens=False)["input_ids"]) for lv in STEERING_LEVELS},
                "tries": {lv: title_trie(tokenizer, v) for lv, v in names.items()}}
        # a tag the model writes: any token ending in '[' and then the tag's other tokens ("[", "NETWORK", "]")
        assert all(tokenizer.decode(tag[:1]) == "[" for tag in spec["tags"].values()), spec["tags"]
        spec["written"] = {lv: (tag[1:], spec["tries"].get(lv)) for lv, tag in spec["tags"].items()}
    lm.__dict__["_title_spec"] = spec
    if not isinstance(lm.__dict__.get("generate"), TitleConstrainedGenerate):
        lm.__dict__["generate"] = TitleConstrainedGenerate(lm, lm.generate)


def title_constraint_note(lm) -> str:
    """'titles: canonical vocabulary (48 network / 83 region names)' or 'titles: free'."""
    spec = lm.__dict__.get("_title_spec")
    return "titles: free" if spec is None else "titles: canonical vocabulary (" + " / ".join(f"{n} {lv}" for lv, n in spec["n"].items()) + " names)"


@contextlib.contextmanager
def free_titles(lm):
    """Inside the block the LM decodes free titles."""
    spec = lm.__dict__.get("_title_spec")
    lm.__dict__["_title_spec"] = None
    try:
        yield
    finally:
        lm.__dict__["_title_spec"] = spec


@contextlib.contextmanager
def constrained_titles(lm, processor):
    """Inside the block ``lm.generate`` also applies ``processor`` (e.g. a TitleConstraint with other names per map,
    which then replaces the default constraint)."""
    from transformers import LogitsProcessorList

    own = lm.__dict__.get("generate")
    lm.generate = functools.partial(lm.generate, logits_processor=LogitsProcessorList([processor]))
    try:
        yield
    finally:
        if own is None:
            del lm.generate
        else:
            lm.generate = own


def pin_neuroqwen(nvlm, *, free_titles: bool = False, title_vocabulary: Path = TITLE_VOCABULARY):
    """Give the NeuroVLM client NeuroQwen3-0.6B at its pinned revision, with the client's tokenizer, dtype and pad settings
    (its lazy loader would read the HF main branch), restricted to Latin-script tokens and, unless ``free_titles``, to
    the canonical [NETWORK] / [REGION] titles of ``title_vocabulary`` (default: the final run's). Keeps an LM that is
    already set."""
    if getattr(nvlm, "_neuro_qwen_model", None) is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        rev = hf_revision(NEURO_QWEN)
        tok = AutoTokenizer.from_pretrained(NEURO_QWEN, revision=rev, use_fast=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        dtype = torch.float32 if nvlm.device.type == "cpu" else torch.bfloat16
        lm = AutoModelForCausalLM.from_pretrained(NEURO_QWEN, revision=rev, torch_dtype=dtype, device_map=None).to(nvlm.device).eval()
        if getattr(lm.config, "pad_token_id", None) is None:
            lm.config.pad_token_id = tok.pad_token_id
        nvlm._neuro_qwen_tokenizer, nvlm._neuro_qwen_model = tok, lm
    lm, tok = nvlm._neuro_qwen_model, nvlm._neuro_qwen_tokenizer
    lm.generation_config.suppress_tokens = list(latin_script_suppress_tokens(tok))
    install_title_constraint(lm, tok, None if free_titles else read_title_vocabulary(title_vocabulary)["titles"])
    return nvlm


def hf_split_pmids() -> dict[str, np.ndarray]:
    """{"train"|"val"|"test": sorted int64 PMIDs} from the split flags of the pinned neuro_image_papers/publications.parquet
    (set-identical to the final run's pmids_split.pt; as scripts/common.hf_split_pmids)."""
    import pandas as pd

    df = pd.read_parquet(hf_file("neurovlm/neuro_image_papers", "publications.parquet"), columns=["pmid", "train", "val", "test"])
    return {k: np.sort(df.loc[df[k].fillna(False).astype(bool), "pmid"].to_numpy(dtype=np.int64)) for k in ("train", "val", "test")}
