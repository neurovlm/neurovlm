#!/usr/bin/env python
"""Comment 2.m6: how often do captions name things no atlas names? Over the 2,971 stage-2 test captions and the a7
caption sets: (1) [NETWORK] / [REGION] titles that are not a title of the stage-14 bank of their level (and, of those,
the ones that are no atlas name either); (2) caption bodies with an anatomical-looking phrase (ending in cortex,
network, gyrus, area, lobe or system) that is no Harvard-Oxford / AAL / Yeo / Du / bank name, and phrases with a word
that occurs in none of these names (Pauli and MNI-lobe names included; a phrase stops at a function word or punctuation).
    .venv/bin/python scripts/figures/review/core/a7_failure_cases_percentiles/audit_names.py   # CPU, ~20 s
Needs ../a6_steering_tokens/outputs/title_vocabulary.json (build_title_vocabulary.py) and the nilearn atlases.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
VOCAB = HERE.parent / "a6_steering_tokens" / "outputs" / "title_vocabulary.json"
TEST_CAPTIONS = HERE.parents[4] / "scripts" / "outputs" / "validation" / "final" / "df_gen_canonical.parquet"
SETS = {"a7_networks": "generated_networks.csv", "a7_ica": "generated_ica.csv", "a7_neurovault": "generated_neurovault.csv",
        "a7_neurovault_allimages": "generated_neurovault_allimages.csv"}
HEADS = {"cortex": "cortex", "cortices": "cortex", "network": "network", "networks": "network", "gyrus": "gyrus", "gyri": "gyrus",
         "area": "area", "areas": "area", "lobe": "lobe", "lobes": "lobe", "system": "system", "systems": "system"}
AAL = ["precentral gyrus", "superior frontal gyrus orbital part", "middle frontal gyrus orbital part", "inferior frontal gyrus opercular part",
       "inferior frontal gyrus triangular part", "inferior frontal gyrus orbital part", "rolandic operculum", "supplementary motor area",
       "olfactory cortex", "medial superior frontal gyrus", "medial orbital frontal gyrus", "gyrus rectus", "insula", "anterior cingulate gyrus",
       "median cingulate gyrus", "middle cingulate gyrus", "posterior cingulate gyrus", "paracingulate gyri", "hippocampus", "parahippocampal gyrus",
       "amygdala", "calcarine fissure", "calcarine cortex", "cuneus", "lingual gyrus", "superior occipital gyrus", "middle occipital gyrus",
       "inferior occipital gyrus", "fusiform gyrus", "postcentral gyrus", "superior parietal gyrus", "inferior parietal gyrus", "supramarginal gyrus",
       "angular gyrus", "precuneus", "paracentral lobule", "caudate nucleus", "lenticular nucleus putamen", "lenticular nucleus pallidum",
       "thalamus", "heschl gyrus", "superior temporal gyrus", "temporal pole", "middle temporal gyrus", "inferior temporal gyrus",
       "cerebellum", "cerebellar crus", "cerebellar hemisphere", "cerebellar vermis", "vermis"]
LOBES = ["frontal lobe", "parietal lobe", "temporal lobe", "occipital lobe", "limbic lobe", "insula"]  # MNI structural atlas
PAULI = ["putamen", "caudate nucleus", "nucleus accumbens", "external globus pallidus", "internal globus pallidus", "substantia nigra",
         "red nucleus", "subthalamic nucleus", "hypothalamus", "ventral tegmental area", "parabrachial pigmented nucleus", "mammillary nucleus",
         "ventral pallidum", "habenular nuclei"]  # Pauli 2017 subcortical atlas (used by a6)
YEO_DU = ["visual", "somatomotor", "dorsal attention", "ventral attention", "salience ventral attention", "salience", "limbic", "frontoparietal",
          "frontoparietal control", "control", "default", "default mode", "temporal parietal", "visual peripheral", "visual central", "auditory",
          "premotor posterior parietal rostral", "cingulo opercular", "salience parietal memory", "parietal memory", "language"]
POSITION = set("anterior posterior superior inferior medial lateral dorsal ventral rostral caudal middle mid central dorsolateral dorsomedial "
               "ventrolateral ventromedial orbital primary secondary left right bilateral upper lower inner outer deep basal intermediate".split())
GENERIC = set("brain cortical subcortical neural functional large scale whole cerebral association sensory higher order multimodal "
              "unimodal heteromodal core".split())
VERBS = set("involves involve involving includes include including links linking connects connecting supports supporting engages engaging "
            "spans spanning lies lying described associated tied linked related known thought considered called termed named comprising "
            "comprises encompassing encompasses covering covers spanning extends extending regions region areas like particularly especially "
            "notably mainly primarily typically commonly often highly".split())


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[-‐–/_,()]", " ", str(s).lower().replace("’", "'").replace("'s", "").replace("'", ""))).strip(" .:")


def atlas_names(bank_titles) -> set[str]:
    from nilearn import datasets

    out = {norm(x) for x in AAL + LOBES + PAULI + list(bank_titles)} | {norm(x + " network") for x in YEO_DU}
    for lab in datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm", verbose=0).labels[1:]:
        base, _, div = str(lab).partition(",")
        out |= {norm(m) for m in re.findall(r"\(formerly ([^)]*)\)", str(lab))}
        base = norm(re.sub(r"\(.*?\)", "", base))
        out.add(base)
        if div:
            out.add(norm(norm(div).replace(" division", "").replace(" part", "") + " " + base))
    for lab in datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm", verbose=0).labels[1:]:
        out.add(norm(re.sub(r"^(Left|Right) ", "", str(lab))))
    return out


class Vocabulary:
    def __init__(self, names: set[str]):
        self.keys = {n.replace(" ", "") for n in names}
        self.words = {w for n in names for w in n.split()} | POSITION | GENERIC | set(HEADS)

    def word_ok(self, w: str) -> bool:
        if w in self.words or w.rstrip("s") in self.words or w.isdigit() or (w.endswith("o") and w[:-1] + "al" in self.words):
            return True
        for i in range(3, len(w) - 2):  # combining forms: occipito+temporal, sensori+motor, fronto+parietal
            if w[i - 1] in "oi" and w[i:] in self.words and any(w[:i - 1] + e in self.words for e in ("al", "ar", "ic", "e", "")):
                return True
        return False

    def phrase(self, words: list[str]) -> tuple[bool, list[str]]:
        """(the phrase or one of its >= 2-word endings is a name, words found in no name)."""
        named = any("".join(words[k:]) in self.keys for k in range(len(words) - 1))
        return named, [w for w in words[:-1] for p in [w.split("-")] if not all(self.word_ok(x) for x in p if x)]


def phrases(body: str, stop: set[str]) -> list[list[str]]:
    """Anatomical-looking phrases: up to 4 content words right before a head noun, inside one clause."""
    out = []
    for clause in re.split(r"[.,;:()\[\]\n]|\band\b|\bor\b", body.lower().replace("’", "'")):
        words = [w.replace("'s", "").strip("'\"") for w in clause.split()]
        for j, w in enumerate(words):
            if w not in HEADS:
                continue
            k = j
            while k > 0 and j - k < 4 and words[k - 1] and words[k - 1] not in stop and re.fullmatch(r"[a-z][a-z-]*", words[k - 1]):
                k -= 1
            if k < j:
                out.append(words[k:j] + [HEADS[w]])
    return out


def load_captions() -> pd.DataFrame:
    rows = []
    d = pd.read_parquet(TEST_CAPTIONS)
    for lv in ("network", "region", "function"):
        rows += [("pubmed_test", str(pm), lv, str(t)) for pm, t in zip(d["pmid"], d[lv])]
    for name, f in SETS.items():
        g = pd.read_csv(OUT / f)
        for i, t in zip(g["id"], g["generated"]):
            m = re.match(r"\s*\[(NETWORK|REGION|FUNCTION)\]\s*", str(t))
            rows.append((name, str(i), m.group(1).lower() if m else "none", str(t)[m.end() if m else 0:]))
    df = pd.DataFrame(rows, columns=["set", "id", "level", "text"])
    df["title"] = df["text"].str.split("\n").str[0].str.strip()
    df["body"] = df["text"].str.split("\n", n=1).str[1].fillna("").str.strip()
    return df


def main() -> None:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    bank = json.loads(VOCAB.read_text())["titles"]
    names = atlas_names(sum(bank.values(), []))
    voc = Vocabulary(names)
    stop = set(ENGLISH_STOP_WORDS) | VERBS
    df = load_captions()
    canon = {lv: {norm(t) for t in bank[lv]} for lv in bank}
    df["title_canonical"] = [norm(t) in canon.get(lv, set()) for t, lv in zip(df["title"], df["level"])]
    df["title_atlas_name"] = [norm(t).replace(" ", "") in voc.keys for t in df["title"]]
    not_named, odd_words = [], []
    for body in df["body"]:
        res = [(" ".join(p), voc.phrase(p)) for p in phrases(body, stop)]
        not_named.append([p for p, (ok, _) in res if not ok])
        odd_words.append([p for p, (_, bad) in res if bad])
    df["body_not_a_name"], df["body_unknown_word"] = not_named, odd_words
    titled = df[df["level"].isin(["network", "region"])]
    summary = titled.groupby(["set", "level"]).agg(
        n=("title", "size"), title_not_canonical=("title_canonical", lambda x: int((~x).sum())),
        title_no_atlas_name_either=("title_atlas_name", lambda x: int((~x[~titled.loc[x.index, "title_canonical"]]).sum()))).reset_index()
    bodies = df.groupby("set").agg(n_captions=("body", "size"), body_phrase_not_a_name=("body_not_a_name", lambda x: int((x.str.len() > 0).sum())),
                                   body_phrase_unknown_word=("body_unknown_word", lambda x: int((x.str.len() > 0).sum()))).reset_index()
    summary.to_csv(OUT / "name_audit_titles.csv", index=False)
    bodies.to_csv(OUT / "name_audit_bodies.csv", index=False)
    rng = np.random.default_rng(0)
    lines = ["# Names outside the atlas and bank vocabulary (comment 2.m6)", "", "Produced by `audit_names.py`.", "",
             "```", summary.to_string(index=False), "", bodies.to_string(index=False), "```", ""]
    for lv in ("network", "region"):
        bad = titled[(titled["level"] == lv) & ~titled["title_canonical"] & ~titled["title_atlas_name"]]
        k = min(10, len(bad))  # titles are canonical by default, so few or none are left
        lines += [f"## {k} [{lv.upper()}] titles that are neither a bank title nor an atlas name (seeded draw)", ""]
        lines += [f"- {r['set']} {r['id']}: {r['title']}" for _, r in bad.iloc[np.sort(rng.choice(len(bad), k, replace=False))].iterrows()]
        lines += ["", f"Most frequent: " + "; ".join(f"{t} ({n})" for t, n in Counter(bad["title"]).most_common(15)), ""]
    for col, what in (("body_unknown_word", "with a word that occurs in no atlas or bank name"), ("body_not_a_name", "that are no atlas or bank name")):
        c = Counter(p for x in df[col] for p in x)
        flat = sorted(c)
        lines += [f"## 10 body phrases {what} (seeded draw of {len(flat)} distinct)", ""]
        lines += [f"- {flat[i]} ({c[flat[i]]})" for i in np.sort(rng.choice(len(flat), 10, replace=False))]
        lines += ["", "Most frequent: " + "; ".join(f"{t} ({n})" for t, n in c.most_common(20)), ""]
    (OUT / "name_audit.md").write_text("\n".join(lines))
    print(summary.to_string(index=False))
    print(bodies.to_string(index=False))


if __name__ == "__main__":
    main()
