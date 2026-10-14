"""Kind of the NeuroVault maps whose uploaders chose the map type 'other', read from the image and file names, the image
and collection descriptions, the declared types of sibling maps and the map values (in-mask quantiles).

First match wins: statistic tokens in the image or file name (zstat, tstat, spmT, spmF, ...), then measure tokens (Cohen's
d, correlation, con/beta images, searchlight accuracy, components, overlap), then mean or variance images (SPM ResMS/RPV,
mean tracer or BOLD images); the same rules on the image description; a map named like a declared T/Z sibling minus its
statistic suffix is that contrast's estimate; a statistic named in the collection description; a single declared T or Z
type among the siblings when the values are t/z-like; t/z-like values alone. Every rule checks the values (a T or Z map
reaches |value| >= 2 on both tails, an F map has no negative tail, a correlation stays within [-1, 1]).
"""
from __future__ import annotations

import re

import pandas as pd

GROUPS = {  # kind -> map type used in the evaluation (None: excluded)
    "T map": "other: t, z, or F", "Z map": "other: t, z, or F", "F map": "other: t, z, or F", "TFCE statistic": "other: t, z, or F",
    "t or z (values)": "other: t, z, or F",
    "contrast estimate": "other: effect estimate", "effect size (d)": "other: effect estimate", "correlation (r)": "other: effect estimate",
    "classifier accuracy": "other: effect estimate", "overlap map": "other: effect estimate",
    "component or gradient": "other: component or gradient",
    "mean or variance image": None, "unresolved": None,
}
STAT_SUFFIX = r"\s+(t ?stat|z ?scr|z ?stat|z ?score|t ?map|z ?map)\s*$"


def _w(alts: str) -> str:
    return r"\b(" + alts + r")\b"


NAME_RULES = [  # (kind, pattern, value condition)
    ("TFCE statistic", r"tfce", "nonneg"),
    ("Z map", r"zstat|zscore|zscr|z stat|z score|spm z|" + _w("z map|zmap"), "tz_scale"),
    ("T map", r"tstat|t stat|spm t|" + _w("t map|tmap|t test|t"), "tz_scale"),
    ("F map", r"spm f|fstat|f stat|f test|anova|" + _w("f|ess"), "nonneg"),
    ("effect size (d)", r"cohen|hedges|effect size", None),
    ("correlation (r)", r"pearson|correlation|" + _w("corr|rsa|isc|spearman|r map"), "unit"),
    ("contrast estimate", r"cope|parameter estimate|" + _w("con|beta|pe"), "small"),
    ("classifier accuracy", r"mvpa|searchlight|classifier|decoding|" + _w("svm"), None),
    ("component or gradient", r"tensor ic|gradient|factor|component|eigenbrain|" + _w("ic|ica|comp|dim"), None),
    ("overlap map", r"overlap|probabilistic|conjunction", None),
    ("mean or variance image", _w("res ?ms|rpv|coefficient of variation|standard deviation|hrf|avg|med|median|iqr|suvr|"
                                  "binding potential|bp|receptors?|transporter"), "nonneg_tail"),
    ("mean or variance image", _w("mean|average|dopamine|serotonin|opioid"), "pet_nonneg_tail"),
]


def text(*parts) -> str:
    """Lower-case words of the given strings (camelCase and letter-digit runs split, .nii.gz dropped)."""
    s = " ".join(p for p in parts if isinstance(p, str))
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    s = re.sub(r"\.nii(\.gz)?", " ", s)
    s = re.sub(r"[^A-Za-z0-9]+", " ", s).lower()
    return " " + re.sub(r"([a-z])(\d)", r"\1 \2", s) + " "


def value_flags(r) -> dict[str, bool]:
    q01, q99, lo, hi = r["q01"], r["q99"], r["q001"], r["q999"]
    signed = r["frac_negative"] >= 0.01 and q01 < 0
    nonneg_tail = q01 >= -0.33 * max(q99, 1e-12)
    return {"nonneg": not signed, "tz_scale": signed and max(abs(lo), hi) >= 2.0, "small": max(abs(lo), abs(hi)) < 2.0,
            "unit": max(abs(r["min"]), abs(r["max"])) <= 1.0001, "nonneg_tail": nonneg_tail,
            "pet_nonneg_tail": str(r["modality"]).startswith("PET") and nonneg_tail,
            "tz_like": signed and min(abs(q01), q99) >= 0.33 * max(abs(q01), q99) and 2.0 <= max(abs(lo), hi) <= 30}


def by_name(t: str, flags: dict[str, bool]) -> str | None:
    for kind, pat, cond in NAME_RULES:
        if re.search(pat, t) and (cond is None or flags[cond]):
            return kind
    return None


def infer(r, sib_types: set, sib_stems: set) -> tuple[str, str]:
    """(kind, evidence) of one 'other' map."""
    f = value_flags(r)
    kind = by_name(text(r["name"], r["file_name"]), f)
    if kind:
        return kind, "image or file name"
    kind = by_name(text(r["description"]), f)
    if kind:
        return kind, "image description"
    if re.sub(r"\s+mean\s*$", "", str(r["name"]).strip().lower()) in sib_stems:
        return "contrast estimate", "named like a declared T/Z sibling"
    m = re.search(r" (z ?stat|z score|t ?stat|eigenbrain)\w* ", text(r["collection_name"], r["collection_description"]))
    if m:
        kind = {"zstat": "Z map", "zscore": "Z map", "tstat": "T map", "eigenbrain": "component or gradient"}[m.group(1).replace(" ", "")]
        return kind, "collection description"
    decl = {k for k in sib_types if k in ("T map", "Z map")}
    if f["tz_like"] and len(decl) == 1:
        return decl.pop(), "declared siblings and values"
    if f["tz_like"]:
        return "t or z (values)", "values"
    return "unresolved", "none"


def other_map_kinds(maps: pd.DataFrame, scrape: pd.DataFrame, collections: pd.DataFrame) -> pd.DataFrame:
    """Kind, evidence and evaluation map type of every row of ``maps`` (id, collection_id, value quantiles) with
    map_type 'other'; ``scrape`` / ``collections`` are the pinned neurovault_images / neurovault_collections tables."""
    img = scrape.assign(collection_id=scrape["collection_id"].astype(int), file_name=scrape["file"].astype(str).str.split("/").str[-1])
    declared = img[img["map_type"] != "other"]
    sib_types = declared.groupby("collection_id")["map_type"].agg(set).to_dict()
    stems = declared["name"].astype(str).str.strip().str.lower().str.replace(STAT_SUFFIX, "", regex=True)
    sib_stems = stems.groupby(declared["collection_id"]).agg(set).to_dict()
    col = collections.set_index("id")
    df = maps[maps["map_type"] == "other"].drop(columns=["map_type"]).merge(
        img[["id", "name", "file_name", "description", "modality"]], on="id", how="left")
    df["collection_id"] = df["collection_id"].astype(int)
    df["collection_name"] = df["collection_id"].map(col["name"])
    df["collection_description"] = df["collection_id"].map(col["description"])
    res = [infer(r, sib_types.get(r["collection_id"], set()), sib_stems.get(r["collection_id"], set())) for _, r in df.iterrows()]
    df["kind"], df["evidence"] = [k for k, _ in res], [e for _, e in res]
    df["map_type_eval"] = df["kind"].map(GROUPS)
    return df.drop(columns=["description", "collection_description"])


def name_rule_check(maps: pd.DataFrame, scrape: pd.DataFrame) -> pd.DataFrame:
    """Declared type vs the kind the image/file-name rules give, over the declared maps of ``maps`` (rule hit rate)."""
    img = scrape.assign(file_name=scrape["file"].astype(str).str.split("/").str[-1])
    df = maps[maps["map_type"] != "other"].merge(img[["id", "name", "file_name", "modality"]], on="id", how="left")
    df["kind"] = [by_name(text(r["name"], r["file_name"]), value_flags(r)) for _, r in df.iterrows()]
    return df.groupby(["map_type", "kind"], dropna=False).size().rename("n_images").reset_index()
