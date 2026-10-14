#!/usr/bin/env python
"""Failure-case pages (2.m6) and example percentiles (3.8) from the outputs of run_analysis.py.

Writes captions.tex, a7_example_percentiles.{md,csv} and the pages of each failure set (a7_<set>.{png,svg}, then
a7_<set>_p2, ...): letter pages (6.5 in wide, at most FIG_LETTER_PAGE_H tall), two cases per page, every text complete.
A case: its rank and name; on the left its maps, each as three brain views stacked (0.9 in tall each), with the scores
below them; on the right its texts in two narrow columns (paper text and generated text side by side, a single text
flowed over both; a bold title runs into its text).  A set whose stacked layout needs more pages or more 7.5 pt pages puts
the maps in a row on top (three views side by side; name and scores beside them) and the texts below; one layout per set
(FAMILIES).  Body text 8 pt at line spacing 1.0; a page whose two cases do not fit at 8 pt is set at 7.5 pt (printed per
page).  CPU, ~4 min (the layout search and the brain views).
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _review_common import apply_style, cut_coords_for, draw_brain, flat_to_nifti  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # scripts/
from common import FIG_LETTER_PAGE_H, FIG_LETTER_W  # noqa: E402

INK, INK2, RULE = "#0b0b0b", "#52514e", "#c3c2b7"
METRICS = {"Fig 3": ("image_cos_cross", "text_cos_cross"), "Fig 2": ("t2b_cos_same", "b2t_cos_network"),
           "Fig S4": ("top1_label_cos", "b2t_cos")}
PAGES = {
    "pubmed_b2t_worst": ("PubMed test set: the 6 worst brain-to-text cases",
                         "The 6 test maps whose generated description is least similar to the paper's own text (title + summary; cosine in "
                         "the shared text-brain space), out of all PubMed test maps."),
    "pubmed_t2b_worst": ("PubMed test set: the 6 worst text-to-brain cases",
                         "The 6 test papers whose generated map (from title + abstract) correlates least with the paper's true map "
                         "(voxel Pearson r), out of all PubMed test papers."),
    "pubmed_random": ("PubMed test set: 6 random cases, for comparison",
                      "Six randomly drawn test papers (fixed seed), both directions: brain to text, cosine of the generated description "
                      "with the paper's text (title + summary) in the shared text-brain space; text to brain, voxel Pearson r of the "
                      "generated map with the true map. Percentiles among all PubMed test papers."),
    "networks_bottom": ("Network atlases: the 6 worst networks",
                        "The 6 atlas networks whose generated network description is least similar to the network map (cosine in the "
                        "shared text-brain space), out of all atlas networks."),
    "neurovault_b2t_best": ("NeuroVault: best-case brain-to-text failures",
                            "Each paper has several maps; every map gets its own generated text and the paper keeps its best map (highest "
                            "cosine of the text with the map). Shown: the 6 papers whose best map is still lowest."),
    "neurovault_t2b_best": ("NeuroVault: best-case text-to-brain failures",
                            "The map generated from the paper text is compared with every map of the paper; the paper keeps its best map "
                            "(highest cosine). Shown: the 6 papers whose best map is still lowest."),
}
SCORES = {  # run_analysis.py score labels -> (direction, measure) on the pages; each set's description defines the score
    "brain-to-text: caption vs true text, contrastive cosine": ("brain to text", "cosine"),
    "text-to-brain: voxel Pearson r, generated vs true map": ("text to brain", "r"),
    "brain-to-text: [NETWORK] text vs map, contrastive cosine": ("brain to text", "cosine"),
    "brain-to-text: best map's generated text vs that map, contrastive cosine": ("brain to text", "cosine"),
    "text-to-brain: generated map vs best true map, contrastive cosine": ("text to brain", "cosine"),
}
PLAIN = {  # run_analysis.py labels -> plain words on the pages
    "brain-to-text: caption vs true text, contrastive cosine": "brain to text, generated description vs the paper's text (cosine)",
    "text-to-brain: voxel Pearson r, generated vs true map": "text to brain, generated map vs true map (voxel Pearson r)",
    "brain-to-text: [NETWORK] text vs map, contrastive cosine": "brain to text, generated network description vs the map (cosine)",
    "brain-to-text: best map's generated text vs that map, contrastive cosine": "brain to text, best map's generated text vs that map (cosine)",
    "text-to-brain: generated map vs best true map, contrastive cosine": "text to brain, generated map vs the paper's best map (cosine)",
    "Generated caption (stage 2, complete)": "Generated description (complete)",
    "Generated [NETWORK] title + text (complete)": "Generated network description (complete)",
    "Reference (title + summary)": "Paper text (title + summary)",
    "Reference label (network_test_set_labels)": "Network name (atlas label)",
    "generated map (text-to-brain head)": "generated map (from the text)",
}

# page layout (inches): pages as wide as the response letter's text and at most as tall as a letter figure page
# (\rfigpart), two cases per page; text wrapped to the measured column widths
W, H_MAX = FIG_LETTER_W, FIG_LETTER_PAGE_H
PER_PAGE = 2
FONT_SIZES = (8.0, 7.5)  # body text, per page: 7.5 pt only on a page whose two cases do not fit at 8 pt
# layout families, in order of preference: maps as stacked views left of the texts, or in a row above them; the case's
# name across the page or in the maps' column; its scores under the name or below / beside the maps.  One family per set.
FAMILIES = (("stack", False, True), ("stack", False, False), ("stack", True, True),
            ("top", True, True), ("top", False, True), ("top", False, False))
SPACING = 1.0  # condensed line spacing: line height = font size
M, GAP, SUBGAP = 0.03, 0.16, 0.08  # page margin, gap between columns, gap between the view stacks of two maps
VIEW_H, VIEW_W = 0.9, 1.0  # one brain view (sagittal, coronal or axial): axes height; width of a stack of three
ORTHO_W = 2.52  # the three views of a map side by side, 0.9 in tall (as the previous pages' narrowest map)
CASE_GAP = 0.14  # between the two cases of a page (a rule in the middle)
FLOW_MIN = 3.6  # a single text wider than this is flowed over two columns (no ~120-character lines)
FS = FONT_SIZES[0]  # current body size (set_font); titles FS + 1


def set_font(size: float) -> None:
    global FS
    FS = size


def lh(size: float | None = None) -> float:
    """Line height in inches."""
    return (size or FS) * SPACING / 72


def pct_str(p: float) -> str:
    return f"{p:.2f}" if p < 1 else (f"{p:.1f}" if p < 10 else f"{p:.0f}")


_MEASURE = {}


def _renderer():
    import matplotlib.pyplot as plt

    if "r" not in _MEASURE:
        _MEASURE["r"] = plt.figure(dpi=300).canvas.get_renderer()
    return _MEASURE["r"]


def text_w(text: str, size: float, bold: bool = False) -> float:
    """Width in inches of one line of text, as drawn at 300 dpi."""
    from matplotlib.font_manager import FontProperties

    if (text, size, bold) not in _MEASURE:
        w, _, _ = _renderer().get_text_width_height_descent(text, FontProperties(size=size, weight="bold" if bold else "normal"),
                                                           ismath=False)
        _MEASURE[text, size, bold] = w / 300
    return _MEASURE[text, size, bold]


@functools.cache
def ascent(size: float) -> float:
    """Baseline below the top of a line (inches), as matplotlib places a line of text with va="top"."""
    from matplotlib.font_manager import FontProperties

    _, h, d = _renderer().get_text_width_height_descent("lp", FontProperties(size=size), ismath=False)
    return (h - d) / 300


def space_w(size: float, bold: bool = False) -> float:
    return text_w("a a", size, bold) - text_w("aa", size, bold)


def line_w(line: tuple[str, str], size: float) -> float:
    """Width of a line (bold part, regular part); the regular part follows the bold one after a space."""
    b, r = line
    return (text_w(b, size, True) if b else 0.0) + (space_w(size) if b and r else 0.0) + (text_w(r, size) if r else 0.0)


def _join(words: list[tuple[str, bool]]) -> tuple[str, str]:
    return " ".join(w for w, b in words if b), " ".join(w for w, b in words if not b)


@functools.cache
def wrap(tokens: tuple[tuple[str, bool], ...], width: float, size: float) -> tuple[tuple[str, str], ...]:
    """Greedy word wrap of one paragraph of (word, bold) tokens (bold words first: an inline title) to ``width`` inches;
    lines are (bold part, regular part).  Words are placed by their summed widths, then every line is measured whole; if
    one is too wide (kerning), the paragraph is wrapped again measuring every candidate line."""
    lines, cur, cw = [], [], 0.0
    for word, bold in tokens:
        ww = text_w(word, size, bold)
        add = ww if not cur else cw + space_w(size, bold and cur[-1][1]) + ww
        if cur and add > width:
            lines.append(cur)
            cur, cw = [(word, bold)], ww
        else:
            cur, cw = cur + [(word, bold)], add
    lines += [cur] if cur else []
    if all(line_w(_join(ln), size) <= width or len(ln) == 1 for ln in lines):
        return tuple(_join(ln) for ln in lines)
    lines, cur = [], []
    for tok in tokens:
        if cur and line_w(_join(cur + [tok]), size) > width:
            lines.append(cur)
            cur = [tok]
        else:
            cur = cur + [tok]
    return tuple(_join(ln) for ln in lines + ([cur] if cur else []))


def wrap_text(text: str, width: float, size: float, bold: bool = False) -> tuple[tuple[str, str], ...]:
    return wrap(tuple((w, bold) for w in str(text).split()), width, size)


def markdown(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def figure_order(sidecar_dir: Path) -> dict[str, list[str]]:
    """Panel order of the Fig. 2 and S4 examples (the percentile table lists them in atlas order)."""
    s2 = json.loads((sidecar_dir / "fig2_network_examples_text.json").read_text())
    s4 = json.loads((sidecar_dir / "figS4_clip_ica_text.json").read_text())
    return {"Fig 2": [f"{r['atlas']}:{r['map']}" for r in s2["rows"]],
            "Fig S4": [f"{c}:{r['component']} (" for c in ("UKBICA", "HCPICA") for r in s4["cohorts"][c]]}


def panel_rank(example: str, order: list[str]) -> int:
    return next((k for k, o in enumerate(order) if example == o or example.startswith(o)), len(order))


def percentile_list(pt: pd.DataFrame, figure: str, metric: str, order: dict) -> tuple[str, int]:
    rows = pt[(pt["figure"] == figure) & (pt["metric"] == metric)]
    if order.get(figure):
        rows = rows.assign(_o=rows["example"].map(lambda ex: panel_rank(ex, order[figure]))).sort_values("_o", kind="stable")
    return ", ".join(f"{p:.0f}" for p in rows["percentile"]), int(rows["n_distribution"].max())


def captions_tex(pt: pd.DataFrame, order: dict) -> str:
    t3, n3 = percentile_list(pt, "Fig 3", "image_cos_cross", order)
    b3, _ = percentile_list(pt, "Fig 3", "text_cos_cross", order)
    t2, n2t = percentile_list(pt, "Fig 2", "t2b_cos_same", order)
    b2, n2b = percentile_list(pt, "Fig 2", "b2t_cos_network", order)
    ts, ns = percentile_list(pt, "Fig S4", "top1_label_cos", order)
    bs, _ = percentile_list(pt, "Fig S4", "b2t_cos", order)
    return "\n".join([
        "% written by a7_failure_cases_percentiles/make_figure.py from outputs/example_percentiles.csv",
        "% final model (scripts/outputs/runs/final) and final paper figures (scripts/figures/main)",
        "% Fig. 3 caption addition (percentiles within the figure's own pool, scripts/figures/main/fig3_newgen_text.json)",
        f"Rows (top to bottom) lie at the {t3} percentiles of the text-to-brain cosine and at the {b3} percentiles of the "
        f"brain-to-text cosine across all {n3} NeuroVault papers.", "",
        "% Fig. 2 caption addition",
        f"Relative to the {n2t} atlas networks with a reference description, the generated maps (rows top to bottom) rank at the "
        f"{t2} percentiles; relative to all {n2b} atlas networks, the generated network descriptions rank at the {b2} percentiles.", "",
        "% Fig. S4 caption addition",
        f"Shown components (UK Biobank, then HCP, in panel order) rank at the {ts} percentiles of top-1 label similarity and the "
        f"{bs} percentiles of generated-text similarity among all {ns} UKB and HCP ICA components.", ""])


def percentile_table(pt: pd.DataFrame, out: Path, order: dict) -> None:
    rows = []
    for figure, (m1, m2) in METRICS.items():
        a = pt[(pt["figure"] == figure) & (pt["metric"] == m1)].set_index("example")
        b = pt[(pt["figure"] == figure) & (pt["metric"] == m2)].set_index("example")
        exs = sorted(a.index, key=lambda ex: panel_rank(ex, order[figure])) if order.get(figure) else a.index
        for ex in exs:
            rows.append({"Figure": figure, "Example": ex.strip()[:60],
                         "Text → brain (S4: label)": f"P{a.loc[ex, 'percentile']:.0f} of {int(a.loc[ex, 'n_distribution'])}",
                         "Brain → text": f"P{b.loc[ex, 'percentile']:.0f} of {int(b.loc[ex, 'n_distribution'])}"})
    df = pd.DataFrame(rows)
    df.to_csv(out / "a7_example_percentiles.csv", index=False)
    (out / "a7_example_percentiles.md").write_text(markdown(df))




@functools.cache
def token_paragraphs(text: str) -> tuple[tuple[tuple[str, bool], ...], ...]:
    """Paragraphs as (word, bold) tokens.  Bold: the [NETWORK] / [REGION] / [FUNCTION] title of a caption and a short first
    paragraph (paper title, network name) followed by more text; a bold title runs into the paragraph after it (as the
    paper's Fig. S4 prints its generated texts)."""
    paras = [p for p in str(text).split("\n") if p.strip()]
    bold = [p.startswith("[") or (k == 0 and len(paras) > 1 and len(p) <= 160) for k, p in enumerate(paras)]
    out, k = [], 0
    while k < len(paras):
        toks = [(w, bold[k]) for w in paras[k].split()]
        if bold[k] and k + 1 < len(paras) and not bold[k + 1]:
            k += 1
            toks += [(w, False) for w in paras[k].split()]
        out.append(tuple(toks))
        k += 1
    return tuple(out)


# A column is a list of rows drawn top down: ("lines", lines, kind) with lines (bold part, regular part) and kind head /
# label / para / score, ("gap", inches), or ("views", map key, "stack" | "ortho").
STYLE = {"head": dict(color=INK), "label": dict(color=INK2, fontstyle="italic"), "para": dict(color=INK),
         "score": dict(color=INK2)}


def kind_size(kind: str) -> float:
    return FS + 1 if kind == "head" else FS


def row_h(row: tuple) -> float:
    if row[0] == "lines":
        return len(row[1]) * lh(kind_size(row[2]))
    if row[0] == "gap":
        return row[1]
    return 3 * VIEW_H if row[2] == "stack" else VIEW_H


def col_h(rows: list[tuple]) -> float:
    return sum(row_h(r) for r in rows)


def fits(rows: list[tuple], width: float) -> bool:
    """Every line within ``width`` (only an unbreakable word, such as a DOI, can exceed it)."""
    return all(line_w(ln, kind_size(r[2])) <= width + 1e-6 for r in rows if r[0] == "lines" for ln in r[1])


def label_rows(label: str, width: float) -> list[tuple]:
    return [("lines", wrap_text(label, width, FS), "label"), ("gap", 0.02)]


def text_lines(text: str, width: float) -> tuple[tuple[str, str], ...]:
    return tuple(ln for toks in token_paragraphs(text) for ln in wrap(toks, width, FS))


def text_rows(label: str, text: str, width: float) -> list[tuple]:
    return label_rows(label, width) + [("lines", text_lines(text, width), "para")]


def flowed_rows(label: str, text: str, width: float) -> tuple[list[tuple], list[tuple]]:
    """One text over two columns of ``width``: its label above the first, its lines split so that both columns end level."""
    lines = text_lines(text, width)
    k = (len(lines) + 1) // 2
    while 1 < k < len(lines) and lines[k - 1][0] and not lines[k - 1][1]:  # a title line stays with the text below it
        k -= 1
    head = label_rows(label, width)
    return head + [("lines", lines[:k], "para")], [("gap", col_h(head)), ("lines", lines[k:], "para")]


def score_rows(c: dict, width: float) -> list[tuple]:
    """Each score with its percentile (the set's description says what is compared)."""
    rows = []
    for lab, v, pct, n in c["metrics"]:
        direction, measure = SCORES.get(lab, (PLAIN.get(lab, lab), ""))
        rows.append(("lines", wrap_text(f"{direction}: {measure} {v:.3f} (percentile {pct_str(pct)} of {n:,})", width, FS), "score"))
    return rows


def name_rows(c: dict, width: float) -> list[tuple]:
    """Rank + title or id (bold; paper titles are drawn with their text)."""
    name = c["id"] if str(c["true_text"]).startswith(c["title"]) else c["title"]
    return [("lines", wrap_text(f"{c['rank']}. {name}", width, FS + 1, True), "head"), ("gap", 0.04)]


def map_rows(label: str, key: str, mode: str, width: float, label_h: float = 0.0) -> list[tuple]:
    """A map's label and views; ``label_h``: pad the label to this height so that maps side by side line up."""
    lab = label_rows(label, width)
    return lab + [("gap", max(0.0, label_h - col_h(lab))), ("views", key, mode), ("gap", 0.04)]


def map_cols(maps: list[tuple[str, str]], mode: str, x_step: float, width: float, dy: float = 0.0) -> list[tuple]:
    """Maps side by side, x_step apart, their views level."""
    lab_h = max(col_h(label_rows(lab, width)) for lab, _ in maps)
    return [(k * x_step, dy, width, map_rows(lab, key, mode, width, lab_h)) for k, (lab, key) in enumerate(maps)]


def case_items(c: dict) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """[(label, map key)], [(label, text)]: the true text, then the generated text if any."""
    maps = [(PLAIN.get(lab, lab), key) for key, lab in c["maps"]]
    texts = [(PLAIN.get(c["true_label"], c["true_label"]), c["true_text"])]
    if c.get("generated_text"):
        texts.append((PLAIN.get(c["generated_label"], c["generated_label"]), c["generated_text"]))
    return maps, texts


def text_columns(texts: list[tuple[str, str]], x0: float, width: float, dy: float = 0.0) -> list[list[tuple]]:
    """Candidate text columns within [x0, x0 + width]: two texts side by side (their split searched); one text flowed
    over two columns (in one if the area is narrower than FLOW_MIN)."""
    if len(texts) == 2:
        out = []
        for wt in np.round(np.arange(1.5, width - GAP - 1.5 + 1e-6, 0.1), 2):
            w2 = round(width - GAP - wt, 3)
            out.append([(x0, dy, wt, text_rows(*texts[0], wt)), (x0 + wt + GAP, dy, w2, text_rows(*texts[1], w2))])
        return out
    if width < FLOW_MIN:
        return [[(x0, dy, width, text_rows(*texts[0], width))]]
    wc = round((width - GAP) / 2, 3)
    first, second = flowed_rows(*texts[0], wc)
    return [[(x0, dy, wc, first), (x0 + wc + GAP, dy, wc, second)]]


def case_layouts(c: dict, fam: tuple[str, bool, bool]) -> list[tuple[list[tuple], list[tuple]]]:
    """Candidate layouts of one case in family ``fam`` = (maps "stack" | "top", name in the maps' column, scores in the
    maps' column): (head rows across the page, [(x, y offset, width, rows) per column]), offsets from the case's top left
    corner below its head."""
    maps, texts = case_items(c)
    full, n = W - 2 * M, len(maps)
    out = []
    family, name_side, scores_side = fam
    head = [] if name_side else name_rows(c, full)[:-1] + (score_rows(c, full) if not scores_side else []) + [("gap", 0.04)]
    if family == "stack":  # maps as stacks of three views on the left (name above, scores below), texts on the right
        base = n * VIEW_W + (n - 1) * SUBGAP
        for wl in np.round(base + np.array([0.0, 0.2, 0.4]), 2):
            top = name_rows(c, wl) if name_side else []
            if not fits(top, wl):
                continue
            sub = round((wl - (n - 1) * SUBGAP) / n, 3)
            left = ([(0.0, 0.0, wl, top)] if top else []) + map_cols(maps, "stack", sub + SUBGAP, sub, col_h(top))
            if scores_side:
                left.append((0.0, max(dy + col_h(rows) for _, dy, _, rows in left), wl, score_rows(c, wl)))
            out += [(head, left + right) for right in text_columns(texts, wl + GAP, round(full - wl - GAP, 3))]
    else:  # maps in a row on top (three views side by side), name and scores beside them, texts below
        row = map_cols(maps, "ortho", ORTHO_W + GAP, ORTHO_W)
        rest = round(full - n * (ORTHO_W + GAP), 3)
        beside = (name_rows(c, rest) if name_side else []) + (score_rows(c, rest) if scores_side else [])
        if beside:
            if rest < 0.9 or not fits(beside, rest):
                return out
            row.append((n * (ORTHO_W + GAP), 0.0, rest, beside))
        out += [(head, row + right) for right in text_columns(texts, 0.0, full, max(col_h(rows) for *_, rows in row))]
    return out


def case_h(layout: tuple[list[tuple], list[tuple]]) -> float:
    head, cols = layout
    return col_h(head) + max(dy + col_h(rows) for _, dy, _, rows in cols)


_BEST = {}


def best_layout(c: dict, fam: tuple) -> tuple[list[tuple], list[tuple]] | None:
    """The lowest candidate layout of a case in family ``fam`` at the current font size (None: the family cannot hold
    it); among equally low ones, the one with the shortest columns (texts balanced)."""
    if (id(c), FS, fam) not in _BEST:
        _BEST[id(c), FS, fam] = min(case_layouts(c, fam), default=None,
                                    key=lambda lay: (round(case_h(lay), 4), sum(dy + col_h(rows) for _, dy, _, rows in lay[1])))
    return _BEST[id(c), FS, fam]


def page_head(name: str, first: int, last: int, n: int, desc: bool) -> list[tuple]:
    title, text = PAGES[name]
    if n > last - first + 1:
        title += f" ({first}-{last} of {n})" if last > first else f" ({first} of {n})"
    full = W - 2 * M
    rows = [("lines", wrap_text(title, full, FS + 1, True), "head")]
    if desc:  # first page of a set
        rows.append(("lines", wrap_text(text + "  Maps: top 10% of voxels; texts complete.", full, FS), "score"))
    return rows + [("gap", 0.1)]


def page_h(name: str, group: list[dict], n: int, first: bool, fam: tuple) -> float:
    layouts = [best_layout(c, fam) for c in group]
    if any(lay is None for lay in layouts):
        return float("inf")
    head = page_head(name, group[0]["rank"], group[-1]["rank"], n, first)
    return 2 * M + col_h(head) + sum(case_h(lay) for lay in layouts) + CASE_GAP * (len(group) - 1)


def paginate_with(name: str, cases: list[dict], families: tuple[tuple, ...]) -> list[tuple[list[dict], float, tuple]]:
    """Pages in rank order as (cases, font size, family): two cases where they fit at 8 pt, else at 7.5 pt (the first
    of ``families`` that fits), else one case."""
    pages, i, n = [], 0, len(cases)
    while i < n:
        page = None
        for size in FONT_SIZES:
            set_font(size)
            fam = next((f for f in families if i + PER_PAGE <= n and page_h(name, cases[i:i + PER_PAGE], n, not pages, f) <= H_MAX), None)
            if fam:
                page = (cases[i:i + PER_PAGE], size, fam)
                break
        if page is None:
            for size in FONT_SIZES:
                set_font(size)
                hs = {f: page_h(name, cases[i:i + 1], n, not pages, f) for f in families}
                fam = next((f for f in families if hs[f] <= H_MAX), min(hs, key=hs.get))
                if hs[fam] <= H_MAX or size == FONT_SIZES[-1]:
                    page = (cases[i:i + 1], size, fam)
                    break
        pages.append(page)
        i += len(page[0])
    return pages


def paginate(name: str, cases: list[dict]) -> list[tuple[list[dict], float, tuple]]:
    """One layout family for the whole set unless mixing them (page by page) saves a page: fewest pages, then fewest
    7.5 pt pages, then the order of FAMILIES."""
    options = [paginate_with(name, cases, (f,)) for f in FAMILIES] + [paginate_with(name, cases, FAMILIES)]
    return min(options, key=lambda pg: (len(pg), sum(s != FONT_SIZES[0] for _, s, _ in pg), options.index(pg)))


def draw_views(fig, H: float, x: float, y: float, vec, mode: str, masker) -> None:
    """One map below (x, y): three views stacked (sagittal, coronal, axial) or side by side ("ortho")."""
    vec = np.asarray(vec, dtype=np.float32)
    vec = np.where(vec >= max(float(np.quantile(vec, 0.9)), 1e-3), vec, 0.0)  # top 10 % voxels
    img = flat_to_nifti(vec, masker)
    cut = cut_coords_for(img)
    if mode == "ortho":
        ax = fig.add_axes((x / W, (y - VIEW_H) / H, ORTHO_W / W, VIEW_H / H))
        draw_brain(img, ax, cut_coords=cut, threshold=1e-3, display_mode="ortho")
        return
    for k, axis in enumerate("xyz"):
        ax = fig.add_axes((x / W, (y - (k + 1) * VIEW_H) / H, VIEW_W / W, VIEW_H / H))
        draw_brain(img, ax, cut_coords=1 if cut is None else [cut[k]], threshold=1e-3, display_mode=axis)


def draw_rows(fig, H: float, x: float, y: float, rows: list[tuple], maps, masker) -> float:
    """Draw rows top down from (x, y), each text line on its baseline (bold part, then the regular part); returns the y below."""
    for row in rows:
        if row[0] == "lines":
            size = kind_size(row[2])
            st = STYLE[row[2]]
            for k, (b, r) in enumerate(row[1]):
                yb = (y - ascent(size) - k * lh(size)) / H
                if b:
                    fig.text(x / W, yb, b, va="baseline", fontsize=size, fontweight="bold", color=st["color"])
                if r:
                    xr = x + (text_w(b, size, True) + space_w(size) if b else 0.0)
                    fig.text(xr / W, yb, r, va="baseline", fontsize=size, **st)
        elif row[0] == "views":
            draw_views(fig, H, x, y, maps[row[1]], row[2], masker)
        y -= row_h(row)
    return y


def render(name: str, pages: list[tuple[list[dict], float, tuple]], maps, out: Path, masker) -> list[Path]:
    """Draw the pages of one failure set: a7_<set>.png, a7_<set>_p2.png, ... (and .svg)."""
    import matplotlib.pyplot as plt

    for old in out.glob(f"a7_{name}_p*.*"):
        old.unlink()
    n, written = sum(len(g) for g, _, _ in pages), []
    for k, (group, size, family) in enumerate(pages, 1):
        set_font(size)
        H = page_h(name, group, n, k == 1, family)
        if H > H_MAX + 1e-6:
            print(f"[warning] {name} page {k}: {H:.2f} in, taller than a letter figure page ({H_MAX:.2f} in)")
        fig = plt.figure(figsize=(W, H))
        y = draw_rows(fig, H, M, H - M, page_head(name, group[0]["rank"], group[-1]["rank"], n, k == 1), maps, masker)
        for i, c in enumerate(group):
            if i:
                y -= CASE_GAP / 2
                fig.add_artist(plt.Line2D([M / W, (W - M) / W], [y / H] * 2, color=RULE, lw=0.6))
                y -= CASE_GAP / 2
            head, cols = best_layout(c, family)
            yb = draw_rows(fig, H, M, y, head, maps, masker)
            for x, dy, _, rows in cols:
                draw_rows(fig, H, M + x, yb - dy, rows, maps, masker)
            y -= case_h((head, cols))
        stem = f"a7_{name}" + (f"_p{k}" if k > 1 else "")
        for ext in ("png", "svg"):
            fig.savefig(out / f"{stem}.{ext}", dpi=300)
            written.append(out / f"{stem}.{ext}")
        plt.close(fig)
        print(f"[figure] {out / f'{stem}.png'}  ({len(group)} cases, {W:.2f} x {H:.2f} in, {size:g} pt, {family[0]}, "
              f"name {'beside the maps' if family[1] else 'across'}, scores {'beside the maps' if family[2] else 'across'})")
    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "outputs")
    out = p.parse_args().out
    apply_style(8)
    import matplotlib.pyplot as plt
    from _hf_pins import load_masker

    plt.rcParams.update({"savefig.bbox": "standard", "svg.fonttype": "none"})  # exact page width; text stays text
    pt = pd.read_csv(out / "example_percentiles.csv")
    side = Path(json.loads((out / "metrics.json").read_text()).get("sidecar_dir") or "")
    if not (side / "fig2_network_examples_text.json").exists():  # path recorded in another checkout
        side = Path(__file__).resolve().parents[3] / "main"
    order = figure_order(side)
    (out / "captions.tex").write_text(captions_tex(pt, order))
    percentile_table(pt, out, order)
    sets = json.loads((out / "panels.json").read_text())
    maps = np.load(out / "panel_maps.npz")
    masker = load_masker()
    for name in PAGES:
        if sets.get(name):
            render(name, paginate(name, sets[name]), maps, out, masker)


if __name__ == "__main__":
    main()
