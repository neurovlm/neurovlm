#!/usr/bin/env python
"""Failure-case pages (2.m6) and example percentiles (3.8) from the outputs of run_analysis.py.

Writes captions.tex, a7_example_percentiles.{md,csv} and the pages of each failure set (a7_<set>.{png,svg}, then
a7_<set>_p2, ...): one letter page each, two cases per page where both fit with their complete texts at 8 pt, else one.
CPU, ~1 min.
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
                      "Six randomly drawn test papers (fixed seed), both directions; percentiles among all PubMed test papers."),
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
# (\rfigpart), up to two cases per page; text 8 pt, wrapped to the measured column width
W, H_MAX = FIG_LETTER_W, FIG_LETTER_PAGE_H
PER_PAGE = 2
FS, FS_HEAD = 8.0, 9.0
SPACING = 1.15
LINE = FS * SPACING / 72  # line height
M, GAP = 0.03, 0.2  # page margin, column gap
MAP_ASPECT = 2.8  # width / height of an ortho map (three slices)
MAP_W = 3.0  # widest map: slices ~1 in tall
MAP_WS = (MAP_W, 2.8, 2.6, 2.52)  # map widths tried, widest first: slices >= 0.9 in tall

def pct_str(p: float) -> str:
    return f"{p:.2f}" if p < 1 else (f"{p:.1f}" if p < 10 else f"{p:.0f}")


_MEASURE = {}


def text_w(text: str, size: float = FS, bold: bool = False) -> float:
    """Width in inches of one line of text, as drawn at 300 dpi."""
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    if (text, size, bold) in _MEASURE:
        return _MEASURE[text, size, bold]
    if "r" not in _MEASURE:
        _MEASURE["r"] = plt.figure(dpi=300).canvas.get_renderer()
    w, _, _ = _MEASURE["r"].get_text_width_height_descent(text, FontProperties(size=size, weight="bold" if bold else "normal"),
                                                         ismath=False)
    _MEASURE[text, size, bold] = w / 300
    return w / 300


@functools.cache
def wrap(text: str, width: float, size: float = FS, bold: bool = False) -> tuple[str, ...]:
    """Greedy word wrap of one paragraph to ``width`` inches."""
    lines, line = [], ""
    for word in str(text).split():
        cand = f"{line} {word}" if line else word
        if line and text_w(cand, size, bold) > width:
            lines.append(line)
            line = word
        else:
            line = cand
    return tuple(lines + [line] if line else lines)

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


def paragraphs(text: str, width: float) -> list[tuple[list[str], bool]]:
    """Wrapped paragraphs; bold: the [NETWORK] / [REGION] / [FUNCTION] title lines of a caption and a short first line
    (paper title, network name) followed by more text."""
    paras = [p for p in str(text).split("\n") if p.strip()]
    out = []
    for k, para in enumerate(paras):
        bold = para.startswith("[") or (k == 0 and len(paras) > 1 and len(para) <= 160)
        out.append((wrap(para, width, bold=bold), bold))
    return out


def text_h(paras) -> float:
    return sum(len(lines) for lines, _ in paras) * LINE


def map_h(w: float) -> float:
    return w / MAP_ASPECT


def item_h(item: tuple, width: float, map_w: float) -> float:
    """Height of a map ('map', label, key) or a text ('text', label, text) with its label above."""
    h = len(wrap(item[1], width)) * LINE + 0.02
    if item[0] == "map":
        return h + map_h(min(width, map_w)) + 0.06
    return h + text_h(paragraphs(item[2], width)) + 0.08


def case_items(c: dict) -> tuple[list[tuple], list[tuple]]:
    maps = [("map", PLAIN.get(lab, lab), key) for key, lab in c["maps"]]
    texts = [("text", PLAIN.get(c["true_label"], c["true_label"]), c["true_text"])]
    if c.get("generated_text"):
        texts.append(("text", PLAIN.get(c["generated_label"], c["generated_label"]), c["generated_text"]))
    return maps, texts


def case_layout(c: dict, map_w: float = MAP_W) -> tuple[float, list[tuple[float, float, list[tuple]]]]:
    """Lowest of the candidate layouts: (body height, [(x, y offset, width, items) per column])."""
    maps, texts = case_items(c)
    full = W - 2 * M
    cands = []
    for wl in np.round(np.arange(MAP_WS[-1], 3.85, 0.1), 2):  # map columns no narrower than the narrowest map
        wr = full - GAP - wl
        cands.append([(0, 0, wl, maps), (wl + GAP, 0, wr, texts)])  # maps left, texts right
        if len(texts) == 2:
            cands.append([(0, 0, wl, maps + texts[:1]), (wl + GAP, 0, wr, texts[1:])])  # maps + true text | generated
            cands.append([(0, 0, wl, maps + texts[1:]), (wl + GAP, 0, wr, texts[:1])])  # maps + generated | true text
            if wl <= 2.6:
                for wt in np.round(np.arange(1.8, wr - 1.7, 0.1), 2):  # maps | true text | generated
                    cands.append([(0, 0, wl, maps), (wl + GAP, 0, wt, texts[:1]), (wl + 2 * GAP + wt, 0, wr - GAP - wt, texts[1:])])
    wm = min(map_w, (full - GAP * (len(maps) - 1)) / len(maps))  # maps side by side on top, texts below
    top = item_h(maps[0], wm, map_w)
    row = [(k * (wm + GAP), 0, wm, [m]) for k, m in enumerate(maps)]
    cands.append(row + [(0, top, full, texts)])
    if len(texts) == 2:
        for wl in np.round(np.arange(2.6, 3.85, 0.1), 2):
            cands.append(row + [(0, top, wl, texts[:1]), (wl + GAP, top, full - GAP - wl, texts[1:])])
    best = None
    for cols in cands:
        h = max(dy + sum(item_h(it, w, map_w) for it in items) for _, dy, w, items in cols)
        if best is None or h < best[0] - 1e-6:
            best = (h, cols)
    return best


def case_head(c: dict) -> list[tuple[list[str], dict]]:
    """Rank + title or id (bold; paper titles are drawn with their text), then the scores with percentiles."""
    full = W - 2 * M
    name = c["id"] if str(c["true_text"]).startswith(c["title"]) else c["title"]
    head = [(wrap(f"{c['rank']}. {name}", full, FS_HEAD, True), dict(fontsize=FS_HEAD, fontweight="bold", color=INK))]
    for lab, v, pct, n in c["metrics"]:
        head.append((wrap(f"{PLAIN.get(lab, lab)}: {v:.3f} (percentile {pct_str(pct)} of {n:,})", full),
                     dict(fontsize=FS, color=INK2)))
    return head


def head_h(blocks) -> float:
    return sum(len(lines) * (st["fontsize"] * SPACING / 72) for lines, st in blocks) + 0.04


def draw_lines(fig, H, x, y, lines, **style) -> float:
    if lines:
        fig.text(x / W, y / H, "\n".join(lines), va="top", linespacing=SPACING, **style)
    return y - len(lines) * style.get("fontsize", FS) * SPACING / 72


def draw_item(fig, H, x, y, width, item, map_w, maps, masker) -> float:
    y = draw_lines(fig, H, x, y, wrap(item[1], width), fontsize=FS, color=INK2, fontstyle="italic") - 0.02
    if item[0] == "map":
        w = min(width, map_w)
        ax = fig.add_axes((x / W, (y - map_h(w)) / H, w / W, map_h(w) / H))
        vec = np.asarray(maps[item[2]], dtype=np.float32)
        vec = np.where(vec >= max(float(np.quantile(vec, 0.9)), 1e-3), vec, 0.0)  # top 10 % voxels
        img = flat_to_nifti(vec, masker)
        draw_brain(img, ax, cut_coords=cut_coords_for(img), threshold=1e-3, display_mode="ortho")
        return y - map_h(w) - 0.06
    for lines, bold in paragraphs(item[2], width):
        y = draw_lines(fig, H, x, y, lines, fontsize=FS, color=INK, fontweight="bold" if bold else "normal")
    return y - 0.08


def page_head(name: str, first: int, last: int, n: int, desc: bool) -> list[tuple[list[str], dict]]:
    title, text = PAGES[name]
    if n > last - first + 1:
        title += f" ({first}-{last} of {n})" if last > first else f" ({first} of {n})"
    full = W - 2 * M
    head = [(wrap(title, full, FS_HEAD, True), dict(fontsize=FS_HEAD, fontweight="bold", color=INK))]
    if desc:  # first page of a set
        head.append((wrap(text + "  Maps: top 10% of voxels; texts complete.", full), dict(fontsize=FS, color=INK2)))
    return head


def fit(name: str, group: list[dict], n: int, first: bool, force: bool = False) -> list[tuple] | None:
    """[(case, map width, columns, height)] at the widest map with which all cases of ``group`` fit one page; None if
    none does (``force``: the narrowest map anyway)."""
    head = head_h(page_head(name, group[0]["rank"], group[-1]["rank"], n, first))
    for map_w in MAP_WS:
        sized = []
        for c in group:
            h, cols = case_layout(c, map_w)
            sized.append((c, map_w, cols, head_h(case_head(c)) + h + 0.12))
        if 2 * M + head + sum(it[-1] for it in sized) - 0.04 <= H_MAX or (force and map_w == MAP_WS[-1]):
            return sized
    return None


def page(name: str, cases: list[dict], maps, out: Path, masker) -> list[Path]:
    """The pages of one failure set, up to PER_PAGE cases per page where they fit: a7_<set>.png, a7_<set>_p2.png, ..."""
    import matplotlib.pyplot as plt

    pages, i = [], 0
    while i < len(cases):  # greedy, in rank order: add the next case while the page still holds them all
        best = fit(name, cases[i:i + 1], len(cases), not pages, force=True)
        for j in range(i + 2, min(i + PER_PAGE, len(cases)) + 1):
            more = fit(name, cases[i:j], len(cases), not pages)
            if more is None:
                break
            best = more
        pages.append(best)
        i += len(best)
    for old in out.glob(f"a7_{name}_p*.*"):
        old.unlink()
    written = []
    for k, pg in enumerate(pages, 1):
        ph = page_head(name, pg[0][0]["rank"], pg[-1][0]["rank"], len(cases), k == 1)
        H = 2 * M + head_h(ph) + sum(it[-1] for it in pg) - 0.04  # last case: no trailing gap
        fig = plt.figure(figsize=(W, H))
        y = H - M
        for lines, st in ph:
            y = draw_lines(fig, H, M, y, lines, **st)
        y -= 0.1
        for i, (c, map_w, cols, h) in enumerate(pg):
            if i:
                fig.add_artist(plt.Line2D([M / W, (W - M) / W], [(y + 0.07) / H] * 2, color=RULE, lw=0.6))
            yb = y
            for lines, st in case_head(c):
                yb = draw_lines(fig, H, M, yb, lines, **st)
            yb -= 0.04
            for x, dy, w, items in cols:
                yc = yb - dy
                for it in items:
                    yc = draw_item(fig, H, M + x, yc, w, it, map_w, maps, masker)
            y -= h
        stem = f"a7_{name}" + (f"_p{k}" if k > 1 else "")
        for ext in ("png", "svg"):
            fig.savefig(out / f"{stem}.{ext}", dpi=300)
            written.append(out / f"{stem}.{ext}")
        plt.close(fig)
        print(f"[figure] {out / f'{stem}.png'}  ({len(pg)} cases, {W:.2f} x {H:.2f} in)")
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
            page(name, sets[name], maps, out, masker)


if __name__ == "__main__":
    main()
