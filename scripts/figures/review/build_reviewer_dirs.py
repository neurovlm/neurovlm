#!/usr/bin/env python
"""Rebuild reviewer_01/ .. reviewer_03/: per comment of RESPONSE.md, its response (README.md) and its figure / table files.

    .venv/bin/python scripts/figures/review/build_reviewer_dirs.py   (core/run_all.sh runs it last; CPU, a few s)

A figure source brings its further pages (<source>_p2, _p3, ...).  For response_letter.tex (\\rfigparts) every figure the
letter shows (an image line of RESPONSE.md) with several pages, or one page more than twice as tall as wide, also gets
<name>_parts.tex, which lists its pages at the letter's full text width; a page that tall is first cut at white rows
into page-high parts, <name>_part<i>.png.  The other figures are copied for the reviewer folders only (e.g. the five
failure-case sets of 2.12 that the letter does not show).  Figure captions are taken from response_letter.tex, their
only source; links between comments ([2.4](#2.4)) point into the right reviewer's README.
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from check_letter_sync import braced, norm, rfigs  # noqa: E402
CORE = "core"
MAIN = "../main"

# comment id -> [(short name, source relative to this folder)]; a source without suffix is a figure (.png + .svg),
# copied as <comment>_<short>.<ext>
FILES: dict[str, list[tuple[str, str]]] = {
    "1.1": [("text_to_brain_baselines", f"{CORE}/a1_text_to_brain_baselines/outputs/text_to_brain_baselines"),
            ("text_to_brain_baselines_table", f"{CORE}/a1_text_to_brain_baselines/outputs/text_to_brain_baselines_table.tex"),
            ("retrieval_table", f"{CORE}/a1_retrieval_metrics_table/outputs/retrieval_metrics_table.tex"),
            ("brain_to_text_baselines_table", f"{CORE}/a1_brain_to_text_baselines/outputs/brain_to_text_baselines_table.md")],
    "1.2": [("dataflow", f"{CORE}/a4_dataflow_diagram/outputs/a4_dataflow"),
            ("stage_splits", f"{CORE}/a4_counts_table/outputs/stage_splits.md")],
    "1.3": [("neurovault_bounds", f"{CORE}/a3_neurovault_bounds/outputs/neurovault_bounds"),
            ("bounds_table", f"{CORE}/a3_neurovault_bounds/outputs/bounds_table.tex"),
            ("map_type_table", f"{CORE}/a3_neurovault_bounds/outputs/map_type_table.tex")],
    "1.4": [("ood", f"{CORE}/a7_ood_confidence/outputs/a7_ood"),
            ("ood_table", f"{CORE}/a7_ood_confidence/outputs/a7_ood_table.tex")],
    "1.8": [("edge_table", f"{CORE}/a8_edge_benchmark/outputs/a8_summary_table.md")],
    "1.9": [("fig2_network_examples", f"{MAIN}/fig2_network_examples"),
             ("fig3_newgen", f"{MAIN}/fig3_newgen"),
             ("figS4_clip_ica", f"{MAIN}/figS4_clip_ica")],
    "2.1": [("mlp_projector", f"{CORE}/a5_mlp_projector/outputs/a5_mlp_projector"),
            ("mlp_projector_table", f"{CORE}/a5_mlp_projector/outputs/a5_mlp_projector_table.tex")],
    "2.2": [("dataflow", f"{CORE}/a4_dataflow_diagram/outputs/a4_dataflow"),
            ("stage_splits", f"{CORE}/a4_counts_table/outputs/stage_splits.md")],
    "2.3": [("neurovault_bounds", f"{CORE}/a3_neurovault_bounds/outputs/neurovault_bounds"),
            ("bounds_table", f"{CORE}/a3_neurovault_bounds/outputs/bounds_table.tex")],
    "2.4": [("leakage_table", f"{CORE}/a4_leakage_audit/outputs/leakage_table.md")],
    "2.5": [("text_to_brain_baselines", f"{CORE}/a1_text_to_brain_baselines/outputs/text_to_brain_baselines"),
            ("text_to_brain_baselines_table", f"{CORE}/a1_text_to_brain_baselines/outputs/text_to_brain_baselines_table.tex")],
    "2.8": [("fig5_perf_quant", f"{MAIN}/fig5_perf_quant"),
             ("neurovault_bounds", f"{CORE}/a3_neurovault_bounds/outputs/neurovault_bounds")],
    "2.12": [(name, f"{CORE}/a7_failure_cases_percentiles/outputs/a7_{name}")  # letter: pubmed_b2t_worst only
             for name in ("pubmed_b2t_worst", "pubmed_t2b_worst", "pubmed_random", "networks_bottom",
                          "neurovault_b2t_best", "neurovault_t2b_best")],
    "3.2": [("text_to_brain_baselines", f"{CORE}/a1_text_to_brain_baselines/outputs/text_to_brain_baselines"),
            ("figS3_recall_comparison", f"{MAIN}/figS3_recall_comparison")],
    "3.3": [("granularity_tradeoff", f"{CORE}/a2_granularity_tradeoff/outputs/a2_granularity_tradeoff"),
            ("brain_to_text_baselines_table", f"{CORE}/a1_brain_to_text_baselines/outputs/brain_to_text_baselines_table.md")],
    "3.4": [("encoder_swap", f"{CORE}/a5_encoder_swap/outputs/a5_encoder_swap"),
            ("encoder_swap_table", f"{CORE}/a5_encoder_swap/outputs/a5_encoder_swap_table.tex")],
    "3.5": [("latent_interpretation", f"{CORE}/a6_latent_interpretation/outputs/latent_interpretation"),
            ("perturbation_probes", f"{CORE}/a6_perturbation_probes/outputs/perturbation_probes"),
            ("loss_ablation_contrastive", f"{CORE}/a5_loss_ablation/outputs/a5_loss_ablation_contrastive.tex"),
            ("loss_ablation_text_to_brain", f"{CORE}/a5_loss_ablation/outputs/a5_loss_ablation_text_to_brain.tex"),
            ("negation_context_table", f"{CORE}/a6_perturbation_probes/outputs/table_negation_context.md")],
    "3.6": [("steering_correctness", f"{CORE}/a6_steering_tokens/outputs/steering_correctness"),
            ("steering_correctness_table", f"{CORE}/a6_steering_tokens/outputs/correctness_table.tex"),
            ("modality_restricted_table", f"{CORE}/a6_modality_restricted_t2b/outputs/modality_table.md")],
    "3.8": [("caption_usability", f"{CORE}/a4_caption_audit/outputs/caption_usability"),
            ("caption_examples", f"{CORE}/a4_caption_audit/outputs/caption_examples.md")],
    "3.9": [("example_percentiles", f"{CORE}/a7_failure_cases_percentiles/outputs/a7_example_percentiles.md"),
            ("captions", f"{CORE}/a7_failure_cases_percentiles/outputs/captions.tex")],
}


def sections(text: str) -> list[tuple[str, str, list[str]]]:
    """(comment id, title, body lines) for every ``### <id>: <title>`` section of RESPONSE.md (a ``## `` heading, such as
    ``## Reviewer 2``, ends the section before it)."""
    out, cur = [], None
    for line in text.splitlines():
        if re.fullmatch(r'<a id="[^"]+"></a>', line.strip()):
            continue  # anchor of the next heading
        if line.startswith("### "):
            cid, _, title = line[4:].partition(" ")
            cur = (cid.rstrip(":"), title, [])
            out.append(cur)
        elif line.startswith("## "):
            cur = None
        elif cur is not None:
            cur[2].append(line)
    return out


def letter_images(png: Path, max_ratio: float = 2.0, part_ratio: float = 1.15) -> list[Path]:
    """The image itself, or, if it is more than ``max_ratio`` x as tall as wide, its parts cut at white rows (each at most
    ``part_ratio`` x the width tall), saved next to it as <stem>_part<i>.png."""
    im = Image.open(png)
    w, h = im.size
    if h <= max_ratio * w:
        return [png]
    white = (np.asarray(im.convert("L")) > 245).all(axis=1)
    cuts, max_h = [0], int(part_ratio * w)
    while h - cuts[-1] > max_h:
        lo, hi = cuts[-1] + max_h // 2, cuts[-1] + max_h
        rows = np.flatnonzero(white[lo:hi]) + lo
        if len(rows):  # middle of the last white band in range
            end = start = int(rows[-1])
            while start - 1 >= lo and white[start - 1]:
                start -= 1
            cuts.append((start + end) // 2)
        else:
            cuts.append(hi)
    cuts.append(h)
    stem, parts = png.name[:-len(".png")], []
    for k, (a, b) in enumerate(zip(cuts, cuts[1:]), 1):
        parts.append(png.parent / f"{stem}_part{k}.png")
        im.crop((0, a, w, b)).save(parts[-1])
    return parts


def write_parts(pages: list[Path]) -> None:
    """<first page>_parts.tex for response_letter.tex (\\rfigparts): every page of a figure, tall pages cut into parts."""
    images = [im for page in pages for im in letter_images(page)]
    if len(images) < 2:
        return
    first = pages[0]
    lines = [f"% {first.name}: {len(pages)} page(s) shown as {len(images)} letter pages; written by build_reviewer_dirs.py"]
    lines += [f"\\rfigpart{{{first.parent.name}/{im.name}}}{{{k}}}{{{len(images)}}}" for k, im in enumerate(images, 1)]
    (first.parent / f"{first.name[:-len('.png')]}_parts.tex").write_text("\n".join(lines) + "\n")
    print(f"[letter parts] {first.parent.name}/{first.name}: {len(images)}")


def copy_files(cid: str, dest: Path, shown: set[str]) -> list[tuple[str, str]]:
    """Copy the files of one comment into ``dest``; returns (file name, source) pairs.  A figure source also brings its
    further pages (<source>_p2, _p3, ...); a figure in ``shown`` (file names the letter shows) gets its _parts.tex."""
    done: list[tuple[str, str]] = []
    for short, src in FILES.get(cid, []):
        src_path = HERE / src
        if src_path.suffix:
            pages = [(short, src_path, src)]
        else:
            pages, n = [(short, src_path, f"{src}.{{png,svg}}")], 2
            while (p := src_path.parent / f"{src_path.name}_p{n}").with_name(p.name + ".png").exists():
                pages.append((f"{short}_p{n}", p, f"{src}_p{n}.{{png,svg}}"))
                n += 1
        pngs = []
        for name_short, path, label in pages:
            sources = [path] if path.suffix and src_path.suffix else [path.with_name(path.name + ext) for ext in (".png", ".svg")]
            for s in sources:
                name = f"{cid}_{name_short}{s.suffix}"
                if any(n == name for n, _ in done):
                    continue
                if s.exists():
                    shutil.copy2(s, dest / name)
                    done.append((name, label))
                    if s.suffix == ".png":
                        pngs.append(dest / name)
                else:
                    print(f"[missing] {cid}: {s}")
        if pngs and pngs[0].name in shown:
            write_parts(pngs)
    return done


MATH = {r"\\le": "≤", r"\\leq": "≤", r"\\ge": "≥", r"\\geq": "≥", r"\\times": "×", r"\\tau": "τ", r"\\approx": "≈",
        r"\\pm": "±", r"\\rightarrow": "→", r"\\alpha": "α", r"\\textdegree": "°"}


def letter_captions() -> dict[str, tuple[int, str]]:
    """Figure file name -> (number n of Fig. Rn, caption as plain text) of every figure in response_letter.tex."""
    full = (HERE / "response_letter.tex").read_text()
    macros = {}  # \newcommand{\capfoo}{...}: the caption of a figure shown more than once
    for m in re.finditer(r"\\newcommand\{\\(cap[A-Za-z]+)\}(?=\{)", full):
        macros[m.group(1)], _ = braced(full, m.end())
    tex = full[full.index("\\begin{document}"):]
    figs = rfigs(tex)
    labels = {f["label"]: f["n"] for f in figs}
    out = {}
    for f in figs:
        cap = f["caption"]
        for name, body in macros.items():
            cap = re.sub(r"\\%s(\{\})?(?![A-Za-z])" % name, lambda _: body, cap)
        cap = norm(cap, labels)
        for k, v in MATH.items():
            cap = re.sub(k + r"(?![A-Za-z])", v, cap)
        cap = re.sub(r"\\[A-Za-z]+\s*", "", cap).replace("{", "").replace("}", "")
        out.setdefault(Path(f["path"]).name, (f["n"], " ".join(cap.split())))
    return out


def link(line: str, rev: str) -> str:
    """[2.4](#2.4) stays in this reviewer's README if comment 2.4 is in it, else points to the other reviewer's README."""
    return re.sub(r"\]\(#([^)]+)\)", lambda m: f"](#{m.group(1)})" if m.group(1).split(".")[0] == rev
                  else f"](../reviewer_0{m.group(1).split('.')[0]}/README.md#{m.group(1)})", line)


def main() -> int:
    response = (HERE / "RESPONSE.md").read_text()
    captions = letter_captions()
    by_reviewer: dict[str, list] = {}
    for cid, title, body in sections(response):
        by_reviewer.setdefault(cid.split(".")[0], []).append((cid, title, body))
    for rev, items in sorted(by_reviewer.items()):
        dest = HERE / f"reviewer_0{rev}"
        dest.mkdir(exist_ok=True)
        for old in dest.iterdir():  # rebuilt from scratch every time
            if old.is_file():
                old.unlink()
        lines = [f"# Reviewer {rev}", "",
                 "Each comment, our response (copied from [`../RESPONSE.md`](../RESPONSE.md)) and the figure / table files used, "
                 "copied here and named by comment.  Rebuilt by `../build_reviewer_dirs.py`; edit RESPONSE.md, not this file.", ""]
        for cid, title, body in items:
            images = [re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line.strip()) for line in body if line.startswith("![")]
            files = copy_files(cid, dest, {Path(m.group(2)).name for m in images if m})
            lines += [f'<a id="{cid}"></a>', f"## {cid}: {title}", ""]
            responded = False
            for line in body:
                line = line.rstrip()
                if line.startswith("<!--") or line.startswith("!["):
                    continue  # hidden script notes and the RESPONSE image link (shown below from this folder)
                if line.endswith(" \\"):
                    line = line[:-2].rstrip()
                if line.startswith("**Comment:**"):
                    lines += [line, ""]
                    continue
                if line and not responded and not line.startswith("|"):
                    line, responded = "**Response:** " + line, True
                lines.append(link(line, rev))
            while lines and not lines[-1]:
                lines.pop()
            figs = [n for n, _ in files if n.endswith(".png")]
            shown = [(m.group(1), Path(m.group(2)).name) for m in images if m and Path(m.group(2)).name in figs]
            if figs and not shown:
                shown = [("", figs[0])]
            for alt, name in shown:
                n, cap = captions.get(name, (None, ""))
                lines += ["", f"![{alt}]({name})"]
                if alt and cap:
                    lines.append(f"*{alt}.* {cap}")
                elif alt:
                    lines.append(f"*{alt} of the response letter.*")
            if files:
                lines += ["", "Files: " + ", ".join(f"[`{n}`]({n})" for n, _ in files) + ".",
                          "Source: " + ", ".join(f"`{s}`" for s in dict.fromkeys(s for _, s in files)) + "."]
            lines.append("")
        (dest / "README.md").write_text("\n".join(lines))
        print(f"reviewer_0{rev}: {len(items)} comments, {sum(1 for p in dest.iterdir() if p.name != 'README.md')} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
