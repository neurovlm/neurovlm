#!/usr/bin/env python
"""Keep the letter's manuscript references right after a paper change.

The "Changes to the manuscript" lines cite the marked manuscript (paper/sn-article-diff.pdf) by margin line
(\\msl{137}{144}), page (\\msp{39}) or page range (\\mspp{35}{38}), and \\msfig / \\mstab link to the page of a caption
through the table in response_letter_with_manuscript.tex.  Every paper edit moves these numbers.  This script rebuilds
the paper and the marked copy, reads the numbered lines and caption pages of the new marked PDF, aligns them with the
PDF the current numbers refer to (manuscript_lines.json), rewrites every reference, regenerates the caption table and
writes manuscript_links_report.md with one row per reference (old, new, status, first words of the passage).  Only the
rows marked "review" need a look.

    .venv/bin/python scripts/figures/review/manuscript_links.py --snapshot      # record the marked PDF the current numbers refer to
    .venv/bin/python scripts/figures/review/manuscript_links.py                 # rebuild paper and marked copy, remap, report
    .venv/bin/python scripts/figures/review/manuscript_links.py --build marked  # paper already built: marked copy and remap
    .venv/bin/python scripts/figures/review/manuscript_links.py --no-build      # remap against the marked PDF on disk
    .venv/bin/python scripts/figures/review/manuscript_links.py --letter ../../../../paper/response/response_letter_v2.tex
    .venv/bin/python scripts/figures/review/manuscript_links.py --changes       # also the line and page numbers of manuscript_changes.md
    .venv/bin/python scripts/figures/review/manuscript_links.py --dry-run       # report only, write nothing

Marked copy: latexdiff --flatten from the submitted version (paper/.git commit 5d3637a) to paper/sn-article.tex, then
the fixes that make it compile with sn-jnl (additions in plain blue, since pypdf reads ulem's wavy underline as text;
filecontents block and \\bibliography line removed, the current .bbl inlined,
\\DIF...FL begin/end markers dropped inside tables, the prompt boxes restored from the revised file).  Entries cited only
in the submitted version are not listed in red, unlike the hand-made 14 Oct copy.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import io
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
from datetime import date
from pathlib import Path

from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[3] / "paper"
SUBMITTED = "5d3637a"  # paper/.git: the submitted manuscript
MARKED = PAPER / "sn-article-diff"
SNAPSHOT = HERE / "manuscript_lines.json"
REPORT = HERE / "manuscript_links_report.md"
LETTER = HERE / "response_letter.tex"
COMBINED = [HERE / "response_letter_with_manuscript.tex", PAPER / "response" / "response_letter_with_manuscript.tex"]
CHANGES = PAPER.parent / "manuscript_changes.md"
CAPTION = re.compile(r"\\ms(?:fig|tab)n?\{([^{}]*)\}")
REF = re.compile(r"\\msl\{(\d+)\}\{(\d+)\}|\\msp\{(\d+)\}|\\mspp\{(\d+)\}\{(\d+)\}")


def tex_env() -> dict:
    env = dict(os.environ)
    if not shutil.which("pdflatex"):
        for p in sorted((Path.home() / "texlive" / "bin").glob("*")):
            env["PATH"] = f"{p}:{env['PATH']}"
    return env


def run(cmd: list[str], cwd: Path, ok: int = 0) -> None:
    r = subprocess.run(cmd, cwd=cwd, env=tex_env(), capture_output=True, text=True)
    if r.returncode > ok:
        tail = "\n".join((r.stdout + r.stderr).splitlines()[-25:])
        raise SystemExit(f"{' '.join(cmd)} failed in {cwd}:\n{tail}")


def pdflatex(name: str, times: int = 1) -> None:
    for _ in range(times):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", name], cwd=PAPER, env=tex_env(), capture_output=True, text=True)
        errors = [l for l in r.stdout.splitlines() if l.startswith("! ")]
        if r.returncode or errors:
            raise SystemExit(f"pdflatex {name}: {errors[:5] or r.stdout.splitlines()[-10:]}")


def build_paper() -> None:
    """Clean paper: pdflatex, bibtex, pdflatex twice (sn-article.pdf and the .bbl the marked copy inlines)."""
    pdflatex("sn-article")
    run(["bibtex", "sn-article"], PAPER, ok=1)  # 1 = warnings only
    pdflatex("sn-article", 2)


def build_marked() -> None:
    """latexdiff from the submitted commit, the fixes listed in the module docstring, three pdflatex passes."""
    with tempfile.TemporaryDirectory() as tmp:
        old = Path(tmp) / "old"
        old.mkdir()
        archive = subprocess.run(["git", "-C", str(PAPER), "archive", SUBMITTED], capture_output=True, check=True).stdout
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(old, filter="data")
        r = subprocess.run(["latexdiff", "--flatten", "--encoding=utf8", "--add-to-config", "VERBATIMLINEENV=promptbox",
                            str(old / "sn-article.tex"), "sn-article.tex"], cwd=PAPER, env=tex_env(), capture_output=True, text=True)
    if r.returncode or not r.stdout.strip():
        raise SystemExit(f"latexdiff failed:\n{r.stderr[-2000:]}")
    diff = r.stdout.replace("\\uwave{#1}", "#1").replace("\\color{blue}\\uwave]", "\\color{blue}]")  # no wavy underline
    diff = re.sub(r"\\begin\{filecontents\*?\}.*?\\end\{filecontents\*?\}\n?", "", diff, flags=re.S)
    diff = re.sub(r"^\\bibliography\{[^}]*\}\n?", "", diff, flags=re.M)
    bbl = (PAPER / "sn-article.bbl").read_text()
    if "\\begin{thebibliography}" in diff:
        diff = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", lambda m: bbl.strip(), diff, flags=re.S)
    else:
        diff = diff.replace("\\end{document}", bbl.strip() + "\n\\end{document}")

    def strip_fl(m: re.Match) -> str:  # table rows cannot hold the float begin/end markers
        return re.sub(r"\\DIF(?:add|del)(?:begin|end)FL\s*", "", m.group(0))
    diff = re.sub(r"\\begin\{(table\*?|sidewaystable)\}.*?\\end\{\1\}", strip_fl, diff, flags=re.S)
    boxes = re.findall(r"\\begin\{promptbox\}\{.*?\\end\{promptbox\}", (PAPER / "sn-article.tex").read_text(), flags=re.S)
    it = iter(boxes)
    diff = re.sub(r"\\begin\{promptbox\}\[alsolanguage=DIFcode\].*?\\end\{promptbox\}", lambda m: next(it), diff, flags=re.S)
    header = (f"% Marked manuscript: latexdiff from the submitted version (paper/.git {SUBMITTED}) to the revised "
              f"paper/sn-article.tex, built {date.today()} by scripts/figures/review/manuscript_links.py (fixes listed there).\n"
              "% Build: pdflatex sn-article-diff, three times; the bibliography is inlined.\n")
    MARKED.with_suffix(".tex").write_text(header + diff)
    pdflatex("sn-article-diff", 3)
    overleaf = PAPER / "response" / "manuscript_marked.pdf"  # the copy the Overleaf project appends to the letter
    if overleaf.exists():
        shutil.copy(MARKED.with_suffix(".pdf"), overleaf)


# --- reading the marked PDF ---------------------------------------------------------------------------------------
def extract(pdf: Path) -> dict:
    """Numbered lines {line: [page, text]} and caption pages {"Fig.S5": 39} of a marked PDF.

    The line ruler prints the number at the left on even pages and at the right on odd pages; numbers must run in
    sequence, which keeps years and table cells out.  A caption is a line that starts with its bold label.
    """
    reader = PdfReader(str(pdf))
    lines, captions, expect = {}, {}, None
    for pno, page in enumerate(reader.pages, 1):
        for raw in page.extract_text(extraction_mode="layout").split("\n"):
            s = raw.rstrip()
            if not s.strip():
                continue
            num = text = None
            m = re.match(r"^\s*(\d{3,4})(?:\s{2,}(.*))?$", s)
            if m and (expect is None and int(m.group(1)) <= 3 or expect is not None and expect <= int(m.group(1)) <= expect + 3):
                num, text = int(m.group(1)), m.group(2) or ""
            else:
                m = re.match(r"^(.*?)\s{2,}(\d{3,4})$", s)
                if m and (expect is None and int(m.group(2)) <= 3 or expect is not None and expect <= int(m.group(2)) <= expect + 3):
                    num, text = int(m.group(2)), m.group(1)
            c = re.match(r"^(Fig\.|Table)\s*(S?\d+)(?:\s{2,}|(?=[A-Z:]))", (s if num is None else text).strip())
            if c:  # a caption: bold label, then a gap or the caption text; "TableS5gives" in running text is not one
                captions.setdefault(f"{c.group(1).rstrip('.')}.{c.group(2)}", pno)
            if num is None:
                continue
            lines[num] = [pno, " ".join(text.split())]
            expect = num + 1
    rel = str(pdf.relative_to(PAPER.parent)) if pdf.resolve().is_relative_to(PAPER.parent) else str(pdf)
    return {"pdf": rel, "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
            "date": str(date.today()), "pages": len(reader.pages), "lines": lines, "captions": captions}


class Lines:
    """Numbered lines of one marked PDF with paragraph starts and ends."""

    def __init__(self, data: dict):
        self.page = {int(k): v[0] for k, v in data["lines"].items()}
        self.text = {int(k): v[1] for k, v in data["lines"].items()}
        self.nums = sorted(self.text)
        self.captions = data["captions"]
        body = [len(t) for n, t in self.text.items() if t and not self.heading(n)]
        self.full = statistics.median(body) if body else 80

    def heading(self, n: int) -> bool:
        t = self.text.get(n, "")
        return bool(re.match(r"^(S?\d+(\.\d+)*)\s+[A-Z]", t)) and len(t) < 60

    def para_end(self, n: int) -> bool:
        t = self.text.get(n, "")
        nxt = self.text.get(n + 1)
        return bool(t) and (not nxt or self.heading(n + 1) or len(t) < 0.72 * self.full)

    def para_start(self, n: int) -> bool:
        return bool(self.text.get(n)) and (not self.text.get(n - 1) or self.heading(n - 1) or self.para_end(n - 1))

    def to_para_end(self, n: int) -> int:
        while self.text.get(n + 1) and not self.para_end(n):
            n += 1
        return n

    def to_para_start(self, n: int) -> int:
        while not self.para_start(n) and self.text.get(n - 1):
            n -= 1
        return n

    def passage(self, a: int, b: int) -> str:
        return " ".join(self.text.get(n, "") for n in range(a, b + 1))


def norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", t.lower())


def align(old: Lines, new: Lines) -> dict[int, int]:
    """Old line number -> new line number: equal lines by text, the rest interpolated between their neighbours."""
    a, b = [norm(old.text[n]) for n in old.nums], [norm(new.text[n]) for n in new.nums]
    pos = {}
    for i, j, size in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        for k in range(size):
            if a[i + k]:
                pos[i + k] = j + k
    out, prev = {}, (-1, -1)
    anchors = sorted(pos.items())
    for i, n in enumerate(old.nums):
        if i in pos:
            prev = (i, pos[i])
        else:
            nxt = next(((x, y) for x, y in anchors if x > i), (len(a), len(b)))
            j = min(prev[1] + (i - prev[0]), max(nxt[1] - 1, prev[1] + 1))
            pos[i] = max(0, min(j, len(b) - 1))
        out[n] = new.nums[pos[i]]
    return out


# --- remapping ------------------------------------------------------------------------------------------------------
def status_of(ratio: float) -> str:
    return "ok" if ratio >= 0.85 else "changed" if ratio >= 0.5 else "review"


def remap_lines(a: int, b: int, old: Lines, new: Lines, amap: dict) -> tuple[int, int, str, str]:
    if a not in old.text or b not in old.text:
        return a, b, "review", "line not in the snapshot"
    a2, b2 = amap[a], max(amap[b], amap[a])
    if old.para_start(a):
        a2 = new.to_para_start(a2)
    if old.para_end(b):
        b2 = new.to_para_end(b2)
    ratio = difflib.SequenceMatcher(None, norm(old.passage(a, b)), norm(new.passage(a2, b2)), autojunk=False).ratio()
    return a2, b2, status_of(ratio), new.passage(a2, b2)


def page_by_lines(p: int, old: Lines, new: Lines, amap: dict) -> int | None:
    on_page = [n for n in old.nums if old.page[n] == p]
    if not on_page:
        return None
    mid = on_page[len(on_page) // 2]
    return new.page[amap[mid]]


def labels_before(text: str, i: int, width: int = 200) -> list[str]:
    """Caption keys named shortly before position i: \\msfig{S5} -> Fig.S5, or plain "Fig. S5" / "Table S6"."""
    ctx = text[max(0, i - width):i]
    keys = [f"{'Fig' if k[0] == 'f' else 'Table'}.{v}" for k, v in re.findall(r"\\ms(fig|tab)n?\{([^{}]*)\}", ctx)]
    keys += [f"{'Fig' if k.startswith('Fig') else 'Table'}.{v}" for k, v in re.findall(r"\b(Figs?\.|Tables?)\s*(S?\d+)", ctx)]
    return keys


def remap_text(text: str, pattern: re.Pattern, old: Lines, new: Lines, amap: dict, where: str, rows: list, latex: bool = True) -> str:
    """Rewrite every reference that ``pattern`` finds (groups: lines a,b | page | pages a,b) in ``text``."""
    out = []
    last = 0
    for m in pattern.finditer(text):
        before = text[:m.start()]
        if latex and re.search(r"(?<!\\)%[^\n]*$", before.rsplit("\n", 1)[-1]):
            continue  # a LaTeX comment
        g = m.groups()
        ctx = re.findall(r"\\revcomment\{([^}]*)\}", before)
        sec = ctx[-1] if ctx else ("editor" if "\\toeditor" in before else "")
        if g[0] is not None:  # a line range
            a, b = int(g[0]), int(g[1] or g[0])
            a2, b2, st, passage = remap_lines(a, b, old, new, amap)
            old_words = " ".join(old.passage(a, b).split()[:7])
            rows.append((where, sec, f"l. {a}-{b}" if a != b else f"l. {a}", f"l. {a2}-{b2}" if a2 != b2 else f"l. {a2}", st,
                         old_words, " ".join(passage.split()[:7])))
            repl = m.group(0).replace(f"{{{a}}}{{{b}}}", f"{{{a2}}}{{{b2}}}") if latex else \
                (f"l. {a2}-{b2}" if a2 != b2 else f"l. {a2}")
        else:  # a page or page range
            pages = [int(x) for x in g[2:] if x is not None]
            keys = [k for k in labels_before(text, m.start()) if k in old.captions]
            keyed = [k for k in keys if min(pages) <= old.captions[k] <= max(pages) and k in new.captions]
            if keyed:  # a page range keeps its length when a multi-page table is named once (Table S5, pp. 35-38)
                new_pages = sorted({new.captions[k] for k in keyed})
                if len(pages) == 1:
                    p2, st = [new.captions[keyed[-1]]], "ok"
                else:
                    p2 = [new_pages[0], max(new_pages[-1], new_pages[0] + pages[1] - pages[0])]
                    st = "ok" if p2 == pages else "changed"
                note = "caption " + ", ".join(keyed)
            else:
                p2 = [page_by_lines(p, old, new, amap) or p for p in pages]
                st, note = ("review", "no caption named nearby, page mapped by its lines")
            old_s, new_s = "-".join(map(str, pages)), "-".join(map(str, p2))
            rows.append((where, sec, f"p. {old_s}", f"p. {new_s}", st, note, ""))
            if latex:
                repl = m.group(0).replace("".join(f"{{{p}}}" for p in pages), "".join(f"{{{p}}}" for p in p2))
            else:
                repl = ("pp. " if len(p2) > 1 else "p. ") + new_s
        out.append(text[last:m.start()])
        out.append(repl)
        last = m.end()
    out.append(text[last:])
    return "".join(out)


def caption_table(path: Path, captions: dict) -> bool:
    text = path.read_text()
    a, b = text.find("% BEGIN caption pages"), text.find("% END caption pages")
    if a < 0 or b < 0:
        return False
    order = sorted(captions, key=lambda k: (k.split(".")[0] != "Fig", "S" in k, int(k.split(".")[1].lstrip("S"))))
    cells = [f"\\mscaption{{{k}}}{{{captions[k]}}}" for k in order]
    body = "\n".join("".join(cells[i:i + 6]) for i in range(0, len(cells), 6))
    path.write_text(text[:a] + "% BEGIN caption pages (manuscript_links.py)\n" + body + "\n" + text[b:])
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot", action="store_true", help="Record the marked PDF on disk as the one the current numbers refer to.")
    p.add_argument("--build", choices=["all", "marked", "none"], default="all",
                   help="Rebuild the paper and the marked copy (all), the marked copy only, or nothing (none).")
    p.add_argument("--no-build", action="store_const", const="none", dest="build", help="Same as --build none.")
    p.add_argument("--letter", type=Path, action="append", help="Letter file(s) to rewrite (default: the repo letter).")
    p.add_argument("--changes", action="store_true", help="Also rewrite the line and page numbers in manuscript_changes.md.")
    p.add_argument("--dry-run", action="store_true", help="Report only; write no file.")
    args = p.parse_args()
    pdf = MARKED.with_suffix(".pdf")

    if args.snapshot:
        data = extract(pdf)
        SNAPSHOT.write_text(json.dumps(data, indent=0))
        print(f"[snapshot] {pdf.name}: {len(data['lines'])} numbered lines, {len(data['captions'])} captions, {data['pages']} pages")
        for path in COMBINED:
            if path.exists():
                known = dict(re.findall(r"\\mscaption\{([^}]*)\}\{(\d+)\}", path.read_text()))
                diff = {k: (v, data["captions"].get(k)) for k, v in known.items() if int(v) != data["captions"].get(k)}
                print(f"[snapshot] caption pages against {path.name}: {'all agree' if not diff else diff}")
                break
        return

    if not SNAPSHOT.exists():
        raise SystemExit("no manuscript_lines.json: run --snapshot first, on the marked PDF the letter's numbers refer to")
    if args.build == "all":
        print("[build] paper (pdflatex, bibtex, pdflatex x2)")
        build_paper()
    if args.build != "none":
        print("[build] marked copy (latexdiff, pdflatex x3)")
        build_marked()
    old = Lines(json.loads(SNAPSHOT.read_text()))
    data = extract(pdf)
    new = Lines(data)
    same = data["sha256"] == json.loads(SNAPSHOT.read_text())["sha256"]
    amap = {n: n for n in old.nums} if same else align(old, new)
    print(f"[marked] {pdf.name}: {len(new.nums)} numbered lines, {len(new.captions)} captions, {data['pages']} pages"
          f"{' (same PDF as the snapshot: verification only)' if same else ''}")

    lost = [k for k in old.captions if k not in new.captions]
    if len(new.nums) < 0.8 * len(old.nums) or lost:
        raise SystemExit(f"the new marked PDF reads badly ({len(new.nums)} numbered lines against {len(old.nums)}, captions not found: "
                         f"{lost}); nothing rewritten")
    rows: list[tuple] = []
    letters = args.letter or [LETTER]
    for path in letters:
        text = path.read_text()
        where = f"{path.parent.name}/{path.name}"
        text2 = remap_text(text, REF, old, new, amap, where, rows)
        if text2 != text and not args.dry_run:
            path.write_text(text2)
        print(f"[letter] {path}: {sum(1 for r in rows if r[0] == where)} references"
              f"{'' if text2 == text else ', rewritten'}")
    if args.changes:
        text = CHANGES.read_text()
        pat = re.compile(r"\bl\. (\d+)(?:[-\u2013](\d+))?|\bp\. (\d+)(?![-\u2013\d])|\bpp\. (\d+)[-\u2013](\d+)")
        text2 = remap_text(text, pat, old, new, amap, CHANGES.name, rows, latex=False)
        if text2 != text and not args.dry_run:
            CHANGES.write_text(text2)
        print(f"[changes] {CHANGES.name}: {sum(1 for r in rows if r[0] == CHANGES.name)} references")
    if not args.dry_run:
        for path in COMBINED:
            if path.exists() and caption_table(path, new.captions):
                print(f"[captions] table rewritten in {path}")
        SNAPSHOT.write_text(json.dumps(data, indent=0))

    counts = {s: sum(1 for r in rows if r[4] == s) for s in ("ok", "changed", "review")}
    lines = [f"# Manuscript references after remapping to {pdf.name} ({data['pages']} pages, {date.today()})", "",
             f"Status: ok = same passage, changed = passage edited (check the range), review = needs a look. "
             f"Counts: {counts}.", "", "| File | Comment | Old | New | Status | Old passage starts | New passage starts |",
             "|---|---|---|---|---|---|---|"]
    lines += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} |" for r in rows]
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"[report] {REPORT}: {counts}")


if __name__ == "__main__":
    main()
