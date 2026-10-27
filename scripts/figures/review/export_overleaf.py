#!/usr/bin/env python
"""Copy the response letters and every file they read into the Overleaf project, paper/response/.

    .venv/bin/python scripts/figures/review/export_overleaf.py              # writes ../../../../paper/response/
    .venv/bin/python scripts/figures/review/export_overleaf.py --out DIR

Both letters are parsed for \\input, \\include, \\InputIfFileExists, \\includegraphics and \\includepdf, also where the
letter's own macros call them (\\rfig, \\rfigparts, \\rfigpart) and in the files they input.  Figures (and a figure's
_parts.tex) go to response/figures/ under their own name (the folder name is prepended on a clash), other files next to
the letters, and the marked manuscript paper/sn-article-diff.pdf becomes response/manuscript_marked.pdf.  The paths in
the copied .tex files are rewritten relative to response/; the repo files are not touched.  Files left in response/
by an earlier export are removed (the folder must be empty or an earlier export).  Upload response/ and compile
response_letter_with_manuscript.tex there.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[3] / "paper"
LETTERS = ["response_letter.tex", "response_letter_with_manuscript.tex"]
RENAME = {PAPER / "sn-article-diff.pdf": "manuscript_marked.pdf", PAPER / "sn-article.pdf": "manuscript.pdf"}
FILE_CMDS = {"input": "tex", "include": "tex", "InputIfFileExists": "tex", "includegraphics": "graphics",
             "includepdf": "pdf"}
DEFAULT_EXT = {"tex": [".tex"], "graphics": [".pdf", ".png", ".jpg", ".jpeg"], "pdf": [".pdf"]}

README = """\
Export: in the neurovlm repo run `.venv/bin/python scripts/figures/review/export_overleaf.py`; it rewrites this folder.
Upload: this response/ folder to the Overleaf project, then compile response_letter_with_manuscript.tex (letter + marked manuscript) or response_letter.tex (letter only) as the main document.
Manuscript: manuscript_marked.pdf is a copy of paper/sn-article-diff.pdf, which Overleaf cannot rebuild; after any paper change rebuild the marked paper and export again.
"""


def mask_comments(s: str) -> str:
    """s with every LaTeX comment replaced by spaces (same length, so positions carry over)."""
    out = list(s)
    i = 0
    while i < len(s):
        if s[i] == "\\":
            i += 2
            continue
        if s[i] == "%":
            while i < len(s) and s[i] != "\n":
                out[i] = " "
                i += 1
            continue
        i += 1
    return "".join(out)


def group(s: str, i: int, open_: str = "{", close: str = "}") -> int:
    """Index after the group that starts at s[i] == open_ (escaped characters and nested braces skipped)."""
    depth = 0
    j = i
    while j < len(s):
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == open_ or (c == "{" and open_ != "{"):
            depth += 1
        elif c == close or (c == "}" and close != "}"):
            depth -= 1
            if depth == 0:
                return j + 1
        j += 1
    raise ValueError(f"unbalanced group at {s[i:i + 40]!r}")


def skip_space(s: str, i: int) -> int:
    while i < len(s) and s[i] in " \t\n":
        i += 1
    return i


def read_args(s: str, i: int, n: int, optional: bool) -> list[tuple[int, int]] | None:
    """Spans (start, end) of the contents of n brace arguments from s[i]; [..] groups and a * skipped if optional."""
    if optional:
        if s[i:i + 1] == "*":
            i += 1
        i = skip_space(s, i)
        while s[i:i + 1] == "[":
            i = skip_space(s, group(s, i, "[", "]"))
    spans = []
    for _ in range(n):
        i = skip_space(s, i)
        if s[i:i + 1] != "{":
            return None
        j = group(s, i)
        spans.append((i + 1, j - 1))
        i = j
    return spans


def calls(s: str, names: set[str]):
    """(name, i) for every \\name in s not followed by a letter."""
    if not names:
        return
    for m in re.finditer(r"\\(" + "|".join(sorted(names, key=len, reverse=True)) + r")(?![A-Za-z@])", s):
        yield m.group(1), m.end()


def definitions(s: str) -> dict[str, tuple[int, str]]:
    """{macro: (number of arguments, body)} of the \\newcommand-like definitions in s."""
    defs = {}
    for m in re.finditer(r"\\(?:newcommand|renewcommand|providecommand|DeclareRobustCommand)\*?\s*\{?\\([A-Za-z@]+)\}?"
                         r"\s*(?:\[(\d)\])?\s*(?=\{)", s):
        end = group(s, m.end())
        defs[m.group(1)] = (int(m.group(2) or 0), s[m.end() + 1:end - 1])
    return defs


def file_macros(text: str) -> dict[str, list[tuple[int, str, str]]]:
    """{command: [(argument number, suffix, kind)]}: the built-in file commands and every macro that passes one of
    its arguments (plus a suffix such as _parts.tex or .png) to a file command or another such macro."""
    found = {name: [(1, "", kind)] for name, kind in FILE_CMDS.items()}
    defs = definitions(text)
    changed = True
    while changed:
        changed = False
        for name, (nargs, body) in defs.items():
            for callee, i in calls(body, set(found)):
                nmax = max(k for k, _, _ in found[callee])
                spans = read_args(body, i, nmax, callee in FILE_CMDS)
                for k, suffix, kind in found[callee] if spans else []:
                    m = re.fullmatch(r"#(\d)(.*)", body[spans[k - 1][0]:spans[k - 1][1]] + suffix, re.S)
                    if m and int(m.group(1)) <= nargs and (int(m.group(1)), m.group(2), kind) not in found.get(name, []):
                        found.setdefault(name, []).append((int(m.group(1)), m.group(2), kind))
                        changed = True
    return found


def candidates(path: Path, kind: str) -> list[Path]:
    return [path] if path.suffix.lower() in DEFAULT_EXT[kind] else [path.with_name(path.name + e) for e in DEFAULT_EXT[kind]]


class Export:
    def __init__(self, out: Path):
        self.out = out
        self.taken: dict[str, Path] = {}       # file in out/ -> source
        self.written: dict[Path, str] = {}     # source -> file in out/
        self.rewrites: list[tuple[str, str, str]] = []
        self.packages: set[str] = set()
        self.macros: dict[str, list[tuple[int, str, str]]] = {}

    def free(self, stem: str, files: list[tuple[str, Path]]) -> bool:
        return all(self.taken.get(stem + suffix, src) == src for suffix, src in files)

    def place(self, arg: str, kinds: set[str], files: list[tuple[str, Path]]) -> str:
        """New value of a path argument whose files (suffix, source) all move with it."""
        for suffix, src in files:
            if src in RENAME:
                return RENAME[src][:len(RENAME[src]) - len(suffix)] if suffix else RENAME[src]
        folder = "figures/" if "graphics" in kinds else ""
        stem = folder + Path(arg).name
        if not self.free(stem, files):
            stem = folder + Path(arg).parent.name + "_" + Path(arg).name
            n = 2
            while not self.free(stem, files):
                stem, n = f"{folder}{Path(arg).parent.name}{n}_{Path(arg).name}", n + 1
        return stem

    def tex(self, src: Path, dst: str):
        """Copy a .tex file to out/dst with its file paths rewritten; follow what it reads."""
        if src in self.written:
            return
        self.taken[dst], self.written[src] = src, dst
        text = src.read_text()
        masked = mask_comments(text)
        self.packages |= {p.strip() for m in re.finditer(r"\\(?:usepackage|RequirePackage)\s*(?:\[[^]]*\])?\s*\{([^}]*)\}",
                                                          masked) for p in m.group(1).split(",") if p.strip()}
        self.macros = file_macros(masked) if not self.macros else {**file_macros(masked), **self.macros}
        edits = []
        for name, i in calls(masked, set(self.macros)):
            pats = self.macros[name]
            spans = read_args(masked, i, max(k for k, _, _ in pats), name in FILE_CMDS)
            if not spans:
                continue
            for k in sorted({k for k, _, _ in pats}):
                a, b = spans[k - 1]
                arg = text[a:b].strip()
                if "#" in arg or not arg:
                    continue
                files = []
                for _, suffix, kind in (p for p in pats if p[0] == k):
                    hit = next((c for c in candidates(HERE / (arg + suffix), kind) if c.is_file()), None)
                    if hit:
                        files.append((hit.name[len(Path(arg + suffix).name):] if suffix == "" else suffix,
                                      hit.resolve(), kind))
                if not files:
                    raise FileNotFoundError(f"{src.name}: \\{name} reads {arg!r}, which does not exist")
                new = self.place(arg, {kind for *_, kind in files}, [(s, f) for s, f, _ in files])
                for suffix, f, kind in files:
                    if kind == "tex":
                        self.tex(f, new + suffix)
                    elif f not in self.written:
                        self.taken[new + suffix], self.written[f] = f, new + suffix
                if new != arg:
                    edits.append((a, b, new))
                    self.rewrites.append((dst, arg, new))
        for a, b, new in sorted(edits, reverse=True):
            text = text[:a] + new + text[b:]
        header = "% Copied from neurovlm/scripts/figures/review by export_overleaf.py; edit the repo file and export again.\n"
        (self.out / dst).parent.mkdir(parents=True, exist_ok=True)
        (self.out / dst).write_text(header + text)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=PAPER / "response", help="folder to write (default paper/response)")
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    readme = out / "README.txt"
    if any(out.iterdir()) and not (readme.is_file() and readme.read_text().startswith("Export:")):
        raise SystemExit(f"{out} is neither empty nor an earlier export; choose another --out")
    ex = Export(out)
    for name in LETTERS:
        ex.macros = {}
        ex.tex(HERE / name, name)
    for src, dst in ex.written.items():
        if not dst.endswith(".tex"):
            (out / dst).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out / dst)
    local = [p for p in sorted(ex.packages) if (HERE / f"{p}.sty").is_file()]
    for p in local:
        shutil.copy2(HERE / f"{p}.sty", out / f"{p}.sty")
        ex.written[HERE / f"{p}.sty"] = f"{p}.sty"
    (out / "README.txt").write_text(README)
    keep = set(ex.written.values()) | {"README.txt"}
    removed = [p for p in sorted(out.rglob("*")) if p.is_file() and p.relative_to(out).as_posix() not in keep]
    for p in removed:
        p.unlink()
    for d in sorted((d for d in out.rglob("*") if d.is_dir()), reverse=True):
        if not any(d.iterdir()):
            d.rmdir()

    total = 0
    for src, dst in sorted(ex.written.items(), key=lambda x: x[1]):
        size = (out / dst).stat().st_size
        total += size
        try:
            shown = src.relative_to(HERE.parents[3])
        except ValueError:
            shown = src
        print(f"{dst:58s} {size / 1e6:7.2f} MB  <- {shown}")
    print("README.txt  (written)")
    print(f"\n{len(ex.written)} files copied, {total / 1e6:.1f} MB, into {out}")
    for dst, old, new in ex.rewrites:
        print(f"  rewrote in {dst}: {old} -> {new}")
    if removed:
        print("removed from an earlier export: " + ", ".join(p.relative_to(out).as_posix() for p in removed))
    missing = []
    if shutil.which("kpsewhich"):
        for p in sorted(ex.packages - set(local)):
            if not subprocess.run(["kpsewhich", f"{p}.sty"], capture_output=True, text=True).stdout.strip():
                missing.append(p)
    print(f"packages ({len(ex.packages)}): {', '.join(sorted(ex.packages))}; "
          + (f"copied from the repo: {', '.join(local)}; " if local else "")
          + (f"NOT FOUND in TeX Live: {', '.join(missing)}" if missing else "all in TeX Live"))


if __name__ == "__main__":
    main()
