#!/usr/bin/env python
"""Check that response_letter.tex carries the same visible response text as RESPONSE.md.

    .venv/bin/python scripts/figures/review/check_letter_sync.py            # sync check, exit 1 on any mismatch
    .venv/bin/python scripts/figures/review/check_letter_sync.py --counts   # words and numbers per response

Per section (the editor letter and every ``### <id>:`` comment) the letter's text after removing the quoted comment,
figures, tables, "Changes to the manuscript" and LaTeX comments must equal the RESPONSE.md text (hidden comments,
anchors, image lines and the **Comment:** paraphrase removed; a link counts as its text, and the letter's manuscript
links \\msfig{S8}, \\mstab{S5} and \\msfign{S4} read as "Fig. S8", "Table S5" and "S4").  Also checked: every
RESPONSE.md table row is in the letter's section, every RESPONSE.md image ``![Fig. Rn](path)`` is figure Rn of the
letter with the same file, and the comment links of each section (``[2.4](#2.4)`` in RESPONSE.md, ``\\cmt{2.4}`` in
the letter) name the same comments, each of which exists.  Figures are numbered by the order of the ``\\rfig`` calls,
as LaTeX numbers them.
"""
from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def braced(s: str, i: int) -> tuple[str, int]:
    """Content of the brace group starting at s[i] == '{' and the index after it."""
    assert s[i] == "{", s[i:i + 20]
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{" and s[j - 1] != "\\":
            depth += 1
        elif s[j] == "}" and s[j - 1] != "\\":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
    raise ValueError("unbalanced braces")


def rfigs(tex: str) -> list[dict]:
    """Every \\rfig{label}{file}{caption}, \\rfigparts{label}{file without .png}{caption} or \\fig{width}{file}{caption}
    (unlabelled) in order; number = position + 1."""
    out = []
    for m in re.finditer(r"\\(rfigparts|rfig|fig)(?=\{)", tex):
        label, i = braced(tex, m.end())
        label = "" if m.group(1) == "fig" else label
        path, i = braced(tex, i)
        path = path if path.endswith(".png") else path + ".png"  # \\rfigparts takes the name without .png
        caption, i = braced(tex, i)
        out.append({"label": label, "path": path, "caption": caption, "start": m.start(), "end": i, "n": len(out) + 1})
    return out


def norm(s: str, labels: dict[str, int] | None = None) -> str:
    """Plain words of a LaTeX or Markdown string."""
    labels = labels or {}
    s = re.sub(r"\\ref\{([^{}]*)\}", lambda m: f"R{labels[m.group(1)]}" if m.group(1) in labels else "R??", s)
    s = re.sub(r"\\cmt\{([^{}]*)\}", r"\1", s)  # letter: link to a comment
    s = re.sub(r"\\msfig\{([^{}]*)\}", r"Fig. \1", s)  # letter: manuscript figure or table, linked in the combined PDF
    s = re.sub(r"\\mstab\{([^{}]*)\}", r"Table \1", s)
    s = re.sub(r"\\ms(?:fig|tab)n\{([^{}]*)\}", r"\1", s)  # its number alone, after "Figs." or "Tables"
    s = re.sub(r"\\hyperref\[[^\]]*\]\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\[([^\]]+)\]\(#[^)]*\)", r"\1", s)  # RESPONSE.md: anchor link
    for cmd in ("textbf", "emph", "textit", "mbox"):
        s = re.sub(r"\\%s\{([^{}]*)\}" % cmd, r"\1", s)
    for a, b in [(r"\%", "%"), (r"\&", "&"), (r"\_", "_"), ("{[}", "["), ("{]}", "]"),
                 (r"$\le$", "≤"), (r"$\leq$", "≤"), (r"$\ge$", "≥"), (r"$\geq$", "≥"),
                 (r"$>$", ">"), (r"$<$", "<"), (r"$\rightarrow$", "→"), (r"$-$", "−"),
                 ("---", "—"), ("--", "–"), ("``", '"'), ("''", '"'),
                 ("“", '"'), ("”", '"'), ("’", "'"), ("‘", "'"), ("~", " "), (r"\,", " "), (r"\ ", " ")]:
        s = s.replace(a, b)
    s = re.sub(r"\$([^$]*)\$", r"\1", s)
    s = s.replace("**", "")
    return " ".join(s.split())


ANCHOR = re.compile(r'^<a id="([^"]+)"></a>$')
MD_LINK = re.compile(r"\[[^\]]+\]\(#([^)]+)\)")
TEX_LINK = re.compile(r"\\cmt\{([^{}]*)\}|\\hyperref\[sec:([^\]]*)\]")


# RESPONSE.md
def md_sections(md: str) -> dict[str, dict]:
    """id -> {text: [lines], table: [[cells]], images: [(alt, path)], code: [fenced blocks]}; the editor letter has id 'editor'."""
    out, sid, code = {}, None, None
    for line in md.splitlines():
        if code is not None:  # inside a fenced code block: compared with the letter's verbatim boxes, not with its text
            if line.strip().startswith("```"):
                if sid is not None:
                    out[sid]["code"].append("\n".join(code))
                code = None
            else:
                code.append(line)
            continue
        if line.strip().startswith("```"):
            code = []
            continue
        if line.startswith("## "):
            sid = "editor" if line[3:].strip().lower() == "to the editor" else None
            if sid:
                out[sid] = {"text": [], "table": [], "images": [], "code": []}
            continue
        m = re.match(r"^### (\S+?):? ", line)
        if m:
            sid = m.group(1)
            out[sid] = {"text": [], "table": [], "images": [], "code": []}
            continue
        if sid is None or not line.strip():
            continue
        s = line.strip()
        img = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)", s)
        if img:
            out[sid]["images"].append((img.group(1), img.group(2)))
            continue
        if s.startswith(("**Comment:**", "<!--")) or ANCHOR.match(s):
            continue
        if s.startswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                out[sid]["table"].append([re.sub(r"\s*<!--.*?-->", "", c) for c in cells])
            continue
        s = re.sub(r"\s*<!--.*?-->", "", s)
        s = re.sub(r"\s*\\$", "", s)
        s = re.sub(r"^[-*] ", "", s)  # Markdown list item
        out[sid]["text"].append(s)
    return out


# response_letter.tex
BLOCK_START = re.compile(r"\\(toeditor|reviewer\{[^{}]*\}|revgeneral(?:page)?\{[^{}]*\}|revcomment\{([^{}]*)\}\{)"
                         r"|\\end\{document\}")


def tex_blocks(tex: str) -> dict[str, tuple[int, int]]:
    """id -> (start, end) of the editor letter and of every \\revcomment block."""
    marks = [(m.start(), m.group(2) if m.group(2) is not None else ("editor" if m.group(1) == "toeditor" else None))
             for m in BLOCK_START.finditer(tex)]
    out = {}
    for (a, sid), (b, _) in zip(marks, marks[1:]):
        if sid:
            out[sid] = (a, b)
    return out


def strip_env(s: str, env: str) -> str:
    return re.sub(r"\\begin\{%s\}.*?\\end\{%s\}" % (env, env), " ", s, flags=re.S)


def visible_response(block: str) -> str:
    """Response words of one letter block: no heading, quote, figures, tables, changes line or comments."""
    block = re.sub(r"(?<!\\)%.*", "", block)
    if block.startswith("\\revcomment"):
        _, i = braced(block, len("\\revcomment"))
        _, i = braced(block, i)
        block = block[i:]
    elif block.startswith("\\toeditor"):
        block = block[len("\\toeditor"):]
    for env in ("revquote", "center", "tabular", "figure", "table", "verbatim"):
        block = strip_env(block, env)
    block = block.replace("\\FloatBarrier", " ")
    while (m := re.search(r"\\(rfigparts|rfig|fig|rtab|revchanges)(?=\{)", block)):
        i, n = m.end(), {"revchanges": 1, "rtab": 2}.get(m.group(1), 3)
        for _ in range(n):
            _, i = braced(block, i)
        block = block[:m.start()] + " " + block[i:]
    block = re.sub(r"\\(begin|end)\{itemize\}|\\item\b", " ", block)  # lists: their items are text
    block = block.replace("\\response", " ").replace("\\par", " ")
    return block


def first_diff(a: str, b: str, ctx: int = 6) -> tuple[str, str]:
    wa, wb = a.split(), b.split()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, wa, wb, autojunk=False).get_opcodes():
        if tag != "equal":
            return " ".join(wa[max(0, i1 - 3):i2 + ctx]), " ".join(wb[max(0, j1 - 3):j2 + ctx])
    return "", ""


def check(md_path: Path, tex_path: Path) -> int:
    md, tex = md_path.read_text(), tex_path.read_text()
    tex = tex[tex.index("\\begin{document}"):]
    figs = rfigs(tex)
    labels = {f["label"]: f["n"] for f in figs}
    for n, m in enumerate(re.finditer(r"\\rtab(?=\{)", tex), 1):  # tables: Table R1, R2, ... in order of \\rtab
        cap, _ = braced(tex, m.end())
        lab = re.search(r"\\label\{([^{}]*)\}", cap)
        if lab:
            labels[lab.group(1)] = n
    secs, blocks = md_sections(md), tex_blocks(tex)
    bad = 0
    for sid, sec in secs.items():
        if sid not in blocks:
            print(f"MISSING in letter: {sid}")
            bad += 1
            continue
        a, b = blocks[sid]
        block = tex[a:b]
        want, got = norm(" ".join(sec["text"])), norm(visible_response(block), labels)
        if want != got:
            bad += 1
            x, y = first_diff(want, got)
            print(f"TEXT MISMATCH {sid}\n  md:     ...{x}...\n  letter: ...{y}...")
        block_norm = norm(block, labels)
        for cells in sec["table"]:
            row = " & ".join(norm(c) for c in cells)
            if row not in block_norm:
                bad += 1
                print(f"TABLE ROW MISMATCH {sid}: {row}")
        verbatim = [" ".join(v.split()) for v in re.findall(r"\\begin\{verbatim\}(.*?)\\end\{verbatim\}", block, re.S)]
        for code in sec["code"]:  # a fenced block of RESPONSE.md must be a verbatim box of the letter (same words)
            if " ".join(code.split()) not in verbatim:
                bad += 1
                print(f"CODE BLOCK MISMATCH {sid}: {' '.join(code.split())[:60]}...")
        in_block = [f for f in figs if a <= f["start"] < b]
        for alt, path in sec["images"]:
            hit = [f for f in in_block if Path(f["path"]).name == Path(path).name]
            m = re.match(r"Fig\. R(\d+)", alt)
            if not hit:
                bad += 1
                print(f"FIGURE MISSING {sid}: {path} is not a \\rfig of this comment in the letter")
            elif alt and (not m or int(m.group(1)) != hit[0]["n"]):
                bad += 1
                print(f"FIGURE NUMBER MISMATCH {sid}: RESPONSE.md says '{alt}', the letter numbers {path} R{hit[0]['n']}")
    anchors = {m.group(1) for m in re.finditer(ANCHOR.pattern, md, re.M)}
    for sid, sec in secs.items():
        if sid not in blocks:
            continue
        a, b = blocks[sid]
        md_links = MD_LINK.findall(" ".join(sec["text"]))
        tex_links = [x or y for x, y in TEX_LINK.findall(visible_response(tex[a:b]))]
        if md_links != tex_links:
            bad += 1
            print(f"LINK MISMATCH {sid}: RESPONSE.md {md_links}, letter {tex_links}")
        for t in md_links:
            if t not in anchors or t not in blocks:
                bad += 1
                print(f"BROKEN LINK {sid}: #{t} (anchor in RESPONSE.md: {t in anchors}; comment in the letter: {t in blocks})")
    for sid in blocks:
        if sid not in secs:
            bad += 1
            print(f"MISSING in RESPONSE.md: {sid}")
    for f in figs:
        if not (tex_path.parent / f["path"]).exists():
            bad += 1
            print(f"FIGURE FILE MISSING: R{f['n']} {f['path']}")
    unresolved = sorted(set(re.findall(r"\\ref\{([^{}]*)\}", tex)) - set(labels))
    for u in unresolved:
        bad += 1
        print(f"UNRESOLVED \\ref{{{u}}}")
    print(f"{len(secs)} sections (editor letter + {len(secs) - ('editor' in secs)} comments), {len(figs)} figures, "
          f"{bad} mismatches")
    return 1 if bad else 0


NAMES = re.compile(r"(Figs?\.\s+(R?\d+[a-z]?)(\s*(,|and|to|–)\s*R?\d+[a-z]?)*|[Ss]tages?\s+\d+(\s+and\s+(stage\s+)?\d+)?"
                   r"|RTX\s+\d+|[Ll]ines?\s+\d+|Qwen3(-0\.6B)?|recall@\w+|\bS\d\b"
                   r"|\b[123]\.(1[0-2]|[1-9])\b(?!\s*(hours?|million|GB|MB|ms|s\b|%)))")


def counts(md_path: Path, ids: list[str]) -> None:
    secs = md_sections(md_path.read_text())
    for sid in ids or list(secs):
        text = " ".join(secs[sid]["text"])
        plain = NAMES.sub(" ", text)
        nums = re.findall(r"(?<![\w.\-])\d[\d,]*(?:\.\d+)?(?![\w\-])", plain)
        print(f"{sid}\t{len(text.split())} words\t{len(nums)} numbers {nums}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--response", type=Path, default=HERE / "RESPONSE.md")
    p.add_argument("--letter", type=Path, default=HERE / "response_letter.tex")
    p.add_argument("--counts", nargs="*", metavar="ID", help="Print words and numbers of these sections (all if none).")
    args = p.parse_args()
    if args.counts is not None:
        counts(args.response, args.counts)
        return 0
    return check(args.response, args.letter)


if __name__ == "__main__":
    sys.exit(main())
