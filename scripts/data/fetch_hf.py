#!/usr/bin/env python
"""Download every pinned Hugging Face input of NeuroVLM (scripts/config/hf_manifest.json) and check its sha256.

    .venv/bin/python scripts/data/fetch_hf.py                    # everything, into the package data dir
    .venv/bin/python scripts/data/fetch_hf.py --only pipeline    # training + validation inputs only
    .venv/bin/python scripts/data/fetch_hf.py --check            # verify what is there, download nothing
    .venv/bin/python scripts/data/fetch_hf.py --data-dir /scratch/nvlm/.cache/neurovlm --only figures

Files with a local_name are copied into --data-dir under that name (neurovault_images.tar is unpacked there). Files the
library loaders read from the HF cache (hf_cache) are downloaded there at the pinned revision, and the cache's
refs/main is set to that revision: the loaders call hf_hub_download without a revision and try local files first, so
they then resolve the pinned file. A present file whose sha256 differs is never replaced without --overwrite (a stale
copy would silently load other weights or data). Exit status 1 on any mismatch, missing file or failed download.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[1] / "config" / "hf_manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def load_entries(path: Path, only: str) -> list[dict]:
    files = json.loads(path.read_text())["files"]
    return files if only == "all" else [e for e in files if only in e["groups"]]


def label(e: dict) -> str:
    return f"{e['repo']}/{e['path']}@{e['revision'][:8]}"


def cache_ref(e: dict) -> Path:
    """refs/<ref> file of the entry's repo in the HF cache."""
    from huggingface_hub.constants import HF_HUB_CACHE

    folder = f"{e['repo_type']}s--{e['repo'].replace('/', '--')}"
    return Path(HF_HUB_CACHE) / folder / "refs" / e.get("ref", "main")


def cached(e: dict) -> Path | None:
    """Path of the pinned file in the HF cache, or None."""
    from huggingface_hub import try_to_load_from_cache

    p = try_to_load_from_cache(e["repo"], e["path"], revision=e["revision"], repo_type=e["repo_type"])
    return Path(p) if isinstance(p, str) else None


def download(e: dict, staging: Path | None = None, force: bool = False) -> Path:
    """hf_hub_download at the pinned revision: into the HF cache, or into `staging` (data-dir-only files)."""
    from huggingface_hub import hf_hub_download

    kw = {"local_dir": staging} if staging is not None else {}
    return Path(hf_hub_download(e["repo"], e["path"], repo_type=e["repo_type"], revision=e["revision"], force_download=force, **kw))


class Fetcher:
    def __init__(self, data_dir: Path, *, check: bool, overwrite: bool):
        self.data_dir, self.check, self.overwrite = data_dir, check, overwrite
        self.staging = data_dir / ".fetch_hf"
        self.rows: list[tuple[str, str, str]] = []  # (status, entry, detail)
        self.refs: dict[Path, bool] = {}  # refs already checked or pinned
        self.tmp: list[Path] = []  # temp dirs of --check

    def note(self, status: str, e: dict, detail: str = "") -> None:
        self.rows.append((status, label(e), detail))
        print(f"{status:<9} {label(e)}{'  ' + detail if detail else ''}", flush=True)

    def in_cache(self, e: dict) -> Path | None:
        """Make sure the pinned file is in the HF cache with the right sha256 (download unless --check)."""
        p = cached(e)
        if p is None and not self.check:
            p = download(e)
        if p is None:
            self.note("missing", e, "not in the HF cache")
            return None
        if sha256(p) != e["sha256"] and self.overwrite and not self.check:
            p = download(e, force=True)
        if sha256(p) != e["sha256"]:
            self.note("MISMATCH", e, f"HF cache file {p} has another sha256 (--overwrite downloads it again)")
            return None
        return p

    def members_file(self, e: dict) -> Path | None:
        """The tar's sha256 list; --check reads it into a temp dir when it is not cached (no cache write)."""
        if not self.check or cached(e) is not None:
            return self.in_cache(e)
        import tempfile

        self.tmp.append(Path(tempfile.mkdtemp(prefix="fetch_hf_")))
        p = download(e, self.tmp[-1])
        if sha256(p) != e["sha256"]:
            self.note("MISMATCH", e, "downloaded file has another sha256")
            return None
        return p

    def pin_ref(self, e: dict) -> bool:
        """refs/main (or refs/pr/N) of the repo in the HF cache -> the pinned revision."""
        ref = cache_ref(e)
        if ref in self.refs:
            if not self.refs[ref]:
                self.note("MISMATCH", e, f"HF cache {ref.parent.name}/{ref.name} not pinned (above)")
            return self.refs[ref]
        cur = ref.read_text().strip() if ref.exists() else None
        if cur != e["revision"] and self.check:
            self.note("MISMATCH", e, f"HF cache {ref.parent.name}/{ref.name} is {cur}, the loaders would read that revision")
        elif cur != e["revision"]:
            ref.parent.mkdir(parents=True, exist_ok=True)
            ref.write_text(e["revision"])
            print(f"pinned    {ref} -> {e['revision'][:8]} (was {cur[:8] if cur else 'absent'})")
        self.refs[ref] = cur == e["revision"] or not self.check
        return self.refs[ref]

    def hf_cache_entry(self, e: dict) -> None:
        p = self.in_cache(e)
        if p is not None and self.pin_ref(e) and not e["local_name"]:
            self.note("ok", e, "HF cache")
        if p is not None and e["local_name"]:
            self.place(e, p)

    def place(self, e: dict, src: Path | None = None) -> None:
        """Copy the pinned file to data_dir/local_name (from the HF cache if it is there, else a fresh download)."""
        target = self.data_dir / e["local_name"]
        if target.exists():
            if sha256(target) == e["sha256"]:
                self.note("ok", e, f"-> {target}")
                return
            if not self.overwrite:
                self.note("MISMATCH", e, f"{target} differs from the pinned file (--overwrite replaces it)")
                return
        if self.check:
            self.note("missing", e, f"{target}")
            return
        src = src or cached(e)
        tmp = target.with_name(target.name + ".fetch_hf.tmp")
        target.parent.mkdir(parents=True, exist_ok=True)
        if src is not None:
            shutil.copyfile(src, tmp)
        else:
            shutil.move(download(e, self.staging), tmp)
        if sha256(tmp) != e["sha256"]:
            tmp.unlink()
            self.note("MISMATCH", e, "downloaded file has another sha256")
            return
        os.replace(tmp, target)
        self.note("fetched", e, f"-> {target}")

    def tar_entry(self, e: dict, members: dict[str, str]) -> None:
        """Unpack the tar into data_dir; members present with the right sha256 are kept, others extracted."""
        bad, need = [], []
        for name, digest in members.items():
            p = self.data_dir / name
            if not p.exists():
                need.append(name)
            elif sha256(p) != digest:
                bad.append(name)
        if bad and not self.overwrite:
            self.note("MISMATCH", e, f"{len(bad)} files in {self.data_dir / e['local_name']} differ, e.g. {bad[0]} (--overwrite replaces them)")
        todo = set(need + (bad if self.overwrite else []))
        if not todo:
            if not bad:
                self.note("ok", e, f"{len(members)} files in {self.data_dir / e['local_name']}")
            return
        if self.check:
            self.note("missing", e, f"{len(need)} of {len(members)} files in {self.data_dir / e['local_name']}")
            return
        src = cached(e) or download(e, self.staging)
        if sha256(src) != e["sha256"]:
            self.note("MISMATCH", e, f"{src} has another sha256")
            return
        n = 0
        with tarfile.open(src) as tar:
            for m in tar:
                if m.name not in todo or not m.isfile():
                    continue
                target = self.data_dir / m.name
                tmp = target.with_name(target.name + ".fetch_hf.tmp")
                target.parent.mkdir(parents=True, exist_ok=True)
                h = hashlib.sha256()
                with tar.extractfile(m) as fin, open(tmp, "wb") as fout:
                    while chunk := fin.read(1 << 24):
                        h.update(chunk)
                        fout.write(chunk)
                if h.hexdigest() != members[m.name]:
                    tmp.unlink()
                    self.note("MISMATCH", e, f"tar member {m.name} has another sha256")
                    return
                os.replace(tmp, target)
                n += 1
        if src.is_relative_to(self.staging):
            src.unlink()
        if n != len(todo):
            self.note("missing", e, f"{len(todo) - n} members not in the tar")
            return
        self.note("fetched", e, f"{n} files -> {self.data_dir / e['local_name']}")

    def run(self, entries: list[dict], all_entries: list[dict]) -> int:
        by_path = {(x["repo"], x["path"]): x for x in all_entries}
        members = {(x["repo"], x["members"]) for x in all_entries if x.get("unpack")}
        for e in entries:
            if (e["repo"], e["path"]) in members:  # handled with its tar
                continue
            try:
                if e.get("unpack") == "tar":
                    p = self.members_file(by_path[(e["repo"], e["members"])])
                    if p is not None:
                        lines = [ln.split(maxsplit=1) for ln in p.read_text().splitlines() if ln.strip()]
                        self.tar_entry(e, {name.strip(): digest for digest, name in lines})
                elif e["hf_cache"]:
                    self.hf_cache_entry(e)
                else:
                    self.place(e)
            except Exception as exc:  # network errors, gated repos, ...
                self.note("FAILED", e, f"{type(exc).__name__}: {exc}")
        for d in [self.staging, *self.tmp]:
            shutil.rmtree(d, ignore_errors=True)
        counts = {s: sum(r[0] == s for r in self.rows) for s in ("ok", "fetched", "MISMATCH", "missing", "FAILED")}
        print("\n" + "  ".join(f"{k} {v}" for k, v in counts.items()) + f"  ({len(self.rows)} entries checked, data dir {self.data_dir})")
        bad = [r for r in self.rows if r[0] in ("MISMATCH", "missing", "FAILED")]
        for status, name, detail in bad:
            print(f"  {status}: {name}  {detail}")
        return 1 if bad else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=None, help="Target data dir (default: neurovlm.data.data_dir).")
    p.add_argument("--only", choices=["pipeline", "figures", "all"], default="all",
                   help="pipeline = training + validation inputs, figures = main and review figures, all = every entry.")
    p.add_argument("--check", action="store_true", help="Verify only: no download, no write.")
    p.add_argument("--overwrite", action="store_true", help="Replace present files whose sha256 differs from the manifest.")
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    args = p.parse_args(argv)
    if args.data_dir is None:
        from neurovlm.data import data_dir

        args.data_dir = Path(data_dir)
    if not args.check:
        args.data_dir.mkdir(parents=True, exist_ok=True)
    entries = load_entries(args.manifest, args.only)
    all_entries = load_entries(args.manifest, "all")
    return Fetcher(args.data_dir, check=args.check, overwrite=args.overwrite).run(entries, all_entries)


if __name__ == "__main__":
    sys.exit(main())
