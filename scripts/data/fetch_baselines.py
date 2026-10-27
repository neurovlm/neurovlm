#!/usr/bin/env python
"""Fig. S3's baselines from their authors' releases: the NiCLIP checkpoint and the GC-LDA model fit on its training split
(OSF project dsj56), and NeuroConText's text pipeline and training PMIDs (Zenodo record 14169410). Each file is pinned
by sha256 in scripts/config/hf_manifest.json (not_on_hub, group "baselines") and placed in <data dir>/baselines/. Members
of the 8 GB NeuroConText archive are read through HTTP range requests, so the archive is never downloaded whole. The
NeuroConText checkpoint is a Hub file (neurovlm/neurocontext-baseline, pinned in the manifest's files) that the figure
script fetches through common.hf_pinned.

    .venv/bin/python scripts/data/fetch_baselines.py            # fetch what is missing, verify what is there
    .venv/bin/python scripts/data/fetch_baselines.py --check    # verify only
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))  # scripts/pipeline/
from common import data_dir, hf_manifest, sha256_file  # noqa: E402

GROUP = "baselines"


def entries() -> list[dict]:
    return [e for e in hf_manifest()["not_on_hub"] if e.get("group") == GROUP]


class RangeFile(io.RawIOBase):
    """Read-only, seekable view of a URL through HTTP range requests (zipfile reads the directory, then one member)."""

    def __init__(self, url: str):
        head = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60)
        self.size, self.url, self.pos = int(head.headers["Content-Length"]), head.geturl(), 0

    def seek(self, offset, whence=0):
        self.pos = (offset, self.pos + offset, self.size + offset)[whence]
        return self.pos

    def tell(self):
        return self.pos

    def readable(self):
        return True

    def seekable(self):
        return True

    def readinto(self, buffer):
        n = min(len(buffer), self.size - self.pos)
        if n <= 0:
            return 0
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        data = urllib.request.urlopen(request, timeout=300).read()
        buffer[: len(data)] = data
        self.pos += len(data)
        return len(data)


def download(e: dict, target: Path) -> None:
    """The URL, or one member of the zip at the URL, to target; sha256-checked."""
    tmp = target.with_name(target.name + ".tmp")
    target.parent.mkdir(parents=True, exist_ok=True)
    if "member" in e:
        archive = zipfile.ZipFile(io.BufferedReader(RangeFile(e["url"]), buffer_size=1 << 22))
        src = archive.open(e["member"])
    else:
        src = urllib.request.urlopen(e["url"], timeout=300)
    with src, open(tmp, "wb") as dst:
        while chunk := src.read(1 << 24):
            dst.write(chunk)
    if sha256_file(tmp) != e["sha256"]:
        tmp.unlink()
        raise RuntimeError(f"{e['url']}{' ' + e['member'] if 'member' in e else ''}: sha256 differs from the pinned {e['sha256'][:12]}")
    os.replace(tmp, target)


def ensure(root: Path | None = None, *, check: bool = False) -> dict[str, Path]:
    """Path of every pinned baseline file by manifest name. Present files are verified, missing ones fetched (not with check)."""
    root = root or data_dir() / "baselines"
    paths = {}
    for e in entries():
        target = root / e["name"]
        if target.exists() and sha256_file(target) == e["sha256"]:
            print(f"ok        {target}")
        elif check:
            raise FileNotFoundError(f"{target} missing or differs from the pinned sha256 (run without --check)")
        else:
            print(f"fetching  {e['url']}{' ' + e['member'] if 'member' in e else ''} -> {target}")
            download(e, target)
        paths[e["name"]] = target
    return paths


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=None, help="Data dir (default: neurovlm.data.data_dir); files go to its baselines/.")
    p.add_argument("--check", action="store_true", help="Verify only: no download, no write.")
    args = p.parse_args(argv)
    ensure(args.data_dir / "baselines" if args.data_dir else None, check=args.check)
    return 0


if __name__ == "__main__":
    sys.exit(main())
