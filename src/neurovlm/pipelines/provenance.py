"""Offline provenance and fingerprint helpers for reproducible runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

from .serialization import json_safe


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA256 digest of a local file without loading it at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_value(value: Any) -> str:
    """Return a stable SHA256 digest for a JSON-compatible value."""

    payload = json.dumps(
        json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def fingerprint_path(path: str | Path) -> dict[str, Any]:
    """Fingerprint a file or directory tree using relative file names."""

    path = Path(path)
    if not path.exists():
        return {"kind": "missing", "path": str(path), "exists": False}
    if path.is_file():
        return {
            "kind": "file",
            "path": str(path),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    files = []
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        files.append(
            {
                "path": item.relative_to(path).as_posix(),
                "size": item.stat().st_size,
                "sha256": sha256_file(item),
            }
        )
    return {
        "kind": "directory",
        "path": str(path),
        "files": files,
        "sha256": sha256_value(files),
    }


def fingerprint_references(references: Mapping[str, Any]) -> dict[str, Any]:
    """Fingerprint local references and canonicalize remote/logical references.

    Strings are treated as local paths only when they currently exist.  URLs,
    Hugging Face identifiers, and other logical references are never opened.
    """

    output: dict[str, Any] = {}
    for name, reference in sorted(references.items()):
        exists = False
        if isinstance(reference, (str, os.PathLike)):
            try:
                exists = Path(reference).exists()
            except OSError:
                exists = False
        if exists:
            output[str(name)] = fingerprint_path(reference)
        else:
            safe = json_safe(reference)
            output[str(name)] = {
                "kind": "reference",
                "value": safe,
                "sha256": sha256_value(safe),
            }
    return output


def environment_provenance(packages: Iterable[str] = ("neurovlm", "numpy", "torch")) -> dict[str, Any]:
    """Collect a compact environment description without importing packages."""

    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": versions,
    }


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.strip()


def git_provenance(repo: str | Path = ".") -> dict[str, Any]:
    """Read Git revision and dirty state; never mutate the repository."""

    repo = Path(repo)
    try:
        root = Path(_git(repo, "rev-parse", "--show-toplevel"))
        status = _git(root, "status", "--porcelain", "--untracked-files=normal")
        branch = _git(root, "branch", "--show-current")
        return {
            "available": True,
            "commit": _git(root, "rev-parse", "HEAD"),
            "branch": branch or None,
            "dirty": bool(status),
            "status": status.splitlines(),
        }
    except (FileNotFoundError, subprocess.SubprocessError, OSError) as error:
        return {"available": False, "error": str(error)}


__all__ = [
    "environment_provenance",
    "fingerprint_path",
    "fingerprint_references",
    "git_provenance",
    "sha256_file",
    "sha256_value",
]
