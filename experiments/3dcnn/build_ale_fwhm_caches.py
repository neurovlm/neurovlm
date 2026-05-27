#!/usr/bin/env python
"""Build atlas-free ALE caches for a small FWHM sweep."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from neurovlm.gnn.ale_dataset import ALEPreprocessConfig, build_or_load_ale_cache


def _fwhm_label(value: float) -> str:
    return str(float(value)).replace(".", "p")


def cache_name(args: argparse.Namespace, fwhm: float) -> str:
    return (
        f"{args.mode}_ale_{int(args.resolution_mm)}mm_"
        f"fwhm{_fwhm_label(fwhm)}_"
        f"{'crop' if args.crop_to_brain else 'full'}_{args.cache_dtype}.pt"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build ALE packed caches for FWHM variants.")
    p.add_argument("--mode", choices=["atlas_free", "difumo_compatible"], default="atlas_free")
    p.add_argument("--fwhm-values", default="12,15")
    p.add_argument("--resolution-mm", type=float, default=4.0)
    p.add_argument("--cache-dir", default="data/ale_caches")
    p.add_argument("--cache-dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p.add_argument("--crop-to-brain", dest="crop_to_brain", action="store_true", default=True)
    p.add_argument("--no-crop-to-brain", dest="crop_to_brain", action="store_false")
    p.add_argument("--normalize", choices=["max", "mass", "none"], default="max")
    p.add_argument("--no-clamp", action="store_true")
    p.add_argument("--max-papers", type=int, default=None)
    p.add_argument("--force-rebuild", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fwhm_values = [float(v.strip()) for v in args.fwhm_values.split(",") if v.strip()]
    for fwhm in fwhm_values:
        config = ALEPreprocessConfig(
            mode=args.mode,
            kernel_fwhm_mm=fwhm,
            resolution_mm=args.resolution_mm,
            crop_to_brain=args.crop_to_brain,
            normalize=args.normalize,
            clamp=not args.no_clamp,
            cache_dtype=args.cache_dtype,
            max_papers=args.max_papers,
        )
        path = cache_dir / cache_name(args, fwhm)
        payload = build_or_load_ale_cache(path, config, force_rebuild=args.force_rebuild)
        print(f"cache={path}")
        print(f"  n_volumes={payload['metadata']['n_volumes']:,}")
        print(f"  shape={tuple(payload['metadata']['shape'])}")


if __name__ == "__main__":
    main()
