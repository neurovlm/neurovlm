#!/usr/bin/env python
"""Canonical title vocabulary for run_correctness.py --title-constraint: the shared one of the final run
(scripts/models/_common_models.title_vocabulary: the title of every row of the stage-14 banks, written next to the model
as <model-dir>/title_vocabulary.json, which every caption reads by default), plus the NeuroAdapter map of every
[NETWORK] / [REGION] bank row for --title-constraint shortlist.

A bank row is the projected SPECTER embedding of one curated stage-14 text ("Title [SEP] body"); the texts are rebuilt
with scripts/models/14_qformer_anat.py and each bank row is matched to its text (_common_models.bank_titles).
Writes <model-dir>/title_vocabulary.json (if missing or stale), a copy as outputs/title_vocabulary.json (read by
a7_failure_cases_percentiles/audit_names.py) and outputs/cache/bank_maps.npz (top 5 % of each row's NeuroAdapter map,
the stage-14 binarisation).
    .venv/bin/python scripts/figures/review/core/a6_steering_tokens/build_title_vocabulary.py   # GPU or CPU, ~1 min
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODELS = HERE.parents[3] / "models"
sys.path[:0] = [str(MODELS), str(MODELS.parent)]
import _final_model as M  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, default=M.DEFAULT_MODEL_DIR, help="Training run with the final model.")
    p.add_argument("--out", type=Path, default=HERE / "outputs", help="Output directory (cache/bank_maps.npz).")
    p.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = p.parse_args()
    import torch
    import _common_models as CM

    run = args.model_dir
    vocab = CM.title_vocabulary(run, device=args.device)
    specter = CM.released_model("specter", args.device)
    head = torch.load(run / "proj_head_text_infonce.pt", map_location=args.device, weights_only=False).eval()
    adapter = torch.load(run / "adapter_anat_full.pt", map_location=args.device, weights_only=False).eval()
    banks = torch.load(run / "canonical_semantic_banks.pt", map_location="cpu", weights_only=False)
    titles, rows = CM.bank_titles(banks, head, specter, device=args.device, return_rows=True)
    if any(titles[lv] != vocab["bank_rows"][lv] for lv in titles):
        raise SystemExit(f"{run / CM.TITLE_VOCABULARY_FILE} does not match the banks; delete it and rerun")
    _, _, texts = CM._stage14_texts()
    with torch.no_grad():
        txt = torch.cat([torch.nn.functional.normalize(specter(texts[s:s + 256]), dim=1).float() for s in range(0, len(texts), 256)])
    maps = {}
    for lv in CM.TITLE_LEVELS:
        with torch.no_grad():
            m = torch.sigmoid(adapter(txt[torch.as_tensor(rows[lv])].to(args.device))).float()
            top = (m >= torch.quantile(m, 0.95, dim=1, keepdim=True)).cpu().numpy()
        maps[lv] = np.packbits(top, axis=1)
        maps[f"{lv}_n_voxels"] = np.array(top.shape[1])
    for lv in ("network", "region", "function"):
        print(f"{lv}: {len(vocab['bank_rows'][lv])} bank rows, {len(vocab['titles'][lv])} titles")
    (args.out / "cache").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "cache" / "bank_maps.npz", **maps)
    # a copy for ../a7_failure_cases_percentiles/audit_names.py
    (args.out / "title_vocabulary.json").write_text(json.dumps(vocab, indent=1) + "\n")


if __name__ == "__main__":
    main()
