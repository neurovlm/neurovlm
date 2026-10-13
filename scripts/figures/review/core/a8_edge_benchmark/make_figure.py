#!/usr/bin/env python
"""A8 summary table (outputs/a8_summary_table.{md,csv}): latency and peak memory of both pipelines at batch 1.

Reads run_analysis.py's outputs/results.csv and outputs/hardware.json.  CPU, < 1 s.

    .venv/bin/python scripts/figures/review/core/a8_edge_benchmark/make_figure.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
COMPONENTS = {"text_to_brain": "text-to-brain (SPECTER2 + head + decoder)",
              "brain_to_text": "brain-to-text (Q-Former + NeuroQwen3-0.6B, 128 tokens)"}
SETTINGS = [("cuda", "bf16"), ("cpu", "fp32"), ("cpu", "bf16"), ("cpu", "int8")]


def markdown(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def latency(ms: float) -> str:
    return f"{ms:.0f} ms" if ms < 1000 else f"{ms / 1000:.1f} s"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=HERE / "outputs")
    out = p.parse_args().out
    r = pd.read_csv(out / "results.csv")
    r = r[r["batch_size"] == 1].set_index(["component", "device", "precision"])
    hw = json.loads((out / "hardware.json").read_text())
    rows = []
    for comp, label in COMPONENTS.items():
        rec = {"Pipeline": label, "Parameters": f"{r.loc[(comp, 'cpu', 'fp32'), 'params_total'] / 1e6:.0f} M"}
        for device, prec in SETTINGS:
            x = r.loc[(comp, device, prec)]
            mem_mb = x["gpu_peak_mb"] if device == "cuda" else x["cpu_peak_rss_mb"]
            rec[f"{'GPU' if device == 'cuda' else 'CPU'} {prec}"] = f"{latency(1000 * x['latency_median_s'])}, {mem_mb / 1024:.1f} GB"
        rows.append(rec)
    df = pd.DataFrame(rows)
    note = (f"Latency per query at batch 1, then peak memory during the batch-1 calls, after loading (GPU: allocated; CPU: "
            f"process RSS). GPU: {hw['gpu_name']}; CPU: {hw['cpu_model']}, {hw['torch_threads']} threads. int8 = PyTorch "
            "dynamic quantisation of linear layers, measured after the fp32 model it is made from is released. No phone or "
            "embedded device was tested.")
    df.to_csv(out / "a8_summary_table.csv", index=False)
    (out / "a8_summary_table.md").write_text(markdown(df) + "\n" + note + "\n")
    print(markdown(df) + "\n" + note)


if __name__ == "__main__":
    main()
