# Memory, latency, hardware, precision and quantisation (comment 1.8)
**Comment 1.8:** "report total memory requirements, inference latency, hardware, numerical precision, and any quantization used". **Goes to** the letter (one table).
```bash
.venv/bin/python scripts/figures/review/core/a8_edge_benchmark/run_analysis.py --device auto --require-idle-gpu --force   # GPU + CPU, 12 min, idle GPU only
.venv/bin/python scripts/figures/review/core/a8_edge_benchmark/make_figure.py                                            # summary table, < 1 s
```
**Inputs:** released weights at their pinned revisions (`_hf_pins.py`; the released Q-Former has 28.8 M parameters, the final MLP projector 28.6 M), NeuroQwen3-0.6B,
the first 16 PubMed test papers (title [SEP] abstract and map, cached in `outputs/cache/inputs.pt`).
**Method:** each (component, precision, device) runs in its own process, on an RTX 5090 (fp32 / bf16 / fp16) and an 8-thread Ryzen 7 9800X3D (fp32 / bf16 /
int8 = dynamic quantisation of every `nn.Linear`). Brain-to-text = projector + NeuroQwen3-0.6B, greedy, exactly 128 new tokens. Latency = median of 5 timed calls
(20 for light GPU components) at batch 1 and 16; peak memory = `torch.cuda.max_memory_allocated` (GPU) or the process max RSS (CPU), per batch
size and from the end of loading: the kernel's peak-RSS counter is reset (`/proc/self/clear_refs`) after loading, and int8 first releases the fp32
model it was quantised from (`quantize_dynamic` returns a copy); the load-time peak is kept as `cpu_load_peak_rss_mb`. The table gives batch 1.
**Outputs** (`outputs/`): `a8_summary_table.{md,csv}` (the response table, from `results.csv` and `hardware.json`), `results.csv` (every component, setting and
batch size), `raw/*.json`, `hardware.json`, `metrics.json`.
**Result:** CPU bf16: brain-to-text (685 M parameters) 5.9 s and 2.2 GB per 128-token description, text-to-brain 52 ms and 1.2 GB per query; int8 is the
fastest CPU brain-to-text (3.3 s, 2.8 GB; fp32 11.9 s, 3.5 GB). Before the fix (2026-10-09 run) the int8 process kept its fp32 source and peaked at 7.0 GB,
and every CPU peak included load-time copies (fp32 brain-to-text 5.5 GB: its load-time peak). CPU rows rerun on 2026-10-12; GPU rows from 2026-10-09
(unchanged code path). No phone or embedded device was tested.
