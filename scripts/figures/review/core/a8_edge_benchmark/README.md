# Memory, latency, hardware, precision and quantisation (comment 1.8)
**Comment 1.8:** "report total memory requirements, inference latency, hardware, numerical precision, and any quantization used". **Goes to** the letter (one table).
```bash
.venv/bin/python scripts/figures/review/core/a8_edge_benchmark/run_analysis.py --device auto --require-idle-gpu --force   # GPU + CPU, 12 min, idle GPU only
.venv/bin/python scripts/figures/review/core/a8_edge_benchmark/make_figure.py                                            # summary table, < 1 s
```
**Inputs:** released weights at their pinned revisions (`_hf_pins.py`; the released Q-Former has 28.8 M parameters, the final MLP projector 28.6 M), NeuroQwen3-0.6B,
the first 16 PubMed test papers (title [SEP] abstract and map, cached in `outputs/cache/inputs.pt`).
**Method:** each (component, precision, device) runs in its own process, on an RTX 5090 (fp32 / bf16 / fp16) and an 8-thread Ryzen 7 9800X3D (fp32 / bf16 /
int8 = `quantize_dynamic` of every `nn.Linear`). `quantize_dynamic` matches the exact type: SPECTER2's adapter library wraps 60 of its 97 linear layers (78 of
87 M linear weights) in a `LoRALinearTorch` subclass that it skips, so SPECTER2 stays mostly fp32 in int8 (quantizing those too drops the embedding cosine
to fp32 from 0.999 to 0.90, checked 2026-10-13, so they are left as shipped). Brain-to-text = projector + NeuroQwen3-0.6B, greedy, exactly 128 new tokens.
Latency = median of 5 timed calls (20 for light GPU components) at batch 1 and 16. Peak memory = what the pipeline adds (weights + activations), per batch
size and from the end of loading: `torch.cuda.max_memory_allocated` (GPU) or the process max RSS minus the RSS before any weight is loaded (CPU,
`cpu_peak_above_preload_mb`; the pre-load RSS, 0.84 GB of Python, PyTorch, transformers, adapters and the inputs, is `rss_pre_load_mb`; the total max RSS is
`cpu_peak_rss_mb`). Before each batch size the freed heap is returned to the OS (`malloc_trim`) and the kernel's peak-RSS counter reset
(`/proc/self/clear_refs`); int8 first releases the fp32 model it was quantised from (`quantize_dynamic` returns a copy); the load-time peak is kept as
`cpu_load_peak_rss_mb`. `weights_mb` = parameter and buffer bytes in memory, tied tensors once. The table gives batch 1.
**Outputs** (`outputs/`): `a8_summary_table.{md,csv}` (the response table, from `results.csv` and `hardware.json`), `results.csv` (every component, setting and
batch size), `raw/*.json`, `hardware.json`, `metrics.json`.
**Result:** CPU, per 128-token description: brain-to-text bf16 6.3 s and 1.4 GB, int8 3.5 s and 1.4 GB, fp32 13.1 s and 2.7 GB; text-to-brain bf16 53 ms
and 0.4 GB per query (GPU bf16: 3.2 s, 1.3 GB and 7 ms, 0.4 GB). int8 is not smaller than bf16: dynamic quantisation keeps NeuroQwen3's 156 M-parameter token
embedding in fp32 (0.58 GB) and gives the output layer, which shares that embedding in bf16, its own int8 copy, so the brain-to-text weights take 1.26 GB in
int8 against 1.28 GB in bf16 (`weights_mb`), and the activations stay fp32. Fixed 2026-10-13: (1) the int8 peaks held 0.56 GB of freed heap left by counting
`weights_mb` (`state_dict` unpacks every int8 weight) before the peak counter was reset: brain-to-text int8 max RSS 2.81 -> 2.25 GB (bf16 2.25 -> 2.23);
(2) the CPU cells gave the total RSS, about 0.5 GB of libraries included, while the GPU cells gave allocated bytes; both now give what the pipeline adds;
(3) `weights_mb` counted tied tensors twice (brain-to-text bf16 1604 -> 1307 MB; the GPU rows' value recomputed from the CPU rows, old one kept as
`weights_mb_uncorrected`). CPU rows rerun on 2026-10-13 (brain-to-text twice, the first pass was disturbed: p90 up to 1.5x the median); CPU latencies are
3-10 % above the 2026-10-12 run, and the old and new code timed back to back give the same int8 latency (3.6 s), so this is run-to-run variation. GPU rows
from 2026-10-09 (unchanged code path). `metrics.json` describes the last invocation. No phone or embedded device was tested.
