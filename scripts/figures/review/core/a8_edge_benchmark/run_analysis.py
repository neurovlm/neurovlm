#!/usr/bin/env python
"""A8 (comment 1.m4): parameters, memory and latency of every NeuroVLM component per precision and device.

Each (component, precision, device) runs in its own subprocess; method and outputs in README.md.  Run on an idle GPU:
    .venv/bin/python scripts/figures/review/core/a8_edge_benchmark/run_analysis.py --device auto --require-idle-gpu --force
"""
from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _review_common import (  # noqa: E402
    PAPER_TEMP, Timings, count_params, env_packages, get_logger, load_pubmed_latents, load_pubmed_maps, load_pubmed_text_rows,
    match_lm_token_norm, read_json, seed_all, set_offline_env, split_pmids, write_json,
)

set_offline_env()
log = get_logger("a8_edge_benchmark")
COMPONENTS = ("text_encoder", "projection_heads", "contrastive_retrieval", "text_to_brain", "qformer", "brain_to_text")
PRECISIONS = ("fp32", "bf16", "fp16", "int8")  # int8 = dynamic quantization of nn.Linear (CPU only); fp16 GPU only


def hardware_info(check_gpu: bool) -> dict:
    import psutil
    import torch

    info = {"platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
            "cpu_model": None, "cpu_logical": psutil.cpu_count(logical=True), "cpu_physical": psutil.cpu_count(logical=False),
            "ram_gb": round(psutil.virtual_memory().total / 2 ** 30, 1), "torch_threads": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available()}
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                info["cpu_model"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        info["cpu_model"] = platform.processor()
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update({"gpu_name": p.name, "gpu_mem_gb": round(p.total_memory / 2 ** 30, 1), "cuda": torch.version.cuda,
                     "gpu_capability": f"{p.major}.{p.minor}"})
        if check_gpu:
            try:
                q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,driver_version", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=20).stdout.strip().split(",")
                info.update({"gpu_util_pct_at_start": float(q[0]), "gpu_mem_used_mb_at_start": float(q[1]), "nvidia_driver": q[2].strip()})
                apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                      capture_output=True, text=True, timeout=20).stdout.strip()
                info["gpu_other_compute_apps"] = [a for a in apps.splitlines() if a.strip()]
            except Exception as exc:
                info["nvidia_smi_error"] = str(exc)
    return info


def module_bytes(module) -> int:
    """Bytes of params + buffers, counting dynamically quantized packed weights at their int8 size."""
    import torch

    def size(v) -> int:
        if torch.is_tensor(v):
            return v.numel() * v.element_size()
        if isinstance(v, (tuple, list)):
            return sum(size(x) for x in v)
        return 0

    return int(sum(size(v) for v in module.state_dict().values()))


def cast(module, precision: str, device):
    import torch

    if precision == "int8":
        from torch.ao.quantization import quantize_dynamic

        return quantize_dynamic(module.float().cpu(), {torch.nn.Linear}, dtype=torch.qint8).eval()
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    return module.to(device=device, dtype=dtype).eval()


def release_memory() -> None:
    """Collect freed modules and hand the freed heap pages back to the OS (glibc), so RSS drops."""
    import ctypes
    import gc

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def reset_peak_rss() -> bool:
    """Reset the kernel's peak-RSS counter (Linux >= 4.0), so that ru_maxrss covers only what follows."""
    try:
        Path("/proc/self/clear_refs").write_text("5")
        return True
    except OSError:
        return False


def worker(cfg: dict) -> dict:
    """Load one component in one precision on one device and time it; GPU peak = max allocated, CPU peak = max RSS, both
    per batch size and after loading (load-time copies, such as the fp32 model int8 is quantised from, are released)."""
    import warnings

    warnings.filterwarnings("ignore")
    import psutil
    import torch
    import torch.nn.functional as F
    from _hf_pins import load_model

    comp, precision, dev = cfg["component"], cfg["precision"], cfg["device"]
    device = torch.device(dev)
    if cfg.get("threads"):
        torch.set_num_threads(int(cfg["threads"]))
    proc = psutil.Process()
    res = {"component": comp, "precision": precision, "device": dev, "torch_threads": torch.get_num_threads(),
           "rss_baseline_mb": proc.memory_info().rss / 2 ** 20}
    inputs = torch.load(cfg["inputs"], weights_only=False)
    texts, maps = inputs["texts"], inputs["maps"].float()
    wdtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16, "int8": torch.float32}[precision]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    mods: dict = {}
    n_params: dict = {}

    def get(name):
        m = load_model(name)
        if name == "specter":
            n_params[name] = count_params(m.specter)
            m.specter = cast(m.specter, precision, device)
            m.device = device if precision != "int8" else torch.device("cpu")
            m.ref = m.ref.to(device=m.device, dtype=wdtype if precision != "int8" else torch.float32)
            return m
        n_params[name] = count_params(m)
        return cast(m, precision, device)

    if comp in ("text_encoder", "contrastive_retrieval", "text_to_brain"):
        mods["specter"] = get("specter")
    if comp in ("projection_heads", "contrastive_retrieval"):
        mods["text_head"] = get("proj_head_text_infonce")
    if comp == "projection_heads":
        mods["image_head"] = get("proj_head_image_infonce")
    if comp in ("projection_heads", "text_to_brain", "qformer", "brain_to_text"):
        mods["ae"] = get("autoencoder")
    if comp == "text_to_brain":
        mods["mse_head"] = get("proj_head_text_mse")
    if comp in ("qformer", "brain_to_text"):
        from _hf_pins import released_qformer

        nq = released_qformer("canonical", "cpu", projection_temp=PAPER_TEMP, canonical_basis="network", use_canonical_projection=True)
        n_params["neuro_qformer"] = count_params(nq)
        mods["qformer"] = cast(nq, precision, device)
    if comp == "brain_to_text":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from neurovlm.resources.loaders import NEURO_QWEN_REPO_ID
        from _hf_pins import hf_revision

        rev = hf_revision(NEURO_QWEN_REPO_ID)
        tok = AutoTokenizer.from_pretrained(NEURO_QWEN_REPO_ID, revision=rev, use_fast=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        # load straight into the target dtype so the CPU peak is not inflated by an fp32 copy (int8 quantizes from fp32)
        lm = AutoModelForCausalLM.from_pretrained(NEURO_QWEN_REPO_ID, revision=rev, dtype=wdtype)
        n_params["neuro_qwen3"] = count_params(lm)
        lm_norm = lm.get_input_embeddings().weight.detach().float().norm(dim=1).mean().item()  # (bf16 weights: <0.5% difference)
        mods["lm"] = cast(lm, precision, device)
    run_dev = torch.device("cpu") if precision == "int8" else device
    index = None
    if comp == "contrastive_retrieval":  # shared embeddings of every PubMed map (built with fp32 heads, stored in run precision)
        ih, enc_lat = load_model("proj_head_image_infonce").eval(), load_pubmed_latents()[0]
        with torch.no_grad():
            index = F.normalize(ih(enc_lat), dim=1).to(run_dev, wdtype)
        res["index_size"] = int(len(index))
        del ih, enc_lat
    if precision == "int8":  # quantize_dynamic returns an int8 copy: drop the fp32 models it was made from
        lm = nq = None  # noqa: F841
    release_memory()
    res["load_seconds"] = time.perf_counter() - t0
    res["params"] = n_params
    res["params_total"] = int(sum(n_params.values()))
    res["weights_mb"] = sum(module_bytes(m.specter if k == "specter" else m) for k, m in mods.items()) / 2 ** 20
    res["rss_after_load_mb"] = proc.memory_info().rss / 2 ** 20
    res["cpu_load_peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # loading, incl. int8's fp32 source
    if device.type == "cuda":
        torch.cuda.synchronize()
        res["gpu_mem_after_load_mb"] = torch.cuda.memory_allocated() / 2 ** 20

    def enc_text(tx):
        return mods["specter"](list(tx)).to(run_dev)

    def make_fn(bs: int):
        tx = texts[:bs]
        x = maps[:bs].to(run_dev, wdtype)
        if comp == "text_encoder":
            return lambda: enc_text(tx)
        if comp == "projection_heads":
            e = torch.randn(bs, 768).to(run_dev, wdtype)
            return lambda: (mods["text_head"](F.normalize(e, dim=1)), mods["image_head"](mods["ae"].encoder(x)))
        if comp == "contrastive_retrieval":
            return lambda: torch.topk(F.normalize(mods["text_head"](F.normalize(enc_text(tx), dim=1)), dim=1) @ index.T, k=10, dim=1)
        if comp == "text_to_brain":
            return lambda: torch.sigmoid(mods["ae"].decoder(mods["mse_head"](F.normalize(enc_text(tx), dim=1))))
        if comp == "qformer":
            return lambda: mods["qformer"](mods["ae"].encoder(x), basis="network", projection_temp=PAPER_TEMP, use_canonical_projection=True)
        if comp == "brain_to_text":
            lm = mods["lm"]
            prefix = tok("[NETWORK]", add_special_tokens=False, return_tensors="pt")["input_ids"].to(run_dev)
            T = int(cfg["max_new_tokens"])

            def fn():
                vis = mods["qformer"](mods["ae"].encoder(x), basis="network", projection_temp=PAPER_TEMP, use_canonical_projection=True)
                vis = match_lm_token_norm(vis.float(), lm_norm).to(next(lm.parameters()).dtype if precision != "int8" else torch.float32)
                pe = lm.get_input_embeddings()(prefix).to(vis.dtype).expand(vis.shape[0], -1, -1)
                emb = torch.cat([vis, pe], dim=1)
                return lm.generate(inputs_embeds=emb, attention_mask=torch.ones(emb.shape[:2], dtype=torch.long, device=run_dev),
                                   max_new_tokens=T, min_new_tokens=T, num_beams=int(cfg["b2t_beams"]), do_sample=False,
                                   repetition_penalty=1.25, no_repeat_ngram_size=4, pad_token_id=tok.pad_token_id or tok.eos_token_id)
            return fn
        raise ValueError(comp)

    res["batches"] = {}
    for bs in cfg["batch_sizes"]:
        fn = make_fn(bs)
        if precision in ("bf16", "fp16"):  # weights are cast; autocast handles modules that upcast internally (e.g. the Q-Former)
            inner = fn

            def fn(inner=inner):
                with torch.autocast(device_type=run_dev.type, dtype=wdtype):
                    return inner()
        sync = (lambda: torch.cuda.synchronize()) if run_dev.type == "cuda" else (lambda: None)
        if run_dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        res["peak_rss_reset"] = reset_peak_rss()
        with torch.inference_mode():
            for _ in range(int(cfg["warmup"])):
                fn()
            sync()
            times = []
            for _ in range(int(cfg["reps"])):
                t = time.perf_counter()
                out = fn()
                sync()
                times.append(time.perf_counter() - t)
        b = {"batch_size": bs, "latency_median_s": float(np.median(times)), "latency_p90_s": float(np.quantile(times, 0.9)),
             "latency_min_s": float(np.min(times)), "reps": len(times)}
        b["per_sample_ms"] = 1000 * b["latency_median_s"] / bs
        if comp == "brain_to_text":
            n_new = int(out.shape[1])
            b["new_tokens"] = n_new
            b["tokens_per_s"] = bs * n_new / b["latency_median_s"]
        b["cpu_peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # this batch size (counter reset above)
        if run_dev.type == "cuda":
            b["gpu_peak_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
            b["gpu_peak_reserved_mb"] = torch.cuda.max_memory_reserved() / 2 ** 20
        res["batches"][str(bs)] = b
    res["cpu_peak_rss_mb"] = max(b["cpu_peak_rss_mb"] for b in res["batches"].values())
    return res


def prepare_inputs(path: Path, n: int = 16) -> None:
    """Cache the first ``n`` test papers ("title [SEP] abstract" and map) as the benchmark inputs."""
    import torch

    if path.exists():
        return
    test = np.sort(split_pmids()["test"])[:n]
    rows = load_pubmed_text_rows(test)
    texts = [f"{a} [SEP] {b}" for a, b in zip(rows["name"], rows["description"])]
    maps = load_pubmed_maps(test)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"texts": texts, "maps": maps.half(), "pmids": test}, path)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=HERE / "outputs", help="Output directory.")
    p.add_argument("--device", default="auto", help="auto (cuda if available, then cpu), cuda or cpu.")
    p.add_argument("--cpu-only", action="store_true")
    p.add_argument("--components", default=",".join(COMPONENTS))
    p.add_argument("--precisions", default=",".join(PRECISIONS))
    p.add_argument("--batch-sizes", default="1,16")
    p.add_argument("--reps", type=int, default=None, help="Timed calls (default 20; 5 for brain_to_text and on CPU; 2 with --smoke).")
    p.add_argument("--warmup", type=int, default=None, help="Untimed calls (default 3; 1 for brain_to_text, on CPU and with --smoke).")
    p.add_argument("--max-new-tokens", type=int, default=None, help="Forced generation length (default 128; 8 with --smoke).")
    p.add_argument("--b2t-beams", type=int, default=1)
    p.add_argument("--threads", type=int, default=None, help="Torch CPU threads (default: PyTorch default).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--require-idle-gpu", action="store_true", help="Abort if other processes use the GPU.")
    p.add_argument("--smoke", action="store_true", help="Quick check: 2 reps, 8 new tokens.")
    p.add_argument("--force", action="store_true", help="Rerun configurations that already have a result.")
    p.add_argument("--worker", type=Path, default=None, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.worker is not None:
        cfg = json.loads(args.worker.read_text())
        seed_all(cfg.get("seed", 0))
        try:
            res = worker(cfg)
        except Exception as exc:  # recorded in the result, the sweep goes on
            import traceback

            res = {"component": cfg["component"], "precision": cfg["precision"], "device": cfg["device"],
                   "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-2000:]}
        write_json(cfg["result"], res)
        return 0

    import pandas as pd
    import torch

    out = args.out
    (out / "raw").mkdir(parents=True, exist_ok=True)
    timings = Timings(out / "timings.json")
    if args.cpu_only or args.device == "cpu" or not torch.cuda.is_available():
        devices = ["cpu"]
    elif args.device == "cuda":
        devices = ["cuda"]
    else:
        devices = ["cuda", "cpu"]
    hw = hardware_info(check_gpu="cuda" in devices)
    if "cuda" not in devices and (out / "hardware.json").exists():  # CPU-only rerun: keep the GPU check of the cached GPU rows
        hw = {**read_json(out / "hardware.json"), **hw}
    write_json(out / "hardware.json", hw)
    if "cuda" in devices and hw.get("gpu_other_compute_apps"):
        msg = f"GPU is not idle: {hw['gpu_other_compute_apps']} -- GPU numbers would be contaminated"
        if args.require_idle_gpu:
            log.error(msg)
            return 2
        log.warning(msg)
    max_new = args.max_new_tokens or (8 if args.smoke else 128)
    prepare_inputs(out / "cache" / "inputs.pt")
    comps = [c for c in args.components.split(",") if c]
    precs = [x for x in args.precisions.split(",") if x]
    bss = [int(b) for b in args.batch_sizes.split(",")]
    t_all = time.perf_counter()
    for dev in devices:
        for comp in comps:
            for prec in precs:
                tag = f"{comp}__{prec}__{dev}"
                res_path = out / "raw" / f"{tag}.json"
                if prec == "int8" and dev != "cpu":
                    continue  # dynamic quantization kernels are CPU-only
                if prec == "fp16" and dev == "cpu":
                    continue
                if res_path.exists() and not args.force:
                    r = read_json(res_path)
                    if "error" not in r and r.get("smoke") == args.smoke:
                        log.info("%s cached", tag)
                        continue
                heavy = comp == "brain_to_text" or dev == "cpu"
                cfg = {"component": comp, "precision": prec, "device": dev, "batch_sizes": bss, "max_new_tokens": max_new,
                       "b2t_beams": args.b2t_beams, "threads": args.threads, "seed": args.seed, "inputs": str(out / "cache" / "inputs.pt"),
                       "result": str(res_path), "reps": args.reps or (2 if args.smoke else 5 if heavy else 20),
                       "warmup": args.warmup if args.warmup is not None else (1 if (args.smoke or heavy) else 3)}
                cfg_path = out / "raw" / f"{tag}.cfg.json"
                write_json(cfg_path, cfg)
                log.info("running %s", tag)
                with timings.time(tag):
                    proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(cfg_path)], capture_output=True, text=True)
                if proc.returncode != 0 or not res_path.exists():
                    write_json(res_path, {"component": comp, "precision": prec, "device": dev, "error": f"worker exit {proc.returncode}",
                                          "stderr": proc.stderr[-3000:]})
                r = read_json(res_path)
                r["smoke"] = args.smoke
                write_json(res_path, r)
                if "error" in r:
                    log.warning("%s failed: %s", tag, r["error"])
                else:
                    log.info("%s | params %.1fM | bs1 %.1f ms | peak %s", tag, r["params_total"] / 1e6, r["batches"][str(bss[0])]["per_sample_ms"],
                             f"{r['batches'][str(bss[0])].get('gpu_peak_mb', r.get('cpu_peak_rss_mb')):.0f} MB")
    rows = []
    for f in sorted((out / "raw").glob("*.json")):
        if f.name.endswith(".cfg.json"):
            continue
        r = read_json(f)
        base = {k: r.get(k) for k in ("component", "precision", "device", "params_total", "weights_mb", "load_seconds", "rss_baseline_mb",
                                       "rss_after_load_mb", "cpu_load_peak_rss_mb", "cpu_peak_rss_mb", "gpu_mem_after_load_mb",
                                       "torch_threads", "error", "smoke")}
        if "batches" not in r:
            rows.append(base)
            continue
        for bs, b in r["batches"].items():
            rows.append({**base, **b})
    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)
    write_json(out / "metrics.json", {"smoke": args.smoke, "devices": devices, "components": comps, "precisions": precs, "batch_sizes": bss,
                                      "max_new_tokens": max_new, "b2t_beams": args.b2t_beams, "hardware": hw, "packages": env_packages(),
                                      "n_configs": int(len(df)), "n_errors": int(df["error"].notna().sum()) if "error" in df else 0,
                                      "wall_seconds_this_run": time.perf_counter() - t_all,
                                      "runtime_estimate_gpu_seconds": {"total": 900, "note": "GPU sweep (6 components x fp32/bf16/fp16 x bs 1,16; "
                                                                       "brain_to_text 5 reps x 128 tokens) ~5 min + CPU sweep ~10-25 min"}})
    cols = [c for c in ("component", "precision", "device", "batch_size", "params_total", "per_sample_ms", "tokens_per_s", "gpu_peak_mb", "cpu_peak_rss_mb", "error") if c in df]
    with pd.option_context("display.width", 220, "display.max_colwidth", 60):
        print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
