#!/usr/bin/env python3
"""
Profile DeepSeek V4 with TP=4 using REAL LongBench datasets as prompts.

Usage:
    cd /home/tjy/codebases/deepseek_deploy
    bash phase0/run.sh    # if you want the original synthetic benchmark
    python phase2/profile_dsv4_real.py   # this script for real workloads

This mirrors phase0/profile_dsv4.py but replaces synthetic prompts with
pre-downloaded LongBench data.
"""

import csv
import gc
import json
import os
import sys
import time

import torch
from vllm import LLM, SamplingParams

# ---- project-root path ----
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

import phase0.config as config
from phase0.gpu_utils import gpu_mem
from phase0.results_utils import (
    ALL_RESULTS,
    RESULTS_DIR,
    TIMESTAMP,
    ensure_results_dir,
    write_report_and_summary,
)
from phase2.data_loader import list_available_datasets, load_longbench

# ---- Benchmark knobs ----
OUTPUT_LEN = 256        # override from config if needed
MAX_SAMPLES = 20        # per dataset (cap for runtime)


def _timed_generate(llm: LLM, prompt: str, sp: SamplingParams) -> dict:
    """Single generate call with wall-clock + vLLM per-request metrics."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = llm.generate([prompt], sp)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    out = outputs[0]
    prompt_tok = len(out.prompt_token_ids)
    out_tok = len(out.outputs[0].token_ids)

    m = out.metrics
    if m and m.first_token_latency is not None:
        ttft_ms = m.first_token_latency * 1000
        if m.num_generation_tokens > 1 and m.last_token_ts and m.first_token_ts:
            tpot_sec = (m.last_token_ts - m.first_token_ts) / (m.num_generation_tokens - 1)
            tpot_ms = tpot_sec * 1000
            decode_tps = 1.0 / tpot_sec
        else:
            tpot_ms = 0.0
            decode_tps = 0.0
        prefill_ms = max(0.0, ttft_ms - tpot_ms)
        prefill_tps = prompt_tok / (prefill_ms / 1000) if prefill_ms > 0 else 0.0
    else:
        ttft_ms = 0.0
        tpot_ms = 0.0
        decode_tps = 0.0
        prefill_ms = 0.0
        prefill_tps = 0.0

    return {
        "prompt_tok": prompt_tok,
        "out_tok": out_tok,
        "total_ms": (t1 - t0) * 1000,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "decode_tps": decode_tps,
        "prefill_ms": prefill_ms,
        "prefill_tps": prefill_tps,
    }


def run_real_benchmark(tp: int = 4):
    tag = f"TP{tp}_longbench"

    print(f"\n{'='*60}")
    print(f"Loading model  TP={tp}  max_model_len={config.MAX_MODEL_LEN}")
    print(f"{'='*60}")

    llm = LLM(
        model=config.MODEL_PATH,
        tensor_parallel_size=tp,
        max_model_len=config.MAX_MODEL_LEN,
        gpu_memory_utilization=config.GPU_MEM_UTIL,
        trust_remote_code=True,
        kv_cache_dtype=config.KV_CACHE_DTYPE,
        enforce_eager=False,
        disable_log_stats=False,
    )
    print("Model loaded.\n")

    # GPU state after load
    used_init, total_init = gpu_mem(tp)
    for i in range(tp):
        print(f"  GPU {i}: {used_init[i]:.0f} MB / {total_init[i]:.0f} MB ({used_init[i]/total_init[i]*100:.1f}%)")

    # Warmup
    warmup_prompt = "Hello, this is a warmup request for CUDA graph capture."
    llm.generate([warmup_prompt], SamplingParams(temperature=0, max_tokens=1, ignore_eos=True))
    torch.cuda.synchronize()

    # Discover datasets
    available = list_available_datasets()
    tasks = [name for name, status in available.items() if status == "local"]
    print(f"\nFound {len(tasks)} real dataset(s): {tasks}")

    ensure_results_dir()

    # Open our own CSV (different columns from the synthetic benchmark)
    csv_path = os.path.join(RESULTS_DIR, f"tp{tp}_longbench.csv")
    csv_fh = open(csv_path, "w", newline="")
    csv_fields = [
        "tp", "task", "sample_idx",
        "context_length", "prompt_tokens", "output_tokens",
        "ttft_ms", "prefill_ms", "prefill_tps", "decode_tps", "tpot_ms", "total_ms",
        "avg_gpu_mem_mb",
    ] + [f"gpu{i}_mem_mb" for i in range(tp)]
    csv_w = csv.DictWriter(csv_fh, fieldnames=csv_fields)
    csv_w.writeheader()
    csv_fh.flush()

    header = (
        f"\n{'task':>20s} {'ctx':>6s} {'out':>4s} | "
        f"{'TTFT_ms':>8s} | {'prefill_t/s':>10s} | {'decode_t/s':>10s} | "
        f"{'total_ms':>8s} | {'GPU_avg_MB':>10s}"
    )
    print(header)
    print("-" * 95)

    results: list[dict] = []
    tokenizer = llm.get_tokenizer()
    max_prompt_len = config.MAX_MODEL_LEN - OUTPUT_LEN

    for task_name in tasks:
        items = load_longbench(task_name)
        items = items[:MAX_SAMPLES]  # cap per dataset
        print(f"\n  Task: {task_name}  ({len(items)} samples)")

        skipped = 0
        for idx, item in enumerate(items):
            context = item.get("context", "") or item.get("input", "")
            question = item.get("input", "") or item.get("question", "")
            prompt = f"{context}\n\n{question}"

            # Skip samples that exceed max_model_len
            tok_len = len(tokenizer.encode(prompt))
            if tok_len > max_prompt_len:
                skipped += 1
                continue

            sp = SamplingParams(
                temperature=0,
                max_tokens=OUTPUT_LEN,
                ignore_eos=True,
            )
            
            # Per-sample warmup (CUDA graph / JIT for this shape)
            if idx == 0:
                llm.generate([prompt], SamplingParams(temperature=0, max_tokens=1, ignore_eos=True))
                torch.cuda.synchronize()

            r = _timed_generate(llm, prompt, sp)

            used, _ = gpu_mem(tp)
            avg_mem = sum(used) / len(used)

            row = {
                "tp": tag,
                "task": task_name,
                "sample_idx": idx,
                "context_length": r["prompt_tok"],
                "prompt_tokens": r["prompt_tok"],
                "output_tokens": r["out_tok"],
                "ttft_ms": round(r["ttft_ms"], 2),
                "prefill_ms": round(r["prefill_ms"], 2),
                "prefill_tps": round(r["prefill_tps"], 1),
                "decode_tps": round(r["decode_tps"], 1),
                "tpot_ms": round(r["tpot_ms"], 2),
                "total_ms": round(r["total_ms"], 1),
                "avg_gpu_mem_mb": round(avg_mem, 1),
            }
            for i in range(tp):
                row[f"gpu{i}_mem_mb"] = used[i]

            results.append(row)
            csv_w.writerow(row)
            csv_fh.flush()
            ALL_RESULTS.append(row)

            print(
                f"{task_name:>20s} {r['prompt_tok']:>6d} {r['out_tok']:>4d} | "
                f"{r['ttft_ms']:>8.1f} | {r['prefill_tps']:>10.0f} | "
                f"{r['decode_tps']:>10.1f} | {r['total_ms']:>8.0f} | {avg_mem:>10.0f}"
            )

        if skipped:
            print(f"  (skipped {skipped} samples exceeding max_model_len={config.MAX_MODEL_LEN})")

    csv_fh.close()

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    # Write metadata + full results JSON
    metadata = {
        "model_path": config.MODEL_PATH,
        "timestamp": TIMESTAMP,
        "max_model_len": config.MAX_MODEL_LEN,
        "gpu_memory_utilization": config.GPU_MEM_UTIL,
        "kv_cache_dtype": config.KV_CACHE_DTYPE,
        "output_len": OUTPUT_LEN,
        "max_samples_per_dataset": MAX_SAMPLES,
        "datasets": tasks,
    }
    json_path = os.path.join(RESULTS_DIR, "full_results.json")
    with open(json_path, "w") as f:
        json.dump({"metadata": metadata, "results": ALL_RESULTS}, f, indent=2)

    write_report_and_summary()

    return results


if __name__ == "__main__":
    # Determine GPU devices from env or default to 4,5,6,7 (matching run.sh)
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    print(f"CUDA_VISIBLE_DEVICES = {cuda_devices}")
    run_real_benchmark(tp=4)
