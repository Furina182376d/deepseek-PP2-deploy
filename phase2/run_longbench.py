#!/usr/bin/env python3
"""
LongBench real-workload benchmark — reuses phase0's run_prompt_benchmark.

Usage:
    cd /home/tjy/codebases/deepseek_deploy
    python phase2/run_longbench.py
"""

import gc
import json
import os
import sys

import torch
from vllm import SamplingParams

# ---- project-root path ----
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

import phase0.config as config
from phase0.gpu_utils import gpu_mem
from phase0.run_tp import _load_model, _print_gpu_mem, _timed_generate
from phase0.results_utils import (
    ALL_RESULTS,
    RESULTS_DIR,
    TIMESTAMP,
    ensure_results_dir,
    open_csv,
    write_csv_row,
    close_csv,
    write_report_and_summary,
)
from phase2.data_loader import list_available_datasets, load_longbench

# ---- Knobs ----
OUTPUT_LEN = 256
MAX_SAMPLES = 20


def iter_longbench_prompts(tokenizer):
    """Yield {"label": task, "prompt": text, "task": task} for every
    valid LongBench sample."""
    max_prompt_len = config.MAX_MODEL_LEN - OUTPUT_LEN
    available = list_available_datasets()
    tasks = [name for name, status in available.items() if status == "local"]

    for task_name in sorted(tasks):
        for item in load_longbench(task_name)[:MAX_SAMPLES]:
            context = item.get("context", "") or item.get("input", "")
            question = item.get("input", "") or item.get("question", "")
            prompt_text = f"{context}\n\n{question}"

            # Skip samples that exceed max model length
            if len(tokenizer.encode(prompt_text)) > max_prompt_len:
                continue

            yield {"label": task_name, "prompt": prompt_text, "task": task_name}


if __name__ == "__main__":
    tp = 4
    tag = f"tp{tp}_longbench"

    llm = _load_model(tp)
    _print_gpu_mem(tp)

    # Global warmup
    llm.generate(
        ["Hello, this is a warmup request for CUDA graph capture."],
        SamplingParams(temperature=0, max_tokens=1, ignore_eos=True),
    )
    torch.cuda.synchronize()

    # Collect prompts (need tokenizer for length filtering)
    tokenizer = llm.get_tokenizer()
    prompts = list(iter_longbench_prompts(tokenizer))
    print(f"\nLoaded {len(prompts)} valid prompts across datasets")

    # Group by task — prompts arrive sorted by task from iter_longbench_prompts
    from itertools import groupby
    tasks: list[tuple[str, list[dict]]] = [
        (task, list(group))
        for task, group in groupby(prompts, key=lambda p: p["task"])
    ]

    ensure_results_dir()
    open_csv(tp=tp, tag=tag, extra_fields=["task"])

    header = (
        f"\n{'label':>20s} {'ctx':>6s} {'out':>4s} | "
        f"{'TTFT_ms':>8s} | {'prefill_t/s':>10s} | {'decode_t/s':>10s} | "
        f"{'total_ms':>8s} | {'GPU_avg_MB':>10s}"
    )
    print(header)
    print("-" * 95)

    for task_name, task_prompts in tasks:
        for i, p in enumerate(task_prompts):
            prompt_text = p["prompt"]

            # Per-task first-sample warmup: absorb CUDA graph / JIT for this
            # bucket.  Remaining samples in the same task run without warmup
            # so any bucket not covered by the first sample will still show
            # graph-capture overhead in the timed measurement.
            if i == 0:
                llm.generate(
                    [prompt_text],
                    SamplingParams(temperature=0, max_tokens=1, ignore_eos=True),
                )
                torch.cuda.synchronize()

            sp = SamplingParams(
                temperature=0,
                max_tokens=OUTPUT_LEN,
                ignore_eos=True,
            )
            r = _timed_generate(llm, prompt_text, sp)

            used, _ = gpu_mem(tp)
            avg_mem = sum(used) / len(used)

            row = {
                "tp": tag,
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
            for f in (["task"]):
                row[f] = p.get(f, "")
            for i_gpu in range(tp):
                row[f"gpu{i_gpu}_mem_mb"] = used[i_gpu]

            write_csv_row(row)
            ALL_RESULTS.append(row)

            print(
                f"{p['label']:>20s} {r['prompt_tok']:>6d} {r['out_tok']:>4d} | "
                f"{r['ttft_ms']:>8.1f} | {r['prefill_tps']:>10.0f} | "
                f"{r['decode_tps']:>10.1f} | {r['total_ms']:>8.0f} | {avg_mem:>10.0f}"
            )

    close_csv()

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    # Metadata + JSON
    available = list_available_datasets()
    task_names = [name for name, status in available.items() if status == "local"]
    metadata = {
        "model_path": config.MODEL_PATH,
        "timestamp": TIMESTAMP,
        "max_model_len": config.MAX_MODEL_LEN,
        "gpu_memory_utilization": config.GPU_MEM_UTIL,
        "kv_cache_dtype": config.KV_CACHE_DTYPE,
        "output_len": OUTPUT_LEN,
        "max_samples_per_dataset": MAX_SAMPLES,
        "datasets": task_names,
    }
    json_path = os.path.join(RESULTS_DIR, "full_results.json")
    with open(json_path, "w") as f:
        json.dump({"metadata": metadata, "results": ALL_RESULTS}, f, indent=2)

    write_report_and_summary()
