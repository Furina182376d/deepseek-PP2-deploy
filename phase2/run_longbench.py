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
import time

import torch
from vllm import SamplingParams

# ---- project-root path ----
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

import phase0.config as config
from phase0.run_tp import _load_model, _print_gpu_mem, run_batch_benchmark, run_prompt_benchmark
from phase0.results_utils import (
    ALL_RESULTS,
    RESULTS_DIR,
    TIMESTAMP,
    write_aggregate_summary,
    write_batch_summary,
    write_report_and_summary,
)
from phase2.data_loader import list_available_datasets, load_longbench

# ---- Knobs ----
OUTPUT_LEN = 256
# Samples per dataset; None = all.  Override via LONGBENCH_MAX_SAMPLES env
# (int, or ""/"all" for every sample in the dataset).
def _parse_max_samples() -> int | None:
    v = os.environ.get("LONGBENCH_MAX_SAMPLES", "20")
    if v == "" or v.lower() == "all":
        return None
    return int(v)


MAX_SAMPLES = _parse_max_samples()
# Batch mode: 0 = sequential per-sample timing (legacy default);
# N > 0 = chunked batch of N per generate call; -1 = all prompts at once.
BATCH_SIZE = int(os.environ.get("LONGBENCH_BATCH_SIZE", "0"))
# Parallelism: LONGBENCH_TP (default 4), LONGBENCH_EP (default 0 = 纯 TP;
# >0 → enable_expert_parallel，vLLM 自动 EP = world//TP).
LONGBENCH_TP = int(os.environ.get("LONGBENCH_TP", "4"))
LONGBENCH_EP = int(os.environ.get("LONGBENCH_EP", "0"))
# SJF-style submission: sort prompts by length before batching.
SORT_BY_LEN = os.environ.get("LONGBENCH_SORT_BY_LEN") == "1"


def iter_longbench_prompts(tokenizer):
    """Yield {"label": task, "prompt": text, "prompt_token_ids": ids,
    "tokenize_ms": ms, "task": task} for every valid LongBench sample.

    Tokenizes ONCE here (the encode is needed for the length check
    anyway) and reuses the ids for both the per-sample warmup and the
    timed generation — Python-side tokenization never enters the timed
    path (same finding as profile_detailed_timing: ~2ms/K was hidden in
    TTFT/wall when passing text).
    """
    max_prompt_len = config.MAX_MODEL_LEN - OUTPUT_LEN
    available = list_available_datasets()
    tasks = [name for name, status in available.items() if status == "local"]

    for task_name in sorted(tasks):
        for item in load_longbench(task_name)[:MAX_SAMPLES]:
            context = item.get("context", "") or item.get("input", "")
            question = item.get("input", "") or item.get("question", "")
            prompt_text = f"{context}\n\n{question}"

            t0 = time.perf_counter()
            ids = tokenizer.encode(prompt_text)
            tokenize_ms = (time.perf_counter() - t0) * 1000

            # Skip samples that exceed max model length
            if len(ids) > max_prompt_len:
                continue

            yield {
                "label": task_name,
                "prompt": prompt_text,
                "prompt_token_ids": ids,
                "tokenize_ms": tokenize_ms,
                "task": task_name,
            }


if __name__ == "__main__":
    tp = LONGBENCH_TP
    ep_on = LONGBENCH_EP > 0
    world = tp * LONGBENCH_EP if ep_on else tp
    tag = f"tp{tp}_longbench" if not ep_on else f"tp{tp}_ep{LONGBENCH_EP}_longbench"

    llm = _load_model(tp, enable_expert_parallel=ep_on)
    _print_gpu_mem(world)

    # Global warmup
    llm.generate(
        ["Hello, this is a warmup request for CUDA graph capture."],
        SamplingParams(temperature=0, max_tokens=1, ignore_eos=True),
    )
    torch.cuda.synchronize()

    # Collect prompts (need tokenizer for length filtering)
    tokenizer = llm.get_tokenizer()
    prompts = list(iter_longbench_prompts(tokenizer))
    if SORT_BY_LEN:
        prompts.sort(key=lambda p: len(p["prompt_token_ids"]))
        print("Prompts sorted by length (SJF-style submission)")
    print(f"\nLoaded {len(prompts)} valid prompts across datasets")

    # Run
    batch_stats = None
    if BATCH_SIZE != 0:
        tag = f"{tag}_b{BATCH_SIZE}" if BATCH_SIZE > 0 else f"{tag}_ball"
        _, batch_stats = run_batch_benchmark(
            llm, prompts, world,
            tag=tag,
            batch_size=BATCH_SIZE,
            output_len=OUTPUT_LEN,
            extra_fields=["task"],
        )
    else:
        run_prompt_benchmark(
            llm, prompts, world,
            tag=tag,
            warmup_per_sample=True,
            output_len=OUTPUT_LEN,
            extra_fields=["task"],
        )

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    # Metadata + JSON
    available = list_available_datasets()
    tasks = [name for name, status in available.items() if status == "local"]
    metadata = {
        "model_path": config.MODEL_PATH,
        "timestamp": TIMESTAMP,
        "max_model_len": config.MAX_MODEL_LEN,
        "gpu_memory_utilization": config.GPU_MEM_UTIL,
        "kv_cache_dtype": config.KV_CACHE_DTYPE,
        "output_len": OUTPUT_LEN,
        "max_samples_per_dataset": MAX_SAMPLES,
        "datasets": tasks,
        "pre_tokenized": True,
        "batch_size": BATCH_SIZE,
        "tp": tp,
        "expert_parallel": ep_on,
        "sort_by_len": SORT_BY_LEN,
    }
    json_path = os.path.join(RESULTS_DIR, "full_results.json")
    with open(json_path, "w") as f:
        json.dump({"metadata": metadata, "results": ALL_RESULTS}, f, indent=2)

    if batch_stats is not None:
        write_batch_summary(ALL_RESULTS, batch_stats)
    else:
        write_report_and_summary()
        write_aggregate_summary()
