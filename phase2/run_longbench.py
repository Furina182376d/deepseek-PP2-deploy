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
from phase0.run_tp import _load_model, _print_gpu_mem, run_prompt_benchmark
from phase0.results_utils import (
    ALL_RESULTS,
    RESULTS_DIR,
    TIMESTAMP,
    write_aggregate_summary,
    write_report_and_summary,
)
from phase2.data_loader import list_available_datasets, load_longbench

# ---- Knobs ----
OUTPUT_LEN = 256
# Samples per dataset; None = all.  Override via LONGBENCH_MAX_SAMPLES env.
MAX_SAMPLES = (
    int(os.environ["LONGBENCH_MAX_SAMPLES"])
    if "LONGBENCH_MAX_SAMPLES" in os.environ
    else 20
)


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

    # Run
    run_prompt_benchmark(
        llm, prompts, tp,
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
    }
    json_path = os.path.join(RESULTS_DIR, "full_results.json")
    with open(json_path, "w") as f:
        json.dump({"metadata": metadata, "results": ALL_RESULTS}, f, indent=2)

    write_report_and_summary()
    write_aggregate_summary()
