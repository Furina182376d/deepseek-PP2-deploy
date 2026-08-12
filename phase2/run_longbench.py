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
from phase0.run_tp import _load_model, _print_gpu_mem, run_prompt_benchmark
from phase0.results_utils import (
    ALL_RESULTS,
    RESULTS_DIR,
    TIMESTAMP,
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
    }
    json_path = os.path.join(RESULTS_DIR, "full_results.json")
    with open(json_path, "w") as f:
        json.dump({"metadata": metadata, "results": ALL_RESULTS}, f, indent=2)

    write_report_and_summary()
