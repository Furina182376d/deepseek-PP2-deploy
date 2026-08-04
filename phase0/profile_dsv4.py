#!/usr/bin/env python3
"""
Profile DeepSeek V4 Flash inference with vLLM — TP=4 vs TP=8.

This is the main entry point.  The heavy-lifting functions live in:

    config.py        — shared constants (model path, context lengths, etc.)
    prompt_utils.py  — synthetic prompt generation
    gpu_utils.py     — GPU memory monitoring
    run_tp.py        — core TP inference run
    results_utils.py — CSV / JSON / report logging
"""

import json
import os
import sys

# ---- project-root path (for cross-phase imports) ----
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

import phase0.config as config
from phase0.results_utils import ALL_RESULTS, RESULTS_DIR, TIMESTAMP, write_report_and_summary
from phase0.run_tp import run_tp

if __name__ == "__main__":
    for tp in [4, 8]:
        try:
            run_tp(tp)
        except Exception as e:
            print(f"\nTP={tp} failed: {e}")
            # Retry with smaller max_model_len
            try:
                print(f"Retrying TP={tp} with max_model_len=32768...")
                old = config.MAX_MODEL_LEN
                config.MAX_MODEL_LEN = 32768
                run_tp(tp)
                config.MAX_MODEL_LEN = old
            except Exception as e2:
                print(f"TP={tp} still failed: {e2}")

    # Write full_results.json
    metadata = {
        "model_path": config.MODEL_PATH,
        "timestamp": TIMESTAMP,
        "max_model_len": config.MAX_MODEL_LEN,
        "gpu_memory_utilization": config.GPU_MEM_UTIL,
        "kv_cache_dtype": config.KV_CACHE_DTYPE,
        "output_len": config.OUTPUT_LEN,
    }
    json_path = os.path.join(RESULTS_DIR, "full_results.json")
    with open(json_path, "w") as f:
        json.dump({"metadata": metadata, "results": ALL_RESULTS}, f, indent=2)

    write_report_and_summary()
