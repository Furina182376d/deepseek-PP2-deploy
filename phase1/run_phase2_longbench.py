#!/usr/bin/env python3
"""Run a small, reproducible LongBench workload on the PP=3/TP=8 deployment.

Every torchrun rank executes the same requests; only global rank 0 writes the
CSV/JSON summary.  Use ``LONG_BENCH_SAMPLES`` to cap samples per task and
``LONG_BENCH_OUTPUT`` to control generated tokens.
"""
import json
import os
import sys
import traceback

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

from vllm import LLM
from phase1 import config_pp as cfg
from phase2.data_loader import list_available_datasets, load_longbench
from phase2.run_bench import _timed_generate
from vllm import SamplingParams


def main():
    if not cfg.IS_LEADER:
        # All ranks still create LLM and execute requests below.
        pass
    llm = LLM(
        model=cfg.MODEL_PATH,
        pipeline_parallel_size=cfg.PP_SIZE,
        tensor_parallel_size=cfg.TP_SIZE_PER_PP,
        max_model_len=cfg.MAX_MODEL_LEN,
        max_num_seqs=cfg.MAX_NUM_SEQS,
        gpu_memory_utilization=cfg.GPU_MEM_UTIL,
        trust_remote_code=True,
        kv_cache_dtype=cfg.KV_CACHE_DTYPE,
        enable_flashinfer_autotune=cfg.ENABLE_FLASHINFER_AUTOTUNE,
        enforce_eager=False,
        distributed_executor_backend=cfg.DISTRIBUTED_EXECUTOR_BACKEND,
        nnodes=cfg.NNODES,
        node_rank=cfg.NODE_RANK,
        master_addr=cfg.MASTER_ADDR,
        master_port=cfg.MASTER_PORT,
        distributed_timeout_seconds=10800,
        cpu_distributed_timeout_seconds=10800,
    )

    # Keep the workload representative but bounded: one deterministic sample
    # per task by default, with a longer generation than the decode probe.
    limit = int(os.environ.get("LONG_BENCH_SAMPLES", "1"))
    output_len = int(os.environ.get("LONG_BENCH_OUTPUT", "128"))
    tasks = ["narrativeqa", "qmsum", "gov_report", "hotpotqa"]
    available = list_available_datasets()
    tasks = [t for t in tasks if available.get(t) == "local"]
    rows = []
    for task in tasks:
        items = load_longbench(task)[:limit]
        for index, item in enumerate(items):
            context = item.get("context", "") or item.get("input", "")
            question = item.get("input", "") or item.get("question", "")
            prompt = f"{context}\n\n{question}"
            result = _timed_generate(
                llm,
                prompt,
                SamplingParams(temperature=0, max_tokens=output_len, ignore_eos=True),
            )
            row = {
                "task": task,
                "sample": index,
                "prompt_tokens": result["prompt_tok"],
                "output_tokens": result["out_tok"],
                "ttft_ms": round(result["ttft_ms"], 2),
                "prefill_ms": round(result["prefill_ms"], 2),
                "prefill_tps": round(result["prefill_tps"], 1),
                "tpot_ms": round(result["tpot_ms"], 2),
                "decode_tps": round(result["decode_tps"], 2),
                "total_ms": round(result["total_ms"], 1),
            }
            rows.append(row)
            if cfg.IS_LEADER:
                print(json.dumps(row, ensure_ascii=False), flush=True)
    if cfg.IS_LEADER:
        out = os.environ.get("LONG_BENCH_RESULT", "phase2_longbench_results.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"config": {"samples_per_task": limit, "output_len": output_len},
                       "results": rows}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
