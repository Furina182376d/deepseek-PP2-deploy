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
import time
import torch

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

from vllm import LLM
from phase1 import config_pp as cfg
from phase2.data_loader import list_available_datasets, load_longbench
from vllm import SamplingParams


def timed_generate(llm, prompt, sampling_params):
    torch.cuda.synchronize()
    start = time.perf_counter()
    outputs = llm.generate([prompt], sampling_params)
    torch.cuda.synchronize()
    total_ms = (time.perf_counter() - start) * 1000
    out = outputs[0]
    metrics = getattr(out, "metrics", None)
    fields = {}
    if metrics is not None:
        for name in ("first_token_latency", "first_token_ts", "last_token_ts",
                     "num_generation_tokens"):
            value = getattr(metrics, name, None)
            if value is not None:
                fields[name] = value
    first_latency = fields.get("first_token_latency")
    first_ts = fields.get("first_token_ts")
    last_ts = fields.get("last_token_ts")
    n_tokens = len(out.outputs[0].token_ids)
    if first_latency is not None:
        ttft_ms = first_latency * 1000
    else:
        ttft_ms = 0.0
    if first_ts is not None and last_ts is not None and n_tokens > 1:
        tpot_ms = (last_ts - first_ts) * 1000 / (n_tokens - 1)
    else:
        tpot_ms = 0.0
    return {
        "prompt_tok": len(out.prompt_token_ids),
        "out_tok": n_tokens,
        "total_ms": total_ms,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "decode_tps": 1000.0 / tpot_ms if tpot_ms else 0.0,
        "metrics_fields": fields,
    }


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
    repeats = int(os.environ.get("LONG_BENCH_REPEATS", "3"))
    requested_tasks = os.environ.get("LONG_BENCH_TASKS", "qmsum,gov_report")
    tasks = [t.strip() for t in requested_tasks.split(",") if t.strip()]
    available = list_available_datasets()
    tasks = [t for t in tasks if available.get(t) == "local"]
    rows = []
    for task in tasks:
        items = load_longbench(task)[:limit]
        for index, item in enumerate(items):
            context = item.get("context", "") or item.get("input", "")
            question = item.get("input", "") or item.get("question", "")
            prompt = f"{context}\n\n{question}"
            sampling_params = SamplingParams(temperature=0, max_tokens=output_len,
                                              ignore_eos=True)
            # One throwaway request removes the first-request JIT from measured rows.
            timed_generate(llm, prompt, sampling_params)
            for repeat in range(repeats):
                result = timed_generate(llm, prompt, sampling_params)
                row = {
                    "task": task,
                    "sample": index,
                    "repeat": repeat,
                    "prompt_tokens": result["prompt_tok"],
                    "output_tokens": result["out_tok"],
                    "ttft_ms": round(result["ttft_ms"], 2),
                    "tpot_ms": round(result["tpot_ms"], 2),
                    "decode_tps": round(result["decode_tps"], 2),
                    "total_ms": round(result["total_ms"], 1),
                    "metrics_fields": result["metrics_fields"],
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
