"""
Core PP=2 multi-node inference — one process per GPU, orchestrated by torchrun.

In external_launcher mode, torchrun launches ``nproc_per_node`` processes on
each node.  Every process creates its own ``LLM`` instance (one GPU worker)
and they coordinate via NCCL for TP within a PP stage and P2P for cross-stage
activations.  All processes call ``llm.generate()`` redundantly — only the
**leader** (global rank 0) collects and logs results.
"""

import gc
import os
import sys
import time

import torch
import torch.distributed as dist
from vllm import LLM, SamplingParams

# ---- project-root path (for cross-phase imports) ----
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

import phase1.config_pp as cfg
from phase0.gpu_utils import gpu_mem
from phase0.prompt_utils import make_prompt
from phase0.results_utils import (
    ALL_RESULTS,
    close_csv,
    ensure_results_dir,
    open_csv,
    write_csv_row,
)


def _timed_generate(llm: LLM, prompt: str, sp: SamplingParams, ctx_len: int):
    """
    Run a single generate call with wall-clock timing.
    All ranks call this; timing is only meaningful on the leader.
    """
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = llm.generate([prompt], sp)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    out = outputs[0]
    prompt_tok = len(out.prompt_token_ids)
    out_tok = len(out.outputs[0].token_ids)
    total_ms = (t1 - t0) * 1000

    # --- Extract vLLM timing metrics ---
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
        first_decode_ms = tpot_ms
        prefill_ms = max(0.0, ttft_ms - first_decode_ms)
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
        "total_ms": total_ms,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "decode_tps": decode_tps,
        "prefill_ms": prefill_ms,
        "prefill_tps": prefill_tps,
    }


def run_pp():
    """Main PP profiling entry point — called by every torchrun process."""

    tp_per_pp = cfg.TP_SIZE_PER_PP  # 8

    if cfg.IS_LEADER:
        print(
            f"\n{'='*60}\n"
            f"Loading model  PP={cfg.PP_SIZE}  TP_per_stage={tp_per_pp}  "
            f"nodes={cfg.NNODES}  backend={cfg.DISTRIBUTED_EXECUTOR_BACKEND}\n"
            f"Global rank={cfg.GLOBAL_RANK}  Local rank={cfg.LOCAL_RANK}  "
            f"World size={cfg.WORLD_SIZE}\n"
            f"max_model_len={cfg.MAX_MODEL_LEN}\n"
            f"{'='*60}"
        )

    # ---- Pre-flight: verify model path exists (all ranks) ----
    import os as _os
    _model_path = cfg.MODEL_PATH
    if not _os.path.exists(_model_path):
        raise FileNotFoundError(
            f"Model path does not exist on this node: {_model_path}\n"
            f"  Node rank: {cfg.NODE_RANK}  Global rank: {cfg.GLOBAL_RANK}\n"
            f"  Host: {_os.uname().nodename}\n"
            f"  Ensure the model directory exists and is accessible on ALL nodes."
        )
    _config_json = _os.path.join(_model_path, "config.json")
    if not _os.path.exists(_config_json):
        raise FileNotFoundError(
            f"config.json not found in model directory on this node: {_config_json}\n"
            f"  Node rank: {cfg.NODE_RANK}  Global rank: {cfg.GLOBAL_RANK}\n"
            f"  Host: {_os.uname().nodename}\n"
            f"  The model directory exists but may be incomplete."
        )
    if cfg.IS_LEADER:
        print(f"Model path verified: {_model_path}")

    # ---- All ranks: create LLM instance ----
    llm = LLM(
        model=cfg.MODEL_PATH,
        pipeline_parallel_size=cfg.PP_SIZE,
        tensor_parallel_size=tp_per_pp,
        max_model_len=cfg.MAX_MODEL_LEN,
        gpu_memory_utilization=cfg.GPU_MEM_UTIL,
        trust_remote_code=True,
        kv_cache_dtype=cfg.KV_CACHE_DTYPE,
        enforce_eager=True,           # disable CUDA graphs to save GPU memory
        max_num_seqs=64,              # limit concurrent sequences → smaller KV cache pool
        disable_log_stats=False,
        distributed_executor_backend=cfg.DISTRIBUTED_EXECUTOR_BACKEND,
        # Multi-node settings forwarded via EngineArgs
        nnodes=cfg.NNODES,
        node_rank=cfg.NODE_RANK,
        master_addr=cfg.MASTER_ADDR,
        master_port=cfg.MASTER_PORT,
    )

    # ---- All ranks: synchronize after model load ----
    if dist.is_initialized():
        dist.barrier()
        if cfg.IS_LEADER:
            print("All ranks synchronized after model load (barrier passed).")

    if cfg.IS_LEADER:
        print("Model loaded.\n")
        # Report GPU memory on this node (leader can see all local GPUs)
        used_init, total_init = gpu_mem(tp_per_pp)
        for i in range(tp_per_pp):
            print(
                f"  GPU {i}: {used_init[i]:.0f} MB / {total_init[i]:.0f} MB "
                f"({used_init[i] / total_init[i] * 100:.1f}%)"
            )

    # ---- All ranks: warmup ----
    llm.generate(
        [make_prompt(cfg.CONTEXT_LENGTHS[0])],
        SamplingParams(temperature=0, max_tokens=cfg.OUTPUT_LEN, ignore_eos=True),
    )

    # ---- Leader: prepare result logging ----
    if cfg.IS_LEADER:
        ensure_results_dir()
        open_csv(tp=tp_per_pp)  # reuse same CSV column schema

        print(
            f"\n{'ctx':>6s} | {'prompt_tok':>5s} {'out_tok':>3s} | "
            f"{'TTFT_ms':>8s} | {'prefill_t/s':>10s} | {'decode_t/s':>10s} | "
            f"{'total_ms':>8s} | {'GPU_avg_MB':>10s}"
        )
        print("-" * 85)

    # ---- All ranks: profile loop ----
    for ctx_len in cfg.CONTEXT_LENGTHS:
        prompt = make_prompt(ctx_len)
        sp = SamplingParams(temperature=0, max_tokens=cfg.OUTPUT_LEN, ignore_eos=True)

        result = _timed_generate(llm, prompt, sp, ctx_len)

        if cfg.IS_LEADER:
            # GPU memory snapshot (leader's local GPUs)
            used, _ = gpu_mem(tp_per_pp)
            avg_mem = sum(used) / len(used)

            ttft_ms = result["ttft_ms"]
            prefill_tps = result["prefill_tps"]
            decode_tps = result["decode_tps"]
            total_ms = result["total_ms"]

            print(
                f"{ctx_len:>6d} | "
                f"{result['prompt_tok']:>5d} {result['out_tok']:>3d} | "
                f"{ttft_ms:>8.1f} | {prefill_tps:>10.0f} | {decode_tps:>10.1f} | "
                f"{total_ms:>8.0f} | {avg_mem:>10.0f}"
            )

            # Build result row (tagged with PP info via the tp field convention)
            row = {
                "tp": f"PP{cfg.PP_SIZE}_TP{tp_per_pp}",
                "tp_size": tp_per_pp,       # actual TP count (int) — used by report generator
                "context_length": ctx_len,
                "prompt_tokens": result["prompt_tok"],
                "output_tokens": result["out_tok"],
                "ttft_ms": round(ttft_ms, 2),
                "prefill_ms": round(result["prefill_ms"], 2),
                "prefill_tps": round(prefill_tps, 1),
                "decode_tps": round(decode_tps, 1),
                "tpot_ms": round(result["tpot_ms"], 2),
                "total_ms": round(total_ms, 1),
                "avg_gpu_mem_mb": round(avg_mem, 1),
            }
            # Per-GPU memory columns (leader's local GPUs)
            for i in range(tp_per_pp):
                row[f"gpu{i}_mem_mb"] = used[i]

            write_csv_row(row)
            ALL_RESULTS.append(row)

    # ---- Cleanup ----
    if cfg.IS_LEADER:
        close_csv()

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)
