"""
Core PP multi-node inference — one process per GPU, orchestrated by torchrun.

In external_launcher mode, torchrun launches ``nproc_per_node`` processes on
each node.  Every process creates its own ``LLM`` instance (one GPU worker)
and they coordinate via NCCL for TP within a PP stage and P2P for cross-stage
activations.  All processes call ``llm.generate()`` redundantly — only the
**leader** (global rank 0) collects and logs results.

当前配置 (config_pp.py): Kimi-K3, PP=3 x TP=8, 三节点 (192.168.0.224/225/226)。
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
    profiler_config = None
    if cfg.TORCH_PROFILER_DIR:
        profiler_config = {
            "profiler": "torch",
            "torch_profiler_dir": cfg.TORCH_PROFILER_DIR,
            "torch_profiler_with_stack": False,
            "torch_profiler_with_flops": False,
            "torch_profiler_use_gzip": False,
            "torch_profiler_dump_cuda_time_total": True,
            "torch_profiler_record_shapes": True,
            "torch_profiler_with_memory": False,
            "warmup_iterations": cfg.TORCH_PROFILER_WARMUP_ITERS,
            "active_iterations": cfg.TORCH_PROFILER_ACTIVE_ITERS,
            "wait_iterations": cfg.TORCH_PROFILER_WAIT_ITERS,
        }
        if cfg.IS_LEADER:
            print(
                "Torch profiler enabled: "
                f"dir={cfg.TORCH_PROFILER_DIR} "
                f"wait={cfg.TORCH_PROFILER_WAIT_ITERS} "
                f"warmup={cfg.TORCH_PROFILER_WARMUP_ITERS} "
                f"active={cfg.TORCH_PROFILER_ACTIVE_ITERS}"
            )

    llm = LLM(
        model=cfg.MODEL_PATH,
        pipeline_parallel_size=cfg.PP_SIZE,
        tensor_parallel_size=tp_per_pp,
        max_model_len=cfg.MAX_MODEL_LEN,
        # K3 含 Mamba 层, 默认 1024 > 单卡可容纳的 Mamba cache block 数 (900),
        # 必须显式降低, 否则 CUDA graph 捕获检查直接失败
        max_num_seqs=cfg.MAX_NUM_SEQS,
        gpu_memory_utilization=cfg.GPU_MEM_UTIL,
        trust_remote_code=True,
        kv_cache_dtype=cfg.KV_CACHE_DTYPE,
        enable_flashinfer_autotune=cfg.ENABLE_FLASHINFER_AUTOTUNE,
        enforce_eager=cfg.ENFORCE_EAGER,
        disable_log_stats=False,
        distributed_executor_backend=cfg.DISTRIBUTED_EXECUTOR_BACKEND,
        # Multi-node settings forwarded via EngineArgs
        nnodes=cfg.NNODES,
        node_rank=cfg.NODE_RANK,
        master_addr=cfg.MASTER_ADDR,
        master_port=cfg.MASTER_PORT,
        # 超时: distributed_timeout_seconds 只管 NCCL 组; gloo (CPU) 组默认
        # 仅 1800s, 三机权重加载不均衡 >30min 时先加载完的 worker 会在
        # is_in_the_same_node 的 barrier 上超时崩溃 (Kimi_deploy 已踩坑),
        # 必须显式加大 cpu_distributed_timeout_seconds
        distributed_timeout_seconds=10800,
        cpu_distributed_timeout_seconds=10800,
        profiler_config=profiler_config,
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
    for warmup_index in range(cfg.NUM_WARMUPS):
        if cfg.IS_LEADER:
            print(f"Warmup {warmup_index + 1}/{cfg.NUM_WARMUPS}")
        llm.generate(
            [make_prompt(cfg.CONTEXT_LENGTHS[0])],
            SamplingParams(temperature=0, max_tokens=cfg.OUTPUT_LEN, ignore_eos=True),
        )

    # ---- Leader: prepare result logging ----
    if cfg.IS_LEADER:
        ensure_results_dir()
        open_csv(tp=tp_per_pp)  # reuse same CSV column schema

        print(
            f"\n{'run':>3s} {'ctx':>6s} | {'prompt_tok':>5s} {'out_tok':>3s} | "
            f"{'TTFT_ms':>8s} | {'prefill_t/s':>10s} | {'decode_t/s':>10s} | "
            f"{'total_ms':>8s} | {'GPU_avg_MB':>10s}"
        )
        print("-" * 85)

    # ---- All ranks: profile loop ----
    # The profiler request is deliberately isolated from the normal result
    # loop: its trace overhead would invalidate the TPOT measurements.
    if cfg.TORCH_PROFILER_DIR:
        if cfg.IS_LEADER:
            print("Starting worker CUDA profiler request")
        llm.start_profile("moe_decode")
        llm.generate(
            [make_prompt(cfg.CONTEXT_LENGTHS[0])],
            SamplingParams(temperature=0, max_tokens=cfg.OUTPUT_LEN, ignore_eos=True),
        )
        llm.stop_profile()
        if cfg.IS_LEADER:
            print("Worker CUDA profiler request complete")

    if not cfg.TORCH_PROFILER_ONLY:
        for ctx_len in cfg.CONTEXT_LENGTHS:
            prompt = make_prompt(ctx_len)
            sp = SamplingParams(temperature=0, max_tokens=cfg.OUTPUT_LEN, ignore_eos=True)

            for repeat_index in range(cfg.NUM_REPEATS):
                result = _timed_generate(llm, prompt, sp, ctx_len)

                if not cfg.IS_LEADER:
                    continue
                # GPU memory snapshot (leader's local GPUs)
                used, _ = gpu_mem(tp_per_pp)
                avg_mem = sum(used) / len(used)

                ttft_ms = result["ttft_ms"]
                prefill_tps = result["prefill_tps"]
                decode_tps = result["decode_tps"]
                total_ms = result["total_ms"]

                print(
                    f"{repeat_index + 1:>3d} {ctx_len:>6d} | "
                    f"{result['prompt_tok']:>5d} {result['out_tok']:>3d} | "
                    f"{ttft_ms:>8.1f} | {prefill_tps:>10.0f} | {decode_tps:>10.1f} | "
                    f"{total_ms:>8.0f} | {avg_mem:>10.0f}"
                )

                # Build result row (tagged with PP info via the tp field convention)
                row = {
                    "tp": f"PP{cfg.PP_SIZE}_TP{tp_per_pp}",
                    "repeat": repeat_index + 1,
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
