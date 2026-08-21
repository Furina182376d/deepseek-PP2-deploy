#!/usr/bin/env python3
"""
Profile Kimi-K3 inference with vLLM — PP=3 TP=8 across three nodes.

Launched via torchrun on each node (see ``launch_pp.sh``).  Every process runs
the same script; only global rank 0 collects and logs results.

Usage (indirect — via launch_pp.sh)::

    torchrun --nnodes=3 --nproc_per_node=8 --node_rank=0 \\
        --master_addr=192.168.0.224 --master_port=29500 \\
        profile_dsv4_pp.py

The heavy-lifting functions live in:

    config_pp.py     — PP-specific constants + torchrun env detection
    run_pp.py        — core PP inference (all ranks)
    results_utils.py — CSV / JSON / report logging (leader only)
"""

import json
import os
import sys
import time
import traceback

# ---- Force local file resolution for HuggingFace libraries ----
# Must be set BEFORE any vLLM/transformers import, otherwise the Hub
# validator rejects local paths like /data/model/... on newer versions.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ---- project-root path (for cross-phase imports) ----
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

# Install encrypted PP P2P hooks BEFORE vLLM imports.
# No-op if VLLM_COMM_PSK is not set in the environment.
from phase3.comm_crypto import install_encrypted_pp_hooks

install_encrypted_pp_hooks()

from phase0.results_utils import (
    ALL_RESULTS,
    RESULTS_DIR,
    TIMESTAMP,
    write_report_and_summary,
)
from phase1.run_pp import run_pp

import phase1.config_pp as cfg

if __name__ == "__main__":
    print(
        f"[rank {cfg.GLOBAL_RANK}/{cfg.WORLD_SIZE}  "
        f"local {cfg.LOCAL_RANK}/{cfg.LOCAL_WORLD_SIZE}  "
        f"pp={cfg.PP_SIZE}  tp_per_pp={cfg.TP_SIZE_PER_PP}  "
        f"backend={cfg.DISTRIBUTED_EXECUTOR_BACKEND}]"
    )

    # ---- Validate config consistency (only leader reports) ----
    cfg.validate_config()
    if cfg.IS_LEADER:
        print("Config validation passed.")

    try:
        run_pp()
    except Exception as e:
        print(f"\n[rank {cfg.GLOBAL_RANK}] PP run failed: {e}", file=sys.stderr)
        # 首次失败必须打完整 traceback —— 这是定位真凶的唯一途径
        traceback.print_exc()

        # ---- Force cleanup before retry ----
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(5)  # wait for NCCL connections to fully release

        # vLLM 引擎初始化失败后, 权重/NCCL 缓冲/CUDA graph 的显存通常不会随
        # 异常释放; 此时在同一进程内重试必然死在 request_memory 的显存检查上,
        # 报出误导性的 "Free memory ... less than desired"。 重试前先看真实
        # 空闲显存, 不足预算就直接以原始错误退出 (torchrun 层可以重启)。
        free_gb, total_gb = torch.cuda.mem_get_info(cfg.LOCAL_RANK)
        free_gb, total_gb = free_gb / 2**30, total_gb / 2**30
        if cfg.IS_LEADER:
            print(
                f"[rank {cfg.GLOBAL_RANK}] GPU {cfg.LOCAL_RANK}: "
                f"free {free_gb:.1f} / {total_gb:.1f} GiB after cleanup"
            )
        if free_gb < cfg.GPU_MEM_UTIL * total_gb:
            print(
                f"[rank {cfg.GLOBAL_RANK}] free {free_gb:.1f} GiB < "
                f"gpu_memory_utilization budget {cfg.GPU_MEM_UTIL * total_gb:.1f} GiB — "
                f"retry would OOM again; aborting. See traceback above for the real error.",
                file=sys.stderr,
            )
            raise  # re-raise the ORIGINAL first-attempt error

        # Retry with smaller max_model_len (OOM / 长上下文兜底)
        try:
            if cfg.IS_LEADER:
                print(f"Retrying with max_model_len=16384...")
            old = cfg.MAX_MODEL_LEN
            cfg.MAX_MODEL_LEN = 16384
            run_pp()
            cfg.MAX_MODEL_LEN = old
        except Exception as e2:
            print(f"[rank {cfg.GLOBAL_RANK}] PP retry still failed: {e2}", file=sys.stderr)
            raise

    # ---- Leader: finalize results ----
    if cfg.IS_LEADER:
        metadata = {
            "model_path": cfg.MODEL_PATH,
            "timestamp": TIMESTAMP,
            "pp_size": cfg.PP_SIZE,
            "tp_per_pp": cfg.TP_SIZE_PER_PP,
            "nnodes": cfg.NNODES,
            "distributed_executor_backend": cfg.DISTRIBUTED_EXECUTOR_BACKEND,
            "max_model_len": cfg.MAX_MODEL_LEN,
            "gpu_memory_utilization": cfg.GPU_MEM_UTIL,
            "kv_cache_dtype": cfg.KV_CACHE_DTYPE,
            "output_len": cfg.OUTPUT_LEN,
        }
        json_path = os.path.join(RESULTS_DIR, "full_results.json")
        with open(json_path, "w") as f:
            json.dump({"metadata": metadata, "results": ALL_RESULTS}, f, indent=2)

        write_report_and_summary(title="PP=3 TP=8 Kimi-K3 三节点部署",
                                 metadata=metadata)
