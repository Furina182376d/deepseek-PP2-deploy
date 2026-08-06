#!/usr/bin/env python3
"""
Profile GLM5.2 inference with vLLM — PP=2 TP=8 across two nodes.

Launched via torchrun on each node (see ``launch_pp.sh``).  Every process runs
the same script; only global rank 0 collects and logs results.

Usage (indirect — via launch_pp.sh)::

    torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \\
        --master_addr=192.168.0.63 --master_port=29500 \\
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

        # ---- Force cleanup before retry ----
        # run_pp()'s finally block already ran del llm + gc + empty_cache,
        # but we do an extra round here to be safe.  Keep the process group
        # alive — retry needs it (external_launcher uses torchrun's group).
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(5)  # wait for NCCL connections to fully release

        # Retry with smaller max_model_len
        try:
            if cfg.IS_LEADER:
                print(f"Retrying with max_model_len=32768...")
            old = cfg.MAX_MODEL_LEN
            cfg.MAX_MODEL_LEN = 32768
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

        write_report_and_summary(title="PP=2 TP=8 GLM5.2 两节点部署")
