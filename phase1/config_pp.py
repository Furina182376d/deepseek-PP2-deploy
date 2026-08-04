"""
PP=2 multi-node constants for DeepSeek V4 Flash profiling.
"""
import os

# Keep a shared reference to the base config for model / cache settings.
from config import (  # noqa: F401 — re-export for convenience
    CONTEXT_LENGTHS,
    GPU_MEM_UTIL,
    KV_CACHE_DTYPE,
    MAX_MODEL_LEN,
    MODEL_PATH,
    OUTPUT_LEN,
)

# ---- PP / TP / multi-node ----
PP_SIZE = 2
TP_SIZE_PER_PP = 4         # each PP stage uses 4 GPUs for tensor parallelism
TOTAL_TP = PP_SIZE * TP_SIZE_PER_PP  # 8 — total logical TP across both nodes

NNODES = 2
MASTER_ADDR = "192.168.0.63"
MASTER_PORT = 29500
DISTRIBUTED_EXECUTOR_BACKEND = "external_launcher"

# ---- torchrun supplies these via env vars ----
GLOBAL_RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", TP_SIZE_PER_PP))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", TOTAL_TP))
NODE_RANK = GLOBAL_RANK // LOCAL_WORLD_SIZE
IS_LEADER = GLOBAL_RANK == 0
