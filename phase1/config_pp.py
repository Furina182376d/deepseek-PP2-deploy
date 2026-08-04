"""
PP=2 multi-node constants for DeepSeek V4 Flash profiling.
"""
import os
import sys

# ---- project-root path (for cross-phase imports) ----
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

# Keep a shared reference to the base config for model / cache settings.
from phase0.config import (  # noqa: F401 — re-export for convenience
    CONTEXT_LENGTHS,
    GPU_MEM_UTIL,
    KV_CACHE_DTYPE,
    MAX_MODEL_LEN,
    MODEL_PATH,
    OUTPUT_LEN,
)

# ---- PP / TP / multi-node ----
PP_SIZE = 2
TP_SIZE_PER_PP = 8         # each PP stage uses all 8 GPUs on the node
WORLD_SIZE_PP = PP_SIZE * TP_SIZE_PER_PP  # 16 — total workers across both nodes

NNODES = 2
MASTER_ADDR = "192.168.0.63"
MASTER_PORT = 29500
DISTRIBUTED_EXECUTOR_BACKEND = "external_launcher"

# ---- torchrun supplies these via env vars ----
GLOBAL_RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", TP_SIZE_PER_PP))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", WORLD_SIZE_PP))
NODE_RANK = GLOBAL_RANK // LOCAL_WORLD_SIZE
IS_LEADER = GLOBAL_RANK == 0
