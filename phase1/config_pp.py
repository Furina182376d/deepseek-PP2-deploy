"""
PP=2 multi-node constants for GLM5.2 profiling.
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
    OUTPUT_LEN,
)

# ---- PP / TP / multi-node ----
MODEL_PATH = "/data/model/GLM-5.2-FP8"
PP_SIZE = 2
TP_SIZE_PER_PP = 4         # each PP stage uses 4 GPUs on the node
WORLD_SIZE_PP = PP_SIZE * TP_SIZE_PER_PP  # 8 — total workers across both nodes

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


def validate_config():
    """Verify that environment-supplied values are consistent with config."""
    errors = []
    if LOCAL_WORLD_SIZE != TP_SIZE_PER_PP:
        errors.append(
            f"LOCAL_WORLD_SIZE={LOCAL_WORLD_SIZE} != TP_SIZE_PER_PP={TP_SIZE_PER_PP}"
        )
    if WORLD_SIZE != PP_SIZE * TP_SIZE_PER_PP:
        errors.append(
            f"WORLD_SIZE={WORLD_SIZE} != PP_SIZE*TP_SIZE_PER_PP="
            f"{PP_SIZE * TP_SIZE_PER_PP}"
        )
    if WORLD_SIZE != NNODES * LOCAL_WORLD_SIZE:
        errors.append(
            f"WORLD_SIZE={WORLD_SIZE} != NNODES*LOCAL_WORLD_SIZE="
            f"{NNODES * LOCAL_WORLD_SIZE}"
        )
    if errors:
        raise RuntimeError(
            "Config validation failed:\n  " + "\n  ".join(errors)
        )
    return True
