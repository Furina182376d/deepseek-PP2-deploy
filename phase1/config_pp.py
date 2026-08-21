"""
PP=3 multi-node constants for Kimi-K3 profiling.
"""
import os
import sys

# ---- project-root path (for cross-phase imports) ----
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

# Keep a shared reference to the base config for generic benchmark settings.
from phase0.config import (  # noqa: F401 — re-export for convenience
    GPU_MEM_UTIL,
    OUTPUT_LEN,
)

# K3 专用: MAX_MODEL_LEN=32768, 32k 的 prompt 会放不下 256 个输出 token,
# 所以 context 扫描止步 16384 (如需 32768 请把 MAX_MODEL_LEN 提到 65536,
# 但要注意 KV cache 显存余量)
# Decode-only observation run: single short context, so the dmon window is
# almost entirely decode. Restore the full sweep afterwards:
#   CONTEXT_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384]
CONTEXT_LENGTHS = [512]

# ---- 模型 (三台机器均为 /data/models/Kimi-K3) ----
MODEL_PATH = "/data/models/Kimi-K3"

# ---- PP / TP / multi-node ----
PP_SIZE = 3
TP_SIZE_PER_PP = 8         # each PP stage uses 8 GPUs on the node
WORLD_SIZE_PP = PP_SIZE * TP_SIZE_PER_PP  # 24 — total workers across 3 nodes

NNODES = 3
# 三机内网: aliyun1=.224 (rank0/master), aliyun2=.225 (rank1), aliyun3=.226 (rank2)
MASTER_ADDR = "192.168.0.224"
MASTER_PORT = 29500
DISTRIBUTED_EXECUTOR_BACKEND = "external_launcher"

# ---- 模型相关 (K3 专用, 覆盖 phase0.config 的 DeepSeek 默认值) ----
# K3 是 MXFP4 量化 MoE (1.5TB 压缩权重), 每卡 62.5GB 权重; 32k 上下文下
# KV/激活余量 ~23GB, 保守取 32768
MAX_MODEL_LEN = 32768
# KV cache 交给 vllm 自动选择 (K3 MLA 支持 fp8_ds_mla, auto 最稳)
KV_CACHE_DTYPE = "auto"

# FlashInfer autotune 会在 warmup 阶段对 24 个 rank 做 gloo 广播 (基准测试前)。
# 三节点环境下若某节点 flashinfer 版本不一致, 该 rank 会在广播前崩溃并拖垮全局
# (Connection closed by peer)。 这是纯优化项, 关闭后 flashinfer 走启发式选择。
# 定位问题期间置 False, 修复后可改回 True。
ENABLE_FLASHINFER_AUTOTUNE = False

# K3 含 Mamba (SSM) 层: 每个 decode 序列占一个 Mamba cache block。 PP=3 时
# stage-1 节点显存只够 900 个 block, 而 vLLM 默认 max_num_seqs=1024, 会在
# CUDA graph 捕获检查直接失败 (max_num_seqs exceeds available Mamba cache
# blocks)。 profiling 每次只发 1 条序列, 512 足够且有安全余量。
MAX_NUM_SEQS = 512

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
