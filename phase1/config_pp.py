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
    OUTPUT_LEN as _BASE_OUTPUT_LEN,
)

# PP experiments can override the decode window without changing phase0.
OUTPUT_LEN = int(os.environ.get("PP_OUTPUT_LEN", _BASE_OUTPUT_LEN))
NUM_WARMUPS = int(os.environ.get("PP_NUM_WARMUPS", "1"))
NUM_REPEATS = int(os.environ.get("PP_NUM_REPEATS", "1"))

# Optional worker-side PyTorch CUDA profiler. Keep disabled for normal latency
# runs; when set, the output directory must be an absolute path shared by the
# node-local worker processes (for example /home/tjy/kimi_bench/moe_profile).
TORCH_PROFILER_DIR = os.environ.get("PP_TORCH_PROFILER_DIR", "")
TORCH_PROFILER_WARMUP_ITERS = int(os.environ.get("PP_PROFILE_WARMUP_ITERS", "2"))
TORCH_PROFILER_ACTIVE_ITERS = int(os.environ.get("PP_PROFILE_ACTIVE_ITERS", "8"))
TORCH_PROFILER_WAIT_ITERS = int(os.environ.get("PP_PROFILE_WAIT_ITERS", "0"))
TORCH_PROFILER_ONLY = os.environ.get("PP_PROFILE_ONLY", "0") == "1"
ENFORCE_EAGER = os.environ.get("PP_ENFORCE_EAGER", "0") == "1"

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
TP_SIZE_PER_PP = int(os.environ.get("PP_TP_SIZE", "8"))
WORLD_SIZE_PP = PP_SIZE * TP_SIZE_PER_PP  # 24 — total workers across 3 nodes

# vLLM context/expert parallel knobs.  CP partitions work within the TP group;
# EP changes MoE expert placement while preserving the PP topology.
PREFILL_CP_SIZE = int(os.environ.get("PP_PREFILL_CP_SIZE", "1"))
DECODE_CP_SIZE = int(os.environ.get("PP_DECODE_CP_SIZE", "1"))
ENABLE_EXPERT_PARALLEL = os.environ.get("PP_ENABLE_EXPERT_PARALLEL", "0") == "1"

_BATCH_RAW = os.environ.get("PP_BATCH_SIZES", os.environ.get("PP_BATCH_SIZE", "1"))
BATCH_SIZES = tuple(int(x) for x in _BATCH_RAW.split(",") if x.strip())

# Optional non-uniform pipeline split. vLLM reads VLLM_PP_LAYER_PARTITION
# before model construction; keep the parsed value here for validation and
# experiment metadata. Kimi-K3 has 93 transformer layers.
NUM_HIDDEN_LAYERS = 93
_PP_PARTITION_RAW = os.environ.get("VLLM_PP_LAYER_PARTITION", "")
PP_LAYER_PARTITION = (
    tuple(int(value) for value in _PP_PARTITION_RAW.split(","))
    if _PP_PARTITION_RAW
    else None
)
PP_EXPERIMENT_ID = os.environ.get("PP_EXPERIMENT_ID", "baseline_31_31_31")

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
MAX_NUM_SEQS = int(os.environ.get("PP_MAX_NUM_SEQS", str(max(BATCH_SIZES))))

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
    if OUTPUT_LEN <= 1:
        errors.append(f"PP_OUTPUT_LEN must be greater than 1, got {OUTPUT_LEN}")
    if NUM_WARMUPS < 0:
        errors.append(f"PP_NUM_WARMUPS must be non-negative, got {NUM_WARMUPS}")
    if NUM_REPEATS <= 0:
        errors.append(f"PP_NUM_REPEATS must be positive, got {NUM_REPEATS}")
    if MAX_NUM_SEQS <= 0:
        errors.append(f"PP_MAX_NUM_SEQS must be positive, got {MAX_NUM_SEQS}")
    if not BATCH_SIZES or any(batch <= 0 for batch in BATCH_SIZES):
        errors.append(f"PP_BATCH_SIZES must contain positive integers, got {BATCH_SIZES}")
    if max(BATCH_SIZES) > MAX_NUM_SEQS:
        errors.append(
            f"max(PP_BATCH_SIZES)={max(BATCH_SIZES)} exceeds PP_MAX_NUM_SEQS={MAX_NUM_SEQS}"
        )
    for name, value in (("PP_PREFILL_CP_SIZE", PREFILL_CP_SIZE),
                        ("PP_DECODE_CP_SIZE", DECODE_CP_SIZE)):
        if value <= 0:
            errors.append(f"{name} must be positive, got {value}")
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
    if PP_LAYER_PARTITION is not None:
        if len(PP_LAYER_PARTITION) != PP_SIZE:
            errors.append(
                f"VLLM_PP_LAYER_PARTITION has {len(PP_LAYER_PARTITION)} entries; "
                f"expected PP_SIZE={PP_SIZE}"
            )
        if any(layers <= 0 for layers in PP_LAYER_PARTITION):
            errors.append("VLLM_PP_LAYER_PARTITION entries must all be positive")
        if sum(PP_LAYER_PARTITION) != NUM_HIDDEN_LAYERS:
            errors.append(
                f"VLLM_PP_LAYER_PARTITION sums to {sum(PP_LAYER_PARTITION)}; "
                f"expected {NUM_HIDDEN_LAYERS} model layers"
            )
    if errors:
        raise RuntimeError(
            "Config validation failed:\n  " + "\n  ".join(errors)
        )
    return True
