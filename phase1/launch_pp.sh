#!/bin/bash
# ---------------------------------------------------------------------------
# Launch PP=3 TP=8 Kimi-K3 profiling across THREE nodes via torchrun.
#
# Usage (run on EACH of the three nodes, with different <node_rank>):
#   Baseline: ./launch_pp.sh <node_rank>
#   Rebalance: ./launch_pp.sh <node_rank> 30,32,31 pp_30_32_31
#
# Prerequisites:
#   - Passwordless SSH between the three nodes.
#   - The ``vllm`` conda environment (vllm 0.27.1, supports Kimi-K3) on all nodes.
#   - Identical codebase path on both nodes.
#   - /data/models/Kimi-K3 present on all three nodes.
# ---------------------------------------------------------------------------
set -eo pipefail

NODE_RANK="${1:?Usage: $0 <node_rank> [layer_partition] [experiment_id]}"
PP_LAYER_PARTITION="${2:-${VLLM_PP_LAYER_PARTITION:-}}"
PP_EXPERIMENT_ID="${3:-${PP_EXPERIMENT_ID:-baseline_31_31_31}}"
MASTER_ADDR="192.168.0.224"
MASTER_PORT=29500
NNODES=3
NPROC_PER_NODE=8   # one process per GPU — 8 H20s used per node

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BASE="${HOME}/miniconda3"

# ---------------------------------------------------------------------------
# Activate conda environment
# ---------------------------------------------------------------------------
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate vllm

# ---------------------------------------------------------------------------
# NCCL settings (from prior working multi-node deployment on these nodes)
# ---------------------------------------------------------------------------
export NCCL_IB_DISABLE=1                 # no InfiniBand — use TCP over eth0
export NCCL_SOCKET_IFNAME="=eth0"
export GLOO_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_CUMEM_HOST_ENABLE=0          # cuMem unavailable, fallback to /dev/shm
export NCCL_TIMEOUT=1800                 # 30 min timeout — avoid hanging forever
export NCCL_ASYNC_ERROR_HANDLING=1       # async error handling to detect dead peers
export TORCH_DISTRIBUTED_TIMEOUT=1800    # PyTorch distributed timeout

# Force HuggingFace libraries to use local files only (skip Hub repo-id validation)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Required by external_launcher executor (deterministic scheduling)
export VLLM_ENABLE_V1_MULTIPROCESSING=0

if [ -n "${PP_LAYER_PARTITION}" ]; then
    if ! [[ "${PP_LAYER_PARTITION}" =~ ^[1-9][0-9]*,[1-9][0-9]*,[1-9][0-9]*$ ]]; then
        echo "Invalid PP layer partition: ${PP_LAYER_PARTITION}" >&2
        exit 2
    fi
    IFS=',' read -r PP_LAYERS_0 PP_LAYERS_1 PP_LAYERS_2 <<< "${PP_LAYER_PARTITION}"
    if [ "$((PP_LAYERS_0 + PP_LAYERS_1 + PP_LAYERS_2))" -ne 93 ]; then
        echo "PP layer partition must sum to 93: ${PP_LAYER_PARTITION}" >&2
        exit 2
    fi
    export VLLM_PP_LAYER_PARTITION="${PP_LAYER_PARTITION}"
fi
export PP_EXPERIMENT_ID

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
cd "${SCRIPT_DIR}"

echo "============================================"
echo "Node rank  : ${NODE_RANK} / ${NNODES}"
echo "Master     : ${MASTER_ADDR}:${MASTER_PORT}"
echo "Procs/node : ${NPROC_PER_NODE}"
echo "Total procs: $((NNODES * NPROC_PER_NODE))"
echo "Partition  : ${PP_LAYER_PARTITION:-31,31,31}"
echo "Experiment : ${PP_EXPERIMENT_ID}"
echo "Output len : ${PP_OUTPUT_LEN:-512}"
echo "Max seqs   : ${PP_MAX_NUM_SEQS:-512}"
echo "K3 timing  : ${VLLM_K3_TIMING:-0}"
echo "============================================"

# ---- Pre-flight: verify cross-node connectivity ----
for PEER_IP in 192.168.0.224 192.168.0.225 192.168.0.226; do
    [ "${PEER_IP}" = "$(hostname -I | grep -o "${PEER_IP}" | head -1)" ] && continue
    echo "Checking connectivity to peer node (${PEER_IP})..."
    if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -o BatchMode=yes "${PEER_IP}" "echo 'Peer reachable'" 2>/dev/null; then
        echo "Peer node ${PEER_IP} is reachable."
    else
        echo "WARNING: Cannot reach peer node ${PEER_IP} via SSH."
        echo "         Make sure the other node has started launch_pp.sh."
    fi
done

torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    profile_dsv4_pp.py
