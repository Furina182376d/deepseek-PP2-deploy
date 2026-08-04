#!/bin/bash
# ---------------------------------------------------------------------------
# Launch PP=2 TP=4 profiling across TWO nodes via torchrun.
#
# Usage (run on BOTH nodes, with different <node_rank>):
#   Node 0 (192.168.0.63):  ./launch_pp.sh 0
#   Node 1 (192.168.0.65):  ./launch_pp.sh 1
#
# Prerequisites:
#   - Passwordless SSH between the two nodes.
#   - The ``ds`` conda environment available on both nodes.
#   - Identical codebase path on both nodes.
# ---------------------------------------------------------------------------
set -euo pipefail

NODE_RANK="${1:?Usage: $0 <node_rank (0 or 1)>}"
MASTER_ADDR="192.168.0.63"
MASTER_PORT=29500
NNODES=2
NPROC_PER_NODE=4   # one process per GPU within the TP=4 group

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BASE="${HOME}/miniconda3"

# ---------------------------------------------------------------------------
# Activate conda environment
# ---------------------------------------------------------------------------
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate ds

# ---------------------------------------------------------------------------
# NCCL settings (from prior working multi-node deployment on these nodes)
# ---------------------------------------------------------------------------
export NCCL_IB_DISABLE=1                 # no InfiniBand — use TCP over eth0
export NCCL_SOCKET_IFNAME="=eth0"
export GLOO_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_CUMEM_HOST_ENABLE=0          # cuMem unavailable, fallback to /dev/shm

# Required by external_launcher executor (deterministic scheduling)
export VLLM_ENABLE_V1_MULTIPROCESSING=0

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
cd "${SCRIPT_DIR}"

echo "============================================"
echo "Node rank  : ${NODE_RANK} / ${NNODES}"
echo "Master     : ${MASTER_ADDR}:${MASTER_PORT}"
echo "Procs/node : ${NPROC_PER_NODE}"
echo "Total procs: $((NNODES * NPROC_PER_NODE))"
echo "============================================"

torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    profile_dsv4_pp.py
