#!/usr/bin/env bash
# Start the unified benchmark on one pipeline node.
# Run with node rank 0 on MASTER_ADDR and rank 1 (etc.) on the other nodes.
set -euo pipefail

NODE_RANK="${1:?Usage: $0 <node_rank>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "${SCRIPT_DIR}"

# Importing run_benchmark.py is side-effect free and only reads its config.
NNODES="$(${PYTHON_BIN} -c 'import run_benchmark as c; print(c.NNODES)')"
TP_SIZE="$(${PYTHON_BIN} -c 'import run_benchmark as c; print(c.TP_SIZE_PER_STAGE)')"
MASTER_ADDR="$(${PYTHON_BIN} -c 'import run_benchmark as c; print(c.MASTER_ADDR)')"
MASTER_PORT="$(${PYTHON_BIN} -c 'import run_benchmark as c; print(c.MASTER_PORT)')"
IFACE_NAME="$(${PYTHON_BIN} -c 'import run_benchmark as c; print(c.NETWORK_INTERFACE)')"

if ! [[ "${NODE_RANK}" =~ ^[0-9]+$ ]] || (( NODE_RANK >= NNODES )); then
    echo "node rank must be an integer in [0, $((NNODES - 1))]" >&2
    exit 2
fi

export GLOO_SOCKET_IFNAME="${IFACE_NAME}"
export NCCL_SOCKET_IFNAME="${IFACE_NAME}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_CUMEM_HOST_ENABLE="${NCCL_CUMEM_HOST_ENABLE:-0}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
export TORCH_DISTRIBUTED_TIMEOUT="${TORCH_DISTRIBUTED_TIMEOUT:-1800}"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

exec "${PYTHON_BIN}" -m torch.distributed.run \
    --nnodes="${NNODES}" \
    --nproc_per_node="${TP_SIZE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    run_benchmark.py
