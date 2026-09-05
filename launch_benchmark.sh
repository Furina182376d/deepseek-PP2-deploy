#!/usr/bin/env bash
# Start the unified benchmark on one pipeline node.
# Run with node rank 0 on MASTER_ADDR and rank 1 (etc.) on the other nodes.
set -euo pipefail

NODE_RANK="${1:?Usage: $0 <node_rank>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ``launch_benchmark.sh`` is commonly started through SSH/non-interactive
# shells, where the conda shell function has not been initialized.  Activate
# the same vLLM environment on every node before reading the Python config.
CONDA_BASE="${CONDA_BASE:-/home/tjy/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vllm}"
CONDA_SH="${CONDA_SH:-${CONDA_BASE}/etc/profile.d/conda.sh}"
if [[ ! -r "${CONDA_SH}" ]]; then
    echo "conda initialization script not found: ${CONDA_SH}" >&2
    echo "Set CONDA_BASE to the Miniconda installation on this node." >&2
    exit 2
fi
source "${CONDA_SH}"
conda activate "${CONDA_ENV_NAME}"

PYTHON_BIN="${PYTHON_BIN:-${CONDA_PREFIX}/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found in the activated environment: ${PYTHON_BIN}" >&2
    exit 2
fi
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
