#!/usr/bin/env bash
# Start the unified vLLM benchmark on one pipeline node.
# Run with node rank 0 on MASTER_ADDR and rank 1 (etc.) on the other nodes.
#
# The second argument selects which benchmark runner executes: ``standard``
# runs vllm_benchmark.py, ``speculative`` runs vllm_benchmark_speculative.py.
# Each script carries its own PP/TP topology, so the launcher reads the
# distributed config from whichever script was selected.
set -euo pipefail

NODE_RANK="${1:?Usage: $0 <node_rank> [0|1|standard|speculative]}"
SPECULATIVE_MODE="${2:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${SPECULATIVE_MODE,,}" in
    0|false|no|off|standard|target)
        BENCHMARK_SCRIPT="${SCRIPT_DIR}/vllm_benchmark.py"
        MODE_LABEL="standard"
        ;;
    1|true|yes|on|speculative|dspark)
        BENCHMARK_SCRIPT="${SCRIPT_DIR}/vllm_benchmark_speculative.py"
        MODE_LABEL="speculative"
        ;;
    *)
        echo "speculative mode must be 0/1, standard, or speculative" >&2
        exit 2
        ;;
esac

if [[ ! -f "${BENCHMARK_SCRIPT}" ]]; then
    echo "Benchmark script not found: ${BENCHMARK_SCRIPT}" >&2
    exit 2
fi

# ``launch_vllm_benchmark.sh`` is commonly started through SSH/non-interactive
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

# Importing the benchmark module is side-effect free (its config section only
# uses the standard library), so the selected script can be queried by path.
read_config() {
    "${PYTHON_BIN}" - "${BENCHMARK_SCRIPT}" "$1" <<'PY'
import importlib.util
import sys

script_path, attribute = sys.argv[1:]
module_spec = importlib.util.spec_from_file_location("vllm_benchmark_config", script_path)
if module_spec is None or module_spec.loader is None:
    raise RuntimeError(f"cannot load benchmark script: {script_path}")
module = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(module)
print(getattr(module, attribute))
PY
}

NNODES="$(read_config NNODES)"
TP_SIZE="$(read_config TP_SIZE_PER_STAGE)"
MASTER_ADDR="$(read_config MASTER_ADDR)"
MASTER_PORT="$(read_config MASTER_PORT)"
IFACE_NAME="$(read_config NETWORK_INTERFACE)"

if ! [[ "${NODE_RANK}" =~ ^[0-9]+$ ]] || (( NODE_RANK >= NNODES )); then
    echo "node rank must be an integer in [0, $((NNODES - 1))]" >&2
    exit 2
fi

export GLOO_SOCKET_IFNAME="${IFACE_NAME}"
export NCCL_SOCKET_IFNAME="${IFACE_NAME}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_CUMEM_HOST_ENABLE="${NCCL_CUMEM_HOST_ENABLE:-0}"
# cuMem (CUMEM) allocation is broken in this containerized environment and
# NCCL's P2P/CUMEM channels (used by alltoall-style send/recv traffic) hang.
# Keep both the device and host switches off so NCCL falls back to NVLink P2P
# and TCP.
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
export TORCH_DISTRIBUTED_TIMEOUT="${TORCH_DISTRIBUTED_TIMEOUT:-1800}"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

echo "Starting vLLM node ${NODE_RANK} (${MODE_LABEL} mode); benchmark script: ${BENCHMARK_SCRIPT}"
exec "${PYTHON_BIN}" -m torch.distributed.run \
    --nnodes="${NNODES}" \
    --nproc_per_node="${TP_SIZE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${BENCHMARK_SCRIPT}"
