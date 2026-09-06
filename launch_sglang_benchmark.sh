#!/usr/bin/env bash
# Start one SGLang distributed-server node. Run this on every pipeline node.
set -euo pipefail

NODE_RANK="${1:?Usage: $0 <node_rank>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BASE="${CONDA_BASE:-/home/tjy/miniconda3}"
CONFIG_PYTHON="${CONFIG_PYTHON:-${CONDA_BASE}/bin/python}"

if [[ ! -x "${CONFIG_PYTHON}" ]]; then
    echo "Configuration Python not found: ${CONFIG_PYTHON}" >&2
    exit 2
fi
cd "${SCRIPT_DIR}"

# Read the one central configuration file before activating the SGLang env.
CONDA_ENV_NAME="${SGLANG_CONDA_ENV:-$(${CONFIG_PYTHON} -c 'import sglang_benchmark as c; print(c.SGLANG_CONDA_ENV)')}"
NNODES="$(${CONFIG_PYTHON} -c 'import sglang_benchmark as c; print(c.NNODES)')"
MASTER_ADDR="$(${CONFIG_PYTHON} -c 'import sglang_benchmark as c; print(c.MASTER_ADDR)')"
SGLANG_PORT="$(${CONFIG_PYTHON} -c 'import sglang_benchmark as c; print(c.SGLANG_PORT)')"
IFACE_NAME="$(${CONFIG_PYTHON} -c 'import sglang_benchmark as c; print(c.NETWORK_INTERFACE)')"

if ! [[ "${NODE_RANK}" =~ ^[0-9]+$ ]] || (( NODE_RANK >= NNODES )); then
    echo "node rank must be an integer in [0, $((NNODES - 1))]" >&2
    exit 2
fi

CONDA_SH="${CONDA_SH:-${CONDA_BASE}/etc/profile.d/conda.sh}"
if [[ ! -r "${CONDA_SH}" ]]; then
    echo "conda initialization script not found: ${CONDA_SH}" >&2
    exit 2
fi
source "${CONDA_SH}"
conda activate "${CONDA_ENV_NAME}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_PREFIX}/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found in the activated environment: ${PYTHON_BIN}" >&2
    exit 2
fi
if ! command -v sglang >/dev/null 2>&1; then
    echo "sglang is not installed in conda environment: ${CONDA_ENV_NAME}" >&2
    exit 2
fi

export GLOO_SOCKET_IFNAME="${IFACE_NAME}"
export NCCL_SOCKET_IFNAME="${IFACE_NAME}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_CUMEM_HOST_ENABLE="${NCCL_CUMEM_HOST_ENABLE:-0}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
# CUDA 13.0's nvcc currently crashes while compiling SGLang's DSV4 indexer
# metadata JIT kernel. DeepGEMM provides the equivalent implementation, so use
# it by default; set this to 1 only after the toolchain can compile the JIT path.
export SGLANG_OPT_USE_JIT_INDEXER_METADATA="${SGLANG_OPT_USE_JIT_INDEXER_METADATA:-0}"
# The current image may ship a FlashInfer Python package newer than its
# precompiled cubin package. Keep the launcher usable in that image while
# allowing strict dependency validation with FLASHINFER_DISABLE_VERSION_CHECK=0.
FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
if [[ "${FLASHINFER_DISABLE_VERSION_CHECK}" == "1" ]]; then
    export FLASHINFER_DISABLE_VERSION_CHECK
    echo "Warning: FlashInfer cubin version checking is disabled; install matching" >&2
    echo "flashinfer and flashinfer-cubin versions for a strict production setup." >&2
else
    # FlashInfer checks only whether this variable is present, so exporting
    # the string "0" would unexpectedly keep the check disabled.
    unset FLASHINFER_DISABLE_VERSION_CHECK
fi
# Keep FlashInfer JIT artifacts writable even when the launcher runs from a
# service account with a read-only home-directory cache.
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/tmp/sglang_flashinfer_workspace}"
mkdir -p "${FLASHINFER_WORKSPACE_BASE}"

# Check the installed CLI and selected optional features before backgrounding a
# distributed worker. This prevents rank 0 from waiting for an endpoint after
# an invalid worker launch.
"${PYTHON_BIN}" sglang_benchmark.py validate-serve "${NODE_RANK}"

LOG_FILE="${SGLANG_LOG_FILE:-/tmp/sglang_benchmark_node${NODE_RANK}.log}"
echo "Starting SGLang node ${NODE_RANK}; log: ${LOG_FILE}"
setsid "${PYTHON_BIN}" sglang_benchmark.py serve "${NODE_RANK}" >"${LOG_FILE}" 2>&1 &
SERVER_PID=$!
export SGLANG_SERVER_PID="${SERVER_PID}"

cleanup() {
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill -- -"${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

if (( NODE_RANK == 0 )); then
    # The root node owns the HTTP endpoint and executes the benchmark after it
    # becomes ready. Ending this process also ends the distributed job.
    if [[ -z "${SGLANG_BENCHMARK_OUTPUT_DIR:-}" ]]; then
        export SGLANG_BENCHMARK_OUTPUT_DIR="results/sglang_$(date -u +%Y%m%d_%H%M%S)"
    fi
    "${PYTHON_BIN}" sglang_benchmark.py benchmark
else
    # A non-root node has no benchmark client. Once it has observed the root
    # endpoint healthy, three failed health checks mean rank 0 completed or
    # failed, so terminate this local worker as well.
    endpoint_seen=0
    failed_checks=0
    while kill -0 "${SERVER_PID}" 2>/dev/null; do
        if curl -sf --connect-timeout 2 "http://${MASTER_ADDR}:${SGLANG_PORT}/v1/models" >/dev/null; then
            endpoint_seen=1
            failed_checks=0
        elif (( endpoint_seen )); then
            failed_checks=$((failed_checks + 1))
            if (( failed_checks >= 3 )); then
                echo "Root SGLang endpoint stopped; terminating node ${NODE_RANK}."
                break
            fi
        fi
        sleep 5
    done
    # Surface an early worker failure instead of silently returning to the
    # shell while rank 0 waits for a peer that can no longer join.
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        set +e
        wait "${SERVER_PID}"
        worker_status=$?
        set -e
        echo "SGLang worker on node ${NODE_RANK} exited with status ${worker_status}." >&2
        echo "Last node log lines:" >&2
        tail -40 "${LOG_FILE}" >&2 || true
        exit "${worker_status}"
    fi
fi
