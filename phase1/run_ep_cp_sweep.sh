#!/usr/bin/env bash
# Run the supported Kimi-K3 EP experiments sequentially.
#
# Run this script on the master node (192.168.0.224).  It starts the
# existing launch_pp.sh on all three nodes over SSH, waits for all ranks to
# exit, and only then starts the next configuration.
#
# Usage:
#   ./run_ep_cp_sweep.sh
#   SWEEP_LOG_DIR=/tmp/k3-sweep ./run_ep_cp_sweep.sh
#
# The nodes and repository path can be overridden when the deployment uses a
# different network or checkout location:
#   SWEEP_HOSTS="192.168.0.224 192.168.0.225 192.168.0.226"
#   SWEEP_REPO=/home/tjy/codebases/deepseek-PP2-deploy/phase1
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SWEEP_REPO:-${SCRIPT_DIR}}"
HOST_LIST="${SWEEP_HOSTS:-192.168.0.224 192.168.0.225 192.168.0.226}"
read -r -a HOSTS <<< "${HOST_LIST}"
MASTER_HOST="${SWEEP_MASTER_HOST:-${HOSTS[0]}}"
LOG_DIR="${SWEEP_LOG_DIR:-${SCRIPT_DIR}/sweep_results/$(date +%Y%m%d_%H%M%S)}"
SSH_OPTS=( -o BatchMode=yes -o StrictHostKeyChecking=no )

if (( ${#HOSTS[@]} != 3 )); then
    echo "SWEEP_HOSTS must contain exactly 3 hosts (got ${#HOSTS[@]})." >&2
    exit 2
fi
if [[ "$(hostname -I 2>/dev/null || true)" != *"${MASTER_HOST}"* ]]; then
    echo "Warning: this host does not appear to have MASTER_HOST=${MASTER_HOST}." >&2
    echo "Run this script on the torchrun master node, or set SWEEP_MASTER_HOST." >&2
fi
mkdir -p "${LOG_DIR}"

cleanup() {
    local status=$?
    if (( status != 0 )); then
        echo "Sweep aborted (status=${status}); inspect logs in ${LOG_DIR}." >&2
    fi
    exit "${status}"
}
trap cleanup EXIT

# Kimi-K3's MultiHeadLatentAttention in the installed vLLM explicitly rejects
# context parallelism, so DCP/PCP > 1 cannot be benchmarked for this model.
# Keep the sweep to the two valid EP comparisons. Do not silently retry the
# unsupported CP configurations after loading the 1.5 TB model.
CONFIGS=(
    "tp8_ep0_dcp1 0 1"
    "tp8_ep1_dcp1 1 1"
)

echo "Kimi-K3 EP/CP sweep"
echo "Note: Kimi-K3 MLA does not support context parallelism in this vLLM;"
echo "      DCP/PCP > 1 configurations are skipped."
echo "Hosts: ${HOSTS[*]}"
echo "Logs : ${LOG_DIR}"
echo "Repo : ${REPO_DIR}"

for spec in "${CONFIGS[@]}"; do
    read -r name ep dcp <<< "${spec}"
    config_log="${LOG_DIR}/${name}"
    mkdir -p "${config_log}"
    echo
    echo "===== ${name}: EP=${ep}, prefill_CP=1, decode_CP=${dcp} ====="

    pids=()
    for rank in 0 1 2; do
        host="${HOSTS[$rank]}"
        log_file="${config_log}/node${rank}.log"
        echo "Starting node ${rank} (${host}); log=${log_file}"
        # Keep SSH itself in the background so all three torchrun jobs start
        # together. The remote launch script remains in the foreground.
        ssh "${SSH_OPTS[@]}" "${host}" \
            "cd '${REPO_DIR}' && \
             PP_TP_SIZE=8 \
             PP_PREFILL_CP_SIZE=1 PP_DECODE_CP_SIZE=${dcp} \
             PP_ENABLE_EXPERT_PARALLEL=${ep} \
             PP_EXPERIMENT_ID=${name} ./launch_pp.sh ${rank}" \
            >"${log_file}" 2>&1 &
        pids+=("$!")
    done

    failed=0
    for i in "${!pids[@]}"; do
        if ! wait "${pids[$i]}"; then
            echo "Node ${i} failed for ${name}; see ${config_log}/node${i}.log" >&2
            failed=1
            # Stop the sibling SSH sessions immediately. This also closes the
            # remote torchrun channels instead of waiting for NCCL timeout.
            for pid in "${pids[@]}"; do
                if [[ "${pid}" != "${pids[$i]}" ]]; then
                    kill "${pid}" 2>/dev/null || true
                fi
            done
        fi
    done
    if (( failed != 0 )); then
        echo "Configuration ${name} failed; later configurations were not run." >&2
        exit 1
    fi
    echo "Completed ${name}."
done

echo
echo "All five configurations completed. Results and logs are under ${LOG_DIR}."
