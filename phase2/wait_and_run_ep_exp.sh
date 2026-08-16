#!/bin/bash
# 等待至少 4 张空闲 GPU（used < 10GB）后自动启动 EP/调度实验。
# 最多等待 6 小时，每 120 秒检查一次。
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MAX_WAIT_S=21600
START=$(date +%s)

while :; do
    NFREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | awk -F',' '$2+0 < 10000' | wc -l)
    if [ "$NFREE" -ge 4 ]; then
        echo "$(date '+%F %T') 检测到 ${NFREE} 张空闲 GPU，启动实验"
        bash "$SCRIPT_DIR/run_ep_scheduler_exp.sh"
        exit $?
    fi
    if [ $(( $(date +%s) - START )) -gt $MAX_WAIT_S ]; then
        echo "$(date '+%F %T') 等待超时 ${MAX_WAIT_S}s，放弃"
        exit 2
    fi
    echo "$(date '+%F %T') 空闲 GPU ${NFREE}/4，120s 后再查"
    sleep 120
done
