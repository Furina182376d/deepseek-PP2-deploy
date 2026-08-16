#!/bin/bash
# EP 与调度实验（batch 扫描结论的后续）：
#   Exp1: TP=2+EP=2 vs TP=4 基准（各跑 b=1 / b=4，对比 TPOT 与边际成本）
#   Exp2: length-aware 提交（按 ctx 排序 + 全量一次提交，对比 FIFO 的 -1）
#
# 用法:
#   bash phase2/run_ep_scheduler_exp.sh
#   自动挑选 4 张空闲 GPU；每个组合独立 python 进程（独立时间戳结果目录）。
#   结果目录: phase0/profile_results/<时间戳>/，汇总见 reports/ep_scheduler_experiments/

set -eo pipefail
cd "$(dirname "$0")"

source /home/tjy/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate vllm

# 自动创建 libcuda.so 软链接（如果缺失）
mkdir -p "$CONDA_PREFIX/lib64"
if [ ! -f "$CONDA_PREFIX/lib64/libcuda.so" ]; then
    ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so "$CONDA_PREFIX/lib64/libcuda.so"
fi
if [ -f "$CONDA_PREFIX/lib64/stubs/libcuda.so" ] && [ ! -f "$CONDA_PREFIX/lib64/stubs/libcuda.so.bak" ]; then
    mv "$CONDA_PREFIX/lib64/stubs/libcuda.so" "$CONDA_PREFIX/lib64/stubs/libcuda.so.bak"
fi

export CUDA_HOME=$CONDA_PREFIX
export CPATH=$CUDA_HOME/targets/x86_64-linux/include:$CPATH
export C_INCLUDE_PATH=$CUDA_HOME/targets/x86_64-linux/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$CUDA_HOME/targets/x86_64-linux/include:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH

# flashinfer JIT 缓存清理一次，后续运行复用
rm -rf ~/.cache/flashinfer

# 自动挑 4 张空闲 GPU（used < 10GB）
FREE_GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | awk -F',' '$2+0 < 10000 {print $1}' | paste -sd, -)
NFREE=$(echo "$FREE_GPUS" | grep -o "," | wc -l); NFREE=$((NFREE + 1))
if [ "$NFREE" -lt 4 ]; then
    echo "ERROR: 空闲 GPU 不足 4 张（当前空闲: ${FREE_GPUS:-无}）"; exit 1
fi
GPUS=$(echo "$FREE_GPUS" | cut -d, -f1-4)
echo "使用 GPU: $GPUS"
export CUDA_VISIBLE_DEVICES=$GPUS

run_one() {  # $1=描述  $2=TP  $3=EP  $4=BATCH  $5=SORT_BY_LEN
    echo ""
    echo "######################################################"
    echo " # $1  (TP=$2 EP=$3 batch=$4 sort=$5)"
    echo "######################################################"
    if LONGBENCH_TP=$2 LONGBENCH_EP=$3 LONGBENCH_BATCH_SIZE=$4 \
       LONGBENCH_SORT_BY_LEN=$5 python run_longbench.py; then
        echo "OK: $1"
    else
        echo "FAILED: $1 (继续下一个)"
    fi
}

# Exp1: EP 对照（对比基准: TP4/EP1 b1≈b0 目录 20260814_233513、b4 目录 20260815_232828）
run_one "TP2+EP2 b=1" 2 1 1 0
run_one "TP2+EP2 b=4" 2 1 4 0

# Exp2: length-aware（对比基准: FIFO 的 -1 目录 20260815_000022）
run_one "TP4 length-aware -1" 4 0 -1 1

echo ""
echo "全部完成。最近 3 个结果目录:"
ls -dt ../phase0/profile_results/*/ 2>/dev/null | head -3 || true
