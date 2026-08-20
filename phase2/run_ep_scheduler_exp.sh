#!/bin/bash
# EP 与调度实验 v2 — 合成 workload（LongBench 数据已不在本机，用 make_prompt 合成）
#
# A 组：EP 对照（短 ctx 集 {1K,2K,4K}×10，maxlen 6144，4×GPU）
#   A1 TP4      b=1   ← EP 基准
#   A2 TP4      b=4
#   A3 TP2+EP2  b=1   ← 假设：专家按卡分片+定向 dispatch 降低 TPOT 与边际成本
#   A4 TP2+EP2  b=4
# B 组：length-aware（混合 ctx 40 条 {32K×4,16K×5,8K×6,4K×7,2K×8,1K×10}，maxlen 64K，TP4）
#   B1 -1 FIFO（长→短提交） ← 长 prompt 堵死短请求
#   B2 -1 SJF （按 ctx 升序） ← 假设：TTFT 分位数大幅下降
#
# 用法: bash phase2/run_ep_scheduler_exp.sh
# 结果: phase0/profile_results/<时间戳>/，汇总见 reports/ep_scheduler_experiments/

set -eo pipefail
cd "$(dirname "$0")"

source /home/tjy/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ds

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

# ---- workload 定义 ----
SHORT="1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,\
2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,\
4096,4096,4096,4096,4096,4096,4096,4096,4096,4096"
SHORT=$(echo "$SHORT" | tr -d '\n')
MIXED_DESC="32768,32768,32768,32768,\
16384,16384,16384,16384,16384,\
8192,8192,8192,8192,8192,8192,\
4096,4096,4096,4096,4096,4096,4096,\
2048,2048,2048,2048,2048,2048,2048,2048,\
1024,1024,1024,1024,1024,1024,1024,1024,1024,1024"
MIXED_DESC=$(echo "$MIXED_DESC" | tr -d '\n')

run_one() {  # $1=描述 $2=TP $3=EP $4=BATCH $5=SORT $6=LENGTHS $7=MAXLEN
    echo ""
    echo "######################################################"
    echo " # $1  (TP=$2 EP=$3 batch=$4 sort=$5 maxlen=$7)"
    echo "######################################################"
    if LONGBENCH_TP=$2 LONGBENCH_EP=$3 LONGBENCH_BATCH_SIZE=$4 \
       LONGBENCH_SORT_BY_LEN=$5 LONGBENCH_SYNTH_LENGTHS="$6" \
       LONGBENCH_MAX_MODEL_LEN=$7 python run_longbench.py; then
        echo "OK: $1"
    else
        echo "FAILED: $1 (继续下一个)"
    fi
}

# ---- A 组：EP 对照 ----
run_one "A1 TP4 b=1"        4 0 1  0 "$SHORT"     6144
run_one "A2 TP4 b=4"        4 0 4  0 "$SHORT"     6144
run_one "A3 TP2+EP2 b=1"    2 2 1  0 "$SHORT"     6144
run_one "A4 TP2+EP2 b=4"    2 2 4  0 "$SHORT"     6144

# ---- B 组：length-aware ----
run_one "B1 TP4 -1 FIFO"    4 0 -1 0 "$MIXED_DESC" 65536
run_one "B2 TP4 -1 SJF"     4 0 -1 1 "$MIXED_DESC" 65536

echo ""
echo "全部完成。最近 6 个结果目录:"
ls -dt ../phase0/profile_results/*/ 2>/dev/null | head -6 || true
