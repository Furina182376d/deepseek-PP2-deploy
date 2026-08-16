#!/bin/bash
# Workload 批量基准测试 — 基于 LongBench workload，按 batch 分块提交。
#
# 用法（任意目录执行）:
#   bash phase2/run_workload_batch.sh                         # 默认 batch=16
#   BATCH_SIZES="8 16 -1" bash phase2/run_workload_batch.sh   # 扫描多个 batch（每个一次模型加载）
#   LONGBENCH_MAX_SAMPLES=all bash phase2/run_workload_batch.sh # 全量样本（真·长 workload）
#
# batch 语义（传给 run_longbench.py 的 LONGBENCH_BATCH_SIZE）:
#   0   = 逐样本串行计时（旧路径基线）
#   N>0 = 每块提交 N 个 prompt，引擎连续批处理
#   -1  = 全部 prompt 一次提交（最接近线上连续批处理）
#
# 每个 batch 一次独立 python 进程（各自加载模型、独立时间戳结果目录），
# 结果目录: phase0/profile_results/<时间戳>/，report.txt 末尾为 BATCHED WORKLOAD SUMMARY。

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

# flashinfer JIT 缓存清理一次（避免陈旧 .so 的 GLIBCXX 不兼容问题）；
# 同一脚本内的后续 batch 复用本次重新编译的缓存，省去重复 JIT。
rm -rf ~/.cache/flashinfer

BATCH_SIZES="${BATCH_SIZES:-16}"
MAX_SAMPLES="${LONGBENCH_MAX_SAMPLES:-20}"

echo "======================================================"
echo " Workload batch benchmark"
echo "   batch sizes : $BATCH_SIZES"
echo "   max samples : $MAX_SAMPLES (per dataset; 'all' = 全部)"
echo "   GPUs        : ${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
echo "======================================================"

for b in $BATCH_SIZES; do
    echo ""
    echo "######################################################"
    echo " # batch_size = $b"
    echo "######################################################"

    export LONGBENCH_BATCH_SIZE=$b
    if [ "$MAX_SAMPLES" = "all" ] || [ -z "$MAX_SAMPLES" ]; then
        export LONGBENCH_MAX_SAMPLES=all
    else
        export LONGBENCH_MAX_SAMPLES=$MAX_SAMPLES
    fi

    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}" python run_longbench.py
done

echo ""
echo "全部完成。最近 $(echo "$BATCH_SIZES" | wc -w) 个结果目录（每个 batch 一个）:"
ls -dt ../phase0/profile_results/*/ 2>/dev/null | head -n "$(echo "$BATCH_SIZES" | wc -w)" || true
