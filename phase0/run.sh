#!/bin/bash
source /home/tjy/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate vllm

# 自动创建 libcuda.so 软链接（如果缺失）
if [ ! -f "$CONDA_PREFIX/lib64/libcuda.so" ]; then
    ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so "$CONDA_PREFIX/lib64/libcuda.so"
fi
# 屏蔽可能存在的 stub（防止运行时加载错误的 stub 库）
if [ -f "$CONDA_PREFIX/lib64/stubs/libcuda.so" ] && [ ! -f "$CONDA_PREFIX/lib64/stubs/libcuda.so.bak" ]; then
    mv "$CONDA_PREFIX/lib64/stubs/libcuda.so" "$CONDA_PREFIX/lib64/stubs/libcuda.so.bak"
fi

export CUDA_HOME=$CONDA_PREFIX
export CPATH=$CUDA_HOME/targets/x86_64-linux/include:$CPATH
export C_INCLUDE_PATH=$CUDA_HOME/targets/x86_64-linux/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$CUDA_HOME/targets/x86_64-linux/include:$CPLUS_INCLUDE_PATH

# 保持 conda lib 优先（解决 libstdc++ 版本），但系统路径放在后面保底
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH

rm -rf ~/.cache/flashinfer

CUDA_VISIBLE_DEVICES=4,5,6,7 python profile_dsv4.py