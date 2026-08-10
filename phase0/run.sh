#!/bin/bash
# 初始化 conda，使其在脚本中可用
source /home/tjy/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate vllm

export CUDA_HOME=$CONDA_PREFIX
export CPATH=$CUDA_HOME/targets/x86_64-linux/include:$CPATH
export C_INCLUDE_PATH=$CUDA_HOME/targets/x86_64-linux/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$CUDA_HOME/targets/x86_64-linux/include:$CPLUS_INCLUDE_PATH
export LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib:$CUDA_HOME/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

rm -rf ~/.cache/flashinfer
CUDA_VISIBLE_DEVICES=4,5,6,7 python profile_dsv4.py