#!/usr/bin/env bash
set +e
NSYS=/usr/local/cuda-13.0/nsight-systems-2025.3.2/target-linux-x64/nsys
mkdir -p /home/tjy/nsys_smoke
"$NSYS" profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cuda-memory-usage=false \
  --force-overwrite=true \
  -o /home/tjy/nsys_smoke/gemm_plain \
  -- /home/tjy/miniconda3/envs/vllm/bin/python /home/tjy/cuda_gemm_smoke.py 2>&1
printf 'ARTIFACTS\n'
find /home/tjy/nsys_smoke -maxdepth 1 -type f -name 'gemm_plain*' -printf '%f %s bytes\n' 2>/dev/null
