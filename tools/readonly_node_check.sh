#!/usr/bin/env bash
set +e

printf 'HOST '; hostname
printf '\n== NVIDIA ==\n'
nvidia-smi --query-gpu=name,driver_version,pstate,compute_mode,mig.mode.current --format=csv,noheader
printf '\n== CONF_COMPUTE ==\n'
nvidia-smi conf-compute -q 2>&1 | head -35
printf '\n== DRIVER ==\n'
cat /proc/driver/nvidia/version
printf '\n== MODULE PARAMS ==\n'
grep -i -E 'RestrictProfiling|Profil' /proc/driver/nvidia/params 2>&1
printf '\n== PYTORCH ==\n'
/home/tjy/miniconda3/envs/vllm/bin/python -c 'import torch; print("torch=", torch.__version__); print("torch_cuda=", torch.version.cuda); print("cuda_available=", torch.cuda.is_available()); print("device=", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")' 2>&1
printf '\n== CUPTI FILES ==\n'
find /usr/local /home/tjy/miniconda3/envs/vllm -name 'libcupti.so*' -print 2>/dev/null
printf '\n== LDCONFIG ==\n'
ldconfig -p 2>/dev/null | grep -i libcupti
printf '\n== ENV ==\n'
/home/tjy/miniconda3/envs/vllm/bin/python -c 'import os; print("CUDA_HOME=", os.environ.get("CUDA_HOME")); print("LD_LIBRARY_PATH=", os.environ.get("LD_LIBRARY_PATH"))' 2>&1
printf '\n== USER ==\n'
id
printf '\n== DEVICE PERMISSIONS ==\n'
ls -l /dev/nvidia* 2>&1 | head -40
printf '\n== CAPABILITIES ==\n'
getcap /home/tjy/miniconda3/envs/vllm/bin/python 2>/dev/null || true
getcap /usr/bin/nvidia-smi 2>/dev/null || true
printf '\n== MODULE DETAIL ==\n'
grep -i -E 'RmProfilingAdminOnly|Enable|Persistence' /proc/driver/nvidia/params 2>&1
