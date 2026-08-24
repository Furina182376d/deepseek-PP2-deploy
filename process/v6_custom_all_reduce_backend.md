# TP Custom All-Reduce Backend A/B

日期：2026-08-24 至 2026-08-25  
结论：保留 vLLM 默认 custom all-reduce；禁用后 TPOT 明显回退且 CUDA Graph 占用增加。

## 1. 优化假设

当前每个 PP stage 在单节点 8 张 H20 上组成 TP=8。启动日志确认默认 TP group
按以下顺序选择 all-reduce backend：

```text
TP group: ['CUSTOM', 'PYNCCL']
PP group: ['PYNCCL']
EP group: ['PYNCCL']
```

本轮测试 vLLM 的节点内 custom all-reduce 是否优于禁用 custom backend 后的
通用通信路径。控制入口是 `LLM(disable_custom_all_reduce=...)`：

```text
baseline:  disable_custom_all_reduce=False
candidate: disable_custom_all_reduce=True
```

测试期间临时加入 `PP_DISABLE_CUSTOM_ALL_REDUCE` 以把该参数传给 `LLM`，并将
实际布尔值写入 `full_results.json`。候选无效后，该临时接口已从工作代码中移除，
恢复 vLLM 默认值 `False`。

## 2. 固定实验设置

两组只有 `disable_custom_all_reduce` 不同，且都保留 `v5` 验证有效的默认
shared-expert overlap：

```text
model=/data/models/Kimi-K3
vLLM=0.27.1
GPU=3 nodes x 8 NVIDIA H20
PP=3
TP per PP stage=8
VLLM_PP_LAYER_PARTITION=31,31,31
context_length=512
actual_prompt_tokens=336
output_len=256
ignore_eos=True
temperature=0
max_model_len=32768
max_num_seqs=384
gpu_memory_utilization=0.9
kv_cache_dtype=auto
warmups=2
repeats=5
VLLM_K3_TIMING=0
PP_ENFORCE_EAGER=0
VLLM_DISABLE_SHARED_EXPERTS_STREAM=0
PP_TORCH_PROFILER_DIR=
PP_PROFILE_ONLY=0
enable_flashinfer_autotune=False
distributed_executor_backend=external_launcher
CUDA Graph=FULL_AND_PIECEWISE
VLLM_USE_BREAKABLE_CUDAGRAPH=auto-enabled
NCCL_IB_DISABLE=1
NCCL_SOCKET_IFNAME==eth0
```

三台节点分别使用 node rank 0、1、2，等价启动方式为：

```bash
env \
  PP_OUTPUT_LEN=256 \
  PP_MAX_NUM_SEQS=384 \
  PP_NUM_WARMUPS=2 \
  PP_NUM_REPEATS=5 \
  VLLM_K3_TIMING=0 \
  PP_ENFORCE_EAGER=0 \
  VLLM_DISABLE_SHARED_EXPERTS_STREAM=0 \
  PP_DISABLE_CUSTOM_ALL_REDUCE=<0-or-1> \
  PP_TORCH_PROFILER_DIR= \
  PP_PROFILE_ONLY=0 \
  bash phase1/launch_pp.sh <node_rank> 31,31,31 <experiment_id>
```

`full_results.json` 分别确认：

```text
v6_custom_ar_baseline: disable_custom_all_reduce=false
v6_custom_ar_disabled: disable_custom_all_reduce=true
```

## 3. 原始结果

### 3.1 Custom all-reduce 开启（控制组）

```text
experiment_id=v6_custom_ar_baseline
result_dir=phase0/profile_results/20260824_233450
remote_logs=/home/tjy/kimi_bench/infra_v6_custom_allreduce/baseline_stage{0,1,2}.log
```

| Run | TTFT | Prefill throughput | Decode throughput | TPOT | Total | Avg GPU memory |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5099.88 ms | 68.1 tok/s | 6.1 tok/s | 164.01 ms | 46925.1 ms | 80071.5 MB |
| 2 | 3625.41 ms | 97.1 tok/s | 6.1 tok/s | 164.23 ms | 45504.9 ms | 80071.5 MB |
| 3 | 3451.78 ms | 102.1 tok/s | 6.2 tok/s | 162.38 ms | 44859.7 ms | 80071.5 MB |
| 4 | 3783.99 ms | 92.8 tok/s | 6.1 tok/s | 164.03 ms | 45612.6 ms | 80071.5 MB |
| 5 | 4373.51 ms | 79.8 tok/s | 6.1 tok/s | 164.35 ms | 46285.2 ms | 80059.0 MB |

```text
TPOT mean=163.80 ms, P50=164.03 ms, P95=164.35 ms
TPOT min=162.38 ms, max=164.35 ms
TTFT mean=4066.91 ms
prefill throughput mean=87.98 tok/s
decode throughput mean=6.12 tok/s
total latency mean=45837.50 ms
GPU memory mean=80069.00 MB
```

### 3.2 Custom all-reduce 禁用（候选）

```text
experiment_id=v6_custom_ar_disabled
result_dir=phase0/profile_results/20260824_234744
remote_logs=/home/tjy/kimi_bench/infra_v6_custom_allreduce/disabled_stage{0,1,2}.log
```

| Run | TTFT | Prefill throughput | Decode throughput | TPOT | Total | Avg GPU memory |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 38744.68 ms | 8.7 tok/s | 5.9 tok/s | 168.34 ms | 81672.5 ms | 80077.7 MB |
| 2 | 6158.81 ms | 56.1 tok/s | 5.9 tok/s | 169.50 ms | 49381.8 ms | 80077.7 MB |
| 3 | 4867.93 ms | 71.5 tok/s | 5.9 tok/s | 168.94 ms | 47949.7 ms | 80077.7 MB |
| 4 | 4574.25 ms | 76.3 tok/s | 5.9 tok/s | 169.42 ms | 47777.3 ms | 80077.7 MB |
| 5 | 4596.00 ms | 75.9 tok/s | 5.9 tok/s | 168.76 ms | 47632.2 ms | 80077.7 MB |

```text
TPOT mean=168.99 ms, P50=168.94 ms, P95=169.50 ms
TPOT min=168.34 ms, max=169.50 ms
TTFT mean=11788.33 ms
prefill throughput mean=57.70 tok/s
decode throughput mean=5.90 tok/s
total latency mean=54882.70 ms
GPU memory mean=80077.70 MB
```

## 4. A/B 对比

| 指标 | Custom 开启 | Custom 禁用 | 候选相对变化 | 判断 |
|---|---:|---:|---:|---|
| TPOT mean | 163.80 ms | 168.99 ms | +5.19 ms / +3.17% | 明显变慢 |
| TPOT P50 | 164.03 ms | 168.94 ms | +4.91 ms / +2.99% | 明显变慢 |
| TPOT P95 | 164.35 ms | 169.50 ms | +5.15 ms / +3.13% | 明显变慢 |
| Decode throughput mean | 6.12 tok/s | 5.90 tok/s | -3.59% | 变慢 |
| GPU memory mean | 80069.0 MB | 80077.7 MB | +8.7 MB | 略增 |

候选五个 TPOT 样本全部高于控制组最大值 `164.35 ms`，两组区间没有重叠。
因此这不是由单个异常值造成的均值差异。

### CUDA Graph 与 KV memory

启动日志还观察到以下方向一致的资源差异：

```text
custom enabled graph memory: 约 1.18--1.20 GiB/stage
custom disabled graph memory: 约 1.27--1.29 GiB/stage
```

禁用 custom backend 后 graph capture 约多占 `0.09 GiB/stage`。可用 KV cache
也从控制组约 `12.59/7.76/6.83 GiB` 变为候选约
`12.50/7.75/6.82 GiB`。这与候选的显存方向一致。

候选 run 1 的 TTFT `38.74 s` 是明显异常值；即使不使用 TTFT 或 total latency
判定，TPOT mean/P50/P95、每个样本、decode throughput 和 graph memory 都一致
支持 custom backend。

## 5. 结论与后续配置

当前 PCIe/NVLink 节点内 TP=8 拓扑上，vLLM custom all-reduce 是有效的默认
优化。禁用它使 TPOT 回退约 3.2%，同时增加 CUDA Graph 显存。因此后续保留：

```text
disable_custom_all_reduce=False
```

不保留候选设置 `True`。本轮临时加入的环境传参接口也已经撤销，工作代码回到
实验前的默认行为。下一项优化继续从 custom all-reduce 开启、shared-expert
overlap 开启、`31,31,31` 的配置出发。

## 6. 实验限制

- 当前 workload 是单请求 decode batch=1；高并发下 collective tensor size 和
  backend 选择可能变化，需要另做吞吐实验。
- CUPTI 不可用，本轮不能直接给出 custom kernel 与 NCCL kernel 的 GPU 时间；
  结论来自端到端 TPOT 和稳定的五次 A/B。
- FlashInfer all-reduce multicast 在当前无 NVSwitch 拓扑上初始化失败并自动回退，
  因此本轮比较的是当前可用 custom backend 与禁用 custom 后的通用路径，而不是
  可工作的 FlashInfer multicast 实现。
