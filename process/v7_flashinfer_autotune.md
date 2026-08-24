# FlashInfer Autotune A/B

日期：2026-08-25  
结论：本 workload 下没有可重复收益，继续关闭 FlashInfer autotune。

## 1. 优化假设与前置检查

`enable_flashinfer_autotune` 会在初始化/warmup 阶段为可用的 FlashInfer kernel
选择实现。当前配置因历史三节点版本不一致风险固定为 `False`，本轮重新确认三节点：

```text
torch=2.13.0
flashinfer-python=0.6.16.post3
vllm=0.27.1
install=/home/tjy/miniconda3/envs/vllm/lib/python3.12/site-packages
```

版本与路径完全一致，因此进行以下 A/B：

```text
control:   enable_flashinfer_autotune=False
candidate: enable_flashinfer_autotune=True
```

实验期间临时增加 `PP_ENABLE_FLASHINFER_AUTOTUNE` 传参，并将实际值写入 JSON。
候选没有达到有效阈值后，临时接口已移除，代码恢复固定 `False`。

## 2. 固定设置

```text
model=/data/models/Kimi-K3
GPU=3 nodes x 8 NVIDIA H20
PP=3, TP per stage=8
partition=31,31,31
context_length=512 (actual prompt=336 tokens)
output_len=256, ignore_eos=True, temperature=0
max_model_len=32768
max_num_seqs=384
gpu_memory_utilization=0.9
kv_cache_dtype=auto
warmups=2
repeats=5
VLLM_K3_TIMING=0
enforce_eager=False
CUDA Graph=FULL_AND_PIECEWISE
shared-expert stream overlap=enabled
disable_custom_all_reduce=False
torch profiler=disabled
enable_flashinfer_autotune=<False or True>  # 唯一变量
NCCL_IB_DISABLE=1
NCCL_SOCKET_IFNAME==eth0
```

三节点分别使用 node rank 0、1、2。等价命令为：

```bash
env PP_OUTPUT_LEN=256 PP_MAX_NUM_SEQS=384 \
  PP_NUM_WARMUPS=2 PP_NUM_REPEATS=5 \
  VLLM_K3_TIMING=0 PP_ENFORCE_EAGER=0 \
  VLLM_DISABLE_SHARED_EXPERTS_STREAM=0 \
  PP_ENABLE_FLASHINFER_AUTOTUNE=<0-or-1> \
  PP_TORCH_PROFILER_DIR= PP_PROFILE_ONLY=0 \
  bash phase1/launch_pp.sh <node_rank> 31,31,31 <experiment_id>
```

启动日志中的 `KernelConfig` 分别确认
`enable_flashinfer_autotune=False/True`。两组均完成三 stage CUDA Graph capture，
没有复现历史 gloo broadcast 或版本不一致崩溃。

## 3. 逐次结果

### 3.1 Autotune 关闭控制组

```text
experiment_id=v7_fi_autotune_off
result_dir=phase0/profile_results/20260825_000922
remote_logs=/home/tjy/kimi_bench/infra_v7_flashinfer_autotune/off_stage{0,1,2}.log
```

| Run | TTFT | Prefill tok/s | Decode tok/s | TPOT | Total | Avg GPU memory |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6210.63 ms | 55.6 | 6.1 | 164.35 ms | 48121.9 ms | 80056.2 MB |
| 2 | 8760.83 ms | 39.1 | 6.1 | 164.72 ms | 50765.5 ms | 80056.2 MB |
| 3 | 6911.58 ms | 49.8 | 6.1 | 163.27 ms | 48547.6 ms | 80056.2 MB |
| 4 | 52994.75 ms | 6.4 | 6.0 | 166.40 ms | 95428.5 ms | 80056.2 MB |
| 5 | 39549.77 ms | 8.5 | 6.0 | 165.40 ms | 81728.4 ms | 80056.2 MB |

```text
TPOT mean=164.83 ms, P50=164.72 ms, P95=166.40 ms
TPOT min=163.27 ms, max=166.40 ms
decode throughput mean=6.06 tok/s
TTFT mean=22885.51 ms
GPU memory mean=80056.20 MB
```

### 3.2 Autotune 开启候选

```text
experiment_id=v7_fi_autotune_on
result_dir=phase0/profile_results/20260825_002444
remote_logs=/home/tjy/kimi_bench/infra_v7_flashinfer_autotune/on_stage{0,1,2}.log
```

| Run | TTFT | Prefill tok/s | Decode tok/s | TPOT | Total | Avg GPU memory |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 22291.39 ms | 15.2 | 6.1 | 163.67 ms | 64029.5 ms | 80080.5 MB |
| 2 | 18176.31 ms | 18.7 | 6.0 | 166.35 ms | 60598.4 ms | 80080.5 MB |
| 3 | 44602.99 ms | 7.6 | 6.0 | 166.43 ms | 87045.3 ms | 80080.5 MB |
| 4 | 49900.31 ms | 6.8 | 6.2 | 161.81 ms | 91164.0 ms | 80080.5 MB |
| 5 | 26851.21 ms | 12.6 | 6.1 | 164.12 ms | 68703.4 ms | 80073.0 MB |

```text
TPOT mean=164.48 ms, P50=164.12 ms, P95=166.43 ms
TPOT min=161.81 ms, max=166.43 ms
decode throughput mean=6.08 tok/s
TTFT mean=32364.44 ms
GPU memory mean=80079.00 MB
```

## 4. 对比与判定

| 指标 | Autotune off | Autotune on | 候选变化 | 判断 |
|---|---:|---:|---:|---|
| TPOT mean | 164.83 ms | 164.48 ms | -0.35 ms / -0.21% | 小于噪声 |
| TPOT P50 | 164.72 ms | 164.12 ms | -0.60 ms / -0.36% | 小于噪声 |
| TPOT P95 | 166.40 ms | 166.43 ms | +0.03 ms / +0.02% | 无改善 |
| Decode throughput mean | 6.06 tok/s | 6.08 tok/s | +0.33% | 小于显示/采样精度 |
| GPU memory mean | 80056.2 MB | 80079.0 MB | +22.8 MB | 略增 |

两组 TPOT 区间高度重叠。候选均值的 `0.21%` 改善由 run 4 的
`161.81 ms` 低值显著拉动，而候选 P95 没有改善。该差异不足以覆盖当前运行波动，
也没有达到“可重复改善”的保留标准。

TTFT 在两组都存在秒级到几十秒级抖动，且候选 TTFT 均值更差；它不能支持
autotune 有效。CUDA Graph 显存两组均约 `1.18--1.20 GiB/stage`，无明显变化。

## 5. 结论

FlashInfer autotune 在当前 Kimi-K3、单请求 batch=1、256-token decode workload
上可以成功初始化，但没有产生稳定端到端收益。按无效候选处理：

```text
ENABLE_FLASHINFER_AUTOTUNE=False
```

后续继续从 autotune 关闭、custom all-reduce 开启、shared-expert overlap 开启、
`31,31,31` 的有效配置出发。

## 6. 限制

- 结论只覆盖当前 latency workload；更高 decode batch 可能触发不同 kernel 选择。
- 五样本 P95 按现有报告规则取最大值。
- CUPTI 不可用，不能直接列出 autotune 选择的 kernel 名与单 kernel 时间。
- 当前日志中的 FlashInfer all-reduce multicast 警告属于无 NVSwitch 拓扑的回退；
  本轮 autotune 并未使该不可用 backend 变为可用。
