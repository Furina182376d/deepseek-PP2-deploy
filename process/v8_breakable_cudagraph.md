# Breakable CUDA Graph A/B

日期：2026-08-25  
结论：保留 Kimi-K3 自动开启的 breakable CUDA Graph；显式关闭会降低 decode 性能，
并显著压缩 KV cache 容量。

## 1. 优化假设

Kimi-K3 的 GDN/KDA 路径包含不能直接放入普通 CUDA Graph 的区间。vLLM 0.27.1
会为该模型自动设置：

```text
VLLM_USE_BREAKABLE_CUDAGRAPH=1
```

本轮检验显式关闭这一模型专用路径是否能减少 graph break 或降低 replay 开销：

```text
control:   未显式设置，vLLM 对 Kimi-K3 自动置 1
candidate: VLLM_USE_BREAKABLE_CUDAGRAPH=0
```

这并不是 eager 对比。候选仍使用 `FULL_AND_PIECEWISE` CUDA Graph，但配置路径从
breakable graph 的 `CompilationMode.NONE` 切换为普通
`CompilationMode.VLLM_COMPILE`。

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
enable_flashinfer_autotune=False
torch profiler=disabled
VLLM_USE_BREAKABLE_CUDAGRAPH=<auto-1 or explicit-0>  # 唯一变量
NCCL_IB_DISABLE=1
NCCL_SOCKET_IFNAME==eth0
```

候选在三节点使用的等价命令为：

```bash
env PP_OUTPUT_LEN=256 PP_MAX_NUM_SEQS=384 \
  PP_NUM_WARMUPS=2 PP_NUM_REPEATS=5 \
  VLLM_K3_TIMING=0 PP_ENFORCE_EAGER=0 \
  VLLM_DISABLE_SHARED_EXPERTS_STREAM=0 \
  VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
  PP_TORCH_PROFILER_DIR= PP_PROFILE_ONLY=0 \
  bash phase1/launch_pp.sh <node_rank> 31,31,31 \
  v8_breakable_cudagraph_off
```

控制组直接使用紧邻本轮之前、运行配置相同的稳定基线
`v7_fi_autotune_off`，没有为控制组重复加载模型。

## 3. 实际配置差异

控制组自动开启 breakable graph 后：

```text
CompilationMode.NONE
CUDAGraphMode.FULL_AND_PIECEWISE
custom_ops=['all']
ir_enable_torch_wrap=False
norm priority=['vllm_c', 'native']
fuse_norm_quant=True
fuse_act_quant=True
```

显式关闭后：

```text
CompilationMode.VLLM_COMPILE
CUDAGraphMode.FULL_AND_PIECEWISE
custom_ops=['none']
ir_enable_torch_wrap=True
norm priority=['native']
fuse_norm_quant=False
fuse_act_quant=False
```

所有 rank 均出现以下警告：

```text
torch.compile is turned on, but model /data/models/Kimi-K3 does not support it
```

因此候选不是“使用普通 compile 获得更多融合”，而是进入模型不支持的普通 compile
配置，同时丢失 breakable 路径启用的一组 vLLM custom op/fusion 设置。

## 4. 逐次结果

### 4.1 自动开启控制组

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

### 4.2 显式关闭候选

```text
experiment_id=v8_breakable_cudagraph_off
result_dir=phase0/profile_results/20260825_004809
remote_logs=/home/tjy/kimi_bench/infra_v8_breakable_cudagraph/off_stage{0,1,2}.log
```

| Run | TTFT | Prefill tok/s | Decode tok/s | TPOT | Total | Avg GPU memory |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 43852.14 ms | 7.7 | 6.0 | 165.82 ms | 86137.5 ms | 79149.0 MB |
| 2 | 60127.50 ms | 5.6 | 6.0 | 165.78 ms | 102403.7 ms | 79149.0 MB |
| 3 | 36021.13 ms | 9.4 | 5.8 | 171.67 ms | 79799.0 ms | 79149.0 MB |
| 4 | 21055.24 ms | 16.1 | 6.0 | 167.47 ms | 63761.4 ms | 79149.0 MB |
| 5 | 19107.10 ms | 17.7 | 6.0 | 167.44 ms | 61806.3 ms | 79149.0 MB |

```text
TPOT mean=167.64 ms, P50=167.44 ms, P95=171.67 ms
TPOT min=165.78 ms, max=171.67 ms
decode throughput mean=5.96 tok/s
TTFT mean=36032.62 ms
GPU memory mean=79149.00 MB
```

## 5. 对比与判定

| 指标 | Auto breakable | Explicit off | 候选变化 | 判断 |
|---|---:|---:|---:|---|
| TPOT mean | 164.83 ms | 167.64 ms | +2.81 ms / +1.70% | 回退 |
| TPOT P50 | 164.72 ms | 167.44 ms | +2.72 ms / +1.65% | 回退 |
| TPOT P95 | 166.40 ms | 171.67 ms | +5.27 ms / +3.17% | 明显回退 |
| Decode throughput mean | 6.06 tok/s | 5.96 tok/s | -1.65% | 回退 |
| TTFT mean | 22885.51 ms | 36032.62 ms | +57.46% | 回退且波动大 |
| Node 0 GPU memory mean | 80056.2 MB | 79149.0 MB | -907.2 MB | 不能抵消容量损失 |

TTFT 在两组中都有较大系统抖动，因此不能只凭 TTFT 判断；TPOT 均值、中位数和尾部
则一致变差，候选没有延迟收益。

## 6. KV cache 与图内存

| Stage | Auto 可用 KV | Off 可用 KV | Auto KV tokens | Off KV tokens | Auto graph | Off graph |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 12.59 GiB | 9.82 GiB | 825474 | 696494 | 1.18 GiB | 1.05 GiB |
| 1 | 7.76 GiB | 6.71 GiB | 737992 | 622592 | 1.20 GiB | 1.03 GiB |
| 2 | 6.83 GiB | 5.77 GiB | 656072 | 553494 | 1.19 GiB | 1.06 GiB |

候选的实际 graph pool 略小，但普通 compile 路径的 memory profiling/工作区占用导致可用
KV cache 明显下降：stage 0 少 `2.77 GiB`，stage 1/2 各少约 `1.05/1.06 GiB`。
这会降低可承载的并发或上下文容量，不属于可接受的显存交换。

## 7. 结论

显式设置 `VLLM_USE_BREAKABLE_CUDAGRAPH=0` 对当前 Kimi-K3 不构成优化：decode
TPOT 回退、P95 恶化、TTFT 没有改善，且 KV cache 容量显著下降。按无效候选处理，
不保留该环境变量；后续继续使用 vLLM 对 Kimi-K3 自动启用的 breakable graph：

```text
VLLM_USE_BREAKABLE_CUDAGRAPH=1  # 由 vLLM 自动设置
```

本实验没有修改仓库代码或 site-packages，候选设置只存在于已结束进程的环境中。

## 8. 限制

- 控制组复用紧邻的稳定基线，而不是在候选后再次重复一轮控制组。
- 五样本 P95 按现有报告规则取最大值。
- Node 0 的 GPU memory 是请求结束后的进程占用快照，不等于可用 KV cache；容量判断
  使用三个 stage 启动日志中的 `Available KV cache memory` 与 token 数。
- 本轮只覆盖单请求 batch=1；更高 batch 仍应优先保留模型官方支持的 graph 路径。
