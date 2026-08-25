# CUDA Device Max Connections A/B

日期：2026-08-25  
结论：`CUDA_DEVICE_MAX_CONNECTIONS=1` 没有稳定降低 decode TPOT，不保留该设置。

## 1. 优化假设

`CUDA_DEVICE_MAX_CONNECTIONS` 控制一个 CUDA context 可使用的并发硬件工作队列数量。
设置为 `1` 会让 compute、custom all-reduce 和辅助 stream 的提交顺序更确定，可能改善
通信/计算依赖的调度；也可能减少本来有效的并发。本轮比较：

```text
control:   未设置 CUDA_DEVICE_MAX_CONNECTIONS（CUDA 默认值）
candidate: CUDA_DEVICE_MAX_CONNECTIONS=1
```

候选首次相对旧稳定基线出现 `0.82%` 的 TPOT 均值改善。由于幅度小于历史运行间波动，
随后按反向顺序补跑一组紧邻默认控制，避免将时间窗口漂移误判成优化。

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
breakable CUDA Graph=auto-enabled
shared-expert stream overlap=enabled
disable_custom_all_reduce=False
enable_flashinfer_autotune=False
torch profiler=disabled
CUDA_DEVICE_MAX_CONNECTIONS=<unset or 1>  # 唯一变量
NCCL_IB_DISABLE=1
NCCL_SOCKET_IFNAME==eth0
```

候选在三节点使用的等价命令：

```bash
env -u VLLM_USE_BREAKABLE_CUDAGRAPH \
  PP_OUTPUT_LEN=256 PP_MAX_NUM_SEQS=384 \
  PP_NUM_WARMUPS=2 PP_NUM_REPEATS=5 \
  VLLM_K3_TIMING=0 PP_ENFORCE_EAGER=0 \
  VLLM_DISABLE_SHARED_EXPERTS_STREAM=0 \
  CUDA_DEVICE_MAX_CONNECTIONS=1 \
  PP_TORCH_PROFILER_DIR= PP_PROFILE_ONLY=0 \
  bash phase1/launch_pp.sh <node_rank> 31,31,31 \
  v9_cuda_max_connections_1
```

紧邻控制组将该变量清除，其余命令完全相同：

```bash
env -u VLLM_USE_BREAKABLE_CUDAGRAPH \
  -u CUDA_DEVICE_MAX_CONNECTIONS \
  PP_OUTPUT_LEN=256 PP_MAX_NUM_SEQS=384 \
  PP_NUM_WARMUPS=2 PP_NUM_REPEATS=5 \
  VLLM_K3_TIMING=0 PP_ENFORCE_EAGER=0 \
  VLLM_DISABLE_SHARED_EXPERTS_STREAM=0 \
  PP_TORCH_PROFILER_DIR= PP_PROFILE_ONLY=0 \
  bash phase1/launch_pp.sh <node_rank> 31,31,31 \
  v9_cuda_max_connections_default_recheck
```

两组启动日志均确认默认 Kimi-K3 breakable graph 配置：

```text
CompilationMode.NONE
CUDAGraphMode.FULL_AND_PIECEWISE
custom_ops=['all']
fuse_norm_quant=True
fuse_act_quant=True
disable_custom_all_reduce=False
```

## 3. 逐次结果

### 3.1 旧稳定默认基线

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
```

### 3.2 `CUDA_DEVICE_MAX_CONNECTIONS=1` 候选

```text
experiment_id=v9_cuda_max_connections_1
result_dir=phase0/profile_results/20260825_085844
remote_logs=/home/tjy/kimi_bench/infra_v9_cuda_max_connections1_stage{0,1,2}.log
```

| Run | TTFT | Prefill tok/s | Decode tok/s | TPOT | Total | Avg GPU memory |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 22911.41 ms | 14.8 | 6.1 | 164.89 ms | 64959.3 ms | 79892.1 MB |
| 2 | 4826.23 ms | 72.1 | 6.1 | 163.52 ms | 46525.8 ms | 79892.1 MB |
| 3 | 6950.52 ms | 49.5 | 6.2 | 162.11 ms | 48289.1 ms | 79892.1 MB |
| 4 | 13562.35 ms | 25.1 | 6.2 | 162.02 ms | 54879.4 ms | 79892.1 MB |
| 5 | 12029.63 ms | 28.3 | 6.1 | 164.84 ms | 54066.6 ms | 79892.1 MB |

```text
TPOT mean=163.48 ms, P50=163.52 ms, P95=164.89 ms
TPOT min=162.02 ms, max=164.89 ms
decode throughput mean=6.14 tok/s
TTFT mean=12056.03 ms
GPU memory mean=79892.10 MB
```

### 3.3 紧邻默认复核

```text
experiment_id=v9_cuda_max_connections_default_recheck
result_dir=phase0/profile_results/20260825_091942
remote_logs=/home/tjy/kimi_bench/infra_v9_cuda_connections_default_stage{0,1,2}.log
```

| Run | TTFT | Prefill tok/s | Decode tok/s | TPOT | Total | Avg GPU memory |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 19703.21 ms | 17.2 | 6.2 | 161.94 ms | 60998.8 ms | 80041.5 MB |
| 2 | 22393.53 ms | 15.1 | 6.2 | 162.07 ms | 63723.9 ms | 80041.5 MB |
| 3 | 41956.58 ms | 8.0 | 6.2 | 161.97 ms | 83259.8 ms | 80041.5 MB |
| 4 | 34940.68 ms | 9.7 | 5.8 | 171.35 ms | 78637.4 ms | 80041.5 MB |
| 5 | 20523.51 ms | 16.5 | 6.2 | 160.19 ms | 61373.3 ms | 80039.0 MB |

```text
TPOT mean=163.50 ms, P50=161.97 ms, P95=171.35 ms
TPOT min=160.19 ms, max=171.35 ms
decode throughput mean=6.12 tok/s
TTFT mean=27903.50 ms
GPU memory mean=80041.00 MB
```

## 4. 对比与判定

| 指标 | Old default | Connections=1 | Adjacent default | 候选相对相邻控制 |
|---|---:|---:|---:|---:|
| TPOT mean | 164.83 ms | 163.48 ms | 163.50 ms | -0.02 ms / -0.01% |
| TPOT P50 | 164.72 ms | 163.52 ms | 161.97 ms | +1.55 ms / +0.96% |
| TPOT P95 | 166.40 ms | 164.89 ms | 171.35 ms | -6.46 ms / -3.77% |
| Decode throughput mean | 6.06 tok/s | 6.14 tok/s | 6.12 tok/s | +0.33% |
| Graph capture, stage 0 | 31 s | 20 s | 29 s | -9 s |

候选相对旧基线的 `1.35 ms` 改善没有在相邻控制中保持：相邻默认控制均值只比候选
慢 `0.02 ms`，实际上相同。候选 P50 还比相邻控制慢 `1.55 ms`。候选 P95 较好是因为
相邻控制 run 4 出现 `171.35 ms` 单点，而不是候选整体分布向下移动。

`CUDA_DEVICE_MAX_CONNECTIONS=1` 将 stage 0 graph capture 从相邻默认的 29 秒缩短到
20 秒，但 graph capture 是一次性启动成本，不在稳态 TPOT 临界路径中。本项目以 decode
性能为首要目标，不能用这 9 秒启动收益替代稳态收益。

## 5. 容量与显存

候选与相邻控制的 stage 0 均为：

```text
Available KV cache memory: 12.59 GiB
GPU KV cache size: 825474 tokens
CUDA graph memory: 1.18 GiB
```

因此该变量没有提供 KV 容量或 graph 显存收益。Node 0 请求结束快照相差约 149 MB，
但没有对应的可用 KV cache 变化，不作为有效容量改善。

## 6. 结论

`CUDA_DEVICE_MAX_CONNECTIONS=1` 只在本轮观察到更快的 graph capture，没有稳定 decode
收益，也没有 KV 容量收益。按无效候选处理：后续不设置该变量，从默认 CUDA 连接数
继续实验。

本轮没有修改仓库代码或 site-packages；候选值只存在于已结束进程环境中。

## 7. 限制

- 相邻 A/B 各五次，仍不足以证明候选是否稳定降低尾延迟；目前 P95 差异由单个异常点
  决定，不能作为保留依据。
- TTFT 在三组均有秒级到几十秒级抖动，不用于判定 CUDA 工作队列对 decode 的效果。
- 本轮仅覆盖单请求 batch=1；高并发时多个 CUDA stream 的排队特征可能不同。
