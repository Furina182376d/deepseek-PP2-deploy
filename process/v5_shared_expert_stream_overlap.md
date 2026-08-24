# Shared Expert CUDA Stream Overlap A/B

日期：2026-08-24  
结论：保留 vLLM 默认的 shared-expert 辅助 CUDA stream；禁用 overlap 没有收益。

## 1. 优化假设

`v4_moe_profilling.md` 的 CUDA Event 结果表明，shared expert 约占单次 MoE
调用 GPU 时间的 26%。当前 vLLM 0.27.1 默认在 token 数不超过 256 时，把
shared expert 放到辅助 CUDA stream，并与 router、dispatch 和 routed expert
路径重叠：

```text
VLLM_DISABLE_SHARED_EXPERTS_STREAM=0                 # 默认，启用辅助 stream
VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=256       # 默认阈值
```

本实验测试这一默认策略在当前 Kimi-K3、单请求 decode workload 上是否真的有益。
候选设置 `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`，使 shared expert 回到主 stream
串行执行。两组分别初始化模型并分别捕获 CUDA Graph，避免复用错误的 graph。

## 2. 固定实验设置

基线与候选只有 `VLLM_DISABLE_SHARED_EXPERTS_STREAM` 不同，其余设置完全相同：

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
PP_TORCH_PROFILER_DIR=
PP_PROFILE_ONLY=0
enable_flashinfer_autotune=False
distributed_executor_backend=external_launcher
CUDA Graph=FULL_AND_PIECEWISE
VLLM_USE_BREAKABLE_CUDAGRAPH=auto-enabled
disable_custom_all_reduce=False
NCCL_IB_DISABLE=1
NCCL_SOCKET_IFNAME==eth0
```

启动环境的等价命令如下，三台节点分别使用 node rank 0、1、2：

```bash
env \
  PP_OUTPUT_LEN=256 \
  PP_MAX_NUM_SEQS=384 \
  PP_NUM_WARMUPS=2 \
  PP_NUM_REPEATS=5 \
  VLLM_K3_TIMING=0 \
  PP_ENFORCE_EAGER=0 \
  PP_TORCH_PROFILER_DIR= \
  PP_PROFILE_ONLY=0 \
  VLLM_DISABLE_SHARED_EXPERTS_STREAM=<0-or-1> \
  bash phase1/launch_pp.sh <node_rank> 31,31,31 <experiment_id>
```

三节点运行文件在实验前做了 SHA-256 校验，以下文件完全一致：

```text
phase1/config_pp.py
phase1/run_pp.py
phase1/profile_dsv4_pp.py
phase1/launch_pp.sh
phase0/results_utils.py
phase0/prompt_utils.py
phase0/gpu_utils.py
```

## 3. 原始结果

### 3.1 默认 overlap 基线

```text
experiment_id=v5_shared_stream_baseline
VLLM_DISABLE_SHARED_EXPERTS_STREAM=0
result_dir=phase0/profile_results/20260824_224727
remote_logs=/home/tjy/kimi_bench/infra_v5_shared_expert/baseline_stage{0,1,2}.log
```

| Run | TTFT | Prefill throughput | Decode throughput | TPOT | Total | Avg GPU memory |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 43300.73 ms | 7.8 tok/s | 6.0 tok/s | 165.33 ms | 85462.8 ms | 80087.0 MB |
| 2 | 36524.57 ms | 9.2 tok/s | 6.0 tok/s | 167.91 ms | 79342.1 ms | 80087.0 MB |
| 3 | 81339.00 ms | 4.1 tok/s | 5.8 tok/s | 172.82 ms | 125409.5 ms | 80087.0 MB |
| 4 | 34597.10 ms | 9.8 tok/s | 5.9 tok/s | 168.07 ms | 77457.8 ms | 80087.0 MB |
| 5 | 71133.26 ms | 4.7 tok/s | 6.0 tok/s | 167.47 ms | 113839.3 ms | 80072.0 MB |

汇总：

```text
TPOT mean=168.32 ms, P50=167.91 ms, P95=172.82 ms
TPOT min=165.33 ms, max=172.82 ms
TTFT mean=53378.93 ms
prefill throughput mean=7.12 tok/s
decode throughput mean=5.94 tok/s
total latency mean=96302.30 ms
GPU memory mean=80084.00 MB
```

### 3.2 禁用 overlap 候选

```text
experiment_id=v5_shared_stream_disabled
VLLM_DISABLE_SHARED_EXPERTS_STREAM=1
result_dir=phase0/profile_results/20260824_231354
remote_logs=/home/tjy/kimi_bench/infra_v5_shared_expert/disabled_stage{0,1,2}.log
```

| Run | TTFT | Prefill throughput | Decode throughput | TPOT | Total | Avg GPU memory |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 55299.00 ms | 6.1 tok/s | 5.8 tok/s | 173.90 ms | 99645.2 ms | 80049.5 MB |
| 2 | 97899.17 ms | 3.4 tok/s | 5.8 tok/s | 171.20 ms | 141555.7 ms | 80049.5 MB |
| 3 | 19026.13 ms | 17.8 tok/s | 5.9 tok/s | 169.82 ms | 62330.9 ms | 80049.5 MB |
| 4 | 24590.18 ms | 13.8 tok/s | 5.9 tok/s | 169.75 ms | 67877.6 ms | 80049.5 MB |
| 5 | 16340.42 ms | 20.8 tok/s | 5.9 tok/s | 168.73 ms | 59368.5 ms | 80037.0 MB |

汇总：

```text
TPOT mean=170.68 ms, P50=169.82 ms, P95=173.90 ms
TPOT min=168.73 ms, max=173.90 ms
TTFT mean=42630.98 ms
prefill throughput mean=12.38 tok/s
decode throughput mean=5.86 tok/s
total latency mean=86155.58 ms
GPU memory mean=80047.00 MB
```

## 4. A/B 对比

| 指标 | 默认 overlap | 禁用 overlap | 候选相对变化 | 判断 |
|---|---:|---:|---:|---|
| TPOT mean | 168.32 ms | 170.68 ms | +2.36 ms / +1.40% | 变慢 |
| TPOT P50 | 167.91 ms | 169.82 ms | +1.91 ms / +1.14% | 变慢 |
| TPOT P95 | 172.82 ms | 173.90 ms | +1.08 ms / +0.62% | 变慢 |
| Decode throughput mean | 5.94 tok/s | 5.86 tok/s | -1.35% | 变慢 |
| GPU memory mean | 80084 MB | 80047 MB | -37 MB / -0.05% | 无实质差异 |

TTFT 在两组内部都有很大波动，基线范围为 34.60--81.34 秒，候选范围为
16.34--97.90 秒。由于候选轮在基线之后运行，权重文件页缓存和节点瞬时状态也
不同，不能把候选较低的 TTFT 均值归因于禁用 shared-expert stream。TPOT 的五次
样本更稳定，而且 mean、P50、P95 和 decode throughput 四个判据方向一致。

## 5. 结论与后续配置

禁用 shared-expert 辅助 CUDA stream 是无效优化：TPOT 均值回退 1.40%，P50
回退 1.14%，decode throughput 均值下降 1.35%。这与 `v4` 中 shared expert
约占 MoE 时间 26% 的结果一致：当前默认 overlap 已经隐藏了其中一部分时间，
改为串行执行只会延长 decode 临界路径。

因此不设置以下候选变量：

```text
VLLM_DISABLE_SHARED_EXPERTS_STREAM=1
```

下一项优化继续使用默认值 `0`，并从本实验前的有效运行配置继续，而不是从
禁用 overlap 的候选配置继续。当前实验没有修改模型或 vLLM 源码。

## 6. 实验限制

- 每组只有五个正式请求，P95 按现有报告规则取五个样本中的最大值。
- 这是单请求、实际 decode batch=1 的 latency workload；不能代替并发吞吐测试。
- 本轮没有启用 stage CUDA Event，因为诊断插桩会改变 timing 路径；端到端 TPOT
  是本次唯一的主要判据。
- 启动日志多次显示 FlashInfer all-reduce workspace 因当前无 NVSwitch 拓扑而
  无法使用 multicast 并回退。该警告不影响本 A/B 的公平性，因为两组完全相同；
  后续应优先测试实际使用的 custom all-reduce 与 PYNCCL 路径，而不是强制一个
  已知无法初始化的 FlashInfer multicast backend。
