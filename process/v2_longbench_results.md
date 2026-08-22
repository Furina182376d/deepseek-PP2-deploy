# Kimi-K3 真实长文本 Workload 实测

测试配置：`PP=3、TP=8`，三节点 H20，模型 `/data/models/Kimi-K3`，vLLM 0.27.1。
Workload 使用 LongBench 原始 JSONL 数据，而不是重复 filler prompt。数据同步在三台节点的
`/home/tjy/benchmarks/longbench/data/`，runner 为 `phase1/run_phase2_longbench.py`。

## 实测结果

本轮每个任务取 1 个样本，`max_tokens=128`、`temperature=0`、`ignore_eos=True`。
成功完成的两个真实长文本请求如下：

| 任务 | prompt tokens | output tokens | wall total | output throughput |
|---|---:|---:|---:|---:|
| qmsum | 6,334 | 128 | 85.960 s | **1.49 tok/s** |
| gov_report | 4,169 | 128 | 57.490 s | **2.23 tok/s** |

output throughput 按 `output_tokens / wall_total` 计算。原始 rank-0 日志为
[`rank0.log`](k3_longbench_raw/rank0.log)。

## 重要限制

这轮测试的 vLLM external-launcher PP 路径没有向 `out.metrics` 填充
`first_token_latency/first_token_ts/last_token_ts`，所以 runner 输出的 `ttft_ms`、
`prefill_ms`、`tpot_ms` 为 0；不能把这些 0 当作真实耗时。可信的本轮指标是 prompt/output
token 数和 wall-clock 总耗时，输出速度也可由二者直接计算。

此外，日志显示首个真实请求触发了 Triton JIT：

```text
Triton kernel JIT compilation during inference:
_causal_conv1d_fwd_kernel
_gather_initial_states_kernel
layer_norm_gated_fwd_kernel
```

因此 qmsum/gov_report 结果包含首次真实 workload 的 JIT 冷启动开销，不应直接作为稳态
TPOT 基线。模型初始化和 CUDA graph 捕获本身也不计入上述 `wall total`，但首请求 kernel
JIT 计入。

## 判断

这次运行已经证明 phase2 方案可以在真实 LongBench 长文本上完成 PP=3/TP=8 推理，并得到
真实 prompt 长度和端到端请求耗时。要得到可用于性能比较的稳态 TTFT/TPOT，下一轮应：

1. 在同一任务/长度上先发送一次 warm-up，再发送至少 3 次测量请求；
2. 在 runner 中直接记录每个 token 的 wall-clock 或修复 external-launcher 的 vLLM metrics；
3. 将 `qmsum`、`gov_report`、`narrativeqa` 等任务按真实 token 长度分桶；
4. 同时记录 stage model event、GPU busy 和 JIT warning，区分长 prefill 与 decode 固定开销。

当前最可靠的结论是：真实长文本请求在该部署上的端到端速度约为 **1.49--2.23 output tok/s**，
但这轮数据仍是冷启动样本，不能据此宣称稳态性能已经测完。
