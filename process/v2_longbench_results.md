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

## 与既有 decode 瓶颈结论的关系

需要把“稳态 decode 结论”和“本轮长文本端到端结果”分开解释：

### 稳态 decode：原结论仍成立

此前启用 K3 CUDA Event 的稳态 decode 测量已经确认：

```text
PP stage 0 model: 161.27 ms/step
MoE/MLP:          106.05 ms = 65.8%
Attention:         48.39 ms = 30.0%
Other:              6.82 ms =  4.2%
```

目前没有新的证据推翻该结论。对 decode 临界路径而言，最慢的仍是 PP stage 0，最大
组成仍是 MoE/MLP。

### 本轮 LongBench：没有重新验证 decode 内部分解

本轮真实长文本测试没有启用之前的 K3 Event 细分插桩，且 external-launcher PP 路径没有
提供有效的 `TTFT/TPOT` metrics。因此：

- `qmsum` 的 6334 prompt tokens、128 output tokens、85.96 s 是端到端 wall-clock；
- `gov_report` 的 4169 prompt tokens、128 output tokens、57.49 s 也是端到端 wall-clock；
- `1.49--2.23 output tok/s` 不能直接与之前稳态 decode 的约 `5.9--6.2 tok/s` 对比；
- 输出速度不是 decode-only 速度，因为它包含长文本 prefill 和首次真实 workload 的 JIT。

本轮日志明确出现了 `_causal_conv1d_fwd_kernel`、`_gather_initial_states_kernel` 和
`layer_norm_gated_fwd_kernel` 的 Triton JIT warning。因而长文本请求的总时间很可能被
长 prefill 与冷启动 JIT 显著污染；仅凭本轮 wall-clock，不能断言端到端总时间仍主要由
decode 贡献，也不能断言真实 workload 下 decode 仍严格保持 `65.8%/30.0%` 的内部比例。

## 当前最终判断

> 真实 workload 证明系统在长文本场景下仍然很慢，但本轮没有证明端到端总时间仍由 decode
> 主导；稳态 decode 的 MoE/MLP 瓶颈结论仍有效，但其内部时间分布尚未在真实 LongBench
> 请求上重新测量。

要完成这一验证，应对同一 LongBench 样本先 warm-up，再重复至少 3 次，并同时启用 K3
CUDA Event，记录 prefill、decode、每步 MoE/attention 以及有效的 TTFT/TPOT。只有这样
才能判断真实长文本 workload 下 decode 的内部时间分布是否仍为约 `65.8%/30.0%`。




既然你说下一步应该做一次真正可比较的 workload 测试：

    1. 同一 LongBench 样本先 warm-up；
    2. 再重复执行 3 次以上；
    3. 启用 K3 CUDA Event 插桩；
    4. 修复或绕过 PP external-launcher 的 metrics 缺失；
    5. 分别记录 prefill、decode 和每个 decode step 的 MoE/attention 时间。

    这样才能回答“真实长文本 workload 下，decode 的内部时间分布是否仍然是 65.8%/30.0%”。

新打开这段历史对话后，直接说：

  继续之前暂停的 LongBench warm-up 对照测试。
  请先检查三台节点是否仍有 runner 进程，再采集结果。

  如果历史对话不可见，也可以在项目目录中重新开始，并让 Codex 先阅读：

  process/v2_longbench_results.md
  process/v2_decode_breakdown.md
  process/v2_future.md
  phase1/run_phase2_longbench.py

  然后说明：

  继续执行真实 LongBench workload：
  warm-up 1 次，重复 3 次，开启 VLLM_K3_TIMING=1，
  分析 prefill、decode、MoE/attention 时间分布。

  当前关键状态和实验目标都已保存在项目文件中。