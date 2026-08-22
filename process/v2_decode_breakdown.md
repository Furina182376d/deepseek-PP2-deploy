# Kimi-K3 Decode 阶段时间消耗实测报告

测试日期：2026-08-22。三节点、每节点 8 张 H20；每个节点对应一个 PP stage。

## 结论摘要

在 `PP=3、TP=8` 的部署上，稳态 decode 的确定性瓶颈是 **PP stage 0 的本地 GPU model execution**。stage 0 的一次 model forward 平均耗时 **161.27 ms**，与此前端到端观测的 **TPOT 161--174 ms** 基本闭合，因此它决定了单请求 decode 的节拍。

stage 0 内部，最大耗时项是 **MoE/MLP 路径**：`106.05 ms`，占 **65.8%**；attention 为 `48.39 ms`，占 **30.0%**；其它约 `6.82 ms`，占 **4.2%**。其中 24 个 KDA 层的 MoE/MLP 单独耗时 `81.53 ms`，占整个 decode 临界路径 **50.6%**。

## 方法

- 使用 vLLM 内置 Kimi-K3 NVIDIA 实现中的 CUDA Event 插桩。
- 每层分别记录 attention、MoE/MLP 及层内其余操作，并记录整个 stage model forward。
- 单请求生成 64 token（API 返回 `completion_tokens=64`、`finish_reason=length`）。
- 三个 stage 各得到 8 TP rank × 64 decode step = 512 个样本。
- 剔除每个 rank 的首个 decode step（明显的冷启动异常点），稳态样本为每 stage 504 个。
- 原始日志位于 `process/k3_decode_timing_raw/stage{0,1,2}.log`，解析器为
  `tools/analyze_k3_decode_events.py`。

分析脚本只纳入 `tok=1` 的 decode 记录，并剔除每个 rank 的首个冷启动 step；因此下表不混入 prefill 或初始化时间。

## 结果

| PP stage | model mean | P50 | P95 | attention mean | MoE/MLP mean |
|---|---:|---:|---:|---:|---:|
| stage 0（24 KDA + 7 full） | **161.27 ms** | 160.20 | 166.10 | 48.39 | **106.05** |
| stage 1（23 KDA + 8 full） | 92.81 ms | 92.20 | 97.70 | 27.87 | 62.26 |
| stage 2（22 KDA + 9 full） | 103.60 ms | 91.10 | 143.50 | 31.84 | 68.49 |

stage 0 比 stage 1、stage 2 分别慢约 `68.46 ms`、`57.67 ms`。流水线每个
decode step 必须等待最慢 stage，因此 stage 0 是临界路径。其 161.27ms 与此前观测的
TPOT 161-174ms 闭合，以下分布就是当前 decode 瓶颈的主体：

| stage 0 组成 | mean | 占 model |
|---|---:|---:|
| KDA 层 MoE/MLP | **81.53 ms** | **50.6%** |
| KDA attention | 31.72 ms | 19.7% |
| full-attention 层 MoE/MLP | 24.53 ms | 15.2% |
| full attention | 16.67 ms | 10.3% |
| 层内其余操作 | 5.76 ms | 3.6% |
| 未归入层边界的模型开销 | 1.06 ms | 0.7% |

合并后：MoE/MLP 106.05ms（65.8%），attention 48.39ms（30.0%），其余约
6.82ms（4.2%）。stage 2 的 logits P50 为 0.7ms，不是瓶颈。

同类项目的计算为：

```text
MoE/MLP  = 81.53 + 24.53 = 106.05 ms = 65.8%
Attention = 31.72 + 16.67 =  48.39 ms = 30.0%
Other     ≈  6.82 ms                  =  4.2%
```

这是 decode 子阶段的直接排序：**MoE/MLP > attention >> other**。

## 结论

decode 的第一瓶颈是 **PP stage 0 的 MoE/MLP 路径**，尤其是 24 个 KDA 层中的
MoE/MLP：单独占整个 decode 临界路径约 50.6%。attention 是第二大项，占约 30%。
跨节点 PP 和 logits 无法解释现有 TPOT；最慢 stage 的纯 GPU model event 已经覆盖
约 161ms，与端到端 TPOT 相符。

这里的 MoE/MLP 区间包含该模块内部的 expert 路由、token dispatch、MARLIN expert
kernel、shared expert 以及模块内 TP collective。当前证据确认了 MoE/MLP 这一级的
总耗时，但尚未声称 MARLIN 或 NCCL 各自的单独耗时。若继续优化，应在该模块内部
增加 CUDA Event，分别测量 router、dispatch、MARLIN GEMM、combine、shared expert
和 TP collective，而不是继续把主要精力放在 PP 网络或 logits 上。

## 最终判断

对当前配置，decode 时间分布已经被实测锁定：**约 65.8% 花在 MoE/MLP，约
30.0% 花在 attention，约 4.2% 是其它模型操作；其中 KDA 层 MoE/MLP 单独贡献约
50.6%。** 优化优先级应从 stage 0 的 KDA MoE/MLP 内部开始。
