# Kimi-K3 Decode 性能优化方向（AI Infra 视角）

## 核心原则

优化应围绕“临界路径上的可消除时间”展开，而不是只看某个 kernel 的理论 FLOPS。
当前实测表明，问题更像是 **MoE decode 的固定开销 + PP stage 负载不均衡**，而不是 H20 理论算力不足。

## 当前证据

三个 PP stage 的稳态 model forward 平均时间为：

```text
stage 0: 161.27 ms
stage 1:  92.81 ms
stage 2: 103.60 ms
```

stage 0 是 decode 临界路径。其内部：

```text
MoE/MLP: 106.05 ms = 65.8%
Attention: 48.39 ms = 30.0%
Other:     6.82 ms =  4.2%
```

其中 KDA 层 MoE/MLP 单独占 `81.53 ms`，即整个临界路径的 `50.6%`。

## 优化优先级

### 1. 先拆分 MoE/MLP

当前 MoE/MLP 是模块级计时，下一步应继续拆成：

- router/gating；
- token dispatch、permutation；
- MARLIN expert GEMM；
- shared expert；
- token combine；
- TP collective。

需要确认 GPU 时间究竟花在 GEMM、kernel launch、routing 还是通信上。当前不能把整个
MoE/MLP 时间直接归因于 MARLIN。

### 2. 优化 batch=1 的 MoE 路径

decode batch=1 时，重点不是提高单个 GEMM 的峰值 TFLOPS，而是降低每 token 的固定开销：

- 合并 expert kernel，减少 kernel 数量和 launch 次数；
- 调整 `moe_align_block_size`、group size、tile size；
- 对比 MARLIN、CUTLASS、Triton 等 backend；
- 减少 token permutation 和 intermediate buffer；
- 检查 `top-k=16`、`896 experts` 对单 token 调度的影响；
- 尝试融合或重叠 shared expert 与 routed expert。

### 3. 重新平衡 PP stage

目标不是让每个 stage 的层数相同，而是最小化：

```text
max(stage_time)
```

stage 1 有约 68 ms 的相对余量，stage 2 也快于 stage 0。可以评估将部分 KDA 层从
stage 0 迁移到 stage 1，但必须重新测量 PP 通信、流水线 bubble 和端到端 TPOT。

### 4. 确认 TP collective 是否隐藏在 MoE 时间内

MoE/MLP 区间可能包含 TP all-reduce、all-gather 或 MoE dispatch collective。需要单独统计 NCCL kernel，确认：

- collective 的实际占比；
- 是否存在同步等待；
- 是否可以通信计算重叠；
- TP 分组或 MoE parallelism 是否需要调整；
- 是否存在不必要的全量 all-reduce。

若 NCCL 占比较小，应继续优化计算和调度；若占比较大，才转向通信拓扑、collective 算法和 overlap。

### 5. 提高 decode 的有效 batch size

单请求 batch=1 对 MoE 不友好。生产系统应通过 continuous batching 合并多个 decode 请求，摊薄：

- kernel launch；
- routing；
- permutation；
- NCCL collective；
- expert GEMM 的低利用率。

这是吞吐优化的重要方向，但可能增加单请求延迟。因此必须分别衡量 throughput 和 TPOT。

### 6. 减少 CUDA graph/eager 边界

需要确认：

- decode 是否完全命中 full CUDA graph；
- 哪些 KDA/GDN 层被迫 eager；
- graph replay 与 eager kernel 的时间差；
- 是否可以扩大 graph 覆盖范围；
- 是否存在更适合动态 MoE 的 graph 方案。

频繁的 eager break 可能带来额外 Python、调度和 kernel launch 成本。

### 7. 暂不优先优化 logits、采样和 PP 网络

当前 logits 约 `0.7 ms`，远小于 `161 ms` 的临界路径；跨节点 PP 通信也无法解释现有 TPOT。
这些方向的收益上限低，不应作为第一优先级。

## 推荐执行顺序

```text
MoE 内部 CUDA Event / NCCL 细分
    ↓
确认 MARLIN、routing、collective 的占比
    ↓
优化 batch=1 kernel 路径
    ↓
重新平衡 PP stage
    ↓
测试 continuous batching 对吞吐的收益
    ↓
检查 CUDA graph 覆盖范围
```

## 预期判断标准

每一步优化都应至少记录：

- stage 0/1/2 model mean、P50、P95；
- 端到端 TPOT；
- decode throughput；
- MoE 子项时间；
- NCCL 时间及通信占比；
- GPU busy、kernel 数量和 kernel launch 间隙。

只有当 `max(stage_time)`、TPOT 或目标吞吐出现可重复改善时，才算真正优化成功。
