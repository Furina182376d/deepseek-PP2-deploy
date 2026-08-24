# Kimi-K3 PP Stage 重平衡实验

测试日期：2026-08-23。部署配置为三节点、每节点 8 张 H20，`PP=3、TP=8`。

## 背景

原始等层数分配为：

```text
stage 0: 31 层，model forward 161.27 ms
stage 1: 31 层，model forward  92.81 ms
stage 2: 31 层，model forward 103.60 ms
```

流水线的 decode 节拍由最慢 stage 决定，即：

```text
max(stage 0 time, stage 1 time, stage 2 time)
```

因此，本轮实验通过修改 PP 分层边界，将部分层从 stage 0 移到其它 stage，目标是降低
`max(stage_time)` 和端到端 TPOT。

## 如何控制三个节点的层数

使用 vLLM 原生环境变量：

```bash
VLLM_PP_LAYER_PARTITION=30,32,31
```

三个节点必须使用完全相同的值。三个数字依次表示 PP rank 0、1、2 负责的 Transformer
层数，总和必须等于 Kimi-K3 的 93 层。

vLLM 根据当前 worker 的 PP rank，自动从这个分区列表中选择对应的连续层区间：

| 节点 | Global ranks | PP rank | TP rank | 层数 | 层编号区间 |
|---|---:|---:|---:|---:|---|
| node 0 | 0--7 | 0 | 0--7 | 30 | `[0, 30)`，即 0--29 |
| node 1 | 8--15 | 1 | 0--7 | 32 | `[30, 62)`，即 30--61 |
| node 2 | 16--23 | 2 | 0--7 | 31 | `[62, 93)`，即 62--92 |

每个节点上的 8 个进程组成该 stage 内部的 `TP=8`。因此，这不是把一个 stage 的层继续
平均分给 8 张 GPU，而是 8 张 GPU 通过 Tensor Parallel 共同执行同一段模型：

```text
node 0 的 8 张 GPU：共同执行前 30 层
node 1 的 8 张 GPU：共同执行中间 32 层
node 2 的 8 张 GPU：共同执行最后 31 层
```

三个节点仍然按以下顺序组成流水线：

```text
输入
  ↓
node 0 / PP stage 0 / layers [0, 30)
  ↓ PP hidden states
node 1 / PP stage 1 / layers [30, 62)
  ↓ PP hidden states
node 2 / PP stage 2 / layers [62, 93)
  ↓
logits
```

### vLLM 内部的区间计算

vLLM 的 `get_pp_indices()` 使用以下逻辑：

```python
start_layer = sum(partitions[:pp_rank])
end_layer = start_layer + partitions[pp_rank]
```

对 `partitions = [30, 32, 31]`：

```text
PP rank 0: start=0,  end=30
PP rank 1: start=30, end=62
PP rank 2: start=62, end=93
```

模型构造时，每个 PP rank 只加载和执行 `[start_layer, end_layer)` 对应的连续层。

## 仓库中的实现

启动脚本 `phase1/launch_pp.sh` 接收：

```text
./launch_pp.sh <node_rank> [layer_partition] [experiment_id]
```

实验使用的三节点命令等价于：

```bash
# node 0
./launch_pp.sh 0 30,32,31 pp_30_32_31

# node 1
./launch_pp.sh 1 30,32,31 pp_30_32_31

# node 2
./launch_pp.sh 2 30,32,31 pp_30_32_31
```

脚本执行以下检查：

1. 第二个参数必须是三个由逗号分隔的正整数；
2. 三个数字的总和必须是 93；
3. 校验通过后，在启动 torchrun 前导出：

```bash
export VLLM_PP_LAYER_PARTITION="30,32,31"
```

所有节点都执行同一个启动脚本，但 `node_rank` 不同。torchrun 生成的 global rank 和
vLLM 进程组共同决定 worker 属于哪个 PP rank，因此无需为每台机器单独修改模型代码。

`phase1/config_pp.py` 还会再次解析和校验分区：

```python
_PP_PARTITION_RAW = os.environ.get("VLLM_PP_LAYER_PARTITION", "")
PP_LAYER_PARTITION = tuple(
    int(value) for value in _PP_PARTITION_RAW.split(",")
)
```

运行结果的 `full_results.json` 会记录：

```json
{
  "pp_layer_partition": [30, 32, 31],
  "pp_experiment_id": "pp_30_32_31"
}
```

这样可以确认结果文件与实际分区一致，避免只根据日志文件名推断配置。

## 实验参数

公平对比轮使用：

```python
# phase1/config_pp.py
CONTEXT_LENGTHS = [512]
```

```bash
PP_OUTPUT_LEN=64
PP_MAX_NUM_SEQS=384
VLLM_K3_TIMING=1
VLLM_PP_LAYER_PARTITION=<待测分区>
```

其中 `PP_MAX_NUM_SEQS=384` 是因为一次 `29,33,31` 启动中，stage 2 仅有 469 个可用
Mamba cache blocks，无法满足原来的 `max_num_seqs=512`。将所有公平对比轮统一为 384，
可以避免 CUDA Graph 初始化失败，并保持候选之间配置一致。

## 实验结果

| 分区 | 运行 | TPOT | Decode rate | 判断 |
|---|---|---:|---:|---|
| `31,31,31` | 历史基线 | 161--174 ms | 5.7--6.2 tok/s | stage 0 瓶颈 |
| `30,32,31` | timed run，`max_num_seqs=512` | 136.56 ms | 7.3 tok/s | 成功 |
| `30,32,31` | `max_num_seqs=384` | 138.49 ms | 7.2 tok/s | 成功 |
| `29,33,31` | `max_num_seqs=384` | 156.81 ms | 6.4 tok/s | stage 1 过载 |
| `30,31,32` | run 1 | 137.02 ms | 7.3 tok/s | 成功但波动较大 |
| `30,31,32` | run 2 | 142.59 ms | 7.0 tok/s | 成功但波动较大 |

重复结果：

```text
30,32,31: mean TPOT = 137.53 ms，range = 1.93 ms
30,31,32: mean TPOT = 139.81 ms，range = 5.57 ms
```

在此前 `output_len=64` 的短测中，`30,32,31` 相对历史 `161--174 ms` 的 TPOT 曾表现出
约 14%--21% 的改善；但该改善没有在后面的 `output_len=256` 长窗口复核中复现。因而这
一数字目前只能作为短测观察，不能作为已确认的稳定端到端收益。继续将 stage 0 的一层
迁移到 stage 1，得到 `29,33,31` 后，短测 TPOT 回退到 156.81 ms，说明 stage 1 成为
新的瓶颈。

## 各候选分区的 Stage Capture

对每个 stage 的 8 个 TP rank，分析器先找到 `tok=1` 的逐层 CUDA Event 记录，再丢弃
每个 rank 的第一个 `tok=1` capture（冷启动/capture 异常点），对剩余样本计算均值：

| 分区 | Stage 0 | Stage 1 | Stage 2 | TPOT | 方向性判断 |
|---|---:|---:|---:|---:|---|
| `30,32,31` | 100.81 ms | 128.10 ms | 118.33 ms | 138.49 ms | rank 波动较大 |
| `29,33,31` | 96.75 ms | 124.85 ms | 101.31 ms | 156.81 ms | stage 0 下降，但 stage 1 成为瓶颈 |
| `30,31,32` run 1 | 100.49 ms | 100.71 ms | 136.90 ms | 137.02 ms | stage 0/1 拉平，stage 2 方差较高 |
| `30,31,32` run 2 | 99.91 ms | 100.69 ms | 116.80 ms | 142.59 ms | stage 0/1 稳定，stage 2 仍是波动来源 |

从层迁移方向看：

```text
31,31,31 → 30,32,31
    stage 0 的原始重负载得到释放，TPOT 明显下降

30,32,31 → 29,33,31
    stage 0 继续下降，但新增层使 stage 1 过载，TPOT 回退

30,32,31 → 30,31,32
    stage 0/1 的 capture 时间接近，但 stage 2 的运行间波动增大
```

因此 stage capture 支持以下边界判断：不应继续把 stage 0 的层迁往 stage 1；
`30,31,32` 虽然在单次 capture 中更接近计算拉平，但重复 TPOT 均值和稳定性仍不如
`30,32,31`。

## 测量限制

K3 的逐层 CUDA Event 在 CUDA Graph capture 时执行；稳态 decode 使用 graph replay，
不会每步重新进入 Python 层 forward。因此新实验中的逐层 stage event 主要用于判断负载
向哪个 stage 转移，不能当作 64 个独立的稳态 replay 样本。

最终分区判断以 vLLM request metrics 中的端到端 TPOT 为主，stage capture 时间只作为
方向性证据。

具体来说，上表每个 stage 只有一个 post-cold capture × 8 个 TP rank，并不是 64 个
decode token 的 64 组 replay 测量。因此，不能用表中的 `max(stage capture)` 直接替代
TPOT，也不能根据 capture 均值之间几毫秒的差异判断最终优劣。候选排序必须以重复的
端到端 TPOT 为准。

## 长窗口复核：未复现短测 TPOT 改善

在候选分区比较之后，对 `30,32,31` 做了较长 decode 窗口和重复测量。该运行的分区配置
确实是重平衡配置，但端到端 TPOT 没有复现此前 64-token 短测中的 `136--138 ms`，而是
回到了原始 `31,31,31` 历史基线的 `161--174 ms` 范围。因此，这一轮应归类为“重平衡
配置下的长窗口复核/非回归结果”，不能称为已经建立了 `30,32,31` 的性能基线。

### 固定配置

```text
PP=3、TP=8，三节点，每节点 8 张 H20
VLLM_PP_LAYER_PARTITION=30,32,31
context_length=512（实际 prompt=336 tokens）
output_len=256，ignore_eos=True
PP_MAX_NUM_SEQS=384
PP_NUM_WARMUPS=2
PP_NUM_REPEATS=5
VLLM_K3_TIMING=0（关闭诊断插桩）
enforce_eager=False（CUDA Graph：FULL_AND_PIECEWISE）
gpu_memory_utilization=0.9，max_model_len=32768，kv_cache_dtype=auto
```

三台节点均使用同一份代码和同一组环境变量，实验 ID 为
`pp_30_32_31_stable_baseline`。本次运行时间戳为 `20260823_163032`，报告位于：

```text
phase0/profile_results/20260823_163032/report.txt
phase0/profile_results/20260823_163032/full_results.json
phase0/profile_results/20260823_163032/tp8.csv
```

原始三节点日志及报告副本已归档到：

```text
process/pp_rebalance/pp_30_32_31_stable_baseline/
```

### Decode TPOT 结果

5 次正式测量（均为 `context=512`、`output=256`）的 TPOT 为：

```text
run 1: 172.96 ms
run 2: 168.43 ms
run 3: 170.44 ms
run 4: 171.50 ms
run 5: 171.36 ms
```

统计结果：

| 指标 | 数值 |
|---|---:|
| mean TPOT | 170.94 ms |
| P50 TPOT | 171.36 ms |
| P95 TPOT | 172.96 ms |
| min / max | 168.43 / 172.96 ms |
| range | 4.53 ms（2.65%） |
| decode throughput（约） | 5.8--5.9 tok/s |

TPOT 范围和 P95 与均值的差异较小，说明本轮运行本身稳定，但它测到的是约 `171 ms` 的
基线级性能，而不是此前短测宣称的重平衡收益。报告元数据中的
`pp_layer_partition=[30,32,31]` 以及三端启动日志中的 `Partition : 30,32,31` 证明配置
已传入；vLLM 的 `get_pp_indices()` 也确实按该分区计算层区间。因此问题不是“忘记设置
分区”，而是该配置在长窗口端到端指标上没有复现短测收益。

需要注意，本轮 TTFT/prefill 受三节点启动后
首次请求和流水线请求边界影响，报告中的 prefill 为 24.7--307.0 s、波动很大；因此本
节不把 prefill 作为可比较数值。若下一阶段优化目标包含 prefill，应另行安排预热后的
独立 prefill 基线，并使用相同的请求调度条件。

后续应先在完全相同的 `output_len=256`、预热次数、重复次数、`max_num_seqs` 和
`VLLM_K3_TIMING` 设置下重跑 `31,31,31` 与 `30,32,31`，再判断 PP rebalance 是否真的
改善端到端 TPOT。此前 `output_len=64` 的 `136--138 ms` 结果只能作为方向性短测，不能
直接与本轮长窗口结果混合排序。

### Controlled recheck

随后按上述完全相同的参数再次运行 `30,32,31`，实验 ID 为
`pp_30_32_31_controlled_recheck`，时间戳为 `20260823_202154`。本轮启动日志确认：

```text
Partition : 30,32,31
Output len: 256
Max seqs  : 384
Warmups   : 2
Repeats   : 5
K3 timing : 0
```

5 次 TPOT 为 `165.69、165.99、169.05、167.01、168.60 ms`，统计为：

| 指标 | 数值 |
|---|---:|
| mean TPOT | 167.27 ms |
| P50 TPOT | 167.01 ms |
| P95 TPOT | 169.05 ms |
| min / max | 165.69 / 169.05 ms |

报告位于 `phase0/profile_results/20260823_202154/`，原始日志归档于
`process/pp_rebalance/pp_30_32_31_controlled_recheck/`。该轮结果与前一轮长窗口复核的
`170.94 ms` 均值一致地落在历史 `31,31,31` 的 `161--174 ms` 水平，进一步确认：在
`output_len=256、VLLM_K3_TIMING=0` 的可比条件下，当前没有复现短测的 `136--138 ms`
端到端收益。

### 为什么此前短测看起来更快

目前能确认的差异不是 PP 分区是否生效，而是测量条件不同：

| 运行 | 分区 | output_len | K3 timing | TPOT |
|---|---|---:|---:|---:|
| timed short run | `30,32,31` | 64 | 1 | 136.56 ms |
| fair short run | `30,32,31` | 64 | 1 | 138.49 ms |
| long run | `30,32,31` | 512 | 未设置/0 | 169.33 ms |
| stable baseline | `30,32,31` | 256 | 0 | 170.94 ms |
| controlled recheck | `30,32,31` | 256 | 0 | 167.27 ms |

`VLLM_K3_TIMING=1` 会在 K3 model forward 和 CUDA Graph replay 路径插入 CUDA Event，且
在 settle 阶段执行 GPU synchronize；它不是只读的观测开关，可能改变 graph replay 的主机
调度和流水线同步行为。与此同时，64-token 请求的端到端 TPOT 只有很短的稳态窗口，不能
自动代表长 decode 的稳定节拍。因此当前最稳妥的结论是：`136--138 ms` 是特定短测/插桩
条件下的观察，不是已经被 `output_len=256/512、VLLM_K3_TIMING=0` 复现的真实收益。

为隔离插桩因素，随后固定 `output_len=256`、`warmups=2`、`repeats=5`、
`max_num_seqs=384` 和分区 `30,32,31`，仅将 timing 改为 `1` 重新运行。该轮时间戳为
`20260823_204405`，5 次 TPOT 为 `174.79、170.74、171.01、170.22、168.64 ms`，
均值 `171.08 ms`，仍与 timing=0 的两轮长窗口结果一致。由此可以排除“只要开启
`VLLM_K3_TIMING=1` 就能得到 136 ms”的解释；插桩确实会改变观测和同步路径，但不是
这次长窗口差异的充分原因。

因此目前最可靠的解释是：此前 64-token 单次短测的 `136--138 ms` 受短窗口、流水线
调度瞬态和运行间波动影响，不能代表长窗口稳态 TPOT。该轮报告及原始日志归档于
`process/pp_rebalance/pp_30_32_31_timing1_recheck/`。当前证据只支持“分区配置生效”，
不支持“端到端 TPOT 稳定下降约 20%”。

### `31,31,31` 严格对照

为完成同口径 A/B 对照，随后仅将分区改回 `31,31,31`，其余参数与 controlled recheck
完全一致：`output_len=256`、`warmups=2`、`repeats=5`、`max_num_seqs=384`、
`VLLM_K3_TIMING=0`、`context=512`，CUDA Graph 配置不变。实验 ID 为
`pp_31_31_31_controlled_baseline`，时间戳为 `20260823_210908`。

5 次 TPOT 为 `163.29、164.72、165.11、162.67、160.47 ms`，统计为：

| 分区/运行 | mean | P50 | P95 | min / max |
|---|---:|---:|---:|---:|
| `31,31,31` controlled baseline | **163.25 ms** | **163.29 ms** | **165.11 ms** | 160.47 / 165.11 ms |
| `30,32,31` controlled recheck | 167.27 ms | 167.01 ms | 169.05 ms | 165.69 / 169.05 ms |
| `30,32,31` previous long run | 170.94 ms | 171.36 ms | 172.96 ms | 168.43 / 172.96 ms |

按均值计算，`30,32,31` 相对本轮 `31,31,31` 对照分别回退约 `2.46%` 和 `4.71%`。
因此在当前单请求长窗口代理 workload 下，证据不仅未显示 PP rebalance 收益，反而支持
保留 `31,31,31`。对照报告及原始日志归档于
`process/pp_rebalance/pp_31_31_31_controlled_baseline/`。

### 只改变 output length 的 A/B 对照

为隔离 `output_len` 的影响，随后对同一分区 `30,32,31` 做了严格 A/B。两轮均固定：
`context=512`（实际 prompt=336 tokens）、`max_num_seqs=384`、`warmups=2`、
`repeats=5`、`VLLM_K3_TIMING=0`、`enforce_eager=False`（CUDA Graph
`FULL_AND_PIECEWISE`）、`max_model_len=32768`、`gpu_memory_utilization=0.9`、
`kv_cache_dtype=auto`。唯一改变的是 `PP_OUTPUT_LEN`。

| 运行 | output_len | 5 次 TPOT (ms) | mean | P50 | P95 | min / max |
|---|---:|---|---:|---:|---:|---:|
| `pp_30_32_31_output64_controlled` | 64 | 154.88, 162.34, 167.12, 168.19, 170.11 | **164.53 ms** | 167.12 ms | 170.11 ms | 154.88 / 170.11 ms |
| `pp_30_32_31_output256_controlled_ab` | 256 | 173.62, 166.26, 169.86, 169.91, 169.03 | **169.74 ms** | 169.86 ms | 173.62 ms | 166.26 / 173.62 ms |

256-token 相对 64-token 的均值高 `5.21 ms`，约 `3.17%`；两组范围明显重叠，当前样本
不能证明 output length 单独造成了稳定的性能回退。更重要的是，本次只改变 output length
且关闭 timing 的 64-token 均值为 `164.53 ms`，没有复现此前旧版单次、timing-on 短测的
`138.49 ms`。因此此前 `138--139 ms` 的结果不能归因于 `output_len=64` 本身，旧实验中
还同时存在 timing 插桩、单次测量和不同的预热/重复协议。

本次两轮的原始日志和结构化报告分别归档于：

```text
process/pp_rebalance/pp_30_32_31_output64_controlled/
process/pp_rebalance/pp_30_32_31_output256_controlled_ab/
```

### 复用旧短测 settings 的 timing-on 稳态复测

为进一步排除 `output_len` 和 `VLLM_K3_TIMING` 的组合因素，复用了旧的“有提升”配置：
`30,32,31`、`output_len=64`、`context=512`、`max_num_seqs=384`、
`VLLM_K3_TIMING=1`、CUDA Graph `FULL_AND_PIECEWISE`；同时采用当前稳态协议的
`warmups=2`、`repeats=5`。实验 ID 为 `pp_30_32_31_output64_timing1_steady`。

5 次 TPOT 为：

```text
158.65、150.61、149.99、176.71、148.67 ms
```

统计为：

| mean | P50 | P95 | min / max |
|---:|---:|---:|---:|
| **156.93 ms** | 150.61 ms | 176.71 ms | 148.67 / 176.71 ms |

该均值比旧的单次 `138.49 ms` 高 `18.44 ms`（约 13.3%），且重复运行范围达到
`27.72 ms`。因此即使完整复用 `output_len=64 + VLLM_K3_TIMING=1`，只把测量改为
两次预热、五次正式重复，也没有复现旧提升。旧结果应继续视为短测/单次运行的表观值，
不能作为 PP rebalance 的稳定收益证据。

本轮归档于：

```text
process/pp_rebalance/pp_30_32_31_output64_timing1_steady/
```

### 当前 workload 的 stage timing 诊断

在严格对照之后，对 `31,31,31` 额外执行了一轮 stage timing 诊断。除开启
`VLLM_K3_TIMING=1` 外，其余条件与严格对照相同：`context=512`、`output_len=256`、
`warmups=2`、`repeats=5`、`max_num_seqs=384`，并保持 CUDA Graph enabled。实验 ID 为
`pp_31_31_31_stage_timing`，时间戳为 `20260823_213703`。

逐层 `[k3t]` 事件只在 CUDA Graph capture 时重新进入 Python model forward。丢弃每个
TP rank 的第一个冷 capture 后，本轮 post-cold capture 结果为：

| stage | 层数 | model mean | P50 | P95 | 样本口径 |
|---|---:|---:|---:|---:|---|
| Stage 0 | 31 | 102.69 ms | 101.65 ms | 114.00 ms | 8 ranks x 1 capture |
| Stage 1 | 31 | 105.58 ms | 98.90 ms | 147.60 ms | 8 ranks x 1 capture |
| Stage 2 | 31 | 99.80 ms | 99.30 ms | 105.80 ms | 8 ranks x 1 capture |

这组数值只能用于方向性判断：本轮 capture 中 stage 1 的均值最高，但三者均值仅相差
约 5.8 ms，且 stage 1 的 rank 间方差明显较高；不能据此断言 stage 1 是稳定 decode
瓶颈。

同时解析 `_replay()` 前后的 `[k3t-gpu]` CUDA Event。分析时按累计序号移除前两个
warmup 请求（前 512 次 replay），并移除每个正式请求第一个混有 prefill 的 16-sample
bucket。剩余稳态 decode bucket 的结果为：

| stage | local bucket mean / P50 / P95 | TP critical mean / P50 / P95 | 完整 bucket |
|---|---:|---:|---:|
| Stage 0 | 6.20 / 6.20 / 6.20 ms | 6.20 / 6.20 / 6.20 ms | 74 |
| Stage 1 | 6.20 / 6.20 / 6.20 ms | 6.20 / 6.20 / 6.20 ms | 73 |
| Stage 2 | 6.20 / 6.20 / 6.20 ms | 6.20 / 6.20 / 6.20 ms | 74 |

这里的 `local` 是各 rank 的 16-replay bucket 均值；`TP critical` 是同一 bucket 内 8 个
TP rank 的最大值。日志只保留一位小数，因此三组结果在当前精度下无法区分。stage 1
少一个完整 bucket，是 rank 13 少打印了一条 lazy-settle 聚合记录；最后一个请求也会有
未达到 settle 阈值的尾部 event 留在进程内，所以上表不是 5 x 255 个 decode replay 的
全量记录。

`6.20 ms` 仅是 vLLM 专用 CUDA stream 上 graph `_replay()` 的 GPU 区间，不包含 engine
step 之间的主机调度、PP 等待和结果处理；`[k3t-cg] dt` 虽包含这些时间，却是相邻 dispatch
的 wall period，也不能视为某个 stage 的纯计算时间。本轮 timing-on 的端到端 TPOT 为
`172.43、172.00、174.91、177.17、169.54 ms`，均值 `173.21 ms`。它与约 6.20 ms 的
replay 区间有很大差距，说明当前单请求 workload 的 TPOT 主要不由可见的单次 stage GPU
replay 决定；这也解释了单纯移动一层没有在严格长窗口对照中带来收益。端到端基线仍应
采用上一节 timing-off 的 `163.25 ms`，不能用本轮插桩结果替换。

本轮原始数据与解析结果归档于：

```text
process/pp_rebalance/pp_31_31_31_stage_timing/
```

稳态 replay 解析器为 `tools/analyze_k3_replay_events.py`。

## 失败实验记录

第一次运行 `29,33,31` 时使用了 `max_num_seqs=512`。stage 2 仅报告 469 个可用
Mamba cache blocks，CUDA Graph 初始化因此失败。node 0/1 在 node 2 退出后仍等待失联
rank，残留 launcher 随后被停止。

失败日志保存在：

```text
process/pp_rebalance/pp_29_33_31_timed/failed_stage0.log
process/pp_rebalance/pp_29_33_31_timed/failed_stage1.log
process/pp_rebalance/pp_29_33_31_timed/failed_stage2.log
```

后续候选统一使用 `PP_MAX_NUM_SEQS=384`，以保证 Mamba cache 能通过 CUDA Graph 初始化，
并保持比较条件一致。

## 最终建议

根据相同参数下的长窗口 A/B 对照，当前推荐恢复为：

```bash
export VLLM_PP_LAYER_PARTITION=31,31,31
```

`30,32,31` 在 64-token 短测中曾得到更低 TPOT，并改变了 stage capture 的负载分布，
但该收益没有在 256-token 重复测试中复现；严格对照中 `31,31,31` 的 mean/P50/P95 均
更低。若继续评估 PP rebalance，应转向目标生产并发的 continuous batching 吞吐测试，
并始终让候选保持相同的 `max_num_seqs`、输入长度、输出长度、预热、重复次数、timing
开关和 CUDA Graph 配置。

详细原始日志、失败日志和 JSON 结果位于：

```text
process/pp_rebalance/
```
