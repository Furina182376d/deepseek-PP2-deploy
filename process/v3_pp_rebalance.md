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

`30,32,31` 相对历史 `161--174 ms` 的 TPOT 改善约为 14%--21%。继续将 stage 0
的一层迁移到 stage 1，得到 `29,33,31` 后，TPOT 回退到 156.81 ms，说明 stage 1
成为新的瓶颈。

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

当前推荐：

```bash
export VLLM_PP_LAYER_PARTITION=30,32,31
```

它在已测试候选中具有最好的重复 TPOT 均值和最小的运行间波动。投入生产前，应使用目标
并发和更长的 decode window 重复测试，并统计多请求 TPOT P50/P95；所有候选必须保持
相同的 `max_num_seqs`、输入长度、输出长度和 CUDA Graph 配置。

详细原始日志、失败日志和 JSON 结果位于：

```text
process/pp_rebalance/
```
