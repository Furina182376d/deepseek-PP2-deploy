# Phase 1 — PP=2 两节点部署

## 阶段目标

在**两个节点**上以 **PP=2（Pipeline Parallelism）+ TP=8** 部署 DeepSeek-V4-Flash，
使用全部 16 张 H20 GPU（每节点 8 卡）：

- **PP Stage 0（Node 0, layers 0-21）**：192.168.0.63，8 GPU（TP=8）
- **PP Stage 1（Node 1, layers 22-42）**：192.168.0.65，8 GPU（TP=8）

通过 `torchrun` + `external_launcher` 后端在两节点各启动 8 个进程（每 GPU 一个），
total world_size=16，所有进程通过 NCCL over TCP 协调，仅 rank 0 收集和记录结果。

> **设计注**：当前实现使用 `external_launcher` 后端。Plan agent 在 vLLM 源码级
> 分析后建议切换为 `mp` 后端 + Leader/Follower 架构（更符合 PP 语义），
> 该重构待后续迭代完成。

## 文件说明

| 文件 | 作用 |
|------|------|
| `config_pp.py` | PP 专用常量：`PP_SIZE=2`、`TP_SIZE_PER_PP=8`（每节点 8 卡全用）、`NNODES=2`、`WORLD_SIZE_PP=16`、`MASTER_ADDR`/`MASTER_PORT`；自动从 torchrun 环境变量检测 `GLOBAL_RANK`、`LOCAL_RANK`、`NODE_RANK`、`IS_LEADER`；re-export 基础 config 的模型/缓存常量 |
| `run_pp.py` | PP 推理核心 — `run_pp()`：每个 torchrun 进程创建一个 `LLM` 实例（`pipeline_parallel_size=2, tensor_parallel_size=8, distributed_executor_backend="external_launcher"`），warmup 后按 context 长度遍历；仅 `IS_LEADER`（global rank 0）负责计时、打印、CSV/JSON 写入；指标采集逻辑与 `run_tp.py` 完全一致 |
| `profile_dsv4_pp.py` | PP 入口脚本：在 vLLM 导入之前安装 `comm_crypto` 加密 hooks（若 `VLLM_COMM_PSK` 设了则生效），然后调用 `run_pp()`；leader 在完成后输出 `full_results.json` 和 report；失败时自动降级 `max_model_len=32768` 重试 |
| `launch_pp.sh` | torchrun 启动包装脚本：在两节点上分别运行（传入不同的 `--node_rank`）；设置 NCCL 环境变量（`NCCL_IB_DISABLE=1`、`NCCL_SOCKET_IFNAME=eth0`、`GLOO_SOCKET_IFNAME=eth0` 等）和 `VLLM_ENABLE_V1_MULTIPROCESSING=0`；激活 `ds` conda 环境后执行 torchrun |

## 使用方式

```bash
cd phase1
# Node 0 (192.168.0.63):  ./launch_pp.sh 0
# Node 1 (192.168.0.65):  ./launch_pp.sh 1
```

## 前置条件

- 两节点间免密 SSH
- 两节点均有 `ds` conda 环境（vLLM 0.26.0）
- 模型路径 `/data/model/DeepSeek-V4-Flash-0731/` 在两节点一致（共享存储或副本）
- 两节点代码库路径一致
