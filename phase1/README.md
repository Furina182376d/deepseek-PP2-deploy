# Phase 1 — PP=3 TP=8 Kimi-K3 三节点部署

## 阶段目标

在**三个节点**上以 **PP=3（Pipeline Parallelism）+ TP=8** 部署 Kimi-K3
(`/data/models/Kimi-K3`, MXFP4 量化 1.5TB, 93 层 ÷ 3 = 每 stage 31 层)，
每节点使用全部 8 张 H20 GPU：

| 节点 | 内网 IP | PP stage | 角色 |
|------|---------|----------|------|
| Node 0 (aliyun1) | 192.168.0.224 | stage 0 (layers 0-30) | master, 收集结果 |
| Node 1 (aliyun2) | 192.168.0.225 | stage 1 (layers 31-61) | worker |
| Node 2 (aliyun3) | 192.168.0.226 | stage 2 (layers 62-92) | worker |

通过 `torchrun` + `external_launcher` 后端在三节点各启动 8 个进程（每 GPU 一个），
total world_size=24，所有进程通过 NCCL over TCP 协调，仅 rank 0 收集和记录结果。

## 使用方式

在**三台服务器上分别执行**（先 worker 后 master 或同时均可，torchrun 会等待）：

```bash
cd /home/tjy/codebases/deepseek-PP2-deploy/phase1

# aliyun1 (192.168.0.224):  ./launch_pp.sh 0
# aliyun2 (192.168.0.225):  ./launch_pp.sh 1
# aliyun3 (192.168.0.226):  ./launch_pp.sh 2
```

运行流程：三机同时加载权重（页缓存热时 ~10 min，冷盘可能 2h）→ 全部 rank 就绪后，
leader 使用真实 LongBench `qmsum/gov_report` 长文本请求进行 warmup 和重复测速 →
结果写入 `phase0/profile_results/`（CSV + `full_results.json` + report，仅 leader 本机）。

## EP/CP 顺序测速

如果要在保持 `PP=3、TP=8` 的前提下依次测试 EP/CP 组合，可只在主节点
（默认 `192.168.0.224`）运行：

```bash
./run_ep_cp_sweep.sh
```

脚本会通过 SSH 同时启动三台机器的 `launch_pp.sh`，等待一组完整结束后再运行下一组。
测速请求来自本地 LongBench 的真实文本，不再使用合成 prompt：默认任务为 `qmsum` 和
`gov_report`，每个任务选择一个在 32K 上限内最长的样本，生成 128 token，每个样本重复 3 次。
当前安装的 vLLM 对 Kimi-K3 的 MultiHeadLatentAttention 明确不支持 context
parallelism，因此 `DCP/PCP > 1` 会在模型初始化时失败；脚本默认只运行下面两组有效配置：

```text
TP=8 EP=0 DCP=1
TP=8 EP=1 DCP=1
```

Prefill CP 和 decode CP 均固定为 1。每组的三节点日志保存在
`sweep_results/<timestamp>/<experiment>/`；任一节点失败时脚本会停止后续测试。
如主机地址或代码路径不同，可设置 `SWEEP_HOSTS`、`SWEEP_MASTER_HOST`、`SWEEP_REPO`
和 `SWEEP_LOG_DIR` 覆盖默认值。

## 文件说明

| 文件 | 作用 |
|------|------|
| `config_pp.py` | K3 专用常量：`MODEL_PATH=/data/models/Kimi-K3`、`PP_SIZE=3`、`TP_SIZE_PER_PP=8`、`NNODES=3`、`MASTER_ADDR=192.168.0.224`、`MAX_MODEL_LEN=32768`；默认从 LongBench 加载 `qmsum/gov_report` 真实长文本；自动从 torchrun 环境变量检测 rank/is_leader；`validate_config()` 校验一致性 |
| `run_pp.py` | PP 推理核心 — 每个 torchrun 进程创建 `LLM(pipeline_parallel_size=3, tensor_parallel_size=8, distributed_executor_backend="external_launcher", distributed_timeout_seconds=10800, cpu_distributed_timeout_seconds=10800, ...)`；指标采集 (TTFT/prefill_tps/decode_tps/tpot) 与 phase2/phase0 一致 |
| `profile_dsv4_pp.py` | 入口脚本：vLLM 导入前安装 `comm_crypto` hooks（无 `VLLM_COMM_PSK` 时 no-op），调用 `run_pp()`；leader 写 `full_results.json` 与 report；失败时降级 `max_model_len=16384` 重试 |
| `launch_pp.sh` | torchrun 启动包装：`./launch_pp.sh <0|1|2>`；NCCL TCP 环境变量（`NCCL_IB_DISABLE=1`、`NCCL_SOCKET_IFNAME=eth0`、`NCCL_CUMEM_HOST_ENABLE=0`）、`HF_HUB_OFFLINE=1`、`VLLM_ENABLE_V1_MULTIPROCESSING=0`；激活 `vllm` conda 环境 |

## 前置条件

- 三节点间免密 SSH（aliyun1 自身公钥也已加入本机 authorized_keys）
- 三节点均有 `vllm` conda 环境（vLLM 0.27.1，官方支持 Kimi-K3）
- `/data/models/Kimi-K3` 三机一致
- 三节点代码库路径一致

## ⚠️ 必须先打的 vllm 补丁

vLLM 0.27.1 有两个会导致三机部署直接崩溃/输出垃圾的已知问题，**必须在三台机器上都打好补丁**
（补丁脚本在 `/home/tjy/codebases/Kimi_deploy/`）：

```bash
# 1) FA3 "cp_world_size must be positive" 崩溃 (vllm PR #50625/#50404):
#    MLACommonImpl 把 dcp_world_size 覆盖为 -1 哨兵, Kimi-K3 直调 forward_mqa
#    绕过惰性修复 -> CUDA graph 捕获时 FA3 收到 -1 崩溃。不修则无法启动。
# 2) gloo 组 1800s 超时: 三机加载不均衡 >30min 时先加载完的节点超时崩溃。
#    修复已在 run_pp.py 里 (cpu_distributed_timeout_seconds=10800), 无需额外操作。
source ~/miniconda3/etc/profile.d/conda.sh && conda activate vllm
for ip in 192.168.0.224 192.168.0.225 192.168.0.226; do
  scp /home/tjy/codebases/Kimi_deploy/patch_vllm.py tjy@$ip:/home/tjy/kimi_bench/bin/
  ssh tjy@$ip 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate vllm \
    && python3 /home/tjy/kimi_bench/bin/patch_vllm.py'
done
```

## 与 Phase 2 的集成

`run_pp()` 加载完的 `LLM` 实例可直接交给 phase2 的 benchmark runner：

```python
from phase1.run_pp import run_pp          # 或自行创建 LLM
from phase2.run_bench import run_needle_benchmark

llm = LLM(pipeline_parallel_size=3, tensor_parallel_size=8, ...)
run_needle_benchmark(llm, tag="K3_PP3_TP8_needle")
```

## 已知问题 (2026-08-20)

- **输出质量**: 模型输出可能退化为重复的 "@"（logprobs 含 NaN）。已确认与
  CUDA graphs 无关（eager 模式同样出现），怀疑是 mxfp4 MoE kernel 路径
  （MARLIN 后端, H20/sm_90）的问题，类似 vllm issue #47303。诊断工具：
  `/home/tjy/codebases/Kimi_deploy/nan_hook.py`（`apply`/`remove`，
  在每层 forward 后检查 NaN 并打印首个 NaN 层号，三机各跑一次）。
  吞吐/通信/计算的**计时数据不受影响**（kernel 仍在执行）。
- **32k 上下文**: `MAX_MODEL_LEN=32768` 时 32k prompt + 输出放不下,
  `CONTEXT_LENGTHS` 止步 16384；测 32768 需把 `MAX_MODEL_LEN` 提到 65536
  （注意 KV cache 显存余量，每卡权重已占 62.5GB）。
