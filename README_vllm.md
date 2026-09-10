# Unified vLLM Benchmark

本项目使用统一脚本运行多节点 vLLM benchmark。每次测试只需修改对应模式
脚本顶部的配置区，然后在两台机器上运行统一启动脚本
[`launch_vllm_benchmark.sh`](launch_vllm_benchmark.sh)：

- 普通模式：修改 [`vllm_benchmark.py`](vllm_benchmark.py)。
- 投机解码模式：修改 [`vllm_benchmark_speculative.py`](vllm_benchmark_speculative.py)。

启动脚本的第二个参数选择执行哪一个脚本（`0`/`standard` 或
`1`/`speculative`），详见下文「启动」。

## 快速配置

两份脚本结构完全相同，各自携带独立的配置（包括 PP/TP 拓扑和
`SPECULATIVE_CONFIG`）。以普通模式的 `vllm_benchmark.py` 为例，配置集中在
文件顶部的 `CONFIGURATION` 配置区（约 28 行起）。常用配置包括：

- `PP_SIZE`：pipeline parallel 的 stage 数量。当前实现按一个节点承载一个 stage，因此节点数由它决定。
- `TP_SIZE_PER_STAGE`：每个 pipeline stage 使用的 tensor parallel GPU 数量。
- `MODEL_PATH`：模型地址，可以是 `/data/models` 下的本地模型目录或其他可访问的模型路径。
- `BENCHMARK_TYPE`：`"longbench"`、`"classic"` 或 `"custom"`。
- `BENCHMARK_DATA_DIR`：文件型 benchmark 的数据目录。
- `BENCHMARK_TASKS`：要运行的任务或文件名元组；设为空元组时使用数据目录中的标准 `.json`/`.jsonl` 文件。
- `MAX_SAMPLES_PER_TASK`：每个任务最多读取的样本数，设为 `0` 表示读取全部样本。
- `OUTPUT_TOKENS`：每个请求的最大输出 token 数。
- `NUM_WARMUPS`：正式计时前的 warmup 次数。
- `NUM_REPEATS`：每个样本的正式重复次数。
- `REQUEST_CONCURRENCY`：每批同时提交的请求数。设为 `1` 测单流，设为 `4` 测四流并行；实际批大小不会超过已加载请求数。
- `ENABLE_EXPERT_PARALLEL`：是否开启 expert parallel（EP）。
- `KV_CACHE_DTYPE`：KV cache 数据类型，例如 `"fp8"`。
- `BLOCK_SIZE`、`MAX_MODEL_LEN`、`MAX_NUM_SEQS`、`GPU_MEMORY_UTILIZATION`：vLLM 资源和长度配置。
- `PREFILL_CONTEXT_PARALLEL_SIZE`、`DECODE_CONTEXT_PARALLEL_SIZE`：context parallel 配置。
- `TOKENIZER_MODE`、`REASONING_PARSER`、`TRUST_REMOTE_CODE`、`COMPILATION_CONFIG`：模型和推理运行选项。
- `SPECULATIVE_CONFIG`：投机解码配置，为 `None`（默认）时关闭投机解码；投机脚本在该项设置非 `None` 值后即启用，并会作为 vLLM 的 `speculative_config` 传入。

当前默认部署对应两台机器：

```python
PP_SIZE = 2
TP_SIZE_PER_STAGE = 8
MODEL_PATH = "/data/models/DeepSeek-V4-Pro-DSpark"
MASTER_ADDR = "192.168.0.224"
```

`NNODES` 默认自动设置为 `PP_SIZE`。如果调整 `PP_SIZE`，需要保证每个 pipeline stage
都有一台对应的节点，并在所有节点使用相同的代码、模型路径、数据路径和 Python/vLLM 环境。

## LongBench

将 `BENCHMARK_TYPE` 设为 `"longbench"`，并把 `BENCHMARK_DATA_DIR` 指向 LongBench
数据目录。例如：

```python
BENCHMARK_TYPE = "longbench"
BENCHMARK_DATA_DIR = "/path/to/longbench/data"
BENCHMARK_TASKS = ("qmsum", "gov_report")
```

任务名可以写成不带后缀的任务名（例如 `qmsum`），脚本会尝试读取对应的 `.jsonl` 文件；
也可以直接填写 `dataset.json` 或 `dataset.jsonl`。LongBench 请求使用记录中的
`context` 和 `input` 字段拼接为 prompt。

## Classic Benchmark

将 `BENCHMARK_TYPE` 设为 `"classic"`。`BENCHMARK_TASKS` 中填写数据文件名：

```python
BENCHMARK_TYPE = "classic"
BENCHMARK_DATA_DIR = "/path/to/classic/data"
BENCHMARK_TASKS = ("dataset.jsonl",)
```

每条 JSON/JSONL 记录支持以下 prompt 格式，按顺序尝试：

- `prompt`
- `text`
- `context` + `input`
- `context` + `question`

其中组合格式会将上下文和问题用空行连接。对于 JSON 数组文件和 JSONL 文件均可使用。

## 自定义 Prompt

将 `BENCHMARK_TYPE` 设为 `"custom"`，并在配置区修改样本数量和
`build_custom_prompt(index)`：

```python
BENCHMARK_TYPE = "custom"
CUSTOM_PROMPT_COUNT = 10


def build_custom_prompt(index):
    return f"你的自定义 prompt，编号是 {index}"
```

`index` 从 `0` 开始递增到 `CUSTOM_PROMPT_COUNT - 1`。自定义模式不读取
`BENCHMARK_DATA_DIR` 或 `BENCHMARK_TASKS`。

## 启动

先确保两台机器可以通过内网互通，并且 `MASTER_ADDR` 与网卡名
`NETWORK_INTERFACE` 配置正确。然后在两台机器分别执行：

```bash
# 192.168.0.224
./launch_vllm_benchmark.sh 0

# 192.168.0.225
./launch_vllm_benchmark.sh 1
```

启动脚本的第二个参数控制运行模式，可以用 `0`/`1`，也可以用
`standard`/`speculative`：

```bash
# 普通模式（无投机解码）
./launch_vllm_benchmark.sh 0 standard

# 投机解码模式
./launch_vllm_benchmark.sh 1 speculative
```

第二个参数省略时默认为 `0`（`standard`），与上面的两行命令等价。普通模式
执行 `vllm_benchmark.py`，投机模式执行 `vllm_benchmark_speculative.py`；
两台机器必须传入相同的模式参数，并使用相同的代码、模型路径和数据路径。

两种模式各执一份配置，拓扑可以不同：启动脚本会先激活名为 `vllm` 的 conda
环境，再自动从**被选中的脚本**读取 PP/TP、node 数、master 地址和端口，
并设置 `GLOO_SOCKET_IFNAME`、`NCCL_SOCKET_IFNAME` 等分布式环境变量。因此
投机模式若要改用 PP=1、TP=16 之类的拓扑，只需修改投机脚本自己的配置区，
无需改动启动脚本。

投机脚本与普通脚本同构。在投机脚本的配置区把 `SPECULATIVE_CONFIG` 设为
非 `None`（例如配置好投机模型与算法）即启用投机解码，同时需要确认两台
节点上的 vLLM 版本支持所选的投机模型与算法。

默认 Miniconda 目录是 `/home/tjy/miniconda3`；如果节点安装位置不同，可以
覆盖：

```bash
CONDA_BASE=/path/to/miniconda3 ./launch_vllm_benchmark.sh 0
```

环境名也可以通过 `CONDA_ENV_NAME` 覆盖，但两台节点必须使用包含相同 vLLM
依赖的环境。只有需要特殊解释器时才设置 `PYTHON_BIN`，并确保它位于已激活的
conda 环境内。

## DSpark 投机模式不支持 PP>1

vLLM 的 DSpark 投机解码（DeepSeek-V4 checkpoint 自带 draft head）目前只支持
PP=1。这是 vLLM 0.27.1（conda 环境 `vllm`）源码中的硬性限制，与 SGLang 一侧的
DSpark 要求 `pp_size == 1` 一致，不是配置写法问题。三条互相独立的证据：

**1. DSpark 加载器硬性禁止 PP**

`/home/tjy/miniconda3/envs/vllm/lib/python3.12/site-packages/vllm/v1/worker/gpu/spec_decode/dspark/utils.py:50`，
`load_dspark_model()` 内：

```python
if get_pp_group().world_size != 1:
    raise NotImplementedError("DSpark does not support pipeline parallelism.")
```

PP=2 时每个 rank 装载 draft 模型都会在此抛 `NotImplementedError`，没有可绕
过的参数。

**2. DSpark 强制 V2 model runner，而 V2 不支持 external_launcher + PP>1**

- `/home/tjy/miniconda3/envs/vllm/lib/python3.12/site-packages/vllm/config/vllm.py`：
  `method == "dspark"` 使 `use_v2_model_runner` 返回 `True`（DeepSeek-V4 本身
  不是默认 V2 架构，是为 dspark 强制 V2；若 V2 不支持其余配置会直接 raise，
  不会回退到 V1）。
- 同上（V2 unsupported 列表）：V2 没有实现 V1 用于 torchrun
  （`external_launcher`）的 PP 输出广播 `broadcast_pp_output`，因此
  `external_launcher + pipeline_parallel_size > 1` 不受支持，而本套件正是以
  torchrun 两机、每机一个 PP stage 的方式启动。
- `_validate_v2_model_runner()` 在 unsupported 非空时 raise `ValueError`，
  因此即使绕过第 1 条，也会被这道检查拦下。

**3. 架构机制：DSpark draft 必须与目标层同进程**

`load_dspark_model()` 会把 draft 模型的 `embed_tokens` / `lm_head` 在进程内
直接替换为目标模型的张量（weight sharing），draft 头取目标模型指定层
（`/data/models/DeepSeek-V4-Pro-DSpark/config.json` 中的
`dspark_target_layer_ids=[58,59,60]`）的 hidden state 作为输入。PP 切分后这
些层分散在不同进程，draft 头无法工作。

因此投机模式使用 PP=1：配合 TP=16 跨两机（每机 8 卡）仍可占用全部 16 张卡，
总规模与普通模式的 PP=2 × TP=8 相当，便于对比。投机脚本内 `NNODES` 不再等
于 `PP_SIZE`（校验逻辑需相应放宽）。

## 输出

rank 0 会把结果写入 `results/<UTC timestamp>/`，包含：

- `summary.json`：运行配置和完整结果。
- `report.json`：与 summary 同步更新的实验报告，包含当前已完成 batch 的聚合指标，尤其是 `aggregate_decode_tps`。
- `requests.csv`：每个任务、样本和重复运行的延迟、吞吐及显存指标。

每个正式 batch 完成后都会立即刷新这三个文件，因此中途终止时仍可查看已完成部分。
`decode_tps` 按 `(生成 token 数 - 1) / decode 时间` 计算；`aggregate_decode_tps` 是并发 batch 的
`total_decode_tokens / batch_decode_time`，单位均为 token/s；每个请求还会记录 `batch_decode_tps`。
`request_sum_decode_tps` 按请求 decode 时间相加，是串行口径，不代表并发总吞吐。离线 engine 不提供有效时间戳时，
报告中的 `timing_source` 为 `step_wall_clock`，TTFT、TPOT 和 decode 吞吐均来自逐 step 的 wall-clock
观测，而不是错误地记为 0。

真实模型启动前，可以先运行以下检查：

```bash
python -m py_compile vllm_benchmark.py vllm_benchmark_speculative.py
bash -n launch_vllm_benchmark.sh
```

## 问题解决过程

真实集群上遇到的启动问题与解决方案，按时间顺序记录。两机同时卡住时，先用
以下手段区分「死锁」与「仍在推进」：

- `ps -eo pid,stat,etime,%cpu,cmd | grep vllm_benchmark`：进程为 `S` 且 CPU
  接近 0 更像等待；为 `R` 且 CPU 高说明在计算。
- `nvidia-smi`：GPU 利用率 0% + 显存不增长 = 卡在初始化；GPU 100% 或显存
  增长 = 在推进。
- 死锁特征：主线程 `wchan` 为空、`timeout 6 strace -p <pid> -c` 六秒零系统
  调用（纯用户态自旋）、模型尚未开始加载（显存为空）。

### 问题 1：跨机 TP 下 custom all-reduce 的 symm_mem fd 传递导致两机死锁（已解决）

**现象**：PP=1/TP=16 跨两机启动时，16 个 rank 的 NCCL 组建立完成、跨机 TCP
全部 ESTABLISHED 后进程无限卡住：主线程纯用户态自旋、GPU 显存为空（模型
从未开始加载）。带 `VLLM_LOGGING_LEVEL=DEBUG` 重跑后，192.168.0.225 打印：

```text
DEBUG ... [custom_all_reduce.py:312] MNNVL AG/RS initialization failed: Failed to send fd: No such file or directory
WARNING ... [custom_all_reduce.py:254] Custom collectives are disabled because this multi-node group does not support MNNVL multicast.
```

**原因**：vLLM 0.27.1（`/home/tjy/miniconda3/envs/vllm/lib/python3.12/site-packages/vllm`）
初始化 custom all-reduce 时，在
`distributed/device_communicators/custom_all_reduce.py` 的 `_init_mnnvl_buffer`
里通过 `torch_symm_mem.rendezvous` 用 UNIX socket **fd 传递**分发跨机共享
显存 buffer；UNIX socket fd 只能在同一台机器内传递，跨机必然失败。失败处理
不对称：发送方打印 warning 后放弃并继续（`disabled=True`），接收方卡在
rendezvous 中永远等不到 fd → 单边状态不一致，两机互相等待。普通 PP=2/TP=8
模式不受影响（TP 组在单机内、无跨机 fd 传递），这是跨机 TP 首次触发该路径。
`disable_custom_all_reduce` 不会因跨机自动打开（`config/parallel.py` 只在平台
不支持或 `VLLM_BATCH_INVARIANT` 时置 True）。

**解决**：对 vLLM 显式传 `disable_custom_all_reduce=True`（`vllm_benchmark_speculative.py`
的 `main()` kwargs），完全跳过 custom-AR 的 MNNVL/symm_mem 初始化，回退标准
NCCL all-reduce。修复后权重正常加载（66/66 shards、每 rank 58.29 GiB），
DSpark draft 模型正常装载（日志 `DSpark draft model loaded: 96 params`）。

/tmp/dummy_tp16_224.log