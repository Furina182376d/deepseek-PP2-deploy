# Unified vLLM Benchmark

本项目使用一个总脚本运行多节点 vLLM benchmark。每次测试只需要修改
[`run_benchmark.py`](run_benchmark.py) 顶部的配置区，然后在两台机器上运行统一启动脚本。

## 快速配置

配置集中在 `run_benchmark.py:28` 附近。常用配置包括：

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
- `TOKENIZER_MODE`、`REASONING_PARSER`、`TRUST_REMOTE_CODE`、`COMPILATION_CONFIG`、`SPECULATIVE_CONFIG`：模型和推理运行选项。

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
./launch_benchmark.sh 0

# 192.168.0.225
./launch_benchmark.sh 1
```

启动脚本会先在每个节点激活名为 `vllm` 的 conda 环境，再从
`run_benchmark.py` 自动读取 PP、TP、master 地址和端口，并设置
`GLOO_SOCKET_IFNAME`、`NCCL_SOCKET_IFNAME` 等分布式环境变量。默认 Miniconda
目录是 `/home/tjy/miniconda3`；如果节点安装位置不同，可以覆盖：

```bash
CONDA_BASE=/path/to/miniconda3 ./launch_benchmark.sh 0
```

环境名也可以通过 `CONDA_ENV_NAME` 覆盖，但两台节点必须使用包含相同 vLLM
依赖的环境。只有需要特殊解释器时才设置 `PYTHON_BIN`，并确保它位于已激活的
conda 环境内。

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
python -m py_compile run_benchmark.py
bash -n launch_benchmark.sh
```
