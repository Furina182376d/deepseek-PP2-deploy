# Phase 2 — 真实 Benchmark 数据集

## 阶段目标

用**真实长上下文数据集**替代 Phase 0/1 中的合成重复段落 prompt，
测量模型在接近生产环境的 workload 下的吞吐表现。
同时支持本地 JSON/JSONL 文件（LongBench）和内置的
needle-in-a-haystack（无需下载）。

## 文件说明

| 文件 | 作用 |
|------|------|
| `benchmark_config.py` | Benchmark 配置：定义 LongBench 任务列表（narrative_qa, qasper, gov_report, qmsum 等）、本地数据搜索路径、output 长度扫描范围、采样上限；内置 needle-in-a-haystack 的 context 长度、depth、passkey 和 filler 段落 |
| `data_loader.py` | 数据加载器 — 提供两个入口：① `load_longbench(task_name)` 从本地 JSON/JSONL 加载 LongBench 任务；② `iter_needle_prompts()` 生成 needle-in-a-haystack prompt（将 passkey 按指定 depth 埋入 filler 段落中，末尾追加检索指令）；`list_available_datasets()` 检测哪些数据集本地可用 |
| `run_bench.py` | Benchmark 运行器 — `run_needle_benchmark(llm, tag=...)` 按 (ctx_len, output_len, depth) 三维度扫描吞吐，输出 CSV 行；`run_longbench_benchmark(llm, tag=...)` 遍历本地 LongBench 任务并逐样本测吞吐；内部 `_timed_generate()` 与 run_tp/run_pp 共享相同的 TTFT/prefill/decode 指标采集逻辑 |

## 数据集获取

由于本环境 HuggingFace 不可达，LongBench 数据需通过 **ModelScope** 下载：

```bash
# LongBench v1（114 MB，21 个子任务 JSONL）
wget https://www.modelscope.cn/api/v1/datasets/ZhipuAI/LongBench/repo?Revision=master&FilePath=data.zip
unzip data.zip -d /data/benchmarks/longbench/

# LongBench v2（465 MB）
wget https://www.modelscope.cn/api/v1/datasets/ZhipuAI/LongBench-v2/repo?Revision=master&FilePath=data.zip
unzip data.zip -d /data/benchmarks/longbench_v2/
```

## 与 Phase 0/1 的集成

`run_bench.py` 接受一个已加载的 `LLM` 实例——可以来自 TP（Phase 0）、PP（Phase 1）
或任意配置。结果统一走 `results_utils`（CSV + JSON + report）。

```python
# 示例：在 PP=2 配置下运行 needle benchmark
from run_pp import run_pp       # Phase 1
from run_bench import run_needle_benchmark

llm = LLM(pipeline_parallel_size=2, tensor_parallel_size=4, ...)
run_needle_benchmark(llm, tag="PP2_TP4_needle")
```
