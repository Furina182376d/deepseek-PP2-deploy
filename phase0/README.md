# Phase 0 — 单节点 TP 基线（重构后保留）

## 阶段目标

在单节点上用 **仅 TP（Tensor Parallelism）** 的方式部署 DeepSeek-V4-Flash，
测得 TP=4 与 TP=8 下的基础吞吐（prefill_tok/s, decode_tok/s），
作为后续 PP=2 多节点部署的**性能基线**。

## 文件说明

| 文件 | 作用 |
|------|------|
| `config.py` | 共享常量：模型路径、context 长度列表、output 长度、max_model_len、GPU 显存利用率、KV cache dtype |
| `prompt_utils.py` | `make_prompt(ctx_len)` — 用重复英文段落生成指定 token 数的合成 prompt（用于基准测试） |
| `gpu_utils.py` | `gpu_mem(tp)` — 返回每个 GPU 的 used/total 显存（MB） |
| `results_utils.py` | CSV/JSON/report 全链路：创建结果目录、打开 CSV、写入行、关闭 CSV、生成终端摘要和 report.txt；维护全局 `ALL_RESULTS` 列表 |
| `run_tp.py` | `run_tp(tp)` — 核心 TP 推理函数：加载模型、warmup、按 context 长度遍历、计时（TTFT/prefill/decode/TPOT）、记录显存 |
| `profile_dsv4.py` | 入口脚本：依次运行 TP=4 和 TP=8，失败时自动降级 max_model_len 重试；最后输出 `full_results.json` 和 report.txt |

## 使用方式

```bash
cd phase0
conda activate ds
python profile_dsv4.py
```

结果输出到 `phase0/profile_results/<timestamp>/`，每个 TP 配置一个 CSV，
外加 `full_results.json` 和 `report.txt`。
