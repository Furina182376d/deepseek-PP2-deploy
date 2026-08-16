# EP 与调度实验（2026-08-16）

Batch 扫描（六轮曲线，见 `phase2/compare_batch.py`）结论的后续实验：攻 4.5ms/seq 的边际成本与 FIFO 排队问题。

模型关键事实（config.json）：256 routed experts 的 MoE；GQA=1（每 token KV ≈ 5.5KB fp8，attention 不是瓶颈）；权重 FP8 e4m3（128×128 分块）+ deep_gemm。→ 边际成本的主角是 MoE 专家层 GEMM 与通信。

## 实验设计

| 实验 | 配置 | 对比基准 |
|---|---|---|
| Exp1a | TP=2+EP=2, batch=1 | TP4/EP1 b0（`phase0/profile_results/20260814_233513`，TPOT 10.05ms） |
| Exp1b | TP=2+EP=2, batch=4 | TP4/EP1 b4（`phase0/profile_results/20260815_232828`，TPOT 17.6ms / 128.7 tok/s / TTFT p50 2.3s） |
| Exp2 | TP4/EP1, batch=-1, 按 ctx 排序提交（SJF） | FIFO 的 -1（`phase0/profile_results/20260815_000022`，TTFT p50 56.6s / 196.2 tok/s） |

## 假设

- **Exp1（EP）**：TP=4 下每层专家 GEMM 跨 4 卡切分 + all-reduce；EP=2 让专家权重按卡分片、token 定向 dispatch → batch=1 TPOT 与 4.5ms/seq 边际成本应同时下降，吞吐上升。
- **Exp2（length-aware）**：FIFO 提交时 61K 长样本堵死短请求（队列时间总和占 wall 3475%）；按长度升序提交后短请求先跑 → TTFT 分位数大幅下降（期望个位数秒），吞吐基本不变（~196 tok/s）。

## 判定标准

- Exp1 有效：TPOT(b=1) < 10.05ms 或 TPOT(b=4) < 17.6ms，且 out_t/s 上升
- Exp2 有效：TTFT p50 << 56.6s（期望 <10s），out_t/s ≈ 196

## 结果

（实验完成后填写：结果目录、compare 汇总表、结论）

## 文件

- 实验脚本：`phase2/run_ep_scheduler_exp.sh`（自动挑选 4 张空闲 GPU）
- 守护脚本：`phase2/wait_and_run_ep_exp.sh`（每 120s 检查 GPU，空闲即启动，最多等 6h）
- 运行日志：`wait_and_run.log`
