# EP 与调度实验（2026-08-16）

Batch 扫描结论的后续实验：攻 4.5ms/seq 边际成本与 FIFO 排队问题。

模型关键事实（config.json）：256 routed experts 的 MoE；GQA=1（每 token KV ≈ 5.5KB fp8，attention 非瓶颈）；权重 FP8 e4m3（128×128 分块）+ deep_gemm。→ 边际成本的主角是 MoE 专家层 GEMM 与通信。

## v2 修订说明（重要）

v1 实验失败原因，已逐一修复：

1. **LongBench 数据与旧基线目录已不在本机**（项目目录迁移时丢失，git 中也无）。→ 改用 harness 自带的合成 workload（`phase0/prompt_utils.make_prompt`，FILLER_PARAGRAPHS 填充 + 精确截断到目标长度），全部对比在同一 workload 上自洽进行。
2. **TP=2 拓扑 OOM**：TP=2（含 TP2+EP2）每卡权重 ≈ model/2，64K ctx 的 KV cache（13.32 GiB）放不下（实测仅剩 2.52 GiB，vLLM 估出 maxlen ≈ 6256）。→ A 组用 maxlen 6144 + 短 ctx 集；B 组维持 TP4 64K。
3. **batch=-1 时 0 prompt 崩溃**（range step=0）：数据丢失导致 0 条 prompt。→ `run_batch_benchmark` 增加空集守卫，报错清晰。
4. **conda 环境名 vllm → ds**（环境被重命名）。→ 脚本已更新。

## 实验设计（v2）

### A 组：EP 对照（4×GPU，maxlen 6144，短 ctx 集 {1K,2K,4K}×10 = 30 条）

| 实验 | 配置 | 回答的问题 |
|---|---|---|
| A1 | TP4, b=1 | 基准 TPOT |
| A2 | TP4, b=4 | 基准边际成本 |
| A3 | TP2+EP2, b=1 | EP 是否降低单请求 TPOT |
| A4 | TP2+EP2, b=4 | EP 是否降低边际成本 |

假设：TP=4 下每层专家 GEMM 跨 4 卡切分 + all-reduce；EP=2 专家按卡分片、token 定向 dispatch → TPOT 与边际成本应下降。

### B 组：length-aware（TP4，maxlen 64K，混合 ctx 40 条 {32K×4,16K×5,8K×6,4K×7,2K×8,1K×10}）

| 实验 | 配置 | 回答的问题 |
|---|---|---|
| B1 | batch=-1，长→短提交（FIFO） | 长 prompt 堵死短请求的量化 |
| B2 | batch=-1，短→长提交（SJF） | 排序提交的 TTFT 收益 |

假设：B2 相对 B1，短请求 TTFT p50 大幅下降（期望个位数秒），吞吐基本不变。

## 判定标准

- A 组 EP 有效：TPOT(A3) < TPOT(A1)，且 A2→A4 边际成本下降
- B 组 SJF 有效：TTFT p50/p90(B2) << B1，out_t/s 基本持平

## 结果（2026-08-16 晚）

### A 组：EP 对照 —— 假设证伪，EP 无效（甚至更慢）

| 实验 | 配置 | TPOT p50 | TTFT p50 | 吞吐 out tok/s |
|---|---|---|---|---|
| A1 | TP4, b=1 | **9.95ms** | 66ms | 98.1 |
| A3 | TP2+EP2, b=1 | 11.37ms（**+14%**） | 63ms | 86.2 |
| A2 | TP4, b=4 | **11.29ms** | — | 261.2 |
| A4 | TP2+EP2, b=4 | 14.01ms（**+24%**） | — | 264.8 |

结论：**4 卡规模、本模型上，EP 的 all-to-all dispatch 开销 > 专家分片的 GEMM 收益**。
判定标准 TPOT(A3)<TPOT(A1) 不成立。原假设"EP 是 MoE decode 最经典杠杆"在本环境证伪——且 batch 越大 EP 相对越差（+14% → +24%），说明 dispatch 通信随并发放大。
对照价值：A1 的 TPOT 9.95ms ≈ 旧 LongBench 基准 10.05ms → 合成 workload 度量有效，且 TPOT 与 ctx 无关（GQA=1 推论验证）。

### B 组：length-aware —— 效果巨大，6× 量级

| 实验 | 配置 | wall | 吞吐 | TTFT p50 | 短请求(≤4K) TTFT p50 |
|---|---|---|---|---|---|
| B1 | -1 FIFO（长→短） | 40.3s | 254.2 tok/s | 27.52s | **34.30s** |
| B2 | -1 SJF（短→长） | **6.6s** | **1545.6 tok/s** | **0.74s** | **0.74s** |

结论：
1. **提交顺序是 harness 层可控的 6× 杠杆**（wall 40.3s→6.6s，吞吐 254→1545 tok/s），零引擎改动。
2. FIFO 下短请求 TTFT p50=34.30s > 长请求的 27.11s——短请求被 4×32K 巨兽完全堵死；SJF 下全部 ≤0.81s。
3. **重要度量发现**：B1 所有请求 `queue_wait_ms` 均为 0——vLLM V1 对全量提交的请求立即 accept 进 running 队列，阻塞发生在调度等待而非 queue_time 指标。单看 queue_wait 会误判"无排队"，TTFT 分位数才是真度量。这正是度量体系的价值实例。
4. SJF 下 prefill/decode 完全重叠（engine est 3948% of wall），墙钟 ≈ max(prefill, decode) 的理论下界。

### 对 4.5ms/seq 边际成本的新认知

合成短 ctx 集（≤4K）上 TP4 b=1→b=4 的边际成本仅 **+0.45ms/seq**（9.95→11.29ms），远小于旧 LongBench 混合长 ctx 的 4.5ms/seq。说明旧诊断需修正：**边际成本的大头可能不是专家层本身，而是长 ctx 并发时的 KV/attention 带宽**（4×长序列每 token KV 读取放大）。待验证：见下节复测。

### C/D 组：边际成本归因复测（2026-08-16 晚）

| 实验 | workload | b=1 TPOT | b=4 TPOT | 边际成本/seq |
|---|---|---|---|---|
| A1/A2 | 短 ctx {1K,2K,4K}×10 | 9.95ms | 11.29ms | 0.45ms |
| C1/C2 | 均匀 8×32K | 10.04ms | 11.99ms | **0.65ms** |
| D1 | 异构 chunk {40K,1K,40K,2K} | — | 12.16ms p50 | ~0.7ms |

结论：**4.5ms/seq 无法在受控 workload 上复现**。均匀 32K 并发只有 0.65ms/seq；异构 chunk（复现旧 LongBench b=4 的"巨兽+短 prompt 同 chunk"条件）也只有 ~0.7ms/seq。旧扫描的 4.5ms/seq 很可能来自更极端条件（4×60K+ 同 chunk 的 KV 读取，按 C 组斜率外推 ≈ 14ms TPOT 仍不到 17.6ms）或当时 harness 的测量差异。

工程含义：
1. **旧"能力边界"诊断被证伪**——引擎比之前认为的更好，TPOT 10ms 附近无 4.5ms/seq 的确定性空间可挖。
2. 剩余的真实杠杆按收益排序：**SJF 提交（6×，已验证）** >> KV/attention 带宽（ctx 增长才显现，~0.2ms/seq/30K）> 专家层（EP 证伪，dispatch 更贵）。
3. 若还要攻 TPOT 本身，下一步是 ncu/nsys profile 单个 decode step 看 10ms 内部构成（专家 GEMM / all-reduce / attention / dispatch 占比）——但这已从"明确可挖"降级为"探索性"。

## 最终结论（实验顺序的三个问题）

| 问题 | 假设 | 结果 | 判定 |
|---|---|---|---|
| EP（L1，配置级） | 专家分片+定向 dispatch 降低 TPOT/边际成本 | b=1 +14%、b=4 +24% 更慢 | **证伪**，不进引擎不改 vLLM |
| length-aware（L2，harness 级） | 短先提交消除长请求阻塞 | wall 6.1×、吞吐 6.1×、TTFT p50 34.3s→0.74s | **成立**，零引擎改动，直接可用 |
| 4.5ms/seq 边际成本 | 引擎能力边界，需攻内核/调度 | 受控 workload 上仅 0.45-0.7ms/seq | **证伪**，无确定性空间可挖 |

下一步按收益排序：
1. **SJF 提交进生产路径**（harness 排序，已验证 6×，零风险）
2. 度量修正：queue_wait_ms 在 batch=-1 下失真（恒为 0），排队度量改用 TTFT 分位数
3. 探索性：ncu/nsys profile 单个 decode step，看 10ms TPOT 内部构成（若还想动内核/调度，先拿剖面证据，走 plugin > upstream PR > fork 顺序）

## 文件

- 实验脚本：`phase2/run_ep_scheduler_exp.sh`
- 运行日志：`reports/ep_scheduler_experiments/run.log`
- 结果目录：`phase0/profile_results/<时间戳>/`（每实验一个）
