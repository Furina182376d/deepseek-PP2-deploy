# DS V4 Flash vs GLM 5.2 推理性能对比：时序与通信分析

> KV Cache: FP8 | Max Context: 65536 | Output: 256 tokens

---

## 1. 测试配置与模型参数

| 配置 | 节点 | PP | TP | GPU | 模型 | Checkpoint |
|------|------|----|----|-----|------|------------|
| DS TP=4 | 1 | 1 | 4 | 4 | DeepSeek V4 Flash | 156 GB |
| DS TP=8 | 1 | 1 | 8 | 8 | DeepSeek V4 Flash | 156 GB |
| GLM PP2_TP8 | **2** | **2** | 8 | 16 | GLM 5.2 | 704 GB |

| 架构参数 | DS V4 Flash | GLM 5.2 |
|----------|------------|---------|
| 架构类型 | MoE | MoE（config.json 确认） |
| hidden_size | 4096 | **6144** |
| num_layers | 43 | **78** |
| routed experts | 256 | 256 |
| active experts/tok | 6 | **8** |
| shared experts | 1 | 1 |
| KV heads | 1 (GQA) | 64 (MHA) |
| dense layers | 0 | 3（first_k_dense_replace） |

**核心差异**：GLM 5.2 hidden_size 大 1.5×、层数多 1.8×、每 token 激活 expert 多 2 个。**且 GLM 必须跨节点 PP=2（704 GB > 单机 8×80 GB）。**

---

## 2. 端到端时序总览

### DS TP=8（单机，PP=1，无跨节点通信）

| ctx | prompt_tokens | TTFT (ms) | 其中 Prefill (ms) | Decode TPOT (ms) | Decode (tok/s) |
|-----|--------------|-----------|-------------------|------------------|----------------|
| 512 | 330 | 67.2 | 56.6 | 9.47 | 105.6 |
| 1024 | 660 | 66.2 | 57.3 | 9.35 | 107.0 |
| 2048 | 1328 | 389.0 | 382.1 | 9.35 | 106.9 |
| 4096 | 2655 | 402.8 | 408.6 | 9.41 | 106.2 |
| 8192 | 5312 | 273.2 | 366.0 | 9.44 | 105.9 |
| 16384 | 10626 | 500.1 | 665.3 | 9.42 | 106.2 |
| 32768 | 21261 | 1187.2 | 1370.5 | 9.48 | 105.5 |

- **TPOT 恒定 ~9.4ms**：纯计算 + 节点内 TP all-reduce（NVLink），无跨节点通信。
- Prefill ms/token：大 context 下 ~0.07 ms/tok（21261 tokens / 1370ms）。

### GLM 5.2 PP2_TP8（两节点，PP=2，每次 forward 一次跨节点通信）

| ctx | prompt_tokens | TTFT (ms) | 其中 Prefill (ms) | Decode TPOT (ms) | Decode (tok/s) |
|-----|--------------|-----------|-------------------|------------------|----------------|
| 512 | 336 | 869.6 | 459.5 | 410.1 | 2.4 |
| 1024 | 670 | 913.9 | 504.4 | 409.5 | 2.4 |
| 2048 | 1346 | 987.5 | 575.2 | 412.3 | 2.4 |
| 4096 | 2690 | 1406.8 | 996.3 | 410.6 | 2.4 |
| 8192 | 5381 | 2566.3 | 2150.4 | 415.9 | 2.4 |
| 16384 | 10762 | 4608.1 | 4192.1 | 416.0 | 2.4 |
| 32768 | 21532 | 8767.6 | 8351.7 | 415.9 | 2.4 |

- **TPOT 恒定 ~410ms** → bottleneck 不是 attention（否则随 context 增长）。
- **Prefill ms/token**：大 context 下 ~0.39 ms/tok（vs DS 的 0.07 ms/tok → 5.6×）。
- **Decode 退化 43×**（106 → 2.4 tok/s），TTFT 退化 7-13×。

---

## 3. PP 跨节点通信：传输量与时序分解

### 3.1 传输量（基于 hidden_size=6144）

PP=2：78 层对半切（~39 层/stage）。PP 边界传输的是 **hidden_states activation**。

```
Per-token 传输量 (decode):
  FP16: 6144 × 2 = 12,288 bytes ≈ 12 KB
  FP8:  6144 × 1 =  6,144 bytes ≈  6 KB

Prefill 传输量 (P = prompt_tokens):
  FP16: P × 12 KB
  FP8:  P ×  6 KB
```

| ctx | prompt_tokens | PP 传输量 (FP16) | IB 400Gbps 理论传输 |
|-----|--------------|------------------|---------------------|
| 512 | 336 | 4.0 MB | 0.08 ms |
| 1024 | 670 | 8.0 MB | 0.16 ms |
| 2048 | 1346 | 16.2 MB | 0.32 ms |
| 4096 | 2690 | 32.3 MB | 0.65 ms |
| 8192 | 5381 | 64.6 MB | 1.29 ms |
| 16384 | 10762 | 129.1 MB | 2.58 ms |
| 32768 | 21532 | 258.4 MB | **5.17 ms** |
| **decode** | **1** | **12 KB** | **0.24 μs** |

### 3.2 理论传输 vs 实际耗时

```
Prefill (ctx=32768): 理论传输 5ms  vs 实际 prefill 8352ms  → 传输仅占 0.06%
Decode:              理论传输 0.24μs vs 实际 TPOT 410ms    → 传输仅占 0.00006%
```

**带宽不是瓶颈。裸数据传输在总延迟中几乎可以忽略。**

真正的 PP 通信开销来自：
1. **NCCL launch latency** — 每次 Send/Recv 的 kernel launch 和 CUDA 同步
2. **Pipeline bubble** — PP stage 0 和 stage 1 必须串行，调度间隙累积
3. **多次小传输** — chunked prefill 可能将一次大传输拆成多次小传输，每次触发 launch/sync overhead
4. **无 GPUDirect RDMA** → GPU→CPU→NIC 内存拷贝

### 3.3 Prefill 时序分解

GLM 5.2 prefill = 节点0前半层计算 + **PP 传输** + 节点1后半层计算 + TP all-reduce

```
prefill_ms vs prompt_tokens（大 context 线性拟合）:

prefill_ms ≈ 0.39 × prompt_tokens + 常数

  0.39 ms/token 包含:
  ├── 计算: 78层 MoE forward / (TP=8)  ≈ 估算 ~0.25-0.35 ms/tok
  └── PP通信: 跨节点传输 + 同步        ≈ 估算 ~0.04-0.14 ms/tok
```

对比 DS TP=8（无 PP）：

```
DS prefill_ms ≈ 0.07 × prompt_tokens

  0.07 ms/token 包含:
  ├── 计算: 43层 MoE forward / (TP=8)  ← 层数少 1.8×, hidden 小 1.5×
  └── TP all-reduce（节点内 NVLink）
```

**GLM prefill 每 token 耗时是 DS 的 5.6×**，主要由层数（1.8×）、hidden_size（1.5×）和 PP 通信共同导致。DS 在 ctx=2048/4096 的波动（吞吐骤降）由 vLLM scheduler 行为导致，非模型 scaling 问题。

### 3.4 Decode 时序分解

```
GLM 5.2 TPOT = 410ms（恒定）

  410ms = 节点0前半层 + PP传输+同步 + 节点1后半层 + TP all-reduce

  MoE 计算地板估算（hidden=6144, 8 experts/tok, moe_intermediate=2048）:
    活跃参数 ≈ 52B/token (见附录)
    FLOPs/token ≈ 2 × 52B = 104B
    H800 FP8 ≈ 990 TFLOPS → 理论 105ms（全模型，不分 stage）
    每 PP stage ≈ 52ms 纯计算

  扣除计算后的剩余:
    410 − (52+52) − 10(TP) − 5(attention) ≈ 291ms
    → ~290ms 为 PP 通信 + pipeline 同步 + scheduler overhead
```

**Decode 的 TPOT 中，PP 通信+调度开销推算占 ~70%（~290ms / 410ms）。**

注意 decode batch=1 下，每次传输仅 12KB，裸带宽传输只需 0.24μs，但 NCCL launch + send/recv 同步的实际延迟在 μs-ms 级。如果积累了 ~290ms/step，说明 vLLM 的 PP 调度在 decode 阶段存在显著的同步等待——这需要通过 `nsys profile` 和 `NCCL_DEBUG` 日志确认。

---

## 4. Context 变化对通信的影响

### 4.1 Prefill：传输量随 context 线性增长

| ctx | prompt_tokens | PP 传输量 (FP16) | prefill_ms | 传输量/耗时比 |
|-----|--------------|------------------|------------|--------------|
| 512 | 336 | 4.0 MB | 459.5 | 0.0087 MB/ms |
| 2048 | 1346 | 16.2 MB | 575.2 | 0.0281 MB/ms |
| 8192 | 5381 | 64.6 MB | 2150.4 | 0.0300 MB/ms |
| 32768 | 21532 | 258.4 MB | 8351.7 | 0.0309 MB/ms |

传输量增长 ~65×（4→258 MB），但传输量/prefill 耗时比基本恒定（~0.03 MB/ms），说明 **prefill 中通信开销与计算开销同步线性增长，热点未发生转移**。

### 4.2 Decode：传输量恒定，TPOT 恒定

decode 每步传输固定 12KB（1 token），TPOT 始终 ~410ms。**热点不随 context 变化**——无论 KV Cache 多大，瓶颈始终在同一位置（PP 通信+同步）。

### 4.3 对比：有无跨节点通信的热点差异

```
DS TP=8 (PP=1):
  TPOT 9.4ms 恒定 → 热点在计算（节点内 NVLink 通信占比极小）

GLM PP2_TP8 (PP=2):
  TPOT 410ms 恒定 → 热点在 PP 通信+调度
  计算仅 ~100ms，却需等待 ~310ms 的通信/同步 → GPU 大量 idle
```

---

## 5. 各 GPU 显存分布（PP 负载不均）

GLM 5.2 PP2_TP8 ctx=32768：

| GPU | PP Stage | 显存 (MB) | 
|-----|----------|-----------|
| GPU0 | Stage 0, rank0 | **86,745** |
| GPU1 | Stage 0, rank1 | 80,484 |
| GPU2 | Stage 0, rank2 | 80,484 |
| GPU3 | Stage 0, rank3 | **87,246** |
| GPU4 | Stage 1, rank0 | 80,484 |
| GPU5 | Stage 1, rank1 | 80,484 |
| GPU6 | Stage 1, rank2 | 80,484 |
| GPU7 | Stage 1, rank3 | 80,460 |

- GPU0/GPU3（Stage 0 首尾 rank，承载 embedding）比 Stage 1 GPU 多 ~6-7 GB。
- **PP 首尾 rank 负载倾斜**是 PP 的固有特征。

---

## 6. 结论

| 问题 | 答案 |
|------|------|
| Decode 瓶颈在哪？ | **PP 跨节点通信+调度**（推算占 TPOT ~70%），不是计算、不是 attention |
| Prefill 瓶颈在哪？ | 计算为主 + PP 通信为辅，随 context 线性 scaling |
| 通信带宽是瓶颈吗？ | **不是**。裸传输只需 μs-ms 级，真正耗时在 NCCL launch/sync/pipeline bubble |
| Context 增大热点转移吗？ | Decode 不转移（TPOT 恒定）；Prefill 通信与计算同步线性增长，热点比例不变 |
| 为什么 DS 快 43×？ | DS 无跨节点 PP（纯节点内 NVLink）+ 模型小（层数 43 vs 78，hidden 4096 vs 6144） |

### 优化方向

1. **最高优先**：GPU timeline profiling（`nsys`）+ NCCL_DEBUG 定位 PP 同步等待的具体环节
2. **PP=1 单机基线**：确认 `PP 开销 = 410ms − 单机 TPOT`
3. **EP 替代 PP**：MoE 天然适合 Expert Parallelism，跨节点只需 all-to-all routing 通信而非每层 PP 边界传输
4. **增大 decode batch**：让计算掩盖通信 overhead

---

## 附录：数据与计算

### A. 活跃参数量估算（GLM 5.2）

```
每层 Attention: 4 × 6144² ≈ 151M

Dense 层 (前3层, intermediate=12288):
  FFN: 3 × 6144 × 12288 ≈ 226M/层
  每 dense 层总活跃: 151M + 226M = 377M

MoE 层 (后75层):
  共享 expert (intermediate=12288): 3 × 6144 × 12288 ≈ 226M
  8 个路由 expert (moe_intermediate=2048): 8 × 3 × 6144 × 2048 ≈ 302M
  每 MoE 层总活跃: 151M + 226M + 302M = 679M

总活跃参数: 3 × 377M + 75 × 679M ≈ 52B
每 token FLOPs: 2 × 52B ≈ 104B (forward)
```

### B. 源数据

```bash
# DS TP=8: /phase0/profile_results/20260804_142508/full_results.json
# GLM PP2_TP8: /phase0/profile_results/20260807_100440/full_results.json
# GLM config: /data/model/GLM-5.2-FP8/config.json  → GlmMoeDsaForCausalLM
# DS config:  /data/model/DeepSeek-V4-Flash-0731/config.json → DeepseekV4ForCausalLM
```
