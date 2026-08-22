# v1: Decode 黑盒内部时间消耗拆分分析

> 目标:TPOT ≈ 170ms(5.9 t/s)的构成。TP=8×PP=3,24×H20,Kimi-K3。
> 方法:CUDA graph dispatch 层 wall-clock(`__call__` 插桩)+ CUDA event 层分解 + CUPTI 不可用(机密计算)。

## 一、Decode 每步的执行结构(已确认)

```
engine step:
  1. __call__  →  FULL graph replay(整个模型主体在 CUDA graph 里,Python forward 完全不执行)
  2. compute_logits( eager,0.1ms )—— graph 外的唯一 Python 层
  3. sampler / scheduler / PP 通信
```

关键机制(`vllm/compilation/breakable_cudagraph.py`):
- capture 阶段建 102 个 graph:51 个 PIECEWISE(prefill chunks,tok=512→1)+ 51 个 FULL(decode,tok=512→1,每 batch 尺寸一个)
- decode 命中 `num_tokens=1` 的 FULL graph → `_replay` 直接 `entry.capture.replay()`,**不经过模型 Python forward**
- 后果:模型层内 CUDA event 插桩在 decode 稳态**完全失效**(只对 eager/capture 有效);decode 每步唯一的插桩点是 `__call__` 与 `compute_logits`

## 二、每步时间分解(第 5 次运行,main gen 30 tokens,11.6s)

### 2.1 三 stage 的 dispatch 间隔(dt)逐毫秒同步(±5ms)

r0/r8/r16 的 30 个 FULL decode dt 几乎相同(全局节拍,三 stage 一起快一起慢):

```
r0:  7769, 115, 151, 118, 106, 94, 101, 100, 95, 93, 97, 102, 96, 103, 97,
     175, 140, 124, 99, 151, 151, 149, 169, 215, 142, 152, 200, 168, 97, 97
r8:  7763, 132, 150, 106, 97, 103, 93, 107, 88, 102, 89, 110, 98, 105, 95,
     176, 137, 137, 137, 115, 150, 144, 163, 225, 131, 210, 196, 126, 88, 105
r16: 7762, 129, 150, 106, 97, 103, 93, 107, 88, 102, 89, 110, 98, 105, 95,
     176, 137, 137, 137, 115, 151, 144, 163, 225, 130, 210, 196, 125, 88, 106
```

结构:
- **step 1:死区 7769ms**(prefill→decode 切换,三 stage 同步;warmup 时同样位置只有 1444ms)
- **step 2-15:快段,avg ≈ 106ms**
- **step 16-28:慢段,avg ≈ 152ms**(与用户生产观测的 170ms 同量级)
- **step 29-30:尾段,≈ 97ms**

### 2.2 GPU 侧时间(模型 forward,以 capture/eager 等价测量)

capture 阶段 tok=1 的 FULL graph 的 model 时间(stage 0,层插桩):

```
model 总 ≈ 108ms
├─ KDA × 24: attn ≈ 11-12ms | mlp(MoE) ≈ 42.7ms | other ≈ 11ms → 层合计 ≈ 65ms
├─ full-attn × 7: attn ≈ 9-12ms | mlp ≈ 12.7ms | other ≈ 3.3ms → 层合计 ≈ 28ms
└─ 层外(logits 前)≈ 0
```

- stage 1 的 KDA mlp = 52ms(比 stage 0 的 42.7ms 慢 ~10ms,MoE 负载/等待差异)
- r16 的 prefill 类 forward(step=108):model=237ms 但 KDA 仅 26.7ms → **~210ms 在等 PP 上游(recv 阻塞)**
- logits/lm_head = 0.1ms(可忽略)

### 2.3 分解结论(当前状态)

| 段 | dt(dispatch 周期) | GPU replay(估) | GPU 外(待第 7 次测量确认) |
|---|---|---|---|
| 快段 | ~106ms | ~108ms | ≈0(GPU 满载) |
| 慢段 | ~152ms | ~108ms | **~44ms** |
| 尾段 | ~97ms | ~108ms | <0(测量噪声) |

- 慢段 44ms 的 GPU 空闲:**唯一候选是 CPU 侧(engine/sampler/PP 通信等待)**,TP 节点内通信不是它(见下)
- dmon 历史观测:stage 2 节点 66-71% busy ≈ 108/152 ✓ 与"GPU 每步 108ms、周期 152ms"吻合

## 三、已排除/已量化的候选

| 候选 | 结论 | 依据 |
|---|---|---|
| TP 节点内 NVLink 通信 | **不是慢段 44ms 的来源** | 层内 attn+mlp 的 GPU 时间已含 all-reduce/all-to-all;且快段 GPU 满载时 TP 通信照常发生 |
| PP 跨节点通信量 | **不是瓶颈** | decode 每步 `num_tokens=1`(非 512 padding!),hidden+residual ≈ 28KB/跳,TCP 下 <1ms |
| PP 通信等待(流水线) | **部分嫌疑** | r16 prefill forward 210ms 等待;但 decode 稳态三 stage dt 同步,r16 无额外等待迹象 |
| CPU 侧(engine/sampler) | **慢段 44ms 的主要嫌疑** | 快段 GPU 满载、慢段 GPU 有空闲,且三 stage 同步——CPU 节拍 |
| CUDA graph replay 本身 | 每步 ~108ms GPU | capture 等价测量 |
| logits/lm_head | 0.1ms | event 测量 |

## 四、待第 7 次运行确认(插桩已修复)

1. **decode 每步 replay 的真实 GPU 时间**(修好的 event 对:`_replay` 前 record ev0、后 record ev1,延迟 16 步结算,零阻塞)
   - 若 replay ≈ 108ms → 慢段 44ms 为 GPU 外,CPU/PP 等待实锤
   - 若 replay ≈ 152ms → GPU 本身变慢(需查降频/负载)
2. **死区 7769ms 的性质**(prefill→decode 间;warmup 时 1444ms;第 6 次运行 main gen 19.2s 疑似死区翻倍→每次 generate 一次?)
3. **快→慢切换的触发条件**(step 15 左右;与 step 数还是时间相关)

## 五、测量方法备注

- CUPTI/torch profiler/nsys 全部不可用(机密计算)→ CUDA event + wall-clock 是唯一通道
- `__call__` 插桩:`breakable_cudagraph.py`,env `VLLM_K3_TIMING=1` 启用
- graph replay 期间 Python forward 不执行 → 层插桩只在 capture/eager 有效
- 三节点日志:nsys_run_0.log(stage0)/1.log(stage1)/2.log(stage2),每节点 8 rank
