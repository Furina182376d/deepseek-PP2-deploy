# MoE/MLP 细粒度 Profiling 与 CUPTI 排查

日期：2026-08-24  
模型：Kimi-K3，PP=3、TP=8，`VLLM_PP_LAYER_PARTITION=31,31,31`。  
运行环境：vLLM 0.27.1，三台 H20 节点。

## 1. 本次 profiling 实验

实验使用 `phase1/launch_pp.sh` 的 worker-side PyTorch profiler。profiler 请求与正常延迟测试隔离，避免 trace 开销污染 TPOT：

```text
PP_OUTPUT_LEN=256
PP_MAX_NUM_SEQS=384
PP_NUM_WARMUPS=2
PP_NUM_REPEATS=1
PP_PROFILE_ONLY=1
PP_TORCH_PROFILER_DIR=/home/tjy/kimi_bench/moe_profile_31_31_31_20260824
PP_PROFILE_WARMUP_ITERS=2
PP_PROFILE_ACTIVE_ITERS=8
PP_PROFILE_WAIT_ITERS=0
VLLM_K3_TIMING=0
```

模型加载日志确认当前 MoE 路径为：

```text
Using 'MARLIN' Mxfp4 MoE backend.
Using MarlinExperts.
```

因此目标拆分是 router/top-k、dispatch/permutation、Marlin expert GEMM、shared expert、combine 以及 TP/NCCL collective。

## 2. 实验观测

node0 保存了 global ranks 0--7 的 8 份 trace 和 8 份 `profiler_out_*.txt`；node1/node2 没有保存对应文件。实验过程中后两节点日志出现：

```text
Rank 0: Torch profiler disabled for CUDA graph capture
```

node0 trace 的统计为：

```text
traceEvents                 = 3029
CUDA kernel events          = 0
Marlin kernel-name matches  = 0
NCCL-name matches           = 24（仅用户态通信标记）
```

所有 rank 的 profiler 日志都出现：

```text
CUPTI_ERROR_CONFIDENTIAL_COMPUTING_NOT_SUPPORTED
CUPTI initialization failed - CUDA profiler activities will be missing
```

这意味着 PyTorch Kineto 可以记录 CPU 调度事件，但无法从 CUPTI 得到 CUDA kernel activity。`profiler_out_0.txt` 中的 `ProfilerStep* = 1.335 s`、`aten::copy_ self CPU = 1.093 s` 不能解释为 GPU 上的 MoE GEMM 时间。

本次实验因此只能确认 profiler 配置、三机启动和 MARLIN backend 选择成功，不能据此判断任何 MoE 子阶段是瓶颈，也不能比较三个 PP stage 的 GPU 时间。原始产物位于：

[实验产物](/home/tjy/codebases/deepseek-PP2-deploy/process/moe_profiling/pp_31_31_31_20260824/)

## 3. CUPTI 错误的含义

`CUPTI_ERROR_CONFIDENTIAL_COMPUTING_NOT_SUPPORTED` 与普通的文件权限或 `/home/tjy` 写权限不是同一个问题。它表示 CUPTI 在当前 NVIDIA 驱动/GPU/runtime 组合下无法启用 CUDA activity，错误名称指向 Confidential Computing 兼容性路径；但**仅凭这个错误不能证明 Confidential Computing 当前已开启**。也可能是驱动、CUDA runtime、CUPTI 版本组合不匹配，或云平台安全策略使 CUPTI 不可用。是否启用 CC 必须通过 `nvidia-smi -q`、云实例配置和管理员信息独立确认。

因此以下操作通常不能解决这个特定错误：

- 反复执行 `llm.start_profile()`；
- 只修改 `PP_TORCH_PROFILER_DIR` 或 trace 文件权限；
- 只设置 `CUPTI_DISABLE_PERFWORKS`、`CUDA_LAUNCH_BLOCKING` 等环境变量；
- 只把 `enforce_eager` 改成 `True`。

`enforce_eager=True` 可以绕过 CUDA graph 对 profiler 的额外限制，但不能绕过 CUPTI 的机密计算限制；它应放在 CUPTI smoke test 通过之后使用。

### 3.1 已完成的 CC 状态检查

三台节点都执行了当前 driver 支持的显式查询：

```text
nvidia-smi conf-compute -q
CC State                   : OFF
GPU CC Capabilities        : CC Capable
CC GPUs Ready State        : Ready
nvidia-smi conf-compute -f
CC status: OFF
nvidia-smi conf-compute -d
DevTools Mode: OFF
```

三台节点的 driver 都是 `595.71.05`，GPU 都是 H20。因此可以排除“当前已经
开启 Confidential Computing”这一假设；`CC Capable` 只表示硬件支持，`Ready`
只表示具备进入 CC 的条件。CUPTI 错误的根因仍应在 driver/runtime/CUPTI
兼容性、生产环境安全策略或 DevTools profiling 限制中继续定位。

## 4. 三台节点的无破坏性排查

以下检查只读，范围局限在节点本机和 `/home/tjy`。三台节点都要执行，不能只检查 node0：

```bash
source /home/tjy/miniconda3/etc/profile.d/conda.sh
conda activate vllm

nvidia-smi
nvidia-smi -q | grep -i -E 'Driver Version|CUDA Version|Confidential|CC|Compute Mode'
cat /proc/driver/nvidia/version

python - <<'PY'
import torch
print('torch =', torch.__version__)
print('torch cuda =', torch.version.cuda)
print('cuda available =', torch.cuda.is_available())
print('device =', torch.cuda.get_device_name(0))
PY

find /usr/local /home/tjy/miniconda3/envs/vllm -name 'libcupti.so*' -print 2>/dev/null
ldconfig -p 2>/dev/null | grep libcupti || true
```

重点记录：驱动版本、`torch.version.cuda`、`libcupti.so` 路径，以及 `nvidia-smi -q` 是否报告 Confidential Compute 状态。这里的 CC 状态是待验证信息，不应从 CUPTI 错误码反推。三台节点的结果必须一致；版本或状态不一致会导致只在部分 PP stage 产生 trace。

### 4.1 最小 CUPTI smoke test

先不加载 1.5 TB 模型，使用单卡小矩阵验证 CUPTI 是否能产生 CUDA activity：

```bash
rm -rf /home/tjy/cupti_smoke
mkdir -p /home/tjy/cupti_smoke
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import torch
from torch.profiler import profile, ProfilerActivity

x = torch.randn((4096, 4096), device='cuda', dtype=torch.float16)
y = torch.randn((4096, 4096), device='cuda', dtype=torch.float16)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
    for _ in range(3):
        z = x @ y
        torch.cuda.synchronize()
print(p.key_averages().table(sort_by='self_cuda_time_total', row_limit=10))
p.export_chrome_trace('/home/tjy/cupti_smoke/trace.json')
PY
```

能看到 GEMM 的 CUDA 时间表示 CUPTI 基础功能可用；若仍报同一个错误且 CUDA kernel 数为 0，则不是 vLLM 配置问题，需要处理驱动/Confidential Computing 状态。

本次已经在三台节点实际运行该 smoke test，结果均为：

```text
torch=2.13.0+cu130  torch.version.cuda=13.0  device=NVIDIA H20
CUPTI_ERROR_CONFIDENTIAL_COMPUTING_NOT_SUPPORTED
events=17  kernel_cat=0  cuda_text=0
```

所以 CUPTI 失败与 vLLM、CUDA Graph 和 PP 配置无关；最小 GEMM 也无法获得
CUDA kernel activity。

## 5. 需要管理员处理的修复

如果 smoke test 仍失败，应把三台节点的检查输出交给节点管理员，请其确认：

1. GPU 是否运行在启用 Confidential Computing/受保护 VM 的模式。先确认状态，再决定是否关闭该模式或迁移到允许 CUPTI activity 的实例/驱动配置；不能仅根据本次错误直接认定已启用。
2. NVIDIA driver、CUDA runtime、CUPTI 版本是否为受支持组合，并且三台节点完全一致。必要时重新安装匹配组件。
3. 是否变成 `CUPTI_ERROR_INSUFFICIENT_PRIVILEGES`。若是，需要管理员调整 GPU capability、设备节点权限或 perf 访问策略。
4. 节点是否运行在不允许 CUPTI 的虚拟化/安全策略下。用户态无法绕过 hypervisor 或 GPU 固件策略。

不要在没有确认目标的情况下重启 persistence daemon、卸载驱动、修改 BIOS/安全启动或改变整机安全策略；这些操作超出本实验范围。

## 6. CUPTI 无法开放时的替代 profiling

### 6.1 CUDA Event 边界计时（首选）

CUDA Event 不依赖 CUPTI activity，可以在 vLLM MoE 调用边界测 GPU elapsed time，至少覆盖：

```text
prepare/dispatch
router + top-k
MarlinExperts
shared expert
finalize/combine
TP all-reduce/all-gather
```

插桩应放在仓库维护的 vLLM patch 中同步到三台节点，不要修改 `/data/models/Kimi-K3`。边界计时形式为：

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
value = original_call(...)
end.record()
end.synchronize()
elapsed_ms = start.elapsed_time(end)
```

实际实现应复用 event，并只记录聚合统计。CUDA graph replay 下不要直接插入动态 event；先用 `enforce_eager=True` 测量，再恢复 graph 做端到端 TPOT A/B。

### 6.2 NVTX 与外部工具

可以加入 `torch.cuda.nvtx.range_push/pop`，但 NVTX 本身不提供时间，只有 Nsight 能正常采集 kernel 时才有用。本机此前 nsys 与 CUDA driver 不兼容，因此目前不能把 NVTX 当作可用替代品。

### 6.3 仅 CPU trace

当前 PyTorch profiler 仍可统计 Python 调度、tensor copy、NCCL/Gloo 调用发起开销，但这些数据不能替代 GPU kernel 时间。

## 7. 推荐执行顺序

1. 三节点完成 `nvidia-smi`/版本检查和最小 CUPTI smoke test。
2. 若 smoke test 通过：用 `enforce_eager=True`、短 decode 窗口重新采集，确认三个 PP stage 都有 CUDA kernel trace。
3. 若 smoke test 失败：暂停继续启动大模型 profiling，把结果交给管理员处理。
4. 若 CUPTI 无法获得：实施 CUDA Event 边界插桩，先测 eager，再与 CUDA graph 的端到端 TPOT 做 A/B 对照。

## 8. CUDA Event 实测结果（eager 诊断轮）

由于三台节点的最小 CUPTI smoke test 均失败，本轮改用实际 vLLM fused-MoE
路径上的 CUDA Event 插桩，并设置：

```text
PP_ENFORCE_EAGER=1
VLLM_MOE_EVENT_TIMING=1
PP_OUTPUT_LEN=256
PP_MAX_NUM_SEQS=384
PP_NUM_WARMUPS=2
PP_PROFILE_ONLY=1
VLLM_PP_LAYER_PARTITION=31,31,31
```

插桩位置是 vLLM 运行时实际调用的：

```text
MoERunner.router.select_experts       -> router_topk
FusedMoEKernel._prepare               -> prepare_dispatch
FusedMoEKernel._fused_experts         -> marlin_experts
SharedExperts                         -> shared_expert
FusedMoEKernel._finalize              -> finalize_combine
MoERunner._maybe_reduce_final_output   -> tp_reduce_final
```

每个 rank 的 decode 事件按 `tokens=1` 聚合；每个完整模型约 60 个 MoE 层，
因此每个 PP stage 约 20 个 MoE 层。下表是每次 MoE 调用的 CUDA elapsed mean，
以及同一 stage 内各部分占比：

| PP stage | router/top-k | prepare/dispatch | Marlin experts | shared expert | finalize/combine | TP reduce | 合计/次 |
|---|---:|---:|---:|---:|---:|---:|---:|
| stage 0 (ranks 0--7) | 0.095 ms (24.8%) | 0.013 ms (3.4%) | 0.149 ms (39.0%) | 0.101 ms (26.3%) | 0.013 ms (3.5%) | 0.011 ms (3.0%) | 0.383 ms |
| stage 1 (ranks 8--15) | 0.078 ms (24.2%) | 0.011 ms (3.5%) | 0.126 ms (39.3%) | 0.085 ms (26.3%) | 0.011 ms (3.5%) | 0.010 ms (3.1%) | 0.322 ms |
| stage 2 (ranks 16--23) | 0.072 ms (24.0%) | 0.010 ms (3.5%) | 0.119 ms (39.6%) | 0.079 ms (26.3%) | 0.011 ms (3.6%) | 0.009 ms (3.2%) | 0.301 ms |

按每 stage 约 20 个 MoE 层粗略折算，MoE/MLP 本身每 decode token 的 CUDA
工作量约为：stage 0 `7.66 ms`、stage 1 `6.44 ms`、stage 2 `6.01 ms`。
这是边界事件的累加估计，不是 pipeline stage wall time；attention、KDA、
layernorm、调度以及跨 stage 通信不在该表内。

### 结果解读

1. **Marlin expert GEMM 是 MoE 内部最大单项**，约占 39%；它是首要优化候选。
2. **shared expert 不是小项**，约占 26%。如果能与 routed experts 或 TP
   通信更好地 overlap，理论收益比单独优化 finalize 更大。
3. **router/top-k 约占 24%**，说明 batch=1 decode 下路由和小张量 kernel
   启动成本显著，不能只盯着 GEMM 算力。
4. prepare/dispatch、finalize/combine 和本轮可见的 TP reduce 各约 3%，不是
   当前 MoE 时间的主要部分。这里的 dispatch 是 `NoDPEP` 本地 prepare；没有
   发生跨节点 MoE all-to-all。
5. 三个 stage 的比例几乎一致，但 stage 0 的绝对值约高 20--27%，与此前
   stage 0 较慢的方向一致；这支持“继续 PP 重分区只能搬运整组层，不能消除
   单个 MoE 层内部的小 kernel 开销”的判断。

### 限制

- 这是 `enforce_eager=True` 的诊断轮。eager 会改变 kernel 调度和总延迟，不能
  直接当作 CUDA Graph 稳态 TPOT。
- CUDA Event 测的是边界内当前 stream 的 GPU elapsed；它不提供 kernel 名称，
  也无法像 CUPTI 一样进一步拆出 Marlin 的 GEMM1/GEMM2、SwiGLU 和内部
  permutation kernel。
- 事件输出按调用聚合，原始日志中包含 warmup、prefill 和 decode；表格只取
  `tokens=1` 的 decode 记录。

原始事件日志已归档到：

[`process/moe_profiling/pp_31_31_31_20260824/event_logs/`](</home/tjy/codebases/deepseek-PP2-deploy/process/moe_profiling/pp_31_31_31_20260824/event_logs/>)

### 下一步优化优先级

基于这轮数据，优先顺序应是：

1. 测试 Marlin batch=1 的 kernel 组合、workspace/permutation 开销，确认是否
   能减少每层小 kernel 启动次数；
2. 检查 shared expert 是否能和 routed Marlin 路径使用独立 stream overlap；
3. 对 router/top-k 做 fused kernel 或批量化实验；
4. 最后再考虑 finalize/TP reduce，因为它们在 MoE 内部只占约 10%。

## 9. 结论摘要

本轮已经获得了可用于优化决策的 MoE/MLP 时间分配：在 batch=1 decode 的
eager 诊断环境中，Marlin expert GEMM 约占 MoE 时间 39%，shared expert 约
占 26%，router/top-k 约占 24%，prepare/finalize/TP reduce 合计约 10%。
三个 PP stage 的比例一致，但 stage 0 的绝对 MoE 时间最高（约 0.383 ms/次，
相对 stage 1/2 的 0.323/0.301 ms）。

因此当前优化重点应从 PP 层数重分配转向：

1. 降低 Marlin batch=1 的 grouped-GEMM、workspace 和 permutation 启动开销；
2. 尝试 shared expert 与 routed Marlin 路径的 stream overlap；
3. 再优化 router/top-k 的融合和批量化。

本轮数据不能直接代表 CUDA Graph 稳态 TPOT，不能把 eager 的绝对时间当成
线上端到端延迟；但它已经明确了 MoE 内部的相对时间优先级。三台节点的
Confidential Computing 状态均为 `OFF`，CUPTI 仍因 driver/runtime 或生产
环境 profiling 限制无法提供 kernel activity，因此当前 Event 结果是可用的
GPU 边界计时替代方案。
