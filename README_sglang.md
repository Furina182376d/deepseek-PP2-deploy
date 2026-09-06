# SGLang 分布式基准测试

本套件独立于 vLLM 基准测试文件。运行本套件时，不要修改
`run_benchmark.py` 或 `launch_benchmark.sh`。

只需编辑 [`sglang_benchmark.py`](sglang_benchmark.py)。模型、PP/TP 拓扑、
SGLang 服务参数、基准测试数据、输出长度以及请求并发数都集中在其中的
一个配置区内。

默认拓扑为 PP=2、每个阶段 TP=8，使用两台机器：

```python
MODEL_PATH = "/data/models/DeepSeek-V4-Pro-DSpark"
PP_SIZE = 2
TP_SIZE_PER_STAGE = 8
MASTER_ADDR = "192.168.0.224"
REQUEST_CONCURRENCY = 4
```

生成的 SGLang 命令包含以下指定参数：

```text
sglang serve --trust-remote-code --model-path ... --tp 8 --pp 2 \
  --moe-runner-backend flashinfer_mxfp4 \
  --chunked-prefill-size 8192 --disable-flashinfer-autotune \
  --swa-full-tokens-ratio 0.1 --mem-fraction-static 0.90 \
  --host 0.0.0.0 --port 30000
```

此外，命令还会传入 `--nnodes`、`--node-rank` 和 `--dist-init-addr`，使两台
服务器组成 PP=2 分布式任务。启动时，运行器会读取 `sglang serve --help`，
并接受 PP 参数别名 `--pp`、`--pp-size` 或 `--pipeline-parallel-size`，以及
TP 参数别名 `--tp` 或 `--tp-size`。如果安装的 SGLang 版本不支持多机 PP，
脚本会在启动工作进程之前失败。

在配置中将 `SGLANG_CONDA_ENV` 设置为包含 SGLang 的 conda 环境名称。启动
脚本会在两台机器上激活该环境。预期的环境名称是 `sglang`，并且两台流水线
节点必须安装相同的 SGLang 构建版本。FlashInfer 的 JIT 缓存默认放在
`/tmp/sglang_flashinfer_workspace` 下；如果需要使用持久化的可写缓存，可以
覆盖 `FLASHINFER_WORKSPACE_BASE`。

默认情况下关闭投机解码：

```python
SPECULATIVE_ALGORITHM = None
```

如需启用投机解码，只需修改这一项，例如设置为
`SPECULATIVE_ALGORITHM = "DSPARK"`。随后，启动脚本会先验证已安装的 CLI
是否列出了该算法，再将工作进程放入后台运行。SGLang `0.5.10.post1` 的 CLI
没有列出 DSPARK（仅列出 `EAGLE`、`EAGLE3`、`NEXTN`、`STANDALONE` 和
`NGRAM`），因此 DSPARK 需要两台流水线节点都安装支持该算法的匹配版本。

CUDA 13.0 的 `nvcc` 目前可能在编译 DSV4 indexer metadata JIT kernel 时发生
内部编译器崩溃。启动脚本默认设置
`SGLANG_OPT_USE_JIT_INDEXER_METADATA=0`，改用环境中 `deep_gemm` 提供的实现，
以避免该问题。确认 CUDA 工具链可以正常编译该 JIT kernel 后，可通过设置
`SGLANG_OPT_USE_JIT_INDEXER_METADATA=1` 恢复 JIT 路径。

启动脚本默认设置 `FLASHINFER_DISABLE_VERSION_CHECK=1`，用于兼容镜像中
`flashinfer` 与预编译 `flashinfer-cubin` 版本暂时不一致的情况；启动时会
打印警告。长期运行或发布环境应在两台节点安装相同版本后，用
`FLASHINFER_DISABLE_VERSION_CHECK=0 ./launch_sglang_benchmark.sh <node_rank>`
启用严格校验。

对于当前 `flashinfer_python==0.6.18` 环境，匹配 cubin 可用以下命令安装
（两台节点都执行）：

```bash
python -m pip install --upgrade --no-deps --index-url https://flashinfer.ai/whl \
  flashinfer-cubin==0.6.18
```

目前已检查的节点都包含 `/data/models/DeepSeek-V4-Pro-DSpark`；因此默认的
`MODEL_PATH` 使用该本地目录。如果每台流水线节点上都存在该目录，也可以只
修改 `MODEL_PATH` 来选择参考路径 `DeepSeek-V4-Pro-0813`。

在两个终端中分别运行：

```bash
# 192.168.0.224
./launch_sglang_benchmark.sh 0

# 192.168.0.225
./launch_sglang_benchmark.sh 1
```

0 号节点会等待 `/v1/models` 就绪，然后向
`http://192.168.0.224:30000` 发送兼容 OpenAI 接口的流式补全请求。
每个基准测试批次会同时提交 `REQUEST_CONCURRENCY` 个请求。另一个节点只
负责承载分布式模型的一部分，并在 0 号节点停止服务后退出。

结果会写入 `results/sglang_<UTC timestamp>/report.json`。报告会在运行开始
前创建，在每个批次完成后更新，并包含最终的 `status`（`completed` 或
`failed`；失败时附带错误信息）以及：

- `decode_tps`：单个请求的解码速率。
- `batch_decode_tps`：该并发请求批次的总速率。
- `aggregate_decode_tps`：所有已完成批次按 token 数加权计算的总速率。
- `ttft_ms`、`tpot_ms`、`decode_ms`：根据流式 SSE token 事件测得的时间指标。

`aggregate_decode_tps` 使用批次解码的总耗时，而不是将各个流的耗时相加，
因此它是 4 流总吞吐率的正确指标。基准测试优先使用服务器提供的 OpenAI
`usage` 数据；如果 SGLang 不返回该数据，则回退到本地模型分词器，并在
每条结果中标明对应的数据来源。
