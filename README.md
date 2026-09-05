# Unified vLLM Benchmark

All experiment settings live at the top of [`run_benchmark.py`](run_benchmark.py):

- `PP_SIZE` and `TP_SIZE_PER_STAGE` control pipeline and tensor parallelism.
- `MODEL_PATH` selects the local model directory.
- `BENCHMARK_TYPE` selects `longbench`, `classic`, or `custom`.
- `BENCHMARK_DATA_DIR`, `BENCHMARK_TASKS`, and `MAX_SAMPLES_PER_TASK` select
  file-based data.
- `build_custom_prompt()` defines the generated prompt when using `custom`.

One pipeline stage is placed on each node, so `NNODES` is derived from
`PP_SIZE`. The launcher reads the values from the Python file automatically.
Run `./launch_benchmark.sh 0` on `MASTER_ADDR`, and run the same command with
the corresponding node rank on every other node. For the current two-node
deployment:

```bash
# 192.168.0.224
./launch_benchmark.sh 0

# 192.168.0.225
./launch_benchmark.sh 1
```

Both nodes must have the same checkout, model path, benchmark data path, and
Python/vLLM environment. Results are written to `results/<UTC timestamp>/` by
rank 0. The runner uses the original `context` + `input` fields for LongBench;
it does not synthesize filler text.
