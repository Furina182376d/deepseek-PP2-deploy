#!/usr/bin/env python3
"""Unified multi-node vLLM benchmark runner.

Edit the CONFIGURATION section below, then run ``launch_benchmark.sh`` on
each pipeline node.  The launcher reads PP/TP, node count, model path and
network settings from this file, so no second configuration file is needed.

Supported benchmark modes:
  * ``longbench``: LongBench JSON/JSONL files.  Select files with
    ``BENCHMARK_TASKS`` or leave it empty to use all non-``*_e`` files.
  * ``classic``: generic JSON/JSONL records with ``context`` + ``input``,
    ``prompt``, or ``text`` fields.
  * ``custom``: prompts returned by ``build_custom_prompt`` below.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# =============================================================================
# CONFIGURATION: edit this block for a new experiment.
# =============================================================================
MODEL_PATH = "/data/models/DeepSeek-V4-Pro-DSpark"

# One node hosts one pipeline stage.  Therefore NNODES must equal PP_SIZE.
PP_SIZE = 2
TP_SIZE_PER_STAGE = 8
NNODES = PP_SIZE
MASTER_ADDR = "192.168.0.224"
MASTER_PORT = 29500
NETWORK_INTERFACE = "eth0"

# ``longbench``, ``classic``, or ``custom``.
BENCHMARK_TYPE = "longbench"
BENCHMARK_DATA_DIR = "/home/tjy/benchmarks/longbench/data"
# Empty means all standard (*.jsonl/*.json) files in BENCHMARK_DATA_DIR.
BENCHMARK_TASKS: tuple[str, ...] = ("qmsum", "gov_report")
MAX_SAMPLES_PER_TASK = 1

# Custom prompt workload.  Change build_custom_prompt() for custom logic.
CUSTOM_PROMPT_COUNT = 4

OUTPUT_TOKENS = 128
NUM_WARMUPS = 1
NUM_REPEATS = 3
MAX_MODEL_LEN = 200000
MAX_NUM_SEQS = 16
GPU_MEMORY_UTILIZATION = 0.95
KV_CACHE_DTYPE = "fp8"
BLOCK_SIZE = 256
ENABLE_EXPERT_PARALLEL = True
PREFILL_CONTEXT_PARALLEL_SIZE = 1
DECODE_CONTEXT_PARALLEL_SIZE = 1
# Leave these unset for generic models.  DSpark is detected from MODEL_PATH
# and receives its DeepSeek-V4 tokenizer/parser automatically.
TOKENIZER_MODE: str | None = None
REASONING_PARSER: str | None = None
TRUST_REMOTE_CODE = True
COMPILATION_CONFIG: dict[str, Any] | None = {
    "mode": 0,
    "cudagraph_mode": "FULL_DECODE_ONLY",
}
ENFORCE_EAGER = False
ENABLE_FLASHINFER_AUTOTUNE = False
SPECULATIVE_CONFIG: dict[str, Any] | None = None

RESULTS_DIR = "results"


def build_custom_prompt(index: int) -> str:
    """Return one prompt for the ``custom`` benchmark mode.

    Replace this function with any deterministic prompt generator.  The
    ``index`` argument ranges from zero to ``CUSTOM_PROMPT_COUNT - 1``.
    """
    return (
        "You are evaluating a language model. Analyze the following item and "
        f"respond with a concise numbered answer. Item index: {index}."
    )


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


def _rank_info() -> dict[str, int | bool]:
    local_world_size = _env_int("LOCAL_WORLD_SIZE", TP_SIZE_PER_STAGE)
    global_rank = _env_int("RANK", 0)
    world_size = _env_int("WORLD_SIZE", PP_SIZE * TP_SIZE_PER_STAGE)
    local_rank = _env_int("LOCAL_RANK", 0)
    node_rank = global_rank // local_world_size
    return {
        "global_rank": global_rank,
        "local_rank": local_rank,
        "local_world_size": local_world_size,
        "world_size": world_size,
        "node_rank": node_rank,
        "is_leader": global_rank == 0,
    }


def validate_config() -> None:
    errors: list[str] = []
    if not MODEL_PATH:
        errors.append("MODEL_PATH must not be empty")
    if PP_SIZE <= 0 or TP_SIZE_PER_STAGE <= 0:
        errors.append("PP_SIZE and TP_SIZE_PER_STAGE must be positive")
    if NNODES != PP_SIZE:
        errors.append("NNODES must equal PP_SIZE (one pipeline stage per node)")
    if not MASTER_ADDR:
        errors.append("MASTER_ADDR must not be empty")
    if BENCHMARK_TYPE not in {"longbench", "classic", "custom"}:
        errors.append("BENCHMARK_TYPE must be longbench, classic, or custom")
    if BENCHMARK_TYPE != "custom" and not BENCHMARK_DATA_DIR:
        errors.append("BENCHMARK_DATA_DIR is required for file-based benchmarks")
    if BENCHMARK_TYPE == "custom" and CUSTOM_PROMPT_COUNT <= 0:
        errors.append("CUSTOM_PROMPT_COUNT must be positive")
    if MAX_SAMPLES_PER_TASK < 0:
        errors.append("MAX_SAMPLES_PER_TASK must be >= 0")
    if OUTPUT_TOKENS <= 0 or MAX_MODEL_LEN <= 0 or MAX_NUM_SEQS <= 0:
        errors.append("OUTPUT_TOKENS, MAX_MODEL_LEN and MAX_NUM_SEQS must be positive")
    if NUM_WARMUPS < 0 or NUM_REPEATS <= 0:
        errors.append("NUM_WARMUPS must be >= 0 and NUM_REPEATS must be positive")
    if not 0 < GPU_MEMORY_UTILIZATION <= 1:
        errors.append("GPU_MEMORY_UTILIZATION must be in (0, 1]")
    if BLOCK_SIZE <= 0:
        errors.append("BLOCK_SIZE must be positive")
    if errors:
        raise ValueError("Invalid benchmark configuration:\n  " + "\n  ".join(errors))


def _model_specific_options() -> dict[str, str]:
    """Return options needed by model families with custom tokenization."""
    model_name = MODEL_PATH.lower()
    if TOKENIZER_MODE or REASONING_PARSER:
        options: dict[str, str] = {}
        if TOKENIZER_MODE:
            options["tokenizer_mode"] = TOKENIZER_MODE
        if REASONING_PARSER:
            options["reasoning_parser"] = REASONING_PARSER
        return options
    if "deepseek-v4" in model_name or "dspark" in model_name:
        return {"tokenizer_mode": "deepseek_v4", "reasoning_parser": "deepseek_v4"}
    return {}


def _task_paths(data_dir: Path, tasks: tuple[str, ...]) -> list[Path]:
    if tasks:
        paths = [
            data_dir / (task if task.endswith((".json", ".jsonl")) else f"{task}.jsonl")
            for task in tasks
        ]
    else:
        paths = sorted(
            path
            for path in data_dir.iterdir()
            if path.suffix in {".json", ".jsonl"} and not path.stem.endswith("_e")
        )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Benchmark file(s) not found: " + ", ".join(missing))
    return paths


def _read_records(path: Path, limit: int) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        if path.suffix == ".jsonl":
            records = [json.loads(line) for line in stream if line.strip()]
        else:
            data = json.load(stream)
            records = data if isinstance(data, list) else [data]
    return records[:limit] if limit > 0 else records


def _record_prompt(record: dict[str, Any]) -> str:
    if record.get("prompt"):
        return str(record["prompt"])
    if record.get("text"):
        return str(record["text"])
    context = str(record.get("context") or "")
    question = str(record.get("input") or record.get("question") or "")
    if context and question:
        return f"{context}\n\n{question}"
    return context or question


def load_requests() -> list[dict[str, Any]]:
    if BENCHMARK_TYPE == "custom":
        return [
            {"task": "custom", "sample": index, "prompt": build_custom_prompt(index)}
            for index in range(CUSTOM_PROMPT_COUNT)
        ]

    data_dir = Path(BENCHMARK_DATA_DIR)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Benchmark data directory does not exist: {data_dir}")
    requests: list[dict[str, Any]] = []
    for path in _task_paths(data_dir, BENCHMARK_TASKS):
        for index, record in enumerate(_read_records(path, MAX_SAMPLES_PER_TASK)):
            prompt = _record_prompt(record)
            if prompt.strip():
                requests.append({"task": path.stem, "sample": index, "prompt": prompt})
    if not requests:
        raise ValueError("No non-empty benchmark prompts were loaded")
    return requests


def _sampling_params(SamplingParams: Any) -> Any:
    return SamplingParams(temperature=0, max_tokens=OUTPUT_TOKENS, ignore_eos=True)


def _request_metrics(output: Any, elapsed_s: float) -> dict[str, float | int]:
    prompt_tokens = len(getattr(output, "prompt_token_ids", []) or [])
    candidates = getattr(output, "outputs", []) or []
    output_tokens = len(getattr(candidates[0], "token_ids", []) or []) if candidates else 0
    metrics = getattr(output, "metrics", None)
    first_latency = getattr(metrics, "first_token_latency", None) if metrics else None
    first_ts = getattr(metrics, "first_token_ts", None) if metrics else None
    last_ts = getattr(metrics, "last_token_ts", None) if metrics else None
    generated = int(getattr(metrics, "num_generation_tokens", output_tokens) or output_tokens)
    if first_ts is not None and last_ts is not None and generated > 1:
        tpot_s = max(0.0, (last_ts - first_ts) / (generated - 1))
    else:
        tpot_s = 0.0
    ttft_s = float(first_latency or 0.0)
    prefill_s = max(0.0, ttft_s - tpot_s)
    return {
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "elapsed_ms": elapsed_s * 1000,
        "ttft_ms": ttft_s * 1000,
        "tpot_ms": tpot_s * 1000,
        "prefill_tps": prompt_tokens / prefill_s if prefill_s > 0 else 0.0,
        "decode_tps": 1.0 / tpot_s if tpot_s > 0 else 0.0,
    }


def _gpu_memory(torch: Any) -> list[float]:
    if not torch.cuda.is_available():
        return []
    memory: list[float] = []
    for index in range(TP_SIZE_PER_STAGE):
        free, total = torch.cuda.mem_get_info(index)
        memory.append(round((total - free) / 1024**2, 1))
    return memory


def _write_results(request_rows: list[dict[str, Any]], rank: dict[str, int | bool]) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(RESULTS_DIR) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": MODEL_PATH,
        "benchmark_type": BENCHMARK_TYPE,
        "benchmark_data_dir": BENCHMARK_DATA_DIR,
        "benchmark_tasks": list(BENCHMARK_TASKS),
        "pp_size": PP_SIZE,
        "tp_size_per_stage": TP_SIZE_PER_STAGE,
        "nnodes": NNODES,
        "enable_expert_parallel": ENABLE_EXPERT_PARALLEL,
        "output_tokens": OUTPUT_TOKENS,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "rank": rank,
    }
    (output_dir / "summary.json").write_text(
        json.dumps({"metadata": metadata, "results": request_rows}, indent=2),
        encoding="utf-8",
    )
    if request_rows:
        with (output_dir / "requests.csv").open("w", newline="", encoding="utf-8") as stream:
            fieldnames = list(request_rows[0])
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(request_rows)
    print(f"Results written to {output_dir}")


def main() -> int:
    validate_config()
    rank = _rank_info()
    expected_world = PP_SIZE * TP_SIZE_PER_STAGE
    if int(rank["local_world_size"]) != TP_SIZE_PER_STAGE:
        raise ValueError(
            f"LOCAL_WORLD_SIZE={rank['local_world_size']} but TP_SIZE_PER_STAGE={TP_SIZE_PER_STAGE}"
        )
    if int(rank["world_size"]) != expected_world:
        raise ValueError(
            f"WORLD_SIZE={rank['world_size']} but PP_SIZE*TP_SIZE_PER_STAGE={expected_world}"
        )

    # CUDA/vLLM are imported only after configuration inspection, which lets
    # launch_benchmark.sh query this file without initializing a GPU runtime.
    import torch
    import torch.distributed as dist
    from vllm import LLM, SamplingParams

    model_dir = Path(MODEL_PATH)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    kwargs: dict[str, Any] = {
        "model": MODEL_PATH,
        "pipeline_parallel_size": PP_SIZE,
        "tensor_parallel_size": TP_SIZE_PER_STAGE,
        "prefill_context_parallel_size": PREFILL_CONTEXT_PARALLEL_SIZE,
        "decode_context_parallel_size": DECODE_CONTEXT_PARALLEL_SIZE,
        "enable_expert_parallel": ENABLE_EXPERT_PARALLEL,
        "block_size": BLOCK_SIZE,
        "max_model_len": MAX_MODEL_LEN,
        "max_num_seqs": MAX_NUM_SEQS,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "trust_remote_code": TRUST_REMOTE_CODE,
        "kv_cache_dtype": KV_CACHE_DTYPE,
        "enable_flashinfer_autotune": ENABLE_FLASHINFER_AUTOTUNE,
        "enforce_eager": ENFORCE_EAGER,
        "distributed_executor_backend": "external_launcher",
        "nnodes": NNODES,
        "node_rank": int(rank["node_rank"]),
        "master_addr": MASTER_ADDR,
        "master_port": MASTER_PORT,
        "distributed_timeout_seconds": 10800,
        "cpu_distributed_timeout_seconds": 10800,
    }
    kwargs.update(_model_specific_options())
    if COMPILATION_CONFIG is not None:
        kwargs["compilation_config"] = COMPILATION_CONFIG
    if SPECULATIVE_CONFIG is not None:
        kwargs["speculative_config"] = SPECULATIVE_CONFIG

    if bool(rank["is_leader"]):
        print(
            f"Loading {MODEL_PATH} | benchmark={BENCHMARK_TYPE} | "
            f"PP={PP_SIZE} TP={TP_SIZE_PER_STAGE} EP={ENABLE_EXPERT_PARALLEL}"
        )
    llm = LLM(**kwargs)
    if dist.is_initialized():
        dist.barrier()

    requests = load_requests()
    params = _sampling_params(SamplingParams)
    if bool(rank["is_leader"]):
        print(f"Loaded {len(requests)} benchmark requests")
    for _ in range(NUM_WARMUPS):
        for request in requests:
            llm.generate([request["prompt"]], params, use_tqdm=False)

    rows: list[dict[str, Any]] = []
    for request in requests:
        for repeat in range(NUM_REPEATS):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            outputs = llm.generate([request["prompt"]], params, use_tqdm=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if not bool(rank["is_leader"]):
                continue
            metrics = _request_metrics(outputs[0], elapsed)
            row = {
                "task": request["task"],
                "sample": request["sample"],
                "repeat": repeat + 1,
                **metrics,
                "gpu_memory_mb": _gpu_memory(torch),
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    if bool(rank["is_leader"]):
        _write_results(rows, rank)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
