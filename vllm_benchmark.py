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
import functools
import json
import logging
import os
import re
import sys
import threading
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
MAX_SAMPLES_PER_TASK = 2

# Custom prompt workload.  Change build_custom_prompt() for custom logic.
CUSTOM_PROMPT_COUNT = 4

OUTPUT_TOKENS = 128
NUM_WARMUPS = 1
NUM_REPEATS = 3
# Number of requests submitted to vLLM in one batch. Set to 1 for the
# original single-stream measurement; 4 measures four concurrent streams.
REQUEST_CONCURRENCY = 4

# How many times a batch may be measured again after it triggered a kernel JIT
# compilation. vLLM builds Triton/TileLang kernels on the first step that needs
# a given shape, and a build inside a measured batch charges seconds to minutes
# of compile time to that batch's TTFT/TPOT: those numbers describe the
# compiler, not the model. Such a batch is discarded and run again, which is
# clean because the kernel is cached after the discarded attempt. A batch that
# is still compiling after this many retries is kept with
# ``jit_contaminated`` set and left out of the aggregate. 0 detects and
# excludes without re-running.
JIT_RETRY_LIMIT = 3
MAX_MODEL_LEN = 200000
MAX_NUM_SEQS = 16
GPU_MEMORY_UTILIZATION = 0.9
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
    if JIT_RETRY_LIMIT < 0:
        errors.append("JIT_RETRY_LIMIT must be >= 0")
    if REQUEST_CONCURRENCY <= 0 or REQUEST_CONCURRENCY > MAX_NUM_SEQS:
        errors.append("REQUEST_CONCURRENCY must be in [1, MAX_NUM_SEQS]")
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


def _sampling_params(SamplingParams: Any, RequestOutputKind: Any) -> Any:
    # DELTA output lets the offline runner observe the first and last token
    # boundaries instead of receiving only one final aggregate output.
    return SamplingParams(
        temperature=0,
        max_tokens=OUTPUT_TOKENS,
        ignore_eos=True,
        output_kind=RequestOutputKind.DELTA,
    )


def _run_request(
    llm: Any,
    prompt: str,
    params: Any,
    request_id: str,
    torch: Any,
) -> tuple[Any, dict[str, float | int]]:
    """Run one request while observing token emission boundaries.

    ``LLM.generate`` intentionally hides intermediate outputs. Driving the
    engine directly is supported by vLLM's offline API and gives us reliable
    wall-clock TTFT/decode boundaries even when RequestOutput.metrics is not
    populated by a particular vLLM build.
    """
    started = time.perf_counter()
    # Start before add_request so tokenization, queueing and the first engine
    # step are included in the observed time-to-first-token.
    llm.llm_engine.add_request(request_id, prompt, params)
    first_token_at: float | None = None
    finished_output: Any | None = None
    generated_tokens = 0

    while llm.llm_engine.has_unfinished_requests():
        step_outputs = llm.llm_engine.step()
        step_finished_at = time.perf_counter()
        for output in step_outputs:
            token_count = sum(
                len(getattr(candidate, "token_ids", ()) or ())
                for candidate in (getattr(output, "outputs", ()) or ())
            )
            generated_tokens += token_count
            if token_count and first_token_at is None:
                first_token_at = step_finished_at
            if getattr(output, "finished", False):
                finished_output = output

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    finished_at = time.perf_counter()
    if finished_output is None:
        raise RuntimeError(f"Request did not produce a finished output: {request_id}")

    if first_token_at is None:
        first_token_at = finished_at
    return finished_output, {
        "wall_elapsed_s": finished_at - started,
        "wall_ttft_s": max(0.0, first_token_at - started),
        "wall_decode_s": max(0.0, finished_at - first_token_at),
        "wall_generated_tokens": generated_tokens,
    }


# =============================================================================
# JIT compilation guard
# =============================================================================
# vLLM compiles kernels on the first step that needs a given shape. When that
# happens inside a measured batch, the kernel build stalls the engine and the
# batch's TTFT/TPOT report compile time instead of model latency. vLLM's
# ``jit_monitor`` announces those events ("<backend> JIT compilation during
# inference: <kernel>"), so the runner counts them, throws the affected batch
# away and measures it again.
#
# Two details matter for the counting to be trustworthy:
#   * The monitor logs through ``warning_once``, which suppresses a kernel that
#     is later compiled again for another shape, so the log records alone can
#     miss events. The counts come from the monitor's own funnel instead
#     (``_handle_jit_event``, shared by the Triton/CuTeDSL/TileLang hooks), with
#     the log handler kept as a fallback.
#   * A compile happens on whichever rank executes that layer, and every other
#     rank stalls behind it in the pipeline. Rank-local counts would therefore
#     hide node-2 compiles from node-0 timings, so ranks sum their counts and
#     only re-run a batch when all of them agree it was clean.

JIT_EVENT_PATTERN = re.compile(r"JIT compilation during inference")


class _JitCompileCounter:
    """Collect the kernel JIT compilations reported while a batch runs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._count = 0
        self._kernels: list[str] = []

    def record(self, kernel: str) -> None:
        with self._lock:
            self._count += 1
            if kernel not in self._kernels:
                self._kernels.append(kernel)

    def take(self) -> tuple[int, list[str]]:
        """Return the events collected so far and reset the counter."""
        with self._lock:
            count, kernels = self._count, self._kernels
            self._count, self._kernels = 0, []
        return count, kernels


class _JitWarningHandler(logging.Handler):
    """Count JIT warnings that reach the ``vllm`` logger.

    Fallback used when the monitor's own funnel cannot be hooked; it cannot see
    events that ``warning_once`` deduplicates, hence the ordering above.
    """

    def __init__(self, counter: _JitCompileCounter) -> None:
        super().__init__(level=logging.WARNING)
        self._counter = counter

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:  # a broken record must never break the benchmark
            return
        if JIT_EVENT_PATTERN.search(message):
            self._counter.record(_jit_event_label(message))


def _jit_event_label(message: str) -> str:
    """Turn a monitor warning into a short kernel label."""
    tail = message.split("during inference:", 1)[-1].strip()
    return tail.split(".", 1)[0].strip() or message


def _jit_event_name(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Read ``fn_name`` out of a monitor call, whatever way it was passed."""
    name = kwargs.get("fn_name")
    if name is None and len(args) >= 3:
        name = args[2]
    return str(name or "unknown")


def install_jit_guard(counter: _JitCompileCounter, announce: bool = False) -> list[str]:
    """Route every JIT signal vLLM exposes into ``counter``.

    Returns the installed signal names so the report can show whether the guard
    was active.
    """
    installed: list[str] = []
    hooked = False
    try:
        from vllm.utils import jit_monitor

        original = jit_monitor._handle_jit_event

        @functools.wraps(original)
        def _counted(*args: Any, **kwargs: Any) -> Any:
            counter.record(_jit_event_name(args, kwargs))
            return original(*args, **kwargs)

        jit_monitor._handle_jit_event = _counted
        installed.append("vllm.utils.jit_monitor")
        hooked = True
    except Exception as exc:  # pragma: no cover - depends on the vLLM build
        if announce:
            print(
                f"Warning: cannot hook vLLM's JIT monitor ({exc}); "
                "relying on its log records instead.",
                flush=True,
            )

    # Only one signal counts a given compilation: every monitor warning is
    # emitted from the funnel, so adding the handler on top would double the
    # counters the report exposes.
    if not hooked:
        try:
            logging.getLogger("vllm").addHandler(_JitWarningHandler(counter))
            installed.append("vllm logger")
        except Exception as exc:  # pragma: no cover - logging itself failing
            if announce:
                print(f"Warning: cannot attach the JIT log handler ({exc}).", flush=True)
    return installed


def count_all_rank_jit_events(torch: Any, dist: Any, local_count: int) -> int:
    """Sum a batch's JIT events over all ranks.

    Every rank has to reach the same verdict, because they re-run the batch
    together or not at all.
    """
    if dist is None or not dist.is_initialized():
        return local_count
    device = torch.device("cuda", torch.cuda.current_device())
    tensor = torch.tensor([local_count], dtype=torch.int64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return int(tensor.item())


def _benchmark_request_id(task: str, sample: int, repeat: int, attempt: int) -> str:
    """Build the request ID for one attempt of a measured batch.

    Attempt 0 keeps the original ID; later attempts add a suffix so a re-run
    never reuses the ID of the request it replaces.
    """
    suffix = "" if attempt == 0 else f"-retry{attempt}"
    return f"benchmark-{task}-{sample}-{repeat}{suffix}"


def _run_batch_until_jit_free(
    llm: Any,
    request_batch: list[dict[str, Any]],
    repeat: int,
    torch: Any,
    dist: Any,
    counter: _JitCompileCounter,
    batch_id: str,
    announce: bool,
    make_params: Any,
) -> tuple[dict[str, tuple[Any, dict[str, float | int]]], dict[str, float | int], int, bool, list[str]]:
    """Run one measured batch, re-running it while it compiles kernels.

    ``make_params`` is called once per request and attempt, so every request
    gets its own sampling parameters exactly as an unguarded run would.
    Returns the results and timing of the accepted attempt plus the number of
    discarded attempts, whether the last attempt still compiled something, and
    the kernels seen. Request IDs carry the attempt number so a re-run never
    reuses the IDs of the discarded attempt.
    """
    attempts = 0
    kernels_seen: list[str] = []
    while True:
        batch = [
            (
                _benchmark_request_id(request["task"], request["sample"], repeat, attempts),
                request["prompt"],
                make_params(),
            )
            for request in request_batch
        ]
        counter.take()
        batch_results, batch_timing = _run_batch(llm, batch, torch)
        local_count, local_kernels = counter.take()
        compiled_here = count_all_rank_jit_events(torch, dist, local_count) > 0
        kernels_seen.extend(kernel for kernel in local_kernels if kernel not in kernels_seen)
        if not compiled_here or attempts >= JIT_RETRY_LIMIT:
            return batch_results, batch_timing, attempts, compiled_here, kernels_seen
        attempts += 1
        if announce:
            named = ", ".join(local_kernels) if local_kernels else "a kernel"
            print(
                f"Warning: {batch_id} compiled {named} mid-batch, so its timings "
                f"measure the compiler; discarding it and measuring again "
                f"({attempts}/{JIT_RETRY_LIMIT}).",
                flush=True,
            )


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _slow_steps(
    step_durations: list[float],
    first_token_step: int | None,
    top: int = 3,
) -> list[dict[str, float | int | str]]:
    """The longest engine steps of a batch, labelled by phase.

    TTFT spread between otherwise identical batches is either one stalled step
    (a compile or a transport hiccup) or every step being slower (compute), and
    this list tells the two apart.
    """
    ranked = sorted(
        range(len(step_durations)), key=lambda index: step_durations[index], reverse=True
    )[:top]
    return [
        {
            "step": index,
            "ms": round(step_durations[index] * 1000, 1),
            "phase": (
                "prefill"
                if first_token_step is None or index <= first_token_step
                else "decode"
            ),
        }
        for index in sorted(ranked)
    ]


def _run_batch(
    llm: Any,
    batch: list[tuple[str, str, Any]],
    torch: Any,
) -> tuple[dict[str, tuple[Any, dict[str, float | int]]], dict[str, float | int]]:
    """Submit and execute several requests concurrently through one engine."""
    started = time.perf_counter()
    for request_id, prompt, params in batch:
        llm.llm_engine.add_request(request_id, prompt, params)

    state: dict[str, dict[str, Any]] = {
        request_id: {
            "first_token_at": None,
            "finished_at": None,
            "finished_output": None,
            "generated_tokens": 0,
        }
        for request_id, _, _ in batch
    }
    step_durations: list[float] = []
    first_token_step: int | None = None
    while llm.llm_engine.has_unfinished_requests():
        step_started_at = time.perf_counter()
        step_outputs = llm.llm_engine.step()
        step_finished_at = time.perf_counter()
        step_durations.append(step_finished_at - step_started_at)
        for output in step_outputs:
            request_id = str(getattr(output, "request_id", ""))
            if request_id not in state:
                continue
            token_count = sum(
                len(getattr(candidate, "token_ids", ()) or ())
                for candidate in (getattr(output, "outputs", ()) or ())
            )
            item = state[request_id]
            item["generated_tokens"] += token_count
            if token_count and item["first_token_at"] is None:
                item["first_token_at"] = step_finished_at
                if first_token_step is None:
                    first_token_step = len(step_durations) - 1
            if getattr(output, "finished", False):
                item["finished_output"] = output
                item["finished_at"] = step_finished_at

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    batch_finished_at = time.perf_counter()
    if any(item["finished_output"] is None for item in state.values()):
        missing = [request_id for request_id, item in state.items() if item["finished_output"] is None]
        raise RuntimeError(f"Requests did not finish: {', '.join(missing)}")

    results: dict[str, tuple[Any, dict[str, float | int]]] = {}
    first_tokens = [item["first_token_at"] for item in state.values() if item["first_token_at"] is not None]
    batch_first_token_at = min(first_tokens) if first_tokens else batch_finished_at
    total_decode_tokens = 0
    for request_id, item in state.items():
        first_token_at = item["first_token_at"] or item["finished_at"] or batch_finished_at
        finished_at = item["finished_at"] or batch_finished_at
        generated_tokens = int(item["generated_tokens"])
        total_decode_tokens += max(generated_tokens - 1, 0)
        results[request_id] = (
            item["finished_output"],
            {
                "wall_elapsed_s": finished_at - started,
                "wall_ttft_s": max(0.0, first_token_at - started),
                "wall_decode_s": max(0.0, finished_at - first_token_at),
                "wall_generated_tokens": generated_tokens,
            },
        )
    return results, {
        "batch_elapsed_s": batch_finished_at - started,
        "batch_decode_s": max(0.0, batch_finished_at - batch_first_token_at),
        "batch_decode_tokens": total_decode_tokens,
        "batch_decode_tps": (
            total_decode_tokens / (batch_finished_at - batch_first_token_at)
            if batch_finished_at > batch_first_token_at
            else 0.0
        ),
        "batch_concurrency": len(batch),
        "batch_steps": len(step_durations),
        "batch_step_p50_ms": (
            _median(step_durations) * 1000 if step_durations else 0.0
        ),
        "batch_step_max_ms": max(step_durations) * 1000 if step_durations else 0.0,
        "batch_slow_steps": _slow_steps(step_durations, first_token_step),
    }


def _request_metrics(
    output: Any,
    timing: dict[str, float | int],
) -> dict[str, float | int | str]:
    prompt_tokens = len(getattr(output, "prompt_token_ids", []) or [])
    candidates = getattr(output, "outputs", []) or []
    output_tokens = len(getattr(candidates[0], "token_ids", []) or []) if candidates else 0
    metrics = getattr(output, "metrics", None)
    generated = int(
        timing.get("wall_generated_tokens", 0)
        or getattr(metrics, "num_generation_tokens", 0)
        or output_tokens
    )

    # Prefer vLLM's engine timestamps when they are non-zero. The direct
    # step-wall-clock values are the fallback for offline builds that return
    # a metrics object with zeroed timestamps.
    first_latency = float(getattr(metrics, "first_token_latency", 0.0) or 0.0)
    first_ts = float(getattr(metrics, "first_token_ts", 0.0) or 0.0)
    last_ts = float(getattr(metrics, "last_token_ts", 0.0) or 0.0)
    scheduled_ts = float(getattr(metrics, "scheduled_ts", 0.0) or 0.0)
    engine_timing_valid = (
        first_latency > 0.0
        and first_ts > 0.0
        and last_ts > first_ts
        and generated > 0
    )
    if engine_timing_valid:
        ttft_s = first_latency
        prefill_s = max(0.0, first_ts - scheduled_ts) if scheduled_ts > 0 else ttft_s
        decode_s = max(0.0, last_ts - first_ts)
        timing_source = "vllm_engine_metrics"
    else:
        ttft_s = float(timing["wall_ttft_s"])
        prefill_s = ttft_s
        decode_s = float(timing["wall_decode_s"])
        timing_source = "step_wall_clock"

    decode_tokens = max(generated - 1, 0)
    tpot_s = decode_s / decode_tokens if decode_tokens > 0 and decode_s > 0 else 0.0
    return {
        "prompt_tokens": prompt_tokens,
        "output_tokens": generated,
        "elapsed_ms": float(timing["wall_elapsed_s"]) * 1000,
        "ttft_ms": ttft_s * 1000,
        "prefill_ms": prefill_s * 1000,
        "decode_ms": decode_s * 1000,
        "tpot_ms": tpot_s * 1000,
        "prefill_tps": prompt_tokens / prefill_s if prefill_s > 0 else 0.0,
        "decode_tps": decode_tokens / decode_s if decode_s > 0 else 0.0,
        "decode_tokens": decode_tokens,
        "timing_source": timing_source,
    }


def _gpu_memory(torch: Any) -> list[float]:
    if not torch.cuda.is_available():
        return []
    memory: list[float] = []
    for index in range(TP_SIZE_PER_STAGE):
        free, total = torch.cuda.mem_get_info(index)
        memory.append(round((total - free) / 1024**2, 1))
    return memory


def _create_results_dir(rank: dict[str, int | bool]) -> Path | None:
    if not bool(rank["is_leader"]):
        return None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(RESULTS_DIR) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _report_aggregate(
    request_rows: list[dict[str, Any]],
    batch_rows: list[dict[str, Any]] | None = None,
) -> dict[str, float | int]:
    """Summarize completed requests, including end-to-end decode throughput.

    A batch that kept compiling kernels after ``JIT_RETRY_LIMIT`` attempts was
    measuring the compiler rather than the model, so it is counted in
    ``contaminated_*`` and left out of every average and throughput below. It
    stays visible in ``results``, flagged with ``jit_contaminated``.
    """
    all_batch_rows = batch_rows or []
    contaminated_requests = sum(1 for row in request_rows if row.get("jit_contaminated"))
    contaminated_batches = sum(1 for row in all_batch_rows if row.get("jit_contaminated"))
    request_rows = [row for row in request_rows if not row.get("jit_contaminated")]
    batch_rows = [row for row in all_batch_rows if not row.get("jit_contaminated")]

    if not request_rows:
        return {
            "completed_experiments": 0,
            "total_prompt_tokens": 0,
            "total_output_tokens": 0,
            "total_decode_tokens": 0,
            "total_elapsed_ms": 0.0,
            "total_decode_ms": 0.0,
            "mean_elapsed_ms": 0.0,
            "mean_ttft_ms": 0.0,
            "mean_tpot_ms": 0.0,
            "mean_decode_tps": 0.0,
            "aggregate_decode_tps": 0.0,
            "request_sum_decode_tps": 0.0,
            "completed_batches": 0,
            "contaminated_requests": contaminated_requests,
            "contaminated_batches": contaminated_batches,
        }

    count = len(request_rows)

    def total(name: str) -> float:
        return sum(float(row.get(name, 0.0) or 0.0) for row in request_rows)

    total_decode_ms = total("decode_ms")
    total_decode_tokens = int(total("decode_tokens"))
    batch_decode_tokens = sum(int(row.get("decode_tokens", 0) or 0) for row in batch_rows)
    batch_decode_ms = sum(float(row.get("decode_ms", 0.0) or 0.0) for row in batch_rows)
    return {
        "completed_experiments": count,
        "total_prompt_tokens": int(total("prompt_tokens")),
        "total_output_tokens": int(total("output_tokens")),
        "total_decode_tokens": total_decode_tokens,
        "total_elapsed_ms": total("elapsed_ms"),
        "total_decode_ms": total_decode_ms,
        "mean_elapsed_ms": total("elapsed_ms") / count,
        "mean_ttft_ms": total("ttft_ms") / count,
        "mean_tpot_ms": total("tpot_ms") / count,
        "mean_decode_tps": total("decode_tps") / count,
        # This is the throughput over all decode time, weighted by tokens.
        # This remains the sum of per-request decode durations and is useful
        # for comparing individual streams, but is not the concurrent rate.
        "request_sum_decode_tps": (
            total_decode_tokens / (total_decode_ms / 1000.0)
            if total_decode_ms > 0
            else 0.0
        ),
        "completed_batches": len(batch_rows),
        # Concurrent throughput uses batch makespan, so each token is counted
        # once and overlapping request times are not added together.
        "aggregate_decode_tps": (
            batch_decode_tokens / (batch_decode_ms / 1000.0)
            if batch_decode_ms > 0
            else 0.0
        ) if batch_rows else (
            total_decode_tokens / (total_decode_ms / 1000.0)
            if total_decode_ms > 0
            else 0.0
        ),
        "contaminated_requests": contaminated_requests,
        "contaminated_batches": contaminated_batches,
    }


def _write_results(
    request_rows: list[dict[str, Any]],
    rank: dict[str, int | bool],
    output_dir: Path,
    status: str,
    batch_rows: list[dict[str, Any]] | None = None,
    jit_summary: dict[str, Any] | None = None,
) -> None:
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
        "request_concurrency": REQUEST_CONCURRENCY,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "rank": rank,
        "status": status,
        "completed_experiments": len(request_rows),
    }
    if jit_summary is not None:
        metadata["jit"] = jit_summary
    summary_payload = {"metadata": metadata, "results": request_rows}
    (output_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    # report.json is updated after every completed repeat and exposes a
    # weighted decode throughput for the experiment so far.
    report_payload = {
        "metadata": metadata,
        "aggregate": _report_aggregate(request_rows, batch_rows),
        "batches": batch_rows or [],
        "results": request_rows,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if request_rows:
        with (output_dir / "requests.csv").open("w", newline="", encoding="utf-8") as stream:
            fieldnames = list(request_rows[0])
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(request_rows)
    print(
        f"Report updated: {output_dir} "
        f"({len(request_rows)} completed experiments, status={status})",
        flush=True,
    )


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
    from vllm.sampling_params import RequestOutputKind

    # Installed before the engine is built so no compilation after warmup can
    # go unnoticed, whichever backend reports it.
    jit_counter = _JitCompileCounter()
    jit_summary: dict[str, Any] = {
        "retry_limit": JIT_RETRY_LIMIT,
        "signals": install_jit_guard(jit_counter, announce=bool(rank["is_leader"])),
        "startup_events": 0,
        "warmup_events": 0,
        "retried_batches": [],
        "contaminated_batches": [],
    }

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
    # Kernel builds during engine init, profiling and graph capture are
    # expected and happen before any measurement; only later ones matter.
    startup_events, _ = jit_counter.take()
    jit_summary["startup_events"] = startup_events
    if dist.is_initialized():
        dist.barrier()

    requests = load_requests()
    if bool(rank["is_leader"]):
        print(f"Loaded {len(requests)} benchmark requests")
        if len(requests) < REQUEST_CONCURRENCY:
            print(
                f"Warning: only {len(requests)} requests loaded; "
                f"effective concurrency is {len(requests)}, not {REQUEST_CONCURRENCY}",
                flush=True,
            )

    output_dir = _create_results_dir(rank)
    if dist.is_initialized():
        dist.barrier()

    for warmup_index in range(NUM_WARMUPS):
        for batch_start in range(0, len(requests), REQUEST_CONCURRENCY):
            request_batch = requests[batch_start : batch_start + REQUEST_CONCURRENCY]
            batch = [
                (
                    f"warmup-{warmup_index}-{batch_start + offset}",
                    request["prompt"],
                    _sampling_params(SamplingParams, RequestOutputKind),
                )
                for offset, request in enumerate(request_batch)
            ]
            if bool(rank["is_leader"]):
                print(
                    f"Warmup {warmup_index + 1}/{NUM_WARMUPS}: "
                    f"batch_size={len(batch)}",
                    flush=True,
                )
            _run_batch(llm, batch, torch)

    # Warmup exists to compile the shapes the measured batches use, so builds
    # here are expected; they are counted to show what the guard had to absorb.
    warmup_events = count_all_rank_jit_events(torch, dist, jit_counter.take()[0])
    jit_summary["warmup_events"] = warmup_events
    if bool(rank["is_leader"]):
        print(
            f"Warmup complete; {warmup_events} kernel JIT compilation(s) seen "
            "across all ranks. Measured batches that still compile a kernel "
            f"are re-run up to {JIT_RETRY_LIMIT} time(s).",
            flush=True,
        )

    rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    for batch_start in range(0, len(requests), REQUEST_CONCURRENCY):
        request_batch = requests[batch_start : batch_start + REQUEST_CONCURRENCY]
        for repeat in range(NUM_REPEATS):
            batch_id = f"batch-{batch_start // REQUEST_CONCURRENCY}-repeat-{repeat + 1}"
            if bool(rank["is_leader"]):
                print(
                    f"Running {batch_id} with {len(request_batch)} concurrent streams "
                    f"({repeat + 1}/{NUM_REPEATS})",
                    flush=True,
                )
            (
                batch_results,
                batch_timing,
                attempts,
                jit_contaminated,
                jit_kernels,
            ) = _run_batch_until_jit_free(
                llm,
                request_batch,
                repeat,
                torch,
                dist,
                jit_counter,
                batch_id,
                announce=bool(rank["is_leader"]),
                make_params=functools.partial(
                    _sampling_params, SamplingParams, RequestOutputKind
                ),
            )
            if not bool(rank["is_leader"]):
                continue
            if attempts:
                jit_summary["retried_batches"].append(batch_id)
            if jit_contaminated:
                jit_summary["contaminated_batches"].append(batch_id)
                print(
                    f"Warning: {batch_id} compiled "
                    f"{', '.join(jit_kernels) or 'a kernel'} during all "
                    f"{attempts} retries; its numbers are reported but excluded "
                    "from the aggregate.",
                    flush=True,
                )
            batch_row = {
                "batch_id": batch_id,
                "batch_size": len(request_batch),
                "decode_tokens": int(batch_timing["batch_decode_tokens"]),
                "decode_ms": float(batch_timing["batch_decode_s"]) * 1000,
                "decode_tps": float(batch_timing["batch_decode_tps"]),
                "elapsed_ms": float(batch_timing["batch_elapsed_s"]) * 1000,
                "jit_retries": attempts,
                "jit_contaminated": jit_contaminated,
                "steps": int(batch_timing["batch_steps"]),
                "step_p50_ms": float(batch_timing["batch_step_p50_ms"]),
                "step_max_ms": float(batch_timing["batch_step_max_ms"]),
                "slow_steps": batch_timing["batch_slow_steps"],
            }
            batch_rows.append(batch_row)
            # A step that takes much longer than the batch median is a stall the
            # engine cannot attribute to the model: a compile, a transport
            # hiccup, or a slow rank. Surface it while the run is still going.
            if float(batch_timing["batch_step_max_ms"]) > 1000:
                slowest = batch_timing["batch_slow_steps"][0]
                print(
                    f"Warning: {batch_id} step {slowest['step']} took "
                    f"{slowest['ms']} ms ({slowest['phase']}); median step "
                    f"{float(batch_timing['batch_step_p50_ms']):.0f} ms over "
                    f"{int(batch_timing['batch_steps'])} steps.",
                    flush=True,
                )
            for request in request_batch:
                request_id = _benchmark_request_id(
                    request["task"], request["sample"], repeat, attempts
                )
                output, timing = batch_results[request_id]
                metrics = _request_metrics(output, timing)
                row = {
                    "task": request["task"],
                    "sample": request["sample"],
                    "repeat": repeat + 1,
                    "batch_id": batch_id,
                    "batch_concurrency": len(request_batch),
                    "batch_decode_tps": batch_row["decode_tps"],
                    "jit_retries": attempts,
                    "jit_contaminated": jit_contaminated,
                    **metrics,
                    "gpu_memory_mb": _gpu_memory(torch),
                }
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
            assert output_dir is not None
            _write_results(
                rows,
                rank,
                output_dir,
                status="in_progress",
                batch_rows=batch_rows,
                jit_summary=jit_summary,
            )

    if bool(rank["is_leader"]):
        assert output_dir is not None
        _write_results(
            rows,
            rank,
            output_dir,
            status="completed",
            batch_rows=batch_rows,
            jit_summary=jit_summary,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
