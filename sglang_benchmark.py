#!/usr/bin/env python3
"""Distributed SGLang server launcher and streaming benchmark client.

Edit the CONFIGURATION block, then run launch_sglang_benchmark.sh on every
node. Rank 0 starts the OpenAI-compatible SGLang endpoint and runs the
benchmark once it is ready. Other ranks participate in the distributed server.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Barrier, Lock
from typing import Any


# =============================================================================
# CONFIGURATION: this is the only file to edit for a new SGLang experiment.
# =============================================================================
# Both pipeline nodes must expose the same local directory. This is the DS
# model currently present under /data/models on 192.168.0.224 and .225.
MODEL_PATH = "/data/models/DeepSeek-V4-Pro-DSpark"

# One node hosts one pipeline stage. PP=2 and TP=8 use 16 GPUs across 2 nodes.
PP_SIZE = 2
TP_SIZE_PER_STAGE = 8
NNODES = PP_SIZE
MASTER_ADDR = "192.168.0.224"
DIST_PORT = 29501
NETWORK_INTERFACE = "eth0"
SGLANG_HOST = "0.0.0.0"
SGLANG_PORT = 30000

# The launcher activates this environment before it invokes this file.
SGLANG_CONDA_ENV = "sglang"

# SGLang serve options, based on the requested reference configuration.
TRUST_REMOTE_CODE = True
MOE_RUNNER_BACKEND = "flashinfer_mxfp4"
# None disables speculative decoding. Set to "DSPARK" after installing an
# SGLang build whose `sglang serve --help` lists it as a supported algorithm.
SPECULATIVE_ALGORITHM: str | None = None
CHUNKED_PREFILL_SIZE = 8192
DISABLE_FLASHINFER_AUTOTUNE = True
SWA_FULL_TOKENS_RATIO = 0.1
MEM_FRACTION_STATIC = 0.90
SGLANG_EXTRA_SERVE_ARGS: tuple[str, ...] = ()

# ``longbench``, ``classic``, or ``custom``.
BENCHMARK_TYPE = "longbench"
BENCHMARK_DATA_DIR = "/home/tjy/benchmarks/longbench/data"
BENCHMARK_TASKS: tuple[str, ...] = ("qmsum", "gov_report")
MAX_SAMPLES_PER_TASK = 2
CUSTOM_PROMPT_COUNT = 4

OUTPUT_TOKENS = 128
NUM_WARMUPS = 1
NUM_REPEATS = 3
# Set 1 for a single stream. Set 4 to measure four concurrently streamed calls.
REQUEST_CONCURRENCY = 4
REQUEST_TIMEOUT_SECONDS = 3600
RESULTS_DIR = "results"


def build_custom_prompt(index: int) -> str:
    """Return one prompt for the ``custom`` benchmark mode."""
    return f"You are evaluating a language model. Return item {index}."


def validate_config() -> None:
    errors: list[str] = []
    if not MODEL_PATH:
        errors.append("MODEL_PATH must not be empty")
    if PP_SIZE <= 0 or TP_SIZE_PER_STAGE <= 0 or NNODES != PP_SIZE:
        errors.append("PP_SIZE/TP_SIZE_PER_STAGE must be positive and NNODES must equal PP_SIZE")
    if not MASTER_ADDR or not NETWORK_INTERFACE:
        errors.append("MASTER_ADDR and NETWORK_INTERFACE must not be empty")
    if not 1 <= SGLANG_PORT <= 65535 or not 1 <= DIST_PORT <= 65535:
        errors.append("SGLANG_PORT and DIST_PORT must be valid TCP ports")
    if BENCHMARK_TYPE not in {"longbench", "classic", "custom"}:
        errors.append("BENCHMARK_TYPE must be longbench, classic, or custom")
    if BENCHMARK_TYPE != "custom" and not BENCHMARK_DATA_DIR:
        errors.append("BENCHMARK_DATA_DIR is required for file benchmarks")
    if MAX_SAMPLES_PER_TASK < 0 or CUSTOM_PROMPT_COUNT <= 0:
        errors.append("sample counts must be non-negative (custom count must be positive)")
    if OUTPUT_TOKENS <= 0 or NUM_WARMUPS < 0 or NUM_REPEATS <= 0:
        errors.append("OUTPUT_TOKENS/NUM_REPEATS must be positive and NUM_WARMUPS non-negative")
    if REQUEST_CONCURRENCY <= 0:
        errors.append("REQUEST_CONCURRENCY must be positive")
    if not 0 < MEM_FRACTION_STATIC <= 1:
        errors.append("MEM_FRACTION_STATIC must be in (0, 1]")
    if errors:
        raise ValueError("Invalid configuration:\n  " + "\n  ".join(errors))


def _task_paths(data_dir: Path, tasks: tuple[str, ...]) -> list[Path]:
    if tasks:
        paths = [
            data_dir / (task if task.endswith((".json", ".jsonl")) else f"{task}.jsonl")
            for task in tasks
        ]
    else:
        paths = sorted(
            path for path in data_dir.iterdir()
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
    return records[:limit] if limit else records


def _record_prompt(record: dict[str, Any]) -> str:
    if record.get("prompt"):
        return str(record["prompt"])
    if record.get("text"):
        return str(record["text"])
    context = str(record.get("context") or "")
    question = str(record.get("input") or record.get("question") or "")
    return f"{context}\n\n{question}" if context and question else context or question


def load_requests() -> list[dict[str, Any]]:
    if BENCHMARK_TYPE == "custom":
        return [{"task": "custom", "sample": i, "prompt": build_custom_prompt(i)}
                for i in range(CUSTOM_PROMPT_COUNT)]
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


class TokenCounter:
    """Use the local tokenizer only when the server omits OpenAI usage data."""

    def __init__(self) -> None:
        self.tokenizer: Any | None = None
        self.failed = False
        self._lock = Lock()

    def count(self, text: str) -> tuple[int, str]:
        # Multiple streaming client threads can request a fallback count at
        # once when the server does not return OpenAI usage. Load only once.
        with self._lock:
            if not self.failed and self.tokenizer is None:
                try:
                    from transformers import AutoTokenizer
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        MODEL_PATH, trust_remote_code=TRUST_REMOTE_CODE, local_files_only=True
                    )
                except Exception:
                    self.failed = True
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text, add_special_tokens=False)), "local_tokenizer"
        # A streamed event need not be one token. This is explicitly labelled
        # as an estimate rather than silently reporting a false token rate.
        return 0, "unavailable"


def _http_json(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"} if data else {}
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


def _endpoint() -> str:
    return f"http://{MASTER_ADDR}:{SGLANG_PORT}"


def wait_for_server() -> str:
    deadline = time.monotonic() + REQUEST_TIMEOUT_SECONDS
    last_error = "not attempted"
    while time.monotonic() < deadline:
        worker_pid = os.environ.get("SGLANG_SERVER_PID")
        if worker_pid and not _worker_alive(worker_pid):
            raise RuntimeError(
                f"SGLang worker process {worker_pid} exited before the HTTP endpoint became ready; "
                "see the node log for the original error"
            )
        try:
            models = _http_json(f"{_endpoint()}/v1/models")
            data = models.get("data") or []
            if data and data[0].get("id"):
                return str(data[0]["id"])
            last_error = f"unexpected /v1/models response: {models}"
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        time.sleep(2)
    raise RuntimeError(f"SGLang endpoint did not become ready: {last_error}")


def _worker_alive(pid_text: str) -> bool:
    """Return false for a missing or already-reaped worker process."""
    try:
        pid = int(pid_text)
        os.kill(pid, 0)
    except (ValueError, ProcessLookupError, PermissionError):
        return False
    try:
        state = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[2]
    except (FileNotFoundError, OSError, IndexError):
        return False
    return state != "Z"


def _stream_request(
    request: dict[str, Any],
    model_id: str,
    token_counter: TokenCounter,
    start_barrier: Barrier,
) -> dict[str, Any]:
    start_barrier.wait()
    started = time.perf_counter()
    payload = {
        "model": model_id,
        "prompt": request["prompt"],
        "max_tokens": OUTPUT_TOKENS,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    http_request = urllib.request.Request(
        f"{_endpoint()}/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    first_token_at: float | None = None
    last_token_at: float | None = None
    output_parts: list[str] = []
    usage: dict[str, Any] | None = None
    with urllib.request.urlopen(http_request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                text = choice.get("text") or ""
                if text:
                    now = time.perf_counter()
                    first_token_at = first_token_at or now
                    last_token_at = now
                    output_parts.append(str(text))
    finished = time.perf_counter()
    output_text = "".join(output_parts)
    prompt_tokens = int((usage or {}).get("prompt_tokens") or 0)
    output_tokens = int((usage or {}).get("completion_tokens") or 0)
    token_source = "server_usage" if output_tokens else ""
    if not prompt_tokens:
        prompt_tokens, prompt_source = token_counter.count(request["prompt"])
    else:
        prompt_source = "server_usage"
    if not output_tokens:
        output_tokens, token_source = token_counter.count(output_text)
    if output_tokens <= 0:
        raise RuntimeError("SGLang response did not include countable completion tokens")
    first_token_at = first_token_at or finished
    last_token_at = last_token_at or finished
    decode_s = max(0.0, last_token_at - first_token_at)
    decode_tokens = max(output_tokens - 1, 0)
    return {
        **request,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "decode_tokens": decode_tokens,
        "elapsed_ms": (finished - started) * 1000,
        "ttft_ms": (first_token_at - started) * 1000,
        "prefill_ms": (first_token_at - started) * 1000,
        "decode_ms": decode_s * 1000,
        "tpot_ms": decode_s * 1000 / decode_tokens if decode_tokens and decode_s else 0.0,
        "prefill_tps": prompt_tokens / (first_token_at - started) if first_token_at > started else 0.0,
        "decode_tps": decode_tokens / decode_s if decode_s else 0.0,
        "timing_source": "http_stream_wall_clock",
        "prompt_token_source": prompt_source,
        "output_token_source": token_source,
        "_decode_first_at": first_token_at,
        "_decode_last_at": last_token_at,
    }


def _run_batch(
    requests: list[dict[str, Any]], model_id: str, token_counter: TokenCounter
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    barrier = Barrier(len(requests))
    with ThreadPoolExecutor(max_workers=len(requests)) as executor:
        futures = [executor.submit(_stream_request, request, model_id, token_counter, barrier)
                   for request in requests]
        rows = [future.result() for future in as_completed(futures)]
    first = min(float(row.pop("_decode_first_at")) for row in rows)
    last = max(float(row.pop("_decode_last_at")) for row in rows)
    decode_s = max(0.0, last - first)
    decode_tokens = sum(int(row["decode_tokens"]) for row in rows)
    return rows, {
        "batch_size": len(rows),
        "decode_tokens": decode_tokens,
        "decode_ms": decode_s * 1000,
        "decode_tps": decode_tokens / decode_s if decode_s else 0.0,
        "elapsed_ms": max(float(row["elapsed_ms"]) for row in rows),
    }


def _aggregate(rows: list[dict[str, Any]], batches: list[dict[str, Any]]) -> dict[str, float | int]:
    def total(name: str) -> float:
        return sum(float(row.get(name, 0.0) or 0.0) for row in rows)
    total_batch_decode_ms = sum(float(batch["decode_ms"]) for batch in batches)
    total_batch_decode_tokens = sum(int(batch["decode_tokens"]) for batch in batches)
    count = len(rows)
    return {
        "completed_experiments": count,
        "completed_batches": len(batches),
        "total_prompt_tokens": int(total("prompt_tokens")),
        "total_output_tokens": int(total("output_tokens")),
        "total_decode_tokens": int(total("decode_tokens")),
        "mean_ttft_ms": total("ttft_ms") / count if count else 0.0,
        "mean_tpot_ms": total("tpot_ms") / count if count else 0.0,
        "mean_decode_tps": total("decode_tps") / count if count else 0.0,
        "aggregate_decode_tps": (
            total_batch_decode_tokens / (total_batch_decode_ms / 1000.0)
            if total_batch_decode_ms else 0.0
        ),
    }


def _write_report(
    rows: list[dict[str, Any]],
    batches: list[dict[str, Any]],
    status: str,
    error: str | None = None,
) -> Path:
    output_dir = Path(RESULTS_DIR) / f"sglang_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    # Persist the same directory for incremental writes in this process.
    output_dir = Path(os.environ.setdefault("SGLANG_BENCHMARK_OUTPUT_DIR", str(output_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "framework": "sglang",
        "model": MODEL_PATH,
        "pp_size": PP_SIZE,
        "tp_size_per_stage": TP_SIZE_PER_STAGE,
        "nnodes": NNODES,
        "speculative_algorithm": SPECULATIVE_ALGORITHM,
        "request_concurrency": REQUEST_CONCURRENCY,
        "output_tokens": OUTPUT_TOKENS,
        "status": status,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if error:
        metadata["error"] = error
    report = {"metadata": metadata, "aggregate": _aggregate(rows, batches), "batches": batches, "results": rows}
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    (output_dir / "report.json").write_text(encoded, encoding="utf-8")
    (output_dir / "summary.json").write_text(encoded, encoding="utf-8")
    if rows:
        with (output_dir / "requests.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Report updated: {output_dir} ({len(rows)} requests, status={status})", flush=True)
    return output_dir


def run_benchmark() -> int:
    validate_config()
    rows: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    _write_report(rows, batches, "starting")
    try:
        model_id = wait_for_server()
        requests = load_requests()
        print(f"SGLang endpoint ready with model={model_id}; loaded {len(requests)} requests", flush=True)
        if len(requests) < REQUEST_CONCURRENCY:
            print(f"Warning: effective concurrency is {len(requests)}, not {REQUEST_CONCURRENCY}", flush=True)
        token_counter = TokenCounter()
        for warmup in range(NUM_WARMUPS):
            for index in range(0, len(requests), REQUEST_CONCURRENCY):
                batch = requests[index : index + REQUEST_CONCURRENCY]
                print(f"Warmup {warmup + 1}/{NUM_WARMUPS}, batch_size={len(batch)}", flush=True)
                _run_batch(batch, model_id, token_counter)

        for index in range(0, len(requests), REQUEST_CONCURRENCY):
            request_batch = requests[index : index + REQUEST_CONCURRENCY]
            for repeat in range(NUM_REPEATS):
                batch_id = f"batch-{index // REQUEST_CONCURRENCY}-repeat-{repeat + 1}"
                print(f"Running {batch_id}, streams={len(request_batch)}", flush=True)
                batch_rows, batch_metrics = _run_batch(request_batch, model_id, token_counter)
                batch_metrics["batch_id"] = batch_id
                batches.append(batch_metrics)
                for row in batch_rows:
                    row["repeat"] = repeat + 1
                    row["batch_id"] = batch_id
                    row["batch_concurrency"] = len(request_batch)
                    row["batch_decode_tps"] = batch_metrics["decode_tps"]
                    print(json.dumps(row, ensure_ascii=False), flush=True)
                rows.extend(batch_rows)
                _write_report(rows, batches, "in_progress")
    except Exception as exc:
        _write_report(rows, batches, "failed", str(exc))
        raise
    _write_report(rows, batches, "completed")
    return 0


def _serve_help() -> str:
    """Read the installed SGLang CLI before forming a distributed command."""
    try:
        result = subprocess.run(
            ["sglang", "serve", "--help"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("sglang command was not found in the activated environment") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("timed out while reading 'sglang serve --help'") from exc
    if not result.stdout:
        raise RuntimeError(f"could not read SGLang CLI help (exit status {result.returncode})")
    return result.stdout


def _select_serve_flag(help_text: str, setting: str, candidates: tuple[str, ...]) -> str:
    for flag in candidates:
        if re.search(rf"(?<![A-Za-z0-9_-]){re.escape(flag)}(?:[ =,]|$)", help_text):
            return flag
    raise RuntimeError(
        f"Installed SGLang does not expose a CLI flag for {setting}. "
        f"Expected one of: {', '.join(candidates)}. Upgrade SGLang or use a release with multi-node PP support."
    )


def _validate_speculative_algorithm(help_text: str) -> None:
    """Reject a different speculative algorithm rather than silently changing it."""
    if not SPECULATIVE_ALGORITHM:
        return
    match = re.search(r"--speculative-algorithm\s+\{([^}]+)\}", help_text)
    if not match:
        return
    supported = {value.strip().upper() for value in match.group(1).split(",")}
    requested = SPECULATIVE_ALGORITHM.upper()
    if requested not in supported:
        raise RuntimeError(
            f"Installed SGLang does not support speculative algorithm {SPECULATIVE_ALGORITHM!r}. "
            f"Its CLI supports: {', '.join(sorted(supported))}. "
            "Install an SGLang build with DSPARK support; do not substitute a different algorithm "
            "when comparing this experiment."
        )


def _build_serve_args(node_rank: int) -> list[str]:
    validate_config()
    if not 0 <= node_rank < NNODES:
        raise ValueError(f"node rank must be in [0, {NNODES - 1}]")
    help_text = _serve_help()
    tp_flag = _select_serve_flag(help_text, "tensor parallelism", ("--tp", "--tp-size"))
    pp_flag = _select_serve_flag(
        help_text, "pipeline parallelism", ("--pp", "--pp-size", "--pipeline-parallel-size")
    )
    nnodes_flag = _select_serve_flag(help_text, "node count", ("--nnodes", "--num-nodes"))
    node_rank_flag = _select_serve_flag(help_text, "node rank", ("--node-rank",))
    init_addr_flag = _select_serve_flag(
        help_text, "distributed init address", ("--dist-init-addr", "--dist-init-address")
    )
    _validate_speculative_algorithm(help_text)
    args = [
        "sglang", "serve",
        "--model-path", MODEL_PATH,
        tp_flag, str(TP_SIZE_PER_STAGE),
        pp_flag, str(PP_SIZE),
        nnodes_flag, str(NNODES),
        node_rank_flag, str(node_rank),
        init_addr_flag, f"{MASTER_ADDR}:{DIST_PORT}",
        "--host", SGLANG_HOST,
        "--port", str(SGLANG_PORT),
        "--moe-runner-backend", MOE_RUNNER_BACKEND,
        "--chunked-prefill-size", str(CHUNKED_PREFILL_SIZE),
        "--swa-full-tokens-ratio", str(SWA_FULL_TOKENS_RATIO),
        "--mem-fraction-static", str(MEM_FRACTION_STATIC),
    ]
    if SPECULATIVE_ALGORITHM:
        args.extend(("--speculative-algorithm", SPECULATIVE_ALGORITHM))
    if TRUST_REMOTE_CODE:
        args.append("--trust-remote-code")
    if DISABLE_FLASHINFER_AUTOTUNE:
        args.append("--disable-flashinfer-autotune")
    args.extend(SGLANG_EXTRA_SERVE_ARGS)
    return args


def _validate_model_compatibility() -> None:
    """Fail before forking distributed workers when HF cannot parse the model."""
    config_path = Path(MODEL_PATH) / "config.json"
    if not config_path.is_file():
        return
    try:
        from transformers import AutoConfig
        AutoConfig.from_pretrained(
            MODEL_PATH,
            trust_remote_code=TRUST_REMOTE_CODE,
            local_files_only=True,
        )
    except Exception as exc:
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            model_type = config.get("model_type", "unknown")
        except Exception:
            model_type = "unknown"
        raise RuntimeError(
            f"Model {MODEL_PATH} (model_type={model_type!r}) is not loadable by the installed "
            "Transformers/SGLang stack. Install a SGLang build with this architecture, or use "
            "a compatible checkpoint. Original error: " + str(exc)
        ) from exc


def validate_serve(node_rank: int) -> int:
    _validate_model_compatibility()
    args = _build_serve_args(node_rank)
    print("SGLang serve preflight passed: " + " ".join(args), flush=True)
    return 0


def serve(node_rank: int) -> int:
    _validate_model_compatibility()
    args = _build_serve_args(node_rank)
    print("Starting: " + " ".join(args), flush=True)
    os.execvp(args[0], args)


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve_parser = subparsers.add_parser("serve")
    serve_parser.add_argument("node_rank", type=int)
    validate_parser = subparsers.add_parser("validate-serve")
    validate_parser.add_argument("node_rank", type=int)
    subparsers.add_parser("benchmark")
    args = parser.parse_args()
    if args.command == "serve":
        return serve(args.node_rank)
    if args.command == "validate-serve":
        return validate_serve(args.node_rank)
    return run_benchmark()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError, urllib.error.URLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
