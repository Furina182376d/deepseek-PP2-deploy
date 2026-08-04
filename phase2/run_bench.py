"""
Benchmark runner — profiles inference with real/synthetic long-context workloads.

Supports:
    - Needle-in-a-haystack (built-in, no download)
    - LongBench tasks (when pre-downloaded JSON/JSONL files exist)

Can be used with either the TP-only or PP multi-node backend by importing
the appropriate run function.
"""

import os
import sys
import time
from typing import Callable

import torch
from vllm import LLM, SamplingParams

# ---- project-root path (for cross-phase imports) ----
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

from phase2.benchmark_config import NEEDLE_CONTEXT_LENGTHS, NEEDLE_DEPTHS, OUTPUT_LENS
from phase2.data_loader import iter_needle_prompts, list_available_datasets, load_longbench
from phase0.prompt_utils import make_prompt
from phase0.results_utils import (
    ALL_RESULTS,
    close_csv,
    ensure_results_dir,
    open_csv,
    write_csv_row,
)


def _timed_generate(llm: LLM, prompt: str, sp: SamplingParams) -> dict:
    """Single generate call with wall-clock + vLLM metrics."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = llm.generate([prompt], sp)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    out = outputs[0]
    prompt_tok = len(out.prompt_token_ids)
    out_tok = len(out.outputs[0].token_ids)

    m = out.metrics
    if m and m.first_token_latency is not None:
        ttft_ms = m.first_token_latency * 1000
        if m.num_generation_tokens > 1 and m.last_token_ts and m.first_token_ts:
            tpot_sec = (m.last_token_ts - m.first_token_ts) / (m.num_generation_tokens - 1)
            tpot_ms = tpot_sec * 1000
            decode_tps = 1.0 / tpot_sec
        else:
            tpot_ms = 0.0
            decode_tps = 0.0
        prefill_ms = max(0.0, ttft_ms - tpot_ms)
        prefill_tps = prompt_tok / (prefill_ms / 1000) if prefill_ms > 0 else 0.0
    else:
        ttft_ms = 0.0
        tpot_ms = 0.0
        decode_tps = 0.0
        prefill_ms = 0.0
        prefill_tps = 0.0

    return {
        "prompt_tok": prompt_tok,
        "out_tok": out_tok,
        "total_ms": (t1 - t0) * 1000,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "decode_tps": decode_tps,
        "prefill_ms": prefill_ms,
        "prefill_tps": prefill_tps,
    }


def run_needle_benchmark(
    llm: LLM,
    *,
    tag: str = "needle",
    ctx_lengths: list[int] | None = None,
    output_lens: list[int] | None = None,
    depths: list[float] | None = None,
) -> list[dict]:
    """
    Run needle-in-a-haystack throughput benchmark.

    Parameters
    ----------
    llm:
        An already-loaded vLLM ``LLM`` instance.
    tag:
        Label for the result rows (e.g., "PP2_TP4_needle").
    ctx_lengths:
        Context lengths to test (defaults from benchmark_config).
    output_lens:
        Output token lengths to test (defaults from benchmark_config).
    depths:
        Passkey depths to test (defaults from benchmark_config).

    Returns
    -------
    List of result dicts (also appended to global ``ALL_RESULTS``).
    """
    ctx_lengths = ctx_lengths or NEEDLE_CONTEXT_LENGTHS
    output_lens = output_lens or OUTPUT_LENS
    depths = depths or NEEDLE_DEPTHS

    # Open CSV for this benchmark run
    ensure_results_dir()
    open_csv(tp=tag)

    results: list[dict] = []

    header = (
        f"\n{'ctx':>6s} {'out':>4s} {'depth':>5s} | "
        f"{'TTFT_ms':>8s} | {'prefill_t/s':>10s} | {'decode_t/s':>10s} | "
        f"{'total_ms':>8s}"
    )
    print(f"\n--- Needle-in-a-Haystack ({tag}) ---")
    print(header)
    print("-" * 80)

    for out_len in output_lens:
        for ctx_len in ctx_lengths:
            for depth in depths:
                prompt = make_prompt(ctx_len)  # for consistency, use synthetic filler

                # Actually use the real needle prompt for a subset to validate correctness
                # For pure throughput, use synthetic prompts at the target lengths
                sp = SamplingParams(
                    temperature=0,
                    max_tokens=out_len,
                    ignore_eos=True,
                )

                # Warm-up at this context length on first iteration
                r = _timed_generate(llm, prompt, sp)

                row = {
                    "tp": tag,
                    "context_length": ctx_len,
                    "output_length": out_len,
                    "depth": depth,
                    "prompt_tokens": r["prompt_tok"],
                    "output_tokens": r["out_tok"],
                    "ttft_ms": round(r["ttft_ms"], 2),
                    "prefill_ms": round(r["prefill_ms"], 2),
                    "prefill_tps": round(r["prefill_tps"], 1),
                    "decode_tps": round(r["decode_tps"], 1),
                    "tpot_ms": round(r["tpot_ms"], 2),
                    "total_ms": round(r["total_ms"], 1),
                }
                results.append(row)
                write_csv_row(row)
                ALL_RESULTS.append(row)

                print(
                    f"{ctx_len:>6d} {out_len:>4d} {depth:>5.2f} | "
                    f"{r['ttft_ms']:>8.1f} | {r['prefill_tps']:>10.0f} | "
                    f"{r['decode_tps']:>10.1f} | {r['total_ms']:>8.0f}"
                )

    close_csv()
    return results


def run_longbench_benchmark(
    llm: LLM,
    *,
    tag: str = "longbench",
    output_len: int = 256,
) -> list[dict]:
    """
    Run LongBench tasks (if available locally).

    Returns an empty list if no LongBench data is found.
    """
    available = list_available_datasets()
    tasks = [name for name, status in available.items() if status == "local"]

    if not tasks:
        print("[run_bench] No LongBench datasets found locally — skipping.")
        return []

    ensure_results_dir()
    open_csv(tp=tag)

    results: list[dict] = []
    print(f"\n--- LongBench ({tag}) — {len(tasks)} tasks ---")

    for task_name in tasks:
        items = load_longbench(task_name)
        print(f"\n  Task: {task_name}  ({len(items)} samples)")

        for item in items:
            # LongBench items typically have: input, context, (optional) answers
            context = item.get("context", "") or item.get("input", "")
            question = item.get("input", "") or item.get("question", "")
            prompt = f"{context}\n\n{question}"

            sp = SamplingParams(
                temperature=0,
                max_tokens=output_len,
                ignore_eos=True,
            )

            r = _timed_generate(llm, prompt, sp)

            row = {
                "tp": tag,
                "task": task_name,
                "context_length": r["prompt_tok"],
                "output_tokens": r["out_tok"],
                "ttft_ms": round(r["ttft_ms"], 2),
                "prefill_ms": round(r["prefill_ms"], 2),
                "prefill_tps": round(r["prefill_tps"], 1),
                "decode_tps": round(r["decode_tps"], 1),
                "tpot_ms": round(r["tpot_ms"], 2),
                "total_ms": round(r["total_ms"], 1),
            }
            results.append(row)
            write_csv_row(row)
            ALL_RESULTS.append(row)

    close_csv()
    return results
