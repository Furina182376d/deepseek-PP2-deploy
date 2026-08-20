"""
Core TP inference — loads the model once per TP setting and profiles
across prompts from any source (synthetic or real datasets).

Public API:
    run_tp(tp)              — synthetic-prompt benchmark (backward compat)
    run_prompt_benchmark()  — generic prompt-driven benchmark
    _load_model(tp)         — build & return a vLLM LLM instance
    _print_gpu_mem(tp)      — print per-GPU memory usage
    _timed_generate()       — single-shot timed generation + metric extraction
"""

import gc
import math
import os
import sys
import time

import torch
from vllm import LLM, SamplingParams

# ---- project-root path (for cross-phase imports) ----
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

import phase0.config as config
from phase0.gpu_utils import gpu_mem
from phase0.prompt_utils import make_prompt
from phase0.results_utils import (
    ALL_RESULTS,
    close_csv,
    ensure_results_dir,
    open_csv,
    write_csv_row,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(
    tp: int, *, enable_expert_parallel: bool = False, max_model_len: int | None = None
) -> LLM:
    """Load the model with TP parallelism and return the LLM instance.

    ``enable_expert_parallel=True`` switches MoE expert layers to expert
    parallelism (vLLM 0.26 derives EP size = world_size // TP
    automatically, e.g. TP=2 on 4 GPUs → EP=2).
    """
    max_model_len = max_model_len or config.MAX_MODEL_LEN
    print(f"\n{'='*60}\nLoading model TP={tp}"
          + (" EP=on" if enable_expert_parallel else "")
          + f", max_model_len={max_model_len}\n{'='*60}")

    llm = LLM(
        model=config.MODEL_PATH,
        tensor_parallel_size=tp,
        max_model_len=max_model_len,
        gpu_memory_utilization=config.GPU_MEM_UTIL,
        trust_remote_code=True,
        kv_cache_dtype=config.KV_CACHE_DTYPE,
        enforce_eager=False,
        disable_log_stats=False,
        enable_expert_parallel=enable_expert_parallel,
    )
    print("Model loaded.\n")
    return llm


def _print_gpu_mem(tp: int):
    """Print per-GPU memory usage after model load."""
    used_init, total_init = gpu_mem(tp)
    for i in range(tp):
        print(f"  GPU {i}: {used_init[i]:.0f} MB / {total_init[i]:.0f} MB "
              f"({used_init[i]/total_init[i]*100:.1f}%)")


def _timed_generate(llm: LLM, prompt, sp: SamplingParams,
                    tokenize_ms: float = 0.0) -> dict:
    """Single generate call with wall-clock + vLLM per-request metrics.

    ``prompt`` may be a str (tokenized inside llm.generate, included in
    wall) or a pre-tokenized list[int] (passed as prompt_token_ids, so
    tokenization stays OUTSIDE the timed path — matching the detailed
    timing benchmark's phase ⑤).

    Returns a dict with keys:
        prompt_tok, out_tok, total_ms, ttft_ms, tpot_ms,
        decode_tps, prefill_ms, prefill_tps,
        queue_wait_ms, overhead_ms, tokenize_ms
    """
    prompt_inputs = (
        [{"prompt_token_ids": prompt}] if isinstance(prompt, list) else [prompt]
    )
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = llm.generate(prompt_inputs, sp)
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
        first_decode_ms = tpot_ms
        prefill_ms = max(0.0, ttft_ms - first_decode_ms)
        prefill_tps = prompt_tok / (prefill_ms / 1000) if prefill_ms > 0 else 0.0
        queue_wait_ms = (
            (m.scheduled_ts - m.queued_ts) * 1000
            if m.queued_ts and m.scheduled_ts
            else 0.0
        )
    else:
        ttft_ms = 0.0
        tpot_ms = 0.0
        decode_tps = 0.0
        prefill_ms = 0.0
        prefill_tps = 0.0
        queue_wait_ms = 0.0

    total_ms = (t1 - t0) * 1000
    # Python-side overhead = wall − (TTFT + remaining decode steps).
    # With pre-tokenized ids this is the ~constant engine handoff cost;
    # with text input it also swallows frontend tokenization.
    decode_ms = max(0, out_tok - 1) * tpot_ms
    overhead_ms = max(0.0, total_ms - ttft_ms - decode_ms)

    return {
        "prompt_tok": prompt_tok,
        "out_tok": out_tok,
        "total_ms": total_ms,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "decode_tps": decode_tps,
        "prefill_ms": prefill_ms,
        "prefill_tps": prefill_tps,
        "queue_wait_ms": queue_wait_ms,
        "overhead_ms": overhead_ms,
        "tokenize_ms": tokenize_ms,
    }


# ---------------------------------------------------------------------------
# Generic prompt benchmark
# ---------------------------------------------------------------------------

def run_prompt_benchmark(
    llm: LLM,
    prompts,  # Iterable[dict] — each dict has "label" and "prompt"; optional extra keys
    tp: int,
    *,
    tag: str | None = None,
    warmup_per_sample: bool = True,
    output_len: int | None = None,
    extra_fields: list[str] | None = None,
) -> list[dict]:
    """Run timed generation over an arbitrary prompt list.

    Parameters
    ----------
    llm:
        Pre-loaded vLLM ``LLM`` instance.
    prompts:
        Iterable of dicts. Each dict must have ``"label"`` (str, shown in
        terminal output) and either ``"prompt"`` (str, the input text) or
        ``"prompt_token_ids"`` (list[int], pre-tokenized — tokenization
        stays outside the timed path; optional ``"tokenize_ms"`` records
        how long that pre-tokenization took).  Extra keys matching
        ``extra_fields`` are forwarded to the CSV row.
    tp:
        Tensor-parallelism count (used for GPU-memory columns).
    tag:
        CSV filename prefix.  Defaults to ``f"tp{tp}"`` (backward compat).
    warmup_per_sample:
        If True, run a cheap ``max_tokens=1`` generation before each timed
        call to absorb CUDA-graph capture / JIT costs.
    output_len:
        Number of tokens to generate per request.  Defaults to
        ``config.OUTPUT_LEN``.
    extra_fields:
        Extra CSV column names to include in the header and row dicts.

    Returns
    -------
    List of result dicts (also appended to global ``ALL_RESULTS``).
    """
    output_len = output_len if output_len is not None else config.OUTPUT_LEN

    ensure_results_dir()
    open_csv(tp=tp, tag=tag, extra_fields=extra_fields)

    header = (
        f"\n{'label':>20s} {'ctx':>6s} {'out':>4s} | "
        f"{'TTFT_ms':>8s} | {'prefill_t/s':>10s} | {'decode_t/s':>10s} | "
        f"{'total_ms':>8s} | {'tok_ms':>7s} | {'ovh_ms':>7s} | {'GPU_avg_MB':>10s}"
    )
    print(header)
    print("-" * 110)

    results: list[dict] = []

    for p in prompts:
        prompt = p.get("prompt_token_ids") or p["prompt"]
        label = p["label"]
        tokenize_ms = p.get("tokenize_ms", 0.0)

        sp = SamplingParams(
            temperature=0,
            max_tokens=output_len,
            ignore_eos=True,
        )

        if warmup_per_sample:
            warmup_inputs = (
                [{"prompt_token_ids": prompt}]
                if isinstance(prompt, list)
                else [prompt]
            )
            llm.generate(warmup_inputs, SamplingParams(temperature=0, max_tokens=1, ignore_eos=True))
            torch.cuda.synchronize()

        r = _timed_generate(llm, prompt, sp, tokenize_ms=tokenize_ms)

        used, _ = gpu_mem(tp)
        avg_mem = sum(used) / len(used)

        row = {
            "tp": tag or f"tp{tp}",
            "context_length": r["prompt_tok"],
            "prompt_tokens": r["prompt_tok"],
            "output_tokens": r["out_tok"],
            "ttft_ms": round(r["ttft_ms"], 2),
            "prefill_ms": round(r["prefill_ms"], 2),
            "prefill_tps": round(r["prefill_tps"], 1),
            "decode_tps": round(r["decode_tps"], 1),
            "tpot_ms": round(r["tpot_ms"], 2),
            "total_ms": round(r["total_ms"], 1),
            "tokenize_ms": round(r["tokenize_ms"], 2),
            "overhead_ms": round(r["overhead_ms"], 2),
            "queue_wait_ms": round(r["queue_wait_ms"], 2),
            "avg_gpu_mem_mb": round(avg_mem, 1),
        }
        for f in (extra_fields or []):
            row[f] = p.get(f, "")

        for i in range(tp):
            row[f"gpu{i}_mem_mb"] = used[i]

        results.append(row)
        write_csv_row(row)
        ALL_RESULTS.append(row)

        print(
            f"{label:>20s} {r['prompt_tok']:>6d} {r['out_tok']:>4d} | "
            f"{r['ttft_ms']:>8.1f} | {r['prefill_tps']:>10.0f} | "
            f"{r['decode_tps']:>10.1f} | {r['total_ms']:>8.0f} | "
            f"{r['tokenize_ms']:>7.1f} | {r['overhead_ms']:>7.1f} | {avg_mem:>10.0f}"
        )

    close_csv()
    return results


# ---------------------------------------------------------------------------
# Batch benchmark
# ---------------------------------------------------------------------------

def run_batch_benchmark(
    llm: LLM,
    prompts,
    tp: int,
    *,
    tag: str,
    batch_size: int,
    output_len: int | None = None,
    extra_fields: list[str] | None = None,
):
    """Chunked batch benchmark — ``batch_size`` prompts per generate call
    (``batch_size=-1`` submits everything in one call).

    The engine continuous-batches requests within each chunk, so decode
    runs compute-bound instead of comm-bound at batch=1.  Per-request
    metrics come from each RequestOutput; TTFT now INCLUDES scheduler
    queue wait — real per-request latency under load.

    Returns ``(rows, stats)`` — rows are per-request dicts (also appended
    to ALL_RESULTS and written to CSV; ``total_ms`` here is per-request
    latency, not wall), stats holds chunk-level wall totals and aggregate
    token counts for the summary.
    """
    output_len = output_len if output_len is not None else config.OUTPUT_LEN

    if not prompts:
        raise SystemExit("ERROR: 0 valid prompts (workload data missing or all "
                         "filtered by max_model_len) — aborting before warmup")

    ensure_results_dir()
    open_csv(tp=tp, tag=tag, extra_fields=extra_fields)

    sp = SamplingParams(temperature=0, max_tokens=output_len, ignore_eos=True)

    # One batched warmup pass (max_tokens=1) absorbs CUDA-graph capture for
    # the batch shapes before timing.  warmup_per_sample would serialize the
    # engine and defeat the point of batching.
    print(f"\nBatch warmup: {len(prompts)} prompts at once, max_tokens=1 ...")
    warm_inputs = [p.get("prompt_token_ids") or p["prompt"] for p in prompts]
    llm.generate(warm_inputs, SamplingParams(temperature=0, max_tokens=1, ignore_eos=True))
    torch.cuda.synchronize()
    print("Warmup done.")

    rows: list[dict] = []
    stats = {
        "tag": tag,
        "batch_size": batch_size,
        "n_samples": len(prompts),
        "wall_ms": 0.0,
        "prompt_tokens": 0,
        "output_tokens": 0,
        "tokenize_ms": sum(p.get("tokenize_ms", 0.0) for p in prompts),
        "engine_ms": 0.0,   # Σ (last_token − scheduled) per request
        "queue_ms": 0.0,
    }

    n_chunks = 1 if batch_size == -1 else math.ceil(len(prompts) / batch_size)
    step = len(prompts) if batch_size == -1 else batch_size
    for ci in range(0, len(prompts), step):
        chunk = prompts[ci:ci + step]
        chunk_inputs = [p.get("prompt_token_ids") or p["prompt"] for p in chunk]

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = llm.generate(chunk_inputs, sp)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        wall_ms = (t1 - t0) * 1000
        stats["wall_ms"] += wall_ms

        used, _ = gpu_mem(tp)
        avg_mem = sum(used) / len(used)

        for p, out in zip(chunk, outputs):
            prompt_tok = len(out.prompt_token_ids)
            out_tok = len(out.outputs[0].token_ids)
            stats["prompt_tokens"] += prompt_tok
            stats["output_tokens"] += out_tok

            m = out.metrics
            if m and m.first_token_latency is not None:
                ttft_ms = m.first_token_latency * 1000
                if m.num_generation_tokens > 1 and m.last_token_ts and m.first_token_ts:
                    tpot_ms = (
                        (m.last_token_ts - m.first_token_ts)
                        / (m.num_generation_tokens - 1) * 1000
                    )
                else:
                    tpot_ms = 0.0
                queue_wait_ms = (
                    (m.scheduled_ts - m.queued_ts) * 1000
                    if m.queued_ts and m.scheduled_ts
                    else 0.0
                )
                engine_ms = (
                    (m.last_token_ts - m.scheduled_ts) * 1000
                    if m.last_token_ts and m.scheduled_ts
                    else 0.0
                )
                # Per-request latency under load (TTFT includes queue wait).
                req_total_ms = ttft_ms + max(0, out_tok - 1) * tpot_ms
            else:
                ttft_ms = tpot_ms = queue_wait_ms = engine_ms = req_total_ms = 0.0

            stats["engine_ms"] += engine_ms
            stats["queue_ms"] += queue_wait_ms

            row = {
                "tp": tag,
                "context_length": prompt_tok,
                "prompt_tokens": prompt_tok,
                "output_tokens": out_tok,
                "ttft_ms": round(ttft_ms, 2),
                "prefill_ms": round(max(0.0, ttft_ms - tpot_ms), 2),
                "prefill_tps": (
                    round(prompt_tok / ((ttft_ms - tpot_ms) / 1000), 1)
                    if ttft_ms > tpot_ms else 0.0
                ),
                "decode_tps": round(1000.0 / tpot_ms, 1) if tpot_ms > 0 else 0.0,
                "tpot_ms": round(tpot_ms, 2),
                "total_ms": round(req_total_ms, 1),  # per-request latency, not wall
                "tokenize_ms": round(p.get("tokenize_ms", 0.0), 2),
                "overhead_ms": 0.0,  # no per-request wall in batch mode
                "queue_wait_ms": round(queue_wait_ms, 2),
                "avg_gpu_mem_mb": round(avg_mem, 1),
            }
            for f in (extra_fields or []):
                row[f] = p.get(f, "")
            for i in range(tp):
                row[f"gpu{i}_mem_mb"] = used[i]

            rows.append(row)
            write_csv_row(row)
            ALL_RESULTS.append(row)

        chunk_out_tok = sum(len(o.outputs[0].token_ids) for o in outputs)
        print(f"  chunk {ci // step + 1}/{n_chunks}: "
              f"{len(chunk):>3d} req  wall={wall_ms:>8.0f}ms  out={chunk_out_tok:>5d} tok")

    close_csv()
    return rows, stats


# ---------------------------------------------------------------------------
# Synthetic benchmark (backward-compatible entry point)
# ---------------------------------------------------------------------------

def run_tp(tp: int):
    """Synthetic-prompt TP benchmark — original phase0 API, unchanged."""
    llm = _load_model(tp)
    _print_gpu_mem(tp)

    # Global warmup
    llm.generate(
        [make_prompt(config.CONTEXT_LENGTHS[0])],
        SamplingParams(temperature=0, max_tokens=config.OUTPUT_LEN, ignore_eos=True),
    )

    # Build synthetic prompt list
    prompts = [
        {"label": str(ctx_len), "prompt": make_prompt(ctx_len)}
        for ctx_len in config.CONTEXT_LENGTHS
    ]

    run_prompt_benchmark(llm, prompts, tp, output_len=config.OUTPUT_LEN)

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)
