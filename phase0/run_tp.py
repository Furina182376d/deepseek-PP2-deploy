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

def _load_model(tp: int) -> LLM:
    """Load the model with TP parallelism and return the LLM instance."""
    print(f"\n{'='*60}\nLoading model TP={tp}, max_model_len={config.MAX_MODEL_LEN}\n{'='*60}")

    llm = LLM(
        model=config.MODEL_PATH,
        tensor_parallel_size=tp,
        max_model_len=config.MAX_MODEL_LEN,
        gpu_memory_utilization=config.GPU_MEM_UTIL,
        trust_remote_code=True,
        kv_cache_dtype=config.KV_CACHE_DTYPE,
        enforce_eager=False,
        disable_log_stats=False,
    )
    print("Model loaded.\n")
    return llm


def _print_gpu_mem(tp: int):
    """Print per-GPU memory usage after model load."""
    used_init, total_init = gpu_mem(tp)
    for i in range(tp):
        print(f"  GPU {i}: {used_init[i]:.0f} MB / {total_init[i]:.0f} MB "
              f"({used_init[i]/total_init[i]*100:.1f}%)")


def _timed_generate(llm: LLM, prompt: str, sp: SamplingParams) -> dict:
    """Single generate call with wall-clock + vLLM per-request metrics.

    Returns a dict with keys:
        prompt_tok, out_tok, total_ms, ttft_ms, tpot_ms,
        decode_tps, prefill_ms, prefill_tps
    """
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
        first_decode_ms = tpot_ms
        prefill_ms = max(0.0, ttft_ms - first_decode_ms)
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
        terminal output) and ``"prompt"`` (str, the input text).  Extra keys
        matching ``extra_fields`` are forwarded to the CSV row.
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
        f"{'total_ms':>8s} | {'GPU_avg_MB':>10s}"
    )
    print(header)
    print("-" * 95)

    results: list[dict] = []

    for p in prompts:
        prompt_text = p["prompt"]
        label = p["label"]

        sp = SamplingParams(
            temperature=0,
            max_tokens=output_len,
            ignore_eos=True,
        )

        if warmup_per_sample:
            llm.generate([prompt_text], SamplingParams(temperature=0, max_tokens=1, ignore_eos=True))
            torch.cuda.synchronize()

        r = _timed_generate(llm, prompt_text, sp)

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
            f"{r['decode_tps']:>10.1f} | {r['total_ms']:>8.0f} | {avg_mem:>10.0f}"
        )

    close_csv()
    return results


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
