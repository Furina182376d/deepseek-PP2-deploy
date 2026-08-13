#!/usr/bin/env python3
"""
Detailed per-phase timing breakdown for PP=1 TP=4 inference.

Measures every segment from prompt arrival to output completion, using
vLLM's internal RequestStateStats timestamps.

Usage:
    cd /home/tjy/codebases/deepseek_deploy
    python phase2/profile_detailed_timing.py
"""

import gc
import json
import os
import sys
import time

import torch
from vllm import SamplingParams

# ---- project-root path ----
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

import phase0.config as config
from phase0.gpu_utils import gpu_mem
from phase0.prompt_utils import make_prompt
from phase0.run_tp import _load_model, _print_gpu_mem
from phase0.results_utils import TIMESTAMP
from phase2.data_loader import list_available_datasets, load_longbench

# ---- Knobs ----
OUTPUT_LEN = 256
MAX_SAMPLES = 20

# Context lengths to sample (covering short → max)
CONTEXT_LENGTHS = [
    1024, 2048, 4096, 8192, 12288, 16384, 24576, 32768, 49152, 65536,
]


def _encode_prompt(tokenizer, text: str):
    """Tokenize once on the Python side; returns (token_ids, encode_ms).

    Kept OUTSIDE the timed llm.generate path so phase ④ reflects engine
    handoff only, not frontend tokenization.
    """
    t0 = time.perf_counter()
    ids = tokenizer.encode(text)
    return ids, (time.perf_counter() - t0) * 1000


def detailed_timing(llm, prompt_token_ids, sp: SamplingParams,
                    tokenize_ms: float = 0.0) -> dict:
    """Single generate + full phase breakdown from vLLM internal timestamps.

    Prompt is passed as pre-tokenized ids — tokenization is measured
    separately (⑤) and excluded from the timed path.

    Returns a dict with every measurable phase in milliseconds.
    """
    torch.cuda.synchronize()
    wall_t0 = time.perf_counter()
    outputs = llm.generate([{"prompt_token_ids": prompt_token_ids}], sp)
    torch.cuda.synchronize()
    wall_t1 = time.perf_counter()

    out = outputs[0]
    prompt_tok = len(out.prompt_token_ids)
    out_tok = len(out.outputs[0].token_ids)
    wall_ms = (wall_t1 - wall_t0) * 1000

    m = out.metrics
    if m is None or m.first_token_latency is None:
        return {
            "prompt_tok": prompt_tok,
            "out_tok": out_tok,
            "wall_ms": wall_ms,
            "tokenize_ms": round(tokenize_ms, 2),
            "error": "no vLLM metrics",
        }

    # ---- vLLM internal timestamps (all in absolute seconds) ----
    ttft_ms = m.first_token_latency * 1000

    # TPOT from decode phase only (first_token is in TTFT)
    if m.num_generation_tokens > 1 and m.last_token_ts and m.first_token_ts:
        tpot_ms = (
            (m.last_token_ts - m.first_token_ts)
            / (m.num_generation_tokens - 1)
            * 1000
        )
        decode_ms = (m.last_token_ts - m.first_token_ts) * 1000
    else:
        tpot_ms = 0.0
        decode_ms = 0.0

    # ---- Phase breakdown from internal timestamps ----
    # NOTE: arrival_time is wall-clock (time.time()), while queued_ts /
    # scheduled_ts / first_token_ts / last_token_ts are engine-core monotonic
    # (time.monotonic()).  We only subtract within the same clock domain.

    # Queued → scheduled (both monotonic) — actual wait in scheduler queue
    queue_wait_ms = (
        (m.scheduled_ts - m.queued_ts) * 1000
        if m.queued_ts and m.scheduled_ts
        else 0.0
    )

    # Scheduled → first token (both monotonic) — model prefill + first decode step
    prefill_and_fd_ms = (
        (m.first_token_ts - m.scheduled_ts) * 1000
        if m.scheduled_ts and m.first_token_ts
        else 0.0
    )

    # Estimate pure prefill = (prefill+first_decode) - one decode step
    prefill_ms = max(0.0, prefill_and_fd_ms - tpot_ms)

    # Decode: first_token → last_token (both monotonic)
    decode_from_ts_ms = (
        (m.last_token_ts - m.first_token_ts) * 1000
        if m.first_token_ts and m.last_token_ts
        else 0.0
    )

    # Engine total from monotonic timestamps: scheduled → last_token
    engine_monotonic_ms = (
        (m.last_token_ts - m.scheduled_ts) * 1000
        if m.scheduled_ts and m.last_token_ts
        else 0.0
    )

    # Wall-clock overhead = wall_ms - (prefill + decode)
    # (excludes pre-queue bookkeeping, but avoids clock-domain mixing)
    model_time_ms = prefill_and_fd_ms + decode_from_ts_ms
    overhead_ms = max(0.0, wall_ms - model_time_ms)

    # Throughputs
    prefill_tps = prompt_tok / (prefill_ms / 1000) if prefill_ms > 0 else 0.0
    decode_tps = 1.0 / (tpot_ms / 1000) if tpot_ms > 0 else 0.0

    return {
        "prompt_tok": prompt_tok,
        "out_tok": out_tok,
        "wall_ms": round(wall_ms, 2),
        "ttft_ms": round(ttft_ms, 2),
        # --- Phase breakdown ---
        "queue_wait_ms": round(queue_wait_ms, 3),
        "prefill_and_fd_ms": round(prefill_and_fd_ms, 2),
        "prefill_ms": round(prefill_ms, 2),
        "decode_ms": round(decode_from_ts_ms, 2),
        "tpot_ms": round(tpot_ms, 2),
        "engine_monotonic_ms": round(engine_monotonic_ms, 2),
        "overhead_ms": round(overhead_ms, 2),
        "tokenize_ms": round(tokenize_ms, 2),
        # --- Throughputs ---
        "prefill_tps": round(prefill_tps, 1),
        "decode_tps": round(decode_tps, 1),
    }


def print_phase_breakdown(r: dict, label: str = ""):
    """Pretty-print a single sample's phase breakdown."""
    if "error" in r:
        print(f"  [{label}] ERROR: {r['error']}")
        return

    wall = r["wall_ms"]
    p_tok = r["prompt_tok"]
    o_tok = r["out_tok"]

    def pct(v_ms):
        return (v_ms / wall * 100) if wall > 0 else 0.0

    print(f"\n  {'─' * 65}")
    print(f"  {label}  |  prompt={p_tok} tok  output={o_tok} tok  wall={wall:.0f}ms")
    print(f"  {'─' * 65}")
    print(f"  {'Phase':<40s} {'ms':>8s}  {'%wall':>7s}  note")
    print(f"  {'─' * 65}")

    rows = [
        ("① Queue wait (scheduler)",     r["queue_wait_ms"],
         "queued→scheduled (both monotonic)"),
        ("② Prefill (prompt → 1st token)", r["prefill_ms"],
         f"scheduled→first_token minus 1 decode step ({r['prefill_tps']:.0f} tok/s)"),
        ("③ Decode (token generation)",   r["decode_ms"],
         f"first→last token, {o_tok} tokens @ {r['tpot_ms']:.2f}ms TPOT ({r['decode_tps']:.0f} tok/s)"),
        ("   ── Model execution total ──", r["engine_monotonic_ms"],
         "scheduled→last_token (② includes prefill+1st_decode)"),
        ("④ Python / overhead",            r["overhead_ms"],
         "wall_ms − (prefill_and_fd + decode)"),
        ("⑤ Tokenize (untimed)",           r.get("tokenize_ms", 0.0),
         "Python-side encode, excluded from wall"),
    ]

    for name, val, note in rows:
        is_subtotal = name.startswith("   ──")
        marker = "├─" if not is_subtotal else "╞═"
        print(f"  {marker} {name:<37s} {val:>8.1f}  {pct(val):>6.1f}%  {note}")

    # TTFT check
    ttft_check = r["prefill_ms"] + r["tpot_ms"]
    print(f"  {'─' * 65}")
    print(f"  TTFT = prefill({r['prefill_ms']:.1f}) + TPOT({r['tpot_ms']:.2f}) = "
          f"{ttft_check:.1f}ms  (vLLM reports {r['ttft_ms']:.1f}ms)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tp = 4
    tag = f"tp{tp}_detailed"

    # 1. Load model
    llm = _load_model(tp)
    _print_gpu_mem(tp)

    # 2. Global warmup
    llm.generate(
        ["Hello, this is a warmup request."],
        SamplingParams(temperature=0, max_tokens=1, ignore_eos=True),
    )
    torch.cuda.synchronize()

    tokenizer = llm.get_tokenizer()

    # 3. Synthetic prompts at controlled lengths
    print("\n" + "=" * 70)
    print("  PHASE 1: Synthetic prompts (controlled context lengths)")
    print("=" * 70)

    synthetic_results = []
    for ctx_len in CONTEXT_LENGTHS:
        prompt = make_prompt(ctx_len)
        label = f"synth_{ctx_len}"

        # Pre-tokenize once (outside timed path); reuse ids for warmup + timing
        prompt_ids, encode_ms = _encode_prompt(tokenizer, prompt)

        # Per-sample warmup
        llm.generate(
            [{"prompt_token_ids": prompt_ids}],
            SamplingParams(temperature=0, max_tokens=1, ignore_eos=True),
        )
        torch.cuda.synchronize()

        sp = SamplingParams(
            temperature=0,
            max_tokens=OUTPUT_LEN,
            ignore_eos=True,
        )
        r = detailed_timing(llm, prompt_ids, sp, tokenize_ms=encode_ms)
        r["context_length"] = ctx_len
        r["label"] = label
        synthetic_results.append(r)
        print_phase_breakdown(r, label)

    # 4. Real LongBench prompts (sample across datasets)
    print("\n\n" + "=" * 70)
    print("  PHASE 2: Real LongBench samples")
    print("=" * 70)

    available = list_available_datasets()
    tasks = [name for name, status in available.items() if status == "local"]
    max_prompt_len = config.MAX_MODEL_LEN - OUTPUT_LEN

    real_results = []
    for task_name in sorted(tasks):
        items = list(load_longbench(task_name))[:3]  # 3 per task
        for idx, item in enumerate(items):
            context = item.get("context", "") or item.get("input", "")
            question = item.get("input", "") or item.get("question", "")
            prompt = f"{context}\n\n{question}"

            prompt_ids, encode_ms = _encode_prompt(tokenizer, prompt)
            if len(prompt_ids) > max_prompt_len:
                continue

            label = f"{task_name}[{idx}]"

            # Per-sample warmup
            llm.generate(
                [{"prompt_token_ids": prompt_ids}],
                SamplingParams(temperature=0, max_tokens=1, ignore_eos=True),
            )
            torch.cuda.synchronize()

            sp = SamplingParams(
                temperature=0,
                max_tokens=OUTPUT_LEN,
                ignore_eos=True,
            )
            r = detailed_timing(llm, prompt_ids, sp, tokenize_ms=encode_ms)
            r["context_length"] = r["prompt_tok"]
            r["label"] = label
            real_results.append(r)
            print_phase_breakdown(r, label)

    # 5. Summary report
    print("\n\n" + "=" * 70)
    print("  SUMMARY: Phase time vs context length (synthetic prompts)")
    print("=" * 70)
    print(f"  {'ctx':>7s} | {'prefill':>9s} {'':>10s} | {'decode':>8s} | {'queue':>8s} | {'tokenize':>9s} | {'overhead':>8s} | {'wall':>8s} | {'prefill':>9s}")
    print(f"  {'':>7s} | {'ms':>9s} {'tok/s':>10s} | {'ms':>8s} | {'ms':>8s} | {'ms':>9s} | {'ms':>8s} | {'ms':>8s} | {'%':>9s}")
    print(f"  {'─' * 94}")

    for r in synthetic_results:
        wall = r["wall_ms"]
        print(f"  {r['context_length']:>7d} | "
              f"{r['prefill_ms']:>9.1f} {r['prefill_tps']:>10.0f} | "
              f"{r['decode_ms']:>8.1f} | "
              f"{r['queue_wait_ms']:>8.2f} | "
              f"{r['tokenize_ms']:>9.2f} | "
              f"{r['overhead_ms']:>8.2f} | "
              f"{wall:>8.0f} | "
              f"{r['prefill_ms']/wall*100:>8.1f}%")

    # 6. Write summary JSON + report.txt
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    # Custom results dir: profile_results/detailed_timing/<TIMESTAMP>/
    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "phase0", "profile_results", "detailed_timing", TIMESTAMP,
    )
    out_dir = os.path.normpath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    metadata = {
        "model_path": config.MODEL_PATH,
        "timestamp": TIMESTAMP,
        "max_model_len": config.MAX_MODEL_LEN,
        "gpu_memory_utilization": config.GPU_MEM_UTIL,
        "kv_cache_dtype": config.KV_CACHE_DTYPE,
        "output_len": OUTPUT_LEN,
        "tp": tp,
        "pp": 1,
    }
    all_results = {
        "synthetic": synthetic_results,
        "real_longbench": real_results,
    }
    json_path = os.path.join(out_dir, "detailed_timing.json")
    with open(json_path, "w") as f:
        json.dump({"metadata": metadata, "results": all_results}, f, indent=2)

    # ---- Build report.txt ----
    lines = []
    lines.append("DeepSeek V4 Flash — Detailed Per-Phase Timing Report")
    lines.append(f"Timestamp : {TIMESTAMP}")
    lines.append(f"Model     : {config.MODEL_PATH}")
    lines.append(f"Config    : PP=1  TP={tp}  |  Max Len: {config.MAX_MODEL_LEN}")
    lines.append(f"Output Len: {OUTPUT_LEN} tokens (ignore_eos=True)")
    lines.append("")
    lines.append("=" * 90)
    lines.append("PHASE LEGEND")
    lines.append("=" * 90)
    lines.append("  ① Queue wait   : queued_ts → scheduled_ts (scheduler slot wait; near-zero offline)")
    lines.append("  ② Prefill      : scheduled_ts → first_token_ts, minus 1 decode step")
    lines.append("                   (process all prompt tokens → produce first token)")
    lines.append("  ③ Decode       : first_token_ts → last_token_ts")
    lines.append("                   (autoregressive generation, one step per token)")
    lines.append("  ④ Overhead     : Python wall-clock − (prefill+first_decode + decode)")
    lines.append("  ⑤ Tokenize     : Python-side encode, done BEFORE the timed section")
    lines.append("                   (prompt passed as token ids; excluded from wall)")
    lines.append("")
    lines.append("  NOTE: ②③ timestamps are engine-core monotonic; ④ bridges to Python wall-clock.")
    lines.append("")

    def format_breakdown(r, indent=""):
        """Format one sample's phase breakdown as report lines."""
        if "error" in r:
            return [f"{indent}ERROR: {r['error']}"]
        wall = r["wall_ms"]
        out = []
        out.append(f"{indent}prompt={r['prompt_tok']:>6d} tok  output={r['out_tok']:>4d} tok  "
                   f"wall={wall:>7.1f}ms  (TTFT={r['ttft_ms']:>7.1f}ms)")
        out.append(f"{indent}  ① Queue wait : {r['queue_wait_ms']:>8.2f} ms  ({r['queue_wait_ms']/max(wall,1e-9)*100:>5.1f}%)")
        out.append(f"{indent}  ② Prefill    : {r['prefill_ms']:>8.2f} ms  ({r['prefill_ms']/max(wall,1e-9)*100:>5.1f}%)  "
                   f"{r['prefill_tps']:>8.0f} tok/s")
        out.append(f"{indent}  ③ Decode     : {r['decode_ms']:>8.2f} ms  ({r['decode_ms']/max(wall,1e-9)*100:>5.1f}%)  "
                   f"{r['decode_tps']:>6.1f} tok/s  (TPOT={r['tpot_ms']:>5.2f}ms)")
        out.append(f"{indent}  ④ Overhead   : {r['overhead_ms']:>8.2f} ms  ({r['overhead_ms']/max(wall,1e-9)*100:>5.1f}%)")
        out.append(f"{indent}  ⑤ Tokenize   : {r.get('tokenize_ms', 0.0):>8.2f} ms  (pre-timed, excluded from wall)")
        return out

    lines.append("=" * 90)
    lines.append("SYNTHETIC PROMPTS — controlled context lengths")
    lines.append("=" * 90)
    for r in synthetic_results:
        lines.append(f"  ctx={r['context_length']:>6d}  ({r['label']})")
        lines.extend(format_breakdown(r, indent="    "))
        lines.append("")

    lines.append("=" * 90)
    lines.append("SYNTHETIC SUMMARY TABLE (phase ms / % of wall)")
    lines.append("=" * 90)
    lines.append(f"{'ctx':>7s} | {'wall':>7s} | {'prefill':>8s} {'%':>6s} | "
                 f"{'decode':>8s} {'%':>6s} | {'queue':>7s} | {'tokenize':>9s} | {'overhead':>8s} | "
                 f"{'prefill_t/s':>11s} | {'decode_t/s':>10s}")
    lines.append("-" * 120)
    for r in synthetic_results:
        wall = r["wall_ms"]
        lines.append(
            f"{r['context_length']:>7d} | {wall:>7.0f} | "
            f"{r['prefill_ms']:>8.1f} {r['prefill_ms']/max(wall,1e-9)*100:>5.1f}% | "
            f"{r['decode_ms']:>8.1f} {r['decode_ms']/max(wall,1e-9)*100:>5.1f}% | "
            f"{r['queue_wait_ms']:>7.2f} | {r['tokenize_ms']:>9.2f} | {r['overhead_ms']:>8.2f} | "
            f"{r['prefill_tps']:>11.0f} | {r['decode_tps']:>10.1f}"
        )

    lines.append("")
    lines.append("=" * 90)
    lines.append("REAL LONGBENCH SAMPLES")
    lines.append("=" * 90)
    for r in real_results:
        lines.append(f"  {r['label']}  (ctx={r['context_length']})")
        lines.extend(format_breakdown(r, indent="    "))
        lines.append("")

    lines.append("")
    lines.append("=" * 90)
    lines.append("FILES IN THIS RUN")
    lines.append(f"  detailed_timing.json — 完整逐阶段数据 (metadata + results)")
    lines.append(f"  report.txt          — 本文件")

    report_path = os.path.join(out_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    # Print phase legend + paths
    print(f"\n\nPhase breakdown legend:")
    print(f"  ① Queue wait      : time spent waiting for scheduler slot (near-zero for offline API)")
    print(f"  ② Prefill         : process all prompt tokens → produce first token")
    print(f"  ③ Decode          : autoregressive token generation ({OUTPUT_LEN} tokens)")
    print(f"  ④ Overhead        : Python-side (detokenization, post-processing, clock skew)")
    print(f"  ⑤ Tokenize        : Python-side encode before timed section (prompt passed as token ids)")
    print(f"\n  NOTE: All internal timestamps are engine-core monotonic. wall_ms is Python wall-clock.")
    print(f"  vLLM-reported TTFT (first_token_latency) is wall-clock, used for cross-check.")
    print(f"\nResults saved to: {out_dir}/")
    print(f"  detailed_timing.json  report.txt")
