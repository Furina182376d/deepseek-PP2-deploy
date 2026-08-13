"""
Results logging — CSV, JSON, and human-readable report.
"""

import csv
import json
import os
import sys
from datetime import datetime

# ---- project-root path (for cross-phase imports) ----
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

from phase0.config import KV_CACHE_DTYPE, GPU_MEM_UTIL, MAX_MODEL_LEN, MODEL_PATH, OUTPUT_LEN

# --- Module-level state ---
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profile_results", TIMESTAMP)
ALL_RESULTS: list[dict] = []  # accumulated for JSON dump at the end

_csv_writer = None
_csv_file = None


def ensure_results_dir():
    """Create the timestamped results directory."""
    os.makedirs(RESULTS_DIR, exist_ok=True)


def csv_path(tp: int, tag: str | None = None) -> str:
    name = tag if tag else f"tp{tp}"
    return os.path.join(RESULTS_DIR, f"{name}.csv")


def csv_headers(tp: int, extra_fields: list[str] | None = None) -> list[str]:
    cols = [
        "tp", "context_length", "prompt_tokens", "output_tokens",
        "ttft_ms", "prefill_ms", "prefill_tps", "decode_tps", "tpot_ms", "total_ms",
        "tokenize_ms", "overhead_ms", "queue_wait_ms",
        "avg_gpu_mem_mb",
    ]
    if extra_fields:
        cols += extra_fields
    cols += [f"gpu{i}_mem_mb" for i in range(tp)]
    return cols


def open_csv(tp: int, tag: str | None = None, extra_fields: list[str] | None = None):
    """Open the per-TP CSV file and write its header."""
    global _csv_file, _csv_writer
    if _csv_file:
        _csv_file.close()
    _csv_file = open(csv_path(tp, tag), "w", newline="")
    _csv_writer = csv.DictWriter(_csv_file, fieldnames=csv_headers(tp, extra_fields))
    _csv_writer.writeheader()
    _csv_file.flush()


def write_csv_row(row: dict):
    """Append a row to the currently open CSV and flush."""
    global _csv_writer, _csv_file
    if _csv_writer:
        _csv_writer.writerow(row)
        _csv_file.flush()


def close_csv():
    global _csv_file, _csv_writer
    if _csv_file:
        _csv_file.close()
        _csv_file = None
        _csv_writer = None


def write_report_and_summary(title: str = "TP=4 vs TP=8 对比"):
    """Print a terminal summary AND write a human-readable report.txt."""
    if not ALL_RESULTS:
        return

    ctxs = sorted(set(r["context_length"] for r in ALL_RESULTS))
    tps = sorted(set(r["tp"] for r in ALL_RESULTS))

    # Group by (tp, ctx)
    by_key = {}
    for r in ALL_RESULTS:
        by_key[(r["tp"], r["context_length"])] = r

    # ----- Build terminal summary -----
    sep = "=" * 100
    print(f"\n\n{sep}")
    print(f"SUMMARY — {title}")
    print(sep)

    print(f"\n{'ctx':>6s} | " + " | ".join(f"{'—— TP={t} ——':>41s}" for t in tps))
    print(f"{'':>6s} | " + " | ".join(f"{'ttft_ms prefill_t/s decode_t/s mem_mb':>41s}" for _ in tps))
    print("-" * (8 + 45 * len(tps)))

    for ctx in ctxs:
        parts = []
        for tp in tps:
            r = by_key.get((tp, ctx))
            if r:
                parts.append(f"{r['ttft_ms']:>8.1f} {r['prefill_tps']:>10.0f} {r['decode_tps']:>10.1f} {r['avg_gpu_mem_mb']:>8.0f}")
            else:
                parts.append(f"{'N/A':>41s}")
        print(f"{ctx:>6d} | " + " | ".join(parts))

    # ----- Write report.txt -----
    report_path = os.path.join(RESULTS_DIR, "report.txt")
    lines = []
    lines.append("DeepSeek V4 Flash — vLLM Profile Report")
    lines.append(f"Timestamp : {TIMESTAMP}")
    lines.append(f"Model     : {MODEL_PATH}")
    lines.append(f"Max Len   : {MAX_MODEL_LEN}  |  GPU util: {GPU_MEM_UTIL}  |  KV dtype: {KV_CACHE_DTYPE}")
    lines.append(f"Output Len: {OUTPUT_LEN} tokens (ignore_eos=True)")
    lines.append("")
    lines.append("=" * 110)
    lines.append("PER-RUN DETAILS")
    lines.append("=" * 110)

    for r in ALL_RESULTS:
        tp = r["tp"]
        ctx = r["context_length"]
        # Derive TP count from row keys to handle both phase0 (int tp) and
        # phase1 (string tag like "PP2_TP8") without a TypeError on range(tp).
        gpu_keys = sorted(
            [k for k in r if k.startswith("gpu") and k.endswith("_mem_mb")],
            key=lambda k: int(k.replace("gpu", "").replace("_mem_mb", "")),
        )
        gpu_str = "  ".join(f"{k.replace('_mem_mb', '')}: {r[k]:.0f}MB" for k in gpu_keys)
        lines.append(
            f"TP={tp}  ctx={ctx:>5d}  |  "
            f"prompt={r['prompt_tokens']:>5d}tok  output={r['output_tokens']:>3d}tok  |  "
            f"TTFT={r['ttft_ms']:>7.1f}ms  "
            f"prefill={r['prefill_ms']:>7.1f}ms ({r['prefill_tps']:>8.0f} tok/s)  "
            f"decode={r['decode_tps']:>8.1f} tok/s (TPOT={r['tpot_ms']:>6.2f}ms)  "
            f"total={r['total_ms']:>8.0f}ms  |  "
            f"avg_mem={r['avg_gpu_mem_mb']:>7.0f}MB  [{gpu_str}]"
        )

    lines.append("")
    lines.append("=" * 110)
    lines.append(title.upper())
    lines.append("=" * 110)
    header = f"{'ctx':>6s} | " + " | ".join(f"{'—— TP={t} ——':>50s}" for t in tps)
    sub_hdr = f"{'':>6s} | " + " | ".join(f"{'ttft_ms':>8s} {'prefill_t/s':>10s} {'decode_t/s':>10s} {'TPOT_ms':>8s} {'avg_mem_mb':>10s}" for _ in tps)
    lines.append(header)
    lines.append(sub_hdr)
    lines.append("-" * (8 + 54 * len(tps)))

    for ctx in ctxs:
        parts = []
        for tp in tps:
            r = by_key.get((tp, ctx))
            if r:
                parts.append(f"{r['ttft_ms']:>8.1f} {r['prefill_tps']:>10.0f} {r['decode_tps']:>10.1f} {r['tpot_ms']:>8.2f} {r['avg_gpu_mem_mb']:>10.0f}")
            else:
                parts.append(f"{'N/A':>50s}")
        lines.append(f"{ctx:>6d} | " + " | ".join(parts))

    lines.append("")
    lines.append("=" * 110)
    lines.append("FILES IN THIS RUN")
    for tp_val in tps:
        lines.append(f"  tp{tp_val}.csv      — {tp_val} 详细结果")
    lines.append(f"  full_results.json — 全部结果 (含元数据)")
    lines.append(f"  report.txt       — 本文件")

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    csv_files = "  ".join(f"tp{tp}.csv" for tp in tps)
    print(f"\nResults saved to: {RESULTS_DIR}/")
    print(f"  {csv_files}  full_results.json  report.txt")


def write_aggregate_summary(title: str = "WORKLOAD AGGREGATES"):
    """Aggregate per-phase totals over ALL_RESULTS — for long workloads
    where the per-ctx grid of write_report_and_summary is meaningless.

    Prints a console block and appends a section to report.txt.  Call
    AFTER write_report_and_summary so the section lands at the end.
    """
    if not ALL_RESULTS:
        return

    n = len(ALL_RESULTS)
    total_prompt_tok = sum(r["prompt_tokens"] for r in ALL_RESULTS)
    total_out_tok = sum(r["output_tokens"] for r in ALL_RESULTS)
    wall_ms = sum(r["total_ms"] for r in ALL_RESULTS)
    # Engine-core estimate: TTFT + remaining decode steps per sample.
    engine_ms = sum(
        r["ttft_ms"] + max(0, r["output_tokens"] - 1) * r["tpot_ms"]
        for r in ALL_RESULTS
    )
    tokenize_ms = sum(r.get("tokenize_ms", 0.0) for r in ALL_RESULTS)
    overhead_ms = sum(r.get("overhead_ms", 0.0) for r in ALL_RESULTS)
    queue_ms = sum(r.get("queue_wait_ms", 0.0) for r in ALL_RESULTS)

    mean_ttft = sum(r["ttft_ms"] for r in ALL_RESULTS) / n
    mean_tpot = sum(r["tpot_ms"] for r in ALL_RESULTS) / n
    mean_prefill_tps = sum(r["prefill_tps"] for r in ALL_RESULTS) / n
    mean_decode_tps = sum(r["decode_tps"] for r in ALL_RESULTS) / n

    def pct(v_ms):
        return v_ms / wall_ms * 100 if wall_ms > 0 else 0.0

    lines = []
    lines.append("=" * 110)
    lines.append(title.upper())
    lines.append("=" * 110)
    lines.append(f"  Samples         : {n}")
    lines.append(f"  Prompt tokens   : {total_prompt_tok:,}")
    lines.append(f"  Output tokens   : {total_out_tok:,}")
    lines.append("-" * 110)
    lines.append(f"  Wall total      : {wall_ms:>12,.0f} ms  ({pct(wall_ms):5.1f}%)")
    lines.append(f"  Engine est      : {engine_ms:>12,.0f} ms  ({pct(engine_ms):5.1f}%)  TTFT + decode")
    lines.append(f"  Tokenize        : {tokenize_ms:>12,.0f} ms  ({pct(tokenize_ms):5.1f}%)  pre-timed; wall saved vs text input")
    lines.append(f"  Overhead        : {overhead_ms:>12,.0f} ms  ({pct(overhead_ms):5.1f}%)  wall − engine")
    lines.append(f"  Queue wait      : {queue_ms:>12,.0f} ms  ({pct(queue_ms):5.1f}%)")
    lines.append("-" * 110)
    lines.append(f"  Mean TTFT       : {mean_ttft:>12.1f} ms")
    lines.append(f"  Mean TPOT       : {mean_tpot:>12.2f} ms")
    lines.append(f"  Mean prefill    : {mean_prefill_tps:>12,.0f} tok/s")
    lines.append(f"  Mean decode     : {mean_decode_tps:>12.1f} tok/s")

    block = "\n".join(lines)
    print(f"\n\n{block}\n")

    report_path = os.path.join(RESULTS_DIR, "report.txt")
    with open(report_path, "a") as f:
        f.write("\n\n" + block + "\n")