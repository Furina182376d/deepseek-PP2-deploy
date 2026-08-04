"""
Results logging — CSV, JSON, and human-readable report.
"""

import csv
import json
import os
from datetime import datetime

from config import KV_CACHE_DTYPE, GPU_MEM_UTIL, MAX_MODEL_LEN, MODEL_PATH, OUTPUT_LEN

# --- Module-level state ---
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profile_results", TIMESTAMP)
ALL_RESULTS: list[dict] = []  # accumulated for JSON dump at the end

_csv_writer = None
_csv_file = None


def ensure_results_dir():
    """Create the timestamped results directory."""
    os.makedirs(RESULTS_DIR, exist_ok=True)


def csv_path(tp: int) -> str:
    return os.path.join(RESULTS_DIR, f"tp{tp}.csv")


def csv_headers(tp: int) -> list[str]:
    cols = [
        "tp", "context_length", "prompt_tokens", "output_tokens",
        "ttft_ms", "prefill_ms", "prefill_tps", "decode_tps", "tpot_ms", "total_ms",
        "avg_gpu_mem_mb",
    ]
    cols += [f"gpu{i}_mem_mb" for i in range(tp)]
    return cols


def open_csv(tp: int):
    """Open the per-TP CSV file and write its header."""
    global _csv_file, _csv_writer
    if _csv_file:
        _csv_file.close()
    _csv_file = open(csv_path(tp), "w", newline="")
    _csv_writer = csv.DictWriter(_csv_file, fieldnames=csv_headers(tp))
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


def write_report_and_summary():
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
    print("SUMMARY — TP=4 vs TP=8 对比")
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
        gpu_str = "  ".join(f"GPU{i}: {r[f'gpu{i}_mem_mb']:.0f}MB" for i in range(tp))
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
    lines.append("TP=4 vs TP=8 COMPARISON")
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
    lines.append(f"  tp4.csv          — TP=4 详细结果")
    lines.append(f"  tp8.csv          — TP=8 详细结果")
    lines.append(f"  full_results.json — 全部结果 (含元数据)")
    lines.append(f"  report.txt       — 本文件")

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nResults saved to: {RESULTS_DIR}/")
    print(f"  tp4.csv  tp8.csv  full_results.json  report.txt")