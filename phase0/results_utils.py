"""
Results logging — CSV, JSON, and human-readable report.
"""

import csv
import json
import os
import statistics
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


def csv_path(tp: int) -> str:
    return os.path.join(RESULTS_DIR, f"tp{tp}.csv")


def csv_headers(tp: int) -> list[str]:
    cols = [
        "tp", "repeat", "context_length", "prompt_tokens", "output_tokens",
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


def write_report_and_summary(title: str = "TP=4 vs TP=8 对比",
                             metadata: dict | None = None):
    """Print a terminal summary AND write a human-readable report.txt.

    ``metadata`` (可选) 覆盖报告头部的模型/运行信息; 缺省时回退到 phase0.config。
    """
    if not ALL_RESULTS:
        return
    metadata = metadata or {}

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
    lines.append(f"{title} — vLLM Profile Report")
    lines.append(f"Timestamp : {metadata.get('timestamp', TIMESTAMP)}")
    lines.append(f"Model     : {metadata.get('model_path', MODEL_PATH)}")
    lines.append(
        f"Max Len   : {metadata.get('max_model_len', MAX_MODEL_LEN)}  |  "
        f"GPU util: {metadata.get('gpu_memory_utilization', GPU_MEM_UTIL)}  |  "
        f"KV dtype: {metadata.get('kv_cache_dtype', KV_CACHE_DTYPE)}"
    )
    lines.append(f"Output Len: {metadata.get('output_len', OUTPUT_LEN)} tokens (ignore_eos=True)")
    lines.append("")
    lines.append("=" * 110)
    lines.append("PER-RUN DETAILS")
    lines.append("=" * 110)

    for r in ALL_RESULTS:
        tp = r["tp"]
        ctx = r["context_length"]
        # r["tp"] 是标签字符串 (如 "PP3_TP8"), 不能用于 range;
        # GPU 数量从行内的 gpu*_mem_mb 列数推导
        n_gpus = sum(1 for k in r if k.startswith("gpu") and k.endswith("_mem_mb"))
        gpu_str = "  ".join(f"GPU{i}: {r[f'gpu{i}_mem_mb']:.0f}MB" for i in range(n_gpus))
        lines.append(
            f"TP={tp}  run={r.get('repeat', 1):>2d}  ctx={ctx:>5d}  |  "
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

    repeated_groups = {}
    for r in ALL_RESULTS:
        repeated_groups.setdefault((r["tp"], r["context_length"]), []).append(r)
    if any(len(rows) > 1 for rows in repeated_groups.values()):
        lines.append("")
        lines.append("=" * 110)
        lines.append("REPEATED TPOT SUMMARY")
        lines.append("=" * 110)
        lines.append(
            f"{'TP':>12s} {'ctx':>6s} {'n':>3s} {'mean_ms':>9s} "
            f"{'p50_ms':>9s} {'p95_ms':>9s} {'min_ms':>9s} {'max_ms':>9s}"
        )
        for (tp, ctx), rows in sorted(repeated_groups.items()):
            values = sorted(float(row["tpot_ms"]) for row in rows)
            p95 = values[round((len(values) - 1) * 0.95)]
            lines.append(
                f"{tp:>12s} {ctx:>6d} {len(values):>3d} "
                f"{statistics.mean(values):>9.2f} "
                f"{statistics.median(values):>9.2f} {p95:>9.2f} "
                f"{min(values):>9.2f} {max(values):>9.2f}"
            )

    def gpu_count(tp_val):
        r0 = by_key.get((tp_val, ctxs[0])) if ctxs else None
        if not r0:
            return 0
        return sum(1 for k in r0 if k.startswith("gpu") and k.endswith("_mem_mb"))

    lines.append("")
    lines.append("=" * 110)
    lines.append("FILES IN THIS RUN")
    for tp_val in tps:
        lines.append(f"  tp{gpu_count(tp_val)}.csv      — {tp_val} 详细结果")
    lines.append(f"  full_results.json — 全部结果 (含元数据)")
    lines.append(f"  report.txt       — 本文件")

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    csv_files = "  ".join(f"tp{gpu_count(t)}.csv" for t in tps)
    print(f"\nResults saved to: {RESULTS_DIR}/")
    print(f"  {csv_files}  full_results.json  report.txt")
