#!/usr/bin/env python3
"""Compare batch-sweep result dirs — one combined throughput/latency table.

Reads report.txt (wall/batch/engine stats) + the per-run CSV (per-request
metrics) from each results dir and prints a single table for finding the
throughput/latency knee point.

Usage:
    python phase2/compare_batch.py <results_dir> [more_dirs...]

    # all batch runs of the last sweep:
    python phase2/compare_batch.py $(ls -d phase0/profile_results/2026081*)
"""

import csv
import os
import re
import sys


def _pct(vals, q):
    if not vals:
        return float("nan")
    s = sorted(vals)
    return s[min(len(s) - 1, int(round(q / 100 * (len(s) - 1))))]


def _num(text):
    return float(text.replace(",", ""))


def load(dirpath: str) -> dict:
    """Extract summary + per-request stats from one results dir."""
    report = open(os.path.join(dirpath, "report.txt")).read()

    m = re.search(r"Batch size\s*:\s*(-?\d+)", report)
    batch = int(m.group(1)) if m else 0  # 0 = sequential run (no batch line)

    m = re.search(r"Wall total\s*:\s*([\d,.]+)\s*ms", report)
    wall_ms = _num(m.group(1)) if m else float("nan")

    m = re.search(r"Engine est\s*:\s*([\d,.]+)\s*ms", report)
    engine_ms = _num(m.group(1)) if m else float("nan")

    csv_path = next(
        os.path.join(dirpath, f) for f in os.listdir(dirpath) if f.endswith(".csv")
    )
    ttfts, lats, tpots, queues = [], [], [], []
    prompt_tok = out_tok = 0
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            ttfts.append(float(row["ttft_ms"]))
            lats.append(float(row["total_ms"]))
            tpots.append(float(row["tpot_ms"]))
            queues.append(float(row["queue_wait_ms"]))
            prompt_tok += int(row["prompt_tokens"])
            out_tok += int(row["output_tokens"])

    n = len(ttfts)
    wall_s = wall_ms / 1000 if wall_ms == wall_ms else float("nan")
    return {
        "batch": batch,
        "wall_s": wall_s,
        "out_tps": out_tok / wall_s if wall_s else float("nan"),
        "prompt_tps": prompt_tok / wall_s if wall_s else float("nan"),
        "conc": engine_ms / wall_ms if wall_ms else float("nan"),
        "ttft_p50": _pct(ttfts, 50),
        "ttft_p90": _pct(ttfts, 90),
        "ttft_p99": _pct(ttfts, 99),
        "req_p50": _pct(lats, 50),
        "req_p99": _pct(lats, 99),
        "tpot_mean": sum(tpots) / n if n else float("nan"),
        "queue_mean": sum(queues) / n if n else float("nan"),
    }


def main():
    dirs = sys.argv[1:]
    if not dirs:
        print(__doc__)
        sys.exit(1)

    rows = [load(d) for d in dirs]
    rows.sort(key=lambda r: (r["batch"] == -1, r["batch"]))  # -1 (all) last

    hdr = (f"{'batch':>6s} | {'wall_s':>8s} | {'out_t/s':>7s} | {'prompt_t/s':>9s} | "
           f"{'conc':>5s} | {'TTFT p50':>8s} {'p90':>8s} {'p99':>8s} | "
           f"{'req p50':>8s} {'p99':>8s} | {'TPOT':>6s} {'queue':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['batch']:>6d} | {r['wall_s']:>8.1f} | {r['out_tps']:>7.1f} | "
              f"{r['prompt_tps']:>9.0f} | {r['conc']:>5.2f} | "
              f"{r['ttft_p50']:>8.0f} {r['ttft_p90']:>8.0f} {r['ttft_p99']:>8.0f} | "
              f"{r['req_p50']:>8.0f} {r['req_p99']:>8.0f} | "
              f"{r['tpot_mean']:>6.1f} {r['queue_mean']:>7.0f}")

    print("\n列说明: conc=有效平均并发; req=逐请求总延迟(TTFT+decode, ms); TPOT/queue 为负载下均值(ms)")
    print("膝盖点判读: 在可接受 TTFT p50/p90 的约束下取最大 out_t/s; 或看 out_t/s 边际收益骤降处。")


if __name__ == "__main__":
    main()
