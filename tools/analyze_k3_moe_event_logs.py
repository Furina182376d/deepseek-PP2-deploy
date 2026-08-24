"""Summarize K3_MOE_EVENT_V1 lines by PP stage and decode token."""

import argparse
import collections
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()
    totals = collections.defaultdict(lambda: [0, 0.0])
    for path in args.logs:
        for line in path.read_text(errors="ignore").splitlines():
            marker = "K3_MOE_EVENT_V1 "
            if marker not in line:
                continue
            try:
                payload = json.loads(line.split(marker, 1)[1])
            except json.JSONDecodeError:
                continue
            rank = int(payload["rank"])
            stage = rank // 8
            for row in payload["rows"]:
                if row["tokens"] != 1:
                    continue
                count, total = totals[(stage, row["label"])]
                totals[(stage, row["label"])] = [
                    count + row["count"],
                    total + row["total_ms"],
                ]
    labels = [
        "router_topk",
        "prepare_dispatch",
        "marlin_experts",
        "shared_expert",
        "finalize_combine",
        "tp_reduce_final",
    ]
    for stage in range(3):
        values = []
        for label in labels:
            count, total = totals[(stage, label)]
            mean = total / count if count else 0.0
            values.append((label, mean))
        total = sum(value for _, value in values)
        print(f"stage={stage} total_ms={total:.6f}")
        for label, value in values:
            share = 100.0 * value / total if total else 0.0
            print(f"  {label:20s} {value:.6f} ms  {share:5.1f}%")


if __name__ == "__main__":
    main()
