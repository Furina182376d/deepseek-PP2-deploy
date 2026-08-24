#!/usr/bin/env python3
"""Summarize steady-decode CUDA Graph replay timings by PP stage."""

import argparse
import re
import statistics as stats
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


PATTERN = re.compile(
    r"\[k3t-gpu r(?P<rank>\d+)\] replay_avg=(?P<average>[\d.]+)ms "
    r"n=(?P<count>\d+).*?cum=(?P<cumulative>\d+)"
)


@dataclass(frozen=True)
class ReplayBucket:
    rank: int
    run: int
    cumulative: int
    average_ms: float
    count: int


def percentile(values, fraction):
    return sorted(values)[round((len(values) - 1) * fraction)]


def format_stats(values):
    return (
        f"mean={stats.mean(values):6.2f} "
        f"p50={stats.median(values):6.2f} "
        f"p95={percentile(values, 0.95):6.2f} ms"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Parse [k3t-gpu] replay buckets, remove warmup requests and the "
            "first bucket of each request (which includes prefill), and group "
            "the remaining steady-decode buckets by PP stage."
        )
    )
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--measured-requests", type=int, default=5)
    parser.add_argument("--replays-per-request", type=int, default=256)
    parser.add_argument("--tp-per-stage", type=int, default=8)
    parser.add_argument("--pp-stages", type=int, default=3)
    args = parser.parse_args()

    buckets = []
    observed_ranks = set()
    for log in args.logs:
        for line in log.read_text(errors="replace").splitlines():
            match = PATTERN.search(line)
            if not match:
                continue
            rank = int(match["rank"])
            cumulative = int(match["cumulative"])
            count = int(match["count"])
            observed_ranks.add(rank)

            request = (cumulative - 1) // args.replays_per_request
            run = request - args.warmup_requests
            if not 0 <= run < args.measured_requests:
                continue

            # The request's first bucket contains its prefill replay followed
            # by decode replays. Exclude the entire bucket instead of mixing
            # prefill GPU time into the steady-decode estimate.
            offset = (cumulative - 1) % args.replays_per_request + 1
            if offset <= count:
                continue

            buckets.append(
                ReplayBucket(rank, run, cumulative, float(match["average"]), count)
            )

    expected_ranks = set(range(args.pp_stages * args.tp_per_stage))
    if observed_ranks != expected_ranks:
        missing = sorted(expected_ranks - observed_ranks)
        extra = sorted(observed_ranks - expected_ranks)
        raise SystemExit(f"rank mismatch: missing={missing}, extra={extra}")
    if not buckets:
        raise SystemExit("no measured steady-decode replay buckets found")

    print(
        f"measured requests={args.measured_requests}, "
        f"warmups removed={args.warmup_requests}, "
        f"replays/request={args.replays_per_request}"
    )
    print(
        "local: all per-rank 16-replay bucket averages; critical: maximum "
        "rank average for buckets present on every TP rank"
    )

    for stage in range(args.pp_stages):
        start = stage * args.tp_per_stage
        stage_ranks = set(range(start, start + args.tp_per_stage))
        stage_buckets = [bucket for bucket in buckets if bucket.rank in stage_ranks]
        local_values = [bucket.average_ms for bucket in stage_buckets]

        by_run_and_cumulative = defaultdict(dict)
        for bucket in stage_buckets:
            by_run_and_cumulative[(bucket.run, bucket.cumulative)][bucket.rank] = (
                bucket.average_ms
            )

        critical_by_run = defaultdict(list)
        incomplete = 0
        for (run, _), rank_values in by_run_and_cumulative.items():
            if set(rank_values) != stage_ranks:
                incomplete += 1
                continue
            critical_by_run[run].append(max(rank_values.values()))
        critical_values = [
            value for run_values in critical_by_run.values() for value in run_values
        ]

        print(f"stage {stage} (ranks {start}-{start + args.tp_per_stage - 1}):")
        print(f"  local    {format_stats(local_values)} buckets={len(local_values)}")
        print(
            f"  critical {format_stats(critical_values)} "
            f"complete_buckets={len(critical_values)} incomplete={incomplete}"
        )
        run_means = []
        for run in range(args.measured_requests):
            values = critical_by_run[run]
            if not values:
                continue
            run_mean = stats.mean(values)
            run_means.append(run_mean)
            print(f"    run {run + 1}: mean={run_mean:6.2f} ms buckets={len(values)}")
        print(f"  run means {format_stats(run_means)} n={len(run_means)}")


if __name__ == "__main__":
    main()
