#!/usr/bin/env python3
"""Summarize PyTorch Chrome traces without loading event payloads into memory twice."""

import argparse
import collections
import gzip
import json
from pathlib import Path
from typing import List, Tuple


def scan(path: Path) -> Tuple[int, collections.Counter, collections.Counter, List[Tuple[float, str, str]]]:
    with gzip.open(path, "rt") as stream:
        trace = json.load(stream)
    events = trace.get("traceEvents", [])
    cats = collections.Counter(str(event.get("cat", "")) for event in events)
    phases = collections.Counter(str(event.get("ph", "")) for event in events)
    timed = []
    for event in events:
        duration = event.get("dur")
        if isinstance(duration, (int, float)) and duration > 0:
            timed.append((float(duration), str(event.get("name", "")), str(event.get("cat", ""))))
    timed.sort(reverse=True)
    return len(events), cats, phases, timed[:10]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    files = sorted(args.root.rglob("*.pt.trace.json.gz"))
    print(f"FILES {len(files)}")
    for path in files:
        try:
            count, cats, phases, longest = scan(path)
        except Exception as exc:  # Keep a bad artifact from hiding other traces.
            print(f"ERROR {path} {type(exc).__name__}: {exc}")
            continue
        gpu_events = sum(value for key, value in cats.items() if any(token in key.lower() for token in ("cuda", "gpu", "kernel")))
        print(f"TRACE {path.name} EVENTS {count} GPU_CAT_EVENTS {gpu_events} CATS {cats} PHASES {phases}")
        for duration, name, category in longest:
            print(f"  LONGEST dur_us={duration:.3f} cat={category} name={name}")


if __name__ == "__main__":
    main()
