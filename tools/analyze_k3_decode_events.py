#!/usr/bin/env python3
import re
import statistics as stats
from pathlib import Path

PATTERN = re.compile(
    r"Worker_PP(?P<stage>\d+)_TP(?P<tp>\d+).*?step=(?P<step>\d+) "
    r"tok=(?P<tokens>\d+) step=(?P<step_ms>[\d.]+)ms.*?"
    r"KDA: n=\d+ attn=(?P<kda_attn>[\d.]+)ms "
    r"mlp=(?P<kda_mlp>[\d.]+)ms other=(?P<kda_other>[\d.-]+)ms.*?"
    r"full: n=\d+ attn=(?P<full_attn>[\d.]+)ms "
    r"mlp=(?P<full_mlp>[\d.]+)ms other=(?P<full_other>[\d.-]+)ms.*?"
    r"ALL: attn=(?P<attn>[\d.]+)ms mlp=(?P<mlp>[\d.]+)ms "
    r"model=(?P<model>[\d.]+)ms"
)


def percentile(values, fraction):
    return sorted(values)[round((len(values) - 1) * fraction)]


def main():
    root = Path("process/k3_decode_timing_raw")
    for stage in range(3):
        rows = []
        for line in (root / f"stage{stage}.log").read_text(errors="replace").splitlines():
            match = PATTERN.search(line)
            if not match or match["tokens"] != "1" or match["step"] == "5":
                continue
            rows.append(match.groupdict())

        print(f"stage {stage}: {len(rows)} samples, {len(set(r['step'] for r in rows))} steps")
        for name in ("model", "attn", "mlp", "kda_attn", "kda_mlp",
                     "kda_other", "full_attn", "full_mlp", "full_other"):
            values = [float(row[name]) for row in rows]
            print(
                f"  {name:11s} mean={stats.mean(values):7.2f} "
                f"p50={stats.median(values):7.2f} p95={percentile(values, .95):7.2f} ms"
            )


if __name__ == "__main__":
    main()
