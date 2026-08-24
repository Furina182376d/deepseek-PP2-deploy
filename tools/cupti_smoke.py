"""Minimal CUPTI/Kineto CUDA activity smoke test."""

import json
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile


output_dir = Path("/home/tjy/cupti_smoke")
output_dir.mkdir(parents=True, exist_ok=True)
trace_path = output_dir / "trace.json"

print(
    f"torch={torch.__version__} cuda={torch.version.cuda} "
    f"device={torch.cuda.get_device_name(0)}"
)
x = torch.randn((4096, 4096), device="cuda", dtype=torch.float16)
y = torch.randn((4096, 4096), device="cuda", dtype=torch.float16)

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(3):
        _ = x @ y
        torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
prof.export_chrome_trace(str(trace_path))

with trace_path.open() as trace_file:
    events = json.load(trace_file).get("traceEvents", [])

print(
    f"events={len(events)} "
    f"kernel_cat={sum(event.get('cat') == 'kernel' for event in events)} "
    f"cuda_text={sum('cuda' in str(event).lower() for event in events)}"
)
