"""
GPU memory monitoring utilities.
"""

import torch


def gpu_mem(tp: int):
    """Return (used_mb_per_gpu, total_mb_per_gpu)."""
    used, total = [], []
    for i in range(tp):
        free, tot = torch.cuda.mem_get_info(i)
        used.append(round((tot - free) / 1024**2, 1))
        total.append(round(tot / 1024**2, 1))
    return used, total