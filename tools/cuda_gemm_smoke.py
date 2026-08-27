"""Plain CUDA GEMM smoke test for external Nsight collection."""

import torch


def main():
    x = torch.randn((4096, 4096), device="cuda", dtype=torch.float16)
    y = torch.randn((4096, 4096), device="cuda", dtype=torch.float16)
    for _ in range(2):
        _ = x @ y
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(3):
        _ = x @ y
    end.record()
    end.synchronize()
    print("gemm_ms=", start.elapsed_time(end), flush=True)


if __name__ == "__main__":
    main()
