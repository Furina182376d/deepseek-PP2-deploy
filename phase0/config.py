"""
Shared constants for DeepSeek V4 Flash profiling.
"""

MODEL_PATH = "/data/model/DeepSeek-V4-Flash-0731"
CONTEXT_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384, 32768]
OUTPUT_LEN = 512  # decode-only run: longer window for nvidia-smi dmon observation
MAX_MODEL_LEN = 65536
GPU_MEM_UTIL = 0.90
KV_CACHE_DTYPE = "fp8"