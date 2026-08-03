#!/usr/bin/env python3
"""
Profile DeepSeek V4 Flash inference with vLLM — TP=4 vs TP=8.
"""

import csv
import gc
import json
import os
import time
from datetime import datetime

import torch
from vllm import LLM, SamplingParams

MODEL_PATH = "/data/model/DeepSeek-V4-Flash-0731"
CONTEXT_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384, 32768]
OUTPUT_LEN = 256
MAX_MODEL_LEN = 65536
GPU_MEM_UTIL = 0.90
KV_CACHE_DTYPE = "fp8"

# --- Results logging ---
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_PATH = f"profile_results_{TIMESTAMP}.csv"
JSON_PATH = f"profile_results_{TIMESTAMP}.json"
ALL_RESULTS: list[dict] = []  # accumulated for JSON dump at the end

# Determine CSV columns — per-GPU columns vary by TP, so define headers when we know tp
_csv_writer = None
_csv_file = None


def _csv_headers(tp: int) -> list[str]:
    cols = [
        "tp", "context_length", "prompt_tokens", "output_tokens",
        "ttft_ms", "prefill_ms", "prefill_tps", "decode_tps", "tpot_ms", "total_ms",
        "avg_gpu_mem_mb",
    ]
    cols += [f"gpu{i}_mem_mb" for i in range(tp)]
    return cols


def _open_csv(tp: int):
    """Open the CSV file and write the header (called once per TP run)."""
    global _csv_file, _csv_writer
    if _csv_file:
        _csv_file.close()
    write_header = not os.path.exists(CSV_PATH)
    _csv_file = open(CSV_PATH, "a", newline="")
    _csv_writer = csv.DictWriter(_csv_file, fieldnames=_csv_headers(tp))
    if write_header:
        _csv_writer.writeheader()
        _csv_file.flush()


def _close_csv():
    global _csv_file, _csv_writer
    if _csv_file:
        _csv_file.close()
        _csv_file = None
        _csv_writer = None


def _make_prompt(ctx_len: int) -> str:
    para = (
        "Artificial intelligence has revolutionized the way we interact with technology. "
        "Deep learning models have demonstrated remarkable capabilities in understanding "
        "and generating human-like text across a wide range of domains and applications. "
        "The transformer architecture, introduced in 2017, remains the foundation of most "
        "state-of-the-art language models today. Researchers continue to push the boundaries "
        "of what is possible with larger models, more data, and novel training techniques. "
    )
    return (para * max(1, ctx_len // 55 + 1))[: ctx_len * 4]


def _gpu_mem(tp: int):
    """Return (used_mb_per_gpu, total_mb_per_gpu)."""
    used, total = [], []
    for i in range(tp):
        free, tot = torch.cuda.mem_get_info(i)
        used.append(round((tot - free) / 1024**2, 1))
        total.append(round(tot / 1024**2, 1))
    return used, total


def run_tp(tp: int):
    print(f"\n{'='*60}\nLoading model TP={tp}, max_model_len={MAX_MODEL_LEN}\n{'='*60}")

    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=tp,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM_UTIL,
        trust_remote_code=True,
        kv_cache_dtype=KV_CACHE_DTYPE,
        enforce_eager=False,
    )
    print("Model loaded.\n")

    # Initial GPU state
    used_init, total_init = _gpu_mem(tp)
    for i in range(tp):
        print(f"  GPU {i}: {used_init[i]:.0f} MB / {total_init[i]:.0f} MB ({used_init[i]/total_init[i]*100:.1f}%)")

    # Warmup
    llm.generate([_make_prompt(CONTEXT_LENGTHS[0])], SamplingParams(temperature=0, max_tokens=OUTPUT_LEN, ignore_eos=True))

    # Open CSV for this TP run
    _open_csv(tp)

    print(f"\n{'ctx':>6s} | {'prompt_tok':>5s} {'out_tok':>3s} | {'TTFT_ms':>8s} | {'prefill_t/s':>10s} | {'decode_t/s':>10s} | {'total_ms':>8s} | {'GPU_avg_MB':>10s}")
    print("-" * 85)

    for ctx_len in CONTEXT_LENGTHS:
        prompt = _make_prompt(ctx_len)
        sp = SamplingParams(temperature=0, max_tokens=OUTPUT_LEN, ignore_eos=True)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = llm.generate([prompt], sp)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        out = outputs[0]
        prompt_tok = len(out.prompt_token_ids)
        out_tok = len(out.outputs[0].token_ids)
        total_ms = (t1 - t0) * 1000

        m = out.metrics
        # TTFT
        ttft_ms = m.first_token_time * 1000 if m and m.first_token_time else 0
        # Decode tokens/sec from vLLM's time_per_output_token
        tpot_ms = (m.time_per_output_token * 1000) if m and m.time_per_output_token else 0
        decode_tps = (1.0 / m.time_per_output_token) if m and m.time_per_output_token and m.time_per_output_token > 0 else 0
        # Prefill = TTFT minus one decode step
        first_decode_ms = tpot_ms
        prefill_ms = max(0, ttft_ms - first_decode_ms)
        prefill_tps = prompt_tok / (prefill_ms / 1000) if prefill_ms > 0 else 0

        used, _ = _gpu_mem(tp)
        avg_mem = sum(used) / len(used)

        print(f"{ctx_len:>6d} | {prompt_tok:>5d} {out_tok:>3d} | {ttft_ms:>8.1f} | {prefill_tps:>10.0f} | {decode_tps:>10.1f} | {total_ms:>8.0f} | {avg_mem:>10.0f}")

        # --- Build result record ---
        row = {
            "tp": tp,
            "context_length": ctx_len,
            "prompt_tokens": prompt_tok,
            "output_tokens": out_tok,
            "ttft_ms": round(ttft_ms, 2),
            "prefill_ms": round(prefill_ms, 2),
            "prefill_tps": round(prefill_tps, 1),
            "decode_tps": round(decode_tps, 1),
            "tpot_ms": round(tpot_ms, 2),
            "total_ms": round(total_ms, 1),
            "avg_gpu_mem_mb": round(avg_mem, 1),
        }
        for i in range(tp):
            row[f"gpu{i}_mem_mb"] = used[i]

        # Append CSV row and accumulate for JSON
        if _csv_writer:
            _csv_writer.writerow(row)
            _csv_file.flush()
        ALL_RESULTS.append(row)

    _close_csv()

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)


def _print_summary():
    """Print a TP=4 vs TP=8 comparison table."""
    if not ALL_RESULTS:
        return

    print(f"\n\n{'='*100}")
    print("SUMMARY — TP=4 vs TP=8 comparison")
    print(f"{'='*100}")

    ctxs = sorted(set(r["context_length"] for r in ALL_RESULTS))
    tps = sorted(set(r["tp"] for r in ALL_RESULTS))

    # Group by (tp, ctx)
    by_key = {}
    for r in ALL_RESULTS:
        by_key[(r["tp"], r["context_length"])] = r

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

    print(f"\nResults saved to: {CSV_PATH}")
    print(f"Results saved to: {JSON_PATH}")


if __name__ == "__main__":
    for tp in [4, 8]:
        try:
            run_tp(tp)
        except Exception as e:
            print(f"\nTP={tp} failed: {e}")
            # Retry with smaller max_model_len
            try:
                print(f"Retrying TP={tp} with max_model_len=32768...")
                old, MAX_MODEL_LEN = MAX_MODEL_LEN, 32768
                run_tp(tp)
                MAX_MODEL_LEN = old
            except Exception as e2:
                print(f"TP={tp} still failed: {e2}")

    # Write JSON summary
    metadata = {
        "model_path": MODEL_PATH,
        "timestamp": TIMESTAMP,
        "max_model_len": MAX_MODEL_LEN,
        "gpu_memory_utilization": GPU_MEM_UTIL,
        "kv_cache_dtype": KV_CACHE_DTYPE,
        "output_len": OUTPUT_LEN,
    }
    with open(JSON_PATH, "w") as f:
        json.dump({"metadata": metadata, "results": ALL_RESULTS}, f, indent=2)

    _print_summary()
