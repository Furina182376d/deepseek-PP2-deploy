"""
Core TP inference run — loads the model once per TP setting and profiles
across a range of context lengths.
"""

import gc
import time

import torch
from vllm import LLM, SamplingParams

import config
from gpu_utils import gpu_mem
from prompt_utils import make_prompt
from results_utils import close_csv, ensure_results_dir, open_csv, write_csv_row
from results_utils import ALL_RESULTS  # noqa: F401 — accumulated for final JSON dump


def run_tp(tp: int):
    print(f"\n{'='*60}\nLoading model TP={tp}, max_model_len={config.MAX_MODEL_LEN}\n{'='*60}")

    llm = LLM(
        model=config.MODEL_PATH,
        tensor_parallel_size=tp,
        max_model_len=config.MAX_MODEL_LEN,
        gpu_memory_utilization=config.GPU_MEM_UTIL,
        trust_remote_code=True,
        kv_cache_dtype=config.KV_CACHE_DTYPE,
        enforce_eager=False,
        disable_log_stats=False,  # required for per-request timing metrics (TTFT, TPOT, etc.)
    )
    print("Model loaded.\n")

    # Initial GPU state
    used_init, total_init = gpu_mem(tp)
    for i in range(tp):
        print(f"  GPU {i}: {used_init[i]:.0f} MB / {total_init[i]:.0f} MB ({used_init[i]/total_init[i]*100:.1f}%)")

    # Warmup
    llm.generate([make_prompt(config.CONTEXT_LENGTHS[0])], SamplingParams(temperature=0, max_tokens=config.OUTPUT_LEN, ignore_eos=True))

    # Open per-TP CSV
    ensure_results_dir()
    open_csv(tp)

    print(f"\n{'ctx':>6s} | {'prompt_tok':>5s} {'out_tok':>3s} | {'TTFT_ms':>8s} | {'prefill_t/s':>10s} | {'decode_t/s':>10s} | {'total_ms':>8s} | {'GPU_avg_MB':>10s}")
    print("-" * 85)

    for ctx_len in config.CONTEXT_LENGTHS:
        prompt = make_prompt(ctx_len)
        sp = SamplingParams(temperature=0, max_tokens=config.OUTPUT_LEN, ignore_eos=True)

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
        # vLLM v1 uses RequestStateStats — compute TTFT/TPOT from timestamps
        if m and m.first_token_latency is not None:
            ttft_ms = m.first_token_latency * 1000
            # TPOT = decode time / (generated_tokens - 1)  (first_token is already counted in TTFT)
            if m.num_generation_tokens > 1 and m.last_token_ts and m.first_token_ts:
                tpot_sec = (m.last_token_ts - m.first_token_ts) / (m.num_generation_tokens - 1)
                tpot_ms = tpot_sec * 1000
                decode_tps = 1.0 / tpot_sec
            else:
                tpot_ms = 0.0
                decode_tps = 0.0
            first_decode_ms = tpot_ms
            prefill_ms = max(0.0, ttft_ms - first_decode_ms)
            prefill_tps = prompt_tok / (prefill_ms / 1000) if prefill_ms > 0 else 0.0
        else:
            ttft_ms = 0.0
            tpot_ms = 0.0
            decode_tps = 0.0
            prefill_ms = 0.0
            prefill_tps = 0.0

        used, _ = gpu_mem(tp)
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
        write_csv_row(row)
        ALL_RESULTS.append(row)

    close_csv()

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)