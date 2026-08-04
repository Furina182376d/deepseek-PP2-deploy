"""
Dataset loader — real benchmarks and synthetic needle-in-a-haystack prompts.

Public API:
    load_longbench(task_name) -> list[dict]
        Load a LongBench task (requires pre-downloaded JSON/JSONL).

    make_needle_prompt(ctx_len, depth, passkey, filler) -> str
        Generate a single needle-in-a-haystack prompt.

    iter_needle_prompts(config) -> Iterator[dict]
        Iterate over (prompt, ctx_len, depth) tuples for profiling.
"""

import json
import os
import random
from pathlib import Path
from typing import Iterator

from benchmark_config import (
    FILLER_PARAGRAPHS,
    LONGBENCH_DATA_DIRS,
    MAX_SAMPLES_PER_DATASET,
    NEEDLE_CONTEXT_LENGTHS,
    NEEDLE_DEPTHS,
    NEEDLE_PASSKEY,
)


def _find_local_json(name: str) -> str | None:
    """Find a JSON/JSONL file for a LongBench task in known directories."""
    for data_dir in LONGBENCH_DATA_DIRS:
        for ext in (".jsonl", ".json"):
            candidate = os.path.join(data_dir, f"{name}{ext}")
            if os.path.isfile(candidate):
                return candidate
    return None


def load_longbench(task_name: str) -> list[dict]:
    """
    Load a LongBench task from local JSON/JSONL.
    Returns a list of dicts with at least ``{"input": str, "context": str}``.
    """
    path = _find_local_json(task_name)
    if path is None:
        # Try to construct the prompt from partial files
        for d in LONGBENCH_DATA_DIRS:
            d = os.path.join(d, "data")
            for ext in (".jsonl", ".json"):
                p = os.path.join(d, f"{task_name}{ext}")
                if os.path.isfile(p):
                    path = p
                    break
            if path:
                break

    if path is None:
        raise FileNotFoundError(
            f"LongBench task '{task_name}' not found in {LONGBENCH_DATA_DIRS}. "
            f"Download from ModelScope (ZhipuAI/LongBench-v2) or HuggingFace."
        )

    items: list[dict] = []
    with open(path, encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        else:
            data = json.load(f)
            items = data if isinstance(data, list) else [data]

    if len(items) > MAX_SAMPLES_PER_DATASET:
        rng = random.Random(42)
        items = rng.sample(items, MAX_SAMPLES_PER_DATASET)

    return items


def _build_filler(target_chars: int) -> str:
    """Build a filler string of approximately *target_chars* characters."""
    parts: list[str] = []
    total = 0
    while total < target_chars:
        para = random.choice(FILLER_PARAGRAPHS)
        parts.append(para)
        total += len(para)
    return " ".join(parts)[:target_chars]


def make_needle_prompt(
    ctx_len: int,
    depth: float = 0.5,
    passkey: str = NEEDLE_PASSKEY,
    filler_paragraphs: list[str] | None = None,
) -> str:
    """
    Generate a needle-in-a-haystack prompt.

    Parameters
    ----------
    ctx_len:
        Target prompt length in **tokens** (approximate).
    depth:
        Where to place the passkey, as a fraction [0.0 .. 1.0] of the context.
    passkey:
        The secret passkey string to hide in the haystack.
    filler_paragraphs:
        Optional custom filler text.  Uses ``FILLER_PARAGRAPHS`` if omitted.

    Returns
    -------
    A single-string prompt of the form::

        <filler>...<passkey>...<filler>

        Find the passkey hidden in the text above.
    """
    paras = filler_paragraphs or FILLER_PARAGRAPHS
    # Rough char-per-token for English: ~4 chars/token
    target_chars = ctx_len * 4

    passkey_sentence = f"\n\n{passkey}\n\n"
    passkey_chars = len(passkey_sentence)

    # Build the haystack with the passkey at the target depth
    before_chars = int(target_chars * depth) - passkey_chars // 2
    after_chars = target_chars - before_chars - passkey_chars

    before = _build_filler(max(0, before_chars))
    after = _build_filler(max(0, after_chars))

    haystack = before + passkey_sentence + after

    # Append the retrieval instruction
    prompt = (
        haystack
        + "\n\n"
        + "Based on the text above, what is the special passkey? "
        + "Reply with only the passkey string."
    )

    return prompt


def iter_needle_prompts(
    ctx_lengths: list[int] | None = None,
    depths: list[float] | None = None,
    passkey: str | None = None,
) -> Iterator[dict]:
    """
    Iterate over needle-in-a-haystack prompts.

    Yields dicts with keys: ``prompt``, ``ctx_len``, ``depth``, ``passkey``.
    """
    ctx_lengths = ctx_lengths or NEEDLE_CONTEXT_LENGTHS
    depths = depths or NEEDLE_DEPTHS
    passkey = passkey or NEEDLE_PASSKEY

    for ctx_len in ctx_lengths:
        for depth in depths:
            yield {
                "prompt": make_needle_prompt(ctx_len, depth, passkey),
                "ctx_len": ctx_len,
                "depth": depth,
                "passkey": passkey,
            }


# ---------------------------------------------------------------------------
# Dataset index — maps a short name to a loader / iter function
# ---------------------------------------------------------------------------
def list_available_datasets() -> dict[str, str]:
    """Return {name: status} for all known datasets."""
    available: dict[str, str] = {}
    for task in ["narrative_qa", "qasper", "gov_report", "qmsum"]:
        available[task] = "local" if _find_local_json(task) else "not_found"
    available["needle_haystack"] = "builtin"
    return available
