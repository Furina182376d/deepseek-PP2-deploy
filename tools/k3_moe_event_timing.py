"""CUDA Event aggregation for Kimi-K3 fused-MoE profiling without CUPTI."""

import atexit
import functools
import json
import os
import threading

import torch


_ENABLED = os.environ.get("VLLM_MOE_EVENT_TIMING", "0") == "1"
_FLUSH_EVENTS = int(os.environ.get("VLLM_MOE_EVENT_FLUSH_EVENTS", "2048"))
_PENDING = []
_LOCK = threading.Lock()


def _token_count(args, kwargs):
    for name in (
        "hidden_states",
        "shared_experts_input",
        "router_logits",
        "a1q",
        "output",
        "fused_output",
        "states",
    ):
        value = kwargs.get(name)
        if isinstance(value, torch.Tensor) and value.ndim:
            return int(value.shape[0])
    for value in args:
        if isinstance(value, torch.Tensor) and value.ndim:
            return int(value.shape[0])
    return -1


def _append(label, tokens, start, end):
    should_flush = False
    with _LOCK:
        _PENDING.append((label, tokens, start, end))
        should_flush = len(_PENDING) >= _FLUSH_EVENTS
    if should_flush:
        flush()


def timed_call(label, func, args, kwargs, stream=None):
    if not _ENABLED or not torch.cuda.is_available():
        return func(*args, **kwargs)
    tokens = _token_count(args, kwargs)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    if stream is None:
        start.record()
        result = func(*args, **kwargs)
        end.record()
    else:
        with torch.cuda.stream(stream):
            start.record()
        result = func(*args, **kwargs)
        with torch.cuda.stream(stream):
            end.record()
    _append(label, tokens, start, end)
    return result


def wrap_method(cls, method_name, label):
    original = getattr(cls, method_name)
    if getattr(original, "_k3_moe_event_wrapped", False):
        return

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        return timed_call(label, original, args, kwargs)

    wrapped._k3_moe_event_wrapped = True
    setattr(cls, method_name, wrapped)


def wrap_shared_experts(cls, order_cls):
    original_forward = cls.forward
    original_aux = cls._run_in_aux_stream
    if getattr(original_forward, "_k3_moe_event_wrapped", False):
        return

    @functools.wraps(original_forward)
    def forward(self, shared_experts_input, order):
        actual_order = self._determine_shared_experts_order(shared_experts_input)
        if order != actual_order or order == order_cls.MULTI_STREAM_OVERLAPPED:
            return original_forward(self, shared_experts_input, order)
        return timed_call(
            "shared_expert",
            original_forward,
            (self, shared_experts_input, order),
            {},
        )

    @functools.wraps(original_aux)
    def aux(self, shared_experts_input):
        return timed_call(
            "shared_expert",
            original_aux,
            (self, shared_experts_input),
            {},
            stream=self._stream,
        )

    forward._k3_moe_event_wrapped = True
    aux._k3_moe_event_wrapped = True
    cls.forward = forward
    cls._run_in_aux_stream = aux


def flush():
    if not _ENABLED or not torch.cuda.is_available():
        return
    with _LOCK:
        if not _PENDING:
            return
        records = list(_PENDING)
        _PENDING.clear()
    try:
        torch.cuda.synchronize()
        buckets = {}
        for label, tokens, start, end in records:
            key = (label, tokens)
            total, count = buckets.get(key, (0.0, 0))
            buckets[key] = (total + start.elapsed_time(end), count + 1)
        rows = [
            {
                "label": label,
                "tokens": tokens,
                "count": count,
                "total_ms": round(total, 6),
                "mean_ms": round(total / count, 6),
            }
            for (label, tokens), (total, count) in sorted(buckets.items())
        ]
        print(
            "K3_MOE_EVENT_V1 "
            + json.dumps(
                {"rank": int(os.environ.get("RANK", "-1")), "rows": rows},
                separators=(",", ":"),
            ),
            flush=True,
        )
    except Exception as exc:
        print(f"K3_MOE_EVENT_ERROR rank={os.environ.get('RANK', '?')} {exc}", flush=True)


atexit.register(flush)
