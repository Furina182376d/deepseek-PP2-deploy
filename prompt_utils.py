"""
Prompt generation utilities for benchmarking.
"""


def make_prompt(ctx_len: int) -> str:
    """Generate a synthetic prompt of approximately *ctx_len* tokens."""
    para = (
        "Artificial intelligence has revolutionized the way we interact with technology. "
        "Deep learning models have demonstrated remarkable capabilities in understanding "
        "and generating human-like text across a wide range of domains and applications. "
        "The transformer architecture, introduced in 2017, remains the foundation of most "
        "state-of-the-art language models today. Researchers continue to push the boundaries "
        "of what is possible with larger models, more data, and novel training techniques. "
    )
    return (para * max(1, ctx_len // 55 + 1))[: ctx_len * 4]