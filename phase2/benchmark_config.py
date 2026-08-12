"""
Benchmark dataset configuration.

Defines which datasets to use, output lengths, sampling limits, and
needle-in-a-haystack parameters for long-context throughput testing.
"""

# ---- LongBench datasets (when available locally) ----
# Downloaded from ModelScope: ZhipuAI/LongBench-v2
# Or HuggingFace: THUDM/LongBench
LONGBENCH_DATASETS = [
    "narrativeqa",       # QA,      avg 31K tokens, EN
    "qasper",            # QA,      avg  4K tokens, EN
    "gov_report",        # Summ,    avg  9K tokens, EN
    "qmsum",             # Summ,    avg 14K tokens, EN
    "hotpotqa",          # QA,      avg 12K tokens, EN
    "triviaqa",          # QA,      avg 20K tokens, EN
    "multifieldqa_en",   # QA,      avg  7K tokens, EN
]

# Local path(s) to search for pre-downloaded datasets
import os

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LONGBENCH_DATA_DIRS = [
    os.path.join(_PROJ, "data"),
    "/data/benchmarks/longbench",
    "/data/benchmarks/LongBench",
    "/data/benchmarks",
]

# ---- Output lengths to test ----
OUTPUT_LENS = [128, 256, 512]

# ---- Sampling ----
MAX_SAMPLES_PER_DATASET = 50  # cap samples per task for runtime

# ---- Needle-in-a-Haystack (synthetic, no download needed) ----
# Used as the primary throughput workload when real datasets are unavailable.
NEEDLE_CONTEXT_LENGTHS = [
    1024, 2048, 4096, 8192, 16384, 32768,
    # 65536,   # uncomment when PP deployment is stable for longer contexts
]

NEEDLE_DEPTHS = [0.25, 0.50, 0.75]  # passkey positions (fraction of context)
NEEDLE_PASSKEY = "The special passkey is: FIREFLY-9472-DELTA"

# Filler corpus: real-looking paragraphs for the haystack.
# Using PG19 / book excerpts would be ideal, but for self-contained testing
# we ship a few reusable paragraphs.
FILLER_PARAGRAPHS = [
    (
        "The history of artificial intelligence dates back to the mid-20th century, "
        "when researchers first began exploring the possibility of creating machines "
        "that could simulate human reasoning and problem-solving capabilities. "
        "Early pioneers like Alan Turing, John McCarthy, and Marvin Minsky laid the "
        "foundations for what would become one of the most transformative technologies "
        "in human history."
    ),
    (
        "Climate change represents one of the most significant challenges facing "
        "humanity in the twenty-first century. Rising global temperatures, caused "
        "primarily by greenhouse gas emissions from human activities, are leading to "
        "more frequent extreme weather events, sea level rise, and disruptions to "
        "ecosystems and agricultural systems worldwide."
    ),
    (
        "The development of modern cryptography has its roots in ancient methods of "
        "secret communication. From the Caesar cipher used by Roman generals to the "
        "Enigma machine of World War II, humans have long sought ways to protect "
        "sensitive information from unauthorized access. Today's cryptographic systems "
        "rely on complex mathematical principles and computational hardness assumptions."
    ),
    (
        "Quantum computing represents a paradigm shift in how we process information. "
        "Unlike classical computers that use bits representing either 0 or 1, quantum "
        "computers leverage quantum bits or qubits that can exist in superposition "
        "states. This property, combined with quantum entanglement, enables quantum "
        "algorithms to solve certain problems exponentially faster than their classical "
        "counterparts."
    ),
    (
        "The human genome project, completed in 2003, was an international scientific "
        "effort to sequence and map all human genes. This monumental achievement took "
        "thirteen years and cost approximately 2.7 billion dollars. Today, whole genome "
        "sequencing can be completed in a matter of days for less than a thousand "
        "dollars, demonstrating the incredible pace of technological advancement in "
        "genomics and biotechnology."
    ),
]
