"""Install/remove the K3 MoE CUDA Event hooks in the active vLLM environment."""

import argparse
import shutil
from pathlib import Path

import vllm


MARKER = "# K3_MOE_EVENT_TIMING_V1"
ROOT = Path(vllm.__file__).resolve().parent
FILES = {
    ROOT / "model_executor/layers/fused_moe/router/fused_moe_router.py": """
from vllm.model_executor.layers.fused_moe.event_timing import wrap_method as _k3_wrap_method
_k3_wrap_method(FusedMoERouter, "select_experts", "router_topk")
""",
    ROOT / "model_executor/layers/fused_moe/modular_kernel.py": """
from vllm.model_executor.layers.fused_moe.event_timing import wrap_method as _k3_wrap_method
_k3_wrap_method(FusedMoEKernelModularImpl, "_prepare", "prepare_dispatch")
_k3_wrap_method(FusedMoEKernelModularImpl, "_fused_experts", "marlin_experts")
_k3_wrap_method(FusedMoEKernelModularImpl, "_finalize", "finalize_combine")
""",
    ROOT / "model_executor/layers/fused_moe/runner/moe_runner.py": """
from vllm.model_executor.layers.fused_moe.event_timing import wrap_method as _k3_wrap_method
_k3_wrap_method(MoERunner, "forward", "moe_total")
_k3_wrap_method(MoERunner, "apply_routed_input_transform", "routed_input_transform")
_k3_wrap_method(MoERunner, "apply_routed_output_transform", "routed_output_transform")
_k3_wrap_method(MoERunner, "_maybe_reduce_routed_output_before_transform", "tp_reduce_routed")
_k3_wrap_method(MoERunner, "_maybe_reduce_shared_expert_output", "tp_reduce_shared")
_k3_wrap_method(MoERunner, "_maybe_reduce_final_output", "tp_reduce_final")
""",
    ROOT / "model_executor/layers/fused_moe/runner/shared_experts.py": """
from vllm.model_executor.layers.fused_moe.event_timing import wrap_shared_experts as _k3_wrap_shared
_k3_wrap_shared(SharedExperts, SharedExpertsOrder)
""",
}
HELPER = ROOT / "model_executor/layers/fused_moe/event_timing.py"


def install(source_helper: Path):
    shutil.copy2(source_helper, HELPER)
    for path, code in FILES.items():
        text = path.read_text()
        if MARKER in text:
            continue
        backup = path.with_suffix(path.suffix + ".pre_k3_moe_event")
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text(text + "\n" + MARKER + code)
        print(f"patched {path}")
    print(f"installed {HELPER}")


def remove():
    for path in FILES:
        backup = path.with_suffix(path.suffix + ".pre_k3_moe_event")
        if backup.exists():
            shutil.copy2(backup, path)
            print(f"restored {path}")
    HELPER.unlink(missing_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument("action", choices=("install", "remove"))
parser.add_argument("--helper", type=Path)
args = parser.parse_args()
if args.action == "install":
    if args.helper is None:
        parser.error("install requires --helper")
    install(args.helper)
else:
    remove()
