from __future__ import annotations

from collections import defaultdict

import torch

from moesr.models.config import MoESRConfig
from moesr.models.moe import MoELayer
from moesr.models.moesr import MoESR


def count_parameters(module: torch.nn.Module) -> int:
    """Count trainable parameters."""

    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def report_config(name: str, cfg: MoESRConfig) -> None:
    """Print parameter accounting for a config preset."""

    model = MoESR(cfg)

    total_params = count_parameters(model)
    shallow_params = count_parameters(model.shallow_extractor)
    stage1_params = count_parameters(model.stage1)
    stage2_params = count_parameters(model.stage2)
    head_params = count_parameters(model.final_reconstruction)

    moe_layers = [module for module in model.modules() if isinstance(module, MoELayer)]
    expert_params_per_layer = []
    router_params_total = 0
    expert_params_total = 0
    active_expert_params_total = 0

    for layer in moe_layers:
        router_params = count_parameters(layer.router)
        single_expert_params = count_parameters(layer.experts[0])
        total_layer_expert_params = sum(count_parameters(expert) for expert in layer.experts)
        router_params_total += router_params
        expert_params_total += total_layer_expert_params
        active_expert_params_total += int(total_layer_expert_params * cfg.experts_per_token / cfg.num_experts)
        expert_params_per_layer.append(single_expert_params)

    dense_non_expert_params = total_params - expert_params_total
    active_params = dense_non_expert_params + active_expert_params_total
    per_stage = defaultdict(int)
    for stage_name, module in [("stage1", model.stage1), ("stage2", model.stage2)]:
        per_stage[stage_name] = count_parameters(module)

    print(f"[{name}]")
    print(f"Total parameters: {total_params:,}")
    print(f"Active parameters / forward (estimated): {active_params:,}")
    print(f"Shallow extractor parameters: {shallow_params:,}")
    print(f"Stage 1 parameters: {stage1_params:,}")
    print(f"Stage 2 parameters: {stage2_params:,}")
    print(f"Final reconstruction head parameters: {head_params:,}")
    print(f"Parameters per stage: {dict(per_stage)}")
    print(f"Parameters per expert (single expert module): {expert_params_per_layer[0]:,}")
    print(f"Total expert parameters: {expert_params_total:,}")
    print(f"Router parameters: {router_params_total:,}")
    print(f"Router share of total: {100.0 * router_params_total / max(total_params, 1):.4f}%")

    if total_params < 900_000_000:
        print("Budget note: this preset is below 1B class.")
    elif total_params > 1_100_000_000:
        print("Budget note: this preset is above the 1B target window.")
    print()


def main() -> None:
    report_config("default", MoESRConfig())
    report_config("moe_1b", MoESRConfig.moe_1b())


if __name__ == "__main__":
    main()
