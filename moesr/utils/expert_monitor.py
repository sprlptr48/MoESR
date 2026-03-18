from __future__ import annotations

from typing import Dict, List

import torch

from moesr.models.moe import MoELayer


class ExpertUtilizationMonitor:
    """Accumulate router dispatch counts across MoE layers.

    report() returns utilization diagnostics and dead-expert warnings.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.layers: List[MoELayer] = [module for module in model.modules() if isinstance(module, MoELayer)]
        self.reset()

    def reset(self) -> None:
        """Clear accumulated counts."""

        self.dispatch_counts = None
        self.num_updates = 0

    def update(self) -> None:
        """Pull the latest router stats from every MoE layer."""

        layer_counts = []
        for layer in self.layers:
            stats = getattr(layer.router, "last_stats", {})
            if "dispatch_counts" in stats:
                layer_counts.append(stats["dispatch_counts"].float().cpu())
        if not layer_counts:
            return
        stacked = torch.stack(layer_counts).sum(dim=0)
        self.dispatch_counts = stacked if self.dispatch_counts is None else self.dispatch_counts + stacked
        self.num_updates += 1

    def report(self, reset: bool = False) -> Dict[str, object]:
        """Return utilization statistics gathered so far."""

        self.update()
        if self.dispatch_counts is None:
            result = {
                "utilization": [],
                "dispatch_counts": [],
                "utilization_std": 0.0,
                "collapse_detected": False,
                "dead_experts": [],
                "warning": "No router activity captured yet.",
            }
            if reset:
                self.reset()
            return result

        utilization = self.dispatch_counts / self.dispatch_counts.sum().clamp_min(1.0)
        dead_experts = [idx for idx, value in enumerate(utilization.tolist()) if value < 0.05]
        result = {
            "utilization": utilization.tolist(),
            "dispatch_counts": [int(value) for value in self.dispatch_counts.tolist()],
            "utilization_std": float(utilization.std(unbiased=False).item()),
            "collapse_detected": bool(utilization.max().item() > 0.5),
            "dead_experts": dead_experts,
            "warning": "Dead expert detected." if dead_experts else "",
        }
        if reset:
            self.reset()
        return result
