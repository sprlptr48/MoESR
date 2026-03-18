from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .attention import ChannelAttentionBlock
from .config import MoESRConfig


class SRExpert(nn.Module):
    """Single MoE expert operating on tokens.

    Input:
        x: [T, C]
    Output:
        out: [T, C]
    """

    def __init__(self, config: MoESRConfig) -> None:
        super().__init__()
        hidden_dim = int(config.embed_dim * config.mlp_ratio)
        self.norm = nn.LayerNorm(config.embed_dim, eps=config.norm_eps)
        self.fc1 = nn.Linear(config.embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.ca = ChannelAttentionBlock(hidden_dim, reduction=config.channel_attention_reduction)
        self.fc2 = nn.Linear(hidden_dim, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.ca(x.unsqueeze(0)).squeeze(0)
        x = self.dropout(self.fc2(x))
        return x


class TopKRouter(nn.Module):
    """Top-k token router with capacity control.

    Input:
        x: [B, N, C]
    Output:
        dispatch_mask: [B, N, E]
        combine_weights: [B, N, E]
    """

    def __init__(self, config: MoESRConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.experts_per_token = config.experts_per_token
        self.capacity_factor = config.expert_capacity_factor
        self.jitter_noise = config.router_jitter_noise
        self.proj = nn.Linear(config.embed_dim, config.num_experts)
        self.aux_loss = torch.tensor(0.0)
        self.last_stats: Dict[str, Tensor] = {}

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        b, n, _ = x.shape
        if self.training and self.jitter_noise > 0.0:
            x = x + torch.randn_like(x) * self.jitter_noise

        logits = self.proj(x)
        probs = logits.softmax(dim=-1)
        topk_weights, topk_indices = torch.topk(probs, k=self.experts_per_token, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        tokens = b * n
        capacity = math.ceil(self.capacity_factor * tokens * self.experts_per_token / self.num_experts)

        dispatch_mask = torch.zeros_like(probs)
        combine_weights = torch.zeros_like(probs)

        flat_indices = topk_indices.reshape(tokens, self.experts_per_token)
        flat_weights = topk_weights.reshape(tokens, self.experts_per_token)

        for rank in range(self.experts_per_token):
            expert_ids = flat_indices[:, rank]
            one_hot = F.one_hot(expert_ids, num_classes=self.num_experts).to(probs.dtype)
            position_in_expert = torch.cumsum(one_hot, dim=0) - 1
            accepted = position_in_expert < capacity
            accepted = accepted * one_hot
            accepted_any = accepted.sum(dim=-1)
            combine_weights.view(tokens, self.num_experts).add_(accepted * flat_weights[:, rank : rank + 1])
            dispatch_mask.view(tokens, self.num_experts).add_(accepted)

        combine_weights = combine_weights * (dispatch_mask > 0).to(combine_weights.dtype)
        combine_weights = combine_weights / combine_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        total_dispatches = dispatch_mask.sum().clamp_min(1.0)
        f_i = dispatch_mask.sum(dim=(0, 1)) / total_dispatches
        p_i = probs.mean(dim=(0, 1))
        self.aux_loss = self.num_experts * torch.sum(f_i * p_i)
        self.last_stats = {
            "dispatch_counts": dispatch_mask.sum(dim=(0, 1)).detach(),
            "routing_probs": p_i.detach(),
            "topk_indices": topk_indices.detach(),
            "capacity": torch.tensor(capacity, device=x.device),
        }
        return dispatch_mask, combine_weights


class MoELayer(nn.Module):
    """Mixture-of-experts token mixer.

    Input:
        x: [B, N, C]
    Output:
        out: [B, N, C]
    """

    def __init__(self, config: MoESRConfig) -> None:
        super().__init__()
        self.router = TopKRouter(config)
        self.experts = nn.ModuleList([SRExpert(config) for _ in range(config.num_experts)])
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x: Tensor) -> Tensor:
        dispatch_mask, combine_weights = self.router(x)
        b, n, c = x.shape
        flat_x = x.reshape(b * n, c)
        flat_dispatch = dispatch_mask.reshape(b * n, -1)
        flat_combine = combine_weights.reshape(b * n, -1)
        out = torch.zeros_like(flat_x)

        for expert_idx, expert in enumerate(self.experts):
            token_mask = flat_dispatch[:, expert_idx] > 0
            if not token_mask.any():
                continue
            expert_input = flat_x[token_mask]
            expert_output = expert(expert_input)
            out[token_mask] += expert_output * flat_combine[token_mask, expert_idx : expert_idx + 1]

        self.aux_loss = self.router.aux_loss
        return out.view(b, n, c)
