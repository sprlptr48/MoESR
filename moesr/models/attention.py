from __future__ import annotations

from typing import Optional

import torch
from einops import rearrange
from torch import Tensor, nn
import torch.nn.functional as F

from .config import MoESRConfig
from .patch_utils import get_relative_position_index, img_to_windows, pad_to_window_multiple, windows_to_img


class WindowAttention(nn.Module):
    """Shifted window self-attention over BHWC features.

    Input:
        x: [B, H, W, C]
    Output:
        out: [B, H, W, C]
    """

    def __init__(self, config: MoESRConfig, shift_size: int = 0) -> None:
        super().__init__()
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.window_size = config.window_size
        self.shift_size = shift_size
        self.scale = (self.embed_dim // self.num_heads) ** -0.5
        self.qkv = nn.Linear(self.embed_dim, self.embed_dim * 3)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.attn_drop = nn.Dropout(config.attention_dropout)
        self.proj_drop = nn.Dropout(config.proj_dropout)

        num_relative_positions = (2 * self.window_size - 1) ** 2
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(num_relative_positions, self.num_heads)
        )
        self.relative_position_mlp = nn.Sequential(
            nn.Linear(2, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.num_heads),
        )
        self.register_buffer(
            "relative_position_index",
            get_relative_position_index(self.window_size),
            persistent=False,
        )
        self.register_buffer(
            "relative_coords_table",
            self._build_relative_coords_table(self.window_size),
            persistent=False,
        )

    @staticmethod
    def _build_relative_coords_table(window_size: int) -> Tensor:
        coords_h = torch.arange(-(window_size - 1), window_size, dtype=torch.float32)
        coords_w = torch.arange(-(window_size - 1), window_size, dtype=torch.float32)
        relative_coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
        relative_coords = relative_coords / max(window_size - 1, 1)
        relative_coords = torch.sign(relative_coords) * torch.log2(relative_coords.abs() + 1.0) / torch.log2(torch.tensor(8.0))
        return relative_coords

    def _relative_position_bias(self) -> Tensor:
        coords = self.relative_coords_table.view(-1, 2)
        continuous_bias = self.relative_position_mlp(coords)
        bias_table = self.relative_position_bias_table + continuous_bias
        relative_bias = bias_table[self.relative_position_index.view(-1)]
        relative_bias = relative_bias.view(
            self.window_size * self.window_size,
            self.window_size * self.window_size,
            self.num_heads,
        )
        return rearrange(relative_bias, "n m h -> h n m")

    def _build_mask(self, batch_size: int, h: int, w: int, device: torch.device) -> Optional[Tensor]:
        if self.shift_size == 0:
            return None
        img_mask = torch.zeros((1, h, w, 1), device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        count = 0
        for hs in h_slices:
            for ws in w_slices:
                img_mask[:, hs, ws, :] = count
                count += 1
        mask_windows = img_to_windows(img_mask, self.window_size).squeeze(-1)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float("-inf")).masked_fill(attn_mask == 0, 0.0)
        return attn_mask.repeat(batch_size, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        b, h, w, c = x.shape
        x, (pad_h, pad_w) = pad_to_window_multiple(x, self.window_size)
        hp, wp = x.shape[1], x.shape[2]

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        windows = img_to_windows(x, self.window_size)
        qkv = self.qkv(windows)
        q, k, v = rearrange(
            qkv,
            "bn n (three heads dim) -> three bn heads n dim",
            three=3,
            heads=self.num_heads,
        )

        attn_mask = self._build_mask(b, hp, wp, windows.device)
        rel_bias = self._relative_position_bias().to(windows.device)

        if hasattr(F, "scaled_dot_product_attention"):
            mask = rel_bias.unsqueeze(0)
            if attn_mask is not None:
                mask = mask + attn_mask.unsqueeze(1)
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                scale=self.scale,
            )
        else:
            scores = (q * self.scale) @ k.transpose(-2, -1)
            scores = scores + rel_bias.unsqueeze(0)
            if attn_mask is not None:
                scores = scores + attn_mask.unsqueeze(1)
            probs = self.attn_drop(scores.softmax(dim=-1))
            attn_out = probs @ v

        out = rearrange(attn_out, "bn heads n dim -> bn n (heads dim)")
        out = self.proj_drop(self.proj(out))
        out = windows_to_img(out, hp, wp, self.window_size)

        if self.shift_size > 0:
            out = torch.roll(out, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        if pad_h > 0 or pad_w > 0:
            out = out[:, :h, :w, :]
        return out


class OverlapCrossAttention(nn.Module):
    """Cross-window aggregation with local and expanded receptive fields.

    Input:
        x: [B, H, W, C]
    Output:
        out: [B, H, W, C]
    """

    def __init__(self, config: MoESRConfig) -> None:
        super().__init__()
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.local_window = config.window_size
        self.expanded_window = config.overlap_window_size
        self.scale = (self.embed_dim // self.num_heads) ** -0.5
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.kv_proj = nn.Linear(self.embed_dim, self.embed_dim * 2)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        b, h, w, c = x.shape
        local, _ = pad_to_window_multiple(x, self.local_window)
        expanded, _ = pad_to_window_multiple(x, self.expanded_window)
        hl, wl = local.shape[1], local.shape[2]
        he, we = expanded.shape[1], expanded.shape[2]
        local_grid_h = hl // self.local_window
        local_grid_w = wl // self.local_window
        expanded_grid_h = he // self.expanded_window
        expanded_grid_w = we // self.expanded_window

        local_windows = img_to_windows(local, self.local_window)
        expanded_windows = img_to_windows(expanded, self.expanded_window)
        local_windows = local_windows.view(
            b,
            local_grid_h,
            local_grid_w,
            self.local_window * self.local_window,
            c,
        )
        expanded_windows = expanded_windows.view(
            b,
            expanded_grid_h,
            expanded_grid_w,
            self.expanded_window * self.expanded_window,
            c,
        )

        local_rows = torch.arange(local_grid_h, device=x.device)
        local_cols = torch.arange(local_grid_w, device=x.device)
        mapped_rows = torch.clamp((local_rows * self.local_window) // self.expanded_window, max=expanded_grid_h - 1)
        mapped_cols = torch.clamp((local_cols * self.local_window) // self.expanded_window, max=expanded_grid_w - 1)
        row_index = mapped_rows[:, None].expand(local_grid_h, local_grid_w)
        col_index = mapped_cols[None, :].expand(local_grid_h, local_grid_w)
        expanded_windows = expanded_windows[:, row_index, col_index]

        local_windows = local_windows.view(b * local_grid_h * local_grid_w, self.local_window * self.local_window, c)
        expanded_windows = expanded_windows.view(
            b * local_grid_h * local_grid_w,
            self.expanded_window * self.expanded_window,
            c,
        )

        q = rearrange(
            self.q_proj(local_windows),
            "bn n (heads dim) -> bn heads n dim",
            heads=self.num_heads,
        )
        k, v = rearrange(
            self.kv_proj(expanded_windows),
            "bn n (two heads dim) -> two bn heads n dim",
            two=2,
            heads=self.num_heads,
        )

        if hasattr(F, "scaled_dot_product_attention"):
            attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, scale=self.scale)
        else:
            scores = (q * self.scale) @ k.transpose(-2, -1)
            probs = scores.softmax(dim=-1)
            attn_out = probs @ v

        out = rearrange(attn_out, "bn heads n dim -> bn n (heads dim)")
        out = self.out_proj(out)
        out = windows_to_img(out, hl, wl, self.local_window)
        return out[:, :h, :w, :]


class ChannelAttentionBlock(nn.Module):
    """Squeeze-and-excitation channel attention.

    Input:
        x: [B, N, C] or [B, C, H, W]
    Output:
        out: Same shape as input.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 3:
            scale = x.mean(dim=1)
            scale = F.relu(self.fc1(scale), inplace=True)
            scale = torch.sigmoid(self.fc2(scale)).unsqueeze(1)
            return x * scale
        if x.dim() == 4:
            scale = x.mean(dim=(2, 3))
            scale = F.relu(self.fc1(scale), inplace=True)
            scale = torch.sigmoid(self.fc2(scale)).view(x.shape[0], x.shape[1], 1, 1)
            return x * scale
        raise ValueError("ChannelAttentionBlock expects a 3D or 4D tensor.")
