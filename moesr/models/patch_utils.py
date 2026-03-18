from __future__ import annotations

from typing import Tuple

import torch
from einops import rearrange
from torch import Tensor
import torch.nn.functional as F


def pad_to_window_multiple(x: Tensor, window_size: int) -> Tuple[Tensor, Tuple[int, int]]:
    """Pad image/features to a multiple of window size.

    Args:
        x: Tensor of shape [B, H, W, C] or [B, C, H, W].
        window_size: Window size for attention partitioning.

    Returns:
        A tuple of padded tensor and (pad_h, pad_w).
    """

    if x.dim() != 4:
        raise ValueError("Expected a 4D tensor.")

    if x.shape[-1] < 8 and x.shape[1] >= 8:
        is_bhwc = False
        _, _, h, w = x.shape
    else:
        is_bhwc = True
        _, h, w, _ = x.shape

    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)

    if is_bhwc:
        padded = F.pad(x.permute(0, 3, 1, 2), (0, pad_w, 0, pad_h))
        return padded.permute(0, 2, 3, 1), (pad_h, pad_w)
    return F.pad(x, (0, pad_w, 0, pad_h)), (pad_h, pad_w)


def img_to_windows(x: Tensor, window_size: int) -> Tensor:
    """Partition a BHWC tensor into non-overlapping windows.

    Args:
        x: Tensor of shape [B, H, W, C].
        window_size: Spatial window size.

    Returns:
        Tensor of shape [B * num_windows, window_size * window_size, C].
    """

    b, h, w, c = x.shape
    if h % window_size != 0 or w % window_size != 0:
        raise ValueError("Height and width must be divisible by window_size.")
    return rearrange(
        x,
        "b (nh ws1) (nw ws2) c -> (b nh nw) (ws1 ws2) c",
        ws1=window_size,
        ws2=window_size,
    )


def windows_to_img(windows: Tensor, h: int, w: int, window_size: int) -> Tensor:
    """Reconstruct a BHWC tensor from windows.

    Args:
        windows: Tensor of shape [B * num_windows, window_size * window_size, C].
        h: Output height.
        w: Output width.
        window_size: Spatial window size.

    Returns:
        Tensor of shape [B, H, W, C].
    """

    num_windows_per_img = (h // window_size) * (w // window_size)
    b = windows.shape[0] // num_windows_per_img
    return rearrange(
        windows,
        "(b nh nw) (ws1 ws2) c -> b (nh ws1) (nw ws2) c",
        b=b,
        nh=h // window_size,
        nw=w // window_size,
        ws1=window_size,
        ws2=window_size,
    )


def get_relative_position_index(window_size: int) -> Tensor:
    """Build a pairwise relative position index for a square window.

    Returns:
        Tensor of shape [window_size * window_size, window_size * window_size].
    """

    coords = torch.stack(
        torch.meshgrid(
            torch.arange(window_size),
            torch.arange(window_size),
            indexing="ij",
        )
    )
    coords_flatten = rearrange(coords, "c h w -> c (h w)")
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = rearrange(relative_coords, "c i j -> i j c")
    relative_coords[:, :, 0] += window_size - 1
    relative_coords[:, :, 1] += window_size - 1
    relative_coords[:, :, 0] *= 2 * window_size - 1
    return relative_coords.sum(-1)

