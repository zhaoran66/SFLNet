"""
Latent-space VideoInterpolator.

Operates entirely on VAE latents (B, C_z, T, H/8, W/8). Mirrors the design
of the pixel-space interpolator in Syn2Seq but with arbitrary channel
dimension (defaults to 4 for SD VAE) and smaller spatial extent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ResBlock3D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv1 = nn.Conv3d(dim, dim, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, dim)
        self.conv2 = nn.Conv3d(dim, dim, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, dim)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x + residual


class LatentVideoInterpolator(nn.Module):
    def __init__(
        self,
        latent_channels: int = 4,
        num_interp_frames: int = 4,
        hidden_dim: int = 64,
        num_res_blocks: int = 3,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.num_interp_frames = num_interp_frames

        self.in_conv = nn.Conv3d(latent_channels, hidden_dim, 3, padding=1)

        self.res_blocks = nn.ModuleList(
            [ResBlock3D(hidden_dim) for _ in range(num_res_blocks)]
        )

        self.out_conv = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden_dim // 2, latent_channels, 3, padding=1),
        )

    def interpolate(
        self,
        start_latent: torch.Tensor,
        end_latent: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.stack([start_latent, end_latent], dim=2)
        x = F.interpolate(
            x,
            size=(self.num_interp_frames + 2,) + x.shape[3:],
            mode="trilinear",
            align_corners=True,
        )

        h = self.in_conv(x)
        for block in self.res_blocks:
            h = block(h)
        delta = self.out_conv(h)
        out = x + delta

        out_start = start_latent.unsqueeze(2)
        out_end = end_latent.unsqueeze(2)
        out_middle = out[:, :, 1:-1]
        out = torch.cat([out_start, out_middle, out_end], dim=2)
        return out

    def forward(
        self,
        exo_latent: torch.Tensor,
        ego_latent: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        exo_last = exo_latent[:, :, -1]
        ego_first = ego_latent[:, :, 0]

        interpolated = self.interpolate(exo_last, ego_first)
        interp_frames = interpolated[:, :, 1:-1]

        full_sequence = torch.cat([exo_latent, interp_frames, ego_latent], dim=2)
        return full_sequence, interp_frames


class LatentVideoInterpolatorV2(nn.Module):
    """
    Improved version with multi-scale residual connections and more capacity.
    """
    def __init__(
        self,
        latent_channels: int = 4,
        num_interp_frames: int = 4,
        hidden_dim: int = 128,
        num_res_blocks: int = 5,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.num_interp_frames = num_interp_frames

        self.in_conv = nn.Conv3d(latent_channels, hidden_dim, 3, padding=1)

        self.res_blocks = nn.ModuleList([
            ResBlock3D(hidden_dim) for _ in range(num_res_blocks)
        ])

        self.skip_fusion = nn.Conv3d(hidden_dim * 2, hidden_dim, kernel_size=1)

        self.out_conv = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.GroupNorm(4, hidden_dim // 2),
            nn.SiLU(),
            nn.Conv3d(hidden_dim // 2, latent_channels, 3, padding=1),
        )

    def interpolate(
        self,
        start_latent: torch.Tensor,
        end_latent: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.stack([start_latent, end_latent], dim=2)
        x = F.interpolate(
            x,
            size=(self.num_interp_frames + 2,) + x.shape[3:],
            mode="trilinear",
            align_corners=True,
        )

        h = self.in_conv(x)
        skip = h

        for block in self.res_blocks:
            h = block(h)

        h = self.skip_fusion(torch.cat([h, skip], dim=1))
        delta = self.out_conv(h)
        out = x + delta

        out_start = start_latent.unsqueeze(2)
        out_end = end_latent.unsqueeze(2)
        out_middle = out[:, :, 1:-1]
        out = torch.cat([out_start, out_middle, out_end], dim=2)
        return out

    def forward(
        self,
        exo_latent: torch.Tensor,
        ego_latent: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        exo_last = exo_latent[:, :, -1]
        ego_first = ego_latent[:, :, 0]

        interpolated = self.interpolate(exo_last, ego_first)
        interp_frames = interpolated[:, :, 1:-1]

        full_sequence = torch.cat([exo_latent, interp_frames, ego_latent], dim=2)
        return full_sequence, interp_frames
