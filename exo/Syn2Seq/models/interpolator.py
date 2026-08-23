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


class VideoInterpolator(nn.Module):
    def __init__(
        self,
        num_interp_frames: int = 8,
        hidden_dim: int = 64,
        num_res_blocks: int = 3,
    ):
        super().__init__()
        self.num_interp_frames = num_interp_frames
        
        self.in_conv = nn.Conv3d(3, hidden_dim, 3, padding=1)
        
        self.res_blocks = nn.ModuleList([
            ResBlock3D(hidden_dim) for _ in range(num_res_blocks)
        ])
        
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.out_conv = nn.Sequential(
            nn.Conv3d(hidden_dim, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(32, 3, 3, padding=1),
            nn.Tanh(),
        )
    
    def interpolate(
        self,
        start_frame: torch.Tensor,
        end_frame: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = start_frame.shape[0]
        
        x = torch.stack([start_frame, end_frame], dim=2)
        
        x = F.interpolate(
            x,
            size=(self.num_interp_frames + 2,) + x.shape[3:],
            mode="trilinear",
            align_corners=True,
        )
        
        h = self.in_conv(x)
        
        for block in self.res_blocks:
            h = block(h)
        
        out = self.out_conv(h)
        
        out_start = start_frame.unsqueeze(2)
        out_end = end_frame.unsqueeze(2)
        out_middle = out[:, :, 1:-1]
        
        out = torch.cat([out_start, out_middle, out_end], dim=2)
        
        return out
    
    def forward(
        self,
        exo_video: torch.Tensor,
        ego_video: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, channels, num_frames, height, width = exo_video.shape
        
        exo_last = exo_video[:, :, -1]
        ego_first = ego_video[:, :, 0]
        
        interpolated = self.interpolate(exo_last, ego_first)
        interp_frames = interpolated[:, :, 1:-1]
        
        full_sequence = torch.cat([exo_video, interp_frames, ego_video], dim=2)
        
        return full_sequence, interp_frames
