import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class SoftSpectralDecomposition(nn.Module):
    """
    Soft Spectral Decomposition using Gaussian basis functions
    
    F_l = S_l(F)  # low-frequency component
    F_h = S_h(F)  # high-frequency component
    
    where S_l and S_h are Gaussian spectral masks
    """
    def __init__(
        self,
        freq_size: Tuple[int, int] = (16, 16),
        feat_dim: int = 3,
        sigma: float = 0.5,
        use_dct: bool = True,
    ):
        super().__init__()
        self.freq_size = freq_size
        self.use_dct = use_dct
        
        h, w = freq_size
        
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        center_y, center_x = h // 2, w // 2
        
        dist = torch.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
        max_dist = torch.max(dist)
        dist_normalized = dist / max_dist
        
        low_mask = torch.exp(-(dist_normalized ** 2) / (2 * sigma ** 2))
        high_mask = 1.0 - low_mask
        
        self.register_buffer("low_mask", low_mask)
        self.register_buffer("high_mask", high_mask)
        
        if use_dct:
            dct_basis = self._create_dct_basis(h, w)
            self.register_buffer("dct_basis", dct_basis)
    
    def _create_dct_basis(self, h: int, w: int) -> torch.Tensor:
        def dct_coeff(n: int, k: int, N: int) -> float:
            if k == 0:
                return math.sqrt(1.0 / N)
            else:
                return math.sqrt(2.0 / N) * math.cos(math.pi * k * (2 * n + 1) / (2 * N))
        
        basis = torch.zeros(h, w, h, w)
        for k1 in range(h):
            for k2 in range(w):
                for n1 in range(h):
                    for n2 in range(w):
                        basis[k1, k2, n1, n2] = dct_coeff(n1, k1, h) * dct_coeff(n2, k2, w)
        
        return basis
    
    def _apply_dct(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h, w = self.freq_size
        
        x_reshaped = x.reshape(B * C, 1, H, W)
        x_resized = F.interpolate(x_reshaped, size=(h, w), mode="bilinear", align_corners=False)
        
        dct_basis = self.dct_basis.view(h * w, h * w)
        x_flat = x_resized.view(B * C, h * w)
        x_dct = x_flat @ dct_basis.T
        
        return x_dct.view(B, C, h, w)
    
    def _apply_idct(self, x_dct: torch.Tensor, orig_size: Tuple[int, int]) -> torch.Tensor:
        B, C, H, W = x_dct.shape
        h, w = orig_size
        
        dct_basis = self.dct_basis.view(H * W, H * W)
        x_flat = x_dct.view(B * C, H * W)
        x_idct = x_flat @ dct_basis
        
        x_idct = x_idct.view(B * C, 1, H, W)
        x_out = F.interpolate(x_idct, size=(h, w), mode="bilinear", align_corners=False)
        
        return x_out.view(B, C, h, w)
    
    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, T, H, W = feat.shape
        
        feat_reshaped = feat.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        
        if self.use_dct:
            feat_dct = self._apply_dct(feat_reshaped)
            feat_low = feat_dct * self.low_mask.view(1, 1, *self.freq_size)
            feat_high = feat_dct * self.high_mask.view(1, 1, *self.freq_size)
            
            feat_low = self._apply_idct(feat_low, (H, W))
            feat_high = self._apply_idct(feat_high, (H, W))
        else:
            feat_resized = F.interpolate(feat_reshaped, size=self.freq_size, mode="bilinear", align_corners=False)
            
            feat_fft = torch.fft.rfft2(feat_resized, norm="ortho")
            feat_fft_shifted = torch.fft.fftshift(feat_fft, dim=(-2, -1))
            
            low_mask = self.low_mask.view(1, 1, *self.freq_size)
            high_mask = self.high_mask.view(1, 1, *self.freq_size)
            
            feat_low_fft = feat_fft_shifted * low_mask
            feat_high_fft = feat_fft_shifted * high_mask
            
            feat_low = torch.fft.irfft2(torch.fft.ifftshift(feat_low_fft, dim=(-2, -1)), norm="ortho")
            feat_high = torch.fft.irfft2(torch.fft.ifftshift(feat_high_fft, dim=(-2, -1)), norm="ortho")
            
            feat_low = F.interpolate(feat_low, size=(H, W), mode="bilinear", align_corners=False)
            feat_high = F.interpolate(feat_high, size=(H, W), mode="bilinear", align_corners=False)
        
        feat_low = feat_low.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)
        feat_high = feat_high.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)
        
        return feat_low, feat_high


def create_gaussian_pose_mask(
    pose: torch.Tensor,
    spatial_size: Tuple[int, int, int],
    sigma: float = 0.1,
) -> torch.Tensor:
    """
    Create Gaussian mask from pose (camera extrinsic or keypoints)
    
    Args:
        pose: (B, T, 4, 4) camera extrinsic or (B, T, 21, 2) keypoints
        spatial_size: (T, H, W) spatial dimensions
        sigma: Gaussian standard deviation
    
    Returns:
        mask: (B, 1, T, H, W) Gaussian mask
    """
    B, T_ = pose.shape[0], spatial_size[0]
    H, W = spatial_size[1], spatial_size[2]
    
    if pose.shape[2] == 4 and pose.shape[3] == 4:
        center = pose[:, :, :2, 3]
        center = (center + 1.0) / 2.0
        center = torch.clamp(center, 0.0, 1.0)
        
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=pose.device, dtype=torch.float32),
            torch.arange(W, device=pose.device, dtype=torch.float32),
            indexing="ij",
        )
        
        mask = torch.zeros(B, T_, H, W, device=pose.device, dtype=torch.float32)
        
        for b in range(B):
            for t in range(min(T_, center.shape[1])):
                cy, cx = center[b, t, 0], center[b, t, 1]
                dist = torch.sqrt((y_grid - cy) ** 2 + (x_grid - cx) ** 2)
                mask[b, t] = torch.exp(-dist ** 2 / (2 * sigma ** 2))
    else:
        mask = torch.ones(B, 1, T_, H, W, device=pose.device, dtype=torch.float32) * 0.5
    
    return mask.unsqueeze(1)


def structure_weighted_asymmetric_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pose_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Structure-Weighted Asymmetric Loss
    
    L_struct = lambda_fg * M * L_fg + lambda_bg * (1 - M) * L_bg
    
    where:
    - M: Gaussian pose mask
    - L_fg: foreground loss (hand dynamics)
    - L_bg: background loss (scene consistency)
    """
    diff = (pred - target) ** 2
    
    fg_loss = torch.mean(pose_mask * diff)
    bg_loss = torch.mean((1 - pose_mask) * diff)
    
    total_loss = fg_loss + bg_loss
    
    return total_loss, fg_loss, bg_loss
