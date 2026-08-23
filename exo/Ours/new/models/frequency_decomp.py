import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SemanticFrequencyDecomposition(nn.Module):
    def __init__(
        self,
        feat_dim: int = 384,
        fft_size: Tuple[int, int] = (16, 16),
        low_freq_ratio: float = 0.25,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.fft_size = fft_size
        self.low_freq_ratio = low_freq_ratio
        
        h, w = fft_size
        rfft_w = w // 2 + 1
        
        low_h = int(h * low_freq_ratio)
        low_w = int(rfft_w * low_freq_ratio)
        
        low_freq_mask = self._create_circular_mask(h, rfft_w, low_h, low_w)
        self.register_buffer("low_freq_mask", low_freq_mask)
        self.register_buffer("high_freq_mask", 1.0 - low_freq_mask)
        
        self.mask_mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.GELU(),
            nn.Linear(feat_dim // 2, 1),
            nn.Sigmoid(),
        )
    
    def _create_circular_mask(self, h: int, w: int, low_h: int, low_w: int) -> torch.Tensor:
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        center_y, center_x = h // 2, w // 2
        max_radius = min(low_h, low_w)
        dist = torch.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
        mask = (dist <= max_radius).float()
        return mask
    
    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = feat.shape
        
        orig_dtype = feat.dtype
        feat = feat.float()
        
        feat_resized = F.interpolate(feat, size=self.fft_size, mode="bilinear", align_corners=False)
        
        feat_fft = torch.fft.rfft2(feat_resized, norm="ortho")
        feat_fft_shifted = torch.fft.fftshift(feat_fft, dim=(-2, -1))
        
        low_mask = self.low_freq_mask.view(1, 1, *self.fft_size[:-1], -1)
        high_mask = self.high_freq_mask.view(1, 1, *self.fft_size[:-1], -1)
        
        feat_low_fft = feat_fft_shifted * low_mask
        feat_high_fft = feat_fft_shifted * high_mask
        
        feat_low = torch.fft.irfft2(torch.fft.ifftshift(feat_low_fft, dim=(-2, -1)), norm="ortho")
        feat_high = torch.fft.irfft2(torch.fft.ifftshift(feat_high_fft, dim=(-2, -1)), norm="ortho")
        
        feat_low = F.interpolate(feat_low, size=(H, W), mode="bilinear", align_corners=False)
        feat_high = F.interpolate(feat_high, size=(H, W), mode="bilinear", align_corners=False)
        
        feat_low = feat_low.to(orig_dtype)
        feat_high = feat_high.to(orig_dtype)
        
        return feat_low, feat_high


class BackgroundDynamicsInterpolator(nn.Module):
    def __init__(self, feat_dim: int = 384):
        super().__init__()
        self.feat_dim = feat_dim
        
        self.pose_encoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.GELU(),
            nn.Linear(64, feat_dim),
        )
        
        self.interp_net = nn.Sequential(
            nn.Linear(feat_dim * 3, feat_dim * 2),
            nn.GELU(),
            nn.Linear(feat_dim * 2, feat_dim),
        )
    
    def slerp_pose(self, pose_start: torch.Tensor, pose_end: torch.Tensor, t: float) -> torch.Tensor:
        return (1 - t) * pose_start + t * pose_end
    
    def forward(
        self,
        feat_low_start: torch.Tensor,
        feat_low_end: torch.Tensor,
        pose_start: torch.Tensor,
        pose_end: torch.Tensor,
        num_steps: int,
    ) -> torch.Tensor:
        B, C, H, W = feat_low_start.shape
        
        interp_features = []
        
        for step in range(num_steps + 2):
            t = step / (num_steps + 1)
            
            pose_t = self.slerp_pose(pose_start.flatten(1), pose_end.flatten(1), t)
            pose_feat = self.pose_encoder(pose_t)
            
            feat_t = (1 - t) * feat_low_start + t * feat_low_end
            
            pose_feat_spatial = pose_feat.view(B, C, 1, 1).expand(-1, -1, H, W)
            
            net_input = torch.cat([feat_t, pose_feat_spatial], dim=1)
            net_input_flat = net_input.permute(0, 2, 3, 1).reshape(-1, C * 2)
            net_input_flat = torch.cat([net_input_flat, pose_feat.repeat_interleave(H * W, dim=0)], dim=1)
            
            residual = self.interp_net(net_input_flat).view(B, H, W, C).permute(0, 3, 1, 2)
            feat_t = feat_t + residual * 0.1
            
            interp_features.append(feat_t)
        
        return torch.stack(interp_features, dim=2)


class FrequencyRouters(nn.Module):
    def __init__(self, feat_dim: int = 384):
        super().__init__()
        self.routing_mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.GELU(),
            nn.Linear(feat_dim // 2, 2),
            nn.Softmax(dim=-1),
        )
    
    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = feat.shape
        
        feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, C)
        
        weights = self.routing_mlp(feat_flat)
        weights = weights.view(B, H, W, 2).permute(0, 3, 1, 2)
        
        w_low, w_high = weights[:, 0], weights[:, 1]
        
        return w_low.unsqueeze(1), w_high.unsqueeze(1)


def orthogonal_loss(feat_low: torch.Tensor, feat_high: torch.Tensor) -> torch.Tensor:
    B, C, H, W = feat_low.shape
    
    feat_low_flat = feat_low.reshape(B, C, -1)
    feat_high_flat = feat_high.reshape(B, C, -1)
    
    feat_low_norm = F.normalize(feat_low_flat, dim=1)
    feat_high_norm = F.normalize(feat_high_flat, dim=1)
    
    dot_product = torch.sum(feat_low_norm * feat_high_norm, dim=1)
    ortho_loss = torch.mean(torch.abs(dot_product))
    
    return ortho_loss
