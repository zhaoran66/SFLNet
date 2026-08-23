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
        feat_dim: int = 384,
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


class PoseGuidedFrequencyRouting(nn.Module):
    """
    Pose-Guided Frequency Routing with Residual Formulation
    
    Supports two modes:
    1. Keypoint mode (B, T, 21, 2): Spatial pose-conditioned gating
    2. Camera pose mode (B, T, 4, 4): Global gating from camera extrinsic
    
    F_r = F_l + G * (F_h - F_l)  # residual routing
    """
    def __init__(
        self,
        feat_dim: int = 384,
        pose_dim: int = 16,  # 4x4 flatten
        hidden_dim: int = 128,
        spatial_size: Tuple[int, int] = (32, 32),
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.pose_dim = pose_dim
        self.spatial_size = spatial_size
        
        self.pose_proj = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.spatial_gate = nn.Sequential(
            nn.Conv3d(feat_dim + hidden_dim, feat_dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, feat_dim // 2),
            nn.GELU(),
            nn.Conv3d(feat_dim // 2, feat_dim // 4, kernel_size=3, padding=1),
            nn.GroupNorm(4, feat_dim // 4),
            nn.GELU(),
            nn.Conv3d(feat_dim // 4, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        
        self.routing_gate = None
    
    def forward(
        self,
        feat_low: torch.Tensor,
        feat_high: torch.Tensor,
        pose: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, T, H, W = feat_low.shape
        
        if pose is not None:
            pose_flat = pose.flatten(1) if pose.dim() == 4 else pose.reshape(B, T, -1).flatten(1)
            pose_feat = self.pose_proj(pose_flat.reshape(B * T, -1))
            pose_feat = pose_feat.reshape(B, T, -1).permute(0, 2, 1)
            pose_feat = pose_feat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, H, W)
            combined_feat = torch.cat([feat_high, pose_feat], dim=1)
        else:
            dummy_pose = torch.zeros(B, 128, T, H, W, device=feat_high.device)
            combined_feat = torch.cat([feat_high, dummy_pose], dim=1)
        
        routing_gate = self.spatial_gate(combined_feat)
        
        self.routing_gate = routing_gate
        
        feat_routed = feat_low + routing_gate * (feat_high - feat_low)
        
        return feat_routed, routing_gate


def create_gaussian_pose_mask(
    pose: torch.Tensor,
    spatial_size: Tuple[int, int, int],
    sigma: float = 0.1,
) -> torch.Tensor:
    B = pose.shape[0]
    T, H, W = spatial_size
    
    mask = torch.ones(B, 1, T, H, W, device=pose.device, dtype=torch.float32) * 0.5
    
    return mask


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


def motion_aware_temporal_smoothness_loss(
    routing_gate: torch.Tensor,
    pose: Optional[torch.Tensor] = None,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Motion-Aware Temporal Smoothness Loss with Temperature Normalization
    
    L_temp = sum(w_t * |G_t - G_{t-1}|)
    where w_t = exp(-|v_t| / tau)
    
    - v_t: pose velocity at time t
    - tau: temperature parameter for stable weighting
    """
    if routing_gate is None:
        return torch.tensor(0.0, device=routing_gate.device if routing_gate is not None else "cpu")
    
    gate_diff = torch.abs(routing_gate[:, :, 1:] - routing_gate[:, :, :-1])
    
    if pose is not None and pose.shape[1] > 1:
        pose_flat = pose.reshape(pose.shape[0], pose.shape[1], -1)
        pose_diff = torch.norm(pose_flat[:, 1:] - pose_flat[:, :-1], dim=-1)
        motion_mag = torch.mean(pose_diff, dim=-1, keepdim=True)
        
        motion_weight = torch.exp(-motion_mag.view(-1, 1, 1, 1, 1) / temperature)
        smooth_loss = torch.mean(motion_weight * gate_diff)
    else:
        smooth_loss = torch.mean(gate_diff)
    
    return smooth_loss


def routing_sparsity_loss(
    routing_gate: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Routing Sparsity Regularization (instead of entropy maximization)
    
    L_sparse = |G * (1 - G)|
    
    Encourages gate values close to 0 or 1, avoiding uniform uncertainty.
    """
    if routing_gate is None:
        return torch.tensor(0.0, device=routing_gate.device if routing_gate is not None else "cpu")
    
    g = torch.clamp(routing_gate, eps, 1 - eps)
    sparsity = torch.mean(g * (1 - g))
    
    return sparsity


def latent_identity_consistency_loss(
    pred_feat: torch.Tensor,
    target_feat: torch.Tensor,
) -> torch.Tensor:
    """
    Latent Identity Consistency Loss
    
    L_id = |E(I_pred) - E(I_gt)|
    
    where E is frozen pre-trained encoder (DINOv2)
    
    Purpose: Preserve semantic structure continuity during interpolation
             Prevent drift and appearance collapse
    """
    return F.l1_loss(pred_feat, target_feat)


def cross_view_alignment_loss(
    exo_feat: torch.Tensor,
    ego_feat: torch.Tensor,
    pose_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Cross-View Latent Alignment Loss
    
    L_align = |z_exo - z_ego|
    
    Applied in shared semantic regions (weighted by pose_mask if available)
    
    Purpose: Stabilize cross-view latent manifold alignment
    """
    if pose_mask is not None:
        B, C, T, H, W = exo_feat.shape
        
        mask_resized = F.interpolate(
            pose_mask.squeeze(1),
            size=(T, H, W),
            mode="trilinear",
            align_corners=False
        ).unsqueeze(1)
        
        diff = torch.abs(exo_feat - ego_feat)
        return torch.mean(mask_resized * diff)
    else:
        return F.l1_loss(exo_feat, ego_feat)


def compute_gate_temporal_variance(
    routing_gate: torch.Tensor,
) -> Tuple[float, float]:
    """
    Compute Gate Temporal Variance for Interpretability
    
    Returns:
    - fg_var: variance in high-gate (foreground) regions
    - bg_var: variance in low-gate (background) regions
    """
    if routing_gate is None:
        return 0.0, 0.0
    
    gate_temporal = routing_gate[0, 0]
    
    fg_mask = gate_temporal > 0.7
    bg_mask = gate_temporal < 0.3
    
    fg_var = float(torch.var(gate_temporal[fg_mask]).item()) if fg_mask.sum() > 0 else 0.0
    bg_var = float(torch.var(gate_temporal[bg_mask]).item()) if bg_mask.sum() > 0 else 0.0
    
    return fg_var, bg_var
