"""
Simplified Hand Pose Estimation Models (no BatchNorm!)

Four variants:
1. Baseline (Simple CNN)
2. Latent-only (SD-VAE + MLP)
3. Ours (DINO + Frequency Decomposition)
4. Method C (SD-VAE + Frequency Decomposition)

All models use LayerNorm / GroupNorm only!
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class HandPoseBaseline(nn.Module):
    """
    Baseline: Simple CNN Regressor - NO BATCHNORM!
    Input: Exo view video (B, 3, T, 128, 128)
    Output: Hand pose parameters (B*T, 51)
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_frames: int = 8,
        hidden_dim: int = 256,
        num_pose_params: int = 51,
    ):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(),
        )
        
        with torch.no_grad():
            dummy = torch.randn(1, in_channels, num_frames, 128, 128)
            out = self.conv_layers(dummy)
            self.feature_dim = out.shape[1] * out.shape[3] * out.shape[4]
        
        self.regressor = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_pose_params),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        feat = self.conv_layers(x)
        feat = feat.permute(0, 2, 1, 3, 4).reshape(B * T, -1)
        pose = self.regressor(feat)
        return pose


class HandPoseLatentOnly(nn.Module):
    """
    Latent-only: SD-VAE Latent + MLP Regressor - NO BATCHNORM!
    Input: Exo view video latent (B, 4, T, 16, 16)
    Output: Hand pose parameters (B*T, 51)
    """
    def __init__(
        self,
        in_channels: int = 4,
        num_frames: int = 8,
        hidden_dim: int = 512,
        num_pose_params: int = 51,
    ):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv3d(4, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(),
        )
        
        with torch.no_grad():
            dummy = torch.randn(1, 4, num_frames, 16, 16)
            out = self.conv_layers(dummy)
            self.feature_dim = out.shape[1] * out.shape[3] * out.shape[4]
        
        self.regressor = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_pose_params),
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = z.shape
        feat = self.conv_layers(z)
        feat = feat.permute(0, 2, 1, 3, 4).reshape(B * T, -1)
        pose = self.regressor(feat)
        return pose


class SoftSpectralDecomposition(nn.Module):
    """
    Soft Spectral Decomposition (DCT + Gaussian masks)
    """
    def __init__(
        self,
        freq_size: Tuple[int, int] = (8, 8),
        feat_dim: int = 384,
        sigma: float = 0.5,
    ):
        super().__init__()
        self.freq_size = freq_size
        
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
    
    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, T, H, W = feat.shape
        
        feat_reshaped = feat.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        feat_resized = F.interpolate(feat_reshaped, size=self.freq_size, mode="bilinear", align_corners=False)
        
        feat_fft = torch.fft.fft2(feat_resized, norm="ortho")
        feat_fft_shifted = torch.fft.fftshift(feat_fft, dim=(-2, -1))
        
        low_mask = self.low_mask.view(1, 1, *self.freq_size)
        high_mask = self.high_mask.view(1, 1, *self.freq_size)
        
        feat_low_fft = feat_fft_shifted * low_mask
        feat_high_fft = feat_fft_shifted * high_mask
        
        feat_low = torch.fft.ifft2(torch.fft.ifftshift(feat_low_fft, dim=(-2, -1)), norm="ortho").real
        feat_high = torch.fft.ifft2(torch.fft.ifftshift(feat_high_fft, dim=(-2, -1)), norm="ortho").real
        
        feat_low = F.interpolate(feat_low, size=(H, W), mode="bilinear", align_corners=False)
        feat_high = F.interpolate(feat_high, size=(H, W), mode="bilinear", align_corners=False)
        
        feat_low = feat_low.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)
        feat_high = feat_high.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)
        
        return feat_low, feat_high


class HandPoseOurs(nn.Module):
    """
    Ours: DINO Feature + Frequency Decomposition + Pose Regressor - NO BATCHNORM!
    Input: Exo view video features (B, 384, T, 32, 32)
    Output: Hand pose parameters (B*T, 51)
    """
    def __init__(
        self,
        feat_dim: int = 384,
        num_frames: int = 8,
        hidden_dim: int = 512,
        num_pose_params: int = 51,
        use_freq_decomp: bool = True,
    ):
        super().__init__()
        self.use_freq_decomp = use_freq_decomp
        
        if use_freq_decomp:
            self.spectral_decomp = SoftSpectralDecomposition(
                freq_size=(8, 8),
                feat_dim=feat_dim,
                sigma=0.5,
            )
        
        self.conv_layers = nn.Sequential(
            nn.Conv3d(feat_dim, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(),
        )
        
        with torch.no_grad():
            dummy = torch.randn(1, feat_dim, num_frames, 32, 32)
            out = self.conv_layers(dummy)
            self.feature_dim = out.shape[1] * out.shape[3] * out.shape[4]
        
        regressor_input_dim = self.feature_dim * (2 if use_freq_decomp else 1)
        
        self.regressor = nn.Sequential(
            nn.Linear(regressor_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_pose_params),
        )
    
    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = feat.shape
        
        if self.use_freq_decomp:
            feat_low, feat_high = self.spectral_decomp(feat)
            
            feat_low_down = self.conv_layers(feat_low)
            feat_high_down = self.conv_layers(feat_high)
            
            feat_low_flat = feat_low_down.permute(0, 2, 1, 3, 4).reshape(B * T, -1)
            feat_high_flat = feat_high_down.permute(0, 2, 1, 3, 4).reshape(B * T, -1)
            
            feat_combined = torch.cat([feat_low_flat, feat_high_flat], dim=1)
            pose = self.regressor(feat_combined)
        else:
            feat_down = self.conv_layers(feat)
            feat_flat = feat_down.permute(0, 2, 1, 3, 4).reshape(B * T, -1)
            pose = self.regressor(feat_flat)
        
        return pose


class HandPoseMethodC(nn.Module):
    """
    Method C: SD-VAE Latent + Frequency Decomposition + Pose Regressor - NO BATCHNORM!
    Input: Exo view video latent (B, 4, T, 16, 16)
    Output: Hand pose parameters (B*T, 51)
    """
    def __init__(
        self,
        latent_dim: int = 4,
        num_frames: int = 8,
        hidden_dim: int = 512,
        num_pose_params: int = 51,
        use_freq_decomp: bool = True,
    ):
        super().__init__()
        self.use_freq_decomp = use_freq_decomp
        
        if use_freq_decomp:
            self.spectral_decomp = SoftSpectralDecomposition(
                freq_size=(8, 8),
                feat_dim=latent_dim,
                sigma=0.5,
            )
        
        self.conv_layers = nn.Sequential(
            nn.Conv3d(latent_dim, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(),
        )
        
        with torch.no_grad():
            dummy = torch.randn(1, latent_dim, num_frames, 16, 16)
            out = self.conv_layers(dummy)
            self.feature_dim = out.shape[1] * out.shape[3] * out.shape[4]
        
        regressor_input_dim = self.feature_dim * (2 if use_freq_decomp else 1)
        
        self.regressor = nn.Sequential(
            nn.Linear(regressor_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_pose_params),
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = z.shape
        
        if self.use_freq_decomp:
            z_low, z_high = self.spectral_decomp(z)
            
            z_low_down = self.conv_layers(z_low)
            z_high_down = self.conv_layers(z_high)
            
            z_low_flat = z_low_down.permute(0, 2, 1, 3, 4).reshape(B * T, -1)
            z_high_flat = z_high_down.permute(0, 2, 1, 3, 4).reshape(B * T, -1)
            
            z_combined = torch.cat([z_low_flat, z_high_flat], dim=1)
            pose = self.regressor(z_combined)
        else:
            z_down = self.conv_layers(z)
            z_flat = z_down.permute(0, 2, 1, 3, 4).reshape(B * T, -1)
            pose = self.regressor(z_flat)
        
        return pose
