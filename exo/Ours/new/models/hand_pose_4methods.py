"""
Four Hand Pose Estimation Methods - Ablation Study

1. Baseline:       Pixel space, no decomposition
2. Ours:           Pixel space + frequency decomposition
3. Latent-only:    SD-VAE latent, no decomposition
4. Method C:       SD-VAE latent + frequency decomposition

All use the same regression head for fair comparison
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SoftSpectralDecomposition(nn.Module):
    """
    Soft Spectral Decomposition (FFT + Gaussian masks)
    Splits features into low-frequency and high-frequency components
    """
    def __init__(
        self,
        freq_size: Tuple[int, int] = (16, 16),
        feat_dim: int = 3,
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


class PixelEncoder(nn.Module):
    """
    Pixel space encoder (RGB images)
    Input: (B, 3, T, 128, 128)
    Output: (B, C, T, 8, 8)  downsampled features
    """
    def __init__(self, in_channels=3, hidden_dim=128):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim // 2, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.ReLU(),
            nn.Conv3d(hidden_dim // 2, hidden_dim, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(hidden_dim, hidden_dim * 2, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(hidden_dim * 2, hidden_dim * 2, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(),
        )
        
        with torch.no_grad():
            dummy = torch.randn(1, in_channels, 8, 128, 128)
            out = self.conv_layers(dummy)
            self.out_dim = out.shape[1]
            self.spatial_dim = out.shape[3] * out.shape[4]
    
    def forward(self, x):
        return self.conv_layers(x)


class DinoFeatureEncoder(nn.Module):
    """
    DINO feature encoder for high-dimensional features
    Input: (B, feat_dim, T, 32, 32)
    Output: (B, C, T, 8, 8)  downsampled features
    """
    def __init__(self, in_channels=384, hidden_dim=128):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(hidden_dim, hidden_dim * 2, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(),
        )
        
        with torch.no_grad():
            dummy = torch.randn(1, in_channels, 8, 32, 32)
            out = self.conv_layers(dummy)
            self.out_dim = out.shape[1]
            self.spatial_dim = out.shape[3] * out.shape[4]
    
    def forward(self, x):
        return self.conv_layers(x)


class LatentEncoder(nn.Module):
    """
    SD-VAE latent encoder
    Input: (B, 4, T, 16, 16)
    Output: (B, C, T, 2, 2)  downsampled features
    """
    def __init__(self, in_channels=4, hidden_dim=256):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim // 4, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(hidden_dim // 4, hidden_dim // 2, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(hidden_dim // 2, hidden_dim, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(),
        )
        
        with torch.no_grad():
            dummy = torch.randn(1, in_channels, 8, 16, 16)
            out = self.conv_layers(dummy)
            self.out_dim = out.shape[1]
            self.spatial_dim = out.shape[3] * out.shape[4]
    
    def forward(self, z):
        return self.conv_layers(z)


class RegressorHead(nn.Module):
    """
    Shared regression head for all methods
    Input: (B*T, feature_dim)
    Output: (B*T, 51) MANO parameters
    """
    def __init__(self, input_dim, hidden_dim=512, num_pose_params=51):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_pose_params),
        )
    
    def forward(self, x):
        return self.regressor(x)


class HandPoseBaseline(nn.Module):
    """
    1. Baseline: Pixel space, no frequency decomposition
    Input: (B, 3, T, 128, 128)
    Output: (B*T, 51)
    """
    def __init__(self, in_channels=3, num_frames=8, hidden_dim=128, num_pose_params=51, **kwargs):
        super().__init__()
        self.encoder = PixelEncoder(in_channels=in_channels, hidden_dim=hidden_dim)
        
        regressor_input_dim = self.encoder.out_dim * self.encoder.spatial_dim
        self.regressor = RegressorHead(regressor_input_dim, hidden_dim=512, num_pose_params=num_pose_params)
    
    def forward(self, x):
        B, C, T, H, W = x.shape
        feat = self.encoder(x)
        feat_flat = feat.permute(0, 2, 1, 3, 4).reshape(B * T, -1)
        pose = self.regressor(feat_flat)
        return pose


class HandPoseOurs(nn.Module):
    """
    2. Ours: DINO feature + frequency decomposition
    Input: (B, feat_dim, T, 32, 32)  DINO features
    Output: (B*T, 51)
    """
    def __init__(self, feat_dim=384, num_frames=8, hidden_dim=256, num_pose_params=51, use_freq_decomp=True, **kwargs):
        super().__init__()
        self.use_freq_decomp = use_freq_decomp
        self.spectral_decomp = SoftSpectralDecomposition(
            freq_size=(8, 8),
            feat_dim=feat_dim,
            sigma=0.5,
        )
        self.encoder_low = DinoFeatureEncoder(in_channels=feat_dim, hidden_dim=hidden_dim)
        self.encoder_high = DinoFeatureEncoder(in_channels=feat_dim, hidden_dim=hidden_dim)
        
        regressor_input_dim = self.encoder_low.out_dim * self.encoder_low.spatial_dim * 2
        self.regressor = RegressorHead(regressor_input_dim, hidden_dim=512, num_pose_params=num_pose_params)
    
    def forward(self, x):
        B, C, T, H, W = x.shape
        feat_low, feat_high = self.spectral_decomp(x)
        
        feat_low_down = self.encoder_low(feat_low)
        feat_high_down = self.encoder_high(feat_high)
        
        feat_low_flat = feat_low_down.permute(0, 2, 1, 3, 4).reshape(B * T, -1)
        feat_high_flat = feat_high_down.permute(0, 2, 1, 3, 4).reshape(B * T, -1)
        
        feat_combined = torch.cat([feat_low_flat, feat_high_flat], dim=1)
        pose = self.regressor(feat_combined)
        return pose


class HandPoseLatentOnly(nn.Module):
    """
    3. Latent-only: SD-VAE latent, no frequency decomposition
    Input: (B, 4, T, 16, 16)
    Output: (B*T, 51)
    """
    def __init__(self, in_channels=4, num_frames=8, hidden_dim=256, num_pose_params=51, **kwargs):
        super().__init__()
        self.encoder = LatentEncoder(in_channels=in_channels, hidden_dim=hidden_dim)
        
        regressor_input_dim = self.encoder.out_dim * self.encoder.spatial_dim
        self.regressor = RegressorHead(regressor_input_dim, hidden_dim=512, num_pose_params=num_pose_params)
    
    def forward(self, z):
        B, C, T, H, W = z.shape
        feat = self.encoder(z)
        feat_flat = feat.permute(0, 2, 1, 3, 4).reshape(B * T, -1)
        pose = self.regressor(feat_flat)
        return pose


class HandPoseMethodC(nn.Module):
    """
    4. Method C: SD-VAE latent + frequency decomposition
    Input: (B, 4, T, 16, 16)
    Output: (B*T, 51)
    """
    def __init__(self, latent_dim=4, num_frames=8, hidden_dim=256, num_pose_params=51, use_freq_decomp=True, **kwargs):
        super().__init__()
        self.use_freq_decomp = use_freq_decomp
        self.spectral_decomp = SoftSpectralDecomposition(
            freq_size=(8, 8),
            feat_dim=latent_dim,
            sigma=0.5,
        )
        self.encoder_low = LatentEncoder(in_channels=latent_dim, hidden_dim=hidden_dim)
        self.encoder_high = LatentEncoder(in_channels=latent_dim, hidden_dim=hidden_dim)
        
        regressor_input_dim = self.encoder_low.out_dim * self.encoder_low.spatial_dim * 2
        self.regressor = RegressorHead(regressor_input_dim, hidden_dim=512, num_pose_params=num_pose_params)
    
    def forward(self, z):
        B, C, T, H, W = z.shape
        z_low, z_high = self.spectral_decomp(z)
        
        z_low_down = self.encoder_low(z_low)
        z_high_down = self.encoder_high(z_high)
        
        z_low_flat = z_low_down.permute(0, 2, 1, 3, 4).reshape(B * T, -1)
        z_high_flat = z_high_down.permute(0, 2, 1, 3, 4).reshape(B * T, -1)
        
        z_combined = torch.cat([z_low_flat, z_high_flat], dim=1)
        pose = self.regressor(z_combined)
        return pose
