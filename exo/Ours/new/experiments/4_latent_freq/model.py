"""
Experiment 4: Baseline + Latent + Frequency Decomposition
Input:  (B, 3, T, 128, 128)  RGB video
Output: (B*T, 51)              MANO hand pose

Pipeline: RGB -> VAE -> Freq-split -> LatentEncoder(x2) -> RegressorHead
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import sys
import os

sys.path.insert(0, '/data/data5/zhaoran/paper_code/exo/latent')


def load_vae(vae_path="/data/data5/zhaoran/paper_code/exo/latent/outputs/vae/checkpoints/best_model.pt"):
    from models.method_c_vae import FrameVAE
    
    vae = FrameVAE()
    
    if os.path.exists(vae_path):
        checkpoint = torch.load(vae_path, map_location='cpu')
        vae.load_state_dict(checkpoint['model_state_dict'])
    return vae


class SoftSpectralDecomposition(nn.Module):
    def __init__(self, freq_size: Tuple[int, int] = (8, 8), feat_dim: int = 4, sigma: float = 0.5):
        super().__init__()
        self.freq_size = freq_size
        
        h, w = freq_size
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        center_y, center_x = h // 2, w // 2
        
        dist = torch.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
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


class LatentEncoder(nn.Module):
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


class HandPoseLatentFreq(nn.Module):
    def __init__(self, in_channels=3, num_frames=8, hidden_dim=256, num_pose_params=51,
                 vae_path="/data/data5/zhaoran/paper_code/exo/latent/outputs/vae/checkpoints/best_model.pt",
                 use_freq_decomp=True):
        super().__init__()
        self.use_freq_decomp = use_freq_decomp
        self.vae = load_vae(vae_path)
        
        for param in self.vae.parameters():
            param.requires_grad = False
        
        self.spectral_decomp = SoftSpectralDecomposition(
            freq_size=(8, 8),
            feat_dim=4,
            sigma=0.5,
        )
        
        self.encoder_low = LatentEncoder(in_channels=4, hidden_dim=hidden_dim)
        self.encoder_high = LatentEncoder(in_channels=4, hidden_dim=hidden_dim)
        
        regressor_input_dim = self.encoder_low.out_dim * self.encoder_low.spatial_dim * 2
        self.regressor = RegressorHead(regressor_input_dim, hidden_dim=512, num_pose_params=num_pose_params)
    
    def train(self, mode=True):
        super().train(mode)
        self.vae.eval()
        return self
    
    def forward(self, x):
        B, C, T, H, W = x.shape
        
        with torch.no_grad():
            z = self.vae.encode(x)
        
        z_low, z_high = self.spectral_decomp(z)
        
        z_low_down = self.encoder_low(z_low)
        z_high_down = self.encoder_high(z_high)
        
        z_low_flat = z_low_down.permute(0, 2, 1, 3, 4).reshape(B * T, -1)
        z_high_flat = z_high_down.permute(0, 2, 1, 3, 4).reshape(B * T, -1)
        
        z_combined = torch.cat([z_low_flat, z_high_flat], dim=1)
        pose = self.regressor(z_combined)
        return pose
