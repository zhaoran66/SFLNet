"""
Hand Pose Estimation - 4 Unified Methods
All methods have SAME external interface:
  Input:  (B, 3, T, 128, 128) - RGB video
  Output: (B*T, 51) - MANO hand pose parameters

Internal differences:
1. Baseline:    Direct RGB -> Encoder -> Regressor
2. Freq-Decomp: RGB -> DINO -> Freq-split -> Regressor
3. Latent-Only: RGB -> VAE -> Encoder -> Regressor
4. Latent-Freq: RGB -> VAE -> Freq-split -> Regressor
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import sys
import os
import json


class SoftSpectralDecomposition(nn.Module):
    """
    Soft Spectral Decomposition (FFT + Gaussian masks)
    Splits features into low-frequency and high-frequency components
    """
    def __init__(
        self,
        freq_size: Tuple[int, int] = (8, 8),
        feat_dim: int = 3,
        sigma: float = 0.5,
    ):
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
    Input: (B, 384, T, 32, 32)
    Output: (B, C, T, 8, 8)  downsampled features
    """
    def __init__(self, in_channels=384, hidden_dim=256):
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


def load_local_dinov2():
    dinov2_code_path = "/data/data5/zhaoran/paper_code/spl/Dinov2"
    dinov2_model_path = "/data/data5/zhaoran/paper_code/spl/model/Dinov2/pytorch_model.bin"
    dinov2_config_path = "/data/data5/zhaoran/paper_code/spl/model/Dinov2/config.json"
    
    if not os.path.exists(dinov2_model_path):
        print(f"Warning: DINOv2 model not found at {dinov2_model_path}")
        return None
    
    if dinov2_code_path not in sys.path:
        sys.path.insert(0, dinov2_code_path)
    
    try:
        with open(dinov2_config_path, 'r') as f:
            config = json.load(f)
        
        from dinov2.models.vision_transformer import DinoVisionTransformer
        
        embed_dim = config.get("hidden_size", 384)
        depth = config.get("num_hidden_layers", 12)
        num_heads = config.get("num_attention_heads", 6)
        patch_size = config.get("patch_size", 14)
        
        model = DinoVisionTransformer(
            img_size=224,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=4.0,
        )
        
        state_dict = torch.load(dinov2_model_path, map_location="cpu")
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("vit."):
                new_state_dict[k[4:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        
        print("DINOv2 model loaded successfully from local!")
        print(f"  Embed dim: {embed_dim}, Depth: {depth}, Heads: {num_heads}")
        
        return model
    
    except Exception as e:
        print(f"Warning: Failed to load DINOv2: {e}")
        return None


class DINOv2FeatureExtractor(nn.Module):
    """Internal DINOv2 feature extractor, frozen during training"""
    def __init__(self, output_size: Tuple[int, int] = (32, 32)):
        super().__init__()
        self.output_size = output_size
        
        self.dinov2 = load_local_dinov2()
        
        if self.dinov2 is None:
            print("Using simple CNN feature extractor instead")
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 4, stride=2, padding=1),
                nn.GroupNorm(8, 64),
                nn.GELU(),
                nn.Conv2d(64, 128, 4, stride=2, padding=1),
                nn.GroupNorm(8, 128),
                nn.GELU(),
                nn.Conv2d(128, 256, 4, stride=2, padding=1),
                nn.GroupNorm(8, 256),
                nn.GELU(),
                nn.Conv2d(256, 384, 4, stride=2, padding=1),
            )
            self.feat_dim = 384
        else:
            for param in self.dinov2.parameters():
                param.requires_grad = False
            self.dinov2.eval()
            self.feat_dim = self.dinov2.embed_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        
        x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        
        if self.dinov2 is not None:
            with torch.no_grad():
                x_resized = F.interpolate(x_reshaped, size=(224, 224), mode="bilinear", align_corners=False)
                outputs = self.dinov2.forward_features(x_resized)
                patch_feat = outputs["x_norm_patchtokens"]
                h = w = int(patch_feat.shape[1] ** 0.5)
                feat = patch_feat.permute(0, 2, 1).reshape(B * T, -1, h, w)
        else:
            feat = self.encoder(x_reshaped)
        
        feat = F.interpolate(feat, size=self.output_size, mode="bilinear", align_corners=False)
        
        feat = feat.view(B, T, -1, self.output_size[0], self.output_size[1])
        feat = feat.permute(0, 2, 1, 3, 4)
        
        return feat


def load_vae(vae_path="/data/data5/zhaoran/paper_code/exo/latent/outputs/vae/checkpoints/best_model.pt"):
    """Load pretrained VAE for latent methods"""
    sys.path.insert(0, '/data/data5/zhaoran/paper_code/exo/latent')
    from models.method_c_vae import FrameVAE
    
    vae = FrameVAE()
    
    if os.path.exists(vae_path):
        checkpoint = torch.load(vae_path, map_location='cpu')
        vae.load_state_dict(checkpoint['model_state_dict'])
        print(f"VAE loaded from {vae_path}")
    else:
        print(f"Warning: VAE checkpoint not found at {vae_path}")
    
    return vae


# =====================================================================
# 4 UNIFIED METHODS - SAME EXTERNAL INTERFACE FOR ALL
# =====================================================================

class HandPoseBaseline(nn.Module):
    """
    1. Baseline: Direct RGB regression
    Input:  (B, 3, T, 128, 128)  RGB video
    Output: (B*T, 51)              MANO hand pose
    
    Pipeline: RGB -> PixelEncoder -> RegressorHead
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


class HandPoseFreqDecomp(nn.Module):
    """
    2. Baseline + Frequency Decomposition
    Input:  (B, 3, T, 128, 128)  RGB video
    Output: (B*T, 51)              MANO hand pose
    
    Pipeline: RGB -> DINOv2 -> Freq-split -> DinoFeatureEncoder(x2) -> RegressorHead
    """
    def __init__(self, in_channels=3, num_frames=8, hidden_dim=256, num_pose_params=51, use_freq_decomp=True, **kwargs):
        super().__init__()
        self.use_freq_decomp = use_freq_decomp
        
        self.dino_extractor = DINOv2FeatureExtractor(output_size=(32, 32))
        
        for param in self.dino_extractor.parameters():
            param.requires_grad = False
        
        self.spectral_decomp = SoftSpectralDecomposition(
            freq_size=(8, 8),
            feat_dim=384,
            sigma=0.5,
        )
        
        self.encoder_low = DinoFeatureEncoder(in_channels=384, hidden_dim=hidden_dim)
        self.encoder_high = DinoFeatureEncoder(in_channels=384, hidden_dim=hidden_dim)
        
        regressor_input_dim = self.encoder_low.out_dim * self.encoder_low.spatial_dim * 2
        self.regressor = RegressorHead(regressor_input_dim, hidden_dim=512, num_pose_params=num_pose_params)
    
    def train(self, mode=True):
        super().train(mode)
        self.dino_extractor.eval()
        return self
    
    def forward(self, x):
        B, C, T, H, W = x.shape
        
        with torch.no_grad():
            feat = self.dino_extractor(x)
        
        feat_low, feat_high = self.spectral_decomp(feat)
        
        feat_low_down = self.encoder_low(feat_low)
        feat_high_down = self.encoder_high(feat_high)
        
        feat_low_flat = feat_low_down.permute(0, 2, 1, 3, 4).reshape(B * T, -1)
        feat_high_flat = feat_high_down.permute(0, 2, 1, 3, 4).reshape(B * T, -1)
        
        feat_combined = torch.cat([feat_low_flat, feat_high_flat], dim=1)
        pose = self.regressor(feat_combined)
        return pose


class HandPoseLatentOnly(nn.Module):
    """
    3. Baseline + Latent (SD-VAE)
    Input:  (B, 3, T, 128, 128)  RGB video
    Output: (B*T, 51)              MANO hand pose
    
    Pipeline: RGB -> VAE -> LatentEncoder -> RegressorHead
    """
    def __init__(self, in_channels=3, num_frames=8, hidden_dim=256, num_pose_params=51, 
                 vae_path="/data/data5/zhaoran/paper_code/exo/latent/outputs/vae/checkpoints/best_model.pt", **kwargs):
        super().__init__()
        self.vae = load_vae(vae_path)
        
        for param in self.vae.parameters():
            param.requires_grad = False
        
        self.encoder = LatentEncoder(in_channels=4, hidden_dim=hidden_dim)
        
        regressor_input_dim = self.encoder.out_dim * self.encoder.spatial_dim
        self.regressor = RegressorHead(regressor_input_dim, hidden_dim=512, num_pose_params=num_pose_params)
    
    def train(self, mode=True):
        super().train(mode)
        self.vae.eval()
        return self
    
    def forward(self, x):
        B, C, T, H, W = x.shape
        
        with torch.no_grad():
            z = self.vae.encode(x)
        
        feat = self.encoder(z)
        feat_flat = feat.permute(0, 2, 1, 3, 4).reshape(B * T, -1)
        pose = self.regressor(feat_flat)
        return pose


class HandPoseLatentFreq(nn.Module):
    """
    4. Baseline + Latent + Frequency Decomposition
    Input:  (B, 3, T, 128, 128)  RGB video
    Output: (B*T, 51)              MANO hand pose
    
    Pipeline: RGB -> VAE -> Freq-split -> LatentEncoder(x2) -> RegressorHead
    """
    def __init__(self, in_channels=3, num_frames=8, hidden_dim=256, num_pose_params=51, 
                 vae_path="/data/data5/zhaoran/paper_code/exo/latent/outputs/vae/checkpoints/best_model.pt",
                 use_freq_decomp=True, **kwargs):
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
