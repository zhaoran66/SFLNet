"""
Hand Pose Estimation - Syn2Seq Migration
From Synchrony to Sequence: Exo Hand Pose Estimation via Interpolation

Two variants (same regression head for fair comparison):
1. Baseline:      Direct regression from exo video to hand pose (single frame)
2. Syn2Seq-Method: Use pose interpolation between frames (sequence modeling)

Key idea from Syn2Seq: 
- Instead of predicting each frame independently, interpolate between "key frames"
- This enforces temporal consistency and smoothness
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


class RegressorHead(nn.Module):
    """
    Shared regression head for all methods (SAME AS hand_pose_4methods.py)
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
    1. Baseline: Direct regression from exo video to hand pose
    No interpolation - treats each frame independently
    (Same as HandPoseBaseline in hand_pose_4methods.py)
    
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


class PoseInterpolator(nn.Module):
    """
    Pose Interpolator - inspired by Syn2Seq video interpolator
    Learns to interpolate between consecutive poses in the latent space
    Input: Features from frame i and frame j
    Output: Interpolated features for frames between i and j
    
    Key idea from Syn2Seq:
    - Instead of predicting every frame independently
    - Predict "key frame" poses and interpolate between them
    - This enforces temporal smoothness
    """
    def __init__(self, feature_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim
        
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.interp_net = nn.Sequential(
            nn.Linear(feature_dim * 2 + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, feature_dim),
        )
    
    def forward(self, feat_i: torch.Tensor, feat_j: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        Interpolate between feat_i and feat_j
        alpha: interpolation weight (0=feat_i, 1=feat_j)
        """
        B, T, D = feat_i.shape
        
        alpha_tensor = torch.full((B, T, 1), alpha, device=feat_i.device)
        time_emb = self.time_embed(alpha_tensor)
        
        feat_concat = torch.cat([feat_i, feat_j, time_emb], dim=-1)
        
        feat_interp = self.interp_net(feat_concat)
        
        return feat_interp


class HandPoseSyn2Seq(nn.Module):
    """
    2. Syn2Seq Method: Use pose interpolation between key frames
    Inspired by Syn2Seq: From Synchrony to Sequence via Interpolation
    
    Architecture:
    1. Encoder extracts features from all frames
    2. Select "key frames" (first, last, and optionally middle frames)
    3. Predict pose features for key frames
    4. Interpolate between key frames using learned interpolator
    5. Regress final pose from interpolated features
    
    Input: (B, 3, T, 128, 128)
    Output: (B*T, 51)
    
    Same regression head as baseline for fair comparison
    """
    def __init__(
        self,
        in_channels=3,
        num_frames=8,
        hidden_dim=128,
        num_pose_params=51,
        num_key_frames=2,  # Number of key frames to predict (rest interpolated)
        use_freq_decomp=True,
        use_learned_interp=True,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.num_key_frames = num_key_frames
        self.use_freq_decomp = use_freq_decomp
        self.use_learned_interp = use_learned_interp
        
        if use_freq_decomp:
            self.spectral_decomp = SoftSpectralDecomposition(
                freq_size=(8, 8),
                feat_dim=in_channels,
                sigma=0.5,
            )
            self.encoder_low = PixelEncoder(in_channels=in_channels, hidden_dim=hidden_dim)
            self.encoder_high = PixelEncoder(in_channels=in_channels, hidden_dim=hidden_dim)
            feature_dim = self.encoder_low.out_dim * self.encoder_low.spatial_dim * 2
        else:
            self.encoder = PixelEncoder(in_channels=in_channels, hidden_dim=hidden_dim)
            feature_dim = self.encoder.out_dim * self.encoder.spatial_dim
        
        self.key_frame_selector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        if use_learned_interp:
            self.pose_interpolator = PoseInterpolator(feature_dim, hidden_dim=256)
        
        self.temporal_smoothing = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1),
        )
        
        self.regressor = RegressorHead(feature_dim, hidden_dim=512, num_pose_params=num_pose_params)
    
    def _get_key_frame_indices(self):
        """Get indices of key frames (evenly spaced)"""
        indices = torch.linspace(0, self.num_frames - 1, self.num_key_frames).long()
        return indices.tolist()
    
    def forward(self, x):
        """
        x: (B, 3, T, H, W)
        Returns: (B*T, 51)
        """
        B, C, T, H, W = x.shape
        
        if self.use_freq_decomp:
            feat_low, feat_high = self.spectral_decomp(x)
            
            feat_low_down = self.encoder_low(feat_low)
            feat_high_down = self.encoder_high(feat_high)
            
            feat_low_flat = feat_low_down.permute(0, 2, 1, 3, 4).reshape(B, T, -1)
            feat_high_flat = feat_high_down.permute(0, 2, 1, 3, 4).reshape(B, T, -1)
            
            feat_all = torch.cat([feat_low_flat, feat_high_flat], dim=-1)
        else:
            feat = self.encoder(x)
            feat_all = feat.permute(0, 2, 1, 3, 4).reshape(B, T, -1)
        
        key_indices = self._get_key_frame_indices()
        key_features = feat_all[:, key_indices, :]
        
        feat_interpolated = torch.zeros_like(feat_all)
        
        for i in range(len(key_indices) - 1):
            start_idx = key_indices[i]
            end_idx = key_indices[i + 1]
            num_interp = end_idx - start_idx + 1
            
            feat_start = key_features[:, i:i+1, :].expand(-1, num_interp, -1)
            feat_end = key_features[:, i+1:i+2, :].expand(-1, num_interp, -1)
            
            if self.use_learned_interp:
                for j, t in enumerate(range(start_idx, end_idx + 1)):
                    alpha = j / (num_interp - 1) if num_interp > 1 else 0.0
                    interp_feat = self.pose_interpolator(
                        key_features[:, i:i+1, :],
                        key_features[:, i+1:i+2, :],
                        alpha
                    )
                    feat_interpolated[:, t, :] = interp_feat.squeeze(1)
            else:
                alphas = torch.linspace(0, 1, num_interp, device=feat_all.device).view(1, -1, 1)
                linear_interp = (1 - alphas) * feat_start + alphas * feat_end
                feat_interpolated[:, start_idx:end_idx+1, :] = linear_interp
        
        feat_smoothed = self.temporal_smoothing(feat_interpolated.transpose(1, 2)).transpose(1, 2)
        feat_smoothed = feat_interpolated + feat_smoothed
        
        feat_final = feat_smoothed.reshape(B * T, -1)
        pose = self.regressor(feat_final)
        
        return pose
    
    def get_interpolation_weights(self, x):
        """Get interpolation weights for visualization"""
        B, C, T, H, W = x.shape
        
        if self.use_freq_decomp:
            feat_low, feat_high = self.spectral_decomp(x)
            feat_low_down = self.encoder_low(feat_low)
            feat_high_down = self.encoder_high(feat_high)
            feat_low_flat = feat_low_down.permute(0, 2, 1, 3, 4).reshape(B, T, -1)
            feat_high_flat = feat_high_down.permute(0, 2, 1, 3, 4).reshape(B, T, -1)
            feat_all = torch.cat([feat_low_flat, feat_high_flat], dim=-1)
        else:
            feat = self.encoder(x)
            feat_all = feat.permute(0, 2, 1, 3, 4).reshape(B, T, -1)
        
        key_scores = self.key_frame_selector(feat_all).squeeze(-1)
        key_indices = self._get_key_frame_indices()
        
        return {
            'key_scores': key_scores,
            'key_indices': key_indices,
        }


class HandPoseSyn2SeqSimple(nn.Module):
    """
    3. Simplified Syn2Seq - No frequency decomposition
    Just key frame prediction + linear interpolation
    For ablation study
    """
    def __init__(
        self,
        in_channels=3,
        num_frames=8,
        hidden_dim=128,
        num_pose_params=51,
        num_key_frames=2,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.num_key_frames = num_key_frames
        
        self.encoder = PixelEncoder(in_channels=in_channels, hidden_dim=hidden_dim)
        
        feature_dim = self.encoder.out_dim * self.encoder.spatial_dim
        self.regressor = RegressorHead(feature_dim, hidden_dim=512, num_pose_params=num_pose_params)
    
    def _get_key_frame_indices(self):
        indices = torch.linspace(0, self.num_frames - 1, self.num_key_frames).long()
        return indices.tolist()
    
    def forward(self, x):
        B, C, T, H, W = x.shape
        
        feat = self.encoder(x)
        feat_all = feat.permute(0, 2, 1, 3, 4).reshape(B, T, -1)
        
        key_indices = self._get_key_frame_indices()
        key_features = feat_all[:, key_indices, :]
        
        feat_interpolated = torch.zeros_like(feat_all)
        
        for i in range(len(key_indices) - 1):
            start_idx = key_indices[i]
            end_idx = key_indices[i + 1]
            num_interp = end_idx - start_idx + 1
            
            feat_start = key_features[:, i:i+1, :]
            feat_end = key_features[:, i+1:i+2, :]
            
            alphas = torch.linspace(0, 1, num_interp, device=feat_all.device).view(1, -1, 1)
            linear_interp = (1 - alphas) * feat_start + alphas * feat_end
            
            feat_interpolated[:, start_idx:end_idx+1, :] = linear_interp
        
        feat_final = feat_interpolated.reshape(B * T, -1)
        pose = self.regressor(feat_final)
        
        return pose
