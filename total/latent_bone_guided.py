# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple

from geodesic_interpolator import GeodesicInterpolator


class TemporalFourierLayer(nn.Module):
    """Temporal Fourier decomposition: separates temporal low-frequency (static/background) and high-frequency (motion/hand)
    
    Input:  [B, T, V, 3, H, W]  (RGB only)
    Output: [B, T, V, 9, H, W]  (RGB + static + motion)
    
    Low frequency  -> background (static regions across time)
    High frequency -> hand motion (dynamic regions across time)
    
    Note: Uses 1D FFT along time dimension (T), not spatial FFT (HxW)
    """
    def __init__(self, freq_threshold_ratio: float = 0.1):
        super().__init__()
        self.freq_threshold_ratio = freq_threshold_ratio
        
    def forward(self, video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, num_views, C, H, W = video.shape
        
        # Reshape for temporal FFT: [B, V, C, H, W, T]
        x = video.permute(0, 2, 3, 4, 5, 1)  # [B, V, C, H, W, T]
        x = x.reshape(batch_size * num_views * C * H * W, seq_len)
        
        # 1D FFT along time dimension
        freq_spectrum = torch.fft.fftshift(torch.fft.fft(x, dim=-1), dim=-1)
        
        # Create low-frequency mask (center frequencies)
        threshold = int(seq_len * self.freq_threshold_ratio)
        center = seq_len // 2
        low_mask = torch.zeros(seq_len, device=video.device, dtype=torch.bool)
        low_mask[center - threshold: center + threshold + 1] = True
        
        # Apply mask and inverse FFT
        low_freq_spectrum = freq_spectrum * low_mask
        high_freq_spectrum = freq_spectrum * ~low_mask
        
        static_component = torch.fft.ifft(
            torch.fft.ifftshift(low_freq_spectrum, dim=-1), dim=-1
        ).real
        motion_component = torch.fft.ifft(
            torch.fft.ifftshift(high_freq_spectrum, dim=-1), dim=-1
        ).real
        
        # Reshape back: [B, T, V, C, H, W]
        static_component = static_component.reshape(
            batch_size, num_views, C, H, W, seq_len
        ).permute(0, 5, 1, 2, 3, 4)
        motion_component = motion_component.reshape(
            batch_size, num_views, C, H, W, seq_len
        ).permute(0, 5, 1, 2, 3, 4)
        
        # Concatenate: [B, T, V, 9, H, W]
        fused = torch.cat([video, static_component, motion_component], dim=3)
        
        return fused, static_component, motion_component


# Backward compatibility alias
FourierLayerGPU = TemporalFourierLayer


class CNNEncoder(nn.Module):
    """2D CNN encoder with configurable input channels
    
    in_channels=3:  RGB only (baseline mode)
    in_channels=9:  RGB + static + motion (fourier mode)
    """
    def __init__(self, embed_dim: int = 256, num_views: int = 5, in_channels: int = 9):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_views = num_views
        self.in_channels = in_channels
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        self.view_fusion = nn.Sequential(
            nn.Linear(256 * num_views, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, num_views, C, H, W]"""
        batch_size, seq_len, num_views, C, H, W = x.shape
        
        x = x.view(batch_size * seq_len * num_views, C, H, W)
        features = self.conv_layers(x)
        features = features.view(batch_size * seq_len, num_views, 256, -1)
        features = features.mean(dim=-1)
        features = features.view(batch_size * seq_len, -1)
        
        fused_features = self.view_fusion(features)
        fused_features = fused_features.view(batch_size, seq_len, self.embed_dim)
        
        return fused_features


class LatentSpaceKeypointDecoder(nn.Module):
    """Keypoint decoder matching baseline/spl architecture"""
    def __init__(self, hidden_dim: int, num_joints: int = 21):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_joints = num_joints
        
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(4)
        ])
        
        self.keypoint_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)
        )
        
        self.joint_queries = nn.Parameter(torch.randn(1, num_joints, hidden_dim))
        
        self.latent_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        batch_size = memory.shape[0]
        queries = self.joint_queries.repeat(batch_size, 1, 1)
        
        memory = self.latent_projection(memory)
        
        x = queries
        for layer in self.decoder_layers:
            x = layer(x, memory)
            
        keypoints = self.keypoint_head(x)
        
        return keypoints


class LatentHandPoseModel(nn.Module):
    """Total model: Baseline + Temporal Fourier + Latent Interpolation
    
    Two complementary components:
        1. Temporal Fourier: input-level hand/background separation (low freq=bg, high freq=hand)
        2. Latent Interpolation: feature-level exo->ego bridge (SLERP + ego anchor)
    
    Ablation story:
        Baseline (CNN 3ch, no fourier, no interpolation): 104mm
        + Temporal Fourier (CNN 9ch): ???mm
        + Latent Interpolation (SLERP + ego anchor): ???mm
    """
    def __init__(self, embed_dim: int = 256, num_interpolation_steps: int = 4, 
                 num_joints: int = 21, num_views: int = 5,
                 use_fourier: bool = True, fourier_threshold: float = 0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_interpolation_steps = num_interpolation_steps
        self.num_joints = num_joints
        self.use_fourier = use_fourier
        
        # Component 1: Fourier frequency decomposition (input-level separation)
        if use_fourier:
            self.fourier_layer = FourierLayerGPU(freq_threshold_ratio=fourier_threshold)
            in_channels = 9
        else:
            self.fourier_layer = None
            in_channels = 3
        
        # Component 2: CNN feature encoder (9ch if fourier, 3ch if baseline)
        self.encoder = CNNEncoder(embed_dim, num_views, in_channels=in_channels)
        
        # Component 3: Geodesic interpolation (feature-level exo->ego bridge)
        self.geodesic_interpolator = GeodesicInterpolator(
            hidden_dim=embed_dim,
            num_interpolate_steps=num_interpolation_steps
        )
        
        # Component 4: Sequence encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.sequence_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=6
        )
        
        # Component 5: Keypoint decoder
        self.keypoint_decoder = LatentSpaceKeypointDecoder(
            hidden_dim=embed_dim,
            num_joints=num_joints
        )
        
        self.pos_encoding = nn.Parameter(torch.randn(1000, embed_dim))
        
        # Component 6: Ego keypoints encoder (for training)
        self.ego_proj = nn.Sequential(
            nn.Linear(num_joints * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Component 7: Endpoint Predictor (solves train-test gap)
        # Training: uses GT ego_feat as target
        # Inference: predicts ego_feat from exo_feat (per-sample, not global mean)
        self.endpoint_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, exo_video: torch.Tensor, ego_keypoints_gt: torch.Tensor = None,
                exo_pose: torch.Tensor = None, ego_pose: torch.Tensor = None) -> torch.Tensor:
        """Forward pass
        
        Args:
            exo_video: [batch, seq_len, num_views, 3, H, W] (always 3 channels input)
            ego_keypoints_gt: [batch, seq_len, num_joints, 3], optional for training
            exo_pose: [batch, seq_len, 12] exo camera extrinsics (R+T flattened)
            ego_pose: [batch, seq_len, 12] ego camera extrinsics (R+T flattened)
        
        Returns:
            keypoints_sequence: [batch, seq_len, num_joints, 3]
        """
        batch_size, seq_len, num_views, c, h, w = exo_video.shape
        
        # Step 1: Temporal Fourier decomposition (3ch -> 9ch) -- input-level hand/bg separation
        if self.fourier_layer is not None:
            exo_video, _, _ = self.fourier_layer(exo_video)
        
        # Step 2: CNN encoding (9ch or 3ch -> 256d features)
        exo_features = self.encoder(exo_video)
        
        # Step 3: Geodesic interpolation -- feature-level exo->ego bridge
        if self.training and ego_keypoints_gt is not None:
            ego_features = self._encode_ego_keypoints(ego_keypoints_gt)
            self.geodesic_interpolator.update_ego_anchor(ego_features)
            full_sequence = self.geodesic_interpolator(exo_features, ego_features,
                                                       exo_pose=exo_pose, ego_pose=ego_pose)
        else:
            # Inference: use Endpoint Predictor to predict ego_feat from exo_feat
            predicted_ego_features = self.endpoint_predictor(exo_features)
            full_sequence = self.geodesic_interpolator(exo_features, predicted_ego_features,
                                                       exo_pose=exo_pose, ego_pose=ego_pose)
        
        # Step 4: Flatten sequence for transformer
        batch_size, seq_len, num_steps, feat_dim = full_sequence.shape
        full_sequence = full_sequence.view(batch_size, seq_len * num_steps, feat_dim)
        
        seq_len_total = full_sequence.shape[1]
        full_sequence = full_sequence + self.pos_encoding[:seq_len_total, :].unsqueeze(0)
        
        # Step 5: Transformer sequence encoding
        encoded = self.sequence_encoder(full_sequence)
        
        # Step 6: Decode keypoints per frame
        keypoints_sequence = []
        step_size = self.num_interpolation_steps + 2
        
        for t in range(seq_len):
            start_idx = t * step_size
            end_idx = (t + 1) * step_size
            frame_memory = encoded[:, start_idx:end_idx, :]
            kp = self.keypoint_decoder(frame_memory)
            keypoints_sequence.append(kp)
        
        keypoints_sequence = torch.stack(keypoints_sequence, dim=1)
        
        # During training: compute auxiliary endpoint loss
        if self.training and ego_keypoints_gt is not None:
            pred_ego = self.endpoint_predictor(exo_features)
            endpoint_loss = F.mse_loss(pred_ego, ego_features.detach()) * 0.1
            return keypoints_sequence, {'endpoint_loss': endpoint_loss}
        
        return keypoints_sequence
    
    def _encode_ego_keypoints(self, keypoints: torch.Tensor) -> torch.Tensor:
        """Encode ego keypoints to latent space"""
        batch_size, seq_len, num_joints, coords = keypoints.shape
        keypoints_flat = keypoints.view(batch_size * seq_len, -1)
        
        ego_feat = self.ego_proj(keypoints_flat)
        ego_feat = ego_feat.view(batch_size, seq_len, self.embed_dim)
        return ego_feat


class KeypointLoss(nn.Module):
    """MSE + L1 loss matching spl/losses.py"""
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        mse = self.mse_loss(pred, target)
        l1 = self.l1_loss(pred, target)
        loss = mse + 0.5 * l1
        
        return {
            'loss': loss,
            'mse': mse,
            'l1': l1
        }
