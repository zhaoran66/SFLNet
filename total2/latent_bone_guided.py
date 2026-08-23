# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple

from geodesic_interpolator import GeodesicInterpolator


class FeatureFourierLayer(nn.Module):
    """Feature-space Fourier decomposition: separates semantic low-freq (static/background) and high-freq (motion/hand)

    Input:  [B, T, D]  (semantic features from CNN)
    Output: [B, T, D]  (fused: original + low_freq + high_freq)

    Low frequency  -> static semantic features (background semantics)
    High frequency -> dynamic semantic features (hand motion semantics)

    Works on semantic-level temporal signals, more meaningful than pixel-level FFT.
    """
    def __init__(self, freq_threshold_ratio: float = 0.1):
        super().__init__()
        self.freq_threshold_ratio = freq_threshold_ratio

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, feat_dim = feat.shape

        x = feat.permute(0, 2, 1)  # [B, D, T]
        freq_spectrum = torch.fft.fftshift(torch.fft.fft(x, dim=-1), dim=-1)

        threshold = int(seq_len * self.freq_threshold_ratio)
        center = seq_len // 2
        low_mask = torch.zeros(seq_len, device=feat.device, dtype=torch.bool)
        low_mask[center - threshold: center + threshold + 1] = True

        low_freq = torch.fft.ifft(
            torch.fft.ifftshift(freq_spectrum * low_mask, dim=-1), dim=-1
        ).real  # [B, D, T]

        high_freq = torch.fft.ifft(
            torch.fft.ifftshift(freq_spectrum * ~low_mask, dim=-1), dim=-1
        ).real  # [B, D, T]

        low_feat = low_freq.permute(0, 2, 1)   # [B, T, D]
        high_feat = high_freq.permute(0, 2, 1)  # [B, T, D]

        return feat, low_feat, high_feat


class CNNEncoder(nn.Module):
    """2D CNN encoder with 3ch RGB input (no pixel-space Fourier)"""
    def __init__(self, embed_dim: int = 256, num_views: int = 5, in_channels: int = 3):
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
    """Architecture: CNN(3ch) -> Camera-guided -> Feature Fourier -> SLERP Interpolation

    Pipeline:
        1. CNN: extract semantic features from raw 3ch RGB (no pixel-space Fourier)
        2. Camera guidance: modulate features with exo/ego camera parameters
        3. Feature Fourier: decompose semantic features into static/dynamic in freq domain
        4. feat_fusion: fuse original + low_freq + high_freq -> 256d
        5. SLERP interpolation: exo->ego geodesic bridge
        6. Transformer + Keypoint decoder

    Physical motivation:
        Pixel-space Fourier: separates pixel intensity signals
        Feature-space Fourier: separates semantic-level motion signals (more meaningful)
            - Low freq = static semantic features (background semantics)
            - High freq = dynamic semantic features (hand motion semantics)
    """
    def __init__(self, embed_dim: int = 256, num_interpolation_steps: int = 4,
                 num_joints: int = 21, num_views: int = 5,
                 fourier_threshold: float = 0.1, pose_dim: int = 12):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_interpolation_steps = num_interpolation_steps
        self.num_joints = num_joints

        # Component 1: CNN encoder (always 3ch, no pixel-space Fourier)
        self.encoder = CNNEncoder(embed_dim, num_views, in_channels=3)

        # Component 2: Camera pose encoder (guides feature space)
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Component 3: Feature-space Fourier decomposition
        self.feature_fourier = FeatureFourierLayer(freq_threshold_ratio=fourier_threshold)

        # Component 4: Feature fusion (original + low_freq + high_freq -> 256d)
        self.feat_fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )

        # Component 5: Geodesic interpolation (feature-level exo->ego bridge)
        self.geodesic_interpolator = GeodesicInterpolator(
            hidden_dim=embed_dim,
            num_interpolate_steps=num_interpolation_steps
        )

        # Component 6: Sequence encoder
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

        # Component 7: Keypoint decoder
        self.keypoint_decoder = LatentSpaceKeypointDecoder(
            hidden_dim=embed_dim,
            num_joints=num_joints
        )

        self.pos_encoding = nn.Parameter(torch.randn(1000, embed_dim))

        # Component 8: Ego keypoints encoder (for training)
        self.ego_proj = nn.Sequential(
            nn.Linear(num_joints * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Component 9: Endpoint Predictor (inference: predict ego_feat from exo_feat)
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
            exo_pose: [batch, seq_len, 12] or [batch, seq_len, V, 12] exo camera extrinsics
            ego_pose: [batch, seq_len, 12] ego camera extrinsics

        Returns:
            keypoints_sequence: [batch, seq_len, num_joints, 3]
        """
        batch_size, seq_len, num_views, c, h, w = exo_video.shape

        # Step 1: CNN feature extraction (raw 3ch RGB, no pixel-space Fourier)
        exo_feat = self.encoder(exo_video)  # [B, T, 256]

        # Step 2: Camera parameter guidance in feature space
        if exo_pose is not None and ego_pose is not None:
            if exo_pose.dim() == 4:
                exo_pose_flat = exo_pose.mean(dim=2)  # [B,T,V,12] -> [B,T,12]
            elif exo_pose.dim() == 3:
                exo_pose_flat = exo_pose.squeeze(2)  # [B,T,1,12] -> [B,T,12]
            else:
                exo_pose_flat = exo_pose  # [B,T,12]
            if ego_pose.dim() == 2:
                ego_pose_flat = ego_pose.unsqueeze(1).expand(-1, seq_len, -1)
            else:
                ego_pose_flat = ego_pose  # [B,T,12]

            pose_emb = self.pose_encoder(
                torch.cat([exo_pose_flat, ego_pose_flat], dim=-1)
            )  # [B, T, 256]
            exo_feat = exo_feat + pose_emb

        # Step 3: Feature-space Fourier decomposition
        orig_feat, low_feat, high_feat = self.feature_fourier(exo_feat)
        # orig_feat:  [B, T, 256] original semantic features
        # low_feat:   [B, T, 256] static semantic features (background)
        # high_feat:  [B, T, 256] dynamic semantic features (hand motion)

        # Step 4: Fuse original + low_freq + high_freq -> 256d
        fused_feat = self.feat_fusion(
            torch.cat([orig_feat, low_feat, high_feat], dim=-1)
        )  # [B, T, 256]

        # Step 5: SLERP interpolation (feature-level exo->ego bridge)
        if self.training and ego_keypoints_gt is not None:
            ego_features = self._encode_ego_keypoints(ego_keypoints_gt)
            self.geodesic_interpolator.update_ego_anchor(ego_features)
            full_sequence = self.geodesic_interpolator(
                fused_feat, ego_features,
                exo_pose=exo_pose, ego_pose=ego_pose
            )
        else:
            predicted_ego_features = self.endpoint_predictor(fused_feat)
            full_sequence = self.geodesic_interpolator(
                fused_feat, predicted_ego_features,
                exo_pose=exo_pose, ego_pose=ego_pose
            )

        # Step 6: Flatten sequence for transformer
        batch_size, seq_len, num_steps, feat_dim = full_sequence.shape
        full_sequence = full_sequence.view(batch_size, seq_len * num_steps, feat_dim)

        seq_len_total = full_sequence.shape[1]
        full_sequence = full_sequence + self.pos_encoding[:seq_len_total, :].unsqueeze(0)

        # Step 7: Transformer sequence encoding
        encoded = self.sequence_encoder(full_sequence)

        # Step 8: Decode keypoints per frame
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
            pred_ego = self.endpoint_predictor(fused_feat)
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
