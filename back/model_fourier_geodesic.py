import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TemporalFourierLayer(nn.Module):
    """Temporal Fourier decomposition: separates temporal low-frequency (static/background) and high-frequency (motion/hand)"""
    def __init__(self, freq_threshold_ratio: float = 0.1):
        super().__init__()
        self.freq_threshold_ratio = freq_threshold_ratio

    def forward(self, video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, num_views, C, H, W = video.shape

        x = video.permute(0, 2, 3, 4, 5, 1)  # [B, V, C, H, W, T]
        x = x.reshape(batch_size * num_views * C * H * W, seq_len)

        freq = torch.fft.fftshift(torch.fft.fft(x, dim=-1), dim=-1)

        threshold = int(seq_len * self.freq_threshold_ratio)
        center = seq_len // 2

        low_mask = torch.zeros(seq_len, device=video.device, dtype=torch.bool)
        low_mask[center - threshold: center + threshold + 1] = True

        low_freq = freq * low_mask
        high_freq = freq * ~low_mask

        static = torch.fft.ifft(torch.fft.ifftshift(low_freq, dim=-1), dim=-1).real
        motion = torch.fft.ifft(torch.fft.ifftshift(high_freq, dim=-1), dim=-1).real

        static = static.reshape(batch_size, num_views, C, H, W, seq_len).permute(0, 5, 1, 2, 3, 4)
        motion = motion.reshape(batch_size, num_views, C, H, W, seq_len).permute(0, 5, 1, 2, 3, 4)

        fused = torch.cat([video, static, motion], dim=3)  # 9 channels

        return fused, static, motion


class VideoFeatureExtractor(nn.Module):
    """Extract features from multi-view video"""
    def __init__(self, hidden_dim: int, num_views: int = 5, in_channels: int = 9):
        super().__init__()
        self.hidden_dim = hidden_dim
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
            nn.Linear(256 * num_views, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_views, c, h, w = video.shape

        video = video.reshape(batch_size * seq_len * num_views, c, h, w)
        features = self.conv_layers(video)
        features = features.reshape(batch_size * seq_len, num_views, 256, -1)
        features = features.mean(dim=-1)
        features = features.reshape(batch_size * seq_len, -1)

        fused_features = self.view_fusion(features)
        fused_features = fused_features.reshape(batch_size, seq_len, self.hidden_dim)

        return fused_features


class KeypointDecoder(nn.Module):
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

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        batch_size = memory.shape[0]
        queries = self.joint_queries.repeat(batch_size, 1, 1)

        x = queries
        for layer in self.decoder_layers:
            x = layer(x, memory)

        keypoints = self.keypoint_head(x)

        return keypoints


class Syn2SeqKeypointFourierGeodesic(nn.Module):
    """
    Syn2Seq with Temporal Fourier (No Interpolation)
    
    Architecture:
        1. Temporal FFT decomposition: separate temporal low-freq (static/background) and high-freq (motion/hand)
        2. CNN feature extraction on exo views
        3. Transformer sequence encoding + keypoint decoding
    
    This is a pure Fourier ablation: Baseline + Fourier only, no interpolation.
    
    Ablation table:
        Baseline (no Fourier, no interpolation):  104mm
        Back (Fourier only):                      ???mm  <- this model
        SPL (interpolation only):                  63mm
        Total (Fourier + interpolation):          ???mm
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = cfg['model']['hidden_dim']
        self.num_joints = cfg['dataset']['num_joints']
        self.num_views = len(cfg['dataset']['exo_views'])

        self.use_fourier = cfg['dataset'].get('use_fourier', True)
        self.use_fused_features = cfg['dataset'].get('use_fused_features', True)

        if self.use_fourier and self.use_fused_features:
            self.fourier_layer = TemporalFourierLayer(
                freq_threshold_ratio=cfg['dataset'].get('fourier_threshold', 0.1)
            )
            in_channels = 9
        else:
            self.fourier_layer = None
            in_channels = 3

        self.feature_extractor = VideoFeatureExtractor(
            hidden_dim=self.hidden_dim,
            num_views=self.num_views,
            in_channels=in_channels
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=cfg['model']['num_heads'],
            dim_feedforward=self.hidden_dim * 4,
            dropout=cfg['model']['dropout'],
            batch_first=True
        )
        self.sequence_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg['model']['num_layers']
        )

        self.keypoint_decoder = KeypointDecoder(
            hidden_dim=self.hidden_dim,
            num_joints=self.num_joints
        )

        self.pos_encoding = nn.Parameter(torch.randn(1000, self.hidden_dim))

    def forward(self, exo_video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            exo_video: [B, T, 5, 3, H, W] - exo view multi-view video
        
        Returns:
            keypoints: [B, T, 21, 3] - 3D joint coordinates
        """
        batch_size, seq_len, num_views, c, h, w = exo_video.shape

        # Step 1: Temporal Fourier decomposition
        if self.fourier_layer is not None:
            exo_video, _, _ = self.fourier_layer(exo_video)  # [B, T, 5, 9, H, W]

        # Step 2: Extract features
        exo_feat = self.feature_extractor(exo_video)  # [B, T, 256]

        # Step 3: Positional encoding
        seq_len_total = exo_feat.shape[1]
        encoded = exo_feat + self.pos_encoding[:seq_len_total, :].unsqueeze(0)

        # Step 4: Transformer sequence encoding
        encoded = self.sequence_encoder(encoded)

        # Step 5: Decode keypoints frame by frame
        keypoints_sequence = []
        for t in range(seq_len):
            frame_memory = encoded[:, t:t+1, :]
            keypoints = self.keypoint_decoder(frame_memory)
            keypoints_sequence.append(keypoints)

        keypoints_sequence = torch.stack(keypoints_sequence, dim=1)

        return keypoints_sequence
