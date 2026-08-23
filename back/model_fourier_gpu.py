import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SpatialFourierLayer(nn.Module):
    def __init__(self, freq_threshold_ratio: float = 0.1):
        super().__init__()
        self.freq_threshold_ratio = freq_threshold_ratio

    def forward(self, video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, num_views, C, H, W = video.shape

        x = video.reshape(batch_size * seq_len * num_views * C, H, W)

        freq = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))

        h_center, w_center = H // 2, W // 2
        r = int(min(H, W) * self.freq_threshold_ratio)

        low_mask = torch.zeros(H, W, dtype=torch.bool, device=video.device)
        low_mask[h_center - r:h_center + r + 1, w_center - r:w_center + r + 1] = True
        high_mask = ~low_mask

        low_freq = freq * low_mask.unsqueeze(0)
        high_freq = freq * high_mask.unsqueeze(0)

        static = torch.fft.ifft2(torch.fft.ifftshift(low_freq, dim=(-2, -1)), s=(H, W), dim=(-2, -1)).real
        motion = torch.fft.ifft2(torch.fft.ifftshift(high_freq, dim=(-2, -1)), s=(H, W), dim=(-2, -1)).real

        static = static.reshape(batch_size, seq_len, num_views, C, H, W)
        motion = motion.reshape(batch_size, seq_len, num_views, C, H, W)

        fused = torch.cat([video, static, motion], dim=3)

        return fused, static, motion


class CrossViewLinearInterpolator(nn.Module):
    def __init__(self, hidden_dim: int, num_interpolate_steps: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_interpolate_steps = num_interpolate_steps

        self.ego_queries = nn.Parameter(torch.randn(1, 1, hidden_dim))

        self.interp_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * num_interpolate_steps)
        )

        self.step_embeddings = nn.Parameter(torch.randn(num_interpolate_steps, hidden_dim))

    def forward(self, exo_feat: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = exo_feat.shape

        ego_query = self.ego_queries.expand(batch_size, seq_len, -1)

        combined = torch.cat([exo_feat, ego_query], dim=-1)
        interpolated = self.interp_net(combined)
        interpolated = interpolated.view(batch_size, seq_len, self.num_interpolate_steps, self.hidden_dim)

        interpolated = interpolated + self.step_embeddings.unsqueeze(0).unsqueeze(0)

        full_sequence = torch.cat([
            exo_feat.unsqueeze(2),
            interpolated,
            ego_query.unsqueeze(2)
        ], dim=2)

        return full_sequence


class VideoFeatureExtractor(nn.Module):
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


class Syn2SeqKeypointGPUFourier(nn.Module):
    """
    Syn2Seq with Spatial Fourier + Cross-View Linear Interpolation

    Architecture:
        1. Spatial FFT decomposition: separate spatial low-freq (structure) and high-freq (texture/edge)
        2. Cross-view linear interpolation: exo -> ego via learned interpolation
        3. Transformer sequence encoding + keypoint decoding
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = cfg['model']['hidden_dim']
        self.num_interpolate_steps = cfg['model']['interpolate_steps']
        self.num_joints = cfg['dataset']['num_joints']
        self.num_views = len(cfg['dataset']['exo_views'])

        self.use_fourier = cfg['dataset'].get('use_fourier', True)
        self.use_fused_features = cfg['dataset'].get('use_fused_features', True)

        if self.use_fourier and self.use_fused_features:
            self.fourier_layer = SpatialFourierLayer(
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

        self.syn2seq = CrossViewLinearInterpolator(
            hidden_dim=self.hidden_dim,
            num_interpolate_steps=self.num_interpolate_steps
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
        batch_size, seq_len, num_views, c, h, w = exo_video.shape

        if self.fourier_layer is not None:
            exo_video, _, _ = self.fourier_layer(exo_video)

        exo_feat = self.feature_extractor(exo_video)

        full_sequence = self.syn2seq(exo_feat)

        batch_size, seq_len, num_steps, feat_dim = full_sequence.shape
        full_sequence = full_sequence.reshape(batch_size, seq_len * num_steps, feat_dim)

        seq_len_total = full_sequence.shape[1]
        full_sequence = full_sequence + self.pos_encoding[:seq_len_total, :].unsqueeze(0)

        encoded = self.sequence_encoder(full_sequence)

        keypoints_sequence = []
        step_size = self.num_interpolate_steps + 2

        for t in range(seq_len):
            start_idx = t * step_size
            end_idx = (t + 1) * step_size
            frame_memory = encoded[:, start_idx:end_idx, :]
            kp = self.keypoint_decoder(frame_memory)
            keypoints_sequence.append(kp)

        keypoints_sequence = torch.stack(keypoints_sequence, dim=1)

        return keypoints_sequence
