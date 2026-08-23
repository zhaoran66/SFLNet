import torch
import torch.nn as nn

class EgoNet(nn.Module):
    """
    Ego image -> 3D hand keypoints
    Same architecture as Baseline but input is ego image
    """
    def __init__(self, hidden_dim=256, num_joints=21):
        super().__init__()
        self.num_joints = num_joints

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        )
        self.feat_proj = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8,
            dim_feedforward=hidden_dim*4,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.pos_encoding = nn.Parameter(torch.randn(1000, hidden_dim) * 0.02)

        self.joint_queries = nn.Parameter(torch.randn(1, num_joints, hidden_dim) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=8,
            dim_feedforward=hidden_dim*4,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)
        self.keypoint_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(),
            nn.Linear(hidden_dim//2, 3), nn.Tanh()
        )

    def forward(self, ego_video):
        # ego_video: [B, T, 3, H, W]
        B, T, C, H, W = ego_video.shape
        x = ego_video.reshape(B*T, C, H, W)
        feat = self.conv_layers(x)
        feat = feat.mean(dim=[-2, -1])
        feat = self.feat_proj(feat)
        feat = feat.reshape(B, T, -1)
        feat = feat + self.pos_encoding[:T]
        encoded = self.transformer(feat)
        all_kp = []
        for t in range(T):
            mem = encoded[:, t:t+1, :]
            q = self.joint_queries.expand(B, -1, -1)
            dec = self.decoder(q, mem)
            kp = self.keypoint_head(dec)
            all_kp.append(kp)
        return torch.stack(all_kp, dim=1)  # [B, T, 21, 3]
