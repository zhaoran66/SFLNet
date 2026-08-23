"""
Experiment 1: Baseline - Direct RGB Regression
Input:  (B, 3, T, 128, 128)  RGB video
Output: (B*T, 51)              MANO hand pose

Pipeline: RGB -> PixelEncoder -> RegressorHead
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelEncoder(nn.Module):
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
    def __init__(self, in_channels=3, num_frames=8, hidden_dim=128, num_pose_params=51):
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
