"""
FASR Ablation: Latent-only (NO Frequency Routing)
This is the ablation model to test how much gain comes from Latent alone vs Latent + Frequency Routing
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, Optional

from .dino_extractor import DINOv2FeatureExtractor


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class FeatureToVideoDecoder(nn.Module):
    def __init__(self, feat_dim=384, output_channels=3, hidden_dim=256):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feat_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim // 2, output_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )
    
    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        out = self.decoder(x)
        out = out.reshape(B, T, out.shape[1], out.shape[2], out.shape[3]).permute(0, 2, 1, 3, 4)
        return out


class LatentDiffusionTransformer(nn.Module):
    """
    Latent-only Ablation: Just simple diffusion transformer on DINOv2 features
    NO Frequency Routing, NO Pose Guidance
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.dino_extractor = DINOv2FeatureExtractor(output_size=(32, 32))
        feat_dim = self.dino_extractor.feat_dim
        
        hidden_size = config.model.hidden_size
        
        self.input_proj = nn.Linear(feat_dim, hidden_size)
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.SiLU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.pose_embed = nn.Sequential(
            nn.Linear(32, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        
        num_layers = config.model.num_layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=config.model.num_heads,
                dim_feedforward=int(hidden_size * config.model.mlp_ratio),
                dropout=config.model.dropout,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_size)
        self.output_proj = nn.Linear(hidden_size, feat_dim)
        
        self.decoder = FeatureToVideoDecoder(
            feat_dim=feat_dim,
            output_channels=3,
            hidden_dim=256,
        )
    
    def extract_features(self, video: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = self.dino_extractor(video)
        return feat
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, pose_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, T, H, W = x.shape
        
        x_flat = x.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)
        x_proj = self.input_proj(x_flat)
        
        t_emb = timestep_embedding(t, x_proj.shape[-1])
        t_emb = self.time_embed(t_emb)
        x_proj = x_proj + t_emb.unsqueeze(1)
        
        if pose_cond is not None:
            pose_emb = self.pose_embed(pose_cond.flatten(1))
            x_proj = x_proj + pose_emb.unsqueeze(1)
        
        for layer in self.layers:
            x_proj = layer(x_proj)
        
        x_proj = self.norm(x_proj)
        x_out = self.output_proj(x_proj)
        x_out = x_out.view(B, T, H, W, C).permute(0, 4, 1, 2, 3)
        
        return x_out
    
    def compute_losses(
        self,
        exo_video: torch.Tensor,
        ego_video: torch.Tensor,
        exo_pose: torch.Tensor,
        ego_pose: torch.Tensor,
        diffusion,
    ) -> Dict[str, torch.Tensor]:
        B, C, T, H, W = exo_video.shape
        
        exo_feat = self.extract_features(exo_video)
        ego_feat = self.extract_features(ego_video)
        
        num_history = min(4, T)
        pose_cond = torch.cat([exo_pose[:, -1:], ego_pose[:, :1]], dim=1)
        
        full_feat_sequence = torch.cat([exo_feat[:, :, -num_history:], ego_feat[:, :, :4]], dim=2)
        
        t = torch.randint(0, diffusion.num_timesteps, (B,), device=exo_video.device)
        noise = torch.randn_like(full_feat_sequence)
        x_t = diffusion.q_sample(full_feat_sequence, t, noise)
        
        model_output = self.forward(x_t, t, pose_cond)
        
        diffusion_loss = F.mse_loss(model_output, noise)
        
        pred_feat = x_t + model_output
        pred_video = self.decoder(pred_feat)
        target_video = torch.cat([exo_video[:, :, -num_history:], ego_video[:, :, :4]], dim=2)
        
        recon_loss = F.mse_loss(pred_video, target_video)
        
        total_loss = (
            self.config.model.lambda_diff * diffusion_loss
            + recon_loss
        )
        
        return {
            "total_loss": total_loss,
            "diffusion_loss": diffusion_loss,
            "recon_loss": recon_loss,
        }
