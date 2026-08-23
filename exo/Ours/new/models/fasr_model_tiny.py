import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, Optional

from .dino_extractor import DINOv2FeatureExtractor
from .freq_routing import (
    SoftSpectralDecomposition,
    PoseGuidedFrequencyRouting,
    create_gaussian_pose_mask,
    structure_weighted_asymmetric_loss,
    motion_aware_temporal_smoothness_loss,
    routing_sparsity_loss,
    latent_identity_consistency_loss,
    cross_view_alignment_loss,
    compute_gate_temporal_variance,
)


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: float = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, int(hidden_size * mlp_ratio))
        self.fc2 = nn.Linear(int(hidden_size * mlp_ratio), hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = MLP(hidden_size, mlp_ratio, dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        feat_dim: int = 1024,
        hidden_size: int = 192,
        num_heads: int = 3,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        
        self.input_proj = nn.Linear(feat_dim, hidden_size)
        
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.SiLU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        
        self.pose_embed = nn.Sequential(
            nn.Linear(32, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_size)
        self.output_proj = nn.Linear(hidden_size, feat_dim)
    
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
        
        for block in self.blocks:
            x_proj = block(x_proj)
        
        x_proj = self.norm(x_proj)
        x_out = self.output_proj(x_proj)
        
        x_out = x_out.view(B, T, H, W, C).permute(0, 4, 1, 2, 3)
        
        return x_out


class GaussianDiffusion:
    def __init__(self, num_timesteps: int = 1000, beta_schedule: str = "linear"):
        self.num_timesteps = num_timesteps
        
        if beta_schedule == "linear":
            self.beta = torch.linspace(0.0001, 0.02, num_timesteps)
        elif beta_schedule == "cosine":
            s = 0.008
            x = torch.linspace(0, num_timesteps, num_timesteps + 1)
            alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.beta = torch.clip(betas, 0, 0.999)
        
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)
        
        self.posterior_variance = self.beta * (1 - self.alpha_bar.roll(1)) / (1 - self.alpha_bar)
        self.posterior_variance[0] = self.posterior_variance[1]
    
    def to(self, device):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(device))
        return self
    
    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_bar = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1, 1)
        
        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
    
    def p_sample(self, model_output: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        beta = self.beta[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1, 1)
        sqrt_alpha_recip = 1.0 / torch.sqrt(self.alpha[t]).view(-1, 1, 1, 1, 1)
        
        pred_mean = sqrt_alpha_recip * (x_t - beta * model_output / sqrt_one_minus_alpha_bar)
        
        noise = torch.randn_like(x_t) * torch.sqrt(self.posterior_variance[t]).view(-1, 1, 1, 1, 1) if t[0] > 0 else 0
        
        return pred_mean + noise


class FeatureToImageDecoder(nn.Module):
    def __init__(
        self,
        feat_dim: int = 1024,
        output_channels: int = 3,
        hidden_dim: int = 128,
    ):
        super().__init__()
        
        # 32x32 -> 64x64 -> 128x128 (2? upsample)
        self.decoder = nn.Sequential(
            nn.Conv2d(feat_dim, hidden_dim * 2, 3, padding=1),
            nn.GroupNorm(8, hidden_dim * 2),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            
            nn.Conv2d(hidden_dim, output_channels, 3, padding=1),
            nn.Tanh(),
        )
        
        self.align_residual = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1),
        )
    
    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = feat.shape
        
        feat_reshaped = feat.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        
        feat_aligned = feat_reshaped + self.align_residual(feat_reshaped) * 0.1
        
        output = self.decoder(feat_aligned)
        
        output = output.view(B, T, 3, output.shape[2], output.shape[3])
        output = output.permute(0, 2, 1, 3, 4)
        
        return output


class FASRTinyModel(nn.Module):
    """
    FASR Tiny: Memory optimized version for 24GB GPUs
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.dino_extractor = DINOv2FeatureExtractor(output_size=(32, 32))
        feat_dim = self.dino_extractor.feat_dim
        
        self.soft_spectral_decomp = SoftSpectralDecomposition(
            freq_size=config.model.freq_size,
            feat_dim=feat_dim,
            sigma=config.model.spectral_sigma,
            use_dct=config.model.use_dct,
        )
        
        self.pose_freq_routing = PoseGuidedFrequencyRouting(
            feat_dim=feat_dim,
            pose_dim=16,
            hidden_dim=config.model.routing_hidden_dim,
        )
        
        self.diffusion_model = DiffusionTransformer(
            feat_dim=feat_dim,
            hidden_size=config.model.hidden_size,
            num_heads=config.model.num_heads,
            num_layers=config.model.num_layers,
            mlp_ratio=config.model.mlp_ratio,
            dropout=config.model.dropout,
        )
        
        self.decoder = FeatureToImageDecoder(
            feat_dim=feat_dim,
            output_channels=3,
            hidden_dim=64,
        )
        
        self._routing_gate = None
    
    def extract_features(self, video: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = self.dino_extractor(video)
        return feat
    
    def apply_spectral_routing(
        self,
        feat: torch.Tensor,
        pose: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feat_low, feat_high = self.soft_spectral_decomp(feat)
        feat_routed, routing_gate = self.pose_freq_routing(feat_low, feat_high, pose)
        return feat_routed, routing_gate
    
    def compute_losses(
        self,
        exo_video: torch.Tensor,
        ego_video: torch.Tensor,
        exo_pose: torch.Tensor,
        ego_pose: torch.Tensor,
        diffusion: GaussianDiffusion,
    ) -> Dict[str, torch.Tensor]:
        B, C, T, H, W = exo_video.shape
        
        exo_feat = self.extract_features(exo_video)
        ego_feat = self.extract_features(ego_video)
        
        exo_routed, exo_gate = self.apply_spectral_routing(exo_feat, exo_pose)
        ego_routed, ego_gate = self.apply_spectral_routing(ego_feat, ego_pose)
        
        num_history = min(4, T)
        pose_cond = torch.cat([exo_pose[:, -1:], ego_pose[:, :1]], dim=1)
        
        full_feat_sequence = torch.cat([exo_routed[:, :, -num_history:], ego_routed[:, :, :4]], dim=2)
        
        t = torch.randint(0, diffusion.num_timesteps, (B,), device=exo_video.device)
        noise = torch.randn_like(full_feat_sequence)
        x_t = diffusion.q_sample(full_feat_sequence, t, noise)
        
        model_output = self.diffusion_model(x_t, t, pose_cond)
        
        diffusion_loss = F.mse_loss(model_output, noise)
        
        pred_feat = x_t + model_output
        
        pred_video = self.decoder(pred_feat)
        target_video = torch.cat([exo_video[:, :, -num_history:], ego_video[:, :, :4]], dim=2)
        
        pose_for_mask = torch.cat([exo_pose[:, -num_history:], ego_pose[:, :4]], dim=1)
        
        pose_mask = create_gaussian_pose_mask(
            pose_for_mask,
            spatial_size=(num_history + 4, target_video.shape[3], target_video.shape[4]),
            sigma=self.config.model.pose_mask_sigma,
        )
        
        recon_loss, fg_loss, bg_loss = structure_weighted_asymmetric_loss(
            pred_video, target_video, pose_mask
        )
        
        all_gates = torch.cat([exo_gate[:, :, -num_history:], ego_gate[:, :, :4]], dim=2)
        temp_loss = motion_aware_temporal_smoothness_loss(
            all_gates, pose_for_mask, self.config.model.temp_tau
        )
        
        sparse_loss = routing_sparsity_loss(all_gates)
        
        identity_loss = latent_identity_consistency_loss(
            pred_feat, full_feat_sequence.detach()
        )
        
        align_loss = cross_view_alignment_loss(
            exo_routed[:, :, -num_history:],
            ego_routed[:, :, :4],
            pose_mask
        )
        
        total_loss = (
            self.config.model.lambda_diff * diffusion_loss
            + self.config.model.lambda_struct_fg * fg_loss
            + self.config.model.lambda_struct_bg * bg_loss
            + self.config.model.lambda_route_temp * temp_loss
            + self.config.model.lambda_route_sparse * sparse_loss
            + self.config.model.lambda_id * identity_loss
            + self.config.model.lambda_align * align_loss
        )
        
        return {
            "total_loss": total_loss,
            "diffusion_loss": diffusion_loss,
            "recon_loss": recon_loss,
            "fg_loss": fg_loss,
            "bg_loss": bg_loss,
            "temp_loss": temp_loss,
            "sparse_loss": sparse_loss,
            "id_loss": identity_loss,
            "align_loss": align_loss,
        }
    
    @torch.no_grad()
    def sample(
        self,
        exo_video: torch.Tensor,
        exo_pose: torch.Tensor,
        ego_pose: torch.Tensor,
        diffusion: GaussianDiffusion,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, T, H, W = exo_video.shape
        
        exo_feat = self.extract_features(exo_video)
        exo_routed, exo_gate = self.apply_spectral_routing(exo_feat, exo_pose)
        
        num_history = min(4, T)
        pose_cond = torch.cat([exo_pose[:, -1:], ego_pose[:, :1]], dim=1)
        
        total_frames = num_history + 4
        
        x = torch.randn(B, exo_feat.shape[1], total_frames, 32, 32, device=exo_video.device)
        x[:, :, :num_history] = exo_routed[:, :, -num_history:]
        
        for t in reversed(range(diffusion.num_timesteps)):
            t_tensor = torch.full((B,), t, device=exo_video.device, dtype=torch.long)
            model_output = self.diffusion_model(x, t_tensor, pose_cond)
            x = diffusion.p_sample(model_output, x, t_tensor)
            x[:, :, :num_history] = exo_routed[:, :, -num_history:]
        
        gen_video = self.decoder(x)
        
        return gen_video, x, exo_gate
    
    @torch.no_grad()
    def get_gate_stats(
        self,
        exo_gate: torch.Tensor,
    ) -> Dict[str, float]:
        fg_var, bg_var = compute_gate_temporal_variance(exo_gate)
        
        return {
            "fg_gate_variance": fg_var,
            "bg_gate_variance": bg_var,
            "mean_gate_value": float(exo_gate.mean().item()),
            "min_gate_value": float(exo_gate.min().item()),
            "max_gate_value": float(exo_gate.max().item()),
        }
