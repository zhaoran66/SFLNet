import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

from .dino_extractor import DINOv2FeatureExtractor, FeatureToImageDecoder
from .frequency_decomp import (
    SemanticFrequencyDecomposition,
    BackgroundDynamicsInterpolator,
    FrequencyRouters,
    orthogonal_loss,
)


class DiffusionModel(nn.Module):
    def __init__(
        self,
        feat_dim: int = 384,
        hidden_size: int = 256,
        num_heads: int = 4,
        num_layers: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        
        self.pose_embed = nn.Sequential(
            nn.Linear(32, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        
        self.input_proj = nn.Linear(feat_dim, hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=int(hidden_size * mlp_ratio),
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.output_proj = nn.Linear(hidden_size, feat_dim)
    
    def timestep_embedding(self, timesteps: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(10000.0)) * torch.arange(half, dtype=torch.float32) / half
        ).to(timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        pose_cond: torch.Tensor,
    ) -> torch.Tensor:
        B, C, T, H, W = x.shape
        
        x_flat = x.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)
        x_proj = self.input_proj(x_flat)
        
        t_emb = self.timestep_embedding(t, x_proj.shape[-1])
        t_emb = self.time_embed(t_emb)
        x_proj = x_proj + t_emb.unsqueeze(1)
        
        pose_emb = self.pose_embed(pose_cond.flatten(1))
        x_proj = x_proj + pose_emb.unsqueeze(1)
        
        x_out = self.transformer(x_proj)
        x_out = self.output_proj(x_out)
        
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
            alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * 3.14159265 / 2) ** 2
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
    
    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
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


class FrequencyAwareModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.dino_extractor = DINOv2FeatureExtractor(
            output_size=(32, 32),
        )
        
        feat_dim = self.dino_extractor.feat_dim
        
        self.freq_decomp = SemanticFrequencyDecomposition(
            feat_dim=feat_dim,
            fft_size=config.model.fft_size,
            low_freq_ratio=config.model.low_freq_ratio,
        )
        
        self.bg_interpolator = BackgroundDynamicsInterpolator(
            feat_dim=feat_dim,
        )
        
        self.freq_router = FrequencyRouters(
            feat_dim=feat_dim,
        )
        
        self.diffusion_model = DiffusionModel(
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
            hidden_dim=128,
        )
    
    def compute_losses(
        self,
        exo_video: torch.Tensor,
        ego_video: torch.Tensor,
        exo_pose: torch.Tensor,
        ego_pose: torch.Tensor,
        diffusion: GaussianDiffusion,
    ) -> Dict[str, torch.Tensor]:
        B, C, T, H, W = exo_video.shape
        
        with torch.no_grad():
            exo_feat = self.dino_extractor(exo_video)
            ego_feat = self.dino_extractor(ego_video)
        
        exo_low, exo_high = self.freq_decomp(exo_feat[:, :, -1])
        ego_low, ego_high = self.freq_decomp(ego_feat[:, :, 0])
        
        ortho_loss = orthogonal_loss(exo_low, exo_high) + orthogonal_loss(ego_low, ego_high)
        
        pose_cond = torch.cat([exo_pose[:, -1:], ego_pose[:, :1]], dim=1)
        
        full_feat_sequence = torch.cat([exo_feat[:, :, -4:], ego_feat[:, :, :4]], dim=2)
        
        t = torch.randint(0, diffusion.num_timesteps, (B,), device=exo_video.device)
        noise = torch.randn_like(full_feat_sequence)
        x_t = diffusion.q_sample(full_feat_sequence, t, noise)
        
        model_output = self.diffusion_model(x_t, t, pose_cond)
        
        recon_loss = F.mse_loss(model_output, noise)
        
        w_low, w_high = self.freq_router(exo_feat[:, :, -1].detach())
        routed_exo = w_low * exo_low.detach() + w_high * exo_high.detach()
        routing_loss = F.mse_loss(routed_exo, exo_feat[:, :, -1].detach())
        
        total_loss = (
            self.config.model.lambda_recon * recon_loss
            + self.config.model.lambda_ortho * ortho_loss
            + 0.01 * routing_loss
        )
        
        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "ortho_loss": ortho_loss,
            "routing_loss": routing_loss,
        }
    
    @torch.no_grad()
    def sample(
        self,
        exo_video: torch.Tensor,
        exo_pose: torch.Tensor,
        ego_pose: torch.Tensor,
        diffusion: GaussianDiffusion,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, T, H, W = exo_video.shape
        
        with torch.no_grad():
            exo_feat = self.dino_extractor(exo_video)
        
        num_history = min(4, T)
        pose_cond = torch.cat([exo_pose[:, -1:], ego_pose[:, :1]], dim=1)
        
        total_frames = num_history + 4
        
        x = torch.randn(B, exo_feat.shape[1], total_frames, 32, 32, device=exo_video.device)
        x[:, :, :num_history] = exo_feat[:, :, -num_history:]
        
        for t in reversed(range(diffusion.num_timesteps)):
            t_batch = torch.tensor([t] * B, device=exo_video.device)
            
            model_output = self.diffusion_model(x, t_batch, pose_cond)
            
            x[:, :, num_history:] = diffusion.p_sample(
                model_output[:, :, num_history:],
                x[:, :, num_history:],
                t_batch,
            )
        
        generated_video = self.decoder(x)
        
        return generated_video, x
