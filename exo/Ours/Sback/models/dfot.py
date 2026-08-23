import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


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
        self.activation = nn.GELU()
    
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


class PatchEmbed3D(nn.Module):
    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (2, 4, 4),
        in_chans: int = 3,
        embed_dim: int = 512,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class DiffusionForcingTransformer(nn.Module):
    def __init__(
        self,
        hidden_size: int = 512,
        num_heads: int = 8,
        num_layers: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        num_timesteps: int = 1000,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_timesteps = num_timesteps
        
        self.patch_embed = PatchEmbed3D(
            patch_size=(2, 4, 4),
            in_chans=3,
            embed_dim=hidden_size,
        )
        
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
        
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_size)
        
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * 2 * 4 * 4),
        )
    
    def _reconstruct_video(self, x: torch.Tensor, output_shape: Tuple[int, int, int]) -> torch.Tensor:
        B, N, C = x.shape
        T, H, W = output_shape
        
        x = self.head(x)
        x = x.reshape(B, T // 2, H // 4, W // 4, 3, 2, 4, 4)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)
        x = x.reshape(B, 3, T, H, W)
        
        return x
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        pose_cond: Optional[torch.Tensor] = None,
        history_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, C, T, H, W = x.shape
        
        x_patch = self.patch_embed(x)
        
        t_emb = timestep_embedding(t, self.hidden_size)
        t_emb = self.time_embed(t_emb)
        
        if pose_cond is not None:
            pose_emb = self.pose_embed(pose_cond.flatten(1))
            x_patch = x_patch + pose_emb.unsqueeze(1)
        
        x_patch = x_patch + t_emb.unsqueeze(1)
        
        for block in self.blocks:
            x_patch = block(x_patch, history_mask)
        
        x_patch = self.norm(x_patch)
        
        output = self._reconstruct_video(x_patch, (T, H, W))
        
        return output


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
        
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance)
        
        self.posterior_mean_coef1 = self.beta * torch.sqrt(self.alpha_bar.roll(1)) / (1 - self.alpha_bar)
        self.posterior_mean_coef2 = (1 - self.alpha_bar.roll(1)) * torch.sqrt(self.alpha) / (1 - self.alpha_bar)
        self.posterior_mean_coef1[0] = 0
        self.posterior_mean_coef2[0] = 1
    
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
    
    def p_mean_variance(self, model_output: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        beta = self.beta[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1, 1)
        sqrt_alpha_recip = 1.0 / torch.sqrt(self.alpha[t]).view(-1, 1, 1, 1, 1)
        
        pred_mean = sqrt_alpha_recip * (x_t - beta * model_output / sqrt_one_minus_alpha_bar)
        
        posterior_variance = self.posterior_variance[t].view(-1, 1, 1, 1, 1)
        posterior_log_variance = self.posterior_log_variance_clipped[t].view(-1, 1, 1, 1, 1)
        
        return pred_mean, posterior_log_variance
    
    def p_sample(self, model_output: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        pred_mean, pred_log_variance = self.p_mean_variance(model_output, x_t, t)
        
        noise = torch.randn_like(x_t) if t[0] > 0 else 0
        
        return pred_mean + torch.exp(0.5 * pred_log_variance) * noise
