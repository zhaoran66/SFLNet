import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict


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


class FrequencyDecomposition(nn.Module):
    def __init__(
        self,
        fft_size: Tuple[int, int] = (16, 16),
        low_freq_ratio: float = 0.25,
    ):
        super().__init__()
        self.fft_size = fft_size
        self.low_freq_ratio = low_freq_ratio
        
        h, w = fft_size
        rfft_w = w // 2 + 1
        
        low_h = int(h * low_freq_ratio)
        low_w = int(rfft_w * low_freq_ratio)
        
        y, x = torch.meshgrid(torch.arange(h), torch.arange(rfft_w), indexing="ij")
        center_y, center_x = h // 2, rfft_w // 2
        dist = torch.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
        max_radius = min(low_h, low_w)
        low_freq_mask = (dist <= max_radius).float()
        
        self.register_buffer("low_freq_mask", low_freq_mask)
        self.register_buffer("high_freq_mask", 1.0 - low_freq_mask)
    
    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, T, H, W = feat.shape
        
        orig_dtype = feat.dtype
        feat = feat.float()
        
        feat_reshaped = feat.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        
        feat_resized = F.interpolate(feat_reshaped, size=self.fft_size, mode="bilinear", align_corners=False)
        
        feat_fft = torch.fft.rfft2(feat_resized, norm="ortho")
        feat_fft_shifted = torch.fft.fftshift(feat_fft, dim=(-2, -1))
        
        low_mask = self.low_freq_mask.view(1, 1, *self.fft_size[:-1], -1)
        high_mask = self.high_freq_mask.view(1, 1, *self.fft_size[:-1], -1)
        
        feat_low_fft = feat_fft_shifted * low_mask
        feat_high_fft = feat_fft_shifted * high_mask
        
        feat_low = torch.fft.irfft2(torch.fft.ifftshift(feat_low_fft, dim=(-2, -1)), norm="ortho")
        feat_high = torch.fft.irfft2(torch.fft.ifftshift(feat_high_fft, dim=(-2, -1)), norm="ortho")
        
        feat_low = F.interpolate(feat_low, size=(H, W), mode="bilinear", align_corners=False)
        feat_high = F.interpolate(feat_high, size=(H, W), mode="bilinear", align_corners=False)
        
        feat_low = feat_low.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)
        feat_high = feat_high.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)
        
        feat_low = feat_low.to(orig_dtype)
        feat_high = feat_high.to(orig_dtype)
        
        return feat_low, feat_high


def orthogonal_loss(feat_low: torch.Tensor, feat_high: torch.Tensor) -> torch.Tensor:
    B, C, T, H, W = feat_low.shape
    
    feat_low_flat = feat_low.reshape(B, C, -1)
    feat_high_flat = feat_high.reshape(B, C, -1)
    
    feat_low_norm = F.normalize(feat_low_flat, dim=1)
    feat_high_norm = F.normalize(feat_high_flat, dim=1)
    
    dot_product = torch.sum(feat_low_norm * feat_high_norm, dim=1)
    ortho_loss = torch.mean(torch.abs(dot_product))
    
    return ortho_loss


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


class DiffusionForcingTransformerWithFreq(nn.Module):
    def __init__(
        self,
        hidden_size: int = 512,
        num_heads: int = 8,
        num_layers: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        fft_size: Tuple[int, int] = (16, 16),
        low_freq_ratio: float = 0.25,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        
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
            nn.LayerNorm(hidden_size // 2),
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
        
        self.freq_decomp = FrequencyDecomposition(
            fft_size=fft_size,
            low_freq_ratio=low_freq_ratio,
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
        compute_freq_loss: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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
        
        if compute_freq_loss:
            feat_low, feat_high = self.freq_decomp(output)
            ortho_loss_val = orthogonal_loss(feat_low, feat_high)
            freq_losses = {"ortho_loss": ortho_loss_val}
        else:
            freq_losses = {"ortho_loss": torch.tensor(0.0, device=output.device)}
        
        return output, freq_losses


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
        
        posterior_log_variance = self.posterior_log_variance_clipped[t].view(-1, 1, 1, 1, 1)
        
        return pred_mean, posterior_log_variance
    
    def p_sample(self, model_output: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        pred_mean, pred_log_variance = self.p_mean_variance(model_output, x_t, t)
        
        noise = torch.randn_like(x_t) if t[0] > 0 else 0
        
        return pred_mean + torch.exp(0.5 * pred_log_variance) * noise


class VideoInterpolator(nn.Module):
    def __init__(self, num_interp_frames: int = 4, hidden_dim: int = 64, num_res_blocks: int = 3):
        super().__init__()
        self.num_interp_frames = num_interp_frames
        
        self.in_conv = nn.Conv3d(3, hidden_dim, 3, padding=1)
        
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1),
                nn.GroupNorm(8, hidden_dim),
                nn.SiLU(),
                nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1),
                nn.GroupNorm(8, hidden_dim),
            ) for _ in range(num_res_blocks)
        ])
        
        self.out_conv = nn.Sequential(
            nn.Conv3d(hidden_dim, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(32, 3, 3, padding=1),
            nn.Tanh(),
        )
    
    def interpolate(self, start_frame: torch.Tensor, end_frame: torch.Tensor) -> torch.Tensor:
        batch_size = start_frame.shape[0]
        
        x = torch.stack([start_frame, end_frame], dim=2)
        
        x = F.interpolate(
            x,
            size=(self.num_interp_frames + 2,) + x.shape[3:],
            mode="trilinear",
            align_corners=True,
        )
        
        h = self.in_conv(x)
        
        for block in self.res_blocks:
            h = h + block(h)
        
        out = self.out_conv(h)
        
        out_start = start_frame.unsqueeze(2)
        out_end = end_frame.unsqueeze(2)
        out_middle = out[:, :, 1:-1]
        
        out = torch.cat([out_start, out_middle, out_end], dim=2)
        
        return out
    
    def forward(self, exo_video: torch.Tensor, ego_video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        exo_last = exo_video[:, :, -1]
        ego_first = ego_video[:, :, 0]
        
        interpolated = self.interpolate(exo_last, ego_first)
        interp_frames = interpolated[:, :, 1:-1]
        
        full_sequence = torch.cat([exo_video, interp_frames, ego_video], dim=2)
        
        return full_sequence, interp_frames


class Syn2SeqWithFrequency(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.model = DiffusionForcingTransformerWithFreq(
            hidden_size=config.model.hidden_size,
            num_heads=config.model.num_heads,
            num_layers=config.model.num_layers,
            mlp_ratio=config.model.mlp_ratio,
            dropout=config.model.dropout,
            fft_size=config.model.fft_size,
            low_freq_ratio=config.model.low_freq_ratio,
        )
        
        self.interpolator = VideoInterpolator(
            num_interp_frames=config.data.num_interp_frames,
            hidden_dim=64,
            num_res_blocks=3,
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
        
        full_sequence, interp_frames = self.interpolator(exo_video, ego_video)
        
        pose_cond = torch.cat([exo_pose[:, -1:], ego_pose[:, :1]], dim=1)
        
        use_exo_to_interp = torch.rand(1).item() > 0.5
        
        if use_exo_to_interp:
            train_sequence = torch.cat([exo_video[:, :, -4:], interp_frames], dim=2)
        else:
            train_sequence = torch.cat([interp_frames, ego_video[:, :, :4]], dim=2)
        
        t = torch.randint(0, diffusion.num_timesteps, (B,), device=exo_video.device)
        noise = torch.randn_like(train_sequence)
        x_t = diffusion.q_sample(train_sequence, t, noise)
        
        model_output, freq_losses = self.model(x_t, t, pose_cond)
        
        recon_loss = F.mse_loss(model_output, noise)
        
        total_loss = (
            recon_loss
            + self.config.model.lambda_ortho * freq_losses["ortho_loss"]
        )
        
        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "ortho_loss": freq_losses["ortho_loss"],
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
        
        pose_cond = torch.cat([exo_pose[:, -1:], ego_pose[:, :1]], dim=1)
        
        num_history = min(4, T)
        num_gen = self.config.data.num_interp_frames
        
        x = torch.randn(B, 3, num_history + num_gen + 4, H, W, device=exo_video.device)
        x[:, :, :num_history] = exo_video[:, :, -num_history:]
        
        for t in reversed(range(diffusion.num_timesteps)):
            t_batch = torch.tensor([t] * B, device=exo_video.device)
            
            sqrt_alpha_bar_t = diffusion.sqrt_alpha_bar[t_batch]
            sqrt_one_minus_alpha_bar_t = diffusion.sqrt_one_minus_alpha_bar[t_batch]
            
            x_noisy_history = (
                sqrt_alpha_bar_t.view(-1, 1, 1, 1, 1) * exo_video[:, :, -num_history:]
                + sqrt_one_minus_alpha_bar_t.view(-1, 1, 1, 1, 1) * torch.randn_like(exo_video[:, :, -num_history:])
            )
            
            x[:, :, :num_history] = x_noisy_history
            
            model_output, _ = self.model(x, t_batch, pose_cond, compute_freq_loss=False)
            
            x[:, :, num_history:] = diffusion.p_sample(
                model_output[:, :, num_history:],
                x[:, :, num_history:],
                t_batch,
            )
        
        x = torch.tanh(x)
        generated_frames = x[:, :, num_history:]
        
        return generated_frames, x
