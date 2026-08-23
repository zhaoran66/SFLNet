"""
Combine Trainer: SD-VAE Content Latent + DINO Low-Frequency Background Latent

Pipeline:
    RGB video (B, 3, T, 128, 128)
       |--- VAE.encode -> content_latent (B, 4, T, 16, 16)
       |--- DINO.extract + FreqDecomp -> bg_feat (B, 384, T, 32, 32)
             |--- BackgroundLatentProjector -> bg_latent (B, 4, T, 16, 16)
    concat -> combined_latent (B, 8, T, 16, 16) -> DiffusionForcingTransformer
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from typing import Optional
from tqdm import tqdm


class ResBlock3D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv1 = nn.Conv3d(dim, dim, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, dim)
        self.conv2 = nn.Conv3d(dim, dim, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, dim)
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x + residual


class LatentInterpolator(nn.Module):
    def __init__(self, latent_channels: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.latent_channels = latent_channels
        self.in_conv = nn.Conv3d(latent_channels, hidden_dim, 3, padding=1)
        self.res_blocks = nn.ModuleList([ResBlock3D(hidden_dim) for _ in range(3)])
        self.out_conv = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(hidden_dim // 2, latent_channels, 3, padding=1),
        )

    def forward(self, start: torch.Tensor, end: torch.Tensor, num_interp: int) -> torch.Tensor:
        x = torch.stack([start, end], dim=2)
        x = F.interpolate(x, size=(num_interp + 2,) + x.shape[3:], mode="trilinear", align_corners=False)
        h = self.in_conv(x)
        for block in self.res_blocks:
            h = block(h)
        delta = self.out_conv(h)
        return x + delta


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half)
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

    def forward(self, x: torch.Tensor, mask=None):
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


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, int(hidden_size * mlp_ratio))
        self.fc2 = nn.Linear(int(hidden_size * mlp_ratio), hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        h = self.fc2(self.dropout(self.activation(self.fc1(self.norm2(x)))))
        x = x + self.dropout(h)
        return x


class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class CombinedDiffusionTransformer(nn.Module):
    def __init__(
        self,
        in_channels=8, hidden_size=256, num_heads=4, num_layers=6,
        mlp_ratio=4.0, dropout=0.1, patch_size=(2, 2, 2),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed3D(patch_size, in_chans=in_channels, embed_dim=hidden_size)
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.pose_embed = nn.Sequential(
            nn.Linear(32, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        pt, ph, pw = patch_size
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, in_channels * pt * ph * pw),
        )

    def _reconstruct(self, x, output_shape):
        B, N, _ = x.shape
        T, H, W = output_shape
        pt, ph, pw = self.patch_size
        x = self.head(x).reshape(B, T // pt, H // ph, W // pw, self.in_channels, pt, ph, pw)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(B, self.in_channels, T, H, W)
        return x

    def forward(self, x, t, pose_cond=None, history_mask=None):
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
        output = self._reconstruct(x_patch, (T, H, W))
        return output


class GaussianDiffusion:
    def __init__(self, num_timesteps=1000, beta_schedule="linear"):
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
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")
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

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha_bar = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1, 1)
        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

    def p_mean_variance(self, model_output, x_t, t):
        beta = self.beta[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1, 1)
        sqrt_alpha_recip = 1.0 / torch.sqrt(self.alpha[t]).view(-1, 1, 1, 1, 1)
        pred_mean = sqrt_alpha_recip * (x_t - beta * model_output / sqrt_one_minus_alpha_bar)
        return pred_mean

    def p_sample(self, model_output, x_t, t):
        pred_mean = self.p_mean_variance(model_output, x_t, t)
        noise = torch.randn_like(x_t) if t[0] > 0 else 0
        log_var = torch.log(self.posterior_variance[t]).view(-1, 1, 1, 1, 1)
        return pred_mean + torch.exp(0.5 * log_var) * noise


class CombineTrainer:
    def __init__(
        self, vae, dino_extractor, freq_decomp, bg_projector,
                 diff_model, diffusion, train_loader, val_loader, config,
    ):
        self.vae = vae
        self.dino_extractor = dino_extractor
        self.freq_decomp = freq_decomp
        self.bg_projector = bg_projector
        self.model = diff_model
        self.diffusion = diffusion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.device = torch.device(config.training.device)

        trainable_params = list(self.model.parameters()) + list(self.bg_projector.parameters())
        self.optimizer = torch.optim.AdamW(trainable_params, lr=config.training.lr, weight_decay=config.training.weight_decay)
        self.scaler = GradScaler(enabled=config.training.use_amp)
        self.global_step = 0

        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, "samples"), exist_ok=True)

        torch.cuda.empty_cache()
        import gc
        gc.collect()

    def _encode(self, video):
        with torch.no_grad():
            content_latent = self.vae.encode(video)
            dino_feat = self.dino_extractor(video.half())
            bg_feat, _ = self.freq_decomp(dino_feat)
            del dino_feat
        bg_latent = self.bg_projector(bg_feat)
        return torch.cat([content_latent, bg_latent], dim=1)

    def _compute_loss(self, x_0, pose_cond, cfg_dropout_prob):
        B = x_0.shape[0]
        t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=self.device)
        noise = torch.randn_like(x_0)
        x_t = self.diffusion.q_sample(x_0, t, noise)

        use_cfg = torch.rand(B, device=self.device) > cfg_dropout_prob
        pose_input = pose_cond if use_cfg.any() else None

        with autocast(enabled=self.config.training.use_amp):
            model_output = self.model(x_t, t, pose_input)
            loss = F.mse_loss(model_output, noise)
        return loss

    def _train_epoch(self, epoch):
        self.model.train()
        self.bg_projector.train()
        self.vae.eval()
        self.dino_extractor.eval()

        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch in pbar:
            exo_v = batch["exo_video"].to(self.device, non_blocking=True)
            ego_v = batch["ego_video"].to(self.device, non_blocking=True)
            exo_p = batch["exo_pose"].to(self.device, non_blocking=True)
            ego_p = batch["ego_pose"].to(self.device, non_blocking=True)

            with autocast(enabled=self.config.training.use_amp):
                exo_z = self._encode(exo_v)
                ego_z = self._encode(ego_v)
                pose_cond = torch.cat([exo_p[:, -1:], ego_p[:, :1]], dim=1)
                train_seq = torch.cat([exo_z[:, :, -4:], ego_z[:, :, :4]], dim=2)
                loss = self._compute_loss(train_seq, pose_cond, self.config.training.cfg_dropout_prob)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            if self.config.training.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.bg_projector.parameters()),
                    self.config.training.grad_clip,
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            self.global_step += 1
            pbar.set_postfix(loss=loss.item())

        return total_loss / max(1, len(self.train_loader))

    @torch.no_grad()
    def _validate(self, epoch):
        self.model.eval()
        self.bg_projector.eval()
        total_loss = 0.0
        for batch in tqdm(self.val_loader, desc="Val"):
            exo_v = batch["exo_video"].to(self.device)
            ego_v = batch["ego_video"].to(self.device)
            exo_p = batch["exo_pose"].to(self.device)
            ego_p = batch["ego_pose"].to(self.device)
            exo_z = self._encode(exo_v)
            ego_z = self._encode(ego_v)
            pose_cond = torch.cat([exo_p[:, -1:], ego_p[:, :1]], dim=1)
            train_seq = torch.cat([exo_z[:, :, -4:], ego_z[:, :, :4]], dim=2)
            loss = self._compute_loss(train_seq, pose_cond, 0.0)
            total_loss += loss.item()
        if epoch % self.config.training.save_interval == 0:
            self._save_checkpoint(epoch)
        return total_loss / max(1, len(self.val_loader))

    def _save_checkpoint(self, epoch):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "bg_projector_state_dict": self.bg_projector.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "global_step": self.global_step,
            "config": self.config,
        }
        torch.save(
            checkpoint,
            os.path.join(self.config.output_dir, "checkpoints", f"checkpoint_{epoch}.pt"),
        )

    def train(self):
        for epoch in range(self.config.training.num_epochs):
            train_loss = self._train_epoch(epoch)
            val_loss = self._validate(epoch)
            print(f"Epoch {epoch}: Train {train_loss:.4f}, Val {val_loss:.4f}")
