"""
Method C: Syn2Seq + SD-VAE Latent + Background Processing

Components:
1. SD-VAE for image encoding/decoding
2. Soft Spectral Decomposition (DCT + Gaussian masks)
3. Structure-Weighted Asymmetric Loss
4. Latent-space Diffusion Transformer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math
from typing import Optional, Tuple

from .method_c_vae import FrameVAE
from .freq_routing import (
    SoftSpectralDecomposition,
    create_gaussian_pose_mask,
    structure_weighted_asymmetric_loss,
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
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        in_chans: int = 4,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class LatentDiffusionTransformerWithBG(nn.Module):
    """
    Method C: Complete Latent + Background Processing Diffusion Model

    Pipeline:
    1. SD-VAE: RGB -> Latent (4x downsampling)
    2. Spectral Decomposition: high/low frequency splitting
    3. Diffusion Transformer: generation in latent space
    4. Structure-Weighted Loss: foreground/background weighted reconstruction
    """
    def __init__(
        self,
        in_channels: int = 4,
        hidden_size: int = 256,
        num_heads: int = 4,
        num_layers: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        num_timesteps: int = 1000,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        use_freq_decomp: bool = True,
        freq_size: Tuple[int, int] = (8, 8),
        spectral_sigma: float = 0.5,
        use_dct: bool = True,
        pose_mask_sigma: float = 0.1,
        lambda_struct_fg: float = 1.0,
        lambda_struct_bg: float = 1.0,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_timesteps = num_timesteps
        self.use_freq_decomp = use_freq_decomp
        self.lambda_struct_fg = lambda_struct_fg
        self.lambda_struct_bg = lambda_struct_bg
        self.use_checkpoint = use_checkpoint

        self.patch_embed = PatchEmbed3D(patch_size, in_channels, hidden_size)

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

        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_size)

        self.patch_size_t, self.patch_size_h, self.patch_size_w = patch_size

        self.output_proj = nn.Linear(hidden_size, in_channels * self.patch_size_t * self.patch_size_h * self.patch_size_w)

        if use_freq_decomp:
            self.spectral_decomp = SoftSpectralDecomposition(
                freq_size=freq_size,
                feat_dim=in_channels,
                sigma=spectral_sigma,
                use_dct=use_dct,
            )
        self.pose_mask_sigma = pose_mask_sigma

    def unpatchify(self, x: torch.Tensor, output_shape: Tuple[int, int, int, int, int]) -> torch.Tensor:
        B, N, C = x.shape
        T, H, W = output_shape[2], output_shape[3], output_shape[4]

        num_patches_t = T // self.patch_size_t
        num_patches_h = H // self.patch_size_h
        num_patches_w = W // self.patch_size_w

        x = x.reshape(B, num_patches_t, num_patches_h, num_patches_w, C)
        x = x.reshape(B, num_patches_t, num_patches_h, num_patches_w,
                      self.patch_size_t, self.patch_size_h, self.patch_size_w, -1)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
        x = x.reshape(B, -1, T, H, W)
        return x

    def forward(self, x: torch.Tensor, t: torch.Tensor, pose_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, T, H, W = x.shape

        x_patch = self.patch_embed(x)

        t_emb = timestep_embedding(t, self.hidden_size)
        t_emb = self.time_embed(t_emb)
        x_patch = x_patch + t_emb.unsqueeze(1)

        if pose_cond is not None:
            pose_emb = self.pose_embed(pose_cond.flatten(1))
            x_patch = x_patch + pose_emb.unsqueeze(1)

        for layer in self.layers:
            if self.use_checkpoint and self.training:
                x_patch = checkpoint(layer, x_patch, None, use_reentrant=False)
            else:
                x_patch = layer(x_patch)

        x_patch = self.norm(x_patch)
        x_out = self.output_proj(x_patch)
        x_out = self.unpatchify(x_out, (B, C, T, H, W))

        return x_out

    def compute_losses(
        self,
        exo_video: torch.Tensor,
        ego_video: torch.Tensor,
        exo_pose: torch.Tensor,
        ego_pose: torch.Tensor,
        diffusion,
        vae: FrameVAE,
    ):
        """
        Compute diffusion + structure-weighted losses

        Args:
            exo_video: (B, 3, T, H, W) external view video
            ego_video: (B, 3, T, H, W) ego view video
            exo_pose: (B, T, 4, 4) external view pose
            ego_pose: (B, T, 4, 4) ego view pose
            diffusion: Gaussian diffusion object
            vae: FrameVAE encoder/decoder

        Returns:
            loss dict
        """
        B, C, T, H, W = exo_video.shape

        num_history = min(4, T)
        pose_for_mask = torch.cat([exo_pose[:, -num_history:], ego_pose[:, :4]], dim=1)

        with torch.no_grad():
            exo_z = vae.encode(exo_video)
            ego_z = vae.encode(ego_video)

        full_z_sequence = torch.cat([exo_z[:, :, -num_history:], ego_z[:, :, :4]], dim=2)

        t = torch.randint(0, diffusion.num_timesteps, (B,), device=exo_video.device)
        noise = torch.randn_like(full_z_sequence)
        x_t = diffusion.q_sample(full_z_sequence, t, noise)

        pose_cond = torch.cat([exo_pose[:, -1:], ego_pose[:, :1]], dim=1)
        model_output = self.forward(x_t, t, pose_cond)

        diffusion_loss = F.mse_loss(model_output, noise)

        pred_z = x_t + model_output
        target_z = full_z_sequence

        recon_loss_z = F.mse_loss(pred_z, target_z)

        pred_video = vae.decode(pred_z)
        target_video = torch.cat([exo_video[:, :, -num_history:], ego_video[:, :, :4]], dim=2)

        recon_loss_rgb = F.mse_loss(pred_video, target_video)

        if self.use_freq_decomp:
            pose_mask = create_gaussian_pose_mask(
                pose_for_mask,
                spatial_size=(num_history + 4, target_video.shape[3], target_video.shape[4]),
                sigma=self.pose_mask_sigma,
            )

            _, fg_loss, bg_loss = structure_weighted_asymmetric_loss(
                pred_video, target_video, pose_mask
            )

            total_loss = (
                diffusion_loss
                + self.lambda_struct_fg * fg_loss
                + self.lambda_struct_bg * bg_loss
            )
        else:
            fg_loss = torch.tensor(0.0, device=exo_video.device)
            bg_loss = torch.tensor(0.0, device=exo_video.device)
            total_loss = diffusion_loss + recon_loss_rgb

        return {
            "total_loss": total_loss,
            "diffusion_loss": diffusion_loss,
            "recon_loss_z": recon_loss_z,
            "recon_loss_rgb": recon_loss_rgb,
            "fg_loss": fg_loss,
            "bg_loss": bg_loss,
        }
