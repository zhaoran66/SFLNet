"""
Stage 2: Diffusion-based Pixel Hallucination
Latent Diffusion Model for exo-to-ego image synthesis conditioned on ego layout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


# ============================================================================
# UNetܹ - ȥ
# ============================================================================

class SinusoidalPosEmb(nn.Module):
    """λñ"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResidualBlock(nn.Module):
    """ԾӵĲв"""

    def __init__(self, in_channels, out_channels, time_dim=None, condition_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, out_channels),
            nn.SiLU(inplace=True)
        ) if time_dim else None
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb=None, cond=None):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        if time_emb is not None and self.time_mlp is not None:
            h = h + self.time_mlp(time_emb)[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """ռע"""

    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels
        self.head_dim = channels // num_heads

        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.reshape(B, C, H * W)

        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        q = q.reshape(B, self.num_heads, self.head_dim, H * W)
        k = k.reshape(B, self.num_heads, self.head_dim, H * W)
        v = v.reshape(B, self.num_heads, self.head_dim, H * W)

        # Attention
        scale = self.head_dim ** -0.5
        attn = torch.einsum('bhdn,bhdm->bhnm', q, k) * scale
        attn = attn.softmax(dim=-1)

        out = torch.einsum('bhnm,bhdm->bhdn', attn, v)
        out = out.reshape(B, C, H * W)
        out = self.proj(out)
        return (out + h).reshape(B, C, H, W)


class DownBlock(nn.Module):
    """²"""

    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.res1 = ResidualBlock(in_channels, out_channels, time_dim)
        self.res2 = ResidualBlock(out_channels, out_channels, time_dim)
        self.down = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, time_emb=None):
        x = self.res1(x, time_emb)
        x = self.res2(x, time_emb)
        skip = x
        x = self.down(x)
        return x, skip


class UpBlock(nn.Module):
    """ϲ"""

    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.res1 = ResidualBlock(in_channels, out_channels, time_dim)
        self.res2 = ResidualBlock(out_channels, out_channels, time_dim)
        self.up = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x, skip, time_emb=None):
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, time_emb)
        x = self.res2(x, time_emb)
        x = self.up(x)
        return x


class UNet(nn.Module):
    """
    U-Netȥ
    Latent Diffusion Modelȥ
    """

    def __init__(
        self,
        in_channels=4,  # 4ͨ: RGB + layout alpha
        out_channels=3,  # RGB
        base_channels=128,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        time_dim=256,
        attention_resolutions=(16,),
        num_heads=8
    ):
        super().__init__()

        self.base_channels = base_channels
        self.channel_mults = channel_mults

        # ʱ䲽Ƕ
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(inplace=True),
            nn.Linear(time_dim * 4, time_dim)
        )

        # ʼ
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        #  (²)
        self.downs = nn.ModuleList()
        ch = base_channels
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.downs.append(DownBlock(ch, out_ch, time_dim))
                ch = out_ch
            if i < len(channel_mults) - 1:
                self.downs.append(nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1))

        # м
        self.mid_block = nn.Sequential(
            ResidualBlock(ch, ch, time_dim),
            AttentionBlock(ch, num_heads),
            ResidualBlock(ch, ch, time_dim)
        )

        #  (ϲ)
        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):
                self.ups.append(UpBlock(ch + out_ch, out_ch, time_dim))
                ch = out_ch

        # 
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, time_steps, condition=None):
        """
        Args:
            x: [B, C, H, W] Ǳ
            time_steps: [B] ʱ䲽
            condition: [B, C, H, W]  (layout + exo frame)
        Returns:
            noise_pred: [B, C, H, W] Ԥ
        """
        # ʱ䲽Ƕ
        t_emb = self.time_mlp(time_steps)

        # ʼ
        h = self.init_conv(x)
        if condition is not None:
            h = torch.cat([h, condition], dim=1)

        # 
        skips = []
        for layer in self.downs:
            if isinstance(layer, DownBlock):
                h, skip = layer(h, t_emb)
                skips.append(skip)
            else:
                h = layer(h)

        # м
        h = self.mid_block(h, t_emb)

        # 
        for layer in self.ups:
            if isinstance(layer, UpBlock):
                skip = skips.pop()
                h = layer(h, skip, t_emb)
            else:
                h = layer(h)

        return self.out_conv(h)


# ============================================================================
# VAE - ԱǱռ
# ============================================================================

class VAEEncoder(nn.Module):
    """VAE"""

    def __init__(self, in_channels=3, latent_dim=4, base_channels=64):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            # 256 -> 128
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(inplace=True),
            # 128 -> 64
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU(inplace=True),
            # 64 -> 32
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU(inplace=True),
            # 32 -> 16
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 8),
            nn.SiLU(inplace=True),
        )

        self.to_latent = nn.Sequential(
            nn.Conv2d(base_channels * 8, latent_dim * 2, kernel_size=3, padding=1),
        )

    def forward(self, x):
        h = self.encoder(x)
        params = self.to_latent(h)
        mean, logvar = params.chunk(2, dim=1)
        logvar = torch.clamp(logvar, -30, 20)
        std = torch.exp(0.5 * logvar)
        z = mean + std * torch.randn_like(std)
        return z, mean, logvar


class VAEDecoder(nn.Module):
    """VAE"""

    def __init__(self, latent_dim=4, out_channels=3, base_channels=64):
        super().__init__()

        self.from_latent = nn.Sequential(
            nn.Conv2d(latent_dim, base_channels * 8, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels * 8),
            nn.SiLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            # 16 -> 32
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU(inplace=True),
            # 32 -> 64
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU(inplace=True),
            # 64 -> 128
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(inplace=True),
            # 128 -> 256
            nn.ConvTranspose2d(base_channels, base_channels // 2, kernel=4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels // 2),
            nn.SiLU(inplace=True),
            # 
            nn.Conv2d(base_channels // 2, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, z):
        h = self.from_latent(z)
        return self.decoder(h)


class VAE(nn.Module):
    """VAEģ"""

    def __init__(self, in_channels=3, latent_dim=4, base_channels=64):
        super().__init__()
        self.encoder = VAEEncoder(in_channels, latent_dim, base_channels)
        self.decoder = VAEDecoder(latent_dim, in_channels, base_channels)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z, mean, logvar = self.encode(x)
        recon = self.decode(z)
        return recon, mean, logvar


# ============================================================================
# Ǳռɢģ
# ============================================================================

class LatentDiffusionModel(nn.Module):
    """
    Ǳռɢģ (Latent Diffusion Model)
    Stage 2: Diffusion-based Pixel Hallucination
    """

    def __init__(
        self,
        vae_config: Dict,
        unet_config: Dict,
        diffusion_config: Dict
    ):
        super().__init__()

        # VAE for latent space
        self.vae = VAE(
            in_channels=vae_config.get('in_channels', 3),
            latent_dim=vae_config.get('latent_dim', 4),
            base_channels=vae_config.get('base_channels', 64)
        )

        # U-Net for denoising
        self.unet = UNet(
            in_channels=unet_config.get('in_channels', 4 + 4),  # noise + condition channels
            out_channels=unet_config.get('out_channels', 4),
            base_channels=unet_config.get('base_channels', 128),
            channel_mults=unet_config.get('channel_mults', (1, 2, 4, 8)),
            num_res_blocks=unet_config.get('num_res_blocks', 2),
            time_dim=unet_config.get('time_dim', 256),
            attention_resolutions=unet_config.get('attention_resolutions', (16,)),
            num_heads=unet_config.get('num_heads', 8)
        )

        # Diffusion parameters
        self.num_timesteps = diffusion_config.get('num_timesteps', 1000)
        self.beta_schedule = diffusion_config.get('beta_schedule', 'linear')
        self.beta_start = diffusion_config.get('beta_start', 0.0001)
        self.beta_end = diffusion_config.get('beta_end', 0.02)

        # Register buffers for diffusion parameters
        self.register_diffusion_buffers()

    def register_diffusion_buffers(self):
        """עɢ̵Ĳ"""
        if self.beta_schedule == 'linear':
            betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        elif self.beta_schedule == 'cosine':
            steps = self.num_timesteps + 1
            x = torch.linspace(0, self.num_timesteps, steps)
            alphas_cumprod = torch.cos(((x / self.num_timesteps) + 0.008) * torch.pi * 0.5) ** 2
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # 
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))

    def q_sample(self, x_start, t, noise=None):
        """ǰɢ: """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise, noise

    def p_losses(self, x_start, condition, t, noise=None):
        """ȥʧ"""
        if noise is None:
            noise = torch.randn_like(x_start)

        # 뵽Ǳռ
        z_start, _, _ = self.vae.encode(x_start)

        # 
        z_noisy, added_noise = self.q_sample(z_start, t, noise)

        # 
        cond_features = self.encode_condition(condition)

        # ȥԤ
        noise_pred = self.unet(z_noisy, t, cond_features)

        # MSEʧ
        loss = F.mse_loss(noise_pred, added_noise, reduction='mean')
        return loss

    def encode_condition(self, condition):
        """
         (layout + exo frame)
        ѾȾõego layoutͼ
        """
        # ʹVAE
        if condition.shape[1] == 4:  # RGBA layout
            cond = condition[:, :3]  # ȡRGB
        else:
            cond = condition

        with torch.no_grad():
            cond_z, _, _ = self.vae.encode(cond)

        return cond_z

    def p_mean(self, z_noisy, t, condition, noise_pred):
        """ȥľֵ"""
        sqrt_recip_alphas_t = self.sqrt_recip_alphas_cumprod[t][:, None, None, None]
        sqrt_recipm1_alphas_cumprod_t = self.sqrt_recipm1_alphas_cumprod[t][:, None, None, None]

        model_mean = sqrt_recip_alphas_t * (
            z_noisy - sqrt_recipm1_alphas_cumprod_t * noise_pred
        )
        return model_mean

    @torch.no_grad()
    def p_sample(self, z_noisy, t, condition):
        """ȥ"""
        t_tensor = torch.full((z_noisy.shape[0],), t, device=z_noisy.device, dtype=torch.long)

        # 
        cond_features = self.encode_condition(condition)

        # Ԥ
        noise_pred = self.unet(z_noisy, t_tensor, cond_features)

        # ȥľֵ
        model_mean = self.p_mean(z_noisy, t_tensor, condition, noise_pred)

        if t == 0:
            return model_mean
        else:
            alphas_t = self.alphas[t][:, None, None, None]
            alphas_cumprod_t = self.alphas_cumprod[t][:, None, None, None]
            betas_t = self.betas[t][:, None, None, None]

            posterior_variance = betas_t * (1. - alphas_cumprod_t) / (1. - alphas_cumprod_t)
            log_variance = torch.log(posterior_variance.clamp(min=1e-20))

            noise = torch.randn_like(z_noisy)
            return model_mean + torch.exp(0.5 * log_variance) * noise

    @torch.no_grad()
    def p_sample_loop(self, condition, shape, device):
        """ȥѭ"""
        # Ӵʼ
        z_noisy = torch.randn(shape, device=device)

        # ȥ
        for t in reversed(range(self.num_timesteps)):
            z_noisy = self.p_sample(z_noisy, t, condition)

        # 뵽ͼռ
        reconstructed = self.vae.decode(z_noisy)
        return reconstructed

    def forward(self, x_start, condition, t):
        """ѵʱǰ򴫲"""
        return self.p_losses(x_start, condition, t)


# ============================================================================
# 򻯰ɢģ (ڿʵ)
# ============================================================================

class SimplifiedDiffusionModel(nn.Module):
    """
    򻯰ɢģ
    Դ޵ĳ
    """

    def __init__(
        self,
        image_size=256,
        in_channels=3,
        out_channels=4,
        base_channels=64,
        latent_dim=4,
        num_timesteps=100
    ):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim

        # 򻯵ʱ䲽Ƕ
        self.time_embed = nn.Sequential(
            nn.Embedding(num_timesteps, base_channels * 4),
            nn.Linear(base_channels * 4, base_channels * 4),
            nn.SiLU(inplace=True),
            nn.Linear(base_channels * 4, base_channels * 4),
        )

        # 򻯵U-Net
        self.unet = SimplifiedUNet(
            in_channels=in_channels + 4,  # RGB + layout
            out_channels=out_channels,
            base_channels=base_channels,
            time_dim=base_channels * 4
        )

        # VAE
        self.vae = SimplifiedVAE(in_channels, latent_dim, base_channels)

        # ɢ
        self.num_timesteps = num_timesteps
        self.register_buffer('betas', torch.linspace(0.0001, 0.02, num_timesteps))
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_start, t, noise=None):
        """ǰɢ"""
        if noise is None:
            noise = torch.randn_like(x_start)

        alphas_t = self.alphas_cumprod[t.cpu()][:, None, None, None].to(t.device)
        sqrt_alphas = torch.sqrt(alphas_t)
        sqrt_one_minus_alphas = torch.sqrt(1.0 - alphas_t)

        return sqrt_alphas * x_start + sqrt_one_minus_alphas * noise, noise

    def p_losses(self, x_start, condition, t):
        """loss"""
        # encode x_start (ego frame) to latent space
        x_start_z = self.vae.encode(x_start)  # [B,4,H/8,W/8]
        noise = torch.randn_like(x_start_z)
        x_noisy, added_noise = self.q_sample(x_start_z, t, noise)
        # resize condition (exo frame 3ch) to match latent spatial size
        cond_resized = F.interpolate(
            condition, size=x_noisy.shape[-2:],
            mode='bilinear', align_corners=False
        )  # [B,3,H/8,W/8]
        # concat: noise(4ch) + condition(3ch) = 7ch
        x_input = torch.cat([x_noisy, cond_resized], dim=1)
        # time embedding
        t_emb = self.time_embed(t)
        # predict noise
        noise_pred = self.unet(x_input, t_emb)
        return F.mse_loss(noise_pred, added_noise)
    @torch.no_grad()
    def sample(self, condition, device):
        """"""
        B = condition.shape[0]
        x = torch.randn(B, 4, self.image_size // 8, self.image_size // 8, device=device)

        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            import torch.nn.functional as F
            cond_resized = F.interpolate(condition, size=x.shape[-2:], mode="bilinear", align_corners=False)
            noise_pred = self.unet(torch.cat([x, cond_resized], dim=1), self.time_embed(t_batch))

            alphas_t = self.alphas[t]
            alphas_cumprod_t = self.alphas_cumprod[t]
            betas_t = 1 - alphas_t

            x = (x - betas_t / torch.sqrt(1 - alphas_cumprod_t) * noise_pred) / torch.sqrt(alphas_t)

            if t > 0:
                x = x + torch.sqrt(betas_t) * torch.randn_like(x)

        return x

    def forward(self, x_start, condition, t):
        return self.p_losses(x_start, condition, t)


class SimplifiedVAE(nn.Module):
    """򻯰VAE"""

    def __init__(self, in_channels=3, latent_dim=4, base_channels=32):
        super().__init__()
        self.latent_dim = latent_dim

        # 
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, latent_dim * 2, 3, 1, 1),
        )

        # 
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, base_channels * 4, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels, in_channels, 4, 2, 1),
        )

    def encode(self, x):
        params = self.encoder(x)
        mean, logvar = params.chunk(2, dim=1)
        std = torch.exp(0.5 * logvar)
        z = mean + std * torch.randn_like(std)
        return z

    def decode(self, z):
        return self.decoder(z)


class SimplifiedUNet(nn.Module):
    """򻯰U-Net"""

    def __init__(self, in_channels=7, out_channels=3, base_channels=64, time_dim=256):
        super().__init__()

        self.time_mlp = nn.Linear(time_dim, base_channels * 4)

        # 
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        # ʱ
        self.time_conv = nn.Sequential(
            nn.Linear(base_channels * 4, base_channels * 4),
            nn.ReLU(inplace=True)
        )

        # 
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, out_channels, 3, 1, 1)
        )

    def forward(self, x, time_emb):
        # 
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # ʱ
        t = self.time_mlp(time_emb)
        t = self.time_conv(t)[:, :, None, None]

        e3 = e3 + t

        # 
        d3 = self.dec3(e3)
        d3 = torch.cat([d3, e2], dim=1)
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        out = self.dec1(d2)

        return out


# ============================================================================
# 
# ============================================================================

def create_diffusion_model(cfg):
    """
    ôɢģ
    """
    model_type = cfg['model'].get('diffusion_model_type', 'simplified')

    if model_type == 'full':
        model = LatentDiffusionModel(
            vae_config=cfg['model']['vae'],
            unet_config=cfg['model']['unet'],
            diffusion_config=cfg['model']['diffusion']
        )
    else:
        model = SimplifiedDiffusionModel(
            image_size=cfg['dataset']['image_size'][0],
            in_channels=3,
            out_channels=4,
            base_channels=cfg['model'].get('diffusion_base_channels', 64),
            latent_dim=cfg['model'].get('latent_dim', 4),
            num_timesteps=cfg['model'].get('num_timesteps', 100)
        )

    return model


class DiffusionLoss(nn.Module):
    """
    Stage 2ʧ
    MSEʧԤ
    """

    def __init__(self):
        super().__init__()

    def forward(self, noise_pred, noise):
        return F.mse_loss(noise_pred, noise)