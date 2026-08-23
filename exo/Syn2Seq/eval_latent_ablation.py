"""
Ablation: Syn2Seq + DINOv2 Latent (WITHOUT Frequency Routing)
Use EXACT SAME metric pipeline as FASR
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torchvision.utils as vutils

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"


def compute_psnr(img1, img2, data_range=1.0):
    """EXACT SAME as FASR"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(data_range / np.sqrt(mse))


def compute_ssim(img1, img2, multichannel=True, data_range=1.0, channel_axis=-1):
    """EXACT SAME as FASR"""
    if multichannel:
        if channel_axis != -1:
            img1 = np.moveaxis(img1, channel_axis, -1)
            img2 = np.moveaxis(img2, channel_axis, -1)
    
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    H, W, C = img1.shape
    
    mu1 = np.mean(img1, axis=(0, 1))
    mu2 = np.mean(img2, axis=(0, 1))
    
    sigma1_sq = np.var(img1, axis=(0, 1))
    sigma2_sq = np.var(img2, axis=(0, 1))
    sigma12 = np.mean((img1 - mu1[None, None, :]) * (img2 - mu2[None, None, :]), axis=(0, 1))
    
    ssim_numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    ssim_denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim = np.mean(ssim_numerator / ssim_denominator)
    
    return ssim


def compute_temporal_consistency(video_frames):
    """EXACT SAME as FASR"""
    T = len(video_frames)
    diffs = []
    
    for i in range(T - 1):
        frame_diff = np.mean(np.abs(video_frames[i] - video_frames[i+1]))
        diffs.append(frame_diff)
    
    return np.mean(diffs)


def compute_gradient_magnitude(img):
    """EXACT SAME as FASR"""
    gx = img[:, :, 1:] - img[:, :, :-1]
    gy = img[:, 1:, :] - img[:, :-1, :]
    
    gx = np.pad(gx, ((0,0), (0,0), (0,1)), mode='edge')
    gy = np.pad(gy, ((0,0), (0,1), (0,0)), mode='edge')
    
    grad_mag = np.sqrt(gx**2 + gy**2)
    return np.mean(grad_mag)


class DINOv2FeatureExtractor(nn.Module):
    """Copy from FASR - same implementation"""
    def __init__(self, output_size=(32, 32)):
        super().__init__()
        self.output_size = output_size
        self._init_dinov2()
    
    def _init_dinov2(self):
        dinov2_path = "/data/data5/zhaoran/paper_code/exo/model/dinov2_small_14_layer/dinov2_small_final"
        print(f"[DINOv2] Loading from: {dinov2_path}")
        
        import torch.hub
        torch.hub.set_dir("/data/data5/zhaoran/paper_code/exo/model/torch_hub_cache")
        
        self.model = torch.hub.load(
            'facebookresearch/dinov2', 
            'dinov2_vits14',
            source='github',
            pretrained=False
        )
        
        state_dict = torch.load(os.path.join(dinov2_path, "pytorch_model.bin"), map_location="cpu")
        self.model.load_state_dict(state_dict)
        
        self.model.eval()
        self.model.requires_grad_(False)
        
        self.feat_dim = 384
        print(f"[DINOv2] Loaded successfully! Feature dim: {self.feat_dim}")
    
    @torch.no_grad()
    def forward(self, x):
        B, C, T, H, W = x.shape
        x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x_resized = nn.functional.interpolate(x_reshaped, size=(224, 224), mode='bilinear', align_corners=False)
        
        feats = self.model(x_resized)
        feats = feats.reshape(B, T, self.feat_dim)[:, :, :, None, None]
        feats = feats.permute(0, 2, 1, 3, 4)
        feats = feats.expand(-1, -1, -1, self.output_size[0], self.output_size[1])
        
        return feats


class LatentSyn2Seq(nn.Module):
    """
    Syn2Seq + DINOv2 Latent (NO Frequency Routing)
    This is the ablation baseline to compare against FASR
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
            nn.Linear(42, hidden_size // 2),
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
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feat_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )
    
    def forward(self, x, t, pose_cond):
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


class GaussianDiffusion:
    def __init__(self, num_timesteps=1000, beta_schedule="linear"):
        self.num_timesteps = num_timesteps
        
        if beta_schedule == "linear":
            self.beta = torch.linspace(0.0001, 0.02, num_timesteps)
        
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)
        self.sqrt_recip_alpha = torch.sqrt(1.0 / self.alpha)
        self.posterior_variance = self.beta * (1.0 - torch.roll(self.alpha_bar, 1)) / (1.0 - self.alpha_bar)
        self.posterior_variance[0] = self.beta[0]
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = self.beta * torch.sqrt(torch.roll(self.alpha_bar, 1)) / (1.0 - self.alpha_bar)
        self.posterior_mean_coef2 = (1.0 - torch.roll(self.alpha_bar, 1)) * torch.sqrt(self.alpha) / (1.0 - self.alpha_bar)
        self.posterior_mean_coef1[0] = 0.0
    
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1, 1)
        
        return sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise
    
    def p_sample(self, model_output, x, t):
        B = x.shape[0]
        
        sqrt_recip_alpha_t = self.sqrt_recip_alpha[t].view(-1, 1, 1, 1, 1)
        beta_t = self.beta[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1, 1)
        
        pred_mean = sqrt_recip_alpha_t * (x - beta_t / sqrt_one_minus_alpha_bar_t * model_output)
        
        if t[0] > 0:
            noise = torch.randn_like(x)
            posterior_log_variance_t = self.posterior_log_variance_clipped[t].view(-1, 1, 1, 1, 1)
            return pred_mean + torch.exp(0.5 * posterior_log_variance_t) * noise
        else:
            return pred_mean


def main():
    import math
    from data.dataset import create_dataloaders
    from configs.default_config import Config
    
    print("=" * 70)
    print("SYN2SEQ + LATENT ABLATION (NO FREQ ROUTING)")
    print("=" * 70)
    print("EXACT SAME metric pipeline as FASR for fair comparison!")
    print("=" * 70)
    
    config = Config()
    
    checkpoint_path = "/data/data5/zhaoran/paper_code/exo/Ours/new/outputs_fasr_tiny_stable/checkpoints/checkpoint_20.pt"
    print(f"\nLoading: {checkpoint_path}")
    print("(this is FASR checkpoint - we'll just test the metric pipeline)")
    print("=" * 70)
    print("Note: For proper ablation, train a version WITHOUT the routing layers!")
    print("=" * 70)


if __name__ == "__main__":
    main()
