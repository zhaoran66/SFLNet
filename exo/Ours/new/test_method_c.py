"""
Quick test for Method C components
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
import sys
sys.path.insert(0, '.')

print("=" * 60)
print("Testing Method C Components")
print("=" * 60)

print("\n[1] Testing FrameVAE...")
from models.method_c_vae import FrameVAE

vae = FrameVAE(
    pretrained_model_name_or_path="/data/data5/zhaoran/paper_code/exo/latent/vae_cache/sd-vae-ft-mse",
    freeze=True,
)
print(f"  VAE created successfully")
print(f"  VAE latent channels: {vae.latent_channels}")

device = torch.device("cuda")
vae.to(device)

# Test encode
x = torch.randn(1, 3, 8, 128, 128, device=device)
with torch.no_grad():
    z = vae.encode(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Latent shape: {z.shape}")
    x_recon = vae.decode(z)
    print(f"  Reconstructed shape: {x_recon.shape}")
    print(f"  ? VAE works!")

print("\n[2] Testing SoftSpectralDecomposition...")
from models.freq_routing import SoftSpectralDecomposition

spec_decomp = SoftSpectralDecomposition(
    freq_size=(8, 8),
    feat_dim=4,
    sigma=0.5,
    use_dct=True,
)
spec_decomp.to(device)

with torch.no_grad():
    z_low, z_high = spec_decomp(z)
    print(f"  Low freq shape: {z_low.shape}")
    print(f"  High freq shape: {z_high.shape}")
    print(f"  ? Spectral decomposition works!")

print("\n[3] Testing LatentDiffusionTransformerWithBG...")
from models.method_c_model import LatentDiffusionTransformerWithBG

model = LatentDiffusionTransformerWithBG(
    in_channels=4,
    hidden_size=256,
    num_heads=4,
    num_layers=6,
    use_freq_decomp=True,
    use_checkpoint=False,
)
model.to(device)

t = torch.randint(0, 1000, (1,), device=device)
pose_cond = torch.randn(1, 2, 4, 4, device=device)

with torch.no_grad():
    out = model(z, t, pose_cond)
    print(f"  Input latent shape: {z.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  ? Diffusion model works!")

print("\n[4] Testing compute_losses...")

class SimpleDiffusion:
    def __init__(self):
        self.num_timesteps = 1000
    def q_sample(self, x, t, noise):
        return x * 0.5 + noise * 0.5

diffusion = SimpleDiffusion()

exo_video = torch.randn(1, 3, 8, 128, 128, device=device)
ego_video = torch.randn(1, 3, 8, 128, 128, device=device)
exo_pose = torch.randn(1, 8, 4, 4, device=device)
ego_pose = torch.randn(1, 8, 4, 4, device=device)

with torch.no_grad():
    losses = model.compute_losses(exo_video, ego_video, exo_pose, ego_pose, diffusion, vae)
    print(f"  Total loss: {losses['total_loss'].item():.4f}")
    print(f"  Diffusion loss: {losses['diffusion_loss'].item():.4f}")
    print(f"  FG loss: {losses['fg_loss'].item():.4f}")
    print(f"  BG loss: {losses['bg_loss'].item():.4f}")
    print(f"  ? Loss computation works!")

print("\n" + "=" * 60)
print("ALL TESTS PASSED! Method C is ready to train!")
print("=" * 60)
print(f"\nTotal model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
