"""Verify standalone SD VAE loads the official sd-vae-ft-mse weights."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
from models.sd_vae import AutoencoderKL

VAE_DIR = "/data/data5/zhaoran/paper_code/exo/latent/vae_cache/sd-vae-ft-mse"


def main():
    print("Loading...")
    vae = AutoencoderKL.from_local(VAE_DIR).eval()
    print(f"latent_channels = {vae.config.latent_channels}")
    print(f"block_out_channels = {vae.config.block_out_channels}")
    print(f"scaling_factor = {vae.config.scaling_factor}")

    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        post = vae.encode(x)
        z = post.sample() * vae.config.scaling_factor
        x_rec = vae.decode(z / vae.config.scaling_factor)

    print(f"input  : {tuple(x.shape)}")
    print(f"latent : {tuple(z.shape)}")
    print(f"recon  : {tuple(x_rec.shape)}")
    print(f"latent stats: mean={z.mean().item():.3f} std={z.std().item():.3f}")
    print(f"recon  stats: mean={x_rec.mean().item():.3f} std={x_rec.std().item():.3f}")
    print("OK")


if __name__ == "__main__":
    main()
