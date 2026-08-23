"""
Smoke test: builds the latent pipeline with random tensors of the right
shape and runs one forward pass through interpolator + diffusion model
in latent space (without touching the VAE or the real dataset).

Run:
    python smoke_test.py
"""

import os
import sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from configs.default_config import Config
from models.dfot import LatentDiffusionForcingTransformer, GaussianDiffusion
from models.interpolator import LatentVideoInterpolator


def main():
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    B = 1
    T = cfg.data.num_frames
    H, W = cfg.data.frame_size
    Cz = cfg.vae.latent_channels
    Hz, Wz = H // cfg.vae.downsample_factor, W // cfg.vae.downsample_factor

    print(f"Pixel video shape: ({B}, 3, {T}, {H}, {W})")
    print(f"Latent video shape: ({B}, {Cz}, {T}, {Hz}, {Wz})")

    exo_z = torch.randn(B, Cz, T, Hz, Wz, device=device)
    ego_z = torch.randn(B, Cz, T, Hz, Wz, device=device)

    interp = LatentVideoInterpolator(
        latent_channels=Cz,
        num_interp_frames=cfg.data.num_interp_frames,
        hidden_dim=64,
        num_res_blocks=2,
    ).to(device)

    full_z, interp_z = interp(exo_z, ego_z)
    print(f"Interpolator -> full: {tuple(full_z.shape)}, interp: {tuple(interp_z.shape)}")

    model = LatentDiffusionForcingTransformer(
        in_channels=Cz,
        hidden_size=cfg.model.hidden_size,
        num_heads=cfg.model.num_heads,
        num_layers=2,
        mlp_ratio=cfg.model.mlp_ratio,
        dropout=cfg.model.dropout,
        num_timesteps=cfg.model.num_timesteps,
        patch_size=cfg.model.patch_size,
    ).to(device)

    diff = GaussianDiffusion(num_timesteps=cfg.model.num_timesteps).to(device)

    train_seq_z = torch.cat([exo_z[:, :, -4:], interp_z], dim=2)
    t = torch.randint(0, cfg.model.num_timesteps, (B,), device=device)
    noise = torch.randn_like(train_seq_z)
    z_t = diff.q_sample(train_seq_z, t, noise)
    pose_cond = torch.randn(B, 2, 4, 4, device=device)

    pred = model(z_t, t, pose_cond)
    print(f"DFoT -> pred: {tuple(pred.shape)} (should match {tuple(z_t.shape)})")

    assert pred.shape == z_t.shape, "Latent DFoT output shape mismatch"
    print("OK: latent pipeline forward pass succeeded.")


if __name__ == "__main__":
    main()
