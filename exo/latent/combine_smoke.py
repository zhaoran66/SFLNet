"""Smoke test for combine pipeline: shape verification."""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
from configs.combine_config import Config
from models.vae import FrameVAE
from models.combine_dino import (
    DINOv2FeatureExtractor,
    SemanticFrequencyDecomposition,
    BackgroundLatentProjector,
)
from training.combine_trainer import (
    CombinedDiffusionTransformer,
    GaussianDiffusion,
)


def main():
    cfg = Config()
    device = "cpu"

    B = 1
    T = cfg.data.num_frames
    H, W = cfg.data.frame_size

    print("=" * 60)
    print(f"Input shape: ({B}, 3, {T}, {H}, {W})")
    print("=" * 60)

    video = torch.randn(B, 3, T, H, W)

    print("\n1. FrameVAE encode...")
    vae = FrameVAE(pretrained_model_name_or_path="/data/data5/zhaoran/paper_code/exo/latent/vae_cache/sd-vae-ft-mse")
    content_z = vae.encode(video)
    print(f"   Content latent: {tuple(content_z.shape)}")
    assert content_z.shape == (
        B, cfg.vae.latent_channels, T, H // cfg.vae.downsample_factor, W // cfg.vae.downsample_factor
    )

    print("\n2. DINO extract...")
    dino = DINOv2FeatureExtractor(output_size=cfg.dino.output_size)
    dino_feat = dino(video)
    print(f"   DINO feat dim: {dino_feat.shape[1]}, expected: {cfg.dino.feat_dim}")
    print(f"   DINO feat: {tuple(dino_feat.shape)}")
    assert dino_feat.shape[1] == cfg.dino.feat_dim

    print("\n3. Frequency Decomp...")
    freq_decomp = SemanticFrequencyDecomposition(
        feat_dim=cfg.dino.feat_dim,
        fft_size=cfg.dino.fft_size,
        low_freq_ratio=cfg.dino.low_freq_ratio,
    )
    bg_feat, fg_feat = freq_decomp(dino_feat)
    print(f"   Low-freq (background): {tuple(bg_feat.shape)}")
    print(f"   High-freq (foreground): {tuple(fg_feat.shape)}")
    assert bg_feat.shape == dino_feat.shape == fg_feat.shape

    print("\n4. Background Project to latent...")
    bg_proj = BackgroundLatentProjector(in_dim=cfg.dino.feat_dim, bg_latent_dim=cfg.dino.bg_latent_dim)
    bg_z = bg_proj(bg_feat)
    print(f"   Background latent: {tuple(bg_z.shape)}")
    assert bg_z.shape == (
        B, cfg.dino.bg_latent_dim, T, H // cfg.vae.downsample_factor, W // cfg.vae.downsample_factor
    )
    assert bg_z.shape == content_z.shape, "Background latent must match content latent spatial size!"

    print("\n5. Concat content + background...")
    combined_z = torch.cat([content_z, bg_z], dim=1)
    total_ch = cfg.vae.latent_channels + cfg.dino.bg_latent_dim
    print(f"   Combined latent: {tuple(combined_z)} (should have {total_ch} channels)")
    assert combined_z.shape[1] == total_ch

    print("\n6. Combined Diffusion Transformer...")
    model = CombinedDiffusionTransformer(
        in_channels=total_ch,
        hidden_size=cfg.model.hidden_size,
        num_heads=cfg.model.num_heads,
        num_layers=2,
        mlp_ratio=cfg.model.mlp_ratio,
        dropout=cfg.model.dropout,
        patch_size=cfg.model.patch_size,
    )
    diffusion = GaussianDiffusion(num_timesteps=cfg.model.num_timesteps)

    t = torch.randint(0, cfg.model.num_timesteps, (B,))
    pose_cond = torch.randn(B, 2, 4, 4)
    noise = torch.randn_like(combined_z)
    z_t = diffusion.q_sample(combined_z, t, noise)
    pred = model(z_t, t, pose_cond)
    print(f"   DFoT input: {tuple(z_t.shape)}, output: {tuple(pred.shape)}")
    assert pred.shape == z_t.shape

    print("\n" + "=" * 60)
    print(" ALL SHAPE CHECKS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
