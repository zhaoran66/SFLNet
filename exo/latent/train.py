"""
Training entry point for the latent-space variant of Syn2Seq.

This script reuses the DexYCB dataset implementation from the original
Syn2Seq repo (without modifying it) by inserting that path into sys.path.
Everything diffusion / interpolation related is performed in the SD-VAE
latent space, so the heavy 3D Transformer never sees full-resolution RGB
tensors.
"""

import os
import sys
import argparse
import random
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SYN2SEQ_DIR = os.path.abspath(os.path.join(HERE, "..", "Syn2Seq"))
sys.path.insert(0, HERE)
sys.path.insert(0, SYN2SEQ_DIR)

from configs.default_config import Config  # noqa: E402

from data.dataset import create_dataloaders  # noqa: E402  (from Syn2Seq)

from models.vae import FrameVAE  # noqa: E402
from models.dfot import LatentDiffusionForcingTransformer, GaussianDiffusion  # noqa: E402
from models.interpolator import LatentVideoInterpolator  # noqa: E402
from training.trainer import LatentSyn2SeqTrainer  # noqa: E402


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train Syn2Seq in VAE latent space")
    parser.add_argument("--no_temporal", action="store_true",
                        help="Disable temporal consistency loss on latent interpolation")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--vae_model", type=str, default=None,
                        help="Override VAE model id (e.g. madebyollin/taesd)")
    args = parser.parse_args()

    config = Config()
    set_seed(config.seed)

    if args.output_dir:
        config.output_dir = args.output_dir
    if args.vae_model:
        config.vae.pretrained_model_name_or_path = args.vae_model

    print("=" * 60)
    print("Training Syn2Seq - Latent-space variant")
    print("=" * 60)
    print(f"VAE: {config.vae.pretrained_model_name_or_path}")
    print(f"Latent channels: {config.vae.latent_channels}, downsample x{config.vae.downsample_factor}")
    print(f"Output dir: {config.output_dir}")
    print(f"Temporal consistency: {'OFF' if args.no_temporal else 'ON'}")

    print("Creating dataloaders (reusing Syn2Seq dataset)...")
    train_loader, val_loader = create_dataloaders(config)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    print("Creating frozen VAE...")
    vae = FrameVAE(
        pretrained_model_name_or_path=config.vae.pretrained_model_name_or_path,
        cache_dir=config.vae.cache_dir,
        scaling_factor=config.vae.scaling_factor,
        encode_chunk_size=config.vae.encode_chunk_size,
        freeze=config.vae.freeze,
    )
    config.model.in_channels = vae.latent_channels

    print("Creating latent DFoT...")
    model = LatentDiffusionForcingTransformer(
        in_channels=config.model.in_channels,
        hidden_size=config.model.hidden_size,
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers,
        mlp_ratio=config.model.mlp_ratio,
        dropout=config.model.dropout,
        num_timesteps=config.model.num_timesteps,
        patch_size=config.model.patch_size,
    )

    diffusion = GaussianDiffusion(
        num_timesteps=config.model.num_timesteps,
        beta_schedule=config.model.beta_schedule,
    )

    print("Creating latent VideoInterpolator...")
    interpolator = LatentVideoInterpolator(
        latent_channels=config.model.in_channels,
        num_interp_frames=config.data.num_interp_frames,
        hidden_dim=64,
        num_res_blocks=3,
    )

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) \
        + sum(p.numel() for p in interpolator.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_trainable / 1e6:.2f} M")
    print(f"Frozen VAE parameters: {sum(p.numel() for p in vae.parameters()) / 1e6:.2f} M")

    print("Creating trainer...")
    trainer = LatentSyn2SeqTrainer(
        model=model,
        diffusion=diffusion,
        interpolator=interpolator,
        vae=vae,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        use_temporal_consistency=not args.no_temporal,
    )

    print("Starting training...")
    print("=" * 60)
    trainer.train()


if __name__ == "__main__":
    main()
