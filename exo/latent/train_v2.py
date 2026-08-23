"""
Improved Training Script for Latent Syn2Seq.

Changes:
  1. 500 epochs instead of 100
  2. Cosine learning rate scheduler
  3. Larger interpolator (hidden_dim = 128)
  4. Pixel-level reconstruction loss (decode back to RGB)
  5. Gradient clipping
  6. AMP gradient scaler
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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

from configs.default_config import Config
from data.dataset import create_dataloaders
from models.vae import FrameVAE
from models.dfot import LatentDiffusionForcingTransformer, GaussianDiffusion
from models.interpolator import LatentVideoInterpolatorV2
from training.trainer_v2 import LatentSyn2SeqTrainerV2


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train Syn2Seq in VAE latent space (Improved)")
    parser.add_argument("--no_temporal", action="store_true", help="Disable temporal loss")
    parser.add_argument("--output_dir", type=str, default="./outputs_latent_v2", help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="Data loader workers")
    parser.add_argument("--pixel_loss_weight", type=float, default=0.5, help="Weight for pixel-level recon loss")
    args = parser.parse_args()

    config = Config()
    set_seed(config.seed)

    config.output_dir = args.output_dir
    config.training.num_epochs = args.num_epochs
    config.training.lr = args.lr
    config.training.batch_size = args.batch_size

    print("=" * 70)
    print("Training Syn2Seq - Latent-space variant (Improved v2)")
    print("=" * 70)
    print(f"Epochs: {args.num_epochs}, LR: {args.lr}, Batch size: {args.batch_size}")
    print(f"Pixel loss weight: {args.pixel_loss_weight}")
    print(f"Temporal loss: {'ON' if not args.no_temporal else 'OFF'}")

    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    print("\nCreating frozen VAE...")
    vae = FrameVAE(
        pretrained_model_name_or_path=config.vae.pretrained_model_name_or_path,
        cache_dir=config.vae.cache_dir,
        scaling_factor=config.vae.scaling_factor,
        freeze=True,
    )
    config.model.in_channels = vae.latent_channels

    print("\nCreating latent DFoT...")
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

    print("\nCreating latent VideoInterpolator...")
    interpolator = LatentVideoInterpolatorV2(
        latent_channels=config.model.in_channels,
        num_interp_frames=config.data.num_interp_frames,
        hidden_dim=128,
        num_res_blocks=5,
    )

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) \
        + sum(p.numel() for p in interpolator.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {n_trainable / 1e6:.2f} M")
    print(f"  - DFoT: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    print(f"  - Interpolator: {sum(p.numel() for p in interpolator.parameters()) / 1e6:.2f} M")
    print(f"Frozen VAE: {sum(p.numel() for p in vae.parameters()) / 1e6:.2f} M")

    print("\nCreating trainer...")
    trainer = LatentSyn2SeqTrainerV2(
        model=model,
        diffusion=diffusion,
        interpolator=interpolator,
        vae=vae,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        use_temporal_consistency=not args.no_temporal,
        pixel_loss_weight=args.pixel_loss_weight,
    )

    print("\nStarting training...")
    print("=" * 70)
    trainer.train()


if __name__ == "__main__":
    main()
