"""
Train entry for combine: VAE Content + DINO Low-Freq Background.
"""
import os
import sys
import gc
import argparse
import random
import numpy as np

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,garbage_collection_threshold:0.8"

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SYN2SEQ_DIR = os.path.abspath(os.path.join(HERE, "..", "Syn2Seq"))
sys.path.insert(0, HERE)
sys.path.insert(0, SYN2SEQ_DIR)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from configs.combine_config import Config

from data.dataset import create_dataloaders

from models.vae import FrameVAE
from models.combine_dino import (
    DINOv2FeatureExtractor,
    SemanticFrequencyDecomposition,
    BackgroundLatentProjector,
)
from training.combine_trainer import (
    CombinedDiffusionTransformer,
    GaussianDiffusion,
    CombineTrainer,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train Combine: VAE Content + DINO Background")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    config = Config()
    set_seed(config.seed)

    if args.output_dir:
        config.output_dir = args.output_dir

    print("=" * 60)
    print("Training Combine: SD-VAE Content + DINO Low-Freq Background")
    print("=" * 60)
    print(f"Content latent dim: {config.vae.latent_channels}")
    print(f"Background latent dim: {config.dino.bg_latent_dim}")
    print(f"Total input dim to DFoT: {config.model.vae_latent_channels + config.model.bg_latent_channels}")
    print(f"Output dir: {config.output_dir}")

    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    print("\nCreating VAE (frozen)...")
    vae = FrameVAE(
        pretrained_model_name_or_path=config.vae.pretrained_model_name_or_path,
        cache_dir=config.vae.cache_dir,
        scaling_factor=config.vae.scaling_factor,
        freeze=True,
    )

    print("\n=== Creating models with low-memory mode ===")
    device = torch.device(config.training.device)

    # --- 1. 先加载 DINOv2 (最大的模型) ---
    print("\n1. Loading DINOv2 Feature Extractor...")
    dino = DINOv2FeatureExtractor(output_size=cfg.dino.output_size).to(device).half()
    dino.eval()
    for p in dino.parameters():
        p.requires_grad = False
    torch.cuda.empty_cache()
    gc.collect()
    print(f"   DINO loaded: {sum(p.numel() for p in dino.parameters())/1e6:.1f}M params")

    # --- 2. 加载 VAE ---
    print("\n2. Loading VAE (frozen)...")
    vae = FrameVAE(pretrained_model_name_or_path=config.vae.pretrained_model_name_or_path).to(device)
    vae.eval()
    torch.cuda.empty_cache()
    gc.collect()
    print(f"   VAE loaded: {sum(p.numel() for p in vae.parameters())/1e6:.1f}M params")

    # --- 3. 加载可训练模块 (freq_decomp + projecter + DFoT) ---
    print("\n3. Creating trainable modules...")
    freq_decomp = SemanticFrequencyDecomposition(
        feat_dim=cfg.dino.feat_dim,
        fft_size=cfg.dino.fft_size,
        low_freq_ratio=cfg.dino.low_freq_ratio,
    ).to(device)

    bg_projector = BackgroundLatentProjector(
        in_dim=cfg.dino.feat_dim,
        bg_latent_dim=cfg.dino.bg_latent_dim,
    ).to(device)

    total_in_channels = config.model.vae_latent_channels + config.model.bg_latent_channels
    diff_model = CombinedDiffusionTransformer(
        in_channels=total_in_channels,
        hidden_size=config.model.hidden_size,
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers,
        mlp_ratio=config.model.mlp_ratio,
        dropout=config.model.dropout,
        patch_size=config.model.patch_size,
    ).to(device)

    diffusion = GaussianDiffusion(num_timesteps=config.model.num_timesteps).to(device)

    n_trainable = (
        sum(p.numel() for p in diff_model.parameters() if p.requires_grad)
        + sum(p.numel() for p in bg_projector.parameters() if p.requires_grad)
        + sum(p.numel() for p in freq_decomp.parameters() if p.requires_grad)
    )
    print(f"   Trainable: {n_trainable / 1e6:.2f} M")

    torch.cuda.empty_cache()
    gc.collect()
    trainer = CombineTrainer(
        vae=vae,
        dino_extractor=dino,
        freq_decomp=freq_decomp,
        bg_projector=bg_projector,
        diff_model=diff_model,
        diffusion=diffusion,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )

    print("\nStarting training...")
    print("=" * 60)
    trainer.train()


if __name__ == "__main__":
    main()
