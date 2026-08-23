import torch
import random
import numpy as np
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.default_config import Config
from data.dataset import create_dataloaders
from models.dfot import DiffusionForcingTransformer, GaussianDiffusion
from models.interpolator import VideoInterpolator
from training.trainer import Syn2SeqTrainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='Train Syn2Seq - Ours Method')
    parser.add_argument('--baseline', action='store_true', help='Use baseline method (no perceptual loss, no temporal consistency)')
    parser.add_argument('--no_perceptual', action='store_true', help='Disable perceptual loss')
    parser.add_argument('--no_temporal', action='store_true', help='Disable temporal consistency loss')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory name (e.g., outputs_baseline, outputs_ours)')
    args = parser.parse_args()
    
    config = Config()
    set_seed(config.seed)
    
    if args.output_dir:
        config.output_dir = args.output_dir
    elif args.baseline:
        config.output_dir = "outputs_baseline"
    else:
        config.output_dir = "outputs_ours"
    
    print("="*60)
    print("Training Syn2Seq - 'Ours' Method")
    print("="*60)
    
    if args.baseline:
        use_perceptual = False
        use_temporal = False
        print("Mode: Baseline (Simple MSE loss only)")
    else:
        use_perceptual = not args.no_perceptual
        use_temporal = not args.no_temporal
        print("Mode: Ours (Full features)")
        print(f"  - Perceptual Loss: {'? ON' if use_perceptual else '? OFF'}")
        print(f"  - Temporal Consistency: {'? ON' if use_temporal else '? OFF'}")
    
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    print("Creating models...")
    model = DiffusionForcingTransformer(
        hidden_size=config.model.hidden_size,
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers,
        mlp_ratio=config.model.mlp_ratio,
        dropout=config.model.dropout,
        num_timesteps=config.model.num_timesteps,
    )
    
    diffusion = GaussianDiffusion(
        num_timesteps=config.model.num_timesteps,
        beta_schedule=config.model.beta_schedule,
    )
    
    interpolator = VideoInterpolator(
        num_interp_frames=config.data.num_interp_frames,
        hidden_dim=64,
        num_res_blocks=3,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    print(f"Interpolator parameters: {sum(p.numel() for p in interpolator.parameters()) / 1e6:.2f} M")
    
    print("Creating trainer...")
    trainer = Syn2SeqTrainer(
        model=model,
        diffusion=diffusion,
        interpolator=interpolator,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        use_perceptual_loss=use_perceptual,
        use_temporal_consistency=use_temporal,
    )
    
    print("Starting training...")
    print("="*60)
    trainer.train()


if __name__ == "__main__":
    main()
