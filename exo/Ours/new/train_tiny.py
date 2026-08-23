import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

import torch
import random
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config_tiny import Config
from data.dataset import create_dataloaders
from models.fasr_model_tiny import FASRTinyModel, GaussianDiffusion
from training.fasr_trainer import FASRTrainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    torch.cuda.empty_cache()


def main():
    config = Config()
    set_seed(config.seed)
    
    print("=" * 60)
    print("FASR-Tiny: Frequency-Adaptive Structural Routing (Memory Optimized)")
    print("=" * 60)
    
    print("\nModel Configuration:")
    print(f"  Hidden Size: {config.model.hidden_size}")
    print(f"  Num Layers: {config.model.num_layers}")
    print(f"  Num Heads: {config.model.num_heads}")
    print(f"  Use DCT: {config.model.use_dct}")
    
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    print("\nCreating model...")
    model = FASRTinyModel(config)
    
    diffusion = GaussianDiffusion(
        num_timesteps=config.model.num_timesteps,
        beta_schedule=config.model.beta_schedule,
    )
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params / 1e6:.2f} M")
    
    print("\nCreating trainer...")
    trainer = FASRTrainer(
        model=model,
        diffusion=diffusion,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )
    
    print("\nStarting training...")
    trainer.train()


if __name__ == "__main__":
    main()
