import torch
import random
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.default_config import Config
from data.dataset import create_dataloaders
from models.fasr_model import FASRModel, GaussianDiffusion
from training.fasr_trainer import FASRTrainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    config = Config()
    set_seed(config.seed)
    
    print("=" * 50)
    print("FASR: Frequency-Adaptive Structural Routing")
    print("From Synchrony to Sequence: Exo-to-Ego Generation via Interpolation")
    print("=" * 50)
    
    print("\nModel Configuration:")
    print(f"  - Use Spectral Routing: {config.model.use_spectral_routing}")
    print(f"  - Use Pose Guidance: {config.model.use_pose_guidance}")
    print(f"  - Hidden Size: {config.model.hidden_size}")
    print(f"  - Lambda FG: {config.model.lambda_struct_fg}")
    print(f"  - Lambda BG: {config.model.lambda_struct_bg}")
    print(f"  - Lambda Temp: {config.model.lambda_route_temp}")
    print(f"  - Lambda ID: {config.model.lambda_id}")
    print(f"  - Lambda Align: {config.model.lambda_align}")
    print(f"  - Use DCT: {config.model.use_dct}")
    
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    print("\nCreating model...")
    model = FASRModel(config)
    
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
