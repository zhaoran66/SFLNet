"""
Hand Pose Estimation Training Script - Latent Methods
=====================================================
Method 3: Baseline + Latent (SD-VAE)               -> GPU 4
Method 4: Baseline + Latent + Frequency Decomp     -> GPU 5
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.backends.cudnn.enabled = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/data/data5/zhaoran/paper_code/exo/Ours/new')

from configs.default_config import Config
from data.dataset import create_dataloaders
from models.method_c_vae import FrameVAE
from models.hand_pose_4methods import (
    HandPoseLatentOnly,
    HandPoseMethodC,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class HandPoseDatasetWrapper:
    def __init__(self, original_dataset):
        self.dataset = original_dataset
        self.samples = original_dataset.samples
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        exo_video = sample["exo_video"]
        num_frames = exo_video.shape[1]
        
        pose_file = self.samples[idx]["pose_file"]
        pose_data = np.load(pose_file)
        hand_pose = torch.from_numpy(pose_data["pose_m"][:, 0]).float()
        
        if hand_pose.shape[0] > num_frames:
            hand_pose = hand_pose[:num_frames]
        
        return {
            "exo_video": exo_video,
            "hand_pose": hand_pose,
        }


class HandPoseTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config,
        method_name: str,
        vae=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.method_name = method_name
        self.vae = vae

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.model.to(self.device)
        
        if self.vae is not None:
            self.vae.to(self.device)
            self.vae.eval()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        output_dir = os.path.join(config.output_dir, method_name)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
        self.output_dir = output_dir

        self.best_val_loss = float("inf")
        self.train_loss_history = []
        self.val_loss_history = []

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [{self.method_name}]")
        for batch in pbar:
            exo_video = batch["exo_video"].to(self.device, non_blocking=True)
            hand_pose_target = batch["hand_pose"].to(self.device, non_blocking=True)
            
            B, T, D = hand_pose_target.shape
            hand_pose_target = hand_pose_target.reshape(B * T, D)

            self.optimizer.zero_grad()

            with torch.no_grad():
                z = self.vae.encode(exo_video)
            
            pose_pred = self.model(z)
            
            loss = F.mse_loss(pose_pred, hand_pose_target)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        avg_loss = total_loss / num_batches
        return avg_loss

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> float:
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc=f"Validating"):
            exo_video = batch["exo_video"].to(self.device)
            hand_pose_target = batch["hand_pose"].to(self.device)
            
            B, T, D = hand_pose_target.shape
            hand_pose_target = hand_pose_target.reshape(B * T, D)

            z = self.vae.encode(exo_video)
            pose_pred = self.model(z)
            
            loss = F.mse_loss(pose_pred, hand_pose_target)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_loss_history': self.train_loss_history,
            'val_loss_history': self.val_loss_history,
        }
        
        if is_best:
            path = os.path.join(self.output_dir, "checkpoints", "best_model.pt")
        else:
            path = os.path.join(self.output_dir, "checkpoints", f"checkpoint_{epoch}.pt")
        
        torch.save(checkpoint, path)

    def train(self, num_epochs: int):
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Method: {self.method_name}")
        print(f"Output directory: {self.output_dir}")

        for epoch in range(num_epochs):
            train_loss = self._train_epoch(epoch)
            val_loss = self._val_epoch(epoch)
            
            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)
            
            self.scheduler.step(val_loss)

            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(epoch, is_best=True)
                print(f"  -> New best model saved! (val_loss: {self.best_val_loss:.6f})")

            if epoch % 10 == 0:
                self._save_checkpoint(epoch, is_best=False)

        print(f"\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        
        return self.best_val_loss


def main():
    parser = argparse.ArgumentParser(description='Train Hand Pose Estimation - Latent Methods')
    parser.add_argument('--method', type=str, required=True,
                       choices=['latent_only', 'latent_freq'],
                       help='Method to train')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='./outputs_hand_pose',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--vae_path', type=str, 
                       default='./outputs/vae/checkpoints/best_model.pt',
                       help='Path to pretrained VAE')
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    config = Config()
    config.training.num_epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.lr = args.lr
    config.output_dir = args.output_dir
    
    print("="*70)
    print("Hand Pose Estimation Training - Latent Methods")
    print("="*70)
    print(f"Method: {args.method}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Output dir: {args.output_dir}")
    print("="*70)
    
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)
    
    train_dataset = HandPoseDatasetWrapper(train_loader.dataset)
    val_dataset = HandPoseDatasetWrapper(val_loader.dataset)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    print("\nLoading VAE...")
    vae = FrameVAE(in_channels=3, latent_dim=4)
    if os.path.exists(args.vae_path):
        checkpoint = torch.load(args.vae_path, map_location='cpu')
        vae.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded VAE from {args.vae_path}")
    else:
        print(f"Warning: VAE checkpoint not found at {args.vae_path}")
    
    print("\nCreating model...")
    
    if args.method == 'latent_only':
        model = HandPoseLatentOnly(
            in_channels=4,
            num_frames=config.data.num_frames,
            hidden_dim=128,
            num_pose_params=51,
        )
    elif args.method == 'latent_freq':
        model = HandPoseMethodC(
            latent_dim=4,
            num_frames=config.data.num_frames,
            hidden_dim=128,
            num_pose_params=51,
            use_freq_decomp=True,
        )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    trainer = HandPoseTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        method_name=args.method,
        vae=vae,
    )
    
    best_loss = trainer.train(num_epochs=args.epochs)
    
    print(f"\n{'='*70}")
    print("Training Summary")
    print(f"{'='*70}")
    print(f"Method: {args.method}")
    print(f"Best validation loss: {best_loss:.6f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
