"""
Common training utilities for all 4 experiments
"""
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

sys.path.insert(0, '/data/data5/zhaoran/paper_code/exo/Ours/new')

from configs.default_config import Config
from data.dataset import create_dataloaders


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
    def __init__(self, model, train_loader, val_loader, config, method_name):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.method_name = method_name

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.lr,
            weight_decay=1e-5,
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        output_dir = os.path.join(config.output_dir, method_name)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
        self.output_dir = output_dir

        self.best_val_loss = float("inf")

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
            pose_pred = self.model(exo_video)
            
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

            pose_pred = self.model(exo_video)
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
        }
        
        if is_best:
            path = os.path.join(self.output_dir, "checkpoints", "best_model.pt")
        else:
            path = os.path.join(self.output_dir, "checkpoints", f"checkpoint_{epoch}.pt")
        
        torch.save(checkpoint, path)

    def train(self, num_epochs: int):
        print(f"\nStarting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            train_loss = self._train_epoch(epoch)
            val_loss = self._val_epoch(epoch)
            
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


def get_dataloaders(config):
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
    
    return train_loader, val_loader
