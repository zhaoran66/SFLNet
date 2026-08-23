"""
Hand Pose Estimation Training Script (21 joints, 51 DOF)
With GPU selection support
"""
import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config_tiny import Config
from data.dataset import create_dataloaders
from models.hand_pose_estimator import (
    HandPoseBaseline,
    HandPoseLatentOnly,
    HandPoseOurs,
    HandPoseMethodC,
)
from models.method_c_vae import FrameVAE


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class HandPoseDatasetWrapper:
    """
    Wrapper to load hand pose (MANO 51 DOF) as target
    """
    def __init__(self, original_dataset):
        self.dataset = original_dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        exo_video = sample["exo_video"]
        ego_video = sample["ego_video"]
        exo_pose = sample["exo_pose"]
        
        pose_file = self.dataset.samples[idx]["pose_file"]
        import numpy as np
        pose_data = np.load(pose_file)
        hand_pose = torch.from_numpy(pose_data["pose_m"][:8, 0]).float()
        
        return {
            "exo_video": exo_video,
            "ego_video": ego_video,
            "exo_pose": exo_pose,
            "hand_pose": hand_pose,
        }


class HandPoseTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config,
        vae=None,
        feature_extractor=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.vae = vae
        self.feature_extractor = feature_extractor

        self.device = torch.device(config.training.device)
        self.model.to(self.device)
        
        if self.vae is not None:
            self.vae.to(self.device)
            self.vae.eval()
        
        if self.feature_extractor is not None:
            self.feature_extractor.to(self.device)
            self.feature_extractor.eval()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        self.scaler = GradScaler(enabled=config.training.use_amp)
        self.global_step = 0
        self.start_epoch = 0

        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, "checkpoints"), exist_ok=True)

        self.best_val_loss = float("inf")

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            exo_video = batch["exo_video"].to(self.device, non_blocking=True)
            hand_pose_target = batch["hand_pose"].to(self.device, non_blocking=True)
            
            B, T, D = hand_pose_target.shape
            hand_pose_target = hand_pose_target.view(B * T, D)

            self.optimizer.zero_grad()

            with autocast(enabled=self.config.training.use_amp):
                if self.vae is not None:
                    with torch.no_grad():
                        z = self.vae.encode(exo_video)
                    pose_pred = self.model(z)
                elif self.feature_extractor is not None:
                    with torch.no_grad():
                        feat = self.feature_extractor(exo_video)
                    pose_pred = self.model(feat)
                else:
                    pose_pred = self.model(exo_video)
                
                loss = F.mse_loss(pose_pred, hand_pose_target)

            self.scaler.scale(loss).backward()
            
            if self.config.training.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.grad_clip
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

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
            hand_pose_target = hand_pose_target.view(B * T, D)

            with autocast(enabled=self.config.training.use_amp):
                if self.vae is not None:
                    z = self.vae.encode(exo_video)
                    pose_pred = self.model(z)
                elif self.feature_extractor is not None:
                    feat = self.feature_extractor(exo_video)
                    pose_pred = self.model(feat)
                else:
                    pose_pred = self.model(exo_video)
                
                loss = F.mse_loss(pose_pred, hand_pose_target)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def train(self):
        for epoch in range(self.start_epoch, self.config.training.num_epochs):
            print(f"\nEpoch {epoch}/{self.config.training.num_epochs}")
            print("-" * 60)

            train_loss = self._train_epoch(epoch)
            val_loss = self._val_epoch(epoch)
            
            self.scheduler.step(val_loss)

            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                print(f"New best val loss: {self.best_val_loss:.6f}")
                self._save_checkpoint(epoch, "best")

            if epoch % 10 == 0:
                self._save_checkpoint(epoch, f"epoch_{epoch}")

        self._save_checkpoint(self.config.training.num_epochs - 1, "last")

    def _save_checkpoint(self, epoch: int, name: str):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        torch.save(
            checkpoint,
            os.path.join(self.config.output_dir, "checkpoints", f"{name}.pt"),
        )


def main():
    parser = argparse.ArgumentParser(description="Train Hand Pose Estimation (21 joints)")
    parser.add_argument("--method", type=str, default="baseline",
                        choices=["baseline", "latent_only", "ours", "method_c"],
                        help="Method to train")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index to use (default: 0)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

    config = Config()
    set_seed(config.seed)

    if args.output_dir:
        config.output_dir = args.output_dir
    else:
        config.output_dir = f"./outputs_hand_pose_{args.method}_gpu{args.gpu}"

    print("=" * 60)
    print(f"Hand Pose Estimation Training: {args.method.upper()}")
    print(f"Using GPU: {args.gpu}")
    print("=" * 60)
    print(f"Output dir: {config.output_dir}")

    print("Creating dataloaders...")
    train_loader_orig, val_loader_orig = create_dataloaders(config)
    
    train_dataset = HandPoseDatasetWrapper(train_loader_orig.dataset)
    val_dataset = HandPoseDatasetWrapper(val_loader_orig.dataset)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    vae = None
    feature_extractor = None

    print(f"Creating model: {args.method}...")
    if args.method == "baseline":
        model = HandPoseBaseline(
            in_channels=3,
            num_frames=8,
            hidden_dim=256,
            num_pose_params=51,
        )
    elif args.method == "latent_only":
        vae = FrameVAE()
        model = HandPoseLatentOnly(
            in_channels=4,
            num_frames=8,
            hidden_dim=512,
            num_pose_params=51,
        )
    elif args.method == "ours":
        from models.dino_extractor import DINOv2FeatureExtractor
        feature_extractor = DINOv2FeatureExtractor()
        model = HandPoseOurs(
            feat_dim=384,
            num_frames=8,
            hidden_dim=512,
            num_pose_params=51,
            use_freq_decomp=True,
        )
    elif args.method == "method_c":
        vae = FrameVAE()
        model = HandPoseMethodC(
            latent_dim=4,
            num_frames=8,
            hidden_dim=512,
            num_pose_params=51,
            use_freq_decomp=True,
        )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    print("Creating trainer...")
    trainer = HandPoseTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        vae=vae,
        feature_extractor=feature_extractor,
    )

    print(f"Starting training for {config.training.num_epochs} epochs...")
    trainer.train()

    print("=" * 60)
    print("Training Complete!")
    print(f"Best val loss: {trainer.best_val_loss:.6f}")
    print(f"Checkpoints saved to: {config.output_dir}/checkpoints/")
    print("=" * 60)


if __name__ == "__main__":
    main()
