#!/usr/bin/env python
"""
Multi-Checkpoint Evaluation - Simple Version
"""
import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.default_config import Config
from data.dataset import create_dataloaders


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SimpleDatasetWrapper:
    def __init__(self, original_dataset):
        self.dataset = original_dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        exo_video = sample["exo_video"]
        num_frames = exo_video.shape[1]
        
        pose_file = self.dataset.samples[idx]["pose_file"]
        pose_data = np.load(pose_file)
        hand_pose = torch.from_numpy(pose_data["pose_m"][:, 0]).float()
        
        if hand_pose.shape[0] > num_frames:
            hand_pose = hand_pose[:num_frames]
        
        return {
            "exo_video": exo_video,
            "hand_pose": hand_pose,
        }


class PixelEncoder(torch.nn.Module):
    def __init__(self, in_channels=3, hidden_dim=128):
        super().__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels, hidden_dim // 2, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            torch.nn.ReLU(),
            torch.nn.Conv3d(hidden_dim // 2, hidden_dim, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv3d(hidden_dim, hidden_dim * 2, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv3d(hidden_dim * 2, hidden_dim * 2, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            torch.nn.ReLU(),
        )
    
    def forward(self, x):
        return self.conv_layers(x)


class RegressorHead(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_pose_params=51):
        super().__init__()
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_dim // 2, num_pose_params),
        )
    
    def forward(self, x):
        return self.regressor(x)


class HandPoseBaseline(torch.nn.Module):
    def __init__(self, in_channels=3, num_frames=8, hidden_dim=128, num_pose_params=51):
        super().__init__()
        self.encoder = PixelEncoder(in_channels=in_channels, hidden_dim=hidden_dim)
        
        with torch.no_grad():
            dummy = torch.randn(1, in_channels, num_frames, 128, 128)
            out = self.encoder(dummy)
            regressor_input_dim = out.shape[1] * out.shape[3] * out.shape[4]
        
        self.regressor = RegressorHead(regressor_input_dim, hidden_dim=512, num_pose_params=num_pose_params)
    
    def forward(self, x):
        B, C, T, H, W = x.shape
        feat = self.encoder(x)
        feat_flat = feat.permute(0, 2, 1, 3, 4).reshape(B * T, -1)
        pose = self.regressor(feat_flat)
        return pose


def compute_metrics(pred, target):
    mse = F.mse_loss(pred, target).item()
    mae = F.l1_loss(pred, target).item()
    rmse = np.sqrt(mse)
    
    error = torch.sqrt(torch.sum((pred - target) ** 2, dim=-1))
    mpjpe = torch.mean(error).item()
    
    thresholds = [1.0, 2.0, 5.0, 10.0]
    pck_values = {}
    for threshold in thresholds:
        correct = (error < threshold).float()
        pck = torch.mean(correct) * 100
        pck_values[f'pck@{threshold}'] = pck.item()
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'mpjpe': mpjpe,
        **pck_values
    }


@torch.no_grad()
def evaluate_single_checkpoint(model, data_loader, device):
    model.eval()
    
    all_preds = []
    all_targets = []
    
    for batch in tqdm(data_loader, desc="Evaluating", leave=False):
        exo_video = batch["exo_video"].to(device, non_blocking=True)
        hand_pose_target = batch["hand_pose"].to(device, non_blocking=True)
        
        B, T, D = hand_pose_target.shape
        hand_pose_target = hand_pose_target.reshape(B * T, D)
        
        pose_pred = model(exo_video)
        
        all_preds.append(pose_pred.cpu())
        all_targets.append(hand_pose_target.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    metrics = compute_metrics(all_preds, all_targets)
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Multi-Checkpoint Evaluation')
    parser.add_argument('--method', type=str, default='baseline')
    parser.add_argument('--output_dir', type=str, 
                       default='/data/data5/zhaoran/paper_code/exo/Ours/new/outputs_hand_pose')
    parser.add_argument('--num_ckpts', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    config = Config()
    config.training.batch_size = 4
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("=" * 80)
    print("Multi-Checkpoint Evaluation")
    print("=" * 80)
    
    print("\nCreating dataloaders...")
    _, val_loader = create_dataloaders(config)
    
    val_dataset = SimpleDatasetWrapper(val_loader.dataset)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    
    print(f"Validation samples: {len(val_dataset)}")
    
    checkpoint_dir = os.path.join(args.output_dir, args.method, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        return
    
    checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_")])
    
    if len(checkpoint_files) == 0:
        print("No checkpoints found!")
        return
    
    print(f"\nFound {len(checkpoint_files)} checkpoints")
    print(f"Evaluating last {min(args.num_ckpts, len(checkpoint_files))} checkpoints...")
    
    checkpoint_files = checkpoint_files[-args.num_ckpts:]
    
    all_metrics = []
    
    for ckpt_file in checkpoint_files:
        ckpt_path = os.path.join(checkpoint_dir, ckpt_file)
        
        print(f"\n{'=' * 80}")
        print(f"Evaluating: {ckpt_file}")
        print(f"{'=' * 80}")
        
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        model = HandPoseBaseline(
            in_channels=3,
            num_frames=8,
            hidden_dim=128,
            num_pose_params=51,
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        epoch = checkpoint.get('epoch', 'N/A')
        val_loss = checkpoint.get('best_val_loss', 'N/A')
        print(f"Epoch: {epoch}, Val Loss: {val_loss}")
        
        metrics = evaluate_single_checkpoint(model, val_loader, device)
        all_metrics.append(metrics)
        
        print(f"  MSE:      {metrics['mse']:.6f}")
        print(f"  MAE:      {metrics['mae']:.6f}")
        print(f"  MPJPE:    {metrics['mpjpe']:.4f}")
        print(f"  PCK@2:    {metrics['pck@2.0']:.2f} %")
    
    print(f"\n{'=' * 80}")
    print(f"SUMMARY: Aggregated Metrics (Mean +- Std)")
    print(f"{'=' * 80}")
    
    keys = all_metrics[0].keys()
    for key in keys:
        values = [m[key] for m in all_metrics]
        print(f"  {key:<12}: {np.mean(values):.6f} +- {np.std(values):.6f}")
    
    best_idx = np.argmin([m['mse'] for m in all_metrics])
    print(f"\n Best Checkpoint: {checkpoint_files[best_idx]}")
    print(f"   MSE:   {all_metrics[best_idx]['mse']:.6f}")
    print(f"   MPJPE: {all_metrics[best_idx]['mpjpe']:.4f}")
    
    results_path = os.path.join(args.output_dir, f'{args.method}_multi_ckpt_results.npy')
    np.save(results_path, all_metrics)
    print(f"\nResults saved to: {results_path}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
