#!/usr/bin/env python
"""
Evaluation for All 4 Hand Pose Methods
1. Baseline
2. freq_decomp
3. latent_only
4. latent_freq
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


# ============ Models ============
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


class LatentEncoder(torch.nn.Module):
    def __init__(self, in_channels=4, hidden_dim=256):
        super().__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels, hidden_dim // 4, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv3d(hidden_dim // 4, hidden_dim // 2, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv3d(hidden_dim // 2, hidden_dim, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            torch.nn.ReLU(),
        )
    
    def forward(self, z):
        return self.conv_layers(z)


class FrameVAE(torch.nn.Module):
    def __init__(self, in_channels=3, latent_dim=4, hidden_dim=64):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, hidden_dim, 4, stride=2, padding=1),
            torch.nn.BatchNorm2d(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Conv2d(hidden_dim, hidden_dim * 2, 4, stride=2, padding=1),
            torch.nn.BatchNorm2d(hidden_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 4, stride=2, padding=1),
            torch.nn.BatchNorm2d(hidden_dim * 4),
            torch.nn.ReLU(),
        )
        self.fc_mu = torch.nn.Conv2d(hidden_dim * 4, latent_dim, 3, padding=1)
        self.fc_var = torch.nn.Conv2d(hidden_dim * 4, latent_dim, 3, padding=1)
    
    def encode(self, x):
        B, C, T, H, W = x.shape
        x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        h = self.encoder(x_reshaped)
        mu = self.fc_mu(h)
        logvar = self.fc_var(h)
        z = mu
        z = z.view(B, T, -1, 16, 16).permute(0, 2, 1, 3, 4)
        return z


class HandPoseLatentOnly(torch.nn.Module):
    def __init__(self, in_channels=3, num_frames=8, hidden_dim=256, num_pose_params=51, 
                 vae_ckpt='/data/data5/zhaoran/paper_code/exo/latent/outputs/vae/checkpoints/best_model.pt'):
        super().__init__()
        self.vae = FrameVAE(in_channels=3, latent_dim=4)
        
        if os.path.exists(vae_ckpt):
            checkpoint = torch.load(vae_ckpt, map_location='cpu')
            self.vae.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded VAE from {vae_ckpt}")
        
        for param in self.vae.parameters():
            param.requires_grad = False
        
        self.encoder = LatentEncoder(in_channels=4, hidden_dim=hidden_dim)
        
        with torch.no_grad():
            dummy = torch.randn(1, 4, num_frames, 16, 16)
            out = self.encoder(dummy)
            regressor_input_dim = out.shape[1] * out.shape[3] * out.shape[4]
        
        self.regressor = RegressorHead(regressor_input_dim, hidden_dim=512, num_pose_params=num_pose_params)
    
    def forward(self, x):
        B, C, T, H, W = x.shape
        
        with torch.no_grad():
            z = self.vae.encode(x)
        
        feat = self.encoder(z)
        feat_flat = feat.permute(0, 2, 1, 3, 4).reshape(B * T, -1)
        pose = self.regressor(feat_flat)
        return pose


# ============ Metrics ============
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
def evaluate_method(model_class, method_name, val_loader, device, args):
    print(f"\n{'=' * 80}")
    print(f"Evaluating: {method_name.upper()}")
    print(f"{'=' * 80}")
    
    checkpoint_dir = os.path.join(args.output_dir, method_name, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return None
    
    checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_")])
    
    if len(checkpoint_files) == 0:
        print("No checkpoints found!")
        return None
    
    num_ckpts = min(args.num_ckpts, len(checkpoint_files))
    checkpoint_files = checkpoint_files[-num_ckpts:]
    
    all_metrics = []
    
    for ckpt_file in checkpoint_files:
        ckpt_path = os.path.join(checkpoint_dir, ckpt_file)
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        if method_name == 'latent_only':
            model = HandPoseLatentOnly()
        else:
            model = HandPoseBaseline()
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        model.eval()
        all_preds = []
        all_targets = []
        
        for batch in tqdm(val_loader, desc=f"Evaluating {ckpt_file}", leave=False):
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
        all_metrics.append(metrics)
    
    print(f"\nResults for {method_name} (average over {num_ckpts} checkpoints):")
    print("-" * 80)
    
    keys = ['mse', 'mae', 'rmse', 'mpjpe', 'pck@1.0', 'pck@2.0', 'pck@5.0', 'pck@10.0']
    aggregated = {}
    for key in keys:
        values = [m[key] for m in all_metrics]
        aggregated[f'{key}_mean'] = np.mean(values)
        aggregated[f'{key}_std'] = np.std(values)
    
    print(f"  Basic Metrics:")
    print(f"    MSE:      {aggregated['mse_mean']:.6f} +- {aggregated['mse_std']:.6f}")
    print(f"    MAE:      {aggregated['mae_mean']:.6f} +- {aggregated['mae_std']:.6f}")
    print(f"    RMSE:     {aggregated['rmse_mean']:.6f} +- {aggregated['rmse_std']:.6f}")
    print(f"\n  Pose Estimation Metrics:")
    print(f"    MPJPE:    {aggregated['mpjpe_mean']:.4f} +- {aggregated['mpjpe_std']:.4f}")
    print(f"\n  PCK Metrics (%):")
    print(f"    PCK@1:    {aggregated['pck@1.0_mean']:.2f} +- {aggregated['pck@1.0_std']:.2f} %")
    print(f"    PCK@2:    {aggregated['pck@2.0_mean']:.2f} +- {aggregated['pck@2.0_std']:.2f} %")
    print(f"    PCK@5:    {aggregated['pck@5.0_mean']:.2f} +- {aggregated['pck@5.0_std']:.2f} %")
    print(f"    PCK@10:   {aggregated['pck@10.0_mean']:.2f} +- {aggregated['pck@10.0_std']:.2f} %")
    
    best_idx = np.argmin([m['mse'] for m in all_metrics])
    print(f"\n  Best Checkpoint: {checkpoint_files[best_idx]}")
    print(f"    MSE:   {all_metrics[best_idx]['mse']:.6f}")
    print(f"    MPJPE: {all_metrics[best_idx]['mpjpe']:.4f}")
    
    return aggregated


def main():
    parser = argparse.ArgumentParser(description='Evaluate All 4 Hand Pose Methods')
    parser.add_argument('--methods', type=str, nargs='+', 
                       default=['baseline', 'latent_only'],
                       help='Methods to evaluate')
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
    print("Hand Pose Estimation - Evaluate All Methods")
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
    
    all_results = {}
    
    for method in args.methods:
        results = evaluate_method(None, method, val_loader, device, args)
        if results is not None:
            all_results[method] = results
    
    if len(all_results) > 1:
        print(f"\n{'=' * 80}")
        print(f"COMPARISON SUMMARY: MSE (Lower is Better)")
        print(f"{'=' * 80}")
        
        baseline_mse = all_results.get('baseline', {}).get('mse_mean', None)
        
        for method in all_results:
            mse = all_results[method]['mse_mean']
            if baseline_mse is not None and method != 'baseline':
                improvement = (baseline_mse - mse) / baseline_mse * 100
                print(f"  {method:<15}: {mse:.6f}  ({improvement:+.2f}% vs baseline)")
            else:
                print(f"  {method:<15}: {mse:.6f}")
    
    results_path = os.path.join(args.output_dir, 'all_methods_comparison.npy')
    np.save(results_path, all_results)
    print(f"\n{'=' * 80}")
    print(f"Results saved to: {results_path}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
