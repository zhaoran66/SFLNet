"""
Evaluation Script for 4 Unified Hand Pose Methods
All methods use SAME interface: Input (B, 3, T, 128, 128) -> Output (B*T, 51)
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
from models.hand_pose_unified import (
    HandPoseBaseline,
    HandPoseFreqDecomp,
    HandPoseLatentOnly,
    HandPoseLatentFreq,
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


def compute_metrics(pred, target):
    """Compute evaluation metrics"""
    mse = F.mse_loss(pred, target).item()
    mae = F.l1_loss(pred, target).item()
    rmse = np.sqrt(mse)
    
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    
    relative_error = np.mean(np.abs(pred_np - target_np) / (np.abs(target_np) + 1e-8))
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'relative_error': relative_error,
    }


@torch.no_grad()
def evaluate_model(model, data_loader, device, method_name):
    """Evaluate a single model"""
    model.eval()
    
    all_preds = []
    all_targets = []
    all_losses = []
    
    for batch in tqdm(data_loader, desc=f"Evaluating {method_name}"):
        exo_video = batch["exo_video"].to(device, non_blocking=True)
        hand_pose_target = batch["hand_pose"].to(device, non_blocking=True)
        
        B, T, D = hand_pose_target.shape
        hand_pose_target = hand_pose_target.reshape(B * T, D)
        
        pose_pred = model(exo_video)
        
        loss = F.mse_loss(pose_pred, hand_pose_target)
        all_losses.append(loss.item())
        
        all_preds.append(pose_pred.cpu())
        all_targets.append(hand_pose_target.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    metrics = compute_metrics(all_preds, all_targets)
    metrics['avg_loss'] = np.mean(all_losses)
    
    return metrics, all_preds, all_targets


def load_model(checkpoint_path, model_class, model_kwargs, device):
    """Load model from checkpoint"""
    model = model_class(**model_kwargs)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Best val loss: {checkpoint.get('best_val_loss', 'N/A')}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}, using random weights")
    
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description='Evaluate 4 Unified Hand Pose Methods')
    parser.add_argument('--output_dir', type=str, default='./outputs_hand_pose',
                       help='Output directory containing trained models')
    parser.add_argument('--vae_path', type=str, 
                       default='/data/data5/zhaoran/paper_code/exo/latent/outputs/vae/checkpoints/best_model.pt',
                       help='Path to pretrained VAE')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    config = Config()
    config.training.batch_size = args.batch_size
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("="*70)
    print("Hand Pose Estimation - Evaluation of 4 Unified Methods")
    print("="*70)
    print(f"  All methods use SAME interface:")
    print(f"    Input:  (B, 3, T, 128, 128)  RGB video")
    print(f"    Output: (B*T, 51)              MANO hand pose")
    print("="*70)
    
    print("\nCreating dataloaders...")
    _, val_loader = create_dataloaders(config)
    
    val_dataset = HandPoseDatasetWrapper(val_loader.dataset)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    
    print(f"Validation samples: {len(val_dataset)}")
    
    methods = [
        ('baseline', 'Baseline (Direct RGB)'),
        ('freq_decomp', 'Baseline + Frequency Decomp'),
        ('latent_only', 'Baseline + Latent'),
        ('latent_freq', 'Baseline + Latent + Frequency Decomp'),
    ]
    
    all_results = {}
    
    for method_name, method_desc in methods:
        print(f"\n{'='*70}")
        print(f"Evaluating: {method_desc}")
        print(f"{'='*70}")
        
        checkpoint_path = os.path.join(args.output_dir, method_name, 'checkpoints', 'best_model.pt')
        
        if method_name == 'baseline':
            model = load_model(
                checkpoint_path,
                HandPoseBaseline,
                {'in_channels': 3, 'num_frames': config.data.num_frames, 'hidden_dim': 128, 'num_pose_params': 51},
                device
            )
        elif method_name == 'freq_decomp':
            model = load_model(
                checkpoint_path,
                HandPoseFreqDecomp,
                {'in_channels': 3, 'num_frames': config.data.num_frames, 'hidden_dim': 256, 'num_pose_params': 51, 'use_freq_decomp': True},
                device
            )
        elif method_name == 'latent_only':
            model = load_model(
                checkpoint_path,
                HandPoseLatentOnly,
                {'in_channels': 3, 'num_frames': config.data.num_frames, 'hidden_dim': 256, 'num_pose_params': 51, 'vae_path': args.vae_path},
                device
            )
        elif method_name == 'latent_freq':
            model = load_model(
                checkpoint_path,
                HandPoseLatentFreq,
                {'in_channels': 3, 'num_frames': config.data.num_frames, 'hidden_dim': 256, 'num_pose_params': 51, 'vae_path': args.vae_path, 'use_freq_decomp': True},
                device
            )
        
        metrics, _, _ = evaluate_model(model, val_loader, device, method_name)
        all_results[method_name] = metrics
        
        print(f"\nResults for {method_desc}:")
        print(f"  Avg Loss: {metrics['avg_loss']:.6f}")
        print(f"  MSE:      {metrics['mse']:.6f}")
        print(f"  MAE:      {metrics['mae']:.6f}")
        print(f"  RMSE:     {metrics['rmse']:.6f}")
        print(f"  Rel Err:  {metrics['relative_error']:.6f}")
    
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    baseline_name = 'baseline'
    if baseline_name in all_results:
        baseline_mse = all_results[baseline_name]['mse']
        for method_name, method_desc in methods[1:]:
            if method_name in all_results:
                mse_improvement = (baseline_mse - all_results[method_name]['mse']) / baseline_mse * 100
                print(f"\n{method_desc} vs {baseline_name}:")
                print(f"  MSE Improvement: {mse_improvement:+.2f}%")
                if mse_improvement > 0:
                    print(f"  Status: BETTER than baseline")
                else:
                    print(f"  Status: WORSE than baseline")
    
    print(f"\n{'='*70}")
    print("All Metrics Comparison:")
    print(f"{'='*70}")
    
    metric_names = ['avg_loss', 'mse', 'mae', 'rmse', 'relative_error']
    print(f"{'Method':<45}", end="")
    for metric in metric_names:
        print(f"{metric:<15}", end="")
    print()
    
    for method_name, method_desc in methods:
        if method_name in all_results:
            metrics = all_results[method_name]
            print(f"{method_desc:<45}", end="")
            for metric in metric_names:
                print(f"{metrics[metric]:<15.6f}", end="")
            print()
    
    results_path = os.path.join(args.output_dir, 'evaluation_results.npy')
    np.save(results_path, all_results)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
