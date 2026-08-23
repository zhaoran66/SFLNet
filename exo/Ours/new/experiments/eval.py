"""
Evaluation script for hand pose estimation
Evaluates trained models on validation set
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

sys.path.insert(0, '/data/data5/zhaoran/paper_code/exo/Ours/new')
sys.path.insert(0, '/data/data5/zhaoran/paper_code/exo/Ours/new/experiments')
sys.path.insert(0, '/data/data5/zhaoran/paper_code/exo/Ours/new/experiments/1_baseline')
sys.path.insert(0, '/data/data5/zhaoran/paper_code/exo/Ours/new/experiments/2_freq_decomp')
sys.path.insert(0, '/data/data5/zhaoran/paper_code/exo/Ours/new/experiments/3_latent_only')
sys.path.insert(0, '/data/data5/zhaoran/paper_code/exo/Ours/new/experiments/4_latent_freq')

from configs.default_config import Config
from data.dataset import create_dataloaders
from train_common import set_seed, HandPoseDatasetWrapper


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


def load_model_from_checkpoint(checkpoint_path, model_class, model_kwargs, device):
    """Load model from checkpoint"""
    model = model_class(**model_kwargs)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Best val loss: {checkpoint.get('best_val_loss', 'N/A')}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
    
    model.to(device)
    model.eval()
    return model


def evaluate_baseline(val_loader, device, output_dir):
    """Evaluate Experiment 1: Baseline"""
    print("\n" + "=" * 70)
    print("Evaluating: Experiment 1 - Baseline")
    print("=" * 70)
    
    from model import HandPoseBaseline
    
    checkpoint_path = os.path.join(output_dir, "1_baseline/checkpoints/best_model.pt")
    model = load_model_from_checkpoint(
        checkpoint_path,
        HandPoseBaseline,
        {'in_channels': 3, 'num_frames': 8, 'hidden_dim': 128, 'num_pose_params': 51},
        device
    )
    
    metrics, _, _ = evaluate_model(model, val_loader, device, "baseline")
    
    print("\nResults:")
    print(f"  Avg Loss:     {metrics['avg_loss']:.6f}")
    print(f"  MSE:          {metrics['mse']:.6f}")
    print(f"  MAE:          {metrics['mae']:.6f}")
    print(f"  RMSE:         {metrics['rmse']:.6f}")
    print(f"  Rel Err:      {metrics['relative_error']:.6f}")
    
    return metrics


def evaluate_freq_decomp(val_loader, device, output_dir):
    """Evaluate Experiment 2: Frequency Decomposition"""
    print("\n" + "=" * 70)
    print("Evaluating: Experiment 2 - Frequency Decomposition")
    print("=" * 70)
    
    from model2 import HandPoseFreqDecomp
    
    checkpoint_path = os.path.join(output_dir, "2_freq_decomp/checkpoints/best_model.pt")
    if not os.path.exists(checkpoint_path):
        print("Model not trained yet. Skipping...")
        return None
    
    model = load_model_from_checkpoint(
        checkpoint_path,
        HandPoseFreqDecomp,
        {'in_channels': 3, 'num_frames': 8, 'hidden_dim': 256, 'num_pose_params': 51, 'use_freq_decomp': True},
        device
    )
    
    metrics, _, _ = evaluate_model(model, val_loader, device, "freq_decomp")
    
    print("\nResults:")
    print(f"  Avg Loss:     {metrics['avg_loss']:.6f}")
    print(f"  MSE:          {metrics['mse']:.6f}")
    print(f"  MAE:          {metrics['mae']:.6f}")
    print(f"  RMSE:         {metrics['rmse']:.6f}")
    print(f"  Rel Err:      {metrics['relative_error']:.6f}")
    
    return metrics


def evaluate_latent_only(val_loader, device, output_dir):
    """Evaluate Experiment 3: Latent Only"""
    print("\n" + "=" * 70)
    print("Evaluating: Experiment 3 - Latent Only")
    print("=" * 70)
    
    from model3 import HandPoseLatentOnly
    
    checkpoint_path = os.path.join(output_dir, "3_latent_only/checkpoints/best_model.pt")
    if not os.path.exists(checkpoint_path):
        print("Model not trained yet. Skipping...")
        return None
    
    model = load_model_from_checkpoint(
        checkpoint_path,
        HandPoseLatentOnly,
        {
            'in_channels': 3, 
            'num_frames': 8, 
            'hidden_dim': 256, 
            'num_pose_params': 51,
            'vae_path': '/data/data5/zhaoran/paper_code/exo/latent/outputs/vae/checkpoints/best_model.pt'
        },
        device
    )
    
    metrics, _, _ = evaluate_model(model, val_loader, device, "latent_only")
    
    print("\nResults:")
    print(f"  Avg Loss:     {metrics['avg_loss']:.6f}")
    print(f"  MSE:          {metrics['mse']:.6f}")
    print(f"  MAE:          {metrics['mae']:.6f}")
    print(f"  RMSE:         {metrics['rmse']:.6f}")
    print(f"  Rel Err:      {metrics['relative_error']:.6f}")
    
    return metrics


def evaluate_latent_freq(val_loader, device, output_dir):
    """Evaluate Experiment 4: Latent + Frequency"""
    print("\n" + "=" * 70)
    print("Evaluating: Experiment 4 - Latent + Frequency")
    print("=" * 70)
    
    from model4 import HandPoseLatentFreq
    
    checkpoint_path = os.path.join(output_dir, "4_latent_freq/checkpoints/best_model.pt")
    if not os.path.exists(checkpoint_path):
        print("Model not trained yet. Skipping...")
        return None
    
    model = load_model_from_checkpoint(
        checkpoint_path,
        HandPoseLatentFreq,
        {
            'in_channels': 3, 
            'num_frames': 8, 
            'hidden_dim': 256, 
            'num_pose_params': 51,
            'vae_path': '/data/data5/zhaoran/paper_code/exo/latent/outputs/vae/checkpoints/best_model.pt',
            'use_freq_decomp': True
        },
        device
    )
    
    metrics, _, _ = evaluate_model(model, val_loader, device, "latent_freq")
    
    print("\nResults:")
    print(f"  Avg Loss:     {metrics['avg_loss']:.6f}")
    print(f"  MSE:          {metrics['mse']:.6f}")
    print(f"  MAE:          {metrics['mae']:.6f}")
    print(f"  RMSE:         {metrics['rmse']:.6f}")
    print(f"  Rel Err:      {metrics['relative_error']:.6f}")
    
    return metrics


def print_comparison(all_metrics):
    """Print comparison table"""
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    
    methods = {
        'baseline': '1. Baseline',
        'freq_decomp': '2. +Frequency',
        'latent_only': '3. +Latent',
        'latent_freq': '4. +Latent +Freq',
    }
    
    metric_names = ['avg_loss', 'mse', 'mae', 'rmse', 'relative_error']
    metric_labels = ['Avg Loss', 'MSE', 'MAE', 'RMSE', 'Rel Err']
    
    # Print header
    print(f"{'Method':<20}", end="")
    for label in metric_labels:
        print(f"{label:<12}", end="")
    print()
    
    # Print each method
    baseline_mse = None
    for method_key, method_name in methods.items():
        if method_key in all_metrics and all_metrics[method_key] is not None:
            metrics = all_metrics[method_key]
            print(f"{method_name:<20}", end="")
            for metric in metric_names:
                print(f"{metrics[metric]:<12.6f}", end="")
            
            # Calculate improvement over baseline
            if method_key == 'baseline':
                baseline_mse = metrics['mse']
                print()
            else:
                if baseline_mse is not None:
                    improvement = (baseline_mse - metrics['mse']) / baseline_mse * 100
                    print(f" ({improvement:+.2f}% MSE)")
                else:
                    print()
        else:
            print(f"{method_name:<20}{'Not trained':<60}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate hand pose estimation models')
    parser.add_argument('--output_dir', type=str, 
                       default='/data/data5/zhaoran/paper_code/exo/Ours/new/outputs_hand_pose',
                       help='Output directory containing trained models')
    parser.add_argument('--method', type=str, default='all',
                       choices=['all', 'baseline', 'freq_decomp', 'latent_only', 'latent_freq'],
                       help='Which method to evaluate (default: all)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    config = Config()
    config.training.batch_size = 4
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("=" * 70)
    print("Hand Pose Estimation - Evaluation")
    print("=" * 70)
    
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
    
    all_metrics = {}
    
    if args.method == 'all' or args.method == 'baseline':
        all_metrics['baseline'] = evaluate_baseline(val_loader, device, args.output_dir)
    
    if args.method == 'all' or args.method == 'freq_decomp':
        all_metrics['freq_decomp'] = evaluate_freq_decomp(val_loader, device, args.output_dir)
    
    if args.method == 'all' or args.method == 'latent_only':
        all_metrics['latent_only'] = evaluate_latent_only(val_loader, device, args.output_dir)
    
    if args.method == 'all' or args.method == 'latent_freq':
        all_metrics['latent_freq'] = evaluate_latent_freq(val_loader, device, args.output_dir)
    
    print_comparison(all_metrics)
    
    # Save results
    results_path = os.path.join(args.output_dir, 'evaluation_results.npy')
    np.save(results_path, all_metrics)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
