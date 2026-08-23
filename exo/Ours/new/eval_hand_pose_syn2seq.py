"""
Evaluation Script for Syn2Seq Hand Pose Estimation
Compares Baseline vs Syn2Seq-Method on the same test set
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
from models.hand_pose_syn2seq import (
    HandPoseBaseline,
    HandPoseSyn2Seq,
    HandPoseSyn2SeqSimple,
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
        import numpy as np
        pose_data = np.load(pose_file)
        hand_pose = torch.from_numpy(pose_data["pose_m"][:, 0]).float()
        
        if hand_pose.shape[0] > num_frames:
            hand_pose = hand_pose[:num_frames]
        
        return {
            "exo_video": exo_video,
            "hand_pose": hand_pose,
        }


def compute_metrics(pred, target):
    """
    Compute evaluation metrics
    pred, target: (B*T, 51)
    """
    mse = F.mse_loss(pred, target).item()
    mae = F.l1_loss(pred, target).item()
    
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    
    rmse = np.sqrt(mse)
    
    relative_error = np.mean(np.abs(pred_np - target_np) / (np.abs(target_np) + 1e-8))
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'relative_error': relative_error,
    }


def evaluate_model(model, data_loader, device, method_name):
    """Evaluate a single model"""
    model.eval()
    
    all_preds = []
    all_targets = []
    all_losses = []
    
    with torch.no_grad():
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
    parser = argparse.ArgumentParser(description='Evaluate Syn2Seq Hand Pose Estimation')
    parser.add_argument('--output_dir', type=str, default='./outputs_syn2seq',
                       help='Output directory containing trained models')
    parser.add_argument('--baseline_ckpt', type=str, default=None,
                       help='Path to baseline checkpoint (default: <output_dir>/baseline/checkpoints/best_model.pt)')
    parser.add_argument('--syn2seq_ckpt', type=str, default=None,
                       help='Path to Syn2Seq checkpoint (default: <output_dir>/syn2seq/checkpoints/best_model.pt)')
    parser.add_argument('--num_key_frames', type=int, default=2,
                       help='Number of key frames for Syn2Seq method')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--compare_all', action='store_true',
                       help='Compare all variants: baseline, syn2seq_simple, syn2seq')
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    config = Config()
    config.training.batch_size = args.batch_size
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("="*70)
    print("Syn2Seq Hand Pose Estimation Evaluation")
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
    
    models_to_evaluate = []
    
    if args.compare_all:
        methods = ['baseline', 'syn2seq_simple', 'syn2seq']
    else:
        methods = ['baseline', 'syn2seq']
    
    for method in methods:
        if method == 'baseline':
            ckpt_path = args.baseline_ckpt or os.path.join(args.output_dir, 'baseline/checkpoints/best_model.pt')
            model_class = HandPoseBaseline
            model_kwargs = {
                'in_channels': 3,
                'num_frames': config.data.num_frames,
                'hidden_dim': 128,
                'num_pose_params': 51,
            }
        elif method == 'syn2seq_simple':
            ckpt_path = os.path.join(args.output_dir, 'syn2seq_simple/checkpoints/best_model.pt')
            model_class = HandPoseSyn2SeqSimple
            model_kwargs = {
                'in_channels': 3,
                'num_frames': config.data.num_frames,
                'hidden_dim': 128,
                'num_pose_params': 51,
                'num_key_frames': args.num_key_frames,
            }
        else:  # syn2seq
            ckpt_path = args.syn2seq_ckpt or os.path.join(args.output_dir, 'syn2seq/checkpoints/best_model.pt')
            model_class = HandPoseSyn2Seq
            model_kwargs = {
                'in_channels': 3,
                'num_frames': config.data.num_frames,
                'hidden_dim': 128,
                'num_pose_params': 51,
                'num_key_frames': args.num_key_frames,
                'use_freq_decomp': True,
                'use_learned_interp': True,
            }
        
        if os.path.exists(ckpt_path):
            print(f"\nLoading {method} model...")
            model = load_model(ckpt_path, model_class, model_kwargs, device)
            models_to_evaluate.append((method, model))
        else:
            print(f"\nWarning: {method} checkpoint not found at {ckpt_path}, skipping")
    
    if len(models_to_evaluate) == 0:
        print("\nNo models to evaluate!")
        return
    
    print(f"\nEvaluating {len(models_to_evaluate)} models...")
    
    all_results = {}
    for method_name, model in models_to_evaluate:
        metrics, preds, targets = evaluate_model(model, val_loader, device, method_name)
        all_results[method_name] = metrics
        
        print(f"\n{method_name.upper()} Results:")
        print(f"  Avg Loss: {metrics['avg_loss']:.6f}")
        print(f"  MSE:      {metrics['mse']:.6f}")
        print(f"  MAE:      {metrics['mae']:.6f}")
        print(f"  RMSE:     {metrics['rmse']:.6f}")
        print(f"  Rel Err:  {metrics['relative_error']:.6f}")
    
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    baseline_name = 'baseline'
    if baseline_name in all_results:
        baseline_mse = all_results[baseline_name]['mse']
        for method_name in all_results:
            if method_name != baseline_name:
                mse_improvement = (baseline_mse - all_results[method_name]['mse']) / baseline_mse * 100
                print(f"{method_name} vs {baseline_name}:")
                print(f"  MSE Improvement: {mse_improvement:+.2f}%")
                if mse_improvement > 0:
                    print(f"  Status: BETTER than baseline")
                else:
                    print(f"  Status: WORSE than baseline")
    
    print("\n" + "="*70)
    print("All Metrics:")
    print("="*70)
    
    metric_names = ['avg_loss', 'mse', 'mae', 'rmse', 'relative_error']
    print(f"{'Method':<20}", end="")
    for metric in metric_names:
        print(f"{metric:<15}", end="")
    print()
    
    for method_name, metrics in all_results.items():
        print(f"{method_name:<20}", end="")
        for metric in metric_names:
            print(f"{metrics[metric]:<15.6f}", end="")
        print()
    
    results_path = os.path.join(args.output_dir, 'evaluation_results.npy')
    np.save(results_path, all_results)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
