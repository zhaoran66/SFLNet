"""
Multi-Checkpoint Evaluation for Hand Pose Estimation
Evaluates multiple checkpoints and computes average metrics for robustness
"""
import os
import sys
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

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


# ============ Model Definition ============
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


# ============ Metric Functions ============
def compute_metrics(pred, target):
    """Compute all metrics"""
    mse = F.mse_loss(pred, target).item()
    mae = F.l1_loss(pred, target).item()
    rmse = np.sqrt(mse)
    
    # MPJPE in pose space
    error = torch.sqrt(torch.sum((pred - target) ** 2, dim=-1))
    mpjpe = torch.mean(error).item()
    
    # PCK at different thresholds
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


def compute_temporal_consistency(pose_pred, pose_target, num_frames=8):
    B = pose_pred.shape[0] // num_frames
    pose_pred_seq = pose_pred.reshape(B, num_frames, -1)
    pose_target_seq = pose_target.reshape(B, num_frames, -1)
    
    pred_velocity = pose_pred_seq[:, 1:] - pose_pred_seq[:, :-1]
    target_velocity = pose_target_seq[:, 1:] - pose_target_seq[:, :-1]
    velocity_error = torch.mean(torch.abs(pred_velocity - target_velocity)).item()
    
    if num_frames >= 3:
        pred_accel = pred_velocity[:, 1:] - pred_velocity[:, :-1]
        target_accel = target_velocity[:, 1:] - target_velocity[:, :-1]
        accel_error = torch.mean(torch.abs(pred_accel - target_accel)).item()
    else:
        accel_error = 0.0
    
    return velocity_error, accel_error


@torch.no_grad()
def evaluate_single_checkpoint(model, data_loader, device, method_name):
    """Evaluate a single checkpoint"""
    model.eval()
    
    all_preds = []
    all_targets = []
    
    for batch in tqdm(data_loader, desc=f"Evaluating {method_name}", leave=False):
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
    velocity_error, accel_error = compute_temporal_consistency(all_preds, all_targets)
    metrics['velocity_error'] = velocity_error
    metrics['acceleration_error'] = accel_error
    
    return metrics


def aggregate_metrics(metric_list):
    """Aggregate metrics from multiple checkpoints"""
    if not metric_list:
        return {}
    
    aggregated = {}
    keys = metric_list[0].keys()
    
    for key in keys:
        values = [m[key] for m in metric_list]
        aggregated[f'{key}_mean'] = np.mean(values)
        aggregated[f'{key}_std'] = np.std(values)
        aggregated[f'{key}_min'] = np.min(values)
        aggregated[f'{key}_max'] = np.max(values)
    
    return aggregated


def main():
    parser = argparse.ArgumentParser(description='Multi-Checkpoint Evaluation')
    parser.add_argument('--method', type=str, default='baseline',
                       choices=['baseline', 'freq_decomp', 'latent_only', 'latent_freq'],
                       help='Which method to evaluate')
    parser.add_argument('--output_dir', type=str, 
                       default='/data/data5/zhaoran/paper_code/exo/Ours/new/outputs_hand_pose',
                       help='Output directory containing checkpoints')
    parser.add_argument('--num_ckpts', type=int, default=5,
                       help='Number of recent checkpoints to evaluate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    config = Config()
    config.training.batch_size = 4
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("=" * 80)
    print("Multi-Checkpoint Evaluation for Robustness")
    print("=" * 80)
    
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
    ckpt_info = []
    
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
        
        metrics = evaluate_single_checkpoint(model, val_loader, device, args.method)
        all_metrics.append(metrics)
        ckpt_info.append({'epoch': epoch, 'val_loss': val_loss, **metrics})
        
        print(f"  MSE:      {metrics['mse']:.6f}")
        print(f"  MAE:      {metrics['mae']:.6f}")
        print(f"  MPJPE:    {metrics['mpjpe']:.4f}")
        print(f"  PCK@2:    {metrics['pck@2.0']:.2f} %")
    
    # Aggregate results
    print(f"\n{'=' * 80}")
    print(f"SUMMARY: Aggregated Metrics (Mean  Std)")
    print(f"{'=' * 80}")
    
    aggregated = aggregate_metrics(all_metrics)
    
    print(f" Basic Metrics:")
    print(f"   MSE:      {aggregated['mse_mean']:.6f}  {aggregated['mse_std']:.6f}")
    print(f"   MAE:      {aggregated['mae_mean']:.6f}  {aggregated['mae_std']:.6f}")
    print(f"   RMSE:     {aggregated['rmse_mean']:.6f}  {aggregated['rmse_std']:.6f}")
    print(f"\n Pose Estimation Metrics:")
    print(f"   MPJPE:    {aggregated['mpjpe_mean']:.4f}  {aggregated['mpjpe_std']:.4f}")
    print(f"\n PCK Metrics (%):")
    print(f"   PCK@1:    {aggregated['pck@1.0_mean']:.2f}  {aggregated['pck@1.0_std']:.2f} %")
    print(f"   PCK@2:    {aggregated['pck@2.0_mean']:.2f}  {aggregated['pck@2.0_std']:.2f} %")
    print(f"   PCK@5:    {aggregated['pck@5.0_mean']:.2f}  {aggregated['pck@5.0_std']:.2f} %")
    print(f"   PCK@10:   {aggregated['pck@10.0_mean']:.2f}  {aggregated['pck@10.0_std']:.2f} %")
    print(f"\n Temporal Consistency:")
    print(f"   Vel Err:  {aggregated['velocity_error_mean']:.6f}  {aggregated['velocity_error_std']:.6f}")
    print(f"   Acc Err:  {aggregated['acceleration_error_mean']:.6f}  {aggregated['acceleration_error_std']:.6f}")
    
    # Find best checkpoint
    best_idx = np.argmin([m['mse'] for m in all_metrics])
    print(f"\n Best Checkpoint: {checkpoint_files[best_idx]}")
    print(f"   Epoch: {ckpt_info[best_idx]['epoch']}")
    print(f"   MSE:   {all_metrics[best_idx]['mse']:.6f}")
    print(f"   MPJPE: {all_metrics[best_idx]['mpjpe']:.4f}")
    
    # Save results
    results_path = os.path.join(args.output_dir, f'{args.method}_multi_ckpt_results.npy')
    np.save(results_path, {'aggregated': aggregated, 'individual': ckpt_info})
    print(f"\nResults saved to: {results_path}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
