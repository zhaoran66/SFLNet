"""
Full Evaluation for Hand Pose Estimation
Metrics: MPJPE, PA-MPJPE, PCK, Temporal Consistency, SSIM
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
def compute_mpjpe_pose_space(pred, target):
    """
    MPJPE in pose parameter space (51-dim)
    pred, target: (N, 51)
    """
    error = torch.sqrt(torch.sum((pred - target) ** 2, dim=-1))
    mpjpe = torch.mean(error)
    return mpjpe.item()


def compute_pa_mpjpe_pose_space(pred, target):
    """
    Procrustes Analysis in pose parameter space
    """
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    
    pred_centered = pred_np - np.mean(pred_np, axis=0, keepdims=True)
    target_centered = target_np - np.mean(target_np, axis=0, keepdims=True)
    
    H = pred_centered.T @ target_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    pred_aligned = pred_centered @ R.T
    error = np.mean(np.sqrt(np.sum((pred_aligned - target_centered) ** 2, axis=-1)))
    
    return error


def compute_pck_pose_space(pred, target, thresholds=[0.01, 0.05, 0.1]):
    """
    Percentage of Correct Keypoints in pose parameter space
    thresholds: in parameter space
    """
    error = torch.sqrt(torch.sum((pred - target) ** 2, dim=-1))
    
    pck_values = {}
    for threshold in thresholds:
        correct = (error < threshold).float()
        pck = torch.mean(correct) * 100
        pck_values[f'pck@{threshold}'] = pck.item()
    
    return pck_values


def compute_temporal_consistency(pose_pred, pose_target, num_frames=8):
    """
    Compute temporal consistency metrics
    """
    B = pose_pred.shape[0] // num_frames
    pose_pred_seq = pose_pred.reshape(B, num_frames, -1)
    pose_target_seq = pose_target.reshape(B, num_frames, -1)
    
    pred_velocity = pose_pred_seq[:, 1:] - pose_pred_seq[:, :-1]
    target_velocity = pose_target_seq[:, 1:] - pose_target_seq[:, :-1]
    velocity_error = torch.mean(torch.abs(pred_velocity - target_velocity))
    
    if num_frames >= 3:
        pred_accel = pred_velocity[:, 1:] - pred_velocity[:, :-1]
        target_accel = target_velocity[:, 1:] - target_velocity[:, :-1]
        accel_error = torch.mean(torch.abs(pred_accel - target_accel))
    else:
        accel_error = torch.tensor(0.0)
    
    return velocity_error.item(), accel_error.item()


def compute_ssim(img1, img2, window_size=11):
    """
    Simplified SSIM computation
    img1, img2: (C, H, W) in [-1, 1] range
    """
    # Normalize to [0, 1]
    img1 = (img1 + 1) / 2
    img2 = (img2 + 1) / 2
    
    # Compute mean
    mu1 = torch.mean(img1)
    mu2 = torch.mean(img2)
    
    # Compute variance
    sigma1_sq = torch.mean(img1 ** 2) - mu1 ** 2
    sigma2_sq = torch.mean(img2 ** 2) - mu2 ** 2
    
    # Compute covariance
    sigma12 = torch.mean(img1 * img2) - mu1 * mu2
    
    C1 = (0.01 * 1) ** 2
    C2 = (0.03 * 1) ** 2
    
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim = numerator / denominator
    return ssim.item()


@torch.no_grad()
def evaluate_model(model, data_loader, device, method_name):
    """Full evaluation with all metrics"""
    model.eval()
    
    all_preds = []
    all_targets = []
    all_videos = []
    
    for batch in tqdm(data_loader, desc=f"Evaluating {method_name}"):
        exo_video = batch["exo_video"].to(device, non_blocking=True)
        hand_pose_target = batch["hand_pose"].to(device, non_blocking=True)
        
        B, T, D = hand_pose_target.shape
        hand_pose_target = hand_pose_target.reshape(B * T, D)
        
        pose_pred = model(exo_video)
        
        all_preds.append(pose_pred.cpu())
        all_targets.append(hand_pose_target.cpu())
        all_videos.append(exo_video.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_videos = torch.cat(all_videos, dim=0)
    
    metrics = {}
    
    # Basic MSE, MAE, RMSE
    metrics['mse'] = F.mse_loss(all_preds, all_targets).item()
    metrics['mae'] = F.l1_loss(all_preds, all_targets).item()
    metrics['rmse'] = np.sqrt(metrics['mse'])
    
    # Pose MPJPE (in parameter space)
    metrics['mpjpe_pose'] = compute_mpjpe_pose_space(all_preds, all_targets)
    
    # PA-MPJPE
    metrics['pa_mpjpe_pose'] = compute_pa_mpjpe_pose_space(all_preds, all_targets)
    
    # PCK at different thresholds
    pck_values = compute_pck_pose_space(all_preds, all_targets, thresholds=[0.01, 0.05, 0.1])
    metrics.update(pck_values)
    
    # Joint-wise error (treat each dimension as a "joint")
    joint_error = torch.mean(torch.abs(all_preds - all_targets), dim=0)
    metrics['mean_joint_error'] = torch.mean(joint_error).item()
    metrics['max_joint_error'] = torch.max(joint_error).item()
    
    # Temporal Consistency
    velocity_error, accel_error = compute_temporal_consistency(all_preds, all_targets, num_frames=8)
    metrics['velocity_error'] = velocity_error
    metrics['acceleration_error'] = accel_error
    
    # SSIM (between first and last frame of each sequence)
    B, C, T, H, W = all_videos.shape
    ssim_values = []
    for i in range(min(100, B)):  # Compute for first 100 samples
        first_frame = all_videos[i, :, 0]  # (C, H, W)
        for t in range(1, min(T, 8)):
            next_frame = all_videos[i, :, t]  # (C, H, W)
            ssim_val = compute_ssim(first_frame, next_frame)
            ssim_values.append(ssim_val)
    
    if ssim_values:
        metrics['ssim'] = np.mean(ssim_values)
    else:
        metrics['ssim'] = 0
    
    return metrics


def main():
    set_seed(42)
    
    config = Config()
    config.training.batch_size = 4
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("=" * 80)
    print("Hand Pose Estimation - Full Evaluation")
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
    
    methods = [
        ("baseline", "1. Baseline", HandPoseBaseline),
    ]
    
    all_results = {}
    
    for method_name, display_name, model_class in methods:
        print(f"\n{'=' * 80}")
        print(f"Evaluating: {display_name}")
        print(f"{'=' * 80}")
        
        checkpoint_path = f"/data/data5/zhaoran/paper_code/exo/Ours/new/outputs_hand_pose/{method_name}/checkpoints/best_model.pt"
        
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}. Skipping...")
            continue
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model = model_class(
            in_channels=3,
            num_frames=8,
            hidden_dim=128,
            num_pose_params=51,
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Best val loss: {checkpoint.get('best_val_loss', 'N/A'):.6f}")
        
        print("\nComputing metrics...")
        metrics = evaluate_model(model, val_loader, device, method_name)
        all_results[method_name] = metrics
        
        print(f"\n{'=' * 80}")
        print(f"RESULTS: {display_name}")
        print(f"{'=' * 80}")
        print(f" Basic Metrics:")
        print(f"   MSE:              {metrics['mse']:.6f}")
        print(f"   MAE:              {metrics['mae']:.6f}")
        print(f"   RMSE:             {metrics['rmse']:.6f}")
        print(f"\n Pose Parameter Metrics:")
        print(f"   MPJPE (pose):     {metrics['mpjpe_pose']:.6f}")
        print(f"   PA-MPJPE (pose):  {metrics['pa_mpjpe_pose']:.6f}")
        print(f"\n PCK Metrics (%):")
        print(f"   PCK@0.01:         {metrics['pck@0.01']:.2f} %")
        print(f"   PCK@0.05:         {metrics['pck@0.05']:.2f} %")
        print(f"   PCK@0.1:          {metrics['pck@0.1']:.2f} %")
        print(f"\n Joint-wise Errors:")
        print(f"   Mean Joint Err:   {metrics['mean_joint_error']:.6f}")
        print(f"   Max Joint Err:    {metrics['max_joint_error']:.6f}")
        print(f"\n Temporal Consistency Metrics:")
        print(f"   Velocity Error:   {metrics['velocity_error']:.6f}")
        print(f"   Acceleration Err: {metrics['acceleration_error']:.6f}")
        print(f"\n Image Quality Metrics:")
        print(f"   SSIM:             {metrics['ssim']:.4f}")
        print(f"{'=' * 80}")
    
    # Save results
    results_path = "/data/data5/zhaoran/paper_code/exo/Ours/new/outputs_hand_pose/full_evaluation_results.npy"
    np.save(results_path, all_results)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
