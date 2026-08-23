"""
Simple evaluation script for baseline model
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


def main():
    set_seed(42)
    
    config = Config()
    config.training.batch_size = 4
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("=" * 70)
    print("Hand Pose Estimation - Baseline Evaluation")
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
    
    print("\nLoading model...")
    checkpoint_path = "/data/data5/zhaoran/paper_code/exo/Ours/new/outputs_hand_pose/baseline/checkpoints/best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = HandPoseBaseline(
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
    print(f"  Best val loss: {checkpoint.get('best_val_loss', 'N/A')}")
    
    print("\nEvaluating...")
    metrics, _, _ = evaluate_model(model, val_loader, device, "baseline")
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"  Avg Loss:     {metrics['avg_loss']:.6f}")
    print(f"  MSE:          {metrics['mse']:.6f}")
    print(f"  MAE:          {metrics['mae']:.6f}")
    print(f"  RMSE:         {metrics['rmse']:.6f}")
    print(f"  Rel Err:      {metrics['relative_error']:.6f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
