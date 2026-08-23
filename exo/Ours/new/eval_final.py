#!/usr/bin/env python
"""
Final Evaluation - Simplified Version
Just compare MSE of all 3 methods: baseline, freq_decomp, latent_only
"""
import os
import sys
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


def compute_mse(pred, target):
    return F.mse_loss(pred, target).item()


@torch.no_grad()
def eval_single_method(method_name, get_model_fn, val_loader, device, output_dir, num_ckpts=5):
    """Evaluate a single method using best checkpoint"""
    ckpt_dir = os.path.join(output_dir, method_name, "checkpoints")
    ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
    
    if not os.path.exists(ckpt_path):
        # try to find any checkpoint
        ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.startswith("checkpoint_")])
        if not ckpts:
            print(f"No checkpoint found for {method_name}!")
            return None
        ckpt_path = os.path.join(ckpt_dir, ckpts[-1])
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = get_model_fn()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"\n{'='*80}")
    print(f"Evaluating: {method_name.upper()}")
    print(f"{'='*80}")
    print(f"Checkpoint: {os.path.basename(ckpt_path)}")
    
    all_mse = []
    
    for batch in tqdm(val_loader, desc="Evaluating"):
        exo_video = batch["exo_video"].to(device, non_blocking=True)
        hand_pose_target = batch["hand_pose"].to(device, non_blocking=True)
        
        B, T, D = hand_pose_target.shape
        hand_pose_target = hand_pose_target.reshape(B * T, D)
        
        pose_pred = model(exo_video)
        
        mse = compute_mse(pose_pred, hand_pose_target)
        all_mse.append(mse)
    
    avg_mse = np.mean(all_mse)
    print(f"\n  Avg MSE: {avg_mse:.6f}")
    
    return avg_mse


def main():
    set_seed(42)
    
    config = Config()
    config.training.batch_size = 4
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = '/data/data5/zhaoran/paper_code/exo/Ours/new/outputs_hand_pose'
    
    print(f"Using device: {device}")
    print("=" * 80)
    print("Final Evaluation - 3 Methods Comparison")
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
    
    # ========== Method 1: Baseline ==========
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
        def __init__(self):
            super().__init__()
            self.encoder = PixelEncoder(in_channels=3, hidden_dim=128)
            
            with torch.no_grad():
                dummy = torch.randn(1, 3, 8, 128, 128)
                out = self.encoder(dummy)
                regressor_input_dim = out.shape[1] * out.shape[3] * out.shape[4]
            
            self.regressor = RegressorHead(regressor_input_dim, hidden_dim=512, num_pose_params=51)
        
        def forward(self, x):
            B, C, T, H, W = x.shape
            feat = self.encoder(x)
            feat_flat = feat.permute(0, 2, 1, 3, 4).reshape(B * T, -1)
            pose = self.regressor(feat_flat)
            return pose
    
    # ========== Method 2: Latent VAE ==========
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
            z = mu
            z = z.view(B, T, -1, 16, 16).permute(0, 2, 1, 3, 4)
            return z
    
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
    
    class HandPoseLatentOnly(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vae = FrameVAE(in_channels=3, latent_dim=4)
            
            vae_ckpt = '/data/data5/zhaoran/paper_code/exo/latent/outputs/vae/checkpoints/best_model.pt'
            if os.path.exists(vae_ckpt):
                checkpoint = torch.load(vae_ckpt, map_location='cpu')
                self.vae.load_state_dict(checkpoint['model_state_dict'])
            
            for param in self.vae.parameters():
                param.requires_grad = False
            
            self.encoder = LatentEncoder(in_channels=4, hidden_dim=256)
            
            with torch.no_grad():
                dummy = torch.randn(1, 4, 8, 16, 16)
                out = self.encoder(dummy)
                regressor_input_dim = out.shape[1] * out.shape[3] * out.shape[4]
            
            self.regressor = RegressorHead(regressor_input_dim, hidden_dim=512, num_pose_params=51)
        
        def forward(self, x):
            B, C, T, H, W = x.shape
            
            with torch.no_grad():
                z = self.vae.encode(x)
            
            feat = self.encoder(z)
            feat_flat = feat.permute(0, 2, 1, 3, 4).reshape(B * T, -1)
            pose = self.regressor(feat_flat)
            return pose
    
    # ========== Method 3: Freq Decomp (Latent based) ==========
    class SoftSpectralDecomposition(torch.nn.Module):
        def __init__(self, freq_size=(8, 8), feat_dim=4, sigma=0.5):
            super().__init__()
            self.freq_size = freq_size
            
            h, w = freq_size
            y_coords, x_coords = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            center_y, center_x = h // 2, w // 2
            
            dist = torch.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
            max_dist = torch.max(dist)
            dist_normalized = dist / max_dist
            
            low_mask = torch.exp(-(dist_normalized ** 2) / (2 * sigma ** 2))
            high_mask = 1.0 - low_mask
            
            self.register_buffer("low_mask", low_mask)
            self.register_buffer("high_mask", high_mask)
        
        def forward(self, feat):
            B, C, T, H, W = feat.shape
            
            feat_reshaped = feat.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            feat_resized = F.interpolate(feat_reshaped, size=self.freq_size, mode="bilinear", align_corners=False)
            
            feat_fft = torch.fft.fft2(feat_resized, norm="ortho")
            feat_fft_shifted = torch.fft.fftshift(feat_fft, dim=(-2, -1))
            
            low_mask = self.low_mask.view(1, 1, *self.freq_size)
            high_mask = self.high_mask.view(1, 1, *self.freq_size)
            
            feat_low_fft = feat_fft_shifted * low_mask
            feat_high_fft = feat_fft_shifted * high_mask
            
            feat_low = torch.fft.ifft2(torch.fft.ifftshift(feat_low_fft, dim=(-2, -1)), norm="ortho").real
            feat_high = torch.fft.ifft2(torch.fft.ifftshift(feat_high_fft, dim=(-2, -1)), norm="ortho").real
            
            feat_low = F.interpolate(feat_low, size=(H, W), mode="bilinear", align_corners=False)
            feat_high = F.interpolate(feat_high, size=(H, W), mode="bilinear", align_corners=False)
            
            feat_low = feat_low.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)
            feat_high = feat_high.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)
            
            return feat_low, feat_high
    
    class HandPoseFreqDecomp(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vae = FrameVAE(in_channels=3, latent_dim=4)
            
            vae_ckpt = '/data/data5/zhaoran/paper_code/exo/latent/outputs/vae/checkpoints/best_model.pt'
            if os.path.exists(vae_ckpt):
                checkpoint = torch.load(vae_ckpt, map_location='cpu')
                self.vae.load_state_dict(checkpoint['model_state_dict'])
            
            for param in self.vae.parameters():
                param.requires_grad = False
            
            self.spectral_decomp = SoftSpectralDecomposition(freq_size=(8, 8), feat_dim=4, sigma=0.5)
            self.encoder_low = LatentEncoder(in_channels=4, hidden_dim=256)
            self.encoder_high = LatentEncoder(in_channels=4, hidden_dim=256)
            
            with torch.no_grad():
                dummy = torch.randn(1, 4, 8, 16, 16)
                out = self.encoder_low(dummy)
                regressor_input_dim = out.shape[1] * out.shape[3] * out.shape[4] * 2
            
            self.regressor = RegressorHead(regressor_input_dim, hidden_dim=512, num_pose_params=51)
        
        def forward(self, x):
            B, C, T, H, W = x.shape
            
            with torch.no_grad():
                z = self.vae.encode(x)
            
            z_low, z_high = self.spectral_decomp(z)
            
            z_low_down = self.encoder_low(z_low)
            z_high_down = self.encoder_high(z_high)
            
            z_low_flat = z_low_down.permute(0, 2, 1, 3, 4).reshape(B * T, -1)
            z_high_flat = z_high_down.permute(0, 2, 1, 3, 4).reshape(B * T, -1)
            
            z_combined = torch.cat([z_low_flat, z_high_flat], dim=1)
            pose = self.regressor(z_combined)
            return pose
    
    # ========== Run Evaluation ==========
    results = {}
    
    results['baseline'] = eval_single_method('baseline', lambda: HandPoseBaseline(), val_loader, device, output_dir)
    results['freq_decomp'] = eval_single_method('freq_decomp', lambda: HandPoseFreqDecomp(), val_loader, device, output_dir)
    results['latent_only'] = eval_single_method('latent_only', lambda: HandPoseLatentOnly(), val_loader, device, output_dir)
    
    print(f"\n{'='*80}")
    print("FINAL COMPARISON")
    print("=" * 80)
    
    baseline_mse = results['baseline']
    
    for method in ['baseline', 'freq_decomp', 'latent_only']:
        mse = results[method]
        if method != 'baseline' and baseline_mse is not None:
            improvement = (baseline_mse - mse) / baseline_mse * 100
            print(f"  {method:<15}: {mse:.6f}  ({improvement:+.2f}% vs baseline)")
        else:
            print(f"  {method:<15}: {mse:.6f}")
    
    print(f"\nRankings by MSE:")
    sorted_methods = sorted([k for k in results if results[k] is not None], key=lambda m: results[m])
    for i, method in enumerate(sorted_methods):
        print(f"  #{i+1}: {method}")
    
    np.save(os.path.join(output_dir, 'final_results.npy'), results)
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_dir}/final_results.npy")
    print("=" * 80)


if __name__ == "__main__":
    main()
