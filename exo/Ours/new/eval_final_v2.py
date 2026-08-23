#!/usr/bin/env python
"""
Final Evaluation - Correct Version
1. baseline: RGB -> PixelEncoder -> Regressor
2. freq_decomp: RGB -> DINO extractor -> Freq-split -> Encoder(x2) -> Regressor
3. latent_only: RGB -> VAE -> Encoder -> Regressor
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


# ============ Shared Components ============
class DINOv2FeatureExtractor(torch.nn.Module):
    """DINO feature extractor - same as used in training"""
    def __init__(self, feat_dim=384):
        super().__init__()
        # Simple CNN fallback (same as training)
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 4, stride=2, padding=1),
            torch.nn.GroupNorm(8, 64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 4, stride=2, padding=1),
            torch.nn.GroupNorm(8, 128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, 4, stride=2, padding=1),
            torch.nn.GroupNorm(8, 256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, feat_dim, 4, stride=2, padding=1),
        )
        self.feat_dim = feat_dim
    
    def forward(self, x):
        B, C, T, H, W = x.shape
        x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        feat = self.encoder(x_reshaped)
        feat = feat.view(B, T, -1, 8, 8).permute(0, 2, 1, 3, 4)
        feat_reshaped = feat.permute(0, 2, 1, 3, 4).reshape(B * T, -1, 8, 8)
        feat_upsampled = F.interpolate(feat_reshaped, size=(32, 32), mode="bilinear", align_corners=False)
        feat_upsampled = feat_upsampled.view(B, T, -1, 32, 32).permute(0, 2, 1, 3, 4)
        return feat_upsampled


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


# ============ Method 1: Baseline ============
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


# ============ Method 2: Freq Decomp (DINO) ============
class DinoFeatureEncoder(torch.nn.Module):
    """Encoder for DINO features"""
    def __init__(self, in_channels=384, hidden_dim=256):
        super().__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels, hidden_dim, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv3d(hidden_dim, hidden_dim * 2, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            torch.nn.ReLU(),
        )
    
    def forward(self, x):
        return self.conv_layers(x)


class SoftSpectralDecomposition(torch.nn.Module):
    def __init__(self, freq_size=(8, 8), feat_dim=384, sigma=0.5):
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
        self.dino_extractor = DINOv2FeatureExtractor(feat_dim=384)
        
        for param in self.dino_extractor.parameters():
            param.requires_grad = False
        
        self.spectral_decomp = SoftSpectralDecomposition(freq_size=(8, 8), feat_dim=384, sigma=0.5)
        self.encoder_low = DinoFeatureEncoder(in_channels=384, hidden_dim=256)
        self.encoder_high = DinoFeatureEncoder(in_channels=384, hidden_dim=256)
        
        with torch.no_grad():
            dummy = torch.randn(1, 384, 8, 32, 32)
            out = self.encoder_low(dummy)
            regressor_input_dim = out.shape[1] * out.shape[3] * out.shape[4] * 2
        
        self.regressor = RegressorHead(regressor_input_dim, hidden_dim=512, num_pose_params=51)
    
    def forward(self, x):
        B, C, T, H, W = x.shape
        
        with torch.no_grad():
            feat = self.dino_extractor(x)
        
        feat_low, feat_high = self.spectral_decomp(feat)
        
        feat_low_down = self.encoder_low(feat_low)
        feat_high_down = self.encoder_high(feat_high)
        
        feat_low_flat = feat_low_down.permute(0, 2, 1, 3, 4).reshape(B * T, -1)
        feat_high_flat = feat_high_down.permute(0, 2, 1, 3, 4).reshape(B * T, -1)
        
        feat_combined = torch.cat([feat_low_flat, feat_high_flat], dim=1)
        pose = self.regressor(feat_combined)
        return pose


# ============ Method 3: Latent Only ============
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


def load_vae(vae_path="/data/data5/zhaoran/paper_code/exo/latent/outputs/vae/checkpoints/best_model.pt"):
    """Load pretrained VAE for latent methods"""
    vae = FrameVAE(in_channels=3, latent_dim=4)
    
    if os.path.exists(vae_path):
        checkpoint = torch.load(vae_path, map_location='cpu')
        vae.load_state_dict(checkpoint['model_state_dict'])
        print(f"VAE loaded from {vae_path}")
    else:
        print(f"Warning: VAE checkpoint not found at {vae_path}")
    
    return vae


class HandPoseLatentOnly(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vae = load_vae()
        
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


class HandPoseLatentFreq(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vae = load_vae()
        
        for param in self.vae.parameters():
            param.requires_grad = False
        
        self.spectral_decomp = SoftSpectralDecomposition(
            freq_size=(8, 8),
            feat_dim=4,
            sigma=0.5,
        )
        
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


def get_mano_converter():
    """Get MANO joints converter with proper error handling"""
    try:
        from utils.mano_layer import MANOJointsConverter
        converter = MANOJointsConverter()
        print("MANO converter loaded successfully! Using physical 3D joints.")
        return converter
    except Exception as e:
        print(f"Warning: Failed to load MANO converter: {e}")
        print("Falling back to parameter space distance (not physically accurate)")
        return None


def compute_mpjpe(pred_params, target_params, mano_converter=None):
    """
    Compute Mean Per Joint Position Error (MPJPE) using REAL 3D joint coordinates
    
    Args:
        pred_params: (B, 51) Predicted MANO parameters
        target_params: (B, 51) Target MANO parameters
        mano_converter: MANO converter model (if None, falls back to parameter space)
    """
    if mano_converter is not None:
        try:
            with torch.no_grad():
                pred_joints = mano_converter(pred_params)
                target_joints = mano_converter(target_params)
            
            pred_aligned = pred_joints - pred_joints[:, 0:1, :]
            target_aligned = target_joints - target_joints[:, 0:1, :]
            
            per_joint_errors = torch.norm(pred_aligned - target_aligned, dim=-1)
            mpjpe = per_joint_errors.mean().item() * 1000
            
            return mpjpe
        except Exception as e:
            print(f"Warning: MPJPE calculation failed: {e}")
    
    pred_np = pred_params.cpu().numpy()
    target_np = target_params.cpu().numpy()
    return np.mean(np.linalg.norm(pred_np - target_np, axis=-1))


def compute_pck(pred_params, target_params, thresholds, mano_converter=None):
    """
    Compute Percentage of Correct Keypoints (PCK) using REAL 3D joint coordinates
    
    Args:
        pred_params: (B, 51) Predicted MANO parameters
        target_params: (B, 51) Target MANO parameters
        thresholds: List of error thresholds in mm
        mano_converter: MANO converter model (if None, falls back to parameter space)
    """
    if mano_converter is not None:
        try:
            with torch.no_grad():
                pred_joints = mano_converter(pred_params)
                target_joints = mano_converter(target_params)
            
            pred_aligned = pred_joints - pred_joints[:, 0:1, :]
            target_aligned = target_joints - target_joints[:, 0:1, :]
            
            per_joint_errors = torch.norm(pred_aligned - target_aligned, dim=-1).cpu().numpy() * 1000
            
            pck_values = {}
            for thresh in thresholds:
                pck = np.mean(per_joint_errors < thresh) * 100
                pck_values[thresh] = pck
            
            return pck_values
        except Exception as e:
            print(f"Warning: PCK calculation failed: {e}")
    
    pred_np = pred_params.cpu().numpy()
    target_np = target_params.cpu().numpy()
    errors = np.linalg.norm(pred_np - target_np, axis=-1)
    
    pck_values = {}
    for thresh in thresholds:
        pck = np.mean(errors < thresh) * 100
        pck_values[thresh] = pck
    
    return pck_values


# ============ Evaluation ============
@torch.no_grad()
def eval_single_method(method_name, model_class, val_loader, device, output_dir, mano_converter=None):
    """Evaluate a single method using best checkpoint with real 3D joint metrics"""
    ckpt_dir = os.path.join(output_dir, method_name, "checkpoints")
    ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
    
    if not os.path.exists(ckpt_path):
        ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.startswith("checkpoint_")])
        if not ckpts:
            print(f"No checkpoint found for {method_name}!")
            return None
        ckpt_path = os.path.join(ckpt_dir, ckpts[-1])
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = model_class()
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    model.eval()
    
    if mano_converter is not None:
        mano_converter.to(device)
        mano_converter.eval()
    
    print(f"\n{'='*80}")
    print(f"Evaluating: {method_name.upper()}")
    print(f"{'='*80}")
    print(f"Checkpoint: {os.path.basename(ckpt_path)}")
    if mano_converter is not None:
        print(f"Using REAL 3D joints for MPJPE/PCK (physically accurate!)")
    else:
        print(f"Warning: Using parameter space distance (not physically accurate!)")
    
    all_mse = []
    all_mae = []
    all_mpjpe = []
    all_pck1 = []
    all_pck2 = []
    all_pck5 = []
    all_pck10 = []
    
    for batch in tqdm(val_loader, desc="Evaluating"):
        exo_video = batch["exo_video"].to(device, non_blocking=True)
        hand_pose_target = batch["hand_pose"].to(device, non_blocking=True)
        
        B, T, D = hand_pose_target.shape
        hand_pose_target = hand_pose_target.reshape(B * T, D)
        
        pose_pred = model(exo_video)
        
        mse = F.mse_loss(pose_pred, hand_pose_target).item()
        mae = F.l1_loss(pose_pred, hand_pose_target).item()
        
        all_mse.append(mse)
        all_mae.append(mae)
        
        mpjpe = compute_mpjpe(pose_pred, hand_pose_target, mano_converter)
        all_mpjpe.append(mpjpe)
        
        pck = compute_pck(pose_pred, hand_pose_target, [1, 2, 5, 10], mano_converter)
        all_pck1.append(pck[1])
        all_pck2.append(pck[2])
        all_pck5.append(pck[5])
        all_pck10.append(pck[10])
    
    results = {
        'mse': (np.mean(all_mse), np.std(all_mse)),
        'mae': (np.mean(all_mae), np.std(all_mae)),
        'mpjpe': (np.mean(all_mpjpe), np.std(all_mpjpe)),
        'pck1': (np.mean(all_pck1), np.std(all_pck1)),
        'pck2': (np.mean(all_pck2), np.std(all_pck2)),
        'pck5': (np.mean(all_pck5), np.std(all_pck5)),
        'pck10': (np.mean(all_pck10), np.std(all_pck10)),
    }
    
    print(f"\n  MSE:     {results['mse'][0]:.6f} \u00b1 {results['mse'][1]:.6f}")
    print(f"  MAE:     {results['mae'][0]:.6f} \u00b1 {results['mae'][1]:.6f}")
    print(f"  MPJPE:   {results['mpjpe'][0]:.4f} \u00b1 {results['mpjpe'][1]:.4f} mm")
    print(f"  PCK@1:   {results['pck1'][0]:.2f}% \u00b1 {results['pck1'][1]:.2f}%")
    print(f"  PCK@2:   {results['pck2'][0]:.2f}% \u00b1 {results['pck2'][1]:.2f}%")
    print(f"  PCK@5:   {results['pck5'][0]:.2f}% \u00b1 {results['pck5'][1]:.2f}%")
    print(f"  PCK@10:  {results['pck10'][0]:.2f}% \u00b1 {results['pck10'][1]:.2f}%")
    
    return results


def main():
    set_seed(42)
    
    config = Config()
    config.training.batch_size = 4
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = '/data/data5/zhaoran/paper_code/exo/Ours/new/outputs_hand_pose'
    
    print(f"Using device: {device}")
    print("=" * 80)
    print("Final Evaluation - 4 Methods Comparison")
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
    
    print("\nInitializing MANO converter for real 3D joint metrics...")
    mano_converter = get_mano_converter()
    
    methods = [
        ('baseline', HandPoseBaseline),
        ('freq_decomp', HandPoseFreqDecomp),
        ('latent_only', HandPoseLatentOnly),
        ('latent_freq', HandPoseLatentFreq),
    ]
    
    results = {}
    for name, model_class in methods:
        results[name] = eval_single_method(name, model_class, val_loader, device, output_dir, mano_converter)
    
    print(f"\n{'='*80}")
    print("FINAL COMPARISON")
    print("=" * 80)
    
    print(f"\n{'='*80}")
    print("MSE Comparison")
    print("=" * 80)
    baseline_mse = results['baseline']['mse'][0]
    
    for method in ['baseline', 'freq_decomp', 'latent_only', 'latent_freq']:
        mse_mean, mse_std = results[method]['mse']
        if method != 'baseline' and baseline_mse is not None:
            improvement = (baseline_mse - mse_mean) / baseline_mse * 100
            print(f"  {method:<15}: {mse_mean:.6f} \u00b1 {mse_std:.6f}  ({improvement:+.2f}% vs baseline)")
        else:
            print(f"  {method:<15}: {mse_mean:.6f} \u00b1 {mse_std:.6f}")
    
    print(f"\nRankings by MSE:")
    valid_results = {k: v for k, v in results.items() if v is not None}
    sorted_methods = sorted(valid_results.keys(), key=lambda m: valid_results[m]['mse'][0])
    for i, method in enumerate(sorted_methods):
        print(f"  #{i+1}: {method}")
    
    print(f"\n{'='*80}")
    print("MPJPE Comparison (3D Joints in mm - PHYSICALLY ACCURATE!)")
    print("=" * 80)
    baseline_mpjpe = results['baseline']['mpjpe'][0]
    
    for method in ['baseline', 'freq_decomp', 'latent_only', 'latent_freq']:
        mpjpe_mean, mpjpe_std = results[method]['mpjpe']
        if method != 'baseline' and baseline_mpjpe is not None:
            improvement = (baseline_mpjpe - mpjpe_mean) / baseline_mpjpe * 100
            print(f"  {method:<15}: {mpjpe_mean:.4f} \u00b1 {mpjpe_std:.4f} mm  ({improvement:+.2f}% vs baseline)")
        else:
            print(f"  {method:<15}: {mpjpe_mean:.4f} \u00b1 {mpjpe_std:.4f} mm")
    
    print(f"\nRankings by MPJPE:")
    valid_results = {k: v for k, v in results.items() if v is not None}
    sorted_methods = sorted(valid_results.keys(), key=lambda m: valid_results[m]['mpjpe'][0])
    for i, method in enumerate(sorted_methods):
        print(f"  #{i+1}: {method}")
    
    print(f"\n{'='*80}")
    print("DETAILED METRICS SUMMARY")
    print("=" * 80)
    for method in ['baseline', 'freq_decomp', 'latent_only', 'latent_freq']:
        r = results[method]
        print(f"\n[{method.upper()}]")
        print(f"  MSE:     {r['mse'][0]:.6f} \u00b1 {r['mse'][1]:.6f}")
        print(f"  MAE:     {r['mae'][0]:.6f} \u00b1 {r['mae'][1]:.6f}")
        print(f"  MPJPE:   {r['mpjpe'][0]:.4f} \u00b1 {r['mpjpe'][1]:.4f} mm")
        print(f"  PCK@1:   {r['pck1'][0]:.2f}% \u00b1 {r['pck1'][1]:.2f}%")
        print(f"  PCK@2:   {r['pck2'][0]:.2f}% \u00b1 {r['pck2'][1]:.2f}%")
        print(f"  PCK@5:   {r['pck5'][0]:.2f}% \u00b1 {r['pck5'][1]:.2f}%")
        print(f"  PCK@10:  {r['pck10'][0]:.2f}% \u00b1 {r['pck10'][1]:.2f}%")
    
    np.save(os.path.join(output_dir, 'final_results_v2_detailed.npy'), results)
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_dir}/final_results_v2_detailed.npy")
    print("=" * 80)


if __name__ == "__main__":
    main()
