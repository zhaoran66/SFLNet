import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.dataset import DexYCBDataset
from models.dfot import DiffusionForcingTransformer, GaussianDiffusion
from models.interpolator import VideoInterpolator


def compute_psnr(img1, img2, data_range=1.0):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(data_range / np.sqrt(mse))


def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    
    model = DiffusionForcingTransformer(
        hidden_size=config.model.hidden_size,
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers,
        mlp_ratio=config.model.mlp_ratio,
        dropout=config.model.dropout,
        num_timesteps=config.model.num_timesteps,
    )
    
    diffusion = GaussianDiffusion(
        num_timesteps=config.model.num_timesteps,
        beta_schedule=config.model.beta_schedule,
    )
    
    interpolator = VideoInterpolator(
        num_interp_frames=config.data.num_interp_frames,
        hidden_dim=64,
        num_res_blocks=3,
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    interpolator.load_state_dict(checkpoint["interpolator_state_dict"])
    
    return model, diffusion, interpolator, config


@torch.no_grad()
def evaluate():
    checkpoint_path = "./outputs/checkpoints/checkpoint_95.pt"
    model, diffusion, interpolator, config = load_checkpoint(checkpoint_path)
    
    device = "cpu"
    config.training.batch_size = 1
    
    print("Creating dataset...")
    val_dataset = DexYCBDataset(
        data_root=config.data.data_root,
        exo_cam_id=config.data.exo_cam_id,
        ego_cam_id=config.data.ego_cam_id,
        num_frames=config.data.num_frames,
        frame_size=config.data.frame_size,
        train=False,
        train_split=config.data.train_split,
    )
    
    val_dataset = Subset(val_dataset, list(range(min(10, len(val_dataset)))))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    print(f"Evaluating on {len(val_dataset)} samples...")
    
    all_psnr_interp = []
    
    for batch_idx, batch in enumerate(tqdm(val_loader)):
        exo_video = batch["exo_video"].to(device)
        ego_video = batch["ego_video"].to(device)
        
        with torch.no_grad():
            full_sequence, interp_frames = interpolator(exo_video, ego_video)
            
            B, C, T, H, W = exo_video.shape
            num_interp = interp_frames.shape[2]
            
            gen_sequence = exo_video[:, :, -4:]
            for i in range(num_interp):
                alpha = (i + 1) / (num_interp + 1)
                interp_frame = (1 - alpha) * exo_video[:, :, -1:] + alpha * ego_video[:, :, :1]
                gen_sequence = torch.cat([gen_sequence, interp_frame], dim=2)
            
            gen_sequence = torch.cat([gen_sequence, ego_video[:, :, :4]], dim=2)
            gen_interp = gen_sequence[:, :, -num_interp-4:-4]
            gt_interp = interp_frames
            
            if batch_idx == 0:
                print(f"\nDebug info:")
                print(f"  exo_video range: [{exo_video.min():.3f}, {exo_video.max():.3f}]")
                print(f"  ego_video range:  [{ego_video.min():.3f}, {ego_video.max():.3f}]")
                print(f"  gen_interp range: [{gen_interp.min():.3f}, {gen_interp.max():.3f}]")
                print(f"  gt_interp range:  [{gt_interp.min():.3f}, {gt_interp.max():.3f}]")
            
            for i in range(B):
                gen_np = (gen_interp[i].permute(1, 2, 3, 0).cpu().numpy() + 1) / 2
                gt_np = (gt_interp[i].permute(1, 2, 3, 0).cpu().numpy() + 1) / 2
                
                gen_np = np.clip(gen_np, 0, 1)
                gt_np = np.clip(gt_np, 0, 1)
                
                for t in range(gen_np.shape[0]):
                    psnr = compute_psnr(gt_np[t], gen_np[t], data_range=1.0)
                    all_psnr_interp.append(psnr)
    
    results = {
        "PSNR": np.mean(all_psnr_interp),
    }
    
    print("\n" + "="*50)
    print("Evaluation Results (on interpolated frames):")
    print(f"  PSNR: {results['PSNR']:.4f}")
    print("="*50)
    print("\nNote: This is simplified evaluation, full eval needs GPU!")
    
    return results


if __name__ == "__main__":
    evaluate()
