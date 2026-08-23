import os
import sys
import glob
import torch
import numpy as np
from tqdm import tqdm
import torchvision.utils as vutils

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import create_dataloaders
from models.main import Syn2SeqWithFrequency, GaussianDiffusion


def compute_psnr(img1, img2, data_range=1.0):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(data_range / np.sqrt(mse))


def compute_ssim(img1, img2, multichannel=True, data_range=1.0, channel_axis=-1):
    if multichannel:
        img1 = np.moveaxis(img1, channel_axis, 0)
        img2 = np.moveaxis(img2, channel_axis, 0)
    
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    mu1 = np.mean(img1, axis=(-2, -1))
    mu2 = np.mean(img2, axis=(-2, -1))
    
    sigma1_sq = np.var(img1, axis=(-2, -1))
    sigma2_sq = np.var(img2, axis=(-2, -1))
    sigma12 = np.mean((img1 - mu1[..., None, None]) * (img2 - mu2[..., None, None]), axis=(-2, -1))
    
    ssim_numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    ssim_denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim = np.mean(ssim_numerator / ssim_denominator)
    
    return ssim


def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    
    model = Syn2SeqWithFrequency(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    diffusion = GaussianDiffusion(
        num_timesteps=config.model.num_timesteps,
        beta_schedule=config.model.beta_schedule,
    )
    
    return model, diffusion, config


@torch.no_grad()
def evaluate(checkpoint_override=None):
    checkpoint_dirs = ["./outputs", "./outputs_ours"]
    checkpoint_path = None
    
    if checkpoint_override and os.path.exists(checkpoint_override):
        checkpoint_path = checkpoint_override
        print(f"Using checkpoint: {checkpoint_path}")
    else:
        for dir_name in checkpoint_dirs:
            checkpoints = sorted(glob.glob(f"{dir_name}/checkpoints/checkpoint_*.pt"))
            if checkpoints:
                checkpoint_path = checkpoints[-1]
                print(f"Found checkpoint: {checkpoint_path}")
                break
    
    if checkpoint_path is None:
        print("No checkpoints found!")
        sys.exit(1)
    
    model, diffusion, config = load_checkpoint(checkpoint_path)
    
    config.output_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    
    print("Creating dataloaders...")
    _, val_loader = create_dataloaders(config)
    
    device = torch.device(config.training.device)
    model.to(device)
    diffusion.to(device)
    
    model.eval()
    
    all_psnr = []
    all_ssim = []
    
    os.makedirs(os.path.join(config.output_dir, "eval_samples"), exist_ok=True)
    
    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
        exo_video = batch["exo_video"].to(device)
        ego_video = batch["ego_video"].to(device)
        exo_pose = batch["exo_pose"].to(device)
        ego_pose = batch["ego_pose"].to(device)
        
        with torch.no_grad():
            gen_sequence, _ = model.sample(
                exo_video[:, :, -4:],
                exo_pose, ego_pose, diffusion,
            )
            
            num_interp = model.config.data.num_interp_frames
            gen_interp = gen_sequence[:, :, -num_interp-4:-4]
            
            B, C, T, H, W = exo_video.shape
            num_interp = model.config.data.num_interp_frames
            
            exo_last = exo_video[:, :, -1:]
            ego_first = ego_video[:, :, :1]
            
            alpha = torch.linspace(0, 1, num_interp + 2, device=device)[1:-1]
            alpha = alpha.view(1, 1, -1, 1, 1)
            gt_interp = (1 - alpha) * exo_last + alpha * ego_first
            
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
                    ssim = compute_ssim(gt_np[t], gen_np[t], multichannel=True, data_range=1.0, channel_axis=-1)
                    
                    all_psnr.append(psnr)
                    all_ssim.append(ssim)
        
        if batch_idx < 5:
            B, C, T, H, W = exo_video.shape
            num_interp = model.config.data.num_interp_frames
            
            exo_last = exo_video[0, :, -1]
            ego_first = ego_video[0, :, 0]
            
            interp_list = [gt_interp[0, :, i].unsqueeze(0) for i in range(num_interp)]
            row1 = torch.cat([exo_last.unsqueeze(0)] + interp_list + [ego_first.unsqueeze(0)], dim=0)
            
            gen_interp_list = [gen_interp[0, :, i].unsqueeze(0) for i in range(num_interp)]
            row2 = torch.cat([exo_last.unsqueeze(0)] + gen_interp_list + [ego_first.unsqueeze(0)], dim=0)
            
            grid = torch.cat([row1, row2], dim=0)
            
            vutils.save_image(
                grid,
                os.path.join(config.output_dir, "eval_samples", f"batch_{batch_idx}.png"),
                nrow=num_interp + 2,
                normalize=True,
                range=(-1, 1),
            )
    
    print("\n" + "="*60)
    print("Evaluation Results (Ours - Frequency Decomposition):")
    print(f"  PSNR: {np.mean(all_psnr):.4f}")
    print(f"  SSIM: {np.mean(all_ssim):.4f}")
    print("="*60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    evaluate(args.checkpoint)
