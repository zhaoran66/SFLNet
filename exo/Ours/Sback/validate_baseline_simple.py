"""
Simple Syn2Seq baseline validation - SAME metric pipeline as FASR
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from tqdm import tqdm
import torchvision.utils as vutils


def compute_psnr(img1, img2, data_range=1.0):
    """EXACT SAME as FASR validation"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(data_range / np.sqrt(mse))


def compute_ssim(img1, img2, multichannel=True, data_range=1.0, channel_axis=-1):
    """EXACT SAME as FASR validation - VERIFIED"""
    if multichannel:
        if channel_axis != -1:
            img1 = np.moveaxis(img1, channel_axis, -1)
            img2 = np.moveaxis(img2, channel_axis, -1)
    
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    H, W, C = img1.shape
    
    mu1 = np.mean(img1, axis=(0, 1))
    mu2 = np.mean(img2, axis=(0, 1))
    
    sigma1_sq = np.var(img1, axis=(0, 1))
    sigma2_sq = np.var(img2, axis=(0, 1))
    sigma12 = np.mean((img1 - mu1[None, None, :]) * (img2 - mu2[None, None, :]), axis=(0, 1))
    
    ssim_numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    ssim_denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim = np.mean(ssim_numerator / ssim_denominator)
    
    return ssim


def check_and_normalize(img, name="image"):
    """EXACT SAME as FASR validation"""
    orig_min, orig_max = img.min(), img.max()
    
    print(f"  [{name}] original range: [{orig_min:.3f}, {orig_max:.3f}], shape: {img.shape}")
    
    if orig_min < -0.5:
        img = np.clip(img, -1, 1)
        img = (img + 1) / 2
    elif orig_max > 1.5:
        img = np.clip(img, 0, 255) / 255.0
    
    img = np.clip(img, 0, 1)
    
    print(f"  [{name}] final range: [{img.min():.3f}, {img.max():.3f}]")
    
    return img


def main():
    print("=" * 60)
    print("Syn2Seq BASELINE - Direct from evaluate_quick.py")
    print("=" * 60)
    
    from data.dataset import create_dataloaders
    from models.dfot import DiffusionForcingTransformer, GaussianDiffusion
    from models.interpolator import VideoInterpolator
    
    checkpoint_path = "./outputs/checkpoints/checkpoint_95.pt"
    print(f"Loading: {checkpoint_path}")
    
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
    
    device = torch.device("cuda")
    model.to(device)
    diffusion.to(device)
    interpolator.to(device)
    model.eval()
    interpolator.eval()
    
    _, val_loader = create_dataloaders(config)
    
    all_psnr = []
    all_ssim = []
    os.makedirs("./metric_validation_baseline", exist_ok=True)
    
    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validating", total=5)):
        if batch_idx >= 5:
            break
        
        exo_video = batch["exo_video"].to(device)
        ego_video = batch["ego_video"].to(device)
        exo_pose = batch["exo_pose"].to(device)
        ego_pose = batch["ego_pose"].to(device)
        
        with torch.no_grad():
            full_sequence, interp_frames = interpolator(exo_video, ego_video)
            
            pose_cond = torch.cat([exo_pose[:, -1:], ego_pose[:, :1]], dim=1)
            
            B, C, T, H, W = exo_video.shape
            num_history = 4
            num_interp = 4
            
            feat_history = model.encoder(exo_video[:, :, -num_history:])
            
            x = torch.randn(B, config.model.hidden_size, num_history + num_interp, 32, 32, device=device)
            x[:, :, :num_history] = feat_history
            
            for t in tqdm(reversed(range(diffusion.num_timesteps)), desc="Sampling", leave=False):
                t_batch = torch.tensor([t] * B, device=device)
                
                sqrt_alpha_bar_t = diffusion.sqrt_alpha_bar[t]
                sqrt_one_minus_alpha_bar_t = diffusion.sqrt_one_minus_alpha_bar[t]
                
                x_noisy_history = sqrt_alpha_bar_t * feat_history + sqrt_one_minus_alpha_bar_t * torch.randn_like(feat_history)
                x[:, :, :num_history] = x_noisy_history
                
                model_output = model(x, t_batch, pose_cond)
                
                x[:, :, num_history:] = diffusion.p_sample(
                    model_output[:, :, num_history:],
                    x[:, :, num_history:],
                    t_batch,
                )
            
            pred_feat = x[:, :, -num_interp:]
            gen_interp = model.decoder(pred_feat)
            
            gt_interp = torch.cat([
                exo_video[:, :, -num_history:],
                ego_video[:, :, :num_interp],
            ], dim=2)[:, :, -num_interp:]
            
            for i in range(min(B, 2)):
                for t in range(min(num_interp, 2)):
                    gen_np = gen_interp[i, :, t].permute(1, 2, 0).cpu().numpy()
                    gt_np = gt_interp[i, :, t].permute(1, 2, 0).cpu().numpy()
                    
                    gen_norm = check_and_normalize(gen_np, "pred")
                    gt_norm = check_and_normalize(gt_np, "gt  ")
                    
                    psnr = compute_psnr(gen_norm, gt_norm)
                    ssim = compute_ssim(gen_norm, gt_norm, multichannel=True, channel_axis=-1)
                    
                    print(f"  PSNR: {psnr:.4f}, SSIM: {ssim:.4f}")
                    
                    all_psnr.append(psnr)
                    all_ssim.append(ssim)
                    
                    if batch_idx == 0:
                        gen_t = torch.from_numpy(gen_norm).permute(2, 0, 1)
                        gt_t = torch.from_numpy(gt_norm).permute(2, 0, 1)
                        diff_t = torch.abs(gen_t - gt_t)
                        
                        comparison = torch.cat([gt_t, gen_t, diff_t * 3], dim=2)
                        vutils.save_image(
                            comparison,
                            f"./metric_validation_baseline/batch{batch_idx}_sample{i}_frame{t}_psnr{psnr:.2f}_ssim{ssim:.4f}.png",
                            normalize=False
                        )
    
    print("\n" + "=" * 60)
    print("FINAL Syn2Seq BASELINE METRICS")
    print("=" * 60)
    print(f"PSNR: {np.mean(all_psnr):.4f} dB")
    print(f"SSIM: {np.mean(all_ssim):.4f}")
    print(f"Total samples: {len(all_psnr)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
