"""
Sanity Check for Syn2Seq Baseline
- For FAIR comparison with FASR
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
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(data_range / np.sqrt(mse))


def compute_ssim(img1, img2, multichannel=True, data_range=1.0):
    if multichannel:
        img1 = np.moveaxis(img1, 0, -1)
        img2 = np.moveaxis(img2, 0, -1)
    
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


def compute_temporal_consistency(video_frames):
    T = len(video_frames)
    diffs = []
    
    for i in range(T - 1):
        frame_diff = np.mean(np.abs(video_frames[i] - video_frames[i+1]))
        diffs.append(frame_diff)
    
    return np.mean(diffs)


def compute_gradient_magnitude(img):
    gx = img[:, :, 1:] - img[:, :, :-1]
    gy = img[:, 1:, :] - img[:, :-1, :]
    
    gx = np.pad(gx, ((0,0), (0,0), (0,1)), mode='edge')
    gy = np.pad(gy, ((0,0), (0,1), (0,0)), mode='edge')
    
    grad_mag = np.sqrt(gx**2 + gy**2)
    return np.mean(grad_mag)


def main():
    print("=" * 70)
    print("SYN2SEQ BASELINE SANITY CHECK")
    print("=" * 70)
    
    from models.dfot import DiffusionForcingTransformer, GaussianDiffusion
    from models.interpolator import VideoInterpolator
    from data.dataset import create_dataloaders
    
    checkpoint_path = "./outputs/checkpoints/checkpoint_95.pt"
    print(f"\nLoading: {checkpoint_path}")
    
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
    
    num_samples = 5
    os.makedirs("./sanity_check_baseline", exist_ok=True)
    
    results = []
    
    for sample_idx in range(num_samples):
        print(f"\n{'='*70}")
        print(f"Processing sample {sample_idx+1}/{num_samples}")
        print(f"{'='*70}")
        
        batch = next(iter(val_loader))
        exo_video = batch["exo_video"].to(device)
        ego_video = batch["ego_video"].to(device)
        exo_pose = batch["exo_pose"].to(device)
        ego_pose = batch["ego_pose"].to(device)
        
        num_history = 4
        num_interp = 4
        
        with torch.no_grad():
            full_sequence, interp_frames = interpolator(exo_video, ego_video)
            pose_cond = torch.cat([exo_pose[:, -1:], ego_pose[:, :1]], dim=1)
            
            feat_history = model.encoder(exo_video[:, :, -num_history:])
            feat_dim = feat_history.shape[1]
            
            print("Sampling Syn2Seq...")
            x = torch.randn(1, feat_dim, num_history + num_interp, 32, 32, device=device)
            x[:, :, :num_history] = feat_history[:1]
            
            for t in tqdm(reversed(range(diffusion.num_timesteps)), desc="Diffusion", leave=False):
                t_batch = torch.tensor([t], device=device)
                
                sqrt_alpha_bar_t = diffusion.sqrt_alpha_bar[t]
                sqrt_one_minus_alpha_bar_t = diffusion.sqrt_one_minus_alpha_bar[t]
                
                x_noisy_history = sqrt_alpha_bar_t * feat_history[:1] + sqrt_one_minus_alpha_bar_t * torch.randn_like(feat_history[:1])
                x[:, :, :num_history] = x_noisy_history
                
                model_output = model(x, t_batch, pose_cond[:1])
                
                x[:, :, num_history:] = diffusion.p_sample(
                    model_output[:, :, num_history:],
                    x[:, :, num_history:],
                    t_batch,
                )
            
            pred_feat = x[:, :, -num_interp:]
            gen_interp = model.decoder(pred_feat)
            
            gt_interp = torch.cat([
                exo_video[:1, :, -num_history:],
                ego_video[:1, :, :num_interp],
            ], dim=2)[:, :, -num_interp:]
            
            print("Generating visualizations...")
            
            gen_frames = []
            gt_frames = []
            psnrs = []
            ssims = []
            
            for i in range(num_interp):
                gen_frame = gen_interp[0, :, i, :, :]
                gt_frame = gt_interp[0, :, i, :, :]
                
                gen_np = (gen_frame.cpu().numpy() + 1) / 2
                gt_np = (gt_frame.cpu().numpy() + 1) / 2
                
                gen_np = np.clip(gen_np, 0, 1)
                gt_np = np.clip(gt_np, 0, 1)
                
                gen_frames.append(gen_np)
                gt_frames.append(gt_np)
                
                psnr = compute_psnr(gen_np, gt_np)
                ssim = compute_ssim(gen_np, gt_np)
                psnrs.append(psnr)
                ssims.append(ssim)
                
                diff = np.abs(gen_np - gt_np)
                
                gen_t = torch.from_numpy(gen_np)
                gt_t = torch.from_numpy(gt_np)
                diff_t = torch.from_numpy(diff)
                
                comparison = torch.cat([gt_t, gen_t, diff_t * 3], dim=2)
                vutils.save_image(
                    comparison,
                    f"./sanity_check_baseline/sample{sample_idx}_frame{i}_psnr{psnr:.2f}_ssim{ssim:.4f}.png",
                    normalize=False,
                )
            
            tc_gen = compute_temporal_consistency(gen_frames)
            tc_gt = compute_temporal_consistency(gt_frames)
            
            sharpness_gen = np.mean([compute_gradient_magnitude(f) for f in gen_frames])
            sharpness_gt = np.mean([compute_gradient_magnitude(f) for f in gt_frames])
            
            print(f"\nResults for sample {sample_idx}:")
            print(f"  PSNR: {np.mean(psnrs):.2f} dB")
            print(f"  SSIM: {np.mean(ssims):.4f}")
            print(f"  Temporal Consistency (Gen): {tc_gen:.6f}")
            print(f"  Temporal Consistency (GT):  {tc_gt:.6f}")
            print(f"  Sharpness (Gen): {sharpness_gen:.6f}")
            print(f"  Sharpness (GT):  {sharpness_gt:.6f}")
            
            results.append({
                'psnr': np.mean(psnrs),
                'ssim': np.mean(ssims),
                'tc_gen': tc_gen,
                'tc_gt': tc_gt,
                'sharpness_gen': sharpness_gen,
                'sharpness_gt': sharpness_gt,
            })
            
            print(f"\nVisualization saved: ./sanity_check_baseline/sample{sample_idx}_frame*.png")
    
    print(f"\n{'='*70}")
    print("SYN2SEQ BASELINE AGGREGATED RESULTS")
    print(f"{'='*70}")
    
    avg_psnr = np.mean([r['psnr'] for r in results])
    avg_ssim = np.mean([r['ssim'] for r in results])
    avg_tc_gen = np.mean([r['tc_gen'] for r in results])
    avg_tc_gt = np.mean([r['tc_gt'] for r in results])
    avg_sharp_gen = np.mean([r['sharpness_gen'] for r in results])
    avg_sharp_gt = np.mean([r['sharpness_gt'] for r in results])
    
    print(f"\nPerceptual Metrics:")
    print(f"  PSNR: {avg_psnr:.2f} dB")
    print(f"  SSIM: {avg_ssim:.4f}")
    
    print(f"\nTemporal Consistency (lower = more stable):")
    print(f"  Syn2Seq Generated: {avg_tc_gen:.6f}")
    print(f"  Ground Truth:      {avg_tc_gt:.6f}")
    print(f"  Gap to GT:         {avg_tc_gen - avg_tc_gt:.6f}")
    
    print(f"\nSharpness (gradient magnitude, higher = sharper):")
    print(f"  Syn2Seq Generated: {avg_sharp_gen:.6f}")
    print(f"  Ground Truth:      {avg_sharp_gt:.6f}")
    print(f"  Sharpness Ratio:   {avg_sharp_gen/avg_sharp_gt:.2%}")
    
    print(f"\n{'='*70}")
    print("NOW COMPARE WITH FASR!")
    print(f"{'='*70}")
    print("Key comparison points:")
    print("  1. Is FASR sharper or more blurry than Syn2Seq?")
    print("  2. Is FASR more temporally stable?")
    print("  3. Visual inspection: look at hand edges in both!")
    print(f"\nFASR results saved to: ../new/sanity_check/")
    print(f"Syn2Seq results saved to: ./sanity_check_baseline/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
