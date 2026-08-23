"""
Evaluate Latent-only Ablation Model (NO Frequency Routing)
EXACT SAME metric pipeline as FASR for fair comparison
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from tqdm import tqdm
import torchvision.utils as vutils

from configs.config_tiny import Config
from data.dataset import create_dataloaders
from models.fasr_model_ablation import LatentDiffusionTransformer


def compute_psnr(img1, img2, data_range=1.0):
    """EXACT SAME as FASR"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(data_range / np.sqrt(mse))


def compute_ssim(img1, img2, multichannel=True, data_range=1.0, channel_axis=-1):
    """EXACT SAME as FASR"""
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


def compute_temporal_consistency(video_frames):
    """EXACT SAME as FASR"""
    T = len(video_frames)
    diffs = []
    
    for i in range(T - 1):
        frame_diff = np.mean(np.abs(video_frames[i] - video_frames[i+1]))
        diffs.append(frame_diff)
    
    return np.mean(diffs)


def compute_gradient_magnitude(img):
    """EXACT SAME as FASR"""
    gx = img[:, :, 1:] - img[:, :, :-1]
    gy = img[:, 1:, :] - img[:, :-1, :]
    
    gx = np.pad(gx, ((0,0), (0,0), (0,1)), mode='edge')
    gy = np.pad(gy, ((0,0), (0,1), (0,0)), mode='edge')
    
    grad_mag = np.sqrt(gx**2 + gy**2)
    return np.mean(grad_mag)


class GaussianDiffusion:
    def __init__(self, num_timesteps=1000, beta_schedule="linear"):
        self.num_timesteps = num_timesteps
        
        if beta_schedule == "linear":
            self.beta = torch.linspace(0.0001, 0.02, num_timesteps)
        
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)
        self.sqrt_recip_alpha = torch.sqrt(1.0 / self.alpha)
        self.posterior_variance = self.beta * (1.0 - torch.roll(self.alpha_bar, 1)) / (1.0 - self.alpha_bar)
        self.posterior_variance[0] = self.beta[0]
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = self.beta * torch.sqrt(torch.roll(self.alpha_bar, 1)) / (1.0 - self.alpha_bar)
        self.posterior_mean_coef2 = (1.0 - torch.roll(self.alpha_bar, 1)) * torch.sqrt(self.alpha) / (1.0 - self.alpha_bar)
        self.posterior_mean_coef1[0] = 0.0
    
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1, 1)
        
        return sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise
    
    def p_sample(self, model_output, x, t):
        B = x.shape[0]
        
        sqrt_recip_alpha_t = self.sqrt_recip_alpha[t].view(-1, 1, 1, 1, 1)
        beta_t = self.beta[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1, 1)
        
        pred_mean = sqrt_recip_alpha_t * (x - beta_t / sqrt_one_minus_alpha_bar_t * model_output)
        
        if t[0] > 0:
            noise = torch.randn_like(x)
            posterior_log_variance_t = self.posterior_log_variance_clipped[t].view(-1, 1, 1, 1, 1)
            return pred_mean + torch.exp(0.5 * posterior_log_variance_t) * noise
        else:
            return pred_mean


def main():
    print("=" * 70)
    print("EVALUATE: Latent-only Ablation (NO Frequency Routing)")
    print("=" * 70)
    print("EXACT SAME metric pipeline as FASR for fair comparison!")
    print("=" * 70)
    
    config = Config()
    config.output_dir = "./outputs_ablation_latent_only"
    
    checkpoint_path = os.path.join(config.output_dir, "checkpoints", "best_model.pt")
    print(f"\nLoading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    model = LatentDiffusionTransformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    diffusion = GaussianDiffusion(
        num_timesteps=config.model.num_timesteps,
        beta_schedule=config.model.beta_schedule,
    )
    
    device = torch.device("cuda")
    model.to(device)
    model.eval()
    
    _, val_loader = create_dataloaders(config)
    
    num_eval_samples = 20
    os.makedirs(os.path.join(config.output_dir, "eval_samples"), exist_ok=True)
    
    all_psnr = []
    all_ssim = []
    all_tc_gen = []
    all_tc_gt = []
    all_sharp_gen = []
    all_sharp_gt = []
    
    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Evaluating", total=num_eval_samples)):
        if batch_idx >= num_eval_samples:
            break
        
        exo_video = batch["exo_video"].to(device)
        ego_video = batch["ego_video"].to(device)
        exo_pose = batch["exo_pose"].to(device)
        ego_pose = batch["ego_pose"].to(device)
        
        num_history = 4
        num_interp = 4
        
        with torch.no_grad():
            exo_feat = model.extract_features(exo_video)
            ego_feat = model.extract_features(ego_video)
            
            feat_history = exo_feat[:, :, -num_history:]
            feat_dim = feat_history.shape[1]
            
            B = feat_history.shape[0]
            pose_cond = torch.cat([exo_pose[:, -1:], ego_pose[:, :1]], dim=1)
            
            Hz, Wz = 32, 32
            x = torch.randn(B, feat_dim, num_history + num_interp, Hz, Wz, device=device)
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
            
            target_video = torch.cat([exo_video[:, :, -num_history:], ego_video[:, :, :num_interp]], dim=2)
            gt_interp = target_video[:, :, -num_interp:]
            
            if batch_idx < 5:
                for i in range(min(num_interp, 2)):
                    gen_frame = (gen_interp[0, :, i:i+1, :, :].squeeze(2) + 1) / 2
                    gt_frame = (gt_interp[0, :, i:i+1, :, :].squeeze(2) + 1) / 2
                    
                    diff = torch.abs(gen_frame - gt_frame)
                    comparison = torch.cat([gt_frame, gen_frame, diff * 3], dim=2)
                    
                    vutils.save_image(
                        comparison,
                        os.path.join(config.output_dir, "eval_samples", f"batch{batch_idx}_frame{i}.png"),
                        normalize=False,
                    )
            
            gen_frames = []
            gt_frames = []
            
            for i in range(num_interp):
                gen_np = (gen_interp[0, :, i, :, :].cpu().numpy() + 1) / 2
                gt_np = (gt_interp[0, :, i, :, :].cpu().numpy() + 1) / 2
                
                gen_np = np.clip(gen_np, 0, 1)
                gt_np = np.clip(gt_np, 0, 1)
                
                gen_frames.append(gen_np)
                gt_frames.append(gt_np)
                
                psnr = compute_psnr(gen_np, gt_np)
                ssim = compute_ssim(gen_np, gt_np)
                
                all_psnr.append(psnr)
                all_ssim.append(ssim)
            
            tc_gen = compute_temporal_consistency(gen_frames)
            tc_gt = compute_temporal_consistency(gt_frames)
            
            sharp_gen = np.mean([compute_gradient_magnitude(f) for f in gen_frames])
            sharp_gt = np.mean([compute_gradient_magnitude(f) for f in gt_frames])
            
            all_tc_gen.append(tc_gen)
            all_tc_gt.append(tc_gt)
            all_sharp_gen.append(sharp_gen)
            all_sharp_gt.append(sharp_gt)
    
    print(f"\n{'='*70}")
    print("LATENT-ONLY ABLATION RESULTS")
    print("=" * 70)
    
    avg_psnr = np.mean(all_psnr)
    avg_ssim = np.mean(all_ssim)
    avg_tc_gen = np.mean(all_tc_gen)
    avg_tc_gt = np.mean(all_tc_gt)
    avg_sharp_gen = np.mean(all_sharp_gen)
    avg_sharp_gt = np.mean(all_sharp_gt)
    
    print(f"\nPerceptual Metrics:")
    print(f"  PSNR: {avg_psnr:.2f} dB")
    print(f"  SSIM: {avg_ssim:.4f}")
    
    print(f"\nTemporal Consistency (lower = more stable):")
    print(f"  Generated: {avg_tc_gen:.6f}")
    print(f"  Ground Truth: {avg_tc_gt:.6f}")
    
    print(f"\nSharpness (gradient magnitude, higher = sharper):")
    print(f"  Generated: {avg_sharp_gen:.6f}")
    print(f"  Ground Truth: {avg_sharp_gt:.6f}")
    
    print(f"\n{'='*70}")
    print("FULL ABLATION COMPARISON")
    print("=" * 70)
    print(f"\n{'Method':<30} {'PSNR':<10} {'SSIM':<10}")
    print(f"{'-'*50}")
    print(f"{'Syn2Seq (pixel space)':<30} {17.04:<10.2f} {0.0402:<10.4f}")
    print(f"{'Latent-only (ablation)':<30} {avg_psnr:<10.2f} {avg_ssim:<10.4f}")
    print(f"{'FASR (full method)':<30} {17.52:<10.2f} {0.2757:<10.4f}")
    print(f"{'-'*50}")
    print(f"\nLatent contribution: +{avg_psnr - 17.04:.2f} dB PSNR, +{avg_ssim - 0.0402:.4f} SSIM")
    print(f"Frequency Routing contribution: +{17.52 - avg_psnr:.2f} dB PSNR, +{0.2757 - avg_ssim:.4f} SSIM")
    print(f"Total gain: +{17.52 - 17.04:.2f} dB PSNR, +{0.2757 - 0.0402:.4f} SSIM")
    print(f"\n{'='*70}")
    print(f"Visualizations saved to: {config.output_dir}/eval_samples/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
