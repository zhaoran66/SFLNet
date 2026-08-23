import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import torchvision.utils as vutils


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


class FASRTinyEvaluator:
    def __init__(self, model, diffusion, val_loader, config, num_samples=50):
        self.model = model
        self.diffusion = diffusion
        self.val_loader = val_loader
        self.config = config
        self.num_samples = num_samples
        self.device = torch.device(config.training.device)
        
        self.model.to(self.device)
        self.diffusion.to(self.device)
        
        self.model.eval()
        
        print(f"Running evaluation on {num_samples} samples...")
        print(f"Using FULL 1000 diffusion steps")
    
    @torch.no_grad()
    def evaluate(self):
        all_psnr_interp = []
        all_ssim_interp = []
        
        os.makedirs(os.path.join(self.config.output_dir, "eval_samples"), exist_ok=True)
        
        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Evaluating", total=self.num_samples)):
            if batch_idx >= self.num_samples:
                break
            
            exo_video = batch["exo_video"].to(self.device)
            ego_video = batch["ego_video"].to(self.device)
            exo_pose = batch["exo_pose"].to(self.device)
            ego_pose = batch["ego_pose"].to(self.device)
            
            with torch.no_grad():
                num_history = 4
                
                pose_cond = torch.cat([exo_pose[:, -1:], ego_pose[:, :1]], dim=1)
                
                B, C, T, H, W = exo_video.shape
                num_interp = 4
                
                gen_interp = self._sample_diffusion(
                    exo_video[:, :, -num_history:],
                    pose_cond,
                    num_gen_frames=num_interp,
                )
                
                gt_interp = torch.cat([
                    exo_video[:, :, -num_history:],
                    ego_video[:, :, :num_interp],
                ], dim=2)[:, :, -num_interp:]
                
                for i in range(B):
                    gen_np = (gen_interp[i].permute(1, 2, 3, 0).cpu().numpy() + 1) / 2
                    gt_np = (gt_interp[i].permute(1, 2, 3, 0).cpu().numpy() + 1) / 2
                    
                    gen_np = np.clip(gen_np, 0, 1)
                    gt_np = np.clip(gt_np, 0, 1)
                    
                    for t in range(gen_np.shape[0]):
                        psnr = compute_psnr(gt_np[t], gen_np[t], data_range=1.0)
                        ssim = compute_ssim(gt_np[t], gen_np[t], multichannel=True, data_range=1.0, channel_axis=-1)
                        
                        all_psnr_interp.append(psnr)
                        all_ssim_interp.append(ssim)
            
            if batch_idx < 5:
                self._save_visualization(batch_idx, exo_video, ego_video, gen_interp, num_history)
        
        results = {
            "PSNR": np.mean(all_psnr_interp),
            "SSIM": np.mean(all_ssim_interp),
        }
        
        print("\n" + "="*60)
        print("FASR-Tiny Evaluation Results (on interpolated frames):")
        print(f"  Samples evaluated: {len(all_psnr_interp)}")
        print(f"  Diffusion steps: 1000 (full)")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")
        print("="*60)
        
        return results
    
    @torch.no_grad()
    def _sample_diffusion(self, history_frames, pose_cond, num_gen_frames):
        B, C, T_hist, H, W = history_frames.shape
        
        feat_hist = self.model.extract_features(history_frames.view(B, C, T_hist, H, W))
        feat_dim = feat_hist.shape[1]
        
        x = torch.randn(B, feat_dim, T_hist + num_gen_frames, 32, 32, device=self.device)
        x[:, :, :T_hist] = feat_hist
        
        for t in reversed(range(self.diffusion.num_timesteps)):
            t_batch = torch.tensor([t] * B, device=self.device)
            
            sqrt_alpha_bar_t = self.diffusion.sqrt_alpha_bar[t]
            sqrt_one_minus_alpha_bar_t = self.diffusion.sqrt_one_minus_alpha_bar[t]
            
            x_noisy_history = sqrt_alpha_bar_t * feat_hist + sqrt_one_minus_alpha_bar_t * torch.randn_like(feat_hist)
            x[:, :, :T_hist] = x_noisy_history
            
            model_output = self.model.diffusion_model(x, t_batch, pose_cond)
            
            x[:, :, T_hist:] = self.diffusion.p_sample(
                model_output[:, :, T_hist:],
                x[:, :, T_hist:],
                t_batch,
            )
        
        pred_feat = x[:, :, -num_gen_frames:]
        pred_video = self.model.decoder(pred_feat)
        
        return pred_video
    
    def _save_visualization(self, batch_idx, exo_video, ego_video, gen_interp, num_history):
        B, C, T, H, W = exo_video.shape
        
        exo_last = exo_video[0, :, -1]
        ego_first = ego_video[0, :, 0]
        
        num_interp = gen_interp.shape[2]
        
        gen_interp_list = [gen_interp[0, :, i].unsqueeze(0) for i in range(num_interp)]
        row = torch.cat([exo_last.unsqueeze(0)] + gen_interp_list + [ego_first.unsqueeze(0)], dim=0)
        
        vutils.save_image(
            row,
            os.path.join(self.config.output_dir, "eval_samples", f"batch_{batch_idx}.png"),
            nrow=num_interp + 2,
            normalize=True,
            range=(-1, 1),
        )


def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    from models.fasr_model_tiny import FASRTinyModel, GaussianDiffusion
    
    config = checkpoint["config"]
    
    model = FASRTinyModel(config)
    diffusion = GaussianDiffusion(
        num_timesteps=config.model.num_timesteps,
        beta_schedule=config.model.beta_schedule,
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    return model, diffusion, config


if __name__ == "__main__":
    from data.dataset import create_dataloaders
    
    checkpoint_path = "./outputs_fasr_tiny_stable/checkpoints/checkpoint_30.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    print(f"Loading checkpoint: {checkpoint_path}")
    model, diffusion, config = load_checkpoint(checkpoint_path)
    
    config.training.device = "cuda"
    config.training.batch_size = 1
    
    _, val_loader = create_dataloaders(config)
    
    evaluator = FASRTinyEvaluator(model, diffusion, val_loader, config, num_samples=50)
    evaluator.evaluate()
