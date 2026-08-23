import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import glob
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


class FastEvaluator:
    def __init__(self, model, diffusion, interpolator, val_loader, config, num_samples=50):
        self.model = model
        self.diffusion = diffusion
        self.interpolator = interpolator
        self.val_loader = val_loader
        self.config = config
        self.num_samples = num_samples
        self.device = torch.device(config.training.device)
        
        self.model.to(self.device)
        self.diffusion.to(self.device)
        self.interpolator.to(self.device)
        
        self.model.eval()
        self.interpolator.eval()
        
        print(f"Running fast evaluation on {num_samples} samples...")
        print(f"Using 500 diffusion steps (instead of 1000)")
    
    @torch.no_grad()
    def evaluate(self):
        all_psnr_interp = []
        all_ssim_interp = []
        
        os.makedirs(os.path.join(self.config.output_dir, "eval_samples_fast"), exist_ok=True)
        
        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Evaluating", total=self.num_samples)):
            if batch_idx >= self.num_samples:
                break
            
            exo_video = batch["exo_video"].to(self.device)
            ego_video = batch["ego_video"].to(self.device)
            exo_pose = batch["exo_pose"].to(self.device)
            ego_pose = batch["ego_pose"].to(self.device)
            
            with torch.no_grad():
                full_sequence, interp_frames = self.interpolator(exo_video, ego_video)
                
                pose_cond = torch.cat([exo_pose[:, -1:], ego_pose[:, :1]], dim=1)
                
                B, C, T, H, W = exo_video.shape
                num_interp = interp_frames.shape[2]
                
                gen_sequence = self._sample_diffusion_fast(
                    exo_video[:, :, -4:],
                    pose_cond,
                    num_gen_frames=num_interp,
                )
                
                gen_interp = gen_sequence[:, :, -num_interp:]
                gt_interp = interp_frames
                
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
            
            if batch_idx < 3:
                self._save_visualization(batch_idx, exo_video, interp_frames, ego_video, gen_sequence)
        
        results = {
            "PSNR": np.mean(all_psnr_interp),
            "SSIM": np.mean(all_ssim_interp),
        }
        
        print("\n" + "="*60)
        print("Fast Evaluation Results (on interpolated frames):")
        print(f"  Samples evaluated: {len(all_psnr_interp)}")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")
        print("="*60)
        
        return results
    
    @torch.no_grad()
    def _sample_diffusion_fast(self, history_frames, pose_cond, num_gen_frames, num_steps=500):
        B, C, T_hist, H, W = history_frames.shape
        
        x = torch.randn(B, C, T_hist + num_gen_frames, H, W, device=self.device)
        x[:, :, :T_hist] = history_frames
        
        step_size = 1000 // num_steps
        timesteps = list(reversed(range(0, 1000, step_size)))
        
        for t in timesteps:
            t_batch = torch.tensor([t] * B, device=self.device)
            
            sqrt_alpha_bar_t = self.diffusion.sqrt_alpha_bar[t]
            sqrt_one_minus_alpha_bar_t = self.diffusion.sqrt_one_minus_alpha_bar[t]
            
            x_noisy_history = sqrt_alpha_bar_t * history_frames + sqrt_one_minus_alpha_bar_t * torch.randn_like(history_frames)
            x[:, :, :T_hist] = x_noisy_history
            
            model_output = self.model(x, t_batch, pose_cond)
            
            x[:, :, T_hist:] = self.diffusion.p_sample(
                model_output[:, :, T_hist:],
                x[:, :, T_hist:],
                t_batch,
            )
        
        return x
    
    def _save_visualization(self, batch_idx, exo_video, interp_frames, ego_video, gen_sequence):
        B, C, T, H, W = exo_video.shape
        
        exo_last = exo_video[0, :, -1]
        ego_first = ego_video[0, :, 0]
        
        num_interp = interp_frames.shape[2]
        num_history = 4
        
        interp_list = [interp_frames[0, :, i].unsqueeze(0) for i in range(num_interp)]
        row1 = torch.cat([exo_last.unsqueeze(0)] + interp_list + [ego_first.unsqueeze(0)], dim=0)
        
        gen_interp_list = [gen_sequence[0, :, num_history+i].unsqueeze(0) for i in range(num_interp)]
        row2 = torch.cat([exo_last.unsqueeze(0)] + gen_interp_list + [ego_first.unsqueeze(0)], dim=0)
        
        grid = torch.cat([row1, row2], dim=0)
        
        vutils.save_image(
            grid,
            os.path.join(self.config.output_dir, "eval_samples_fast", f"batch_{batch_idx}.png"),
            nrow=num_interp + 2,
            normalize=True,
            range=(-1, 1),
        )


def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    from models.dfot import DiffusionForcingTransformer, GaussianDiffusion
    from models.interpolator import VideoInterpolator
    
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


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    from data.dataset import create_dataloaders
    
    checkpoint_path = "./outputs/checkpoints/checkpoint_95.pt"
    
    if not os.path.exists(checkpoint_path):
        checkpoints = sorted(glob.glob("./outputs/checkpoints/checkpoint_*.pt"))
        if checkpoints:
            checkpoint_path = checkpoints[-1]
            print(f"Using latest checkpoint: {checkpoint_path}")
        else:
            print(f"No checkpoints found!")
            sys.exit(1)
    
    model, diffusion, interpolator, config = load_checkpoint(checkpoint_path)
    
    config.training.device = "cuda"
    config.training.batch_size = 1
    
    _, val_loader = create_dataloaders(config)
    
    evaluator = FastEvaluator(model, diffusion, interpolator, val_loader, config, num_samples=50)
    evaluator.evaluate()
