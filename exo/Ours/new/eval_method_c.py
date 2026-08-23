"""
Method C Evaluation Script
Compute PSNR/SSIM and compare with other methods
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

import sys
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.utils as vutils

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config_tiny import Config
from data.dataset import create_dataloaders
from models.method_c_vae import FrameVAE
from models.method_c_model import LatentDiffusionTransformerWithBG


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

    mu1 = np.mean(img1, axis=(0, 1))
    mu2 = np.mean(img2, axis=(0, 1))

    sigma1_sq = np.var(img1, axis=(0, 1))
    sigma2_sq = np.var(img2, axis=(0, 1))
    sigma12 = np.mean((img1 - mu1[None, None, :]) * (img2 - mu2[None, None, :]), axis=(0, 1))

    ssim_numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    ssim_denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)

    return np.mean(ssim_numerator / ssim_denominator)


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

        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1, 1).to(x_start.device)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1, 1).to(x_start.device)

        return sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise

    def p_sample(self, model_output, x, t):
        B = x.shape[0]

        sqrt_recip_alpha_t = self.sqrt_recip_alpha[t].view(-1, 1, 1, 1, 1).to(x.device)
        beta_t = self.beta[t].view(-1, 1, 1, 1, 1).to(x.device)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1, 1).to(x.device)

        pred_mean = sqrt_recip_alpha_t * (x - beta_t / sqrt_one_minus_alpha_bar_t * model_output)

        if t[0] > 0:
            noise = torch.randn_like(x)
            posterior_log_variance_t = self.posterior_log_variance_clipped[t].view(-1, 1, 1, 1, 1).to(x.device)
            return pred_mean + torch.exp(0.5 * posterior_log_variance_t) * noise
        else:
            return pred_mean

    def to(self, device):
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        self.sqrt_alpha_bar = self.sqrt_alpha_bar.to(device)
        self.sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar.to(device)
        self.sqrt_recip_alpha = self.sqrt_recip_alpha.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(device)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device)
        return self


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./outputs_method_c/checkpoints/best.pt")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="./outputs_method_c")
    args = parser.parse_args()

    device = torch.device("cuda")

    print("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    config.output_dir = args.output_dir

    print("Creating SD-VAE...")
    vae = FrameVAE(
        pretrained_model_name_or_path="/data/data5/zhaoran/paper_code/exo/latent/vae_cache/sd-vae-ft-mse",
        freeze=True,
    )
    vae.to(device)
    vae.eval()

    print("Creating model...")
    model = LatentDiffusionTransformerWithBG(
        in_channels=4,
        hidden_size=config.model.hidden_size,
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers,
        mlp_ratio=config.model.mlp_ratio,
        dropout=config.model.dropout,
        num_timesteps=config.model.num_timesteps,
        patch_size=(1, 2, 2),
        use_freq_decomp=True,
        freq_size=(8, 8),
        use_checkpoint=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    diffusion = GaussianDiffusion(
        num_timesteps=config.model.num_timesteps,
        beta_schedule=config.model.beta_schedule,
    )
    diffusion.to(device)

    print("Creating dataloaders...")
    _, val_loader = create_dataloaders(config)

    os.makedirs(os.path.join(config.output_dir, "eval_samples"), exist_ok=True)

    all_psnr = []
    all_ssim = []

    print("Evaluating...")
    for batch_idx, batch in enumerate(tqdm(val_loader, total=args.num_samples)):
        if batch_idx >= args.num_samples:
            break

        exo_video = batch["exo_video"].to(device)
        ego_video = batch["ego_video"].to(device)
        exo_pose = batch["exo_pose"].to(device)
        ego_pose = batch["ego_pose"].to(device)

        B, C, T, H, W = exo_video.shape
        num_history = min(4, T)

        with torch.no_grad():
            exo_z = vae.encode(exo_video)

        pose_cond = torch.cat([exo_pose[:, -1:], ego_pose[:, :1]], dim=1)

        x = torch.randn(B, 4, num_history + 4, H // 8, W // 8, device=device)
        x[:, :, :num_history] = exo_z[:, :, -num_history:]

        for t in tqdm(reversed(range(diffusion.num_timesteps)), desc="Sampling", leave=False):
            t_batch = torch.tensor([t] * B, device=device)
            model_output = model(x, t_batch, pose_cond)
            x[:, :, num_history:] = diffusion.p_sample(
                model_output[:, :, num_history:],
                x[:, :, num_history:],
                t_batch,
            )

        gen_z = x
        gen_video = vae.decode(gen_z)
        target_video = torch.cat([exo_video[:, :, -num_history:], ego_video[:, :, :4]], dim=2)

        for i in range(B):
            for t in range(num_history, num_history + 4):
                gen_np = (gen_video[i, :, t].cpu().numpy() + 1) / 2
                gt_np = (target_video[i, :, t].cpu().numpy() + 1) / 2

                gen_np = np.clip(gen_np, 0, 1)
                gt_np = np.clip(gt_np, 0, 1)

                psnr = compute_psnr(gen_np, gt_np)
                ssim = compute_ssim(gen_np, gt_np)

                all_psnr.append(psnr)
                all_ssim.append(ssim)

        if batch_idx < 5:
            exo_last = exo_video[0, :, -1]
            ego_first = ego_video[0, :, 0]
            gen_frames = [gen_video[0, :, num_history + i] for i in range(4)]

            row = torch.cat([exo_last] + gen_frames + [ego_first], dim=2)

            vutils.save_image(
                row,
                os.path.join(config.output_dir, "eval_samples", f"batch_{batch_idx}.png"),
                normalize=True,
                range=(-1, 1),
            )

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS: Method C (Syn2Seq + SD-VAE Latent + BG)")
    print("=" * 60)
    print(f"PSNR: {np.mean(all_psnr):.2f} dB")
    print(f"SSIM: {np.mean(all_ssim):.4f}")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    print(f"{'Method':<40} {'PSNR (dB)':<12} {'SSIM':<12}")
    print("-" * 60)
    print(f"{'Syn2Seq (pixel, baseline)':<40} {'17.04':<12} {'0.0402':<12}")
    print(f"{'Latent-only (SD-VAE + Diffusion)':<40} {'~18.5+':<12} {'-':<12}")
    print(f"{'Ours (DINO Latent + BG)':<40} {'17.52':<12} {'-':<12}")
    print(f"{'Method C (SD-VAE + BG)':<40} {np.mean(all_psnr):.2f}          {np.mean(all_ssim):.4f}")
    print("=" * 60)

    with open(os.path.join(config.output_dir, "eval_results.txt"), "w") as f:
        f.write("Method C: Syn2Seq + SD-VAE Latent + Background Processing\n")
        f.write("=" * 60 + "\n")
        f.write(f"PSNR: {np.mean(all_psnr):.2f} dB\n")
        f.write(f"SSIM: {np.mean(all_ssim):.4f}\n")

    print(f"\nResults saved to {config.output_dir}/eval_results.txt")


if __name__ == "__main__":
    main()
