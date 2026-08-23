"""Quick evaluation on a subset, runs on CPU."""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.utils as vutils

HERE = os.path.dirname(os.path.abspath(__file__))
SYN2SEQ_DIR = os.path.abspath(os.path.join(HERE, "..", "Syn2Seq"))
sys.path.insert(0, HERE)
sys.path.insert(0, SYN2SEQ_DIR)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

from configs.default_config import Config
from data.dataset import create_dataloaders
from models.vae import FrameVAE
from models.dfot import LatentDiffusionForcingTransformer, GaussianDiffusion
from models.interpolator import LatentVideoInterpolator


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
    return np.mean(ssim_numerator / ssim_denominator)


class QuickLatentEvaluator:
    def __init__(self, vae, model, diffusion, interpolator, val_loader, config, num_batches=10):
        self.vae = vae
        self.model = model
        self.diffusion = diffusion
        self.interpolator = interpolator
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device("cuda")
        self.num_batches = num_batches

        for m in [vae, model, interpolator]:
            m.to(self.device)
            m.eval()
        diffusion.to(self.device)

    @torch.no_grad()
    def evaluate(self):
        all_psnr = []
        all_ssim = []

        os.makedirs(os.path.join(self.config.output_dir, "eval_samples"), exist_ok=True)

        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Quick Eval")):
            if batch_idx >= self.num_batches:
                break

            exo_video = batch["exo_video"].to(self.device)
            ego_video = batch["ego_video"].to(self.device)
            exo_pose = batch["exo_pose"].to(self.device)
            ego_pose = batch["ego_pose"].to(self.device)

            with torch.no_grad():
                exo_z = self.vae.encode(exo_video)
                ego_z = self.vae.encode(ego_video)

                _, interp_z = self.interpolator(exo_z, ego_z)

                pose_cond = torch.cat([exo_pose[:, -1:], ego_pose[:, :1]], dim=1)
                num_interp = interp_z.shape[2]

                gen_z = self._sample_diffusion(
                    exo_z[:, :, -4:],
                    pose_cond,
                    num_gen_frames=num_interp,
                )

                gen_interp_z = gen_z[:, :, -num_interp:]
                gt_interp_z = interp_z

                gen_rgb = self.vae.decode(gen_interp_z)
                gt_rgb = self.vae.decode(gt_interp_z)

                gen_rgb_np = (gen_rgb.permute(0, 2, 3, 4, 1).cpu().numpy() + 1) / 2
                gt_rgb_np = (gt_rgb.permute(0, 2, 3, 4, 1).cpu().numpy() + 1) / 2

                gen_rgb_np = np.clip(gen_rgb_np, 0, 1)
                gt_rgb_np = np.clip(gt_rgb_np, 0, 1)

                B = gen_rgb_np.shape[0]
                for i in range(B):
                    for t in range(num_interp):
                        psnr = compute_psnr(gt_rgb_np[i, t], gen_rgb_np[i, t], data_range=1.0)
                        ssim = compute_ssim(gt_rgb_np[i, t], gen_rgb_np[i, t], multichannel=True, data_range=1.0, channel_axis=-1)
                        all_psnr.append(psnr)
                        all_ssim.append(ssim)

            if batch_idx < 5:
                self._save_visualization(batch_idx, exo_video, gt_rgb, ego_video, gen_rgb, num_interp)

        results = {
            "PSNR": np.mean(all_psnr),
            "SSIM": np.mean(all_ssim),
        }

        print(f"Evaluation Results (on {len(all_psnr)} frames):")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")

        return results

    @torch.no_grad()
    def _sample_diffusion(self, history_z, pose_cond, num_gen_frames):
        B, Cz, T_hist, Hz, Wz = history_z.shape

        x = torch.randn(B, Cz, T_hist + num_gen_frames, Hz, Wz, device=self.device)
        x[:, :, :T_hist] = history_z

        for t in reversed(range(self.diffusion.num_timesteps)):
            t_batch = torch.tensor([t] * B, device=self.device)

            sqrt_alpha_bar_t = self.diffusion.sqrt_alpha_bar[t]
            sqrt_one_minus_alpha_bar_t = self.diffusion.sqrt_one_minus_alpha_bar[t]

            x_noisy_history = sqrt_alpha_bar_t * history_z + sqrt_one_minus_alpha_bar_t * torch.randn_like(history_z)
            x[:, :, :T_hist] = x_noisy_history

            model_output = self.model(x, t_batch, pose_cond)

            x[:, :, T_hist:] = self.diffusion.p_sample(
                model_output[:, :, T_hist:],
                x[:, :, T_hist:],
                t_batch,
            )

        return x

    def _save_visualization(self, batch_idx, exo_video, gt_interp_rgb, ego_video, gen_rgb, num_interp):
        exo_last = exo_video[0, :, -1]
        ego_first = ego_video[0, :, 0]

        interp_list_gt = [gt_interp_rgb[0, :, i].unsqueeze(0) for i in range(num_interp)]
        row1 = torch.cat([exo_last.unsqueeze(0)] + interp_list_gt + [ego_first.unsqueeze(0)], dim=0)

        interp_list_gen = [gen_rgb[0, :, i].unsqueeze(0) for i in range(num_interp)]
        row2 = torch.cat([exo_last.unsqueeze(0)] + interp_list_gen + [ego_first.unsqueeze(0)], dim=0)

        grid = torch.cat([row1, row2], dim=0)

        vutils.save_image(
            grid,
            os.path.join(self.config.output_dir, "eval_samples", f"batch_{batch_idx}.png"),
            nrow=num_interp + 2,
            normalize=True,
            range=(-1, 1),
        )


def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]

    vae = FrameVAE(
        pretrained_model_name_or_path=getattr(config.vae, "pretrained_model_name_or_path", "/data/data5/zhaoran/paper_code/exo/latent/vae_cache/sd-vae-ft-mse"),
        cache_dir=getattr(config.vae, "cache_dir", "./vae_cache"),
        scaling_factor=getattr(config.vae, "scaling_factor", 0.18215),
    )

    model = LatentDiffusionForcingTransformer(
        in_channels=4,
        hidden_size=config.model.hidden_size,
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers,
        mlp_ratio=config.model.mlp_ratio,
        dropout=config.model.dropout,
        num_timesteps=config.model.num_timesteps,
        patch_size=getattr(config.model, "patch_size", (2, 2, 2)),
    )

    diffusion = GaussianDiffusion(
        num_timesteps=config.model.num_timesteps,
        beta_schedule=config.model.beta_schedule,
    )

    interpolator = LatentVideoInterpolator(
        latent_channels=4,
        num_interp_frames=config.data.num_interp_frames,
        hidden_dim=64,
        num_res_blocks=3,
    )

    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    interpolator.load_state_dict(checkpoint["interpolator_state_dict"], strict=False)

    print(f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint['epoch']})")

    return vae, model, diffusion, interpolator, config


def main():
    import argparse
    import glob
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./outputs_latent")
    parser.add_argument("--num_batches", type=int, default=10)
    args = parser.parse_args()

    if args.checkpoint is None:
        ckpt_dir = "./outputs_latent/checkpoints"
        ckpt_list = sorted(glob.glob(os.path.join(ckpt_dir, "checkpoint_*.pt")), key=lambda x: int(x.split("_")[-1].split(".")[0]))
        if not ckpt_list:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
        args.checkpoint = ckpt_list[-1]
        print(f"Using latest checkpoint: {args.checkpoint}")

    vae, model, diffusion, interpolator, config = load_checkpoint(args.checkpoint)
    config.output_dir = args.output_dir

    _, val_loader = create_dataloaders(config)
    print(f"Val samples: {len(val_loader.dataset)}, evaluating on {args.num_batches} batches")

    evaluator = QuickLatentEvaluator(vae, model, diffusion, interpolator, val_loader, config, args.num_batches)
    results = evaluator.evaluate()

    with open(os.path.join(config.output_dir, "quick_eval_results.txt"), "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v:.4f}\n")

    print(f"Results saved to {os.path.join(config.output_dir, 'quick_eval_results.txt')}")


if __name__ == "__main__":
    main()
