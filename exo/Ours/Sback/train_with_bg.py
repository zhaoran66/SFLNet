import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

import sys
import argparse
import random
import numpy as np
import torch

from configs.default_config import Config
from data.dataset import create_dataloaders
from models.dfot_with_bg import DiffusionForcingTransformerWithBG
from training.trainer_with_bg import Syn2SeqBGTrainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train Syn2Seq with Background Processing")
    parser.add_argument("--output_dir", type=str, default="./outputs_with_bg",
                        help="Output directory")
    parser.add_argument("--no_freq_decomp", action="store_true",
                        help="Disable frequency decomposition")
    args = parser.parse_args()

    config = Config()
    set_seed(config.seed)

    if args.output_dir:
        config.output_dir = args.output_dir

    print("=" * 60)
    print("Training Syn2Seq with Background Processing (Pixel Space)")
    print("=" * 60)
    print(f"GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"Frequency Decomposition: {'DISABLED' if args.no_freq_decomp else 'ENABLED'}")
    print(f"Output dir: {config.output_dir}")

    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    print("Creating DFoT with Background Processing...")
    model = DiffusionForcingTransformerWithBG(
        in_channels=3,
        hidden_size=config.model.hidden_size,
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers,
        mlp_ratio=config.model.mlp_ratio,
        dropout=config.model.dropout,
        num_timesteps=config.model.num_timesteps,
        patch_size=getattr(config.model, "patch_size", (2, 4, 4)),
        use_freq_decomp=not args.no_freq_decomp,
        pose_mask_sigma=getattr(config.model, "pose_mask_sigma", 0.1),
        lambda_struct_fg=getattr(config.model, "lambda_struct_fg", 1.0),
        lambda_struct_bg=getattr(config.model, "lambda_struct_bg", 1.0),
        use_checkpoint=True,
    )

    diffusion = GaussianDiffusion(
        num_timesteps=config.model.num_timesteps,
        beta_schedule=config.model.beta_schedule,
    )

    print("Creating trainer...")
    trainer = Syn2SeqBGTrainer(
        model=model,
        diffusion=diffusion,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )

    print(f"Starting training for {config.training.num_epochs} epochs...")
    trainer.train()

    print("=" * 60)
    print("Training Complete!")
    print(f"Checkpoints saved to: {config.output_dir}/checkpoints/")
    print("=" * 60)


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


if __name__ == "__main__":
    main()
