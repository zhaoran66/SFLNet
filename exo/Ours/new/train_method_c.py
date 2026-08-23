"""
Method C: Syn2Seq + SD-VAE Latent + Background Processing
Training Script
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config_tiny import Config
from data.dataset import create_dataloaders
from models.method_c_vae import FrameVAE
from models.method_c_model import LatentDiffusionTransformerWithBG


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


class MethodCTrainer:
    def __init__(
        self,
        model: nn.Module,
        vae: nn.Module,
        diffusion,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config,
    ):
        self.model = model
        self.vae = vae
        self.diffusion = diffusion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.device = torch.device(config.training.device)
        self.model.to(self.device)
        self.vae.to(self.device)
        self.diffusion.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )

        self.scaler = GradScaler(enabled=config.training.use_amp)
        self.global_step = 0
        self.start_epoch = 0

        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, "samples"), exist_ok=True)

        self.loss_history = []
        self.best_val_loss = float("inf")

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            exo_video = batch["exo_video"].to(self.device, non_blocking=True)
            ego_video = batch["ego_video"].to(self.device, non_blocking=True)
            exo_pose = batch["exo_pose"].to(self.device, non_blocking=True)
            ego_pose = batch["ego_pose"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with autocast(enabled=self.config.training.use_amp):
                losses = self.model.compute_losses(
                    exo_video, ego_video, exo_pose, ego_pose, self.diffusion, self.vae
                )
                loss = losses["total_loss"]

            self.scaler.scale(loss).backward()

            if self.config.training.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.grad_clip
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "diff": f"{losses['diffusion_loss'].item():.4f}",
                "fg": f"{losses['fg_loss'].item():.4f}",
                "bg": f"{losses['bg_loss'].item():.4f}",
            })

        avg_loss = total_loss / num_batches
        return avg_loss

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> float:
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc=f"Validating"):
            exo_video = batch["exo_video"].to(self.device)
            ego_video = batch["ego_video"].to(self.device)
            exo_pose = batch["exo_pose"].to(self.device)
            ego_pose = batch["ego_pose"].to(self.device)

            with autocast(enabled=self.config.training.use_amp):
                losses = self.model.compute_losses(
                    exo_video, ego_video, exo_pose, ego_pose, self.diffusion, self.vae
                )

            total_loss += losses["total_loss"].item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    @torch.no_grad()
    def _sample_and_save(self, epoch: int):
        self.model.eval()

        batch = next(iter(self.val_loader))
        exo_video = batch["exo_video"].to(self.device)
        ego_video = batch["ego_video"].to(self.device)
        exo_pose = batch["exo_pose"].to(self.device)
        ego_pose = batch["ego_pose"].to(self.device)

        B, C, T, H, W = exo_video.shape
        num_history = min(4, T)

        with torch.no_grad():
            exo_z = self.vae.encode(exo_video)

        pose_cond = torch.cat([exo_pose[:, -1:], ego_pose[:, :1]], dim=1)

        x = torch.randn(B, 4, num_history + 4, H // 8, W // 8, device=self.device)
        x[:, :, :num_history] = exo_z[:, :, -num_history:]

        for t in tqdm(reversed(range(self.diffusion.num_timesteps)), desc="Sampling", leave=False):
            t_batch = torch.tensor([t] * B, device=self.device)
            model_output = self.model(x, t_batch, pose_cond)
            x[:, :, num_history:] = self.diffusion.p_sample(
                model_output[:, :, num_history:],
                x[:, :, num_history:],
                t_batch,
            )

        gen_z = x
        gen_video = self.vae.decode(gen_z)

        exo_last = exo_video[0, :, -1]
        ego_first = ego_video[0, :, 0]
        gen_frames = [gen_video[0, :, num_history + i] for i in range(4)]

        row = torch.cat([exo_last] + gen_frames + [ego_first], dim=2)

        vutils.save_image(
            row,
            os.path.join(self.config.output_dir, "samples", f"epoch_{epoch}.png"),
            normalize=True,
            range=(-1, 1),
        )

    def train(self):
        for epoch in range(self.start_epoch, self.config.training.num_epochs):
            print(f"\nEpoch {epoch}/{self.config.training.num_epochs}")
            print("-" * 60)

            train_loss = self._train_epoch(epoch)
            val_loss = self._val_epoch(epoch)

            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            self.loss_history.append({"train": train_loss, "val": val_loss})

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                print(f"New best val loss: {self.best_val_loss:.4f}")
                self._save_checkpoint(epoch, "best")

            if epoch % 5 == 0:
                self._save_checkpoint(epoch, f"epoch_{epoch}")
                self._sample_and_save(epoch)

        self._save_checkpoint(self.config.training.num_epochs - 1, "last")

    def _save_checkpoint(self, epoch: int, name: str):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
            "loss_history": self.loss_history,
        }
        torch.save(
            checkpoint,
            os.path.join(self.config.output_dir, "checkpoints", f"{name}.pt"),
        )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train Method C: Syn2Seq + SD-VAE Latent + Background Processing")
    parser.add_argument("--output_dir", type=str, default="./outputs_method_c",
                        help="Output directory")
    parser.add_argument("--no_freq_decomp", action="store_true",
                        help="Disable frequency decomposition (ablation)")
    args = parser.parse_args()

    config = Config()
    set_seed(config.seed)

    if args.output_dir:
        config.output_dir = args.output_dir

    print("=" * 60)
    print("Method C: Syn2Seq + SD-VAE Latent + Background Processing")
    print("=" * 60)
    print(f"GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"Frequency Decomposition: {'DISABLED' if args.no_freq_decomp else 'ENABLED'}")
    print(f"Output dir: {config.output_dir}")

    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    print("Creating SD-VAE...")
    vae = FrameVAE(
        pretrained_model_name_or_path="/data/data5/zhaoran/paper_code/exo/latent/vae_cache/sd-vae-ft-mse",
        freeze=True,
    )

    print("Creating Diffusion Transformer with Background Processing...")
    model = LatentDiffusionTransformerWithBG(
        in_channels=4,
        hidden_size=config.model.hidden_size,
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers,
        mlp_ratio=config.model.mlp_ratio,
        dropout=config.model.dropout,
        num_timesteps=config.model.num_timesteps,
        patch_size=(1, 2, 2),
        use_freq_decomp=not args.no_freq_decomp,
        freq_size=(8, 8),
        spectral_sigma=0.5,
        use_dct=True,
        pose_mask_sigma=0.1,
        lambda_struct_fg=1.0,
        lambda_struct_bg=1.0,
        use_checkpoint=True,
    )

    diffusion = GaussianDiffusion(
        num_timesteps=config.model.num_timesteps,
        beta_schedule=config.model.beta_schedule,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    print("Creating trainer...")
    trainer = MethodCTrainer(
        model=model,
        vae=vae,
        diffusion=diffusion,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )

    print(f"Starting training for {config.training.num_epochs} epochs...")
    trainer.train()

    print("=" * 60)
    print("Training Complete!")
    print(f"Best val loss: {trainer.best_val_loss:.4f}")
    print(f"Checkpoints saved to: {config.output_dir}/checkpoints/")
    print("=" * 60)


if __name__ == "__main__":
    main()
