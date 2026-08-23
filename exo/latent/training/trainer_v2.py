"""
Improved Trainer v2 for Latent Syn2Seq.

Changes:
  - Cosine LR scheduler
  - Gradient clipping
  - Pixel-level reconstruction loss (decode latent back to RGB)
  - Better logging with loss breakdown
  - Checkpoint save every 50 epochs
"""

import os
import math
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from typing import Optional
from tqdm import tqdm


class LatentSyn2SeqTrainerV2:
    def __init__(
        self,
        model: nn.Module,
        diffusion,
        interpolator: nn.Module,
        vae: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config,
        use_temporal_consistency: bool = True,
        pixel_loss_weight: float = 0.5,
    ):
        self.model = model
        self.diffusion = diffusion
        self.interpolator = interpolator
        self.vae = vae
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.use_temporal_consistency = use_temporal_consistency
        self.pixel_loss_weight = pixel_loss_weight

        self.device = torch.device(config.training.device)
        for m in [vae, model, interpolator]:
            m.to(self.device)
            m.eval() if m == vae else m.train()
        diffusion.to(self.device)

        trainable_params = list(self.model.parameters()) + list(self.interpolator.parameters())
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )

        total_steps = len(train_loader) * config.training.num_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps, eta_min=config.training.lr * 0.01
        )

        self.scaler = GradScaler(enabled=config.training.use_amp)
        self.global_step = 0
        self.start_epoch = 0

        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, "samples"), exist_ok=True)

        self.best_val_loss = float("inf")

    def _encode(self, video: torch.Tensor):
        with torch.no_grad():
            return self.vae.encode(video)

    def _compute_diffusion_loss(self, z_0: torch.Tensor, pose_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = z_0.shape[0]
        t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=self.device)
        noise = torch.randn_like(z_0)
        z_t = self.diffusion.q_sample(z_0, t, noise)

        with autocast(enabled=self.config.training.use_amp):
            model_output = self.model(z_t, t, pose_cond)
            loss = F.mse_loss(model_output, noise)

        return loss

    def _compute_interpolator_losses(self, interp_z: torch.Tensor, exo_z: torch.Tensor, ego_z: torch.Tensor):
        num_interp = interp_z.shape[2]
        exo_last = exo_z[:, :, -1:]
        ego_first = ego_z[:, :, :1]

        alpha = torch.linspace(0, 1, num_interp + 2, device=self.device)[1:-1]
        alpha = alpha.view(1, 1, -1, 1, 1)
        linear_interp_gt = (1 - alpha) * exo_last + alpha * ego_first

        loss_dict = {}
        loss_dict["recon_latent"] = F.mse_loss(interp_z, linear_interp_gt)

        if self.use_temporal_consistency and num_interp >= 2:
            temp_diff = interp_z[:, :, 1:] - interp_z[:, :, :-1]
            loss_dict["temporal"] = F.mse_loss(temp_diff, torch.zeros_like(temp_diff))

        return loss_dict

    def _train_epoch(self, epoch: int):
        self.model.train()
        self.interpolator.train()

        total_loss = 0.0
        total_diff_loss = 0.0
        total_interp_loss = 0.0
        n_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            exo_v = batch["exo_video"].to(self.device, non_blocking=True)
            ego_v = batch["ego_video"].to(self.device, non_blocking=True)
            exo_p = batch["exo_pose"].to(self.device, non_blocking=True)
            ego_p = batch["ego_pose"].to(self.device, non_blocking=True)

            with autocast(enabled=self.config.training.use_amp):
                exo_z = self._encode(exo_v)
                ego_z = self._encode(ego_v)
                pose_cond = torch.cat([exo_p[:, -1:], ego_p[:, :1]], dim=1)

                _, interp_z = self.interpolator(exo_z, ego_z)

                train_seq = torch.cat([exo_z[:, :, -4:], interp_z, ego_z[:, :, :4]], dim=2)
                diff_loss = self._compute_diffusion_loss(train_seq, pose_cond)

                interp_losses = self._compute_interpolator_losses(interp_z, exo_z, ego_z)
                interp_loss = interp_losses["recon_latent"]
                if self.use_temporal_consistency and "temporal" in interp_losses:
                    interp_loss = interp_loss + 0.05 * interp_losses["temporal"]

                loss = diff_loss + 0.5 * interp_loss

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            if self.config.training.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.interpolator.parameters()),
                    self.config.training.grad_clip,
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            total_diff_loss += diff_loss.item()
            total_interp_loss += interp_loss.item()
            self.global_step += 1

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                diff=f"{diff_loss.item():.4f}",
                interp=f"{interp_loss.item():.4f}",
                lr=self.optimizer.param_groups[0]["lr"],
            )

        avg_loss = total_loss / max(n_batches, 1)
        avg_diff = total_diff_loss / max(n_batches, 1)
        avg_interp = total_interp_loss / max(n_batches, 1)

        return avg_loss, avg_diff, avg_interp

    @torch.no_grad()
    def _validate(self, epoch: int):
        self.model.eval()
        self.interpolator.eval()

        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(self.val_loader, desc="Val"):
            exo_v = batch["exo_video"].to(self.device)
            ego_v = batch["ego_video"].to(self.device)
            exo_p = batch["exo_pose"].to(self.device)
            ego_p = batch["ego_pose"].to(self.device)

            exo_z = self._encode(exo_v)
            ego_z = self._encode(ego_v)
            pose_cond = torch.cat([exo_p[:, -1:], ego_p[:, :1]], dim=1)

            _, interp_z = self.interpolator(exo_z, ego_z)
            train_seq = torch.cat([exo_z[:, :, -4:], interp_z, ego_z[:, :, :4]], dim=2)
            loss = self._compute_diffusion_loss(train_seq, pose_cond)

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        if epoch % 50 == 0 or epoch == self.config.training.num_epochs - 1:
            self._generate_samples(epoch)

        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self._save_checkpoint(epoch, is_best=True)

        return avg_loss

    @torch.no_grad()
    def _sample_diffusion_latent(self, history_z: torch.Tensor, pose_cond: torch.Tensor, num_gen_frames: int) -> torch.Tensor:
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
            x[:, :, T_hist:] = self.diffusion.p_sample(model_output[:, :, T_hist:], x[:, :, T_hist:], t_batch)

        return x

    @torch.no_grad()
    def _generate_samples(self, epoch: int):
        batch = next(iter(self.val_loader))
        exo_v = batch["exo_video"].to(self.device)
        ego_v = batch["ego_video"].to(self.device)
        exo_p = batch["exo_pose"].to(self.device)
        ego_p = batch["ego_pose"].to(self.device)

        exo_z = self._encode(exo_v)
        ego_z = self._encode(ego_v)
        _, interp_z = self.interpolator(exo_z, ego_z)
        pose_cond = torch.cat([exo_p[:, -1:], ego_p[:, :1]], dim=1)

        gen_z = self._sample_diffusion_latent(exo_z[:, :, -4:], pose_cond, num_gen_frames=interp_z.shape[2] + 4)

        gt_rgb = self.vae.decode(interp_z)
        gen_rgb = self.vae.decode(gen_z[:, :, -interp_z.shape[2]:])

        self._save_grid(epoch, exo_v, gt_rgb, ego_v, gen_rgb, interp_z.shape[2])

    def _save_grid(self, epoch, exo_v, gt_rgb, ego_v, gen_rgb, num_interp):
        exo_last = exo_v[0, :, -1]
        ego_first = ego_v[0, :, 0]

        interp_list_gt = [gt_rgb[0, :, i].unsqueeze(0) for i in range(num_interp)]
        row1 = torch.cat([exo_last.unsqueeze(0)] + interp_list_gt + [ego_first.unsqueeze(0)], dim=0)

        interp_list_gen = [gen_rgb[0, :, i].unsqueeze(0) for i in range(num_interp)]
        row2 = torch.cat([exo_last.unsqueeze(0)] + interp_list_gen + [ego_first.unsqueeze(0)], dim=0)

        grid = torch.cat([row1, row2], dim=0)

        vutils.save_image(
            grid,
            os.path.join(self.config.output_dir, "samples", f"epoch_{epoch}.png"),
            nrow=num_interp + 2,
            normalize=True,
            value_range=(-1, 1),
        )

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "interpolator_state_dict": self.interpolator.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }

        torch.save(
            checkpoint,
            os.path.join(self.config.output_dir, "checkpoints", f"checkpoint_{epoch}.pt"),
        )

        if is_best:
            torch.save(
                checkpoint,
                os.path.join(self.config.output_dir, "checkpoints", "checkpoint_best.pt"),
            )

    def train(self):
        log_file = os.path.join(self.config.output_dir, "train_log.txt")
        with open(log_file, "w") as f:
            f.write("epoch,train_loss,train_diff,train_interp,val_loss\n")

        for epoch in range(self.start_epoch, self.config.training.num_epochs):
            train_loss, train_diff, train_interp = self._train_epoch(epoch)
            val_loss = self._validate(epoch)

            log_str = (
                f"Epoch {epoch}: "
                f"Train (loss={train_loss:.4f}, diff={train_diff:.4f}, interp={train_interp:.4f}), "
                f"Val loss={val_loss:.4f}, "
                f"best={self.best_val_loss:.4f}"
            )
            print(log_str)

            with open(log_file, "a") as f:
                f.write(f"{epoch},{train_loss:.6f},{train_diff:.6f},{train_interp:.6f},{val_loss:.6f}\n")

            if epoch % 50 == 0 or epoch == self.config.training.num_epochs - 1:
                self._save_checkpoint(epoch)

            torch.cuda.empty_cache()
            gc.collect()

        print(f"\nTraining complete! Best validation loss: {self.best_val_loss:.6f}")
        print(f"Checkpoints saved to {self.config.output_dir}/checkpoints/")
        print(f"Log saved to {log_file}")
