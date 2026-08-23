"""
Trainer for latent-space Syn2Seq.

Pipeline (everything between encode and decode is in latent space):

    pixels (B, 3, T, H, W)
        |  VAE.encode (no_grad, frozen)
        v
    latents (B, 4, T, H/8, W/8)
        |  LatentVideoInterpolator -> full_sequence_z, interp_z
        |  LatentDiffusionForcingTransformer -> noise prediction
        v
    diffusion / interpolator losses (in latent space)

Decoding back to RGB only happens for periodic visualisation.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from typing import Optional
from tqdm import tqdm


class LatentSyn2SeqTrainer:
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
    ):
        self.model = model
        self.diffusion = diffusion
        self.interpolator = interpolator
        self.vae = vae
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.device = torch.device(config.training.device)
        self.model.to(self.device)
        self.diffusion.to(self.device)
        self.interpolator.to(self.device)
        self.vae.to(self.device)

        self.use_temporal_consistency = use_temporal_consistency

        trainable_params = list(self.model.parameters()) + list(self.interpolator.parameters())
        self.optimizer = torch.optim.AdamW(
            trainable_params,
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

    def _encode(self, video: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.vae.encode(video)

    def _decode(self, latent: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.vae.decode(latent)

    def _compute_diffusion_loss(
        self,
        z_0: torch.Tensor,
        pose_cond: Optional[torch.Tensor] = None,
        cfg_dropout_prob: float = 0.0,
    ) -> torch.Tensor:
        batch_size = z_0.shape[0]
        t = torch.randint(0, self.diffusion.num_timesteps, (batch_size,), device=self.device)
        noise = torch.randn_like(z_0)
        z_t = self.diffusion.q_sample(z_0, t, noise)

        use_cfg = torch.rand(batch_size, device=self.device) > cfg_dropout_prob
        pose_cond_input = pose_cond if use_cfg.any() else None

        with autocast(enabled=self.config.training.use_amp):
            model_output = self.model(z_t, t, pose_cond_input)
            loss = F.mse_loss(model_output, noise)
        return loss

    def _compute_interpolator_losses(
        self,
        interp_z: torch.Tensor,
        exo_z: torch.Tensor,
        ego_z: torch.Tensor,
    ):
        num_interp = interp_z.shape[2]
        exo_last = exo_z[:, :, -1:]
        ego_first = ego_z[:, :, :1]

        alpha = torch.linspace(0, 1, num_interp + 2, device=exo_z.device)[1:-1]
        alpha = alpha.view(1, 1, -1, 1, 1)
        linear_interp_gt = (1 - alpha) * exo_last + alpha * ego_first

        loss_dict = {}
        loss_dict['recon'] = F.mse_loss(interp_z, linear_interp_gt)

        if self.use_temporal_consistency and num_interp >= 2:
            temp_diff = interp_z[:, :, 1:] - interp_z[:, :, :-1]
            loss_dict['temporal'] = F.mse_loss(temp_diff, torch.zeros_like(temp_diff))
        return loss_dict

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        self.interpolator.train()
        self.vae.eval()

        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch in pbar:
            exo_video = batch["exo_video"].to(self.device, non_blocking=True)
            ego_video = batch["ego_video"].to(self.device, non_blocking=True)
            exo_pose = batch["exo_pose"].to(self.device, non_blocking=True)
            ego_pose = batch["ego_pose"].to(self.device, non_blocking=True)

            exo_z = self._encode(exo_video)
            ego_z = self._encode(ego_video)

            with autocast(enabled=self.config.training.use_amp):
                _, interp_z = self.interpolator(exo_z, ego_z)

                pose_cond = torch.cat([exo_pose[:, -1:], ego_pose[:, :1]], dim=1)

                use_exo_to_interp = torch.rand(1).item() > 0.5
                if use_exo_to_interp:
                    train_seq_z = torch.cat([exo_z[:, :, -4:], interp_z], dim=2)
                else:
                    train_seq_z = torch.cat([interp_z, ego_z[:, :, :4]], dim=2)

                diff_loss = self._compute_diffusion_loss(
                    train_seq_z,
                    pose_cond,
                    self.config.training.cfg_dropout_prob,
                )

                interp_losses = self._compute_interpolator_losses(interp_z, exo_z, ego_z)
                interp_loss = interp_losses['recon']
                if self.use_temporal_consistency and 'temporal' in interp_losses:
                    interp_loss = interp_loss + 0.05 * interp_losses['temporal']

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

            total_loss += loss.item()
            self.global_step += 1

            log_dict = {'loss': loss.item(), 'diff': diff_loss.item(), 'interp': interp_loss.item()}
            if self.use_temporal_consistency and 'temporal' in interp_losses:
                log_dict['temp'] = interp_losses['temporal'].item()
            pbar.set_postfix(log_dict)

            if self.global_step % self.config.training.log_interval == 0:
                self.loss_history.append(loss.item())

        return total_loss / max(1, len(self.train_loader))

    @torch.no_grad()
    def _validate(self, epoch: int) -> float:
        self.model.eval()
        self.interpolator.eval()

        total_loss = 0.0
        for batch in tqdm(self.val_loader, desc="Validating"):
            exo_video = batch["exo_video"].to(self.device, non_blocking=True)
            ego_video = batch["ego_video"].to(self.device, non_blocking=True)
            exo_pose = batch["exo_pose"].to(self.device, non_blocking=True)
            ego_pose = batch["ego_pose"].to(self.device, non_blocking=True)

            exo_z = self._encode(exo_video)
            ego_z = self._encode(ego_video)

            _, interp_z = self.interpolator(exo_z, ego_z)
            pose_cond = torch.cat([exo_pose[:, -1:], ego_pose[:, :1]], dim=1)

            train_seq_z = torch.cat([exo_z[:, :, -4:], interp_z, ego_z[:, :, :4]], dim=2)
            loss = self._compute_diffusion_loss(train_seq_z, pose_cond, 0.0)
            total_loss += loss.item()

        if epoch % self.config.training.save_interval == 0:
            self._generate_samples(epoch)

        return total_loss / max(1, len(self.val_loader))

    @torch.no_grad()
    def _sample_diffusion_latent(
        self,
        history_z: torch.Tensor,
        pose_cond: torch.Tensor,
        num_gen_frames: int,
    ) -> torch.Tensor:
        B, Cz, T_hist, Hz, Wz = history_z.shape

        z = torch.randn(B, Cz, T_hist + num_gen_frames, Hz, Wz, device=self.device)
        z[:, :, :T_hist] = history_z

        for t in reversed(range(self.diffusion.num_timesteps)):
            t_batch = torch.tensor([t] * B, device=self.device)

            sqrt_alpha_bar_t = self.diffusion.sqrt_alpha_bar[t]
            sqrt_one_minus_alpha_bar_t = self.diffusion.sqrt_one_minus_alpha_bar[t]

            z_noisy_history = (
                sqrt_alpha_bar_t * history_z
                + sqrt_one_minus_alpha_bar_t * torch.randn_like(history_z)
            )
            z[:, :, :T_hist] = z_noisy_history

            model_output = self.model(z, t_batch, pose_cond)
            z[:, :, T_hist:] = self.diffusion.p_sample(
                model_output[:, :, T_hist:],
                z[:, :, T_hist:],
                t_batch,
            )
        return z

    @torch.no_grad()
    def _generate_samples(self, epoch: int):
        batch = next(iter(self.val_loader))
        exo_video = batch["exo_video"].to(self.device)
        ego_video = batch["ego_video"].to(self.device)
        exo_pose = batch["exo_pose"].to(self.device)
        ego_pose = batch["ego_pose"].to(self.device)

        exo_z = self._encode(exo_video)
        ego_z = self._encode(ego_video)
        _, interp_z = self.interpolator(exo_z, ego_z)

        pose_cond = torch.cat([exo_pose[:, -1:], ego_pose[:, :1]], dim=1)

        gen_z = self._sample_diffusion_latent(
            exo_z[:, :, -4:],
            pose_cond,
            num_gen_frames=interp_z.shape[2] + 4,
        )

        interp_pixels = self._decode(interp_z)
        gen_pixels = self._decode(gen_z)

        grid = self._create_grid(exo_video, interp_pixels, ego_video, gen_pixels)

        vutils.save_image(
            grid,
            os.path.join(self.config.output_dir, "samples", f"epoch_{epoch}.png"),
            nrow=8,
            normalize=True,
            value_range=(-1, 1),
        )

    def _create_grid(self, exo_video, interp_frames, ego_video, gen_sequence):
        B, C, T, H, W = exo_video.shape
        exo_last = exo_video[0, :, -1]
        ego_first = ego_video[0, :, 0]
        num_interp = interp_frames.shape[2]

        interp_list = [interp_frames[0, :, i].unsqueeze(0) for i in range(num_interp)]
        row1 = torch.cat([exo_last.unsqueeze(0)] + interp_list + [ego_first.unsqueeze(0)], dim=0)

        gen_interp_list = [
            gen_sequence[0, :, T_hist_i].unsqueeze(0)
            for T_hist_i in range(gen_sequence.shape[2] - num_interp - 4, gen_sequence.shape[2] - 4)
        ]
        row2 = torch.cat([exo_last.unsqueeze(0)] + gen_interp_list + [ego_first.unsqueeze(0)], dim=0)

        return torch.cat([row1, row2], dim=0)

    def train(self):
        for epoch in range(self.start_epoch, self.config.training.num_epochs):
            train_loss = self._train_epoch(epoch)
            val_loss = self._validate(epoch)
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            if epoch % self.config.training.save_interval == 0:
                self._save_checkpoint(epoch)

    def _save_checkpoint(self, epoch: int):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "interpolator_state_dict": self.interpolator.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "global_step": self.global_step,
            "loss_history": self.loss_history,
            "config": self.config,
        }
        torch.save(
            checkpoint,
            os.path.join(self.config.output_dir, "checkpoints", f"checkpoint_{epoch}.pt"),
        )
