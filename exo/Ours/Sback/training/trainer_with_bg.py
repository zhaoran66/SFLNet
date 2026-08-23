import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from typing import Optional
from tqdm import tqdm


class Syn2SeqBGTrainer:
    def __init__(
        self,
        model: nn.Module,
        diffusion,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config,
    ):
        self.model = model
        self.diffusion = diffusion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.device = torch.device(config.training.device)
        self.model.to(self.device)
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
                    exo_video, ego_video, exo_pose, ego_pose, self.diffusion
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
                    exo_video, ego_video, exo_pose, ego_pose, self.diffusion
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

        pose_cond = torch.cat([exo_pose[:, -1:], ego_pose[:, :1]], dim=1)

        x = torch.randn(B, C, num_history + 4, H, W, device=self.device)
        x[:, :, :num_history] = exo_video[:, :, -num_history:]

        for t in reversed(range(self.diffusion.num_timesteps)):
            t_batch = torch.tensor([t] * B, device=self.device)
            model_output = self.model(x, t_batch, pose_cond)
            x[:, :, num_history:] = self.diffusion.p_sample(
                model_output[:, :, num_history:],
                x[:, :, num_history:],
                t_batch,
            )

        gen_video = x

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
