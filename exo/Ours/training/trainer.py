import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import numpy as np
from typing import Dict
from tqdm import tqdm


class Syn2SeqTrainerWithFreq:
    def __init__(
        self,
        model,
        diffusion,
        train_loader,
        val_loader,
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
        
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, "samples"), exist_ok=True)
        
        self.loss_history = []
    
    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            exo_video = batch["exo_video"].to(self.device)
            ego_video = batch["ego_video"].to(self.device)
            exo_pose = batch["exo_pose"].to(self.device)
            ego_pose = batch["ego_pose"].to(self.device)
            
            with autocast(enabled=self.config.training.use_amp):
                losses = self.model.compute_losses(
                    exo_video, ego_video, exo_pose, ego_pose, self.diffusion
                )
                loss = losses["total_loss"]
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            if self.config.training.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            self.global_step += 1
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "diff": f"{losses['recon_loss'].item():.4f}",
                "ortho": f"{losses['ortho_loss'].item():.4f}",
            })
            
            if self.global_step % self.config.training.log_interval == 0:
                self.loss_history.append(loss.item())
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def _validate(self, epoch: int) -> float:
        self.model.eval()
        
        total_loss = 0.0
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            exo_video = batch["exo_video"].to(self.device)
            ego_video = batch["ego_video"].to(self.device)
            exo_pose = batch["exo_pose"].to(self.device)
            ego_pose = batch["ego_pose"].to(self.device)
            
            losses = self.model.compute_losses(
                exo_video, ego_video, exo_pose, ego_pose, self.diffusion
            )
            
            total_loss += losses["total_loss"].item()
        
        if epoch % self.config.training.save_interval == 0:
            self._generate_samples(epoch)
        
        return total_loss / len(self.val_loader)
    
    @torch.no_grad()
    def _generate_samples(self, epoch: int):
        batch = next(iter(self.val_loader))
        exo_video = batch["exo_video"].to(self.device)
        ego_video = batch["ego_video"].to(self.device)
        exo_pose = batch["exo_pose"].to(self.device)
        ego_pose = batch["ego_pose"].to(self.device)
        
        gen_video, gen_feat = self.model.sample(exo_video, exo_pose, ego_pose, self.diffusion)
        
        B, C, T_gen, H, W = gen_video.shape
        T_exo = exo_video.shape[2]
        
        exo_last = exo_video[0, :, -1]
        ego_first = ego_video[0, :, 0]
        
        num_gen = min(4, T_gen - T_exo)
        
        gen_imgs = [gen_video[0, :, T_exo + i].unsqueeze(0) for i in range(num_gen)]
        row = torch.cat([exo_last.unsqueeze(0)] + gen_imgs + [ego_first.unsqueeze(0)], dim=0)
        
        vutils.save_image(
            row,
            os.path.join(self.config.output_dir, "samples", f"epoch_{epoch}.png"),
            nrow=num_gen + 2,
            normalize=True,
            range=(-1, 1),
        )
    
    def train(self):
        for epoch in range(self.config.training.num_epochs):
            train_loss = self._train_epoch(epoch)
            val_loss = self._validate(epoch)
            
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            if epoch % self.config.training.save_interval == 0:
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "config": self.config,
                }
                torch.save(
                    checkpoint,
                    os.path.join(self.config.output_dir, "checkpoints", f"checkpoint_{epoch}.pt"),
                )
