"""
Ablation: Train Latent-only model (NO Frequency Routing)
For fair comparison with full FASR
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import random
import numpy as np
from tqdm import tqdm

from configs.config_tiny import Config as BaseConfig
from data.dataset import create_dataloaders
from models.fasr_model_ablation import LatentDiffusionTransformer


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
    
    def q_sample(self, x_start, t, noise=None):
        device = x_start.device
        t = t.to(self.sqrt_alpha_bar.device)
        
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1, 1).to(device)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1, 1).to(device)
        
        return sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise
    
    def p_sample(self, model_output, x, t):
        device = x.device
        t = t.to(self.sqrt_recip_alpha.device)
        
        sqrt_recip_alpha_t = self.sqrt_recip_alpha[t].view(-1, 1, 1, 1, 1).to(device)
        beta_t = self.beta[t].view(-1, 1, 1, 1, 1).to(device)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1, 1).to(device)
        
        pred_mean = sqrt_recip_alpha_t * (x - beta_t / sqrt_one_minus_alpha_bar_t * model_output)
        
        if t[0] > 0:
            noise = torch.randn_like(x)
            posterior_log_variance_t = self.posterior_log_variance_clipped[t].view(-1, 1, 1, 1, 1).to(device)
            return pred_mean + torch.exp(0.5 * posterior_log_variance_t) * noise
        else:
            return pred_mean


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train():
    config = BaseConfig()
    config.output_dir = "./outputs_ablation_latent_only"
    set_seed(config.seed)
    
    print("=" * 70)
    print("ABLATION: Latent-only (NO Frequency Routing)")
    print("=" * 70)
    print("Model: DINOv2 Latent + Simple Transformer Diffusion")
    print("Comparison: Syn2Seq (pixel) vs FASR (full)")
    print("=" * 70)
    
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, "checkpoints"), exist_ok=True)
    
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    print("\nCreating model...")
    model = LatentDiffusionTransformer(config)
    diffusion = GaussianDiffusion(
        num_timesteps=config.model.num_timesteps,
        beta_schedule=config.model.beta_schedule,
    )
    
    device = torch.device("cuda")
    model.to(device)
    diffusion.to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )
    
    scaler = torch.cuda.amp.GradScaler(enabled=config.training.use_amp)
    
    num_epochs = config.training.num_epochs
    best_val_loss = float('inf')
    
    print(f"\nStarting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            exo_video = batch["exo_video"].to(device)
            ego_video = batch["ego_video"].to(device)
            exo_pose = batch["exo_pose"].to(device)
            ego_pose = batch["ego_pose"].to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=config.training.use_amp):
                losses = model.compute_losses(
                    exo_video, ego_video, exo_pose, ego_pose, diffusion
                )
                loss = losses["total_loss"]
            
            scaler.scale(loss).backward()
            
            if config.training.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
            
            total_train_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "diff": f"{losses['diffusion_loss'].item():.4f}",
                "recon": f"{losses['recon_loss'].item():.4f}",
            })
        
        avg_train_loss = total_train_loss / num_batches
        
        model.eval()
        total_val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                exo_video = batch["exo_video"].to(device)
                ego_video = batch["ego_video"].to(device)
                exo_pose = batch["exo_pose"].to(device)
                ego_pose = batch["ego_pose"].to(device)
                
                with torch.cuda.amp.autocast(enabled=config.training.use_amp):
                    losses = model.compute_losses(
                        exo_video, ego_video, exo_pose, ego_pose, diffusion
                    )
                
                total_val_loss += losses["total_loss"].item()
                val_batches += 1
        
        avg_val_loss = total_val_loss / val_batches
        
        print(f"\nEpoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"New best val loss: {best_val_loss:.4f}")
            
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "config": config,
            }
            torch.save(checkpoint, os.path.join(config.output_dir, "checkpoints", "best_model.pt"))
        
        if epoch % 5 == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "config": config,
            }
            torch.save(checkpoint, os.path.join(config.output_dir, "checkpoints", f"checkpoint_{epoch}.pt"))
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {config.output_dir}/checkpoints/")
    print("=" * 70)
    print("\nNow you can evaluate this checkpoint to get the Latent-only baseline!")


if __name__ == "__main__":
    train()
