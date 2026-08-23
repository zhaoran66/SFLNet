import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import numpy as np
from typing import Dict, Optional
from tqdm import tqdm


class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        from torchvision import models
        vgg16 = models.vgg16(pretrained=True).features[:16].eval().to(device)
        self.vgg16 = vgg16
        for param in self.vgg16.parameters():
            param.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1).to(device))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1).to(device))
    
    def forward(self, x, y):
        B, C, T, H, W = x.shape
        x_norm = (x + 1) / 2
        y_norm = (y + 1) / 2
        x_norm = (x_norm - self.mean) / self.std
        y_norm = (y_norm - self.mean) / self.std
        
        x_reshaped = x_norm.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
        y_reshaped = y_norm.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
        
        feat_x = self.vgg16(x_reshaped)
        feat_y = self.vgg16(y_reshaped)
        
        return F.mse_loss(feat_x, feat_y)


class Syn2SeqTrainer:
    def __init__(
        self,
        model: nn.Module,
        diffusion,
        interpolator: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config,
        use_perceptual_loss: bool = True,
        use_temporal_consistency: bool = True,
    ):
        self.model = model
        self.diffusion = diffusion
        self.interpolator = interpolator
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        self.device = torch.device(config.training.device)
        self.model.to(self.device)
        self.diffusion.to(self.device)
        self.interpolator.to(self.device)
        
        self.use_perceptual_loss = use_perceptual_loss
        self.use_temporal_consistency = use_temporal_consistency
        
        if self.use_perceptual_loss:
            self.perceptual_loss_fn = PerceptualLoss(self.device)
            print("? Enabled Perceptual Loss")
        
        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.interpolator.parameters()),
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
    
    def _compute_diffusion_loss(
        self,
        x_0: torch.Tensor,
        pose_cond: Optional[torch.Tensor] = None,
        cfg_dropout_prob: float = 0.0,
    ) -> torch.Tensor:
        batch_size = x_0.shape[0]
        
        t = torch.randint(0, self.diffusion.num_timesteps, (batch_size,), device=self.device)
        
        noise = torch.randn_like(x_0)
        x_t = self.diffusion.q_sample(x_0, t, noise)
        
        use_cfg = torch.rand(batch_size, device=self.device) > cfg_dropout_prob
        pose_cond_input = pose_cond if use_cfg.any() else None
        
        with autocast(enabled=self.config.training.use_amp):
            model_output = self.model(x_t, t, pose_cond_input)
            loss = F.mse_loss(model_output, noise)
        
        return loss
    
    def _compute_interpolator_losses(
        self,
        interp_frames: torch.Tensor,
        exo_video: torch.Tensor,
        ego_video: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        num_interp = interp_frames.shape[2]
        exo_last = exo_video[:, :, -1:]
        ego_first = ego_video[:, :, :1]
        
        alpha = torch.linspace(0, 1, num_interp + 2, device=exo_video.device)[1:-1]
        alpha = alpha.view(1, 1, -1, 1, 1)
        linear_interp_gt = (1 - alpha) * exo_last + alpha * ego_first
        
        loss_dict = {}
        
        recon_loss = F.mse_loss(interp_frames, linear_interp_gt)
        loss_dict['recon'] = recon_loss
        
        if self.use_perceptual_loss:
            perc_loss = self.perceptual_loss_fn(interp_frames, linear_interp_gt)
            loss_dict['perceptual'] = perc_loss
        
        if self.use_temporal_consistency and num_interp >= 2:
            temp_diff = interp_frames[:, :, 1:] - interp_frames[:, :, :-1]
            temp_loss = F.mse_loss(temp_diff, torch.zeros_like(temp_diff))
            loss_dict['temporal'] = temp_loss
        
        return loss_dict
    
    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        self.interpolator.train()
        
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            exo_video = batch["exo_video"].to(self.device)
            ego_video = batch["ego_video"].to(self.device)
            exo_pose = batch["exo_pose"].to(self.device)
            ego_pose = batch["ego_pose"].to(self.device)
            
            with autocast(enabled=self.config.training.use_amp):
                full_sequence, interp_frames = self.interpolator(exo_video, ego_video)
                
                pose_cond = torch.cat([exo_pose[:, -1:], ego_pose[:, :1]], dim=1)
                
                use_exo_to_interp = torch.rand(1).item() > 0.5
                
                if use_exo_to_interp:
                    train_sequence = torch.cat([exo_video[:, :, -4:], interp_frames], dim=2)
                else:
                    train_sequence = torch.cat([interp_frames, ego_video[:, :, :4]], dim=2)
                
                diff_loss = self._compute_diffusion_loss(
                    train_sequence,
                    pose_cond,
                    self.config.training.cfg_dropout_prob,
                )
                
                interp_losses = self._compute_interpolator_losses(
                    interp_frames, exo_video, ego_video)
                
                interp_loss = interp_losses['recon']
                
                if self.use_perceptual_loss:
                    interp_loss = interp_loss + 0.1 * interp_losses['perceptual']
                
                if self.use_temporal_consistency and 'temporal' in interp_losses:
                    interp_loss = interp_loss + 0.05 * interp_losses['temporal']
                
                loss = diff_loss + 0.5 * interp_loss
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            if self.config.training.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(list(self.model.parameters()) + list(self.interpolator.parameters()), self.config.training.grad_clip)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            self.global_step += 1
            
            log_dict = {'loss': loss.item(), 'diff': diff_loss.item(), 'interp': interp_loss.item()}
            if self.use_perceptual_loss:
                log_dict['perc'] = interp_losses['perceptual'].item()
            if self.use_temporal_consistency:
                log_dict['temp'] = interp_losses['temporal'].item()
            pbar.set_postfix(log_dict)
            
            if self.global_step % self.config.training.log_interval == 0:
                self.loss_history.append(loss.item())
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def _validate(self, epoch: int) -> float:
        self.model.eval()
        self.interpolator.eval()
        
        total_loss = 0.0
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            exo_video = batch["exo_video"].to(self.device)
            ego_video = batch["ego_video"].to(self.device)
            exo_pose = batch["exo_pose"].to(self.device)
            ego_pose = batch["ego_pose"].to(self.device)
            
            full_sequence, interp_frames = self.interpolator(exo_video, ego_video)
            
            pose_cond = torch.cat([exo_pose[:, -1:], ego_pose[:, :1]], dim=1)
            
            train_sequence = torch.cat([exo_video[:, :, -4:], interp_frames, ego_video[:, :, :4]], dim=2)
            
            loss = self._compute_diffusion_loss(train_sequence, pose_cond, 0.0)
            
            total_loss += loss.item()
        
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
        
        full_sequence, interp_frames = self.interpolator(exo_video, ego_video)
        
        pose_cond = torch.cat([exo_pose[:, -1:], ego_pose[:, :1]], dim=1)
        
        B, C, T, H, W = exo_video.shape
        gen_sequence = self._sample_diffusion(
            exo_video[:, :, -4:],
            pose_cond,
            num_gen_frames=interp_frames.shape[2] + 4,
        )
        
        grid = self._create_grid(exo_video, interp_frames, ego_video, gen_sequence)
        
        vutils.save_image(
            grid,
            os.path.join(self.config.output_dir, "samples", f"epoch_{epoch}.png"),
            nrow=8,
            normalize=True,
            range=(-1, 1),
        )
    
    @torch.no_grad()
    def _sample_diffusion(
        self,
        history_frames: torch.Tensor,
        pose_cond: torch.Tensor,
        num_gen_frames: int,
    ) -> torch.Tensor:
        B, C, T_hist, H, W = history_frames.shape
        
        x = torch.randn(B, C, T_hist + num_gen_frames, H, W, device=self.device)
        x[:, :, :T_hist] = history_frames
        
        for t in reversed(range(self.diffusion.num_timesteps)):
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
        
        x = torch.tanh(x)
        
        return x
    
    def _create_grid(self, exo_video, interp_frames, ego_video, gen_sequence):
        B, C, T, H, W = exo_video.shape
        
        exo_last = exo_video[0, :, -1]
        ego_first = ego_video[0, :, 0]
        
        num_interp = interp_frames.shape[2]
        interp_list = [interp_frames[0, :, i].unsqueeze(0) for i in range(num_interp)]
        
        row1 = torch.cat([exo_last.unsqueeze(0)] + interp_list + [ego_first.unsqueeze(0)], dim=0)
        
        gen_interp_list = [gen_sequence[0, :, T+i].unsqueeze(0) for i in range(num_interp)]
        row2 = torch.cat([exo_last.unsqueeze(0)] + gen_interp_list + [ego_first.unsqueeze(0)], dim=0)
        
        grid = torch.cat([row1, row2], dim=0)
        
        return grid
    
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
