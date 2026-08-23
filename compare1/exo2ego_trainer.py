"""
Exo2Ego Trainer - ѵStage 1Stage 2ѵ
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
from typing import Dict, Optional, Tuple
from pathlib import Path


class Exo2EgoTrainer:
    """
    Exo2Ego׶ѵ

    Stage 1: Layout Transformer - Ԥegoӽǵֲ
    Stage 2: Diffusion Model - ڲϸegoͼ
    """

    def __init__(
        self,
        cfg: Dict,
        stage1_model: nn.Module,
        stage2_model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = 'cuda'
    ):
        self.cfg = cfg
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # ģ
        self.stage1_model = stage1_model.to(self.device)
        self.stage2_model = stage2_model.to(self.device)

        # ݼ
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Ż
        self.stage1_optimizer = optim.Adam(
            self.stage1_model.parameters(),
            lr=cfg['training']['lr'],
            weight_decay=cfg['training'].get('weight_decay', 0.0001)
        )
        self.stage2_optimizer = optim.Adam(
            self.stage2_model.parameters(),
            lr=cfg['training']['lr'],
            weight_decay=cfg['training'].get('weight_decay', 0.0001)
        )

        # ѧϰʵ
        self.stage1_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.stage1_optimizer,
            T_max=cfg['training']['num_epochs']
        )
        self.stage2_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.stage2_optimizer,
            T_max=cfg['training']['num_epochs']
        )

        # ʧ
        self.layout_loss_fn = self._create_layout_loss()
        self.diffusion_loss_fn = nn.MSELoss()

        # ѵ״̬
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Ŀ¼
        self.output_dir = Path(cfg['training'].get('output_dir', './outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)

        # ־
        self.log_interval = cfg['training'].get('log_interval', 10)
        self.save_interval = cfg['training'].get('save_interval', 5)

    def _create_layout_loss(self):
        """ʧ"""
        from model_layout_transformer import LayoutLoss
        return LayoutLoss(
            keypoint_weight=self.cfg['training'].get('keypoint_weight', 1.0),
            heatmap_weight=self.cfg['training'].get('heatmap_weight', 0.5),
            visibility_weight=self.cfg['training'].get('visibility_weight', 0.2)
        )

    def train_stage1(self, batch: Dict, train_stage1: bool = True) -> Dict:
        """
        ѵStage 1: Layout Transformer

        Args:
            batch: exo_video, ego_video, ego_keypointsȵ
            train_stage1: ǷStage 1

        Returns:
            losses: ʧֵ
        """
        exo_video = batch['exo_video'].to(self.device)  # [B, T, C, H, W]
        ego_keypoints = batch['ego_keypoints'].to(self.device)  # [B, T, num_joints, 3]

        if train_stage1:
            self.stage1_model.train()
        else:
            self.stage1_model.eval()

        with torch.set_grad_enabled(train_stage1):
            # ǰ򴫲
            pred_keypoints, pred_heatmaps, pred_visibility = self.stage1_model(exo_video)

            # ʧ
            losses = self.layout_loss_fn(
                pred_keypoints=pred_keypoints,
                pred_heatmaps=pred_heatmaps,
                pred_visibility=pred_visibility,
                gt_keypoints=ego_keypoints[..., :2]  # ֻʹxy
            )

            if train_stage1 and losses['total_loss'].requires_grad:
                self.stage1_optimizer.zero_grad()
                losses['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(self.stage1_model.parameters(), max_norm=1.0)
                self.stage1_optimizer.step()

        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}

    def train_stage2(self, batch: Dict, ego_layout: torch.Tensor, train_stage2: bool = True) -> Dict:
        """
        ѵStage 2: Diffusion Model

        Args:
            batch: ego_video
            ego_layout: Stage 1Ԥego
            train_stage2: ǷStage 2

        Returns:
            losses: ʧֵ
        """
        ego_video = batch['ego_video'].to(self.device)  # [B, T, C, H, W]
        exo_video = batch['exo_video'].to(self.device)  # [B, T, C, H, W]

        if train_stage2:
            self.stage2_model.train()
        else:
            self.stage2_model.eval()

        losses = {}
        B, T = ego_video.shape[:2]

        # ÿʱ䲽ɢѵ
        total_loss = 0.0
        num_valid = 0

        with torch.set_grad_enabled(train_stage2):
            for t_idx in range(min(T, self.cfg['model'].get('max_train_timesteps', 8))):
                # ѡʱ䲽
                t = torch.randint(0, self.stage2_model.num_timesteps, (B,), device=self.device)

                # ȡǰ֡
                ego_frame = ego_video[:, t_idx]
                ego_layout_frame = ego_layout[:, t_idx]
                exo_frame = exo_video[:, t_idx]

                # resize exo_frame to match ego_layout spatial size
                layout_h, layout_w = ego_layout_frame.shape[-2], ego_layout_frame.shape[-1]
                import torch.nn.functional as F
                exo_frame_resized = F.interpolate(exo_frame, size=(layout_h, layout_w), mode='bilinear', align_corners=False)
                # 只用exo_frame作为条件（3通道，与VAE兼容）
                condition = exo_frame_resized

                # ʧ
                loss = self.stage2_model.p_losses(ego_frame, condition, t)

                total_loss += loss.item()
                num_valid += 1

                if train_stage2:
                    self.stage2_optimizer.zero_grad()
                    loss.backward()

            if train_stage2:
                torch.nn.utils.clip_grad_norm_(self.stage2_model.parameters(), max_norm=1.0)
                self.stage2_optimizer.step()

        losses['diffusion_loss'] = total_loss / max(num_valid, 1)
        losses['total_loss'] = losses['diffusion_loss']

        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}

    def train_epoch(self, train_stage1: bool = True, train_stage2: bool = True):
        """
        ѵһepoch

        Args:
            train_stage1: ǷѵStage 1
            train_stage2: ǷѵStage 2
        """
        self.current_epoch += 1
        epoch_start_time = time.time()

        stage1_losses = []
        stage2_losses = []

        for batch_idx, batch in enumerate(self.train_loader):
            batch_time = time.time()

            # Stage 1ѵ
            if train_stage1:
                s1_losses = self.train_stage1(batch, train_stage1=True)
                stage1_losses.append(s1_losses)

            # Stage 1ԤⲼ (Stage 2)
            with torch.no_grad():
                exo_video = batch['exo_video'].to(self.device)
                pred_keypoints, pred_heatmaps, pred_visibility = self.stage1_model(exo_video)
                ego_layout = self._keypoints_to_layout(pred_keypoints)

            # Stage 2ѵ
            if train_stage2:
                s2_losses = self.train_stage2(batch, ego_layout, train_stage2=True)
                stage2_losses.append(s2_losses)

            self.global_step += 1

            # ־
            if batch_idx % self.log_interval == 0:
                log_str = f"Epoch {self.current_epoch} | Batch {batch_idx}/{len(self.train_loader)}"
                log_str += f" | Time: {time.time() - batch_time:.2f}s"

                if stage1_losses:
                    log_str += f" | S1 Loss: {stage1_losses[-1].get('total_loss', 0):.4f}"
                if stage2_losses:
                    log_str += f" | S2 Loss: {stage2_losses[-1].get('total_loss', 0):.4f}"

                print(log_str)

        # ѧϰ
        if train_stage1:
            self.stage1_scheduler.step()
        if train_stage2:
            self.stage2_scheduler.step()

        # ƽʧ
        avg_stage1 = {}
        avg_stage2 = {}
        if stage1_losses:
            for key in stage1_losses[0].keys():
                avg_stage1[key] = sum(l[key] for l in stage1_losses) / len(stage1_losses)
        if stage2_losses:
            for key in stage2_losses[0].keys():
                avg_stage2[key] = sum(l[key] for l in stage2_losses) / len(stage2_losses)

        epoch_time = time.time() - epoch_start_time
        print(f"\nEpoch {self.current_epoch} completed in {epoch_time:.2f}s")
        print(f"Stage 1 Avg Losses: {avg_stage1}")
        print(f"Stage 2 Avg Losses: {avg_stage2}")

        return avg_stage1, avg_stage2

    def _keypoints_to_layout(self, keypoints: torch.Tensor) -> torch.Tensor:
        """
        Convert keypoints to layout images

        Args:
            keypoints: [B, T, num_joints, 2] normalized coordinates

        Returns:
            layout: [B, T, 4, H, W] RGBA layout images
        """
        B, T, num_joints, _ = keypoints.shape
        H, W = 64, 64  # layout size

        layouts = []
        for b in range(B):
            batch_layouts = []
            for t in range(T):
                kp = keypoints[b, t]  # [num_joints, 2]
                # simple layout rendering - using heatmap
                heatmap = torch.zeros(num_joints, H, W, device=keypoints.device)
                for j in range(num_joints):
                    x, y = (kp[j] * torch.tensor([W, H], device=keypoints.device)).long()
                    x = x.clamp(0, W - 1)
                    y = y.clamp(0, H - 1)
                    heatmap[j, y, x] = 1.0

                # convert to [B, 4, H, W] format
                layout = torch.zeros(4, H, W, device=keypoints.device)
                layout[:3] = heatmap.sum(dim=0, keepdim=True).expand(3, -1, -1)
                layout[3] = (layout[:3].sum(dim=0) > 0).float()
                batch_layouts.append(layout)

            layouts.append(torch.stack(batch_layouts, dim=0))

        return torch.stack(layouts, dim=0)

    def validate(self) -> Dict:
        """֤"""
        if self.val_loader is None:
            return {}

        self.stage1_model.eval()
        self.stage2_model.eval()

        val_losses = []

        with torch.no_grad():
            for batch in self.val_loader:
                s1_losses = self.train_stage1(batch, train_stage1=False)
                val_losses.append(s1_losses)

        avg_val_loss = sum(l['total_loss'] for l in val_losses) / len(val_losses)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        return {'val_loss': avg_val_loss}

    def save_checkpoint(self, filename: str, include_optimizer: bool = True):
        """"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'stage1_model_state_dict': self.stage1_model.state_dict(),
            'stage2_model_state_dict': self.stage2_model.state_dict(),
        }

        if include_optimizer:
            checkpoint['stage1_optimizer_state_dict'] = self.stage1_optimizer.state_dict()
            checkpoint['stage2_optimizer_state_dict'] = self.stage2_optimizer.state_dict()
            checkpoint['stage1_scheduler_state_dict'] = self.stage1_scheduler.state_dict()
            checkpoint['stage2_scheduler_state_dict'] = self.stage2_scheduler.state_dict()

        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """ؼ"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.stage1_model.load_state_dict(checkpoint['stage1_model_state_dict'])
        self.stage2_model.load_state_dict(checkpoint['stage2_model_state_dict'])

        if 'stage1_optimizer_state_dict' in checkpoint:
            self.stage1_optimizer.load_state_dict(checkpoint['stage1_optimizer_state_dict'])
            self.stage2_optimizer.load_state_dict(checkpoint['stage2_optimizer_state_dict'])
            self.stage1_scheduler.load_state_dict(checkpoint['stage1_scheduler_state_dict'])
            self.stage2_scheduler.load_state_dict(checkpoint['stage2_scheduler_state_dict'])

        print(f"Checkpoint loaded from {checkpoint_path}")

    def train(self, num_epochs: int, train_stage1: bool = True, train_stage2: bool = True):
        """
        ѵ

        Args:
            num_epochs: ѵepoch
            train_stage1: ǷѵStage 1
            train_stage2: ǷѵStage 2
        """
        print(f"Starting training for {num_epochs} epochs")
        print(f"Stage 1: {'Enabled' if train_stage1 else 'Disabled'}")
        print(f"Stage 2: {'Enabled' if train_stage2 else 'Disabled'}")
        print(f"Device: {self.device}")

        for epoch in range(num_epochs):
            # ѵ
            self.train_epoch(train_stage1=train_stage1, train_stage2=train_stage2)

            # ֤
            if self.val_loader is not None and epoch % self.save_interval == 0:
                val_metrics = self.validate()

            # 
            if epoch % self.save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pth")

        print("Training completed!")


# ============================================================================
# Stageѵ
# ============================================================================

class Stage1Trainer:
    """Stage 1 (Layout Transformer) ѵ"""

    def __init__(self, cfg, model, train_loader, val_loader=None, device='cuda'):
        self.cfg = cfg
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = optim.Adam(model.parameters(), lr=cfg['training']['lr'])
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg['training']['num_epochs'])
        self.loss_fn = LayoutLoss()

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0

            for batch in self.train_loader:
                exo_video = batch['exo_video'].to(self.device)
                ego_keypoints = batch['ego_keypoints'].to(self.device)

                pred_kp, pred_hm, pred_vis = self.model(exo_video)

                losses = self.loss_fn(
                    pred_keypoints=pred_kp,
                    pred_heatmaps=pred_hm,
                    pred_visibility=pred_vis,
                    gt_keypoints=ego_keypoints[..., :2]
                )

                self.optimizer.zero_grad()
                losses['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += losses['total_loss'].item()

            self.scheduler.step()
            print(f"Epoch {epoch}: Loss = {total_loss / len(self.train_loader):.4f}")


class Stage2Trainer:
    """Stage 2 (Diffusion Model) ѵ"""

    def __init__(self, cfg, model, train_loader, val_loader=None, device='cuda'):
        self.cfg = cfg
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = optim.Adam(model.parameters(), lr=cfg['training']['lr'])
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg['training']['num_epochs'])

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0

            for batch in self.train_loader:
                ego_video = batch['ego_video'].to(self.device)
                ego_layout = batch.get('ego_layout', torch.randn_like(ego_video[:, :, :4]))
                ego_layout = ego_layout.to(self.device)

                B, T = ego_video.shape[:2]

                for t_idx in range(min(T, 4)):
                    t = torch.randint(0, self.model.num_timesteps, (B,), device=self.device)
                    loss = self.model.p_losses(ego_video[:, t_idx], ego_layout[:, t_idx], t)

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                    total_loss += loss.item()

            self.scheduler.step()
            print(f"Epoch {epoch}: Loss = {total_loss / len(self.train_loader):.4f}")