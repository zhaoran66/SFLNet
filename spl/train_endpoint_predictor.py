# -*- coding: utf-8 -*-
"""
Direction 1: Endpoint Predictor
- Train a network to predict ego_feat from exo_global + camera poses
- Architecture: exo_global + camera_extrinsics ?? Transformer/MLP ?? predicted_ego_feat

This version uses only exo information + camera poses, no ego at all.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config
from losses import KeypointLoss
from dataset import build_dataloader
from model import SPLLatentKeypoint


class EndpointPredictor(nn.Module):
    """
    Predict ego latent feature from exo features + camera extrinsics.
    
    Input:
        exo_global: [B, T, 256] - exo video features
        camera_extrinsics: [B, V, 4, 4] - camera extrinsics (optional)
    
    Output:
        predicted_ego_feat: [B, T, 256] - predicted ego latent feature
    """
    def __init__(self, hidden_dim=256, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Camera pose encoder (if extrinsics are provided)
        self.camera_encoder = nn.Sequential(
            nn.Linear(12, 64),  # 4x4 extrinsics flattened
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim)
        )
        
        # Feature aggregator
        self.feature_aggregator = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # concat exo + camera context
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Residual connection
        self.alpha = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, exo_global, camera_extrinsics=None):
        """
        Args:
            exo_global: [B, T, 256] 
            camera_extrinsics: [B, V, 4, 4] or None
        
        Returns:
            predicted_ego_feat: [B, T, 256]
        """
        B, T, H = exo_global.shape
        
        # Encode camera poses if available
        if camera_extrinsics is not None:
            # Extract rotation (R) and translation (t) from extrinsics
            R = camera_extrinsics[:, :, :3, :3]  # [B, V, 3, 3]
            t = camera_extrinsics[:, :, :3, 3]  # [B, V, 3]
            
            # Flatten rotation matrix (9 params) + translation (3 params) = 12 params
            R_flat = R.reshape(B, -1)[:, :12]  # Take first 9+3=12
            camera_feat = self.camera_encoder(R_flat)  # [B, 256]
            camera_feat = camera_feat.unsqueeze(1).expand(-1, T, -1)  # [B, T, 256]
        else:
            camera_feat = torch.zeros_like(exo_global)
        
        # Aggregate exo features across time
        exo_agg = exo_global  # Keep temporal dimension
        
        # Fuse exo + camera features
        fused = torch.cat([exo_agg, camera_feat], dim=-1)  # [B, T, 512]
        fused = fused[:, :, :self.hidden_dim * 2]  # Truncate
        
        # Predict ego feat
        predicted = self.predictor(fused)  # [B, T, 256]
        
        # Residual connection: ego ?? exo + residual
        ego_estimate = (1 - torch.sigmoid(self.alpha)) * exo_global + torch.sigmoid(self.alpha) * predicted
        
        return ego_estimate


class SPLWithEndpointPredictor(nn.Module):
    """
    SPL with Endpoint Predictor replacing learnable_ego_anchor.
    
    Training: predict ego_feat from exo + camera poses, supervised by GT ego_feat
    Inference: use predicted ego_feat as Slerp endpoint
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = cfg.model.hidden_dim
        self.num_interpolate_steps = cfg.model.interpolate_steps
        self.num_joints = cfg.dataset.num_joints
        
        # Original SPL components
        self.feature_extractor = VideoFeatureExtractor(
            hidden_dim=self.hidden_dim,
            num_views=len(cfg.dataset.exo_views)
        )
        
        # NEW: Endpoint Predictor
        self.endpoint_predictor = EndpointPredictor(
            hidden_dim=self.hidden_dim,
            num_layers=2
        )
        
        self.geodesic_interpolator = GeodesicInterpolator(
            hidden_dim=self.hidden_dim,
            num_interpolate_steps=self.num_interpolate_steps,
            use_mean_endpoint=False  # We don't need anchor anymore
        )
        
        self.seq_norm = nn.LayerNorm(self.hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=cfg.model.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=cfg.model.dropout,
            batch_first=True,
            norm_first=True
        )
        self.sequence_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.model.num_layers
        )
        
        self.keypoint_decoder = LatentSpaceKeypointDecoder(
            hidden_dim=self.hidden_dim,
            num_joints=self.num_joints
        )
        
        self.pos_encoding = nn.Parameter(torch.randn(2000, self.hidden_dim) * 0.02)
    
    def forward(self, exo_video, ego_keypoints_gt=None, return_all_steps=False, use_predicted_endpoint=True):
        batch_size, seq_len, num_views, c, h, w = exo_video.shape
        
        # Extract features
        spatial_features, exo_global = self.feature_extractor(exo_video)
        
        # Get GT ego features (for training)
        if self.training and ego_keypoints_gt is not None:
            ego_features = self.feature_extractor.encode_ego_keypoints(ego_keypoints_gt)
        else:
            ego_features = None
        
        # Predict ego endpoint from exo
        if use_predicted_endpoint:
            predicted_ego_feat = self.endpoint_predictor(exo_global)
        else:
            predicted_ego_feat = None
        
        # Use predicted endpoint during inference, GT during training
        if self.training and ego_features is not None:
            # Training: use GT ego_feat as endpoint
            full_sequence = self.geodesic_interpolator(exo_global, ego_features)
        else:
            # Inference: use predicted ego_feat
            full_sequence = self.geodesic_interpolator(exo_global, predicted_ego_feat)
        
        num_total_steps = full_sequence.shape[2]
        
        full_sequence_flat = full_sequence.view(batch_size, seq_len * num_total_steps, self.hidden_dim)
        full_sequence_flat = self.seq_norm(full_sequence_flat)
        full_sequence_flat = full_sequence_flat + self.pos_encoding[:full_sequence_flat.shape[1], :].unsqueeze(0)
        
        encoded = self.sequence_encoder(full_sequence_flat)
        encoded = encoded.view(batch_size, seq_len, num_total_steps, self.hidden_dim)
        
        all_step_keypoints = []
        for step_idx in range(num_total_steps):
            start_idx = max(0, step_idx - 1)
            end_idx = min(num_total_steps, step_idx + 2)
            context_feat = encoded[:, :, start_idx:end_idx, :].flatten(2, 3)
            step_feat = encoded[:, :, step_idx, :]
            decoder_input = torch.cat([step_feat, context_feat], dim=-1)[:, :, :self.hidden_dim]
            keypoints = self.keypoint_decoder(decoder_input, spatial_features)
            all_step_keypoints.append(keypoints)
        
        keypoints = all_step_keypoints[-1]
        
        if return_all_steps:
            return {
                'final_pose': keypoints,
                'all_steps': torch.stack(all_step_keypoints, dim=1),
                'num_steps': num_total_steps,
                'predicted_ego_feat': predicted_ego_feat,
                'gt_ego_feat': ego_features
            }
        
        return keypoints


class EndpointPredictorTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.training.device if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        
        self.model = SPLWithEndpointPredictor(cfg).to(self.device)
        
        # Main loss: keypoint prediction
        self.criterion = KeypointLoss()
        
        # NEW: Endpoint prediction loss (L2 on latent features)
        self.endpoint_loss_weight = 0.1  # Weight for endpoint prediction loss
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.training.num_epochs
        )
        
        self.train_loader = build_dataloader(cfg, split='train')
        self.val_loader = build_dataloader(cfg, split='val')
        
        self.best_val_loss = float('inf')
    
    def train_epoch(self, epoch):
        self.model.train()
        
        total_loss = 0.0
        total_kp_loss = 0.0
        total_endpoint_loss = 0.0
        n_batches = 0
        
        for batch in self.train_loader:
            exo_video = batch['exo_video'].to(self.device)
            ego_keypoints = batch['ego_keypoints'].to(self.device)
            ego_K = batch['ego_K'].to(self.device)
            ego_2d = batch['ego_keypoints_2d'].to(self.device)
            ego_wrist = batch['ego_wrist'].to(self.device)
            
            pred_dict = self.model(exo_video, ego_keypoints, return_all_steps=True)
            pred_keypoints = pred_dict['final_pose']
            
            # Keypoint loss
            kp_loss_dict = self.criterion(
                pred_dict, ego_keypoints,
                K=ego_K, gt_2d=ego_2d, wrist=ego_wrist, epoch=epoch
            )
            kp_loss = kp_loss_dict['loss']
            
            # Endpoint prediction loss (NEW)
            if 'predicted_ego_feat' in pred_dict and 'gt_ego_feat' in pred_dict:
                endpoint_loss = F.mse_loss(
                    pred_dict['predicted_ego_feat'],
                    pred_dict['gt_ego_feat']
                )
            else:
                endpoint_loss = torch.tensor(0.0, device=self.device)
            
            # Total loss
            loss = kp_loss + self.endpoint_loss_weight * endpoint_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            
            if torch.isnan(loss):
                print(f'[WARNING] NaN loss at batch {n_batches}')
                continue
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_kp_loss += kp_loss.item()
            total_endpoint_loss += endpoint_loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        avg_kp = total_kp_loss / n_batches
        avg_ep = total_endpoint_loss / n_batches
        
        return {'loss': avg_loss, 'kp_loss': avg_kp, 'endpoint_loss': avg_ep}
    
    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        
        from losses import MPJPE, PCK
        
        total_loss = 0.0
        total_mpjpe = 0.0
        n_batches = 0
        
        mpjpe_metric = MPJPE()
        
        for batch in self.val_loader:
            exo_video = batch['exo_video'].to(self.device)
            ego_keypoints = batch['ego_keypoints'].to(self.device)
            ego_K = batch['ego_K'].to(self.device)
            ego_2d = batch['ego_keypoints_2d'].to(self.device)
            ego_wrist = batch['ego_wrist'].to(self.device)
            
            pred_dict = self.model(exo_video, ego_keypoints, return_all_steps=True, use_predicted_endpoint=False)
            pred_keypoints = pred_dict['final_pose']
            
            # Wrist-centering
            pred_wrist = pred_keypoints[..., 0:1, :]
            pred_centered = pred_keypoints - pred_wrist
            gt_centered = ego_keypoints
            
            kp_loss_dict = self.criterion(
                pred_dict, ego_keypoints,
                K=ego_K, gt_2d=ego_2d, wrist=ego_wrist, epoch=epoch
            )
            loss = kp_loss_dict['loss']
            
            mpjpe = mpjpe_metric(pred_centered, gt_centered)
            
            total_loss += loss.item()
            total_mpjpe += mpjpe.item()
            n_batches += 1
        
        return {'loss': total_loss / n_batches, 'mpjpe': total_mpjpe / n_batches}
    
    def save_checkpoint(self, epoch, is_best=False):
        save_dir = '/data/data5/zhaoran/paper_code/spl/check'
        os.makedirs(save_dir, exist_ok=True)
        
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss
        }
        
        torch.save(ckpt, os.path.join(save_dir, f'endpoint_predictor_epoch_{epoch}.pth'))
        
        if is_best:
            torch.save(ckpt, os.path.join(save_dir, 'endpoint_predictor_best.pth'))
            print(f'[!] Saved best model at epoch {epoch}')
    
    def train(self):
        print('\n' + '=' * 60)
        print('TRAINING SPL WITH ENDPOINT PREDICTOR')
        print('=' * 60)
        print(f'Endpoint prediction loss weight: {self.endpoint_loss_weight}')
        print(f'Camera poses: NOT USED (v1 - add in v2)')
        print('=' * 60)
        
        for epoch in range(1, self.cfg.training.num_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)
            
            self.scheduler.step()
            
            print(f'\nEpoch {epoch}/{self.cfg.training.num_epochs}')
            print(f'  Train Loss: {train_metrics["loss"]:.4f} | KP: {train_metrics["kp_loss"]:.4f} | EP: {train_metrics["endpoint_loss"]:.4f}')
            print(f'  Val Loss:   {val_metrics["loss"]:.4f} | MPJPE: {val_metrics["mpjpe"]:.2f} mm')
            
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            
            if epoch % self.cfg.training.save_interval == 0 or is_best:
                self.save_checkpoint(epoch, is_best)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    if args.epochs:
        cfg.training.num_epochs = args.epochs
    
    trainer = EndpointPredictorTrainer(cfg)
    trainer.train()


if __name__ == '__main__':
    main()
