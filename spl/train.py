# -*- coding: utf-8 -*-
import os
import sys
import argparse
import torch
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    class tqdm:
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable
        def __iter__(self):
            return iter(self.iterable)
        def set_postfix(self, **kwargs):
            pass

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None

from config import load_config
from losses import KeypointLoss, MPJPE, PCK
from dataset import build_dataloader
from model import SPLLatentKeypoint


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.training.device if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        
        self.model = SPLLatentKeypoint(cfg).to(self.device)
        
        self.criterion = KeypointLoss()
        self.mpjpe_metric = MPJPE()
        self.pck_metric = PCK()
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.training.num_epochs
        )
        
        self.train_loader = build_dataloader(cfg, split='train')
        self.val_loader = build_dataloader(cfg, split='val')
        
        self.writer = SummaryWriter(log_dir='logs') if HAS_TENSORBOARD else None
        
        self.best_val_loss = float('inf')
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_mpjpe': [],
            'val_mpjpe': [],
            'train_pck': [],
            'val_pck': [],
            'train_mse': [],
            'val_mse': []
        }
        
    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        
        total_loss = 0.0
        total_mpjpe = 0.0
        total_pck = 0.0
        total_mse = 0.0
        total_proj = 0.0
        n_batches = 0
        
        for batch in tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]'):
            exo_video = batch['exo_video'].to(self.device)
            ego_keypoints = batch['ego_keypoints'].to(self.device)
            ego_K = batch['ego_K'].to(self.device)
            ego_2d = batch['ego_keypoints_2d'].to(self.device)
            ego_wrist = batch['ego_wrist'].to(self.device)
            
            pred_dict = self.model(exo_video, ego_keypoints, return_all_steps=True)
            pred_keypoints = pred_dict['final_pose']
            
            loss_dict = self.criterion(
                pred_dict, ego_keypoints,
                K=ego_K, gt_2d=ego_2d, wrist=ego_wrist, epoch=epoch
            )
            loss = loss_dict['loss']
            
            #  Endpoint Predictor ?
            if 'endpoint_loss' in pred_dict:
                loss = loss + pred_dict['endpoint_loss']
            
            self.optimizer.zero_grad()
            loss.backward()
            
            if torch.isnan(loss):
                print(f'[WARNING] NaN loss detected at batch {n_batches}, skipping!')
                continue
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_mpjpe += self.mpjpe_metric(pred_keypoints, ego_keypoints).item()
            total_pck += self.pck_metric(pred_keypoints, ego_keypoints).item()
            total_mse += loss_dict['mse'].item()
            total_proj += loss_dict['proj'].item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        avg_mpjpe = total_mpjpe / n_batches
        avg_pck = total_pck / n_batches
        avg_mse = total_mse / n_batches
        avg_proj = total_proj / n_batches
        
        return {
            'loss': avg_loss,
            'mpjpe': avg_mpjpe,
            'pck': avg_pck,
            'mse': avg_mse,
            'proj': avg_proj
        }
    
    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        self.model.eval()
        
        total_loss = 0.0
        total_mpjpe = 0.0
        total_pck = 0.0
        total_mse = 0.0
        total_proj = 0.0
        n_batches = 0
        
        for batch in tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]'):
            exo_video = batch['exo_video'].to(self.device)
            ego_keypoints = batch['ego_keypoints'].to(self.device)
            ego_K = batch['ego_K'].to(self.device)
            ego_2d = batch['ego_keypoints_2d'].to(self.device)
            ego_wrist = batch['ego_wrist'].to(self.device)
            
            pred_dict = self.model(exo_video, ego_keypoints, return_all_steps=True)
            pred_keypoints = pred_dict['final_pose']
            
            loss_dict = self.criterion(
                pred_dict, ego_keypoints,
                K=ego_K, gt_2d=ego_2d, wrist=ego_wrist, epoch=epoch
            )
            loss = loss_dict['loss']
            
            total_loss += loss.item()
            total_mpjpe += self.mpjpe_metric(pred_keypoints, ego_keypoints).item()
            total_pck += self.pck_metric(pred_keypoints, ego_keypoints).item()
            total_mse += loss_dict['mse'].item()
            total_proj += loss_dict['proj'].item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        avg_mpjpe = total_mpjpe / n_batches
        avg_pck = total_pck / n_batches
        avg_mse = total_mse / n_batches
        avg_proj = total_proj / n_batches
        
        return {
            'loss': avg_loss,
            'mpjpe': avg_mpjpe,
            'pck': avg_pck,
            'mse': avg_mse,
            'proj': avg_proj
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': self.metrics
        }
        
        save_dir = '/data/data5/zhaoran/paper_code/spl/check'
        os.makedirs(save_dir, exist_ok=True)
        
        epoch_path = os.path.join(save_dir, f'epoch_{epoch}.pth')
        torch.save(checkpoint, epoch_path)
        
        if is_best:
            best_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f'Saved best model at epoch {epoch}')
    
    def compute_inference_endpoint(self):
        print("\nComputing mean ego features for inference endpoint...")
        self.model.compute_mean_ego_features(self.train_loader, self.device)
        print("Mean ego features computed successfully!")
    
    def train(self):
        for epoch in range(1, self.cfg.training.num_epochs + 1):
            print(f'\nEpoch {epoch}/{self.cfg.training.num_epochs}')
            
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)
            
            self.scheduler.step()
            
            self.metrics['train_loss'].append(train_metrics['loss'])
            self.metrics['val_loss'].append(val_metrics['loss'])
            self.metrics['train_mpjpe'].append(train_metrics['mpjpe'])
            self.metrics['val_mpjpe'].append(val_metrics['mpjpe'])
            self.metrics['train_pck'].append(train_metrics['pck'])
            self.metrics['val_pck'].append(val_metrics['pck'])
            self.metrics['train_mse'].append(train_metrics['mse'])
            self.metrics['val_mse'].append(val_metrics['mse'])
            
            if self.writer:
                self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
                self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
                self.writer.add_scalar('MPJPE/train', train_metrics['mpjpe'], epoch)
                self.writer.add_scalar('MPJPE/val', val_metrics['mpjpe'], epoch)
                self.writer.add_scalar('PCK/train', train_metrics['pck'], epoch)
                self.writer.add_scalar('PCK/val', val_metrics['pck'], epoch)
                self.writer.add_scalar('MSE/train', train_metrics['mse'], epoch)
                self.writer.add_scalar('MSE/val', val_metrics['mse'], epoch)
            
            print(f'Train Loss: {train_metrics["loss"]:.4f} | MSE: {train_metrics["mse"]:.4f} | Proj: {train_metrics["proj"]:.4f} | MPJPE: {train_metrics["mpjpe"]:.2f}mm | PCK: {train_metrics["pck"]:.2f}%')
            print(f'Val Loss: {val_metrics["loss"]:.4f} | MSE: {val_metrics["mse"]:.4f} | Proj: {val_metrics["proj"]:.4f} | MPJPE: {val_metrics["mpjpe"]:.2f}mm | PCK: {val_metrics["pck"]:.2f}%')
            
            gate_val = torch.sigmoid(self.model.keypoint_decoder.query_gate).item()
            print(f'QueryGate: {gate_val:.4f} | LR: {self.scheduler.get_last_lr()[0]:.6f}')
            
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            
            if epoch % self.cfg.training.save_interval == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
        
        self.compute_inference_endpoint()
        
        if self.writer:
            self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='SPL: Latent Space Keypoint Estimation')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == '__main__':
    main()
