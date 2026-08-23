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

from utils.config import load_config
from utils.losses import KeypointLoss, MPJPE, PCK
from datasets.dexycb_mv import build_dataloader
from models.syn2seq import Syn2SeqKeypoint


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.training.device if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        
        self.model = Syn2SeqKeypoint(cfg).to(self.device)
        
        self.criterion = KeypointLoss(intermediate_weight=0.5)
        self.mpjpe_metric = MPJPE()
        self.pck_metric = PCK()
        
        self.optimizer = optim.Adam(
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
        
    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        
        total_loss = 0.0
        total_mpjpe = 0.0
        total_pck = 0.0
        total_mse = 0.0
        total_interp_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Train Epoch {epoch}')
        num_batches = 0
        
        for batch_idx, batch in enumerate(pbar):
            exo_video = batch['exo_video'].to(self.device)
            ego_video = batch['ego_video'].to(self.device) if 'ego_video' in batch else None
            gt_keypoints = batch['ego_keypoints'].to(self.device)
            
            self.optimizer.zero_grad()
            
            final_kp, intermediate_kp, intermediate_pseudo_gt, _, vt_loss = self.model(exo_video, ego_video)
            
            loss_dict = self.criterion(
                final_kp, 
                gt_keypoints, 
                intermediate_kp, 
                intermediate_pseudo_gt
            )
            loss = loss_dict['loss']
            if vt_loss is not None:
                loss = loss + 0.3 * vt_loss
            
            loss.backward()
            self.optimizer.step()
            
            mpjpe = self.mpjpe_metric(final_kp, gt_keypoints)
            pck = self.pck_metric(final_kp, gt_keypoints)
            
            total_loss += loss.item()
            total_mpjpe += mpjpe.item()
            total_pck += pck.item()
            total_mse += loss_dict['mse'].item()
            total_interp_loss += loss_dict['interp_loss'].item()
            num_batches += 1
            
            if HAS_TQDM:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'mpjpe': f'{mpjpe.item():.2f}mm',
                    'pck': f'{pck.item():.2f}%',
                    'interp': f'{loss_dict["interp_loss"].item():.4f}'
                })
        
        avg_loss = total_loss / num_batches
        avg_mpjpe = total_mpjpe / num_batches
        avg_pck = total_pck / num_batches
        avg_mse = total_mse / num_batches
        avg_interp_loss = total_interp_loss / num_batches
        
        return {
            'loss': avg_loss,
            'mpjpe': avg_mpjpe,
            'pck': avg_pck,
            'mse': avg_mse,
            'interp_loss': avg_interp_loss
        }
    
    def val_epoch(self, epoch: int) -> dict:
        self.model.eval()
        
        total_loss = 0.0
        total_mpjpe = 0.0
        total_pck = 0.0
        total_mse = 0.0
        total_interp_loss = 0.0
        
        pbar = tqdm(self.val_loader, desc=f'Val Epoch {epoch}')
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                exo_video = batch['exo_video'].to(self.device)
                ego_video = batch['ego_video'].to(self.device) if 'ego_video' in batch else None
                gt_keypoints = batch['ego_keypoints'].to(self.device)
                
                final_kp, intermediate_kp, intermediate_pseudo_gt, _, _ = self.model(exo_video, ego_video)
                
                loss_dict = self.criterion(
                    final_kp, 
                    gt_keypoints, 
                    intermediate_kp, 
                    intermediate_pseudo_gt
                )
                
                mpjpe = self.mpjpe_metric(final_kp, gt_keypoints)
                pck = self.pck_metric(final_kp, gt_keypoints)
                
                total_loss += loss_dict['loss'].item()
                total_mpjpe += mpjpe.item()
                total_pck += pck.item()
                total_mse += loss_dict['mse'].item()
                total_interp_loss += loss_dict['interp_loss'].item()
                num_batches += 1
                
                if HAS_TQDM:
                    pbar.set_postfix({
                        'loss': f'{loss_dict["loss"].item():.4f}',
                        'mpjpe': f'{mpjpe.item():.2f}mm',
                        'pck': f'{pck.item():.2f}%',
                        'interp': f'{loss_dict["interp_loss"].item():.4f}'
                    })
        
        avg_loss = total_loss / num_batches
        avg_mpjpe = total_mpjpe / num_batches
        avg_pck = total_pck / num_batches
        avg_mse = total_mse / num_batches
        avg_interp_loss = total_interp_loss / num_batches
        
        return {
            'loss': avg_loss,
            'mpjpe': avg_mpjpe,
            'pck': avg_pck,
            'mse': avg_mse,
            'interp_loss': avg_interp_loss
        }
    
    def train(self):
        for epoch in range(1, self.cfg.training.num_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.val_epoch(epoch)
            
            self.scheduler.step()
            
            print(f'\nEpoch {epoch}/{self.cfg.training.num_epochs}')
            print(f'Train - Loss: {train_metrics["loss"]:.4f}, MPJPE: {train_metrics["mpjpe"]:.2f}mm, PCK: {train_metrics["pck"]:.2f}%, Interp: {train_metrics["interp_loss"]:.4f}')
            print(f'Val   - Loss: {val_metrics["loss"]:.4f}, MPJPE: {val_metrics["mpjpe"]:.2f}mm, PCK: {val_metrics["pck"]:.2f}%, Interp: {val_metrics["interp_loss"]:.4f}')
            
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint(epoch, 'best_model.pth', val_metrics)
            
            if epoch % self.cfg.training.save_interval == 0:
                self.save_checkpoint(epoch, f'epoch_{epoch}.pth', val_metrics)
            
            if self.writer is not None:
                self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
                self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
                self.writer.add_scalar('MPJPE/train', train_metrics['mpjpe'], epoch)
                self.writer.add_scalar('MPJPE/val', val_metrics['mpjpe'], epoch)
                self.writer.add_scalar('PCK/train', train_metrics['pck'], epoch)
                self.writer.add_scalar('PCK/val', val_metrics['pck'], epoch)
                self.writer.add_scalar('Loss/interp_train', train_metrics['interp_loss'], epoch)
                self.writer.add_scalar('Loss/interp_val', val_metrics['interp_loss'], epoch)
        
        print('\nTraining completed!')
        
    def save_checkpoint(self, epoch: int, filename: str, metrics: dict):
        os.makedirs('checkpoints', exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_val_loss,
            'metrics': metrics
        }
        torch.save(checkpoint, os.path.join('checkpoints', filename))


def main():
    parser = argparse.ArgumentParser(description='Train Syn2Seq Keypoint Model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == '__main__':
    main()
