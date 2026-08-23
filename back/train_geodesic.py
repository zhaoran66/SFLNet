import os
import sys
import argparse
import torch
import torch.optim as optim

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

from utils_fourier import load_config, KeypointLoss, MPJPE, PCK
from dataset_dexycb import build_dataloader_dexycb
from model_fourier_geodesic import Syn2SeqKeypointFourierGeodesic


class TrainerGeodesic:
    def __init__(self, cfg, gpu_id: int = 4):
        self.cfg = cfg
        
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        print(f'Model: Fourier + Geodesic Interpolation (Manifold Geometry Preserving)')
        
        self.model = Syn2SeqKeypointFourierGeodesic(cfg).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f'Total parameters: {total_params / 1e6:.2f}M')
        
        self.criterion = KeypointLoss()
        self.mpjpe_metric = MPJPE()
        self.pck_metric = PCK()
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg['training']['lr'],
            weight_decay=cfg['training']['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg['training']['num_epochs']
        )
        
        self.train_loader = build_dataloader_dexycb(cfg, split='train')
        self.val_loader = build_dataloader_dexycb(cfg, split='val')
        
        print(f'Train dataset: {len(self.train_loader.dataset)} sequences')
        print(f'Val dataset: {len(self.val_loader.dataset)} sequences')
        
        self.writer = SummaryWriter(log_dir='logs/geodesic') if HAS_TENSORBOARD else None
        
        self.best_val_loss = float('inf')
        self.best_mpjpe = float('inf')
        self.best_pck = 0.0
        
        self.save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints', 'geodesic')
        os.makedirs(self.save_dir, exist_ok=True)
        
    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        
        total_loss = 0.0
        total_mpjpe = 0.0
        total_pck = 0.0
        total_mse = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Train Epoch {epoch}')
        num_batches = 0
        
        for batch_idx, batch in enumerate(pbar):
            input_data = batch['exo_video'].to(self.device)
            gt_keypoints = batch['ego_keypoints'].to(self.device)
            
            self.optimizer.zero_grad()
            
            pred_keypoints = self.model(input_data)
            
            loss_dict = self.criterion(pred_keypoints, gt_keypoints)
            loss = loss_dict['loss']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            mpjpe = self.mpjpe_metric(pred_keypoints, gt_keypoints)
            pck = self.pck_metric(pred_keypoints, gt_keypoints)
            
            total_loss += loss.item()
            total_mpjpe += mpjpe.item()
            total_pck += pck.item()
            total_mse += loss_dict['mse'].item()
            num_batches += 1
            
            if batch_idx % self.cfg['training'].get('log_interval', 10) == 0 and self.writer is not None:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/batch_loss', loss.item(), global_step)
                self.writer.add_scalar('Train/batch_mpjpe', mpjpe.item(), global_step)
                self.writer.add_scalar('Train/batch_pck', pck.item(), global_step)
            
            if HAS_TQDM:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'mpjpe': f'{mpjpe.item():.2f}mm',
                    'pck': f'{pck.item():.2f}%'
                })
        
        return {
            'loss': total_loss / num_batches,
            'mpjpe': total_mpjpe / num_batches,
            'pck': total_pck / num_batches,
            'mse': total_mse / num_batches
        }
    
    def validate(self, epoch: int) -> dict:
        self.model.eval()
        
        total_loss = 0.0
        total_mpjpe = 0.0
        total_pck = 0.0
        total_mse = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Val Epoch {epoch}')
            
            for batch_idx, batch in enumerate(pbar):
                input_data = batch['exo_video'].to(self.device)
                gt_keypoints = batch['ego_keypoints'].to(self.device)
                
                pred_keypoints = self.model(input_data)
                
                loss_dict = self.criterion(pred_keypoints, gt_keypoints)
                
                mpjpe = self.mpjpe_metric(pred_keypoints, gt_keypoints)
                pck = self.pck_metric(pred_keypoints, gt_keypoints)
                
                total_loss += loss_dict['loss'].item()
                total_mpjpe += mpjpe.item()
                total_pck += pck.item()
                total_mse += loss_dict['mse'].item()
                num_batches += 1
                
                if HAS_TQDM:
                    pbar.set_postfix({
                        'loss': f'{loss_dict["loss"].item():.4f}',
                        'mpjpe': f'{mpjpe.item():.2f}mm',
                        'pck': f'{pck.item():.2f}%'
                    })
        
        return {
            'loss': total_loss / num_batches,
            'mpjpe': total_mpjpe / num_batches,
            'pck': total_pck / num_batches,
            'mse': total_mse / num_batches
        }
    
    def train(self):
        print('\n' + '='*60)
        print('STARTING TRAINING: Fourier + Geodesic Interpolation')
        print('='*60)
        print(f'  Fourier threshold: {self.cfg["dataset"]["fourier_threshold"]}')
        print(f'  Geodesic correction scale: 0.1 (preserve manifold geometry)')
        print(f'  Num epochs: {self.cfg["training"]["num_epochs"]}')
        print(f'  Batch size: {self.cfg["training"]["batch_size"]}')
        print('='*60 + '\n')
        
        for epoch in range(1, self.cfg['training']['num_epochs'] + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)
            
            self.scheduler.step()
            
            if self.writer is not None:
                self.writer.add_scalar('Epoch/train_loss', train_metrics['loss'], epoch)
                self.writer.add_scalar('Epoch/val_loss', val_metrics['loss'], epoch)
                self.writer.add_scalar('Epoch/train_mpjpe', train_metrics['mpjpe'], epoch)
                self.writer.add_scalar('Epoch/val_mpjpe', val_metrics['mpjpe'], epoch)
                self.writer.add_scalar('Epoch/train_pck', train_metrics['pck'], epoch)
                self.writer.add_scalar('Epoch/val_pck', val_metrics['pck'], epoch)
            
            print(f'\nEpoch {epoch}/{self.cfg["training"]["num_epochs"]}')
            print(f'Train - Loss: {train_metrics["loss"]:.4f}, '
                  f'MPJPE: {train_metrics["mpjpe"]:.2f}mm, '
                  f'PCK: {train_metrics["pck"]:.2f}%')
            print(f'Val   - Loss: {val_metrics["loss"]:.4f}, '
                  f'MPJPE: {val_metrics["mpjpe"]:.2f}mm, '
                  f'PCK: {val_metrics["pck"]:.2f}%')
            
            is_best = False
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                is_best = True
            
            if val_metrics['mpjpe'] < self.best_mpjpe:
                self.best_mpjpe = val_metrics['mpjpe']
                is_best = True
            
            if val_metrics['pck'] > self.best_pck:
                self.best_pck = val_metrics['pck']
                is_best = True
            
            if is_best:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'val_mpjpe': val_metrics['mpjpe'],
                    'val_pck': val_metrics['pck']
                }, 'checkpoints/geodesic/best_model.pth')
                print(f'* Best model saved! Best MPJPE: {self.best_mpjpe:.2f}mm, Best PCK: {self.best_pck:.2f}%')
            
            if epoch % self.cfg['training'].get('save_interval', 5) == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, f'checkpoints/geodesic/epoch_{epoch}.pth')
            
            print(f'Best so far - MPJPE: {self.best_mpjpe:.2f}mm, PCK: {self.best_pck:.2f}%')
            print('-' * 60)
        
        print('\n' + '='*60)
        print('TRAINING COMPLETED!')
        print(f'Final Best - MPJPE: {self.best_mpjpe:.2f}mm, PCK: {self.best_pck:.2f}%')
        print('='*60)


def main():
    parser = argparse.ArgumentParser(description='Train with Fourier + Geodesic Interpolation')
    parser.add_argument('--config', type=str, default='config_fourier.yaml',
                        help='Path to config file')
    parser.add_argument('--gpu', type=int, default=4,
                        help='GPU ID to use (default: 4)')
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    trainer = TrainerGeodesic(cfg, gpu_id=args.gpu)
    trainer.train()


if __name__ == '__main__':
    main()
