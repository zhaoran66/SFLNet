# -*- coding: utf-8 -*-
import os
import sys
import argparse
import torch
import torch.optim as optim
import yaml

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

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'

sys.path.insert(0, '/data/data5/zhaoran/paper_code/back')
sys.path.insert(0, '/data/data5/zhaoran/paper_code/total')

from latent_bone_guided import LatentHandPoseModel, KeypointLoss
from utils_fourier import MPJPE, PCK
from dataset_dexycb import DexYCBMultiViewReal
from torch.utils.data import DataLoader


class LatentTrainer:
    def __init__(self, cfg, gpu_id: int = 0):
        self.cfg = cfg
        
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        
        embed_dim = cfg['model'].get('hidden_dim', 256)
        num_interpolation_steps = cfg['model'].get('interpolate_steps', 4)
        num_joints = cfg['dataset'].get('num_joints', 21)
        num_views = len(cfg['dataset'].get('exo_views', [0, 1, 2, 3, 4]))
        
        use_fourier = cfg['dataset'].get('use_fourier', True)
        fourier_threshold = cfg['dataset'].get('fourier_threshold', 0.1)
        
        print(f'Model: embed_dim={embed_dim}, steps={num_interpolation_steps}, joints={num_joints}, views={num_views}')
        print(f'Fourier: use={use_fourier}, threshold={fourier_threshold}')
        print('Architecture: Fourier + CNN + SLERP Interpolation + Pose Guidance + Ego Anchor')
        
        self.model = LatentHandPoseModel(
            embed_dim=embed_dim,
            num_interpolation_steps=num_interpolation_steps,
            num_joints=num_joints,
            num_views=num_views,
            use_fourier=use_fourier,
            fourier_threshold=fourier_threshold
        )
        self.model = self.model.to(self.device)
        
        self.criterion = KeypointLoss()
        
        self.mpjpe_metric = MPJPE()
        self.pck_metric = PCK()
        
        lr = cfg['training'].get('lr', 1e-4)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=cfg['training'].get('weight_decay', 1e-4)
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg['training']['num_epochs']
        )
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=True, init_scale=1024)
        
        self._setup_dataloaders()
        
        log_dir = f'logs/latent_{gpu_id}'
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir) if HAS_TENSORBOARD else None
        
        self.ckpt_dir = 'checkpoints'
        os.makedirs(self.ckpt_dir, exist_ok=True)
        
        self.best_mpjpe = float('inf')
        self.best_pck = 0.0
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Total params: {total_params/1e6:.2f}M, Trainable: {trainable_params/1e6:.2f}M')
        
        torch.cuda.empty_cache()
        
    def _setup_dataloaders(self):
        print('Setting up dataloaders...')
        
        train_dataset = DexYCBMultiViewReal(self.cfg, split='train')
        val_dataset = DexYCBMultiViewReal(self.cfg, split='val')
        
        bs = self.cfg['training']['batch_size']
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=bs,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=bs,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        print(f'Train: {len(train_dataset)} seqs, {len(self.train_loader)} batches')
        print(f'Val: {len(val_dataset)} seqs, {len(self.val_loader)} batches')
        
    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        
        total_loss = 0.0
        total_mpjpe = 0.0
        total_pck = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Train Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            exo_video = batch['exo_video'].to(self.device)
            ego_keypoints = batch['ego_keypoints'].to(self.device)
            exo_pose = batch.get('exo_pose')
            ego_pose = batch.get('ego_pose')
            if exo_pose is not None:
                exo_pose = exo_pose.to(self.device)
            if ego_pose is not None:
                ego_pose = ego_pose.to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(enabled=True):
                output = self.model(exo_video, ego_keypoints,
                                   exo_pose=exo_pose, ego_pose=ego_pose)
                if isinstance(output, tuple):
                    pred_keypoints, aux_dict = output
                    endpoint_loss = aux_dict.get('endpoint_loss', torch.tensor(0.0, device=self.device))
                else:
                    pred_keypoints = output
                    endpoint_loss = torch.tensor(0.0, device=self.device)
                loss_dict = self.criterion(pred_keypoints, ego_keypoints)
                loss = loss_dict['loss'] + endpoint_loss
            
            if not torch.isfinite(loss):
                print(f"\nSkip batch {batch_idx}: loss={loss.item()}")
                continue
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            with torch.no_grad():
                mpjpe = self.mpjpe_metric(pred_keypoints, ego_keypoints)
                pck = self.pck_metric(pred_keypoints, ego_keypoints)
                
                total_loss += loss.item()
                total_mpjpe += mpjpe.item()
                total_pck += pck.item()
                num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mpjpe': f'{mpjpe.item():.2f}',
                'pck': f'{pck.item():.2f}%',
                'ep_loss': f'{endpoint_loss.item():.4f}',
                'mem': f'{torch.cuda.memory_allocated()/1e9:.1f}GB'
            })
            
            if self.writer and batch_idx % 50 == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/loss', loss.item(), global_step)
                self.writer.add_scalar('Train/mpjpe', mpjpe.item(), global_step)
                self.writer.add_scalar('Train/pck', pck.item(), global_step)
                self.writer.add_scalar('Train/endpoint_loss', endpoint_loss.item(), global_step)
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_mpjpe = total_mpjpe / num_batches if num_batches > 0 else 0
        avg_pck = total_pck / num_batches if num_batches > 0 else 0
        
        torch.cuda.empty_cache()
        
        return {'loss': avg_loss, 'mpjpe': avg_mpjpe, 'pck': avg_pck}
    
    def validate(self, epoch: int) -> dict:
        self.model.eval()
        
        total_loss = 0.0
        total_mpjpe = 0.0
        total_pck = 0.0
        num_batches = 0
        
        pbar = tqdm(self.val_loader, desc=f'Val Epoch {epoch}')
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                exo_video = batch['exo_video'].to(self.device)
                ego_keypoints = batch['ego_keypoints'].to(self.device)
                exo_pose = batch.get('exo_pose')
                ego_pose = batch.get('ego_pose')
                if exo_pose is not None:
                    exo_pose = exo_pose.to(self.device)
                if ego_pose is not None:
                    ego_pose = ego_pose.to(self.device)
                
                with torch.cuda.amp.autocast(enabled=True):
                    pred_keypoints = self.model(exo_video, ego_keypoints,
                                                exo_pose=exo_pose, ego_pose=ego_pose)
                    loss_dict = self.criterion(pred_keypoints, ego_keypoints)
                    loss = loss_dict['loss']
                
                mpjpe = self.mpjpe_metric(pred_keypoints, ego_keypoints)
                pck = self.pck_metric(pred_keypoints, ego_keypoints)
                
                total_loss += loss.item()
                total_mpjpe += mpjpe.item()
                total_pck += pck.item()
                num_batches += 1
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'mpjpe': f'{mpjpe.item():.2f}'
                })
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_mpjpe = total_mpjpe / num_batches if num_batches > 0 else 0
        avg_pck = total_pck / num_batches if num_batches > 0 else 0
        
        if self.writer:
            self.writer.add_scalar('Val/loss', avg_loss, epoch)
            self.writer.add_scalar('Val/mpjpe', avg_mpjpe, epoch)
            self.writer.add_scalar('Val/pck', avg_pck, epoch)
        
        torch.cuda.empty_cache()
        
        return {'loss': avg_loss, 'mpjpe': avg_mpjpe, 'pck': avg_pck}
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_mpjpe': self.best_mpjpe,
            'cfg': self.cfg
        }
        
        path = os.path.join(self.ckpt_dir, 'best_latent.pth' if is_best else f'epoch_{epoch}_latent.pth')
        torch.save(checkpoint, path)
        print(f'Saved: {path}')
    
    def train(self):
        num_epochs = self.cfg['training']['num_epochs']
        
        for epoch in range(num_epochs):
            print(f'\n{"="*50}\nEpoch {epoch}/{num_epochs-1}\n{"="*50}')
            
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)
            
            self.scheduler.step()
            
            print(f'Train: loss={train_metrics["loss"]:.4f}, MPJPE={train_metrics["mpjpe"]:.2f}mm, PCK={train_metrics["pck"]:.2f}%')
            print(f'Val:   loss={val_metrics["loss"]:.4f}, MPJPE={val_metrics["mpjpe"]:.2f}mm, PCK={val_metrics["pck"]:.2f}%')
            
            if val_metrics['mpjpe'] < self.best_mpjpe and val_metrics['mpjpe'] > 0:
                self.best_mpjpe = val_metrics['mpjpe']
                self.save_checkpoint(epoch, is_best=True)
                print(f'New best MPJPE: {self.best_mpjpe:.2f}mm')
            
            if val_metrics['pck'] > self.best_pck:
                self.best_pck = val_metrics['pck']
            
            if epoch % self.cfg['training'].get('save_interval', 5) == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        if self.writer:
            self.writer.close()
        
        print('\nTraining complete!')
        print(f'Best MPJPE: {self.best_mpjpe:.2f}mm')
        print(f'Best PCK: {self.best_pck:.2f}%')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_fourier.yaml')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    print(f'Config: {cfg["name"]}')
    print(f'GPU: {args.gpu}')
    
    trainer = LatentTrainer(cfg, gpu_id=args.gpu)
    trainer.train()


if __name__ == '__main__':
    main()
