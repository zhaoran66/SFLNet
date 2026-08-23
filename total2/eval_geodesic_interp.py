# -*- coding: utf-8 -*-
import os
import sys
import argparse
import torch
import yaml
import numpy as np

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

sys.path.insert(0, '/data/data5/zhaoran/paper_code/total')
sys.path.insert(0, '/data/data5/zhaoran/paper_code/back')

from latent_hand_aware_v2 import HandAwareLatentModel
from utils_fourier import MPJPE, PCK
from dataset_dexycb import DexYCBMultiViewReal
from torch.utils.data import DataLoader


class GeodesicEvaluator:
    def __init__(self, cfg, checkpoint_path, gpu_id: int = 0):
        self.cfg = cfg
        
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        
        embed_dim = cfg['model'].get('hidden_dim', 256)
        num_interpolation_steps = cfg['model'].get('interpolate_steps', 4)
        num_joints = cfg['dataset'].get('num_joints', 21)
        
        print(f'Model: embed_dim={embed_dim}, steps={num_interpolation_steps}, joints={num_joints}')
        
        self.model = HandAwareLatentModel(
            embed_dim=embed_dim,
            num_interpolation_steps=num_interpolation_steps,
            num_joints=num_joints
        )
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f'Loaded checkpoint from {checkpoint_path}')
            print(f'Trained epoch: {checkpoint.get("epoch", "unknown")}')
        else:
            print(f'WARNING: Checkpoint not found: {checkpoint_path}')
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.mpjpe_metric = MPJPE()
        self.pck_metric = PCK()
        
        val_dataset = DexYCBMultiViewReal(self.cfg, split='val')
        self.val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2)
        print(f'Val: {len(val_dataset)} sequences')
        
    def _process_batch(self, batch):
        exo_video = batch['exo_video']
        ego_keypoints = batch['ego_keypoints']
        
        batch_size, seq_len, num_views, c, h, w = exo_video.shape
        
        ego_view_idx = 0
        single_view_video = exo_video[:, :, ego_view_idx]
        flat_images = single_view_video.reshape(batch_size * seq_len, c, h, w)
        
        if h != 224 or w != 224:
            flat_images = torch.nn.functional.interpolate(
                flat_images, size=(224, 224), mode='bilinear', align_corners=False
            )
        
        flat_keypoints = ego_keypoints.reshape(batch_size * seq_len, -1, 3)
        
        return flat_images, flat_keypoints, batch_size, seq_len
    
    def evaluate(self):
        print('\n' + '='*60 + '\nEvaluation\n' + '='*60)
        
        all_mpjpe = []
        all_pck = []
        
        pbar = tqdm(self.val_loader, desc='Evaluating')
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                images, gt_keypoints, batch_size, seq_len = self._process_batch(batch)
                
                images = images.to(self.device)
                gt_keypoints = gt_keypoints.to(self.device)
                
                with torch.cuda.amp.autocast(enabled=True):
                    output = self.model(images)
                    pred_keypoints = output['final_pose']
                
                for i in range(pred_keypoints.shape[0]):
                    mpjpe = self.mpjpe_metric(pred_keypoints[i:i+1], gt_keypoints[i:i+1])
                    pck = self.pck_metric(pred_keypoints[i:i+1], gt_keypoints[i:i+1])
                    all_mpjpe.append(mpjpe.item())
                    all_pck.append(pck.item())
                
                if batch_idx % 10 == 0:
                    pbar.set_postfix({
                        'mpjpe': f'{np.mean(all_mpjpe):.2f}',
                        'pck': f'{np.mean(all_pck):.2f}'
                    })
        
        all_mpjpe = np.array(all_mpjpe)
        all_pck = np.array(all_pck)
        
        print('\n' + '='*60 + '\nResults\n' + '='*60)
        print(f'MPJPE (mean): {np.mean(all_mpjpe):.2f} mm')
        print(f'MPJPE (median): {np.median(all_mpjpe):.2f} mm')
        print(f'MPJPE (std): {np.std(all_mpjpe):.2f} mm')
        print(f'PCK (@0.05): {np.mean(all_pck):.2f}%')
        print(f'PCK (median): {np.median(all_pck):.2f}%')
        
        print(f'\nError Distribution:')
        print(f'  < 10mm: {100*np.mean(all_mpjpe < 10):.2f}%')
        print(f'  10-20mm: {100*np.mean((all_mpjpe >= 10) & (all_mpjpe < 20)):.2f}%')
        print(f'  20-50mm: {100*np.mean((all_mpjpe >= 20) & (all_mpjpe < 50)):.2f}%')
        print(f'  > 50mm: {100*np.mean(all_mpjpe >= 50):.2f}%')
        
        print('\n' + '='*60 + '\nDone!')


def load_config(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Evaluate Geodesic Interpolation Hand Pose')
    parser.add_argument('--config', type=str, default='/data/data5/zhaoran/paper_code/back/config_fourier.yaml')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_geodesic_interp.pth')
    parser.add_argument('--gpu', type=int, default=0)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f'Checkpoint not found: {args.checkpoint}')
        ckpt_dir = os.path.dirname(args.checkpoint)
        if os.path.exists(ckpt_dir):
            print('Available:')
            for f in sorted(os.listdir(ckpt_dir)):
                if f.endswith('.pth'):
                    print(f'  {os.path.join(ckpt_dir, f)}')
        return
    
    cfg = load_config(args.config)
    evaluator = GeodesicEvaluator(cfg, args.checkpoint, gpu_id=args.gpu)
    evaluator.evaluate()


if __name__ == '__main__':
    main()
