import os
import sys
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

from config import load_config
from losses import MPJPE, PCK
from dataset import build_dataloader
from model import SPLLatentKeypoint


def compute_pa_epe(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    if pred.dim() == 4:
        batch_size, seq_len, num_joints, coords = pred.shape
        pred = pred.reshape(batch_size * seq_len, num_joints, coords)
        gt = gt.reshape(batch_size * seq_len, num_joints, coords)
    
    pred_center = pred.mean(dim=1, keepdim=True)
    gt_center = gt.mean(dim=1, keepdim=True)
    
    pred_aligned = pred - pred_center
    gt_aligned = gt - gt_center
    
    epe = torch.norm(pred_aligned - gt_aligned, p=2, dim=-1)
    pa_epe = epe.mean(dim=-1) * 100
    
    return pa_epe


def compute_pa_auc(errors: torch.Tensor, max_threshold: float = 10.0) -> float:
    thresholds = torch.linspace(0, max_threshold, 100)
    pck_curve = []
    for thresh in thresholds:
        pck_thresh = (errors < thresh).float().mean()
        pck_curve.append(pck_thresh)
    
    pck_curve = torch.tensor(pck_curve)
    auc = torch.trapz(pck_curve, thresholds) / max_threshold * 100
    
    return auc.item()


def evaluate(cfg, checkpoint_path, max_batches=50):
    device = torch.device(cfg.training.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    model = SPLLatentKeypoint(cfg).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ckpt_state = checkpoint['model_state_dict']
    
    model.load_state_dict(ckpt_state, strict=False)
    
    print(f'Loaded checkpoint from {checkpoint_path}, epoch: {checkpoint.get("epoch", "unknown")}')
    
    model.eval()
    
    val_loader = build_dataloader(cfg, split='val')
    
    mpjpe_metric = MPJPE()
    pck_metric = PCK()
    
    all_mpjpe = []
    all_pck = []
    all_pa_epe = []
    
    print(f'Evaluating max {max_batches} batches...')
    
    count = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Evaluating'):
            if count >= max_batches:
                break
                
            exo_video = batch['exo_video'].to(device)
            gt_keypoints = batch['ego_keypoints'].to(device)
            
            output = model(exo_video)
            if isinstance(output, dict):
                pred_keypoints = output['final_pose']
            else:
                pred_keypoints = output
            
            # CRITICAL FIX: Apply wrist-centering to predictions
            # GT is already wrist-centered in dataset, so predictions must be too
            pred_wrist = pred_keypoints[..., 0:1, :]
            pred_keypoints_centered = pred_keypoints - pred_wrist
            
            mpjpe = mpjpe_metric(pred_keypoints_centered, gt_keypoints)
            pck = pck_metric(pred_keypoints_centered, gt_keypoints)
            pa_epe = compute_pa_epe(pred_keypoints_centered, gt_keypoints)
            
            all_mpjpe.append(mpjpe.item())
            all_pck.append(pck.item())
            all_pa_epe.extend(pa_epe.cpu().numpy())
            count += 1
    
    pa_epe_tensor = torch.tensor(all_pa_epe)
    pa_auc = compute_pa_auc(pa_epe_tensor)
    
    print()
    print('='*60)
    print(f'SPL MODEL EVALUATION RESULTS ({len(all_mpjpe)} batches)')
    print('='*60)
    print(f'MPJPE (mean):   {np.mean(all_mpjpe):.2f} mm')
    print(f'MPJPE (std):    {np.std(all_mpjpe):.2f} mm')
    print(f'MPJPE (median): {np.median(all_mpjpe):.2f} mm')
    print()
    print(f'PCK@0.05 (mean):   {np.mean(all_pck):.2f} %')
    print(f'PCK@0.05 (std):    {np.std(all_pck):.2f} %')
    print(f'PCK@0.05 (median): {np.median(all_pck):.2f} %')
    print()
    print(f'PA-EPE (mean):   {pa_epe_tensor.mean().item():.2f} cm')
    print(f'PA-EPE (std):    {pa_epe_tensor.std().item():.2f} cm')
    print(f'PA-EPE (median): {pa_epe_tensor.median().item():.2f} cm')
    print(f'PA-AUC:          {pa_auc:.2f} %')
    print('='*60)
    
    return {
        'mpjpe_mean': np.mean(all_mpjpe),
        'mpjpe_std': np.std(all_mpjpe),
        'pck_mean': np.mean(all_pck),
        'pck_std': np.std(all_pck),
        'num_batches': len(all_mpjpe)
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate SPL Model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint')
    parser.add_argument('--max_batches', type=int, default=50,
                       help='Max batches to evaluate')
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    evaluate(cfg, args.checkpoint, args.max_batches)


if __name__ == '__main__':
    main()
