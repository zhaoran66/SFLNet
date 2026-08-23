import os
import argparse
import torch
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

from utils_fourier import load_config, MPJPE, PCK
from dataset_dexycb import build_dataloader_dexycb
from model_fourier_geodesic import Syn2SeqKeypointFourierGeodesic


def compute_pa_epe(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Compute Procrustes-aligned EPE"""
    if pred.dim() == 4:
        batch_size, seq_len, num_joints, coords = pred.shape
        pred = pred.reshape(batch_size * seq_len, num_joints, coords)
        gt = gt.reshape(batch_size * seq_len, num_joints, coords)
    
    pred_center = pred.mean(dim=1, keepdim=True)
    gt_center = gt.mean(dim=1, keepdim=True)
    
    pred_aligned = pred - pred_center
    gt_aligned = gt - gt_center
    
    epe = torch.norm(pred_aligned - gt_aligned, p=2, dim=-1)
    pa_epe = epe.mean(dim=-1) * 100  # Convert to cm
    
    return pa_epe


def compute_pa_auc(errors: torch.Tensor, max_threshold: float = 10.0) -> float:
    """Compute PA-AUC (area under PCK curve)"""
    thresholds = torch.linspace(0, max_threshold, 100)
    pck_curve = []
    for thresh in thresholds:
        pck_thresh = (errors < thresh).float().mean()
        pck_curve.append(pck_thresh)
    
    pck_curve = torch.tensor(pck_curve)
    auc = torch.trapz(pck_curve, thresholds) / max_threshold * 100
    
    return auc.item()


def main():
    parser = argparse.ArgumentParser(description='Evaluate Back model on DexYCB')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config_fourier.yaml', help='Path to config file')
    parser.add_argument('--gpu', type=int, default=4, help='GPU ID')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'], help='Evaluation split')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    cfg = load_config(args.config)
    
    model = Syn2SeqKeypointFourierGeodesic(cfg).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded checkpoint from {args.checkpoint}, epoch: {checkpoint.get("epoch", "unknown")}')
    
    model.eval()
    
    val_loader = build_dataloader_dexycb(cfg, split=args.split)
    print(f'Evaluating on {len(val_loader.dataset)} sequences')
    
    mpjpe_metric = MPJPE()
    pck_metric = PCK()
    
    all_mpjpe = []
    all_pck = []
    all_pa_epe = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Evaluating')
        for batch in pbar:
            exo_video = batch['exo_video'].to(device)
            gt_keypoints = batch['ego_keypoints'].to(device)
            
            pred_keypoints = model(exo_video)
            
            # CRITICAL: Apply wrist-centering to predictions
            # GT is already wrist-centered in dataset, so predictions must be too
            pred_wrist = pred_keypoints[..., 0:1, :]
            pred_keypoints_centered = pred_keypoints - pred_wrist
            
            mpjpe = mpjpe_metric(pred_keypoints_centered, gt_keypoints).item()
            pck = pck_metric(pred_keypoints_centered, gt_keypoints).item()
            pa_epe = compute_pa_epe(pred_keypoints_centered, gt_keypoints)
            
            all_mpjpe.append(mpjpe)
            all_pck.append(pck)
            all_pa_epe.extend(pa_epe.cpu().numpy())
    
    all_mpjpe = np.array(all_mpjpe)
    all_pck = np.array(all_pck)
    pa_epe_tensor = torch.tensor(all_pa_epe)
    pa_auc = compute_pa_auc(pa_epe_tensor)
    
    print('\n' + '='*60)
    print(f'BACK MODEL EVALUATION RESULTS ({len(all_mpjpe)} batches)')
    print('='*60)
    print(f'MPJPE (mean):   {all_mpjpe.mean():.2f} mm')
    print(f'MPJPE (std):    {all_mpjpe.std():.2f} mm')
    print(f'MPJPE (median): {np.median(all_mpjpe):.2f} mm')
    print()
    print(f'PCK@50mm (mean):   {all_pck.mean():.2f} %')
    print(f'PCK@50mm (std):    {all_pck.std():.2f} %')
    print(f'PCK@50mm (median): {np.median(all_pck):.2f} %')
    print()
    print(f'PA-EPE (mean):   {pa_epe_tensor.mean().item():.2f} cm')
    print(f'PA-EPE (std):    {pa_epe_tensor.std().item():.2f} cm')
    print(f'PA-EPE (median): {pa_epe_tensor.median().item():.2f} cm')
    print(f'PA-AUC:          {pa_auc:.2f} %')
    print('='*60)


if __name__ == '__main__':
    main()
