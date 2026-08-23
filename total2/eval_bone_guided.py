# -*- coding: utf-8 -*-
import os
import sys
import argparse
import torch
import yaml
import numpy as np

from latent_bone_guided import LatentHandPoseModel
from utils_fourier import MPJPE, PCK
from dataset_dexycb import DexYCBMultiViewReal
from torch.utils.data import DataLoader


def compute_pa_epe(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    # Handle [batch, seq_len, num_joints, 3] -> flatten to [batch*seq_len, num_joints, 3]
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


def main():
    parser = argparse.ArgumentParser(description='Evaluate Bone Guided Model')
    parser.add_argument('--config', type=str, default='config_fourier.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint')
    parser.add_argument('--gpu', type=int, default=-1,
                       help='GPU device id')
    parser.add_argument('--max_batches', type=int, default=50,
                       help='Max batches to evaluate')
    args = parser.parse_args()
    
    cfg = yaml.safe_load(open(args.config))
    checkpoint_path = args.checkpoint
    
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')
    
    embed_dim = cfg['model'].get('hidden_dim', 256)
    num_interpolation_steps = cfg['model'].get('interpolate_steps', 4)
    num_joints = cfg['dataset'].get('num_joints', 21)
    num_views = len(cfg['dataset'].get('exo_views', [0, 1, 2, 3, 4]))
    fourier_threshold = cfg['model'].get('fourier_threshold', 0.1)
    pose_dim = cfg['model'].get('pose_dim', 12)
    
    print(f'Loading model: embed_dim={embed_dim}, steps={num_interpolation_steps}, joints={num_joints}')
    
    model = LatentHandPoseModel(
        embed_dim=embed_dim,
        num_interpolation_steps=num_interpolation_steps,
        num_joints=num_joints,
        num_views=num_views,
        fourier_threshold=fourier_threshold,
        pose_dim=pose_dim
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded checkpoint from {checkpoint_path}, epoch: {checkpoint.get("epoch", "unknown")}')
    
    model = model.to(device)
    model.eval()
    
    mpjpe_metric = MPJPE()
    pck_metric = PCK()
    
    print('Loading dataset...')
    val_dataset = DexYCBMultiViewReal(cfg, split='val')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f'Val: {len(val_dataset)} sequences')
    
    all_mpjpe = []
    all_pck = []
    all_pa_epe = []
    
    print(f'Evaluating first {args.max_batches} batches...')
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= args.max_batches:
                break
                
            exo_video = batch['exo_video'].to(device)
            ego_keypoints = batch['ego_keypoints'].to(device)
            exo_pose = batch.get('exo_pose')
            ego_pose = batch.get('ego_pose')
            if exo_pose is not None:
                exo_pose = exo_pose.to(device)
            if ego_pose is not None:
                ego_pose = ego_pose.to(device)
            
            batch_size, seq_len, num_views, c, h, w = exo_video.shape
            
            # Resize if needed
            if h != 256 or w != 256:
                # Reshape for interpolation: [batch, seq_len, num_views, c, h, w] -> [batch*seq_len*num_views, c, h, w]
                reshaped = exo_video.reshape(batch_size * seq_len * num_views, c, h, w)
                resized = torch.nn.functional.interpolate(
                    reshaped, size=(256, 256), mode='bilinear', align_corners=False
                )
                exo_video = resized.reshape(batch_size, seq_len, num_views, c, 256, 256)
            
            pred_keypoints = model(exo_video, exo_pose=exo_pose, ego_pose=ego_pose)
            
            # Flatten for metrics: [batch, seq_len, num_joints, 3] -> [batch*seq_len, num_joints, 3]
            pred_flat = pred_keypoints.reshape(batch_size * seq_len, -1, 3)
            ego_flat = ego_keypoints.reshape(batch_size * seq_len, -1, 3)
            
            mpjpe = mpjpe_metric(pred_flat, ego_flat)
            pck = pck_metric(pred_flat, ego_flat)
            pa_epe = compute_pa_epe(pred_keypoints, ego_keypoints)
            
            all_mpjpe.append(mpjpe.item())
            all_pck.append(pck.item())
            all_pa_epe.extend(pa_epe.cpu().numpy())
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}: MPJPE={mpjpe.item():.2f}mm, PCK={pck.item():.2f}%')
    
    pa_epe_tensor = torch.tensor(all_pa_epe)
    pa_auc = compute_pa_auc(pa_epe_tensor)
    
    print()
    print('='*60)
    print(f'RESULTS ({args.max_batches} batches):')
    print(f'MPJPE: {np.mean(all_mpjpe):.2f} +/- {np.std(all_mpjpe):.2f} mm')
    print(f'PCK@0.05: {np.mean(all_pck):.2f} +/- {np.std(all_pck):.2f} %')
    print()
    print(f'PA-EPE (mean):   {pa_epe_tensor.mean().item():.2f} cm')
    print(f'PA-EPE (std):    {pa_epe_tensor.std().item():.2f} cm')
    print(f'PA-EPE (median): {pa_epe_tensor.median().item():.2f} cm')
    print(f'PA-AUC:          {pa_auc:.2f} %')
    print('='*60)


if __name__ == '__main__':
    main()
