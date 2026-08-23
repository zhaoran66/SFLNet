# -*- coding: utf-8 -*-
"""
Geometry Baseline Evaluation

Reviewer baseline: Exo 3D Hand Pose -> rigid transformation (using known
camera extrinsics) -> Ego 3D Hand Pose.

This script loads the GT 3D hand joints in the EXO camera coordinate system,
transforms them into the EGO camera coordinate system via the relative rigid
transform computed from camera extrinsics, and evaluates against the GT ego
joints with the same metrics used for SFLNet (MPJPE, PCK, PA-EPE, PA-AUC).
"""

import argparse
import os
import sys
import yaml
import torch
import numpy as np

sys.path.insert(0, '/data/data5/zhaoran/paper_code/total')

from dataset_dexycb import DexYCBMultiViewReal
from utils_fourier import MPJPE, PCK
from torch.utils.data import DataLoader


def pose12_to_matrix(pose12: torch.Tensor) -> torch.Tensor:
    """
    Convert a flattened 3x4 camera extrinsic [R|t] into a 4x4 transformation matrix.

    The input is expected to be in row-major order:
        [R11 R12 R13 t1 R21 R22 R23 t2 R31 R32 R33 t3]
    representing the world-to-camera transform: X_cam = R @ X_world + t.

    Args:
        pose12: [..., 12] tensor.
    Returns:
        T: [..., 4, 4] transformation matrix.
    """
    orig_shape = pose12.shape[:-1]
    pose = pose12.reshape(*orig_shape, 3, 4)  # [..., 3, 4]
    R = pose[..., :, :3]   # [..., 3, 3]
    t = pose[..., :, 3:4]  # [..., 3, 1]

    T = torch.eye(4, dtype=pose12.dtype, device=pose12.device)
    T_shape = (*orig_shape, 4, 4)
    T = T.view(*([1] * len(orig_shape)), 4, 4).expand(T_shape).clone()
    T[..., :3, :3] = R
    T[..., :3, 3:4] = t
    return T


def relative_camera_transform(exo_pose12: torch.Tensor, ego_pose12: torch.Tensor) -> torch.Tensor:
    """
    Compute the rigid transformation that maps points from the EXO camera frame
    to the EGO camera frame.

    Args:
        exo_pose12: [..., 12] exo camera extrinsics (world -> exo cam).
        ego_pose12: [..., 12] ego camera extrinsics (world -> ego cam).
    Returns:
        T_ego_exo: [..., 4, 4] such that X_ego = T_ego_exo @ X_exo.
    """
    T_w2exo = pose12_to_matrix(exo_pose12)  # world -> exo
    T_w2ego = pose12_to_matrix(ego_pose12)  # world -> ego
    T_exo2w = torch.inverse(T_w2exo)        # exo -> world
    T_ego_exo = T_w2ego @ T_exo2w           # exo -> world -> ego
    return T_ego_exo


def transform_points(points: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """
    Apply a 4x4 rigid transform to 3D points.

    Args:
        points: [..., N, 3]
        T: [..., 4, 4]
    Returns:
        transformed: [..., N, 3]
    """
    # Handle broadcasting between points and T.
    # points: [B, T, J, 3], T: [B, T, 4, 4]
    orig_shape = points.shape
    N = orig_shape[-2]
    pts = points.reshape(-1, N, 3)  # [M, N, 3]
    T_flat = T.reshape(-1, 4, 4)    # [M, 4, 4]

    ones = torch.ones(*pts.shape[:-1], 1, dtype=pts.dtype, device=pts.device)
    pts_h = torch.cat([pts, ones], dim=-1)  # [M, N, 4]
    transformed_h = torch.bmm(pts_h, T_flat.transpose(1, 2))  # [M, N, 4]
    transformed = transformed_h[..., :3]

    return transformed.reshape(orig_shape)


def wrist_center(points: torch.Tensor) -> torch.Tensor:
    """Subtract wrist (joint 0) from all joints."""
    return points - points[..., 0:1, :]


def compute_pa_epe(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Per-frame Procrustes-aligned EPE (in cm)."""
    if pred.dim() == 4:
        B, T, J, C = pred.shape
        pred = pred.reshape(B * T, J, C)
        gt = gt.reshape(B * T, J, C)

    pred_center = pred.mean(dim=1, keepdim=True)
    gt_center = gt.mean(dim=1, keepdim=True)
    pred_aligned = pred - pred_center
    gt_aligned = gt - gt_center

    epe = torch.norm(pred_aligned - gt_aligned, p=2, dim=-1)
    pa_epe = epe.mean(dim=-1) * 100  # m -> cm
    return pa_epe


def compute_pa_auc(errors: torch.Tensor, max_threshold: float = 10.0) -> float:
    """Compute PA-EPE AUC (percentage)."""
    thresholds = torch.linspace(0, max_threshold, 100)
    pck_curve = [(errors < t).float().mean() for t in thresholds]
    pck_curve = torch.tensor(pck_curve)
    auc = torch.trapz(pck_curve, thresholds) / max_threshold * 100
    return auc.item()


def evaluate_geometry_baseline(cfg, split='val', max_batches=None, device='cpu'):
    """Evaluate the geometry baseline on the given split."""
    dataset = DexYCBMultiViewReal(cfg, split=split)
    dataloader = DataLoader(dataset, batch_size=cfg['training'].get('batch_size', 4),
                            shuffle=False, num_workers=0)

    mpjpe_metric = MPJPE()
    pck_metric = PCK()

    all_mpjpe = []
    all_pck = []
    all_pa_epe = []

    print(f'Evaluating Geometry Baseline on {split}: {len(dataset)} sequences')

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        ego_keypoints = batch['ego_keypoints'].to(device)        # [B, T, J, 3]
        exo_keypoints = batch['exo_keypoints'].to(device)        # [B, T, J, 3]
        exo_pose = batch['exo_pose'].to(device)                  # [B, T, V, 12]
        ego_pose = batch['ego_pose'].to(device)                  # [B, T, 12]

        B, T, V, _ = exo_pose.shape

        # Use the first exo view for the geometry baseline.
        exo_pose_first = exo_pose[:, :, 0, :]  # [B, T, 12]

        # Compute exo -> ego rigid transform.
        T_ego_exo = relative_camera_transform(exo_pose_first, ego_pose)  # [B, T, 4, 4]

        # Transform exo joints to ego camera space.
        pred_ego = transform_points(exo_keypoints, T_ego_exo)  # [B, T, J, 3]

        # Wrist-center both prediction and ground truth in ego camera space.
        pred_ego_centered = wrist_center(pred_ego)
        ego_keypoints_centered = wrist_center(ego_keypoints)

        # Flatten for MPJPE / PCK.
        pred_flat = pred_ego_centered.reshape(B * T, -1, 3)
        gt_flat = ego_keypoints_centered.reshape(B * T, -1, 3)

        mpjpe = mpjpe_metric(pred_flat, gt_flat)
        pck = pck_metric(pred_flat, gt_flat)
        pa_epe = compute_pa_epe(pred_ego_centered, ego_keypoints_centered)

        all_mpjpe.append(mpjpe.item())
        all_pck.append(pck.item())
        all_pa_epe.extend(pa_epe.cpu().numpy())

        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}: MPJPE={mpjpe.item():.2f}mm, PCK={pck.item():.2f}%')

    all_pa_epe = torch.tensor(all_pa_epe)
    pa_auc = compute_pa_auc(all_pa_epe)

    print()
    print('=' * 60)
    print('Geometry Baseline Results (GT Exo 3D Pose -> Rigid Transform -> Ego)')
    print('=' * 60)
    print(f'MPJPE:   {np.mean(all_mpjpe):.2f} +/- {np.std(all_mpjpe):.2f} mm')
    print(f'PCK@5cm: {np.mean(all_pck):.2f} +/- {np.std(all_pck):.2f} %')
    print(f'PA-EPE:  {all_pa_epe.mean().item():.2f} +/- {all_pa_epe.std().item():.2f} cm')
    print(f'PA-AUC:  {pa_auc:.2f} %')
    print('=' * 60)

    return {
        'mpjpe': float(np.mean(all_mpjpe)),
        'pck': float(np.mean(all_pck)),
        'pa_epe': float(all_pa_epe.mean().item()),
        'pa_auc': pa_auc,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Geometry Baseline')
    parser.add_argument('--config', type=str, default='/data/data5/zhaoran/paper_code/total/config_fourier.yaml',
                        help='Path to config file')
    parser.add_argument('--split', type=str, default='val',
                        help='Dataset split to evaluate')
    parser.add_argument('--max_batches', type=int, default=None,
                        help='Limit number of batches (for quick test)')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU device id, -1 for CPU')
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    results = evaluate_geometry_baseline(cfg, split=args.split,
                                         max_batches=args.max_batches, device=device)


if __name__ == '__main__':
    main()
