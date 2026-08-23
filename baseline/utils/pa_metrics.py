import torch
import numpy as np
from typing import Dict, Tuple


def compute_pa_epe(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Compute Pose Alignment - End Point Error (PA-EPE)
    
    PA-EPE = ||(pred - pred_center) - (gt - gt_center)||_2
    
    This removes global translation and measures only pose shape error.
    
    Args:
        pred: [batch, seq_len, num_joints, 3] or [batch, num_joints, 3]
        gt: same shape as pred
    
    Returns:
        pa_epe: per-sample PA-EPE in cm
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
        gt = gt.unsqueeze(1)
    
    pred_center = pred.mean(dim=2, keepdim=True)
    gt_center = gt.mean(dim=2, keepdim=True)
    
    pred_aligned = pred - pred_center
    gt_aligned = gt - gt_center
    
    epe = torch.norm(pred_aligned - gt_aligned, p=2, dim=-1)
    pa_epe = epe.mean(dim=-1) * 100
    
    return pa_epe


def compute_pck_curve(errors: torch.Tensor, thresholds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute PCK curve for AUC calculation
    
    Args:
        errors: [N] error values (in cm)
        thresholds: [T] threshold values (in cm)
    
    Returns:
        pck: [T] PCK values at each threshold
        thresholds: same as input
    """
    pck = []
    for thresh in thresholds:
        pck_thresh = (errors < thresh).float().mean()
        pck.append(pck_thresh)
    
    return torch.tensor(pck), thresholds


def compute_pa_auc(errors: torch.Tensor, max_threshold: float = 10.0) -> float:
    """
    Compute PA-AUC (Pose Alignment - Area Under Curve)
    Standard threshold range: 0-10cm
    
    Args:
        errors: [N] PA-EPE values (in cm)
        max_threshold: maximum threshold (in cm), default 10cm (standard)
    
    Returns:
        auc: Area Under PCK curve (0-100, percentage)
    """
    thresholds = torch.linspace(0, max_threshold, 100)
    pck_curve, _ = compute_pck_curve(errors, thresholds)
    
    auc = torch.trapz(pck_curve, thresholds) / max_threshold * 100
    
    return auc.item()


def compute_procrustes_alignment(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Apply Procrustes analysis to align prediction to ground truth
    (removes global translation, rotation, scale)
    
    Args:
        pred: [batch, seq_len, num_joints, 3] or [batch, num_joints, 3]
        gt: same shape as pred
    
    Returns:
        pred_aligned: aligned prediction
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
        gt = gt.unsqueeze(1)
    
    batch_size, seq_len, num_joints, _ = pred.shape
    
    pred_flat = pred.reshape(-1, num_joints, 3)
    gt_flat = gt.reshape(-1, num_joints, 3)
    
    pred_centered = pred_flat - pred_flat.mean(dim=1, keepdim=True)
    gt_centered = gt_flat - gt_flat.mean(dim=1, keepdim=True)
    
    pred_scale = torch.norm(pred_centered, p=2, dim=(1, 2), keepdim=True)
    gt_scale = torch.norm(gt_centered, p=2, dim=(1, 2), keepdim=True)
    
    pred_normalized = pred_centered / (pred_scale + 1e-8)
    gt_normalized = gt_centered / (gt_scale + 1e-8)
    
    H = torch.bmm(pred_normalized.transpose(1, 2), gt_normalized)
    U, S, V = torch.svd(H.cpu())
    U = U.to(pred.device)
    S = S.to(pred.device)
    V = V.to(pred.device)
    
    R = torch.bmm(V, U.transpose(1, 2))
    
    det_R = torch.det(R)
    V[:, :, -1] *= det_R.unsqueeze(-1)
    R = torch.bmm(V, U.transpose(1, 2))
    
    pred_aligned = torch.bmm(pred_normalized, R)
    pred_aligned = pred_aligned * gt_scale
    pred_aligned = pred_aligned.reshape(batch_size, seq_len, num_joints, 3)
    
    return pred_aligned


def compute_pa_metrics(pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
    """
    Compute full PA metrics
    
    Args:
        pred: [batch, seq_len, num_joints, 3]
        gt: same shape as pred
    
    Returns:
        metrics dictionary
    """
    pa_epe = compute_pa_epe(pred, gt)
    
    pred_aligned = compute_procrustes_alignment(pred, gt)
    pa_epe_procrustes = compute_pa_epe(pred_aligned, gt)
    
    pa_auc = compute_pa_auc(pa_epe.flatten(), max_threshold=10.0)
    pa_auc_procrustes = compute_pa_auc(pa_epe_procrustes.flatten(), max_threshold=10.0)
    
    metrics = {
        'pa_epe_mean_cm': float(pa_epe.mean()),
        'pa_epe_std_cm': float(pa_epe.std()),
        'pa_epe_median_cm': float(pa_epe.median()),
        'pa_auc_pct': pa_auc,
        'pa_epe_procrustes_mean_cm': float(pa_epe_procrustes.mean()),
        'pa_auc_procrustes_pct': pa_auc_procrustes
    }
    
    return metrics


def print_pa_report(metrics: Dict[str, float]):
    """
    Print PA metrics report
    """
    print('=' * 70)
    print('POSE ALIGNMENT METRICS REPORT (Standard 0-10cm)')
    print('=' * 70)
    
    print(f'\n{"Metric":<40} {"Value":<20}')
    print('-' * 70)
    
    print(f'{"PA-EPE (mean) [cm]":<40} {metrics["pa_epe_mean_cm"]:<20.2f}')
    print(f'{"PA-EPE (std) [cm]":<40} {metrics["pa_epe_std_cm"]:<20.2f}')
    print(f'{"PA-EPE (median) [cm]":<40} {metrics["pa_epe_median_cm"]:<20.2f}')
    print(f'{"PA-AUC [%]":<40} {metrics["pa_auc_pct"]:<20.2f}')
    print()
    print(f'{"PA-EPE (Procrustes aligned) [cm]":<40} {metrics["pa_epe_procrustes_mean_cm"]:<20.2f}')
    print(f'{"PA-AUC (Procrustes aligned) [%]":<40} {metrics["pa_auc_procrustes_pct"]:<20.2f}')
    
    print('=' * 70)
    print('\nInterpretation:')
    print('  - Lower PA-EPE is better (pose shape error)')
    print('  - Higher PA-AUC is better (area under PCK curve, 0-10cm)')
    print('  - Procrustes: removes translation, rotation, scale')
    print()
