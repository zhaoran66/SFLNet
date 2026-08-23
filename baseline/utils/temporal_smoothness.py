import torch
import numpy as np
from typing import Dict, Tuple


def compute_temporal_velocity(keypoints: torch.Tensor) -> torch.Tensor:
    """
    Compute temporal velocity: ||kp[t] - kp[t-1]||_2
    
    Args:
        keypoints: [batch, seq_len, num_joints, 3] or [seq_len, num_joints, 3]
    
    Returns:
        velocity: [..., seq_len-1, num_joints]
    """
    if keypoints.dim() == 3:
        keypoints = keypoints.unsqueeze(0)
    
    diff = keypoints[:, 1:] - keypoints[:, :-1]
    velocity = torch.norm(diff, p=2, dim=-1)
    
    return velocity


def compute_temporal_acceleration(keypoints: torch.Tensor) -> torch.Tensor:
    """
    Compute temporal acceleration: ||v[t] - v[t-1]||_2
    
    Args:
        keypoints: [batch, seq_len, num_joints, 3] or [seq_len, num_joints, 3]
    
    Returns:
        acceleration: [..., seq_len-2, num_joints]
    """
    velocity = compute_temporal_velocity(keypoints)
    acceleration = torch.norm(velocity[..., 1:, :] - velocity[..., :-1, :], p=2, dim=-1)
    
    return acceleration


def compute_temporal_jerk(keypoints: torch.Tensor) -> torch.Tensor:
    """
    Compute temporal jerk: ||a[t] - a[t-1]||_2
    
    Args:
        keypoints: [batch, seq_len, num_joints, 3] or [seq_len, num_joints, 3]
    
    Returns:
        jerk: [..., seq_len-3, num_joints]
    """
    acceleration = compute_temporal_acceleration(keypoints)
    jerk = torch.norm(acceleration[..., 1:, :] - acceleration[..., :-1, :], p=2, dim=-1)
    
    return jerk


def compute_smoothness_metrics(keypoints: torch.Tensor) -> Dict[str, float]:
    """
    Compute full temporal smoothness metrics
    
    Args:
        keypoints: [batch, seq_len, num_joints, 3]
    
    Returns:
        metrics: dictionary of smoothness metrics (in mm)
    """
    velocity = compute_temporal_velocity(keypoints)
    acceleration = compute_temporal_acceleration(keypoints)
    jerk = compute_temporal_jerk(keypoints)
    
    metrics = {
        'mean_velocity_mm': float(velocity.mean() * 1000),
        'std_velocity_mm': float(velocity.std() * 1000),
        'max_velocity_mm': float(velocity.max() * 1000),
        'mean_acceleration_mm': float(acceleration.mean() * 1000),
        'std_acceleration_mm': float(acceleration.std() * 1000),
        'max_acceleration_mm': float(acceleration.max() * 1000),
        'mean_jerk_mm': float(jerk.mean() * 1000),
        'std_jerk_mm': float(jerk.std() * 1000),
        'max_jerk_mm': float(jerk.max() * 1000),
    }
    
    return metrics


def compute_per_joint_velocity(keypoints: torch.Tensor) -> torch.Tensor:
    """
    Compute average temporal velocity for each joint
    
    Args:
        keypoints: [batch, seq_len, num_joints, 3]
    
    Returns:
        per_joint_vel: [num_joints]
    """
    velocity = compute_temporal_velocity(keypoints)
    per_joint_vel = velocity.mean(dim=(0, 1)) * 1000
    
    return per_joint_vel


def compare_pred_gt_smoothness(pred_kp: torch.Tensor, gt_kp: torch.Tensor) -> Dict[str, Dict[str, float]]:
    """
    Compare temporal smoothness between prediction and ground truth
    
    Args:
        pred_kp: [batch, seq_len, num_joints, 3]
        gt_kp: [batch, seq_len, num_joints, 3]
    
    Returns:
        comparison: {'pred': pred_metrics, 'gt': gt_metrics, 'ratio': ratio_metrics}
    """
    pred_metrics = compute_smoothness_metrics(pred_kp)
    gt_metrics = compute_smoothness_metrics(gt_kp)
    
    ratio_metrics = {}
    for key in pred_metrics.keys():
        if gt_metrics[key] > 0:
            ratio_metrics[f'{key}_ratio'] = pred_metrics[key] / gt_metrics[key]
        else:
            ratio_metrics[f'{key}_ratio'] = float('inf')
    
    return {
        'pred': pred_metrics,
        'gt': gt_metrics,
        'ratio': ratio_metrics
    }


def detect_abrupt_changes(keypoints: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
    """
    Detect abrupt motion changes (jump points)
    
    Args:
        keypoints: [batch, seq_len, num_joints, 3]
        threshold: threshold (in meters)
    
    Returns:
        is_abrupt: [batch, seq_len-1, num_joints] boolean mask
    """
    velocity = compute_temporal_velocity(keypoints)
    is_abrupt = velocity > threshold
    
    return is_abrupt


def compute_accumulated_motion(keypoints: torch.Tensor) -> torch.Tensor:
    """
    Compute accumulated motion (path length)
    
    Args:
        keypoints: [batch, seq_len, num_joints, 3]
    
    Returns:
        accumulated: [batch, num_joints]
    """
    velocity = compute_temporal_velocity(keypoints)
    accumulated = velocity.sum(dim=1) * 1000
    
    return accumulated


def print_smoothness_report(comparison: Dict[str, Dict[str, float]]):
    """
    Print temporal smoothness report
    """
    print('=' * 70)
    print('TEMPORAL SMOOTHNESS ANALYSIS REPORT')
    print('=' * 70)
    
    print(f'\n{"Metric":<30} {"Prediction (mm)":<18} {"Ground Truth (mm)":<18} {"Ratio (Pred/GT)":<12}')
    print('-' * 70)
    
    keys = ['mean_velocity_mm', 'std_velocity_mm', 'max_velocity_mm',
            'mean_acceleration_mm', 'std_acceleration_mm', 'max_acceleration_mm',
            'mean_jerk_mm', 'std_jerk_mm', 'max_jerk_mm']
    
    for key in keys:
        pred_val = comparison['pred'][key]
        gt_val = comparison['gt'][key]
        ratio = comparison['ratio'][f'{key}_ratio']
        
        name = key.replace('_mm', '').replace('_', ' ').title()
        print(f'{name:<30} {pred_val:<18.2f} {gt_val:<18.2f} {ratio:<12.2f}')
    
    print('=' * 70)
    print('\nInterpretation:')
    print('  - Ratio close to 1.0: prediction has similar smoothness to ground truth')
    print('  - Ratio > 1.0: prediction is less smooth (more jittery)')
    print('  - Ratio < 1.0: prediction is smoother (more stable)')
    print()
