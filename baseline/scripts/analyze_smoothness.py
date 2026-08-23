import os
import sys
import argparse
import torch
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['DEX_YCB_DIR'] = '/data/data2/kuanghaohong/Multiview/POEM/data/DexYCB'
sys.path.insert(0, '/data/data2/kuanghaohong/Multiview/POEM/dex-ycb-toolkit')

from utils.config import load_config
from utils.temporal_smoothness import (
    compare_pred_gt_smoothness,
    compute_per_joint_velocity,
    detect_abrupt_changes,
    print_smoothness_report
)
from datasets.dexycb_mv import build_dataloader
from models.syn2seq import Syn2SeqKeypoint

JOINT_NAMES = [
    'Wrist',
    'Index_MCP', 'Index_PIP', 'Index_DIP', 'Index_TIP',
    'Middle_MCP', 'Middle_PIP', 'Middle_DIP', 'Middle_TIP',
    'Pinky_MCP', 'Pinky_PIP', 'Pinky_DIP', 'Pinky_TIP',
    'Ring_MCP', 'Ring_PIP', 'Ring_DIP', 'Ring_TIP',
    'Thumb_MCP', 'Thumb_PIP', 'Thumb_DIP', 'Thumb_TIP'
]


def analyze_temporal_smoothness(cfg, checkpoint_path: str) -> Dict:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    model = Syn2SeqKeypoint(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f'Loaded checkpoint from {checkpoint_path}')
    
    val_loader = build_dataloader(cfg, split='val')
    
    all_pred_kp = []
    all_gt_kp = []
    
    with torch.no_grad():
        for batch in val_loader:
            exo_video = batch['exo_video'].to(device)
            gt_keypoints = batch['ego_keypoints'].to(device)
            
            pred_keypoints = model(exo_video)
            
            all_pred_kp.append(pred_keypoints.cpu())
            all_gt_kp.append(gt_keypoints.cpu())
    
    all_pred_kp = torch.cat(all_pred_kp, dim=0)
    all_gt_kp = torch.cat(all_gt_kp, dim=0)
    
    print(f'\nAnalyzing {all_pred_kp.shape[0]} sequences x {all_pred_kp.shape[1]} frames')
    
    comparison = compare_pred_gt_smoothness(all_pred_kp, all_gt_kp)
    
    print_smoothness_report(comparison)
    
    pred_per_joint_vel = compute_per_joint_velocity(all_pred_kp)
    gt_per_joint_vel = compute_per_joint_velocity(all_gt_kp)
    
    print('\n' + '=' * 70)
    print('PER-JOINT MEAN VELOCITY (mm/frame)')
    print('=' * 70)
    print(f'{"Joint":<20} {"Prediction":<15} {"Ground Truth":<15} {"Ratio":<10}')
    print('-' * 70)
    
    for i, (pred_vel, gt_vel, name) in enumerate(zip(pred_per_joint_vel, gt_per_joint_vel, JOINT_NAMES)):
        ratio = pred_vel / gt_vel if gt_vel > 0 else float('inf')
        print(f'{name:<20} {pred_vel:<15.2f} {gt_vel:<15.2f} {ratio:<10.2f}')
    
    pred_abrupt = detect_abrupt_changes(all_pred_kp, threshold=0.05)
    gt_abrupt = detect_abrupt_changes(all_gt_kp, threshold=0.05)
    
    pred_abrupt_ratio = pred_abrupt.float().mean().item() * 100
    gt_abrupt_ratio = gt_abrupt.float().mean().item() * 100
    
    print('\n' + '=' * 70)
    print('ABRUPT MOTION DETECTION (velocity > 50 mm/frame)')
    print('=' * 70)
    print(f'Prediction abrupt frames: {pred_abrupt_ratio:.2f}%')
    print(f'Ground Truth abrupt frames: {gt_abrupt_ratio:.2f}%')
    
    print('=' * 70)
    
    return {
        'smoothness_comparison': comparison,
        'per_joint_velocity': {'pred': pred_per_joint_vel.tolist(), 'gt': gt_per_joint_vel.tolist()},
        'abrupt_motion_ratio': {'pred': pred_abrupt_ratio, 'gt': gt_abrupt_ratio}
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze Temporal Smoothness')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, 
                       default='checkpoints/best_model.pth',
                       help='Path to checkpoint file')
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    analyze_temporal_smoothness(cfg, args.checkpoint)


if __name__ == '__main__':
    main()
