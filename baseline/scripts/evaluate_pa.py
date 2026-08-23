import os
import sys
import argparse
import torch
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['DEX_YCB_DIR'] = '/data/data2/kuanghaohong/Multiview/POEM/data/DexYCB'
sys.path.insert(0, '/data/data2/kuanghaohong/Multiview/POEM/dex-ycb-toolkit')

from utils.config import load_config
from utils.pa_metrics import compute_pa_metrics, print_pa_report
from datasets.dexycb_mv import build_dataloader
from models.syn2seq import Syn2SeqKeypoint


def evaluate_pa_metrics(cfg, checkpoint_path: str) -> Dict:
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
    
    print(f'\nEvaluating {all_pred_kp.shape[0]} sequences x {all_pred_kp.shape[1]} frames')
    
    metrics = compute_pa_metrics(all_pred_kp, all_gt_kp)
    
    print_pa_report(metrics)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate PA-EPE and PA-AUC Metrics')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, 
                       default='checkpoints/best_model.pth',
                       help='Path to checkpoint file')
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    evaluate_pa_metrics(cfg, args.checkpoint)


if __name__ == '__main__':
    main()
