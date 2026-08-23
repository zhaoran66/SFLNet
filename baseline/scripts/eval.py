import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

from utils.config import load_config
from utils.losses import KeypointLoss, MPJPE, PCK
from datasets.dexycb_mv import build_dataloader
from models.syn2seq import Syn2SeqKeypoint


class Evaluator:
    def __init__(self, cfg, checkpoint_path: str):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        
        self.model = Syn2SeqKeypoint(cfg).to(self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded checkpoint from {checkpoint_path}')
        
        self.criterion = KeypointLoss()
        self.mpjpe_metric = MPJPE()
        self.pck_metric = PCK()
        
        self.val_loader = build_dataloader(cfg, split='val')
        
    def evaluate(self):
        self.model.eval()
        
        total_loss = 0.0
        total_mpjpe = 0.0
        total_pck = 0.0
        total_mse = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Evaluating')
            
            for batch_idx, batch in enumerate(pbar):
                exo_video = batch['exo_video'].to(self.device)
                gt_keypoints = batch['ego_keypoints'].to(self.device)
                
                pred_keypoints = self.model(exo_video)
                
                loss_dict = self.criterion(pred_keypoints, gt_keypoints)
                
                mpjpe = self.mpjpe_metric(pred_keypoints, gt_keypoints)
                pck = self.pck_metric(pred_keypoints, gt_keypoints)
                
                total_loss += loss_dict['loss'].item()
                total_mpjpe += mpjpe.item()
                total_pck += pck.item()
                total_mse += loss_dict['mse'].item()
                num_batches += 1
                
                if HAS_TQDM:
                    pbar.set_postfix({
                        'loss': f'{loss_dict["loss"].item():.4f}',
                        'mpjpe': f'{mpjpe.item():.2f}mm',
                        'pck': f'{pck.item():.2f}%'
                    })
        
        print('\n' + '=' * 70)
        print('EVALUATION RESULTS')
        print('=' * 70)
        print(f'Loss:      {total_loss / num_batches:.4f}')
        print(f'MSE:       {total_mse / num_batches:.4f}')
        print(f'MPJPE:     {total_mpjpe / num_batches:.2f} mm')
        print(f'PCK@5cm:  {total_pck / num_batches:.2f} %')
        print('=' * 70)
        
        return {
            'loss': total_loss / num_batches,
            'mpjpe': total_mpjpe / num_batches,
            'pck': total_pck / num_batches,
            'mse': total_mse / num_batches
        }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Syn2Seq Keypoint Model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, 
                       default='checkpoints/best_model.pth',
                       help='Path to checkpoint file')
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    evaluator = Evaluator(cfg, args.checkpoint)
    evaluator.evaluate()


if __name__ == '__main__':
    main()
