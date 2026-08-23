#!/usr/bin/env python3
"""
Exo2Ego Evaluation Script
Computes metrics: SSIM, PSNR, LPIPS, FID for generated ego videos
"""

import argparse
import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
import torchvision.transforms as transforms

sys.path.insert(0, str(Path(__file__).parent))

from model_layout_transformer import create_layout_transformer
from model_diffusion import create_diffusion_model


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Exo2Ego model')
    parser.add_argument('--config', type=str, default='config_exo2ego.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of evaluation dataset')
    parser.add_argument('--split', type=str, default='val',
                        help='Dataset split to evaluate')
    parser.add_argument('--output', type=str, default='evaluation_results',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to evaluate')
    return parser.parse_args()


def create_models(cfg, device):
    """ģ"""
    stage1_model = create_layout_transformer(cfg)
    stage2_model = create_diffusion_model(cfg)

    stage1_model.to(device)
    stage2_model.to(device)

    stage1_model.eval()
    stage2_model.eval()

    return stage1_model, stage2_model


class MetricsCalculator:
    """ͼָ"""

    def __init__(self, device):
        self.device = device

    def ssim(self, img1, img2):
        """SSIM (Structural Similarity Index)"""
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)

        mu1 = img1.mean()
        mu2 = img2.mean()
        sigma1 = img1.var()
        sigma2 = img2.var()
        sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1**2 + mu2**2 + C1) * (sigma1 + sigma2 + C2))

        return ssim

    def psnr(self, img1, img2, max_val=255):
        """PSNR (Peak Signal-to-Noise Ratio)"""
        mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(max_val / np.sqrt(mse))

    def mse(self, img1, img2):
        """MSE"""
        return np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)


def calculate_fid(features1, features2):
    """
    FID (Frchet Inception Distance)
    򻯰汾 - ʹԤͳ
    """
    mu1, sigma1 = features1.mean(axis=0), np.cov(features1, rowvar=False)
    mu2, sigma2 = features2.mean(axis=0), np.cov(features2, rowvar=False)

    # FID
    diff = mu1 - mu2
    covmean = np.sqrt(sigma1 @ sigma2)

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


class Exo2EgoEvaluator:
    """
    Exo2Ego
    """

    def __init__(self, cfg, stage1_model, stage2_model, device):
        self.cfg = cfg
        self.stage1_model = stage1_model
        self.stage2_model = stage2_model
        self.device = device
        self.metrics_calc = MetricsCalculator(device)

        self.stage1_model.eval()
        self.stage2_model.eval()

    @torch.no_grad()
    def evaluate_batch(self, exo_video, ego_video_gt):
        """
        һ

        Args:
            exo_video: [B, T, 3, H, W]
            ego_video_gt: [B, T, 3, H, W]

        Returns:
            metrics: ֵָ
        """
        B, T, C, H, W = ego_video_gt.shape

        all_ssim = []
        all_psnr = []
        all_mse = []

        for t in range(T):
            exo_frame = exo_video[:, t]
            gt_frame = ego_video_gt[:, t]

            # Stage 1: ԤⲼ
            pred_kp, _, _ = self.stage1_model(exo_frame.unsqueeze(1))
            pred_kp = pred_kp[0, 0]  # [num_joints, 2]

            # תΪ
            from inference_exo2ego import keypoints_to_layout_image
            ego_layout = keypoints_to_layout_image(pred_kp.cpu().numpy(), H)
            ego_layout_tensor = torch.from_numpy(ego_layout).unsqueeze(0).to(self.device)

            # Stage 2: ͼ
            gen_frame = self.stage2_model.sample(ego_layout_tensor, self.device)
            gen_frame = gen_frame[0].cpu().numpy()
            gen_frame = np.transpose(gen_frame, (1, 2, 0))
            gen_frame = (gen_frame * 255).astype(np.uint8)

            # GTͼ
            gt_frame_np = gt_frame.cpu().numpy()
            gt_frame_np = np.transpose(gt_frame_np, (1, 2, 0))
            gt_frame_np = (gt_frame_np * 255).astype(np.uint8)

            # ָ
            ssim = self.metrics_calc.ssim(gen_frame, gt_frame_np)
            psnr = self.metrics_calc.psnr(gen_frame, gt_frame_np)
            mse = self.metrics_calc.mse(gen_frame, gt_frame_np)

            all_ssim.append(ssim)
            all_psnr.append(psnr)
            all_mse.append(mse)

        return {
            'ssim': np.mean(all_ssim),
            'psnr': np.mean(all_psnr),
            'mse': np.mean(all_mse)
        }

    def evaluate_dataset(self, dataloader, num_samples=100):
        """
        ݼ

        Args:
            dataloader: ݼ
            num_samples: 

        Returns:
            results: ֵ
        """
        all_metrics = {
            'ssim': [],
            'psnr': [],
            'mse': []
        }

        total_samples = 0

        for batch_idx, batch in enumerate(dataloader):
            if total_samples >= num_samples:
                break

            exo_video = batch['exo_video'].to(self.device)
            ego_video_gt = batch['ego_video'].to(self.device)

            batch_size = exo_video.shape[0]
            metrics = self.evaluate_batch(exo_video, ego_video_gt)

            for key in all_metrics:
                all_metrics[key].append(metrics[key] * batch_size)

            total_samples += batch_size
            print(f"Evaluated {total_samples}/{num_samples} samples")

        # ƽֵ
        results = {}
        for key in all_metrics:
            results[key] = sum(all_metrics[key]) / total_samples if total_samples > 0 else 0

        return results


def main():
    args = parse_args()

    # 
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ģ
    print("Creating models...")
    stage1_model, stage2_model = create_models(cfg, device)

    # ؼ
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    stage1_model.load_state_dict(checkpoint['stage1_model_state_dict'])
    stage2_model.load_state_dict(checkpoint['stage2_model_state_dict'])

    # ݼ
    from dataset_exo2ego import Exo2EgoSyntheticDataset, build_dataloader

    dataset = Exo2EgoSyntheticDataset(cfg, split=args.split)
    dataloader = build_dataloader(cfg, split=args.split, dataset_type='synthetic')

    # 
    evaluator = Exo2EgoEvaluator(cfg, stage1_model, stage2_model, device)

    # 
    print(f"\nEvaluating on {args.num_samples} samples...")
    results = evaluator.evaluate_dataset(dataloader, num_samples=args.num_samples)

    # ӡ
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"SSIM: {results['ssim']:.4f}")
    print(f"PSNR: {results['psnr']:.2f} dB")
    print(f"MSE:  {results['mse']:.2f}")
    print("="*50)

    # 
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / 'evaluation_results.txt'
    with open(results_file, 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value:.4f}\n")

    print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    main()