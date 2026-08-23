# -*- coding: utf-8 -*-
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import yaml
import argparse

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


def mpjpe(pred, gt):
    if pred.dim() == 4:
        pred = pred.reshape(-1, 21, 3)
        gt = gt.reshape(-1, 21, 3)
    per_joint = torch.norm(pred - gt, dim=-1)
    return per_joint.mean().item() * 1000


def pck(pred, gt, threshold=0.05):
    if pred.dim() == 4:
        pred = pred.reshape(-1, 21, 3)
        gt = gt.reshape(-1, 21, 3)
    per_joint = torch.norm(pred - gt, dim=-1)
    correct = (per_joint < threshold).float().mean().item() * 100
    return correct


def pa_epe(pred, gt):
    pred_np = pred.detach().cpu().numpy()
    gt_np = gt.detach().cpu().numpy()
    
    if pred_np.ndim == 4:
        B, T, J, C = pred_np.shape
        pred_np = pred_np.reshape(B * T, J, C)
        gt_np = gt_np.reshape(B * T, J, C)
    
    errors = []
    for i in range(pred_np.shape[0]):
        p = pred_np[i]
        g = gt_np[i]
        p_c = p - p.mean(axis=0)
        g_c = g - g.mean(axis=0)
        H = g_c.T @ p_c
        try:
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            p_aligned = (R @ p_c.T).T
            err = np.sqrt(((p_aligned - g_c) ** 2).sum(axis=1)).mean()
            errors.append(err)
        except:
            errors.append(0.1)
    
    return np.mean(errors) * 1000


def wrist_center(kp):
    if kp.dim() == 4:
        wrist = kp[..., 0:1, :]
        return kp - wrist
    elif kp.dim() == 3:
        wrist = kp[:, 0:1, :]
        return kp - wrist
    return kp


def eval_spl(device, max_batches=50):
    print('\n' + '=' * 70)
    print('[1/4] Evaluating SPL (Latent Space Keypoint)')
    print('=' * 70)
    
    sys.path.insert(0, '/data/data5/zhaoran/paper_code/spl')
    from config import load_config
    from dataset import build_dataloader
    from model import SPLLatentKeypoint
    
    cfg = load_config('/data/data5/zhaoran/paper_code/spl/config.yaml')
    
    model = SPLLatentKeypoint(cfg).to(device)
    
    ckpt_path = '/data/data5/zhaoran/paper_code/spl/check/best_model.pth'
    if not os.path.exists(ckpt_path):
        print(f'  [SKIP] Checkpoint not found: {ckpt_path}')
        return None
    
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    print(f'  Loaded: {ckpt_path}')
    
    val_loader = build_dataloader(cfg, split='val')
    
    all_mpjpe, all_pck, all_paepe = [], [], []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc='SPL')):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            
            exo_video = batch['exo_video'].to(device)
            gt_kp = batch['ego_keypoints'].to(device)
            
            output = model(exo_video, return_all_steps=False)
            if isinstance(output, dict):
                pred_kp = output['final_pose']
            else:
                pred_kp = output
            
            pred_kp = wrist_center(pred_kp)
            
            all_mpjpe.append(mpjpe(pred_kp, gt_kp))
            all_pck.append(pck(pred_kp, gt_kp))
            all_paepe.append(pa_epe(pred_kp, gt_kp))
    
    results = {
        'MPJPE': np.mean(all_mpjpe),
        'PCK': np.mean(all_pck),
        'PA-EPE': np.mean(all_paepe)
    }
    print(f'  SPL: MPJPE={results["MPJPE"]:.2f}mm, PCK={results["PCK"]:.2f}%, PA-EPE={results["PA-EPE"]:.2f}cm')
    return results


def eval_back(device, max_batches=50):
    print('\n' + '=' * 70)
    print('[2/4] Evaluating Back (Fourier + Geodesic)')
    print('=' * 70)
    
    sys.path.insert(0, '/data/data5/zhaoran/paper_code/back')
    from utils_fourier import load_config
    from dataset_dexycb import build_dataloader_dexycb
    from model_fourier_geodesic import Syn2SeqKeypointFourierGeodesic
    
    cfg = load_config('/data/data5/zhaoran/paper_code/back/config_fourier.yaml')
    
    model = Syn2SeqKeypointFourierGeodesic(cfg).to(device)
    
    ckpt_path = '/data/data5/zhaoran/paper_code/back/checkpoints/geodesic/best_model.pth'
    if not os.path.exists(ckpt_path):
        print(f'  [SKIP] Checkpoint not found: {ckpt_path}')
        return None
    
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    print(f'  Loaded: {ckpt_path}')
    
    val_loader = build_dataloader_dexycb(cfg, split='val')
    
    all_mpjpe, all_pck, all_paepe = [], [], []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc='Back')):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            
            exo_video = batch['exo_video'].to(device)
            gt_kp = batch['ego_keypoints'].to(device)
            
            pred_kp = model(exo_video)
            
            pred_kp = wrist_center(pred_kp)
            gt_kp = wrist_center(gt_kp)
            
            all_mpjpe.append(mpjpe(pred_kp, gt_kp))
            all_pck.append(pck(pred_kp, gt_kp))
            all_paepe.append(pa_epe(pred_kp, gt_kp))
    
    results = {
        'MPJPE': np.mean(all_mpjpe),
        'PCK': np.mean(all_pck),
        'PA-EPE': np.mean(all_paepe)
    }
    print(f'  Back: MPJPE={results["MPJPE"]:.2f}mm, PCK={results["PCK"]:.2f}%, PA-EPE={results["PA-EPE"]:.2f}cm')
    return results


def eval_baseline(device, max_batches=50):
    print('\n' + '=' * 70)
    print('[3/4] Evaluating Baseline (Syn2Seq)')
    print('=' * 70)
    
    sys.path.insert(0, '/data/data5/zhaoran/paper_code/baseline')
    from utils.config import load_config
    from datasets.dexycb_mv import build_dataloader
    from models.syn2seq import Syn2SeqKeypoint
    
    cfg = load_config('/data/data5/zhaoran/paper_code/baseline/config.yaml')
    
    model = Syn2SeqKeypoint(cfg).to(device)
    
    ckpt_path = '/data/data5/zhaoran/paper_code/baseline/checkpoints/best_model.pth'
    if not os.path.exists(ckpt_path):
        print(f'  [SKIP] Checkpoint not found: {ckpt_path}')
        return None
    
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    print(f'  Loaded: {ckpt_path}')
    
    val_loader = build_dataloader(cfg, split='val')
    
    all_mpjpe, all_pck, all_paepe = [], [], []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc='Baseline')):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            
            exo_video = batch['exo_video'].to(device)
            gt_kp = batch['ego_keypoints'].to(device)
            
            pred_kp = model(exo_video)
            
            all_mpjpe.append(mpjpe(pred_kp, gt_kp))
            all_pck.append(pck(pred_kp, gt_kp))
            all_paepe.append(pa_epe(pred_kp, gt_kp))
    
    results = {
        'MPJPE': np.mean(all_mpjpe),
        'PCK': np.mean(all_pck),
        'PA-EPE': np.mean(all_paepe)
    }
    print(f'  Baseline: MPJPE={results["MPJPE"]:.2f}mm, PCK={results["PCK"]:.2f}%, PA-EPE={results["PA-EPE"]:.2f}cm')
    return results


def eval_total(device, max_batches=50):
    print('\n' + '=' * 70)
    print('[4/4] Evaluating Total (Hand-Aware Latent + Geodesic)')
    print('=' * 70)
    
    sys.path.insert(0, '/data/data5/zhaoran/paper_code/total')
    sys.path.insert(0, '/data/data5/zhaoran/paper_code/back')
    from latent_hand_aware_v2 import HandAwareLatentModel
    from dataset_dexycb import DexYCBMultiViewReal
    from torch.utils.data import DataLoader
    
    with open('/data/data5/zhaoran/paper_code/total/config_fourier.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    
    embed_dim = cfg['model'].get('hidden_dim', 256)
    num_interp_steps = cfg['model'].get('interpolate_steps', 4)
    num_joints = cfg['dataset'].get('num_joints', 21)
    
    model = HandAwareLatentModel(
        embed_dim=embed_dim,
        num_interpolation_steps=num_interp_steps,
        num_joints=num_joints
    )
    
    ckpt_path = '/data/data5/zhaoran/paper_code/total/checkpoints/best_latent.pth'
    if not os.path.exists(ckpt_path):
        print(f'  [SKIP] Checkpoint not found: {ckpt_path}')
        return None
    
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()
    print(f'  Loaded: {ckpt_path}')
    
    val_dataset = DexYCBMultiViewReal(cfg, split='val')
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2)
    
    all_mpjpe, all_pck, all_paepe = [], [], []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc='Total')):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            
            exo_video = batch['exo_video']
            gt_kp = batch['ego_keypoints'].to(device)
            
            batch_size, seq_len, num_views, c, h, w = exo_video.shape
            
            ego_view_idx = 0
            single_view = exo_video[:, :, ego_view_idx]
            flat_images = single_view.reshape(batch_size * seq_len, c, h, w)
            
            if h != 224 or w != 224:
                flat_images = F.interpolate(flat_images, size=(224, 224), mode='bilinear', align_corners=False)
            
            flat_images = flat_images.to(device)
            flat_kp = gt_kp.reshape(batch_size * seq_len, -1, 3)
            
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                output = model(flat_images)
                pred_kp = output['final_pose']
            
            pred_kp = wrist_center(pred_kp)
            flat_kp = wrist_center(flat_kp)
            
            all_mpjpe.append(mpjpe(pred_kp, flat_kp))
            all_pck.append(pck(pred_kp, flat_kp))
            all_paepe.append(pa_epe(pred_kp, flat_kp))
    
    results = {
        'MPJPE': np.mean(all_mpjpe),
        'PCK': np.mean(all_pck),
        'PA-EPE': np.mean(all_paepe)
    }
    print(f'  Total: MPJPE={results["MPJPE"]:.2f}mm, PCK={results["PCK"]:.2f}%, PA-EPE={results["PA-EPE"]:.2f}cm')
    return results


def main():
    parser = argparse.ArgumentParser(description='Unified Evaluation for All Models')
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--max_batches', type=int, default=50)
    parser.add_argument('--models', type=str, nargs='+', default=['spl', 'back', 'baseline', 'total'],
                       choices=['spl', 'back', 'baseline', 'total'])
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    
    results = {}
    
    eval_funcs = {
        'spl': eval_spl,
        'back': eval_back,
        'baseline': eval_baseline,
        'total': eval_total,
    }
    
    for model_name in args.models:
        try:
            result = eval_funcs[model_name](device, max_batches=args.max_batches)
            if result is not None:
                results[model_name] = result
        except Exception as e:
            print(f'  [ERROR] Failed to evaluate {model_name}: {e}')
            import traceback
            traceback.print_exc()
    
    print('\n' + '=' * 80)
    print('COMPARISON TABLE (Wrist-Centered MPJPE)')
    print('=' * 80)
    print(f'{"Model":<25} {"MPJPE (mm)":<15} {"PCK@5cm (%)":<15} {"PA-EPE (cm)":<15}')
    print('-' * 70)
    
    for name, r in results.items():
        print(f'{name:<25} {r["MPJPE"]:<15.2f} {r["PCK"]:<15.2f} {r["PA-EPE"]:<15.2f}')
    
    print('=' * 80)
    
    if len(results) > 1:
        best_mpjpe = min(results.items(), key=lambda x: x[1]['MPJPE'])
        best_pck = max(results.items(), key=lambda x: x[1]['PCK'])
        print(f'\nBest MPJPE: {best_mpjpe[0]} ({best_mpjpe[1]["MPJPE"]:.2f}mm)')
        print(f'Best PCK:   {best_pck[0]} ({best_pck[1]["PCK"]:.2f}%)')


if __name__ == '__main__':
    main()
