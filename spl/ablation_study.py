# -*- coding: utf-8 -*-
import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config
from losses import MPJPE, PCK
from dataset import build_dataloader
from model import SPLLatentKeypoint


def compute_pa_epe(pred, gt):
    if pred.dim() == 4:
        B, T, J, C = pred.shape
        pred = pred.reshape(B * T, J, C)
        gt = gt.reshape(B * T, J, C)
    pred_center = pred.mean(dim=1, keepdim=True)
    gt_center = gt.mean(dim=1, keepdim=True)
    pred_aligned = pred - pred_center
    gt_aligned = gt - gt_center
    epe = torch.norm(pred_aligned - gt_aligned, p=2, dim=-1)
    pa_epe = epe.mean(dim=-1) * 100
    return pa_epe


def compute_pa_auc(errors, max_threshold=10.0):
    thresholds = torch.linspace(0, max_threshold, 100)
    pck_curve = []
    for thresh in thresholds:
        pck_thresh = (errors < thresh).float().mean()
        pck_curve.append(pck_thresh)
    pck_curve = torch.tensor(pck_curve)
    auc = torch.trapz(pck_curve, thresholds) / max_threshold * 100
    return auc.item()


def wrist_center(kp):
    wrist = kp[..., 0:1, :]
    return kp - wrist


def run_eval(model, val_loader, device, max_batches=50, hook_fn=None):
    model.eval()
    mpjpe_metric = MPJPE()
    pck_metric = PCK()
    all_mpjpe, all_pck, all_pa_epe = [], [], []

    count = 0
    with torch.no_grad():
        for batch in val_loader:
            if count >= max_batches:
                break
            exo_video = batch['exo_video'].to(device)
            gt_kp = batch['ego_keypoints'].to(device)
            ego_kp_gt = batch['ego_keypoints'].to(device)

            if hook_fn:
                pred_kp = hook_fn(model, exo_video, ego_kp_gt)
            else:
                output = model(exo_video)
                pred_kp = output['final_pose'] if isinstance(output, dict) else output

            pred_kp = wrist_center(pred_kp)
            gt_kp = wrist_center(gt_kp)

            all_mpjpe.append(mpjpe_metric(pred_kp, gt_kp).item())
            all_pck.append(pck_metric(pred_kp, gt_kp).item())
            all_pa_epe.extend(compute_pa_epe(pred_kp, gt_kp).cpu().numpy())
            count += 1

    pa_epe_t = torch.tensor(all_pa_epe)
    pa_auc = compute_pa_auc(pa_epe_t)
    return {
        'MPJPE': f'{np.mean(all_mpjpe):.2f} mm',
        'PCK@0.05': f'{np.mean(all_pck):.2f} %',
        'PA-EPE': f'{pa_epe_t.mean().item():.2f} cm',
        'PA-AUC': f'{pa_auc:.2f} %',
    }


def ablation_no_interpolation(model, exo_video, ego_kp_gt):
    """Ablation 1: bypass GeodesicInterpolator, directly use exo_global"""
    spatial_features, exo_global = model.feature_extractor(exo_video)

    full_sequence = exo_global.unsqueeze(2)

    num_total_steps = full_sequence.shape[2]
    full_sequence_flat = full_sequence.view(exo_global.shape[0], -1, model.hidden_dim)
    full_sequence_flat = model.seq_norm(full_sequence_flat)
    full_sequence_flat = full_sequence_flat + model.pos_encoding[:full_sequence_flat.shape[1], :].unsqueeze(0)

    encoded = model.sequence_encoder(full_sequence_flat)
    encoded = encoded.view(exo_global.shape[0], -1, num_total_steps, model.hidden_dim)

    step_feat = encoded[:, :, 0, :]
    keypoints = model.keypoint_decoder(step_feat, spatial_features)
    return keypoints


def ablation_mean_ego_endpoint(model, exo_video, ego_kp_gt):
    """Ablation 2: use mean_ego_feat from training set instead of learnable_ego_anchor"""
    spatial_features, exo_global = model.feature_extractor(exo_video)

    ego_features = model.feature_extractor.encode_ego_keypoints(ego_kp_gt)
    full_sequence = model.geodesic_interpolator(exo_global, ego_features)

    num_total_steps = full_sequence.shape[2]
    full_sequence_flat = full_sequence.view(exo_global.shape[0], -1, model.hidden_dim)
    full_sequence_flat = model.seq_norm(full_sequence_flat)
    full_sequence_flat = full_sequence_flat + model.pos_encoding[:full_sequence_flat.shape[1], :].unsqueeze(0)

    encoded = model.sequence_encoder(full_sequence_flat)
    encoded = encoded.view(exo_global.shape[0], -1, num_total_steps, model.hidden_dim)

    all_step_keypoints = []
    for step_idx in range(num_total_steps):
        start_idx = max(0, step_idx - 1)
        end_idx = min(num_total_steps, step_idx + 2)
        context_feat = encoded[:, :, start_idx:end_idx, :].flatten(2, 3)
        step_feat = encoded[:, :, step_idx, :]
        decoder_input = torch.cat([step_feat, context_feat], dim=-1)[:, :, :model.hidden_dim]
        keypoints = model.keypoint_decoder(decoder_input, spatial_features)
        all_step_keypoints.append(keypoints)

    return all_step_keypoints[-1]


def ablation_gate_fixed_one(model, exo_video, ego_kp_gt):
    """Ablation 3: force query_gate = 1.0 (only query path, no direct MLP)"""
    original_gate = model.keypoint_decoder.query_gate.data.clone()
    model.keypoint_decoder.query_gate.data.fill_(100.0)

    output = model(exo_video)
    pred_kp = output['final_pose'] if isinstance(output, dict) else output

    model.keypoint_decoder.query_gate.data.copy_(original_gate)
    return pred_kp


def ablation_intermediate_steps(model, exo_video, ego_kp_gt):
    """Ablation 4: evaluate each interpolation step separately"""
    spatial_features, exo_global = model.feature_extractor(exo_video)

    ego_features = model.feature_extractor.encode_ego_keypoints(ego_kp_gt)
    full_sequence = model.geodesic_interpolator(exo_global, ego_features)

    num_total_steps = full_sequence.shape[2]
    full_sequence_flat = full_sequence.view(exo_global.shape[0], -1, model.hidden_dim)
    full_sequence_flat = model.seq_norm(full_sequence_flat)
    full_sequence_flat = full_sequence_flat + model.pos_encoding[:full_sequence_flat.shape[1], :].unsqueeze(0)

    encoded = model.sequence_encoder(full_sequence_flat)
    encoded = encoded.view(exo_global.shape[0], -1, num_total_steps, model.hidden_dim)

    step_results = {}
    for step_idx in range(num_total_steps):
        start_idx = max(0, step_idx - 1)
        end_idx = min(num_total_steps, step_idx + 2)
        context_feat = encoded[:, :, start_idx:end_idx, :].flatten(2, 3)
        step_feat = encoded[:, :, step_idx, :]
        decoder_input = torch.cat([step_feat, context_feat], dim=-1)[:, :, :model.hidden_dim]
        keypoints = model.keypoint_decoder(decoder_input, spatial_features)
        step_results[f'Step {step_idx}'] = keypoints

    return step_results


def main():
    cfg = load_config('config.yaml')
    device = torch.device(cfg.training.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    model = SPLLatentKeypoint(cfg).to(device)
    ckpt_path = 'check/best_model.pth'
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    print(f'Loaded checkpoint: epoch={ckpt.get("epoch", "?")}')

    train_loader = build_dataloader(cfg, split='train')
    model.compute_mean_ego_features(train_loader, device)

    val_loader = build_dataloader(cfg, split='val')
    max_batches = 50

    print('\n' + '=' * 70)
    print('SPL ABLATION STUDY')
    print('=' * 70)

    results = {}

    # --- Full model (current) ---
    print('\n[1/5] Full SPL model (with learnable_ego_anchor)...')
    r = run_eval(model, val_loader, device, max_batches)
    results['Full SPL (current)'] = r
    print(f'  MPJPE: {r["MPJPE"]}, PCK: {r["PCK@0.05"]}, PA-EPE: {r["PA-EPE"]}, PA-AUC: {r["PA-AUC"]}')

    # --- Ablation 1: No interpolation ---
    print('\n[2/5] Ablation 1: No Geodesic Interpolation (direct exo_global)...')
    r = run_eval(model, val_loader, device, max_batches, hook_fn=ablation_no_interpolation)
    results['No Interpolation'] = r
    print(f'  MPJPE: {r["MPJPE"]}, PCK: {r["PCK@0.05"]}, PA-EPE: {r["PA-EPE"]}, PA-AUC: {r["PA-AUC"]}')

    # --- Ablation 2: Mean ego endpoint ---
    print('\n[3/5] Ablation 2: mean_ego_feat instead of learnable_ego_anchor...')
    r = run_eval(model, val_loader, device, max_batches, hook_fn=ablation_mean_ego_endpoint)
    results['Mean Ego Endpoint'] = r
    print(f'  MPJPE: {r["MPJPE"]}, PCK: {r["PCK@0.05"]}, PA-EPE: {r["PA-EPE"]}, PA-AUC: {r["PA-AUC"]}')

    # --- Ablation 3: Gate = 1.0 ---
    print('\n[4/5] Ablation 3: Gate fixed to 1.0 (query path only)...')
    r = run_eval(model, val_loader, device, max_batches, hook_fn=ablation_gate_fixed_one)
    results['Gate=1.0 (query only)'] = r
    print(f'  MPJPE: {r["MPJPE"]}, PCK: {r["PCK@0.05"]}, PA-EPE: {r["PA-EPE"]}, PA-AUC: {r["PA-AUC"]}')

    # --- Ablation 4: Per-step MPJPE ---
    print('\n[5/5] Ablation 4: Per-step MPJPE (with ego endpoint)...')
    mpjpe_metric = MPJPE()
    step_mpjpes = {f'Step {i}': [] for i in range(5)}
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            if count >= max_batches:
                break
            exo_video = batch['exo_video'].to(device)
            gt_kp = wrist_center(batch['ego_keypoints'].to(device))
            ego_kp_gt = batch['ego_keypoints'].to(device)
            step_results = ablation_intermediate_steps(model, exo_video, ego_kp_gt)
            for step_name, pred_kp in step_results.items():
                pred_centered = wrist_center(pred_kp)
                step_mpjpes[step_name].append(mpjpe_metric(pred_centered, gt_kp).item())
            count += 1

    print('  Per-step MPJPE:')
    for step_name, vals in step_mpjpes.items():
        mean_mpjpe = np.mean(vals)
        print(f'    {step_name}: {mean_mpjpe:.2f} mm')
        results[f'PerStep_{step_name}'] = {'MPJPE': f'{mean_mpjpe:.2f} mm'}

    # --- Summary ---
    print('\n' + '=' * 70)
    print('ABLATION SUMMARY')
    print('=' * 70)
    print(f'{"Configuration":<30} {"MPJPE":<12} {"PCK@0.05":<12} {"PA-EPE":<12} {"PA-AUC":<12}')
    print('-' * 70)
    for name, r in results.items():
        if name.startswith('PerStep'):
            print(f'  {name:<28} {r["MPJPE"]:<12}')
        else:
            print(f'  {name:<28} {r["MPJPE"]:<12} {r["PCK@0.05"]:<12} {r["PA-EPE"]:<12} {r["PA-AUC"]:<12}')

    # --- Diagnosis ---
    full_mpjpe = float(results['Full SPL (current)']['MPJPE'].replace(' mm', ''))
    no_interp_mpjpe = float(results['No Interpolation']['MPJPE'].replace(' mm', ''))
    mean_ego_mpjpe = float(results['Mean Ego Endpoint']['MPJPE'].replace(' mm', ''))
    gate1_mpjpe = float(results['Gate=1.0 (query only)']['MPJPE'].replace(' mm', ''))

    print('\n' + '=' * 70)
    print('DIAGNOSIS')
    print('=' * 70)

    if no_interp_mpjpe < full_mpjpe * 0.8:
        print('[!] Interpolation is HURTING performance!')
        print(f'    No interp: {no_interp_mpjpe:.1f}mm vs Full: {full_mpjpe:.1f}mm')
        print('    => Interpolation path may be introducing noise')
    elif no_interp_mpjpe > full_mpjpe * 1.2:
        print('[+] Interpolation is HELPING performance!')
        print(f'    No interp: {no_interp_mpjpe:.1f}mm vs Full: {full_mpjpe:.1f}mm')
        print('    => Geodesic path provides useful intermediate supervision')
    else:
        print('[~] Interpolation has NEUTRAL effect')
        print(f'    No interp: {no_interp_mpjpe:.1f}mm vs Full: {full_mpjpe:.1f}mm')

    if mean_ego_mpjpe < full_mpjpe * 0.8:
        print('[!] learnable_ego_anchor is WORSE than mean_ego_feat!')
        print(f'    Mean ego: {mean_ego_mpjpe:.1f}mm vs Anchor: {full_mpjpe:.1f}mm')
        print('    => Anchor is not learning useful ego prior')
    elif mean_ego_mpjpe > full_mpjpe * 1.2:
        print('[+] learnable_ego_anchor is BETTER than mean_ego_feat!')
        print(f'    Mean ego: {mean_ego_mpjpe:.1f}mm vs Anchor: {full_mpjpe:.1f}mm')
    else:
        print('[~] Anchor and mean_ego_feat perform similarly')
        print(f'    Mean ego: {mean_ego_mpjpe:.1f}mm vs Anchor: {full_mpjpe:.1f}mm')

    if gate1_mpjpe < full_mpjpe * 0.8:
        print('[!] Direct MLP path is HURTING performance!')
        print(f'    Gate=1.0: {gate1_mpjpe:.1f}mm vs Gate=learned: {full_mpjpe:.1f}mm')
        print('    => Remove direct_mlp or force gate=1')
    elif gate1_mpjpe > full_mpjpe * 1.2:
        print('[+] Direct MLP path is HELPING as regularizer!')
        print(f'    Gate=1.0: {gate1_mpjpe:.1f}mm vs Gate=learned: {full_mpjpe:.1f}mm')
    else:
        print('[~] Gate has NEUTRAL effect')
        print(f'    Gate=1.0: {gate1_mpjpe:.1f}mm vs Gate=learned: {full_mpjpe:.1f}mm')

    print('=' * 70)


if __name__ == '__main__':
    main()
