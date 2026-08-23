import sys
import torch
import numpy as np
import time
import torch.nn.functional as F
sys.path.insert(0, '/data/data5/zhaoran/paper_code/total')
sys.path.insert(0, '/data/data5/zhaoran/paper_code/ego_estimator')
from dataset_dexycb import DexYCBMultiViewReal
from model import EgoNet
import yaml

def compute_pck(pred, gt, threshold=0.05):
    dist = torch.norm(pred - gt, dim=-1)
    return (dist < threshold).float().mean().item()

def compute_pa_metrics(pred, gt):
    # 和evaluate.py保持一致：只做平移对齐
    pred_c = pred - pred.mean(dim=0, keepdim=True)
    gt_c = gt - gt.mean(dim=0, keepdim=True)
    dist = torch.norm(pred_c - gt_c, dim=-1)
    epe = dist.mean().item() * 100  # cm
    # PA-AUC: max_threshold=10cm
    thresholds = torch.linspace(0, 10.0, 100)
    pck_curve = [(dist*100 < t).float().mean().item() for t in thresholds]
    auc = np.trapz(pck_curve, thresholds.numpy()) / 10.0 * 100
    return epe, auc

def run_pipeline():
    cfg = yaml.safe_load(open('/data/data5/zhaoran/paper_code/compare1/config_exo2ego.yaml'))
    device = torch.device('cuda:0')

    val_cfg = yaml.safe_load(open('/data/data5/zhaoran/paper_code/total/config_fourier.yaml'))
    val_dataset = DexYCBMultiViewReal(val_cfg, split='val')
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=4, shuffle=False, num_workers=0)

    # 加载 Stage1
    from model_layout_transformer import SimplifiedLayoutTranslator
    cfg_stage1 = cfg.copy()
    cfg_stage1['model'] = cfg['model'].copy()
    cfg_stage1['model']['hidden_dim'] = 128
    cfg_stage1['model']['num_layers'] = 2
    stage1 = SimplifiedLayoutTranslator(cfg_stage1).to(device)
    ckpt = torch.load('outputs/exo2ego/checkpoints/stage1_final.pth', map_location=device)
    stage1.load_state_dict(ckpt['stage1_model_state_dict'], strict=True)
    stage1.eval()

    # 加载 Stage2
    from model_diffusion import create_diffusion_model
    stage2 = create_diffusion_model(cfg).to(device)
    ckpt2 = torch.load('outputs/exo2ego/checkpoints/stage2_final.pth', map_location=device)
    stage2.load_state_dict(ckpt2['stage2_model_state_dict'], strict=False)
    stage2.eval()

    # 加载 EgoNet
    ego_net = EgoNet(hidden_dim=256, num_joints=21).to(device)
    ego_net.load_state_dict(
        torch.load('/data/data5/zhaoran/paper_code/ego_estimator/best_ego_net.pth', map_location=device))
    ego_net.eval()

    # 统计参数量
    stage1_params = sum(p.numel() for p in stage1.parameters()) / 1e6
    stage2_params = sum(p.numel() for p in stage2.parameters()) / 1e6
    egonet_params = sum(p.numel() for p in ego_net.parameters()) / 1e6
    total_params = stage1_params + stage2_params + egonet_params
    print(f"Stage1 参数量: {stage1_params:.2f}M")
    print(f"Stage2 参数量: {stage2_params:.2f}M")
    print(f"EgoNet 参数量: {egonet_params:.2f}M")
    print(f"Total  参数量: {total_params:.2f}M")

    all_mpjpe = []
    all_pck = []
    all_pa_epe = []
    all_pa_auc = []
    all_time = []

    with torch.no_grad():
        for batch in val_loader:
            exo_video = batch['exo_video'].to(device)
            gt_kp = batch['ego_keypoints'].to(device)
            B, T, V, C, H, W = exo_video.shape
            exo_for_stage1 = exo_video.squeeze(2)

            t_start = time.time()

            # Stage1
            _, pred_heatmaps, _ = stage1(exo_for_stage1)

            # Stage2
            pseudo_ego_frames = []
            for t in range(T):
                exo_frame = exo_for_stage1[:, t]
                exo_resized = F.interpolate(exo_frame, size=(64, 64),
                                            mode='bilinear', align_corners=False)
                pseudo_frame = stage2.sample(exo_resized, device)
                pseudo_frame_decoded = stage2.vae.decode(pseudo_frame)
                pseudo_ego_frames.append(pseudo_frame_decoded)

            pseudo_ego = torch.stack(pseudo_ego_frames, dim=1)
            B2, T2, C2, h, w = pseudo_ego.shape
            pseudo_ego = F.interpolate(
                pseudo_ego.reshape(B2*T2, C2, h, w),
                size=(H, W), mode='bilinear', align_corners=False
            ).reshape(B2, T2, C2, H, W)

            # EgoNet
            pred_kp = ego_net(pseudo_ego)

            t_end = time.time()
            all_time.append((t_end - t_start) / (B * T) * 1000)

            pred_wrist = pred_kp[:, :, 0:1, :]
            pred_centered = pred_kp - pred_wrist
            mpjpe = torch.norm(pred_centered - gt_kp, dim=-1).mean() * 1000
            pck = compute_pck(pred_centered, gt_kp)
            all_mpjpe.append(mpjpe.item())
            all_pck.append(pck * 100)

            for b in range(B):
                for t in range(T):
                    pa_epe, pa_auc = compute_pa_metrics(pred_centered[b, t], gt_kp[b, t])
                    all_pa_epe.append(pa_epe)
                    all_pa_auc.append(pa_auc)

            print(f"Batch MPJPE: {mpjpe.item():.2f}mm  PCK: {pck*100:.1f}%")

    print(f"\n{'='*60}")
    print(f"PMYS+EgoNet Pipeline Results:")
    print(f"参数量:  {total_params:.2f}M")
    print(f"推理时间: {np.mean(all_time):.2f} ms/frame")
    print(f"MPJPE:   {np.mean(all_mpjpe):.2f} mm")
    print(f"PCK@5cm: {np.mean(all_pck):.2f} %")
    print(f"PA-EPE:  {np.mean(all_pa_epe):.2f} cm")
    print(f"PA-AUC:  {np.mean(all_pa_auc):.2f} %")
    print(f"{'='*60}")

if __name__ == '__main__':
    run_pipeline()
