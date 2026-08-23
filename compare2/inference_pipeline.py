import sys
import torch
import numpy as np
import time
import torch.nn.functional as F
sys.path.insert(0, '/data/data5/zhaoran/paper_code/total')
sys.path.insert(0, '/data/data5/zhaoran/paper_code/ego_estimator')
sys.path.insert(0, '/data/data5/zhaoran/paper_code/compare2')
from dataset_dexycb import DexYCBMultiViewReal
from model import EgoNet
from model_syn2seq import Syn2SeqForcing
import yaml

def compute_pck(pred, gt, threshold=0.05):
    dist = torch.norm(pred - gt, dim=-1)
    return (dist < threshold).float().mean().item()

def compute_pa_metrics(pred, gt):
    pred_c = pred - pred.mean(dim=0, keepdim=True)
    gt_c = gt - gt.mean(dim=0, keepdim=True)
    dist = torch.norm(pred_c - gt_c, dim=-1)
    epe = dist.mean().item() * 100
    thresholds = torch.linspace(0, 10.0, 100)
    pck_curve = [(dist*100 < t).float().mean().item() for t in thresholds]
    auc = np.trapz(pck_curve, thresholds.numpy()) / 10.0 * 100
    return epe, auc

def run():
    cfg_syn = yaml.safe_load(open('/data/data5/zhaoran/paper_code/compare2/config_syn2seq.yaml'))
    device = torch.device('cuda:0')

    val_cfg = yaml.safe_load(open('/data/data5/zhaoran/paper_code/total/config_fourier.yaml'))
    val_dataset = DexYCBMultiViewReal(val_cfg, split='val')
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=4, shuffle=False, num_workers=0)

    # 加载 Syn2Seq
    syn2seq = Syn2SeqForcing(cfg_syn).to(device)
    ckpt = torch.load('/data/data5/zhaoran/paper_code/compare2/outputs/syn2seq/best_model.pth',
                      map_location=device)
    syn2seq.load_state_dict(ckpt['model_state_dict'])
    syn2seq.eval()

    # 加载 EgoNet
    ego_net = EgoNet(hidden_dim=256, num_joints=21).to(device)
    ego_net.load_state_dict(
        torch.load('/data/data5/zhaoran/paper_code/ego_estimator/best_ego_net.pth',
                   map_location=device))
    ego_net.eval()

    # 参数量
    syn_params = sum(p.numel() for p in syn2seq.parameters()) / 1e6
    ego_params = sum(p.numel() for p in ego_net.parameters()) / 1e6
    print(f"Syn2Seq参数量: {syn_params:.2f}M")
    print(f"EgoNet参数量:  {ego_params:.2f}M")
    print(f"Total参数量:   {syn_params+ego_params:.2f}M")

    all_mpjpe = []
    all_pck = []
    all_pa_epe = []
    all_pa_auc = []
    all_time = []
    img_size = cfg_syn['dataset']['image_size'][0]

    with torch.no_grad():
        for batch in val_loader:
            exo_video = batch['exo_video'].to(device)  # [B,T,1,3,H,W]
            gt_kp = batch['ego_keypoints'].to(device)
            B, T, V, C, H, W = exo_video.shape

            t_start = time.time()

            # Syn2Seq推理：生成伪ego视频
            exo_resized = F.interpolate(
                exo_video.squeeze(2).reshape(B*T, C, H, W),
                size=(img_size, img_size), mode='bilinear', align_corners=False
            ).reshape(B, T, C, img_size, img_size)

            pseudo_ego = syn2seq(exo_resized)  # [B,T,3,img_size,img_size]

            # resize回原始尺寸
            pseudo_ego = F.interpolate(
                pseudo_ego.reshape(B*T, C, img_size, img_size),
                size=(H, W), mode='bilinear', align_corners=False
            ).reshape(B, T, C, H, W)

            # EgoNet预测关节点
            pred_kp = ego_net(pseudo_ego)  # [B,T,21,3]

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
                    pa_epe, pa_auc = compute_pa_metrics(pred_centered[b,t], gt_kp[b,t])
                    all_pa_epe.append(pa_epe)
                    all_pa_auc.append(pa_auc)

            print(f"Batch MPJPE: {mpjpe.item():.2f}mm")

    print(f"\n{'='*60}")
    print(f"Syn2Seq+EgoNet Pipeline Results:")
    print(f"推理时间: {np.mean(all_time):.2f} ms/frame")
    print(f"MPJPE:   {np.mean(all_mpjpe):.2f} mm")
    print(f"PCK@5cm: {np.mean(all_pck):.2f} %")
    print(f"PA-EPE:  {np.mean(all_pa_epe):.2f} cm")
    print(f"PA-AUC:  {np.mean(all_pa_auc):.2f} %")
    print(f"{'='*60}")

if __name__ == '__main__':
    run()
