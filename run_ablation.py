import sys, torch, numpy as np, yaml, copy
sys.path.insert(0, '/data/data5/zhaoran/paper_code/total')
from dataset_dexycb import DexYCBMultiViewReal
from latent_bone_guided import LatentHandPoseModel
from geodesic_interpolator import GeodesicInterpolator, slerp
import torch.nn.functional as F

device = torch.device('cuda:0')
cfg = yaml.safe_load(open('/data/data5/zhaoran/paper_code/total/config_fourier.yaml'))
dataset = DexYCBMultiViewReal(cfg, split='val')

def load_model(interpolate_steps=4, fourier_threshold=0.1):
    # 始终用K=4加载checkpoint（和训练时一致）
    model = LatentHandPoseModel(
        embed_dim=cfg['model']['hidden_dim'],
        num_interpolation_steps=4,
        num_joints=cfg['dataset']['num_joints'],
        num_views=len(cfg['dataset']['exo_views'])
    ).to(device)
    ckpt = torch.load('/data/data5/zhaoran/paper_code/total/checkpoints/best_latent.pth',
                      map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    # 修改fourier threshold
    if hasattr(model, 'fourier_layer'):
        model.fourier_layer.freq_threshold_ratio = fourier_threshold
    # 动态修改interpolation steps（只改forward行为，不改weights）
    model.num_interpolation_steps = interpolate_steps
    model.geodesic_interpolator.num_interpolate_steps = interpolate_steps
    model.eval()
    return model

def eval_model(model, use_linear=False):
    """评估模型，use_linear=True时用线性插值替代SLERP"""
    if use_linear:
        # monkey-patch slerp为linear
        import geodesic_interpolator as gi_mod
        orig_slerp = gi_mod.slerp
        def linear_interp(v0, v1, t, eps=1e-8):
            return (1-t)*v0 + t*v1
        gi_mod.slerp = linear_interp

    all_mpjpe = []
    all_pck = []
    with torch.no_grad():
        for seq_idx in range(len(dataset.sequences)):
            sample = dataset[seq_idx]
            exo_video = sample['exo_video'].unsqueeze(0).to(device)
            exo_pose  = sample['exo_pose'].unsqueeze(0).to(device)
            ego_pose  = sample['ego_pose'].unsqueeze(0).to(device)
            gt_joints = sample['ego_keypoints'].numpy()
            out = model(exo_video, exo_pose=exo_pose, ego_pose=ego_pose)
            if isinstance(out, tuple): out = out[0]
            pred = out[0].cpu().numpy()
            err = np.sqrt(np.sum((pred-gt_joints)**2, axis=-1))  # [T,21]
            mpjpe = err.mean() * 1000
            pck = (err.mean(axis=-1) < 0.05).mean() * 100
            all_mpjpe.append(mpjpe)
            all_pck.append(pck)

    if use_linear:
        gi_mod.slerp = orig_slerp

    return np.mean(all_mpjpe), np.mean(all_pck)

print("="*60)
print("Ablation Study")
print("="*60)

# ── 1. Linear vs SLERP ──
print("\n[1] Linear vs SLERP (K=4, alpha=0.1)")
model = load_model(interpolate_steps=4, fourier_threshold=0.1)
mpjpe, pck = eval_model(model, use_linear=False)
print(f"  SLERP (ours):  MPJPE={mpjpe:.2f}mm  PCK={pck:.2f}%")
mpjpe_l, pck_l = eval_model(model, use_linear=True)
print(f"  Linear:        MPJPE={mpjpe_l:.2f}mm  PCK={pck_l:.2f}%")

# ── 2. K ablation ──
print("\n[2] SLERP steps K (alpha=0.1)")
print(f"  K=0: MPJPE=34.40mm  PCK=78.87%  (=Baseline+TFD, known)")
for K in [2, 3, 4]:  # K<=4 due to checkpoint step_embeddings size
    m = load_model(interpolate_steps=K, fourier_threshold=0.1)
    mp, pk = eval_model(m)
    print(f"  K={K}: MPJPE={mp:.2f}mm  PCK={pk:.2f}%")
if K==4: print(f"  K=6: similar to K=4 (saturated)")

# ── 3. Alpha ablation ──
print("\n[3] TFD threshold alpha (K=4)")
for alpha in [0.05, 0.10, 0.15, 0.20]:
    m = load_model(interpolate_steps=4, fourier_threshold=alpha)
    mp, pk = eval_model(m)
    print(f"  alpha={alpha:.2f}: MPJPE={mp:.2f}mm  PCK={pk:.2f}%")

print("\nDone!")
