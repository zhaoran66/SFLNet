import sys, torch, numpy as np, cv2, yaml, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, '/data/data5/zhaoran/paper_code/total')
sys.path.insert(0, '/data/data5/zhaoran')

device = torch.device('cuda:0')
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

def draw_3d_skeleton(ax, joints, color, title, alpha=1.0):
    """joints: [21,3]"""
    for i,j in HAND_CONNECTIONS:
        ax.plot([joints[i,0],joints[j,0]],
                [joints[i,1],joints[j,1]],
                [joints[i,2],joints[j,2]],
                color=color, linewidth=2, alpha=alpha)
    ax.scatter(joints[:,0], joints[:,1], joints[:,2],
               c=color, s=20, zorder=5)
    ax.set_title(title, fontsize=8, pad=2)
    ax.set_xlabel('X',fontsize=6); ax.set_ylabel('Y',fontsize=6); ax.set_zlabel('Z',fontsize=6)
    ax.tick_params(labelsize=5)

def gen_3d_comparison(gt, pred_base, pred_total, seq_idx, err_b, err_t):
    fig = plt.figure(figsize=(6, 2.5), facecolor='white')

    ax1 = fig.add_subplot(131, projection='3d')
    draw_3d_skeleton(ax1, gt, '#22aa22', f'GT')

    ax2 = fig.add_subplot(132, projection='3d')
    draw_3d_skeleton(ax2, pred_base, '#cc3333', f'Baseline\n{err_b:.1f}mm')

    ax3 = fig.add_subplot(133, projection='3d')
    draw_3d_skeleton(ax3, pred_total, '#1a7aaa', f'SFLNet\n{err_t:.1f}mm')

    # 统一坐标轴范围
    all_pts = np.vstack([gt, pred_base, pred_total])
    margin = 0.02
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(all_pts[:,0].min()-margin, all_pts[:,0].max()+margin)
        ax.set_ylim(all_pts[:,1].min()-margin, all_pts[:,1].max()+margin)
        ax.set_zlim(all_pts[:,2].min()-margin, all_pts[:,2].max()+margin)
        ax.view_init(elev=20, azim=-60)

    plt.suptitle(f'seq{seq_idx:03d} — 3D Skeleton Comparison', fontsize=9, y=1.01)
    plt.tight_layout(pad=0.5)

    fig.canvas.draw()
    w,h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h,w,4)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    return img

# 加载模型
from dataset_dexycb import DexYCBMultiViewReal
from latent_bone_guided import LatentHandPoseModel
from baseline_model import BaselineModel

cfg = yaml.safe_load(open('/data/data5/zhaoran/paper_code/total/config_fourier.yaml'))
dataset = DexYCBMultiViewReal(cfg, split='val')

total = LatentHandPoseModel(
    embed_dim=cfg['model']['hidden_dim'],
    num_interpolation_steps=cfg['model']['interpolate_steps'],
    num_joints=cfg['dataset']['num_joints'],
    num_views=len(cfg['dataset']['exo_views'])
).to(device)
ckpt=torch.load('/data/data5/zhaoran/paper_code/total/checkpoints/best_latent.pth',map_location=device)
total.load_state_dict(ckpt['model_state_dict']); total.eval()

baseline=BaselineModel().to(device)
ckpt_b=torch.load('/data/data5/zhaoran/paper_code/baseline/checkpoints/best_model.pth',map_location=device)
baseline.load_state_dict(ckpt_b['model_state_dict'],strict=False); baseline.eval()
print('Models loaded')

os.makedirs('/data/data5/zhaoran/paper_code/comparison_vis',exist_ok=True)
seq_list=[149,10,23]; t=8

rows=[]
for seq_idx in seq_list:
    sample=dataset[seq_idx]
    exo_video=sample['exo_video'].unsqueeze(0).to(device)
    exo_pose=sample['exo_pose'].unsqueeze(0).to(device)
    ego_pose=sample['ego_pose'].unsqueeze(0).to(device)
    gt_joints=sample['ego_keypoints'].numpy()

    with torch.no_grad():
        out=total(exo_video,exo_pose=exo_pose,ego_pose=ego_pose)
        if isinstance(out,tuple): out=out[0]
        pred_t=out[0].cpu().numpy()
        pred_b=baseline(exo_video)[0].cpu().numpy()

    gt_t=gt_joints[t]; pt=pred_t[t]; pb=pred_b[t]
    err_t=np.mean(np.sqrt(np.sum((pred_t-gt_joints)**2,axis=-1)))*1000
    err_b=np.mean(np.sqrt(np.sum((pred_b-gt_joints)**2,axis=-1)))*1000

    img=gen_3d_comparison(gt_t, pb, pt, seq_idx, err_b, err_t)
    rows.append(img)
    print(f'seq{seq_idx:03d}  Baseline={err_b:.1f}mm  SFLNet={err_t:.1f}mm')

# 拼成竖排
sep=np.ones((4,rows[0].shape[1],3),dtype=np.uint8)*200
grid=[]
for i,r in enumerate(rows):
    grid.append(r)
    if i<len(rows)-1: grid.append(sep)
combined=np.vstack(grid)
out_path='/data/data5/zhaoran/paper_code/comparison_vis/fig3_3d.jpg'
cv2.imwrite(out_path,combined,[cv2.IMWRITE_JPEG_QUALITY,95])
print(f'Saved: {out_path}')
