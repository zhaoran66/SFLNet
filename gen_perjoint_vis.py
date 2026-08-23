import sys, torch, numpy as np, cv2, yaml, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, '/data/data5/zhaoran/paper_code/total')
sys.path.insert(0, '/data/data5/zhaoran')

device = torch.device('cuda:0')

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

# 21个关节的2D布局（用于热力图骨架展示）
# 标准手部布局坐标 [x, y]，y轴朝上
JOINT_LAYOUT = np.array([
    [0.0,  0.0],   # 0 wrist
    [-0.4, 0.5],   # 1 thumb_mcp
    [-0.6, 0.85],  # 2 thumb_pip
    [-0.75,1.15],  # 3 thumb_dip
    [-0.85,1.4],   # 4 thumb_tip
    [-0.2, 0.6],   # 5 index_mcp
    [-0.2, 1.0],   # 6 index_pip
    [-0.2, 1.3],   # 7 index_dip
    [-0.2, 1.55],  # 8 index_tip
    [0.0,  0.6],   # 9 middle_mcp
    [0.0,  1.05],  # 10 middle_pip
    [0.0,  1.35],  # 11 middle_dip
    [0.0,  1.6],   # 12 middle_tip
    [0.2,  0.6],   # 13 ring_mcp
    [0.2,  1.0],   # 14 ring_pip
    [0.2,  1.28],  # 15 ring_dip
    [0.2,  1.5],   # 16 ring_tip
    [0.4,  0.55],  # 17 pinky_mcp
    [0.4,  0.88],  # 18 pinky_pip
    [0.4,  1.12],  # 19 pinky_dip
    [0.4,  1.3],   # 20 pinky_tip
])

def draw_perjoint_skeleton(errors_per_joint, title, vmax=80, fig_size=(2.2, 3.2)):
    """画带热力图颜色的骨架，errors_per_joint: [21] mm"""
    cmap = plt.cm.RdYlGn_r  # 绿→黄→红
    norm = mcolors.Normalize(vmin=0, vmax=vmax)

    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    pts = JOINT_LAYOUT.copy()
    pts[:, 1] = -pts[:, 1]  # flip y

    # 画骨骼连接线
    for i, j in HAND_CONNECTIONS:
        x = [pts[i,0], pts[j,0]]
        y = [pts[i,1], pts[j,1]]
        err_avg = (errors_per_joint[i] + errors_per_joint[j]) / 2
        color = cmap(norm(err_avg))
        ax.plot(x, y, color=color, linewidth=3.0, solid_capstyle='round')

    # 画关节点
    for idx in range(21):
        color = cmap(norm(errors_per_joint[idx]))
        ax.scatter(pts[idx,0], pts[idx,1], c=[color],
                  s=90, zorder=5, edgecolors='#333333', linewidths=0.8)

    mean_err = errors_per_joint.mean()
    ax.set_title(f'Mean: {mean_err:.1f} mm', color='black', fontsize=10, fontweight='bold', pad=4)
    ax.set_xlim(-1.1, 0.7)
    ax.set_ylim(-1.8, 0.2)
    ax.axis('off')
    ax.set_aspect('equal')

    # colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('mm', color='black', fontsize=8)
    cbar.ax.yaxis.set_tick_params(color='black', labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='black')
    cbar.outline.set_edgecolor('black')

    plt.tight_layout(pad=0.3)

    # 转成numpy图像
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    return img

# 加载数据集
from dataset_dexycb import DexYCBMultiViewReal
from latent_bone_guided import LatentHandPoseModel
from baseline_model import BaselineModel

cfg = yaml.safe_load(open('/data/data5/zhaoran/paper_code/total/config_fourier.yaml'))
dataset = DexYCBMultiViewReal(cfg, split='val')

# 加载 SFLNet
total = LatentHandPoseModel(
    embed_dim=cfg['model']['hidden_dim'],
    num_interpolation_steps=cfg['model']['interpolate_steps'],
    num_joints=cfg['dataset']['num_joints'],
    num_views=len(cfg['dataset']['exo_views'])
).to(device)
ckpt = torch.load('/data/data5/zhaoran/paper_code/total/checkpoints/best_latent.pth', map_location=device)
total.load_state_dict(ckpt['model_state_dict'])
total.eval()

# 加载 Baseline
baseline = BaselineModel().to(device)
ckpt_b = torch.load('/data/data5/zhaoran/paper_code/baseline/checkpoints/best_model.pth', map_location=device)
baseline.load_state_dict(ckpt_b['model_state_dict'], strict=False)
baseline.eval()
print('Models loaded')

os.makedirs('/data/data5/zhaoran/paper_code/perjoint_vis', exist_ok=True)

seq_list = [149, 10, 23]
t = 8

for seq_idx in seq_list:
    seq = dataset.sequences[seq_idx]
    sample = dataset[seq_idx]
    exo_video = sample['exo_video'].unsqueeze(0).to(device)
    exo_pose  = sample['exo_pose'].unsqueeze(0).to(device)
    ego_pose  = sample['ego_pose'].unsqueeze(0).to(device)
    gt_joints = sample['ego_keypoints'].numpy()  # [T,21,3]

    with torch.no_grad():
        out = total(exo_video, exo_pose=exo_pose, ego_pose=ego_pose)
        if isinstance(out, tuple): out = out[0]
        pred_total = out[0].cpu().numpy()

        out_b = baseline(exo_video)
        pred_base = out_b[0].cpu().numpy()

    # per-joint error at frame t
    err_total = np.sqrt(np.sum((pred_total[t] - gt_joints[t])**2, axis=-1)) * 1000  # [21]
    err_base  = np.sqrt(np.sum((pred_base[t]  - gt_joints[t])**2, axis=-1)) * 1000  # [21]

    # 生成热力图
    img_base  = draw_perjoint_skeleton(err_base,  f'Baseline seq{seq_idx}')
    img_total = draw_perjoint_skeleton(err_total, f'SFLNet seq{seq_idx}')

    cv2.imwrite(f'/data/data5/zhaoran/paper_code/perjoint_vis/seq{seq_idx:03d}_baseline.png', img_base)
    cv2.imwrite(f'/data/data5/zhaoran/paper_code/perjoint_vis/seq{seq_idx:03d}_sflnet.png',   img_total)

    print(f'seq{seq_idx:03d}  Baseline mean={err_base.mean():.1f}mm  SFLNet mean={err_total.mean():.1f}mm')

print('Done! Files in /data/data5/zhaoran/paper_code/perjoint_vis/')

# ── 拼合：3行×2列（Baseline | SFLNet）──
all_imgs = []
for seq_idx in seq_list:
    img_b = cv2.imread(f'/data/data5/zhaoran/paper_code/perjoint_vis/seq{seq_idx:03d}_baseline.png')
    img_s = cv2.imread(f'/data/data5/zhaoran/paper_code/perjoint_vis/seq{seq_idx:03d}_sflnet.png')
    # 统一高度
    h = max(img_b.shape[0], img_s.shape[0])
    def pad_h(img, h):
        if img.shape[0] < h:
            pad = np.ones((h-img.shape[0], img.shape[1], 3), dtype=np.uint8)*255
            img = np.vstack([img, pad])
        return img
    img_b = pad_h(img_b, h)
    img_s = pad_h(img_s, h)
    sep = np.ones((h, 4, 3), dtype=np.uint8)*200
    row = np.hstack([img_b, sep, img_s])
    all_imgs.append(row)

# 列标题
W = all_imgs[0].shape[1]
title_h = 36
title_row = np.ones((title_h, W, 3), dtype=np.uint8)*255
half_w = (W-4)//2
cv2.putText(title_row, 'Baseline', (half_w//2 - 50, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180,60,60), 2)
cv2.putText(title_row, 'SFLNet (Ours)', (half_w + 4 + half_w//2 - 80, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30,130,30), 2)

# 行分隔线
sep_h = np.ones((3, W, 3), dtype=np.uint8)*200

grid = [title_row]
for i, row in enumerate(all_imgs):
    grid.append(row)
    if i < len(all_imgs)-1:
        grid.append(sep_h)

combined = np.vstack(grid)
out_path = '/data/data5/zhaoran/paper_code/perjoint_vis/combined_perjoint.png'
cv2.imwrite(out_path, combined)
print(f'Combined saved: {out_path}  size={combined.shape}')
