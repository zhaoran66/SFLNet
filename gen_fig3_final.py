import sys, torch, numpy as np, cv2, yaml, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, '/data/data5/zhaoran/paper_code/total')
sys.path.insert(0, '/data/data5/zhaoran/paper_code/ego_estimator')

device = torch.device('cuda:0')
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]
SIZE = 256
mean = torch.tensor([0.485,0.456,0.406]).view(1,1,3,1,1)
std  = torch.tensor([0.229,0.224,0.225]).view(1,1,3,1,1)

def draw_3d(ax, joints, color, label, lw=2.0, ms=28, alpha=1.0, ls='-'):
    for i,j in HAND_CONNECTIONS:
        ax.plot([joints[i,0],joints[j,0]],
                [joints[i,1],joints[j,1]],
                [joints[i,2],joints[j,2]],
                color=color, linewidth=lw, alpha=alpha, linestyle=ls)
    ax.scatter(joints[:,0],joints[:,1],joints[:,2],
               c=color, s=ms, zorder=5, edgecolors='white',
               linewidths=0.5, label=label, depthshade=True)

def render_3d_single(gt, pred, pred_color, pred_label, err, title, figsize=(3.5,3.5)):
    fig = plt.figure(figsize=figsize, facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#f5f5f5')

    all_pts = np.vstack([gt, pred])
    mid = (all_pts.max(0)+all_pts.min(0))/2
    rng = (all_pts.max(0)-all_pts.min(0)).max()*0.65 + 0.02
    for set_lim, m in zip([ax.set_xlim,ax.set_ylim,ax.set_zlim], mid):
        set_lim(m-rng, m+rng)

    # GT绿色虚线
    draw_3d(ax, gt, '#22aa22', 'GT', lw=1.8, ms=22, alpha=0.7, ls='--')
    # 预测实线
    draw_3d(ax, pred, pred_color, f'{pred_label}\n({err:.1f}mm)', lw=2.2, ms=32)

    ax.set_title(title, fontsize=9, fontweight='bold', pad=4)
    ax.view_init(elev=25, azim=-55)
    ax.tick_params(labelsize=5, pad=0)
    ax.set_xlabel('X',fontsize=6,labelpad=0)
    ax.set_ylabel('Y',fontsize=6,labelpad=0)
    ax.set_zlabel('Z',fontsize=6,labelpad=0)
    ax.legend(fontsize=8, loc='upper left', framealpha=0.85,
              markerscale=0.8, handlelength=1.5)
    ax.grid(True, alpha=0.25)

    plt.tight_layout(pad=0.4)
    fig.canvas.draw()
    w,h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.buffer_rgba(),dtype=np.uint8).reshape(h,w,4)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    return img

# 加载模型
from dataset_dexycb import DexYCBMultiViewReal
from latent_bone_guided import LatentHandPoseModel
from model import EgoNet

cfg = yaml.safe_load(open('/data/data5/zhaoran/paper_code/total/config_fourier.yaml'))
dataset = DexYCBMultiViewReal(cfg, split='val')

# SFLNet
total = LatentHandPoseModel(
    embed_dim=cfg['model']['hidden_dim'],
    num_interpolation_steps=cfg['model']['interpolate_steps'],
    num_joints=cfg['dataset']['num_joints'],
    num_views=len(cfg['dataset']['exo_views'])
).to(device)
ckpt = torch.load('/data/data5/zhaoran/paper_code/total/checkpoints/best_latent.pth', map_location=device)
total.load_state_dict(ckpt['model_state_dict']); total.eval()

# EgoNet（用exo帧作为伪ego输入，模拟Syn2Seq+EgoNet）
egonet = EgoNet(hidden_dim=256, num_joints=21).to(device)
ckpt_e = torch.load('/data/data5/zhaoran/paper_code/ego_estimator/best_ego_net.pth', map_location=device)
egonet.load_state_dict(ckpt_e, strict=False); egonet.eval()
print('Models loaded')

os.makedirs('/data/data5/zhaoran/paper_code/comparison_vis', exist_ok=True)
mean_d = mean.to(device); std_d = std.to(device)

seq_list = [149, 101, 117]; t = 8
grid_rows = []

for idx, seq_idx in enumerate(seq_list):
    seq = dataset.sequences[seq_idx]
    sample = dataset[seq_idx]
    exo_video = sample['exo_video'].unsqueeze(0).to(device)
    exo_pose  = sample['exo_pose'].unsqueeze(0).to(device)
    ego_pose  = sample['ego_pose'].unsqueeze(0).to(device)
    gt_joints = sample['ego_keypoints'].numpy()

    # 读取 exo 帧作为伪ego输入
    exo_frames = []
    for ti in range(len(seq['frames'])):
        frame_data = seq['frames'][ti]
        raw_exo = dataset.ds[frame_data[dataset.exo_views[0]]]
        img = cv2.imread(raw_exo['color_file'])
        img = cv2.resize(img, (SIZE,SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_t = torch.from_numpy(img).float().permute(2,0,1).unsqueeze(0)/255.0
        exo_frames.append(img_t)
    pseudo_ego = torch.cat(exo_frames,dim=0).unsqueeze(0).to(device)
    pseudo_ego = (pseudo_ego - mean_d) / std_d

    with torch.no_grad():
        out = total(exo_video, exo_pose=exo_pose, ego_pose=ego_pose)
        if isinstance(out, tuple): out = out[0]
        pred_sfl = out[0].cpu().numpy()
        pred_syn = egonet(pseudo_ego)[0].cpu().numpy()

    gt_t   = gt_joints[t]
    sfl_t  = pred_sfl[t]
    syn_t  = pred_syn[t]

    err_sfl = np.mean(np.sqrt(np.sum((pred_sfl-gt_joints)**2,axis=-1)))*1000
    err_syn = np.mean(np.sqrt(np.sum((pred_syn-gt_joints)**2,axis=-1)))*1000

    # exo 输入图
    frame_data = seq['frames'][t]
    raw_exo = dataset.ds[frame_data[dataset.exo_views[0]]]
    exo_img = cv2.resize(cv2.imread(raw_exo['color_file']), (SIZE,SIZE))
    cv2.putText(exo_img,f'seq{seq_idx:03d}',(8,22),
                cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,0),3)
    cv2.putText(exo_img,f'seq{seq_idx:03d}',(8,22),
                cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1)

    # 3D 骨架图
    img_sfl = render_3d_single(gt_t, sfl_t, '#e06010',
                               'SFLNet (Ours)', err_sfl,
                               f'SFLNet  {err_sfl:.1f}mm')
    img_syn = render_3d_single(gt_t, syn_t, '#cc3333',
                               'Baseline', err_syn,
                               f'Baseline  {err_syn:.1f}mm')

    H = img_sfl.shape[0]
    exo_r = cv2.resize(exo_img, (SIZE, H))
    sv = np.ones((H,3,3),dtype=np.uint8)*180

    row = np.hstack([exo_r, sv, img_sfl, sv, img_syn])
    grid_rows.append(row)
    if idx < len(seq_list)-1:
        grid_rows.append(np.ones((4,row.shape[1],3),dtype=np.uint8)*180)
    print(f'seq{seq_idx:03d}  SFLNet={err_sfl:.1f}mm  Syn2Seq+EgoNet={err_syn:.1f}mm')

# 标题行
W = grid_rows[0].shape[1]
title = np.ones((32,W,3),dtype=np.uint8)*245
cv2.putText(title,'Exo Input',(8,22),cv2.FONT_HERSHEY_SIMPLEX,0.52,(50,50,50),1)
cv2.putText(title,'SFLNet (Ours) vs GT',(SIZE+10,22),
            cv2.FONT_HERSHEY_SIMPLEX,0.52,(180,80,10),1)
cv2.putText(title,'Baseline vs GT',(SIZE+W//2+10,22),
            cv2.FONT_HERSHEY_SIMPLEX,0.52,(30,30,180),1)

# 图例
legend = np.ones((28,W,3),dtype=np.uint8)*245
cv2.circle(legend,(16,14),6,(0,180,0),-1)
cv2.putText(legend,'Ground Truth',(26,18),cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,100,0),1)
cv2.circle(legend,(160,14),6,(10,80,220),-1)
cv2.putText(legend,'SFLNet (Ours)',(172,18),cv2.FONT_HERSHEY_SIMPLEX,0.42,(10,80,180),1)
cv2.circle(legend,(310,14),6,(30,30,200),-1)
cv2.putText(legend,'Baseline',(322,18),cv2.FONT_HERSHEY_SIMPLEX,0.42,(30,30,160),1)

final = np.vstack([
    title,
    np.ones((2,W,3),dtype=np.uint8)*180,
    *grid_rows,
    np.ones((2,W,3),dtype=np.uint8)*180,
    legend
])
out_path = '/data/data5/zhaoran/paper_code/comparison_vis/fig3_final.jpg'
cv2.imwrite(out_path, final, [cv2.IMWRITE_JPEG_QUALITY,95])
print(f'Saved: {out_path}  size={final.shape}')
