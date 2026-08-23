import sys, torch, numpy as np, cv2, yaml, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, '/data/data5/zhaoran/paper_code/total')
sys.path.insert(0, '/data/data5/zhaoran')

device = torch.device('cuda:0')
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

def draw_3d_compare(gt, pred, pred_label, pred_color, err_mm,
                    elev=20, azim=-60, figsize=(3.2,3.2)):
    fig = plt.figure(figsize=figsize, facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')

    # 统一轴范围
    all_pts = np.vstack([gt, pred])
    margin = 0.02
    rng = max(all_pts.max(axis=0)-all_pts.min(axis=0)) + margin*2
    mid = (all_pts.max(axis=0)+all_pts.min(axis=0))/2
    for set_lim, m in zip([ax.set_xlim,ax.set_ylim,ax.set_zlim], mid):
        set_lim(m-rng/2, m+rng/2)

    # GT骨架（绿，细虚线）
    for i,j in HAND_CONNECTIONS:
        ax.plot([gt[i,0],gt[j,0]],[gt[i,1],gt[j,1]],[gt[i,2],gt[j,2]],
                color='#22aa22', linewidth=1.5, alpha=0.7, linestyle='--')
    ax.scatter(gt[:,0],gt[:,1],gt[:,2],c='#22aa22',s=25,zorder=4,
               edgecolors='#005500',linewidths=0.5,label='GT')

    # 预测骨架（实线，粗）
    for i,j in HAND_CONNECTIONS:
        ax.plot([pred[i,0],pred[j,0]],[pred[i,1],pred[j,1]],[pred[i,2],pred[j,2]],
                color=pred_color, linewidth=2.0, alpha=0.9)
    ax.scatter(pred[:,0],pred[:,1],pred[:,2],c=pred_color,s=35,zorder=5,
               edgecolors='black',linewidths=0.5,label=pred_label)

    ax.set_title(f'{pred_label}\nMPJPE={err_mm:.1f} mm',
                 fontsize=9, fontweight='bold', pad=4,
                 color=pred_color)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('X',fontsize=6,labelpad=1)
    ax.set_ylabel('Y',fontsize=6,labelpad=1)
    ax.set_zlabel('Z',fontsize=6,labelpad=1)
    ax.tick_params(labelsize=5, pad=0)
    ax.legend(fontsize=7, loc='upper left', framealpha=0.7)
    ax.grid(True, alpha=0.3)

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
from baseline_model import BaselineModel

cfg = yaml.safe_load(open('/data/data5/zhaoran/paper_code/total/config_fourier.yaml'))
dataset = DexYCBMultiViewReal(cfg, split='val')

total = LatentHandPoseModel(
    embed_dim=cfg['model']['hidden_dim'],
    num_interpolation_steps=cfg['model']['interpolate_steps'],
    num_joints=cfg['dataset']['num_joints'],
    num_views=len(cfg['dataset']['exo_views'])
).to(device)
ckpt = torch.load('/data/data5/zhaoran/paper_code/total/checkpoints/best_latent.pth', map_location=device)
total.load_state_dict(ckpt['model_state_dict']); total.eval()

baseline = BaselineModel().to(device)
ckpt_b = torch.load('/data/data5/zhaoran/paper_code/baseline/checkpoints/best_model.pth', map_location=device)
baseline.load_state_dict(ckpt_b['model_state_dict'], strict=False); baseline.eval()
print('Models loaded')

os.makedirs('/data/data5/zhaoran/paper_code/comparison_vis', exist_ok=True)
SIZE = 256
seq_list = [149, 10, 23]; t = 8

grid_rows = []
sep_v = np.ones((320, 3, 3), dtype=np.uint8)*180

for idx, seq_idx in enumerate(seq_list):
    seq = dataset.sequences[seq_idx]
    sample = dataset[seq_idx]
    exo_video = sample['exo_video'].unsqueeze(0).to(device)
    exo_pose  = sample['exo_pose'].unsqueeze(0).to(device)
    ego_pose  = sample['ego_pose'].unsqueeze(0).to(device)
    gt_joints = sample['ego_keypoints'].numpy()

    with torch.no_grad():
        out = total(exo_video, exo_pose=exo_pose, ego_pose=ego_pose)
        if isinstance(out, tuple): out = out[0]
        pred_t = out[0].cpu().numpy()
        pred_b = baseline(exo_video)[0].cpu().numpy()

    gt_t  = gt_joints[t]
    pt    = pred_t[t]
    pb    = pred_b[t]
    err_t = np.mean(np.sqrt(np.sum((pred_t-gt_joints)**2,axis=-1)))*1000
    err_b = np.mean(np.sqrt(np.sum((pred_b-gt_joints)**2,axis=-1)))*1000

    # exo图
    frame_data = seq['frames'][t]
    raw_exo = dataset.ds[frame_data[dataset.exo_views[0]]]
    exo_img = cv2.resize(cv2.imread(raw_exo['color_file']), (SIZE,SIZE))
    # 加序列标签
    cv2.putText(exo_img,f'seq{seq_idx:03d}',(8,22),
                cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,0),3)
    cv2.putText(exo_img,f'seq{seq_idx:03d}',(8,22),
                cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1)

    # 3D 骨架图
    img_sfl  = draw_3d_compare(gt_t, pt, 'SFLNet', '#e06010', err_t)
    img_base = draw_3d_compare(gt_t, pb, 'Baseline', '#3030cc', err_b)

    # resize 到相同高度
    H = 320
    exo_r  = cv2.resize(exo_img, (SIZE, H))
    sfl_r  = cv2.resize(img_sfl,  (H, H))
    base_r = cv2.resize(img_base, (H, H))

    row = np.hstack([exo_r, sep_v, sfl_r, sep_v, base_r])
    grid_rows.append(row)
    if idx < len(seq_list)-1:
        grid_rows.append(np.ones((4, row.shape[1], 3), dtype=np.uint8)*180)
    print(f'seq{seq_idx:03d}  Baseline={err_b:.1f}mm  SFLNet={err_t:.1f}mm')

# 标题行
W = grid_rows[0].shape[1]
title_h = 32
title = np.ones((title_h, W, 3), dtype=np.uint8)*245
cv2.putText(title,'Exo Input',(10,22),cv2.FONT_HERSHEY_SIMPLEX,0.55,(50,50,50),1)
cv2.putText(title,'SFLNet (Ours) vs GT  [3D]',(SIZE+10,22),
            cv2.FONT_HERSHEY_SIMPLEX,0.55,(180,80,10),1)
cv2.putText(title,'Baseline vs GT  [3D]',(SIZE+323+10,22),
            cv2.FONT_HERSHEY_SIMPLEX,0.55,(30,30,180),1)

final = np.vstack([
    title,
    np.ones((2,W,3),dtype=np.uint8)*180,
    *grid_rows
])

out_path = '/data/data5/zhaoran/paper_code/comparison_vis/fig3_3d.jpg'
cv2.imwrite(out_path, final, [cv2.IMWRITE_JPEG_QUALITY,95])
print(f'Saved: {out_path}  size={final.shape}')
