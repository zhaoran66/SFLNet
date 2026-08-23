import sys, torch, numpy as np, cv2, yaml, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, '/data/data5/zhaoran/paper_code/total')
device = torch.device('cuda:0')

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

from dataset_dexycb import DexYCBMultiViewReal
from latent_bone_guided import LatentHandPoseModel

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
print('Model loaded')

os.makedirs('/data/data5/zhaoran/paper_code/comparison_vis',exist_ok=True)

def get_camera_matrix(extrinsic_12):
    """从[12]外参向量提取R[3x3]和t[3]"""
    ext = extrinsic_12.reshape(3,4)
    R = ext[:,:3]
    t = ext[:,3]
    return R, t

def transform_joints(joints_world, R, t):
    """把世界坐标关节点变换到相机坐标系"""
    return (R @ joints_world.T).T + t

def project_to_2d(joints_cam, intr, img_size=256, orig_w=640, orig_h=480):
    """把相机坐标系关节点投影到2D"""
    fx,fy=intr['fx'],intr['fy']
    ppx,ppy=intr['ppx'],intr['ppy']
    sx,sy=img_size/orig_w, img_size/orig_h
    Z=np.clip(joints_cam[:,2],0.01,10.0)
    u=(fx*joints_cam[:,0]/Z+ppx)*sx
    v=(fy*joints_cam[:,1]/Z+ppy)*sy
    return np.stack([u,v],axis=1)

def draw_skel_2d_white(kp2d, color, size=256):
    img=np.ones((size,size,3),dtype=np.uint8)*255
    for i,j in HAND_CONNECTIONS:
        x1,y1=int(kp2d[i,0]),int(kp2d[i,1])
        x2,y2=int(kp2d[j,0]),int(kp2d[j,1])
        if 0<=x1<size and 0<=y1<size and 0<=x2<size and 0<=y2<size:
            cv2.line(img,(x1,y1),(x2,y2),color,3)
    for i in range(21):
        x,y=int(kp2d[i,0]),int(kp2d[i,1])
        if 0<=x<size and 0<=y<size:
            cv2.circle(img,(x,y),6,color,-1)
            cv2.circle(img,(x,y),6,(0,0,0),1)
    return img

ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
COLORS_BGR = [
    (255,80,30),   # 蓝（exo）
    (255,120,60),
    (180,160,80),
    (80,180,120),
    (40,200,80),
    (30,180,30),   # 绿（ego/SFLNet）
]
LABELS = [r'$\alpha$=0 (Exo)', r'$\alpha$=0.2', r'$\alpha$=0.4',
          r'$\alpha$=0.6', r'$\alpha$=0.8', r'$\alpha$=1 (SFLNet)']

seq_list = [149, 101]
t = 8
SIZE = 256

# 图像布局：每行=一个序列，7列=exo图+6个视角
n_rows = len(seq_list)
n_cols = 7
cell_w, cell_h = SIZE, SIZE

fig_w = n_cols * (cell_w/80) + 0.5
fig_h = n_rows * (cell_h/80) + 1.5
fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                          facecolor='white')
if n_rows == 1:
    axes = [axes]

fig.suptitle('SFLNet: Geodesic Viewpoint Interpolation (Exo→Ego)',
             fontsize=12, fontweight='bold', y=1.02)

for row_idx, seq_idx in enumerate(seq_list):
    seq = dataset.sequences[seq_idx]
    sample = dataset[seq_idx]
    exo_video = sample['exo_video'].unsqueeze(0).to(device)
    exo_pose  = sample['exo_pose'].unsqueeze(0).to(device)
    ego_pose  = sample['ego_pose'].unsqueeze(0).to(device)
    gt_joints = sample['ego_keypoints'].numpy()  # [T,21,3] wrist-centered in ego cam

    with torch.no_grad():
        out = total(exo_video, exo_pose=exo_pose, ego_pose=ego_pose)
        if isinstance(out, tuple): out = out[0]
        pred_sfl = out[0].cpu().numpy()  # [T,21,3]

    gt_t   = gt_joints[t]   # [21,3] in ego cam space
    pred_t = pred_sfl[t]    # [21,3] in ego cam space (wrist-centered)

    # 获取exo和ego的外参
    frame_data = seq['frames'][t]
    raw_exo = dataset.ds[frame_data[dataset.exo_views[0]]]
    raw_ego = dataset.ds[frame_data[dataset.ego_view]]

    exo_ext = sample['exo_pose'][t, 0].numpy()  # [12]
    ego_ext = sample['ego_pose'][t].numpy()      # [12]

    R_exo, t_exo = get_camera_matrix(exo_ext)
    R_ego, t_ego = get_camera_matrix(ego_ext)

    intr_exo = raw_exo['intrinsics']
    intr_ego = raw_ego['intrinsics']

    # GT关节点在ego坐标系，加回wrist得到ego坐标系绝对坐标
    data_ego = np.load(raw_ego['label_file'])
    j3d_ego  = data_ego['joint_3d'].squeeze()  # [21,3] 绝对坐标 in ego cam
    wrist_ego = j3d_ego[0]

    # SFLNet预测的绝对坐标（ego cam）
    pred_abs = pred_t + wrist_ego
    gt_abs   = gt_t   + wrist_ego  # == j3d_ego

    # ego cam → world
    # ego cam: X_cam = R_ego @ X_world + t_ego
    # → X_world = R_ego^T @ (X_cam - t_ego)
    R_ego_inv = R_ego.T
    pred_world = (R_ego_inv @ (pred_abs - t_ego).T).T
    gt_world   = (R_ego_inv @ (gt_abs   - t_ego).T).T

    # 对每个alpha，插值R和t，把world坐标投影到中间视角
    exo_img = cv2.resize(cv2.imread(raw_exo['color_file']), (SIZE,SIZE))
    exo_img = cv2.cvtColor(exo_img, cv2.COLOR_BGR2RGB)

    # 第0列：exo图
    ax = axes[row_idx][0]
    ax.imshow(exo_img)
    ax.set_title(f'Exo Input\nseq{seq_idx:03d}', fontsize=8, fontweight='bold')
    ax.axis('off')

    # 6个插值视角
    for ai, alpha in enumerate(ALPHAS):
        # 插值相机参数
        R_interp = (1-alpha)*R_exo + alpha*R_ego
        # 正交化 R
        U,_,Vt = np.linalg.svd(R_interp)
        R_interp = U @ Vt
        t_interp = (1-alpha)*t_exo + alpha*t_ego

        # 插值内参
        fx = (1-alpha)*intr_exo['fx'] + alpha*intr_ego['fx']
        fy = (1-alpha)*intr_exo['fy'] + alpha*intr_ego['fy']
        ppx= (1-alpha)*intr_exo['ppx']+ alpha*intr_ego['ppx']
        ppy= (1-alpha)*intr_exo['ppy']+ alpha*intr_ego['ppy']
        intr_interp={'fx':fx,'fy':fy,'ppx':ppx,'ppy':ppy}

        # 投影 SFLNet 预测（α=1时用pred，α=0时用gt在exo视角）
        pts_world = (1-alpha)*gt_world + alpha*pred_world
        pts_cam = transform_joints(pts_world, R_interp, t_interp)
        kp2d = project_to_2d(pts_cam, intr_interp)

        skel_img = draw_skel_2d_white(kp2d, COLORS_BGR[ai])
        skel_rgb = cv2.cvtColor(skel_img, cv2.COLOR_BGR2RGB)

        ax = axes[row_idx][ai+1]
        ax.imshow(skel_rgb)

        err_t = np.mean(np.sqrt(np.sum((pred_sfl-gt_joints)**2,axis=-1)))*1000
        title = LABELS[ai]
        if ai == 5:
            title += f'\nMPJPE={err_t:.1f}mm'
        ax.set_title(title, fontsize=8, fontweight='bold',
                     color=('green' if ai==5 else ('blue' if ai==0 else 'black')))
        ax.axis('off')

    print(f'seq{seq_idx:03d} done')

# 图例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=(30/255,180/255,30/255), label='α=1.0  SFLNet (Ego view)'),
    Patch(facecolor=(255/255,80/255,30/255), label='α=0.0  Exo view'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=2,
           fontsize=9, bbox_to_anchor=(0.5,-0.04), framealpha=0.9)

plt.tight_layout(pad=0.4)
out_path = '/data/data5/zhaoran/paper_code/comparison_vis/fig3_viewpoint.jpg'
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved: {out_path}')
