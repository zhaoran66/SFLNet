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

def draw_3d_skel(ax, joints, gt, pred_color, elev, azim, title):
    all_pts = np.vstack([joints, gt])
    mid=(all_pts.max(0)+all_pts.min(0))/2
    rng=(all_pts.max(0)-all_pts.min(0)).max()*0.65+0.02
    for sl,m in zip([ax.set_xlim,ax.set_ylim,ax.set_zlim],mid):
        sl(m-rng,m+rng)
    # GT绿虚线
    for i,j in HAND_CONNECTIONS:
        ax.plot([gt[i,0],gt[j,0]],[gt[i,1],gt[j,1]],[gt[i,2],gt[j,2]],
                color='#22aa22',lw=1.5,alpha=0.6,ls='--')
    ax.scatter(gt[:,0],gt[:,1],gt[:,2],c='#22aa22',s=20,alpha=0.6,depthshade=False)
    # 预测实线
    for i,j in HAND_CONNECTIONS:
        ax.plot([joints[i,0],joints[j,0]],[joints[i,1],joints[j,1]],[joints[i,2],joints[j,2]],
                color=pred_color,lw=2.5)
    ax.scatter(joints[:,0],joints[:,1],joints[:,2],
               c=pred_color,s=40,edgecolors='white',linewidths=0.5,depthshade=False,zorder=5)
    ax.set_title(title,fontsize=9,fontweight='bold',pad=3)
    ax.view_init(elev=elev,azim=azim)
    ax.tick_params(labelsize=4,pad=0)
    ax.set_xlabel('X',fontsize=5,labelpad=0)
    ax.set_ylabel('Y',fontsize=5,labelpad=0)
    ax.set_zlabel('Z',fontsize=5,labelpad=0)
    ax.grid(True,alpha=0.2)
    ax.set_facecolor('#f5f5f5')

# 6个视角：从exo侧视(elev=10,azim=90) 渐变到 ego俯视(elev=70,azim=-30)
VIEWPOINTS = [
    (10,  90,  '#3355ff', r'$\alpha$=0.0'+'\nExo view'),
    (22,  66,  '#4477dd', r'$\alpha$=0.2'),
    (34,  42,  '#5599bb', r'$\alpha$=0.4'),
    (46,  18,  '#77bb66', r'$\alpha$=0.6'),
    (58,  -6,  '#99cc33', r'$\alpha$=0.8'),
    (70, -30,  '#ee6610', r'$\alpha$=1.0'+'\nEgo view\n(SFLNet)'),
]

seq_list = [149, 101]
t = 8

fig = plt.figure(figsize=(14, 5.0), facecolor='white')
fig.suptitle('SFLNet Geodesic Interpolation: Progressive Viewpoint Transition (Exo→Ego)',
             fontsize=11, fontweight='bold', y=1.02)

n_rows = len(seq_list)
n_cols = 7  # exo图 + 6个3D视角

for row_idx, seq_idx in enumerate(seq_list):
    seq = dataset.sequences[seq_idx]
    sample = dataset[seq_idx]
    exo_video = sample['exo_video'].unsqueeze(0).to(device)
    exo_pose  = sample['exo_pose'].unsqueeze(0).to(device)
    ego_pose  = sample['ego_pose'].unsqueeze(0).to(device)
    gt_joints = sample['ego_keypoints'].numpy()

    with torch.no_grad():
        out = total(exo_video, exo_pose=exo_pose, ego_pose=ego_pose)
        if isinstance(out,tuple): out=out[0]
        pred_sfl = out[0].cpu().numpy()

    gt_t   = gt_joints[t]
    pred_t = pred_sfl[t]
    err = np.mean(np.sqrt(np.sum((pred_sfl-gt_joints)**2,axis=-1)))*1000

    # exo图
    frame_data = seq['frames'][t]
    raw_exo = dataset.ds[frame_data[dataset.exo_views[0]]]
    exo_img = cv2.resize(cv2.imread(raw_exo['color_file']),(256,256))
    exo_img = cv2.cvtColor(exo_img,cv2.COLOR_BGR2RGB)

    # 第1列：exo图
    ax_img = fig.add_subplot(n_rows, n_cols, row_idx*n_cols+1)
    ax_img.imshow(exo_img)
    ax_img.set_title(f'Exo Input\nseq{seq_idx:03d}',fontsize=8,fontweight='bold')
    ax_img.axis('off')

    # 6个3D视角
    for ai,(elev,azim,color,label) in enumerate(VIEWPOINTS):
        ax = fig.add_subplot(n_rows,n_cols,row_idx*n_cols+ai+2,projection='3d')
        title = label
        if ai==5:
            title += f'\n{err:.1f}mm'
        draw_3d_skel(ax, pred_t, gt_t, color, elev, azim, title)

    print(f'seq{seq_idx:03d}  MPJPE={err:.1f}mm done')

# 图例
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
legend_elements = [
    Line2D([0],[0],color='#22aa22',lw=1.5,ls='--',label='Ground Truth (GT)'),
    Patch(facecolor='#3355ff',label='α=0.0  Exo viewpoint'),
    Patch(facecolor='#ee6610',label='α=1.0  Ego viewpoint (SFLNet output)'),
]
fig.legend(handles=legend_elements,loc='lower center',ncol=3,
           fontsize=9,bbox_to_anchor=(0.5,-0.04),framealpha=0.9)

plt.tight_layout(pad=0.5)
out_path='/data/data5/zhaoran/paper_code/comparison_vis/fig3_viewpoint_3d.jpg'
plt.savefig(out_path,dpi=150,bbox_inches='tight',facecolor='white')
plt.close()
print(f'Saved: {out_path}')
