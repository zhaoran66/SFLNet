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

def slerp_R(R0,R1,alpha):
    R_rel=R1@R0.T
    cos_t=np.clip((np.trace(R_rel)-1)/2,-1,1)
    theta=np.arccos(abs(cos_t))
    if theta<1e-6: return R0.copy()
    K=(R_rel-R_rel.T)/(2*np.sin(theta))
    return (np.eye(3)+np.sin(alpha*theta)*K+(1-np.cos(alpha*theta))*(K@K))@R0

def draw_cam_frustum(ax, R, t, color, scale=0.05, alpha=1.0):
    """画相机视锥"""
    center=-R.T@t
    # 相机朝向（Z轴）
    z_dir=R.T[:,2]*scale
    # 画视锥
    corners_cam=np.array([[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])*scale*0.6
    corners_world=(R.T@corners_cam.T).T+center
    # 连线：center→4个角
    for c in corners_world:
        ax.plot([center[0],c[0]],[center[1],c[1]],[center[2],c[2]],
                color=color,lw=1.0,alpha=alpha*0.7)
    # 画矩形
    for i in range(4):
        c1=corners_world[i]; c2=corners_world[(i+1)%4]
        ax.plot([c1[0],c2[0]],[c1[1],c2[1]],[c1[2],c2[2]],
                color=color,lw=1.5,alpha=alpha*0.8)
    # 中心点
    ax.scatter(*center,c=color,s=60,zorder=6,edgecolors='white',linewidths=0.8,alpha=alpha)

ALPHAS=[0.0,0.2,0.4,0.6,0.8,1.0]
COLORS=['#2244cc','#3B6EC7','#5499B0','#6DBB88','#86CC55','#ee6610']
LABELS=['α=0\n(Exo)','α=0.2','α=0.4','α=0.6','α=0.8','α=1\n(Ego)']

seq_list=[149,101]; t=8
fig=plt.figure(figsize=(14,5.5),facecolor='white')
fig.suptitle('SFLNet: Geodesic Camera Interpolation — Exo→Ego in 3D Space',
             fontsize=11,fontweight='bold',y=1.01)

for row_idx,seq_idx in enumerate(seq_list):
    seq=dataset.sequences[seq_idx]
    sample=dataset[seq_idx]
    exo_video=sample['exo_video'].unsqueeze(0).to(device)
    exo_pose=sample['exo_pose'].unsqueeze(0).to(device)
    ego_pose=sample['ego_pose'].unsqueeze(0).to(device)
    gt_joints=sample['ego_keypoints'].numpy()

    with torch.no_grad():
        out=total(exo_video,exo_pose=exo_pose,ego_pose=ego_pose)
        if isinstance(out,tuple): out=out[0]
        pred_sfl=out[0].cpu().numpy()

    exo_ext=sample['exo_pose'][t,0].numpy()
    ego_ext=sample['ego_pose'][t].numpy()
    R_exo=exo_ext[:9].reshape(3,3); t_exo=exo_ext[9:]
    R_ego=ego_ext[:9].reshape(3,3);  t_ego=ego_ext[9:]

    frame_data=seq['frames'][t]
    raw_ego=dataset.ds[frame_data[dataset.ego_view]]
    data_ego=np.load(raw_ego['label_file'])
    j3d_ego_abs=data_ego['joint_3d'].squeeze()
    wrist_ego=j3d_ego_abs[0]
    pred_abs=pred_sfl[t]+wrist_ego
    R_ego_inv=R_ego.T
    pred_world=(R_ego_inv@(pred_abs-t_ego).T).T
    gt_abs=gt_joints[t]+wrist_ego
    gt_world=(R_ego_inv@(gt_abs-t_ego).T).T
    err=np.mean(np.sqrt(np.sum((pred_sfl[t]-gt_joints[t])**2,axis=-1)))*1000

    # 相机中心轨迹
    cam_centers=np.array([-slerp_R(R_exo,R_ego,a).T@((1-a)*t_exo+a*t_ego)
                           for a in np.linspace(0,1,30)])

    # exo图（缩略）
    raw_exo=dataset.ds[frame_data[dataset.exo_views[0]]]
    exo_img=cv2.resize(cv2.imread(raw_exo['color_file']),(120,90))
    exo_img=cv2.cvtColor(exo_img,cv2.COLOR_BGR2RGB)

    # 主图：3D场景
    ax=fig.add_subplot(1,2,row_idx+1,projection='3d')
    ax.set_facecolor('#f0f2f8')

    # 手部骨架放大显示
    hand_center = gt_world.mean(0)
    gt_vis = (gt_world - hand_center)*8 + hand_center
    pred_vis = (pred_world - hand_center)*8 + hand_center
    for i,j in HAND_CONNECTIONS:
        ax.plot([gt_vis[i,0],gt_vis[j,0]],
                [gt_vis[i,1],gt_vis[j,1]],
                [gt_vis[i,2],gt_vis[j,2]],
                color='#22aa22',lw=2.5,alpha=0.9,zorder=3)
    ax.scatter(gt_vis[:,0],gt_vis[:,1],gt_vis[:,2],
               c='#22aa22',s=30,depthshade=False,zorder=4)

    # SFLNet预测（世界坐标）
    for i,j in HAND_CONNECTIONS:
        ax.plot([pred_vis[i,0],pred_vis[j,0]],
                [pred_vis[i,1],pred_vis[j,1]],
                [pred_vis[i,2],pred_vis[j,2]],
                color='#ee6610',lw=2.0,alpha=0.7,zorder=3,ls='--')
    ax.scatter(pred_vis[:,0],pred_vis[:,1],pred_vis[:,2],
               c='#ee6610',s=20,depthshade=False,zorder=4,alpha=0.7)

    # 相机轨迹
    ax.plot(cam_centers[:,0],cam_centers[:,1],cam_centers[:,2],
            'k--',lw=1.5,alpha=0.35,zorder=1)

    # 6个插值相机视锥
    for ai,alpha in enumerate(ALPHAS):
        R_i=slerp_R(R_exo,R_ego,alpha)
        t_i=(1-alpha)*t_exo+alpha*t_ego
        fac=0.9 if ai in [0,5] else 0.5
        draw_cam_frustum(ax,R_i,t_i,COLORS[ai],scale=0.06,alpha=fac)
        # 标签
        center=-R_i.T@t_i
        ax.text(center[0],center[1],center[2]+0.03,
                LABELS[ai],fontsize=6,color=COLORS[ai],
                ha='center',fontweight='bold',zorder=7)

    ax.set_title(f'seq{seq_idx:03d}  SFLNet MPJPE={err:.1f}mm',
                 fontsize=9,fontweight='bold')
    ax.view_init(elev=35,azim=-30)
    ax.tick_params(labelsize=5,pad=0)
    ax.set_xlabel('X',fontsize=6); ax.set_ylabel('Y',fontsize=6); ax.set_zlabel('Z',fontsize=6)
    ax.grid(True,alpha=0.2)

    print(f'seq{seq_idx:03d} done')

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
leg=[
    Line2D([0],[0],color='#22aa22',lw=2,label='GT skeleton (world)'),
    Line2D([0],[0],color='#ee6610',lw=2,ls='--',label='SFLNet pred (world)'),
    Patch(facecolor='#2244cc',label='α=0  Exo camera'),
    Patch(facecolor='#ee6610',label='α=1  Ego camera'),
]
fig.legend(handles=leg,loc='lower center',ncol=4,fontsize=8,
           bbox_to_anchor=(0.5,-0.04),framealpha=0.9)

plt.tight_layout(pad=0.8)
out_path='/data/data5/zhaoran/paper_code/comparison_vis/fig3_cam_3d_v2.jpg'
plt.savefig(out_path,dpi=150,bbox_inches='tight',facecolor='white')
plt.close()
print(f'Saved: {out_path}')
