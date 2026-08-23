import sys, torch, numpy as np, yaml, os
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

def draw_cam(ax, pos, R, color, scale=0.08, alpha_vis=1.0, label=None):
    """画相机视锥"""
    z_dir = R.T[:,2]*scale  # 相机朝向
    x_dir = R.T[:,0]*scale*0.6
    y_dir = R.T[:,1]*scale*0.6
    # 视锥4个角
    corners = np.array([pos+z_dir+x_dir+y_dir,
                        pos+z_dir-x_dir+y_dir,
                        pos+z_dir-x_dir-y_dir,
                        pos+z_dir+x_dir-y_dir])
    for c in corners:
        ax.plot([pos[0],c[0]],[pos[1],c[1]],[pos[2],c[2]],
                color=color,lw=1.0,alpha=alpha_vis*0.6)
    for i in range(4):
        ax.plot([corners[i,0],corners[(i+1)%4,0]],
                [corners[i,1],corners[(i+1)%4,1]],
                [corners[i,2],corners[(i+1)%4,2]],
                color=color,lw=1.5,alpha=alpha_vis*0.8)
    ax.scatter(*pos,c=color,s=80,zorder=6,edgecolors='white',linewidths=1.0,alpha=alpha_vis)
    if label:
        ax.text(pos[0],pos[1],pos[2]+scale*0.8,label,
                fontsize=7,color=color,ha='center',fontweight='bold',zorder=8)

ALPHAS=[0.0,0.2,0.4,0.6,0.8,1.0]
COLORS=['#2244cc','#3B6EC7','#44AABB','#66BB77','#88CC33','#ee6610']
LABELS=[r'$\alpha$=0'+'\n(Exo)',r'$\alpha$=0.2',r'$\alpha$=0.4',
        r'$\alpha$=0.6',r'$\alpha$=0.8',r'$\alpha$=1'+'\n(Ego)']

seq_list=[149,101]; t=8
fig=plt.figure(figsize=(14,6),facecolor='white')
fig.suptitle('SFLNet: Geodesic Camera Interpolation — Circular Arc from Exo to Ego',
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
    c_exo=-R_exo.T@t_exo
    c_ego=-R_ego.T@t_ego

    # 手部骨架世界坐标
    frame_data=seq['frames'][t]
    raw_ego=dataset.ds[frame_data[dataset.ego_view]]
    data_ego=np.load(raw_ego['label_file'])
    j3d_ego_abs=data_ego['joint_3d'].squeeze()
    wrist_ego=j3d_ego_abs[0]
    gt_abs=gt_joints[t]+wrist_ego
    R_ego_inv=R_ego.T
    gt_world=(R_ego_inv@(gt_abs-t_ego).T).T
    hand_center=gt_world.mean(0)

    # ── 关键：把相机位置投影到以hand_center为圆心的球面上 ──
    # 保持exo和ego到hand_center的平均距离作为球半径
    r_exo=np.linalg.norm(c_exo-hand_center)
    r_ego=np.linalg.norm(c_ego-hand_center)
    R_sphere=( r_exo+r_ego)/2  # 平均半径

    # 把exo和ego投影到球面（单位方向向量×球半径）
    dir_exo=(c_exo-hand_center)/np.linalg.norm(c_exo-hand_center)
    dir_ego=(c_ego-hand_center)/np.linalg.norm(c_ego-hand_center)

    # SLERP两个方向向量（球面插值→圆弧轨迹）
    cos_angle=np.clip(np.dot(dir_exo,dir_ego),-1,1)
    angle=np.arccos(cos_angle)

    # 生成圆弧轨迹（密集点）
    arc_pts=[]
    for a in np.linspace(0,1,50):
        if abs(angle)<1e-6:
            d=dir_exo
        else:
            d=(np.sin((1-a)*angle)*dir_exo+np.sin(a*angle)*dir_ego)/np.sin(angle)
        arc_pts.append(hand_center+d*R_sphere)
    arc_pts=np.array(arc_pts)

    # 6个相机位置（球面上）
    cam_positions=[]
    cam_rotations=[]
    for alpha in ALPHAS:
        if abs(angle)<1e-6:
            d=dir_exo
        else:
            d=(np.sin((1-alpha)*angle)*dir_exo+np.sin(alpha*angle)*dir_ego)/np.sin(angle)
        pos=hand_center+d*R_sphere
        R_i=slerp_R(R_exo,R_ego,alpha)
        cam_positions.append(pos)
        cam_rotations.append(R_i)

    err=np.mean(np.sqrt(np.sum((pred_sfl[t]-gt_joints[t])**2,axis=-1)))*1000

    ax=fig.add_subplot(1,2,row_idx+1,projection='3d')
    ax.set_facecolor('#eef0f8')

    # 手部骨架（放大显示）
    sc=5.0
    gt_vis=(gt_world-hand_center)*sc+hand_center
    for i,j in HAND_CONNECTIONS:
        ax.plot([gt_vis[i,0],gt_vis[j,0]],
                [gt_vis[i,1],gt_vis[j,1]],
                [gt_vis[i,2],gt_vis[j,2]],
                color='#22aa22',lw=2.5,alpha=0.9,zorder=4)
    ax.scatter(gt_vis[:,0],gt_vis[:,1],gt_vis[:,2],
               c='#22aa22',s=30,depthshade=False,zorder=5)

    # 手部中心点
    ax.scatter(*hand_center,c='black',s=50,marker='*',zorder=6)

    # 圆弧轨迹
    ax.plot(arc_pts[:,0],arc_pts[:,1],arc_pts[:,2],
            'k--',lw=2.0,alpha=0.4,zorder=2)

    # 从相机到手部中心的连线（视线）
    for pos,color in zip(cam_positions,COLORS):
        ax.plot([pos[0],hand_center[0]],[pos[1],hand_center[1]],[pos[2],hand_center[2]],
                color=color,lw=0.8,alpha=0.25,ls=':',zorder=1)

    # 6个相机视锥
    for ai,(pos,R_i,color,label) in enumerate(zip(cam_positions,cam_rotations,COLORS,LABELS)):
        fac=1.0 if ai in [0,5] else 0.6
        sc_cam=0.07 if ai in [0,5] else 0.05
        draw_cam(ax,pos,R_i,color,scale=sc_cam,alpha_vis=fac,
                 label=label if ai in [0,5] else None)

    ax.set_title(f'seq{seq_idx:03d}   MPJPE = {err:.1f} mm',
                 fontsize=9,fontweight='bold')
    ax.view_init(elev=20,azim=-40)
    ax.tick_params(labelsize=5,pad=0)
    ax.set_xlabel('X',fontsize=6); ax.set_ylabel('Y',fontsize=6); ax.set_zlabel('Z',fontsize=6)
    ax.grid(True,alpha=0.15)
    print(f'seq{seq_idx:03d} done, MPJPE={err:.1f}mm')

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
leg=[
    Line2D([0],[0],color='#22aa22',lw=2.5,label='GT hand skeleton'),
    Line2D([0],[0],color='black',lw=1.5,ls='--',alpha=0.5,label='Camera arc trajectory'),
    Patch(facecolor='#2244cc',label='α=0  Exo camera'),
    Patch(facecolor='#ee6610',label='α=1  Ego camera'),
]
fig.legend(handles=leg,loc='lower center',ncol=4,fontsize=9,
           bbox_to_anchor=(0.5,-0.04),framealpha=0.9)

plt.tight_layout(pad=0.8)
out_path='/data/data5/zhaoran/paper_code/comparison_vis/fig3_cam_arc.jpg'
plt.savefig(out_path,dpi=150,bbox_inches='tight',facecolor='white')
plt.close()
print(f'Saved: {out_path}')
