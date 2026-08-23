import sys, torch, numpy as np, cv2, yaml, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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

def draw_camera(ax, R, t, color, label, scale=0.08):
    """在3D空间中画相机位置和朝向"""
    # 相机中心
    cam_center = -R.T @ t
    # 相机坐标轴（世界坐标系下）
    x_axis = R.T[:,0]*scale
    y_axis = R.T[:,1]*scale
    z_axis = R.T[:,2]*scale
    # 画坐标轴
    ax.quiver(*cam_center, *x_axis, color='red',   alpha=0.8, linewidth=1.5)
    ax.quiver(*cam_center, *y_axis, color='green', alpha=0.8, linewidth=1.5)
    ax.quiver(*cam_center, *z_axis, color=color,   alpha=0.9, linewidth=2.5, label=label)
    # 相机位置点
    ax.scatter(*cam_center, c=color, s=80, zorder=5, edgecolors='white', linewidths=1)

def slerp_R(R0, R1, alpha):
    R_rel = R1 @ R0.T
    cos_theta = np.clip((np.trace(R_rel)-1)/2, -1, 1)
    theta = np.arccos(abs(cos_theta))
    if theta < 1e-6:
        return R0.copy()
    K = (R_rel - R_rel.T)/(2*np.sin(theta))
    R_i = np.eye(3) + np.sin(alpha*theta)*K + (1-np.cos(alpha*theta))*(K@K)
    return R_i @ R0

ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
COLORS = ['#2244cc','#3B6EC7','#5499B0','#6DBB88','#86CC55','#ee6610']

seq_idx=149; t=8
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

# 相机外参
exo_ext=sample['exo_pose'][t,0].numpy()
ego_ext=sample['ego_pose'][t].numpy()
R_exo=exo_ext[:9].reshape(3,3); t_exo=exo_ext[9:]
R_ego=ego_ext[:9].reshape(3,3);  t_ego=ego_ext[9:]

# GT关节点世界坐标
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

# 图：左=3D场景（相机+手），右=6个2D投影视角
fig=plt.figure(figsize=(18,6),facecolor='white')
fig.suptitle('SFLNet: Geodesic Camera Interpolation — Exo→Ego Viewpoint Transition',
             fontsize=11,fontweight='bold',y=1.01)

# ── 左图：3D场景全局视图 ──
ax3d=fig.add_subplot(1,7,1,projection='3d')
ax3d.set_facecolor('#f0f0f5')

# 画手部骨架（世界坐标）
for i,j in HAND_CONNECTIONS:
    ax3d.plot([gt_world[i,0],gt_world[j,0]],
              [gt_world[i,1],gt_world[j,1]],
              [gt_world[i,2],gt_world[j,2]],
              color='#22aa22',lw=1.8,alpha=0.8)
ax3d.scatter(gt_world[:,0],gt_world[:,1],gt_world[:,2],
             c='#22aa22',s=20,depthshade=False,zorder=5)

# 画所有中间相机
for ai,alpha in enumerate(ALPHAS):
    R_i=slerp_R(R_exo,R_ego,alpha)
    t_i=(1-alpha)*t_exo+alpha*t_ego
    label=f'α={alpha:.1f}' if ai in [0,5] else None
    draw_camera(ax3d,R_i,t_i,COLORS[ai],label,scale=0.06)

# 画相机轨迹
cam_centers=[]
for alpha in np.linspace(0,1,20):
    R_i=slerp_R(R_exo,R_ego,alpha)
    t_i=(1-alpha)*t_exo+alpha*t_ego
    cam_centers.append(-R_i.T@t_i)
cam_centers=np.array(cam_centers)
ax3d.plot(cam_centers[:,0],cam_centers[:,1],cam_centers[:,2],
          'k--',lw=1.2,alpha=0.4,label='Camera path')

ax3d.set_title('3D Scene\n(Camera Trajectory)',fontsize=9,fontweight='bold')
ax3d.view_init(elev=30,azim=-60)
ax3d.tick_params(labelsize=4,pad=0)
ax3d.set_xlabel('X',fontsize=5,labelpad=0)
ax3d.set_ylabel('Y',fontsize=5,labelpad=0)
ax3d.set_zlabel('Z',fontsize=5,labelpad=0)
ax3d.grid(True,alpha=0.2)
ax3d.legend(fontsize=6,loc='upper left')

# ── 右侧6列：每个相机视角下的2D投影 ──
SIZE=200
for ai,(alpha,color) in enumerate(zip(ALPHAS,COLORS)):
    R_i=slerp_R(R_exo,R_ego,alpha)
    t_i=(1-alpha)*t_exo+alpha*t_ego
    # 插值内参
    frame_data=seq['frames'][t]
    raw_exo=dataset.ds[frame_data[dataset.exo_views[0]]]
    intr_exo=raw_exo['intrinsics']
    raw_ego2=dataset.ds[frame_data[dataset.ego_view]]
    intr_ego=raw_ego2['intrinsics']
    fx=(1-alpha)*intr_exo['fx']+alpha*intr_ego['fx']
    fy=(1-alpha)*intr_exo['fy']+alpha*intr_ego['fy']
    ppx=(1-alpha)*intr_exo['ppx']+alpha*intr_ego['ppx']
    ppy=(1-alpha)*intr_exo['ppy']+alpha*intr_ego['ppy']
    sx=SIZE/640; sy=SIZE/480

    # 投影SFLNet预测
    pts_cam=(R_i@pred_world.T).T+t_i
    Z=np.clip(pts_cam[:,2],0.01,10)
    u=(fx*pts_cam[:,0]/Z+ppx)*sx
    v=(fy*pts_cam[:,1]/Z+ppy)*sy
    kp2d=np.stack([u,v],axis=1)

    # 白底图
    img=np.ones((SIZE,SIZE,3),dtype=np.uint8)*255
    for ci,(i,j) in enumerate(HAND_CONNECTIONS):
        x1,y1=int(kp2d[i,0]),int(kp2d[i,1])
        x2,y2=int(kp2d[j,0]),int(kp2d[j,1])
        bc=tuple(int(c*255) for c in matplotlib.colors.to_rgb(color))
        bc_bgr=(bc[2],bc[1],bc[0])
        if 0<=x1<SIZE and 0<=y1<SIZE and 0<=x2<SIZE and 0<=y2<SIZE:
            cv2.line(img,(x1,y1),(x2,y2),bc_bgr,3)
    for i in range(21):
        x,y=int(kp2d[i,0]),int(kp2d[i,1])
        if 0<=x<SIZE and 0<=y<SIZE:
            bc=tuple(int(c*255) for c in matplotlib.colors.to_rgb(color))
            bc_bgr=(bc[2],bc[1],bc[0])
            cv2.circle(img,(x,y),5,bc_bgr,-1)
            cv2.circle(img,(x,y),5,(0,0,0),1)

    ax=fig.add_subplot(1,7,ai+2)
    ax.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    label=f'α={alpha:.1f}'
    if ai==0: label+='\n(Exo view)'
    if ai==5: label+=f'\n(Ego view)\n{np.mean(np.sqrt(np.sum((pred_sfl[t]-gt_joints[t])**2,axis=-1)))*1000:.1f}mm'
    ax.set_title(label,fontsize=8,fontweight='bold',
                 color='blue' if ai==0 else ('darkorange' if ai==5 else 'black'))
    ax.axis('off')

from matplotlib.patches import Patch
legend_elements=[
    Patch(facecolor='#2244cc',label='α=0  Exo camera viewpoint'),
    Patch(facecolor='#ee6610',label='α=1  Ego camera viewpoint (SFLNet)'),
    Patch(facecolor='#22aa22',label='Hand skeleton (world coords)'),
]
fig.legend(handles=legend_elements,loc='lower center',ncol=3,
           fontsize=9,bbox_to_anchor=(0.5,-0.04),framealpha=0.9)

plt.tight_layout(pad=0.5)
out_path='/data/data5/zhaoran/paper_code/comparison_vis/fig3_cam_transition.jpg'
plt.savefig(out_path,dpi=150,bbox_inches='tight',facecolor='white')
plt.close()
print(f'Saved: {out_path}')
