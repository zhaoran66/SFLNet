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

ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
COLORS = ['#2244cc','#3366dd','#4499bb','#66bb77','#88cc44','#ee6610']
LABELS = [r'$\alpha$=0'+'\n(Exo cam)', r'$\alpha$=0.2', r'$\alpha$=0.4',
          r'$\alpha$=0.6', r'$\alpha$=0.8', r'$\alpha$=1'+'\n(Ego cam)']

def slerp_rotation(R0, R1, alpha):
    """SLERP between two rotation matrices via axis-angle"""
    # R_rel = R1 @ R0.T
    R_rel = R1 @ R0.T
    # 转axis-angle
    theta = np.arccos(np.clip((np.trace(R_rel)-1)/2, -1, 1))
    if abs(theta) < 1e-6:
        return R0.copy()
    axis = np.array([R_rel[2,1]-R_rel[1,2],
                     R_rel[0,2]-R_rel[2,0],
                     R_rel[1,0]-R_rel[0,1]]) / (2*np.sin(theta))
    # 插值角度
    theta_i = alpha * theta
    K = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    R_i = np.eye(3) + np.sin(theta_i)*K + (1-np.cos(theta_i))*(K@K)
    return R_i @ R0

def draw_3d(ax, joints, color, title, ref_center=None, ref_scale=None):
    if ref_center is not None:
        mid = ref_center
        rng = ref_scale
    else:
        mid = (joints.max(0)+joints.min(0))/2
        rng = (joints.max(0)-joints.min(0)).max()*0.65+0.02
    for sl,m in zip([ax.set_xlim,ax.set_ylim,ax.set_zlim],mid):
        sl(m-rng, m+rng)
    for i,j in HAND_CONNECTIONS:
        ax.plot([joints[i,0],joints[j,0]],[joints[i,1],joints[j,1]],[joints[i,2],joints[j,2]],
                color=color,lw=2.5)
    ax.scatter(joints[:,0],joints[:,1],joints[:,2],
               c=color,s=40,edgecolors='white',linewidths=0.5,depthshade=False,zorder=5)
    ax.set_title(title,fontsize=9,fontweight='bold',pad=3)
    ax.view_init(elev=25,azim=-55)
    ax.tick_params(labelsize=4,pad=0)
    ax.set_xlabel('X',fontsize=5,labelpad=0)
    ax.set_ylabel('Y',fontsize=5,labelpad=0)
    ax.set_zlabel('Z',fontsize=5,labelpad=0)
    ax.grid(True,alpha=0.2)
    ax.set_facecolor('#f5f5f5')

seq_list=[149,101]; t=8
n_rows=len(seq_list); n_cols=8  # exo图+6步+GT

fig=plt.figure(figsize=(18,6.0),facecolor='white')
fig.suptitle('SFLNet: Camera Coordinate Transformation (Exo→Ego) via Geodesic Interpolation',
             fontsize=11,fontweight='bold',y=1.01)

for row_idx,seq_idx in enumerate(seq_list):
    seq=dataset.sequences[seq_idx]
    sample=dataset[seq_idx]
    exo_video=sample['exo_video'].unsqueeze(0).to(device)
    exo_pose=sample['exo_pose'].unsqueeze(0).to(device)
    ego_pose=sample['ego_pose'].unsqueeze(0).to(device)
    gt_joints=sample['ego_keypoints'].numpy()  # [T,21,3] in ego cam, wrist-centered

    with torch.no_grad():
        out=total(exo_video,exo_pose=exo_pose,ego_pose=ego_pose)
        if isinstance(out,tuple): out=out[0]
        pred_sfl=out[0].cpu().numpy()  # [T,21,3] in ego cam

    # 获取相机外参
    exo_ext=sample['exo_pose'][t,0].numpy()  # [12]
    ego_ext=sample['ego_pose'][t].numpy()    # [12]
    R_exo=exo_ext[:9].reshape(3,3); t_exo=exo_ext[9:]
    R_ego=ego_ext[:9].reshape(3,3); t_ego=ego_ext[9:]

    # GT关节点（ego坐标系绝对坐标）
    frame_data=seq['frames'][t]
    raw_ego=dataset.ds[frame_data[dataset.ego_view]]
    data_ego=np.load(raw_ego['label_file'])
    j3d_ego_abs=data_ego['joint_3d'].squeeze()  # [21,3] abs in ego cam
    wrist_ego=j3d_ego_abs[0]

    # SFLNet预测绝对坐标（ego cam）
    pred_abs=pred_sfl[t]+wrist_ego  # [21,3]
    gt_abs=gt_joints[t]+wrist_ego   # [21,3]

    # 转换到世界坐标
    R_ego_inv=R_ego.T
    pred_world=(R_ego_inv@(pred_abs-t_ego).T).T
    gt_world=(R_ego_inv@(gt_abs-t_ego).T).T

    # 对每个alpha，把world坐标变换到插值后的相机坐标系
    all_pts=[]
    step_joints=[]
    for alpha in ALPHAS:
        # SLERP旋转插值
        R_i=slerp_rotation(R_exo,R_ego,alpha)
        t_i=(1-alpha)*t_exo+alpha*t_ego
        # world→interpolated cam
        joints_cam=(R_i@pred_world.T).T+t_i
        # wrist-centered
        joints_cam=joints_cam-joints_cam[0]
        step_joints.append(joints_cam)
        all_pts.append(joints_cam)

    # GT在ego坐标系（wrist-centered）
    gt_cam=gt_joints[t].copy()
    all_pts.append(gt_cam)
    all_pts_arr=np.vstack(all_pts)
    ref_center=(all_pts_arr.max(0)+all_pts_arr.min(0))/2
    ref_scale=(all_pts_arr.max(0)-all_pts_arr.min(0)).max()*0.65+0.02

    err=np.mean(np.sqrt(np.sum((pred_sfl[t]-gt_joints[t])**2,axis=-1)))*1000

    # exo图
    raw_exo=dataset.ds[frame_data[dataset.exo_views[0]]]
    exo_img=cv2.resize(cv2.imread(raw_exo['color_file']),(256,256))
    exo_img=cv2.cvtColor(exo_img,cv2.COLOR_BGR2RGB)

    ax_img=fig.add_subplot(n_rows,n_cols,row_idx*n_cols+1)
    ax_img.imshow(exo_img)
    ax_img.set_title(f'Exo Input\nseq{seq_idx:03d}',fontsize=8,fontweight='bold')
    ax_img.axis('off')

    # 6个插值步骤
    for ai,(alpha,color,label,joints) in enumerate(zip(ALPHAS,COLORS,LABELS,step_joints)):
        ax=fig.add_subplot(n_rows,n_cols,row_idx*n_cols+ai+2,projection='3d')
        title=label
        if ai==5: title+=f'\n{err:.1f}mm'
        draw_3d(ax,joints,color,title,ref_center,ref_scale)

    # GT
    ax_gt=fig.add_subplot(n_rows,n_cols,row_idx*n_cols+8,projection='3d')
    draw_3d(ax_gt,gt_cam,'#22aa22','GT\n(Ego cam)',ref_center,ref_scale)

    print(f'seq{seq_idx:03d}  MPJPE={err:.1f}mm done')

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements=[
    Patch(facecolor='#2244cc',label='α=0: SFLNet pred in Exo camera space'),
    Patch(facecolor='#ee6610',label='α=1: SFLNet pred in Ego camera space'),
    Patch(facecolor='#22aa22',label='Ground Truth (Ego camera space)'),
]
fig.legend(handles=legend_elements,loc='lower center',ncol=3,
           fontsize=9,bbox_to_anchor=(0.5,-0.04),framealpha=0.9)

plt.tight_layout(pad=0.5)
out_path='/data/data5/zhaoran/paper_code/comparison_vis/fig3_cam_interp.jpg'
plt.savefig(out_path,dpi=150,bbox_inches='tight',facecolor='white')
plt.close()
print(f'Saved: {out_path}')
