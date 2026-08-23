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

# 加载模型并修改forward以返回中间结果
total = LatentHandPoseModel(
    embed_dim=cfg['model']['hidden_dim'],
    num_interpolation_steps=cfg['model']['interpolate_steps'],
    num_joints=cfg['dataset']['num_joints'],
    num_views=len(cfg['dataset']['exo_views'])
).to(device)
ckpt=torch.load('/data/data5/zhaoran/paper_code/total/checkpoints/best_latent.pth',map_location=device)
total.load_state_dict(ckpt['model_state_dict']); total.eval()

def forward_with_steps(model, exo_video, exo_pose, ego_pose):
    """修改版forward，返回每个插值步骤的关节点预测"""
    with torch.no_grad():
        # Step 1: 特征提取
        B, T, V, C, H, W = exo_video.shape
        fused, static, motion = model.fourier_layer(exo_video)  # [B,T,V,9,H,W]
        exo_features = model.encoder(fused)  # [B,T,D]

        # Step 2: 预测ego特征
        predicted_ego = model.geodesic_interpolator.endpoint_predictor(exo_features)

        # Step 3: GI插值，得到full_sequence [B,T,num_steps,D]
        full_sequence = model.geodesic_interpolator(
            exo_features, predicted_ego,
            exo_pose=exo_pose, ego_pose=ego_pose
        )
        B, T_seq, num_steps, D = full_sequence.shape

        # Step 4: 对每个插值步骤单独解码
        step_size = num_steps  # num_interpolation_steps + 2
        all_step_kps = []

        for step_i in range(step_size):
            # 取第 step_i 个插值步骤的特征 [B,T,D]
            step_feat = full_sequence[:, :, step_i, :]  # [B,T,D]

            # 通过sequence encoder
            step_flat = step_feat.view(B, T_seq, D)
            step_flat = step_flat + model.pos_encoding[:T_seq].unsqueeze(0)
            encoded = model.sequence_encoder(step_flat)  # [B,T,D]

            # 解码每帧
            step_kps = []
            for t in range(T_seq):
                frame_mem = encoded[:, t:t+1, :]
                kp = model.keypoint_decoder(frame_mem)  # [B,21,3]
                step_kps.append(kp)
            step_kps = torch.stack(step_kps, dim=1)  # [B,T,21,3]
            all_step_kps.append(step_kps[0].cpu().numpy())  # [T,21,3]

    return all_step_kps  # list of num_steps x [T,21,3]

print('Model loaded')
os.makedirs('/data/data5/zhaoran/paper_code/comparison_vis', exist_ok=True)

# 颜色：蓝→绿（exo→ego）
COLORS = ['#2244cc','#3366dd','#4499bb','#66bb77','#88cc44','#ee6610']
LABELS = [r'$\alpha$=0'+'\n(exo feat)', r'$\alpha$=0.2', r'$\alpha$=0.4',
          r'$\alpha$=0.6', r'$\alpha$=0.8', r'$\alpha$=1'+'\n(SFLNet)']

def draw_3d(ax, joints, color, title, ref_pts=None):
    pts = joints if ref_pts is None else ref_pts
    mid=(pts.max(0)+pts.min(0))/2
    rng=(pts.max(0)-pts.min(0)).max()*0.65+0.02
    for sl,m in zip([ax.set_xlim,ax.set_ylim,ax.set_zlim],mid):
        sl(m-rng,m+rng)
    for i,j in HAND_CONNECTIONS:
        ax.plot([joints[i,0],joints[j,0]],[joints[i,1],joints[j,1]],[joints[i,2],joints[j,2]],
                color=color,lw=2.2)
    ax.scatter(joints[:,0],joints[:,1],joints[:,2],
               c=color,s=35,edgecolors='white',linewidths=0.4,depthshade=False,zorder=5)
    ax.set_title(title,fontsize=9,fontweight='bold',pad=3)
    ax.view_init(elev=25,azim=-55)
    ax.tick_params(labelsize=4,pad=0)
    ax.set_xlabel('X',fontsize=5,labelpad=0)
    ax.set_ylabel('Y',fontsize=5,labelpad=0)
    ax.set_zlabel('Z',fontsize=5,labelpad=0)
    ax.grid(True,alpha=0.2)
    ax.set_facecolor('#f5f5f5')

seq_list=[149,157,101]; t=8
n_rows=len(seq_list); n_cols=8

fig=plt.figure(figsize=(16,8.0),facecolor='white')
fig.suptitle('SFLNet: Geodesic Latent Interpolation Steps (Exo Feature → Ego Feature)',
             fontsize=11,fontweight='bold',y=1.02)

for row_idx,seq_idx in enumerate(seq_list):
    seq=dataset.sequences[seq_idx]
    sample=dataset[seq_idx]
    exo_video=sample['exo_video'].unsqueeze(0).to(device)
    exo_pose=sample['exo_pose'].unsqueeze(0).to(device)
    ego_pose=sample['ego_pose'].unsqueeze(0).to(device)
    gt_joints=sample['ego_keypoints'].numpy()

    # 获取每个插值步骤的骨架
    step_kps = forward_with_steps(total, exo_video, exo_pose, ego_pose)
    n_steps = len(step_kps)
    print(f'seq{seq_idx}: {n_steps} interpolation steps')

    gt_t=gt_joints[t]
    err_final=np.mean(np.sqrt(np.sum((step_kps[-1]-gt_joints)**2,axis=-1)))*1000

    # exo图
    frame_data=seq['frames'][t]
    raw_exo=dataset.ds[frame_data[dataset.exo_views[0]]]
    exo_img=cv2.resize(cv2.imread(raw_exo['color_file']),(256,256))
    exo_img=cv2.cvtColor(exo_img,cv2.COLOR_BGR2RGB)

    ax_img=fig.add_subplot(n_rows,n_cols,row_idx*n_cols+1)
    ax_img.imshow(exo_img)
    ax_img.set_title(f'Exo Input\nseq{seq_idx:03d}',fontsize=8,fontweight='bold')
    ax_img.axis('off')

    # 统一坐标参考范围（所有步骤+GT）
    all_ref = np.vstack([sk[t] for sk in step_kps] + [gt_t])
    colors_used = COLORS[:n_steps] if n_steps<=6 else COLORS
    labels_used = LABELS[:n_steps] if n_steps<=6 else LABELS

    for ai in range(min(n_steps,6)):
        ax=fig.add_subplot(n_rows,n_cols,row_idx*n_cols+ai+2,projection='3d')
        title=labels_used[ai]
        if ai==n_steps-1 or ai==5:
            title+=f'\n{err_final:.1f}mm'
        draw_3d(ax,step_kps[ai][t],colors_used[ai],title,ref_pts=all_ref)

    # 最后一列：GT
    ax_gt=fig.add_subplot(n_rows,n_cols,row_idx*n_cols+8,projection='3d')
    draw_3d(ax_gt, gt_t, '#22aa22', 'GT (ground truth)', ref_pts=all_ref)

    print(f'seq{seq_idx:03d} done, final MPJPE={err_final:.1f}mm')

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
legend_elements=[
    Patch(facecolor='#2244cc',label='α=0  Exo latent feature decoded'),
    Patch(facecolor='#ee6610',label='α=1  SFLNet output'),
    Patch(facecolor='#22aa22',label='Ground Truth (GT)'),
]
fig.legend(handles=legend_elements,loc='lower center',ncol=3,
           fontsize=9,bbox_to_anchor=(0.5,-0.04),framealpha=0.9)

plt.tight_layout(pad=0.5)
out_path='/data/data5/zhaoran/paper_code/comparison_vis/fig3_interp_best.jpg'
plt.savefig(out_path,dpi=150,bbox_inches='tight',facecolor='white')
plt.close()
print(f'Saved: {out_path}')
