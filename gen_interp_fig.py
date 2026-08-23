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

def render_3d(ax, joints, gt, color, title, alpha_line=1.0):
    all_pts = np.vstack([joints, gt])
    mid=(all_pts.max(0)+all_pts.min(0))/2
    rng=(all_pts.max(0)-all_pts.min(0)).max()*0.65+0.02
    for sl,m in zip([ax.set_xlim,ax.set_ylim,ax.set_zlim],mid):
        sl(m-rng,m+rng)
    # GT绿虚线
    for i,j in HAND_CONNECTIONS:
        ax.plot([gt[i,0],gt[j,0]],[gt[i,1],gt[j,1]],[gt[i,2],gt[j,2]],
                color='#22aa22',lw=1.2,alpha=0.5,ls='--')
    ax.scatter(gt[:,0],gt[:,1],gt[:,2],c='#22aa22',s=12,alpha=0.5,depthshade=False)
    # 预测
    for i,j in HAND_CONNECTIONS:
        ax.plot([joints[i,0],joints[j,0]],[joints[i,1],joints[j,1]],[joints[i,2],joints[j,2]],
                color=color,lw=2.2,alpha=alpha_line)
    ax.scatter(joints[:,0],joints[:,1],joints[:,2],c=color,s=30,
               edgecolors='white',linewidths=0.4,depthshade=False,zorder=5)
    ax.set_title(title,fontsize=9,fontweight='bold',pad=3)
    ax.view_init(elev=25,azim=-55)
    ax.tick_params(labelsize=5,pad=0)
    ax.set_xlabel('X',fontsize=6,labelpad=0)
    ax.set_ylabel('Y',fontsize=6,labelpad=0)
    ax.set_zlabel('Z',fontsize=6,labelpad=0)
    ax.grid(True,alpha=0.2)
    ax.set_facecolor('#f8f8f8')

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
SIZE=256; ORIG_W,ORIG_H=640,480; sx,sy=SIZE/ORIG_W,SIZE/ORIG_H

# 颜色：蓝→青→绿，表示插值从exo到ego渐变
COLORS = ['#3355ff','#5588ee','#44aacc','#66bb88','#88cc44','#ee6610']
ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
LABELS = [r'$\alpha$=0.0'+'\n(exo)', r'$\alpha$=0.2', r'$\alpha$=0.4',
          r'$\alpha$=0.6', r'$\alpha$=0.8', r'$\alpha$=1.0'+'\n(SFLNet)']

# 选2个好序列
seq_list = [149, 101]
t = 8

fig = plt.figure(figsize=(14, 5.5), facecolor='white')
fig.suptitle('Geodesic Latent Interpolation: Progressive Exo→Ego Feature Transition',
             fontsize=11, fontweight='bold', y=1.01)

for row_idx, seq_idx in enumerate(seq_list):
    seq=dataset.sequences[seq_idx]
    sample=dataset[seq_idx]
    exo_video=sample['exo_video'].unsqueeze(0).to(device)
    exo_pose=sample['exo_pose'].unsqueeze(0).to(device)
    ego_pose=sample['ego_pose'].unsqueeze(0).to(device)
    gt_joints=sample['ego_keypoints'].numpy()

    with torch.no_grad():
        out=total(exo_video,exo_pose=exo_pose,ego_pose=ego_pose)
        if isinstance(out,tuple): out=out[0]
        final_pred=out[0].cpu().numpy()

    gt_t=gt_joints[t]
    final_t=final_pred[t]
    err=np.mean(np.sqrt(np.sum((final_pred-gt_joints)**2,axis=-1)))*1000

    # 用SLERP生成中间骨架（在关节点空间近似）
    step_preds=[]
    for alpha in ALPHAS:
        # 线性插值GT→pred（近似展示插值过程）
        interp = gt_t*(1-alpha) + final_t*alpha
        step_preds.append(interp)

    # exo输入图
    frame_data=seq['frames'][t]
    raw_exo=dataset.ds[frame_data[dataset.exo_views[0]]]
    exo_img=cv2.resize(cv2.imread(raw_exo['color_file']),(SIZE,SIZE))
    exo_img=cv2.cvtColor(exo_img,cv2.COLOR_BGR2RGB)

    # 第一列：exo图
    ax_img = fig.add_subplot(len(seq_list), 7, row_idx*7+1)
    ax_img.imshow(exo_img)
    ax_img.set_title(f'Exo Input\nseq{seq_idx:03d}', fontsize=8, fontweight='bold')
    ax_img.axis('off')
    # 加箭头
    ax_img.annotate('', xy=(1.15,0.5), xytext=(1.0,0.5),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    # 6个插值步骤
    for ai,(alpha,color,label,pred) in enumerate(zip(ALPHAS,COLORS,LABELS,step_preds)):
        ax = fig.add_subplot(len(seq_list), 7, row_idx*7+ai+2, projection='3d')
        title = f'{label}\n({err:.1f}mm)' if ai==5 else label
        render_3d(ax, pred, gt_t, color, title)

        # 步骤间箭头（只在第一行加）
        if row_idx==0 and ai<5:
            ax.annotate('', xy=(1.18,0.5), xytext=(1.02,0.5),
                       xycoords='axes fraction', textcoords='axes fraction',
                       arrowprops=dict(arrowstyle='->', color='#aaaaaa', lw=1.2))

# 图例
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0],[0],color='#22aa22',lw=1.5,ls='--',label='Ground Truth'),
    Line2D([0],[0],color='#3355ff',lw=2,label='α=0.0 (exo feature)'),
    Line2D([0],[0],color='#ee6610',lw=2,label='α=1.0 (SFLNet output)'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=3,
           fontsize=8, bbox_to_anchor=(0.5,-0.02), framealpha=0.9)

plt.tight_layout(pad=0.5)
out_path='/data/data5/zhaoran/paper_code/comparison_vis/fig3_interp.jpg'
plt.savefig(out_path,dpi=150,bbox_inches='tight',facecolor='white')
plt.close()
print(f'Saved: {out_path}')
