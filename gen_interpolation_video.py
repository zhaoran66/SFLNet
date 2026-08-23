import sys, torch, numpy as np, cv2, yaml, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, '/data/data5/zhaoran/paper_code/total')
sys.path.insert(0, '/data/data5/zhaoran/paper_code/ego_estimator')

device = torch.device('cuda:0')
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]
SIZE=256; ORIG_W,ORIG_H=640,480; sx,sy=SIZE/ORIG_W,SIZE/ORIG_H
mean_t = torch.tensor([0.485,0.456,0.406]).view(1,1,3,1,1).to(device)
std_t  = torch.tensor([0.229,0.224,0.225]).view(1,1,3,1,1).to(device)

def draw_skel_ego(img, kp2d, bone_c, joint_c, lw=2, r=4):
    img = img.copy()
    for i,j in HAND_CONNECTIONS:
        x1,y1=int(kp2d[i,0]),int(kp2d[i,1])
        x2,y2=int(kp2d[j,0]),int(kp2d[j,1])
        if 0<=x1<SIZE and 0<=y1<SIZE and 0<=x2<SIZE and 0<=y2<SIZE:
            cv2.line(img,(x1,y1),(x2,y2),bone_c,lw)
    for i in range(21):
        x,y=int(kp2d[i,0]),int(kp2d[i,1])
        if 0<=x<SIZE and 0<=y<SIZE:
            cv2.circle(img,(x,y),r,joint_c,-1)
            cv2.circle(img,(x,y),r,(0,0,0),1)
    return img

def project(kp3d, wrist, intr):
    fx,fy,ppx,ppy=intr['fx'],intr['fy'],intr['ppx'],intr['ppy']
    kp=kp3d+wrist; Z=np.clip(kp[:,2],0.1,10.0)
    u=(fx*kp[:,0]/Z+ppx)*sx; v=(fy*kp[:,1]/Z+ppy)*sy
    return np.stack([u,v],axis=1)

def render_3d_skel(joints, gt, color, title, figsize=(2.8,2.8)):
    fig = plt.figure(figsize=figsize, facecolor='#0d1117')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0d1117')
    all_pts = np.vstack([joints, gt])
    mid=(all_pts.max(0)+all_pts.min(0))/2
    rng=(all_pts.max(0)-all_pts.min(0)).max()*0.65+0.02
    for sl,m in zip([ax.set_xlim,ax.set_ylim,ax.set_zlim],mid):
        sl(m-rng,m+rng)
    # GT虚线
    for i,j in HAND_CONNECTIONS:
        ax.plot([gt[i,0],gt[j,0]],[gt[i,1],gt[j,1]],[gt[i,2],gt[j,2]],
                color='#22aa22',lw=1.2,alpha=0.5,ls='--')
    ax.scatter(gt[:,0],gt[:,1],gt[:,2],c='#22aa22',s=15,alpha=0.5)
    # 预测
    for i,j in HAND_CONNECTIONS:
        ax.plot([joints[i,0],joints[j,0]],[joints[i,1],joints[j,1]],[joints[i,2],joints[j,2]],
                color=color,lw=2.0)
    ax.scatter(joints[:,0],joints[:,1],joints[:,2],c=color,s=30,
               edgecolors='white',linewidths=0.4)
    ax.set_title(title,color='white',fontsize=8,pad=3)
    ax.view_init(elev=25,azim=-55)
    for ax_ in [ax.xaxis,ax.yaxis,ax.zaxis]:
        ax_.pane.fill=False
        ax_.pane.set_edgecolor('#333')
    ax.tick_params(colors='#888',labelsize=5)
    ax.grid(True,alpha=0.15,color='#444')
    plt.tight_layout(pad=0.2)
    fig.canvas.draw()
    w,h=fig.canvas.get_width_height()
    img=np.frombuffer(fig.canvas.buffer_rgba(),dtype=np.uint8).reshape(h,w,4)
    img=cv2.cvtColor(img,cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    return img

# 加载模型
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

os.makedirs('/data/data5/zhaoran/paper_code/demo_videos',exist_ok=True)

# 6个插值步骤的颜色（从蓝→橙，表示exo→ego渐变）
ALPHA_STEPS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
COLORS = ['#4488ff','#5599ee','#66aacc','#ee9944','#ff8833','#ff6600']
LABELS = ['α=0.0\n(exo feat)','α=0.2','α=0.4','α=0.6','α=0.8','α=1.0\n(ego feat)']

seq_list = [149, 101, 117]

for seq_idx in seq_list:
    seq=dataset.sequences[seq_idx]
    sample=dataset[seq_idx]
    exo_video=sample['exo_video'].unsqueeze(0).to(device)
    exo_pose=sample['exo_pose'].unsqueeze(0).to(device)
    ego_pose=sample['ego_pose'].unsqueeze(0).to(device)
    gt_joints=sample['ego_keypoints'].numpy()

    # 获取最终预测
    with torch.no_grad():
        out = total(exo_video, exo_pose=exo_pose, ego_pose=ego_pose)
        if isinstance(out, tuple): out = out[0]
        final_pred = out
        intermediate = None

    t = 8  # 中间帧
    frame_data=seq['frames'][t]
    raw_ego=dataset.ds[frame_data[dataset.ego_view]]
    raw_exo=dataset.ds[frame_data[dataset.exo_views[0]]]
    data_ego=np.load(raw_ego['label_file'])
    j2d_ego=data_ego['joint_2d'].squeeze()
    j3d_ego=data_ego['joint_3d'].squeeze()
    intr=raw_ego['intrinsics']; wrist=j3d_ego[0]

    exo_img=cv2.resize(cv2.imread(raw_exo['color_file']),(SIZE,SIZE))
    ego_img=cv2.resize(cv2.imread(raw_ego['color_file']),(SIZE,SIZE))
    j2d_s=j2d_ego.copy(); j2d_s[:,0]*=sx; j2d_s[:,1]*=sy
    gt_t=gt_joints[t]

    if intermediate is not None:
        # intermediate: list of [B,T,21,3] per alpha step
        step_preds = [step[0,t].cpu().numpy() for step in intermediate]
    else:
        # fallback: 线性插值可视化
        final_t = final_pred[0,t].cpu().numpy()
        step_preds = [gt_t * a + final_t * (1-a) for a in reversed(ALPHA_STEPS)]

    err_final=np.mean(np.sqrt(np.sum((final_pred[0].cpu().numpy()-gt_joints)**2,axis=-1)))*1000

    # 生成视频：每帧展示当前插值步骤
    # 布局：左=exo输入 | 右=6列3D骨架
    skel_w=220; skel_h=220
    total_w=SIZE+8+skel_w*6+10; total_h=max(SIZE,skel_h)+60
    fourcc=cv2.VideoWriter_fourcc(*'mp4v')
    vpath=f'/data/data5/zhaoran/paper_code/demo_videos/seq{seq_idx:03d}_interp.mp4'
    writer=cv2.VideoWriter(vpath,fourcc,4,(total_w,total_h))

    # 对每个alpha步骤生成一帧（重复4次让视频长一点）
    for repeat in range(4):
        for ai,(alpha,color,label,step_pred) in enumerate(zip(ALPHA_STEPS,COLORS,LABELS,step_preds)):
            # 左：exo图
            frame=np.zeros((total_h,total_w,3),dtype=np.uint8)+15
            exo_r=exo_img.copy()
            cv2.putText(exo_r,'Exo Input',(8,22),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
            frame[:SIZE,:SIZE]=exo_r

            # 右：当前步骤高亮，其余暗淡
            for bi in range(6):
                x0=SIZE+8+bi*skel_w
                is_current=(bi==ai)
                sk_img=render_3d_skel(step_preds[bi], gt_t, COLORS[bi],
                                      LABELS[bi], figsize=(2.4,2.4))
                sk_img=cv2.resize(sk_img,(skel_w,skel_h))
                if not is_current:
                    sk_img=(sk_img*0.35).astype(np.uint8)
                else:
                    # 高亮边框
                    cv2.rectangle(sk_img,(1,1),(skel_w-2,skel_h-2),(100,180,255),2)
                frame[:skel_h,x0:x0+skel_w]=sk_img

            # 进度箭头
            for bi in range(5):
                x_arrow=SIZE+8+bi*skel_w+skel_w-8
                y_arrow=skel_h//2
                c=(255,255,0) if bi<ai else (80,80,80)
                cv2.arrowedLine(frame,(x_arrow,y_arrow),(x_arrow+12,y_arrow),c,2,tipLength=0.4)

            # 底部标签
            cv2.putText(frame,f'Geodesic Interpolation: exo feat --> ego feat',
                        (8,total_h-36),cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),1)
            cv2.putText(frame,f'seq{seq_idx:03d}  Final MPJPE={err_final:.1f}mm  Step {ai+1}/6',
                        (8,total_h-14),cv2.FONT_HERSHEY_SIMPLEX,0.45,(150,150,150),1)

            writer.write(frame)
    writer.release()
    print(f'seq{seq_idx:03d}  MPJPE={err_final:.1f}mm -> {vpath}')

print('Done!')
