import sys, torch, numpy as np, cv2, yaml, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, '/data/data5/zhaoran/paper_code/total')
sys.path.insert(0, '/data/data5/zhaoran')

device = torch.device('cuda:0')
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]
SIZE = 256; ORIG_W,ORIG_H=640,480; sx,sy=SIZE/ORIG_W,SIZE/ORIG_H

JOINT_LAYOUT = np.array([
    [0.0,0.0],[-0.4,0.5],[-0.6,0.85],[-0.75,1.15],[-0.85,1.4],
    [-0.2,0.6],[-0.2,1.0],[-0.2,1.3],[-0.2,1.55],
    [0.0,0.6],[0.0,1.05],[0.0,1.35],[0.0,1.6],
    [0.2,0.6],[0.2,1.0],[0.2,1.28],[0.2,1.5],
    [0.4,0.55],[0.4,0.88],[0.4,1.12],[0.4,1.3],
])

def draw_skeleton(img, kp2d, color_bone, color_joint, thickness=2, radius=3):
    img = img.copy()
    for i,j in HAND_CONNECTIONS:
        x1,y1=int(kp2d[i,0]),int(kp2d[i,1])
        x2,y2=int(kp2d[j,0]),int(kp2d[j,1])
        if 0<=x1<SIZE and 0<=y1<SIZE and 0<=x2<SIZE and 0<=y2<SIZE:
            cv2.line(img,(x1,y1),(x2,y2),color_bone,thickness)
    for i in range(21):
        x,y=int(kp2d[i,0]),int(kp2d[i,1])
        if 0<=x<SIZE and 0<=y<SIZE:
            cv2.circle(img,(x,y),radius,color_joint,-1)
    return img

def project_joints(pred_kp_t, wrist, intr):
    fx,fy,ppx,ppy=intr['fx'],intr['fy'],intr['ppx'],intr['ppy']
    kp_abs=pred_kp_t+wrist
    Z=np.clip(kp_abs[:,2],0.1,10.0)
    u=(fx*kp_abs[:,0]/Z+ppx)*sx
    v=(fy*kp_abs[:,1]/Z+ppy)*sy
    return np.stack([u,v],axis=1)

def add_label(img, text, color=(255,255,255), pos=(8,22)):
    img=img.copy()
    cv2.putText(img,text,pos,cv2.FONT_HERSHEY_SIMPLEX,0.52,(0,0,0),3)
    cv2.putText(img,text,pos,cv2.FONT_HERSHEY_SIMPLEX,0.52,color,1)
    return img

def draw_perjoint(err_b, err_s, vmax=80, fig_w=3.6, fig_h=3.2):
    """两列热力图：左baseline，右SFLNet"""
    cmap = plt.cm.RdYlGn_r
    norm = mcolors.Normalize(vmin=0, vmax=vmax)
    pts = JOINT_LAYOUT.copy(); pts[:,1] = -pts[:,1]

    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))
    fig.patch.set_facecolor('white')

    for ax, errors, title, tcol in zip(
        axes,
        [err_b, err_s],
        [f'Baseline\n{err_b.mean():.1f} mm', f'SFLNet\n{err_s.mean():.1f} mm'],
        ['#c03030', '#1a7a1a']
    ):
        ax.set_facecolor('white')
        for i,j in HAND_CONNECTIONS:
            x=[pts[i,0],pts[j,0]]; y=[pts[i,1],pts[j,1]]
            c=cmap(norm((errors[i]+errors[j])/2))
            ax.plot(x,y,color=c,linewidth=3.0,solid_capstyle='round')
        for idx in range(21):
            c=cmap(norm(errors[idx]))
            ax.scatter(pts[idx,0],pts[idx,1],c=[c],s=80,zorder=5,
                      edgecolors='#333',linewidths=0.8)
        ax.set_title(title,color=tcol,fontsize=8,fontweight='bold',pad=3)
        ax.set_xlim(-1.1,0.7); ax.set_ylim(-1.8,0.2)
        ax.axis('off'); ax.set_aspect('equal')

    # 共享colorbar
    sm=plt.cm.ScalarMappable(cmap=cmap,norm=norm); sm.set_array([])
    cbar=fig.colorbar(sm,ax=axes,shrink=0.7,pad=0.02,aspect=20)
    cbar.set_label('mm',fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    plt.tight_layout(pad=0.4)
    fig.canvas.draw()
    w,h=fig.canvas.get_width_height()
    img=np.frombuffer(fig.canvas.buffer_rgba(),dtype=np.uint8).reshape(h,w,4)
    img=cv2.cvtColor(img,cv2.COLOR_RGBA2BGR)
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
ckpt=torch.load('/data/data5/zhaoran/paper_code/total/checkpoints/best_latent.pth',map_location=device)
total.load_state_dict(ckpt['model_state_dict']); total.eval()

baseline=BaselineModel().to(device)
ckpt_b=torch.load('/data/data5/zhaoran/paper_code/baseline/checkpoints/best_model.pth',map_location=device)
baseline.load_state_dict(ckpt_b['model_state_dict'],strict=False); baseline.eval()
print('Models loaded')

os.makedirs('/data/data5/zhaoran/paper_code/comparison_vis',exist_ok=True)
seq_list=[149,10,23]; t=8

grid_rows=[]
# 列标题
col_w=SIZE; heat_w=360
total_w=col_w*4+6+heat_w+4
title_h=30
title_row=np.ones((title_h,total_w,3),dtype=np.uint8)*245
for ci,(txt,col) in enumerate(zip(
    ['Exo Input','Baseline (62.20mm)','SFLNet Ours (34.42mm)','Ground Truth'],
    [(50,50,50),(100,60,200),(20,130,20),(40,140,40)])):
    cv2.putText(title_row,txt,(ci*(col_w+2)+6,20),
                cv2.FONT_HERSHEY_SIMPLEX,0.45,col,1)
cv2.putText(title_row,'Per-Joint Error (mm)',
            (col_w*4+10,20),cv2.FONT_HERSHEY_SIMPLEX,0.45,(80,80,80),1)
grid_rows.append(title_row)

sep_h=np.ones((2,total_w,3),dtype=np.uint8)*180
sep_v=np.ones((SIZE,2,3),dtype=np.uint8)*180

for idx,seq_idx in enumerate(seq_list):
    seq=dataset.sequences[seq_idx]
    sample=dataset[seq_idx]
    exo_video=sample['exo_video'].unsqueeze(0).to(device)
    exo_pose=sample['exo_pose'].unsqueeze(0).to(device)
    ego_pose=sample['ego_pose'].unsqueeze(0).to(device)
    gt_joints=sample['ego_keypoints'].numpy()

    with torch.no_grad():
        out=total(exo_video,exo_pose=exo_pose,ego_pose=ego_pose)
        if isinstance(out,tuple): out=out[0]
        pred_total=out[0].cpu().numpy()
        pred_base=baseline(exo_video)[0].cpu().numpy()

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
    gt_img=draw_skeleton(ego_img,j2d_s,(0,200,0),(0,255,0))
    kp2d_t=project_joints(pred_total[t],wrist,intr)
    total_img=draw_skeleton(ego_img,kp2d_t,(255,120,0),(255,180,50))
    kp2d_b=project_joints(pred_base[t],wrist,intr)
    base_img=draw_skeleton(ego_img,kp2d_b,(80,80,200),(130,130,230))

    err_t=np.sqrt(np.sum((pred_total[t]-gt_joints[t])**2,axis=-1))*1000
    err_b=np.sqrt(np.sum((pred_base[t]-gt_joints[t])**2,axis=-1))*1000

    # per-joint热力图
    heat_img=draw_perjoint(err_b,err_t)
    # resize热力图高度到SIZE
    heat_img=cv2.resize(heat_img,(heat_w,SIZE))

    row=np.hstack([
        add_label(exo_img,f'seq{seq_idx:03d}',(255,255,255)),sep_v,
        add_label(base_img,f'{err_b.mean():.1f}mm',(160,160,255)),sep_v,
        add_label(total_img,f'{err_t.mean():.1f}mm',(255,180,80)),sep_v,
        add_label(gt_img,'GT',(80,220,80)),
        np.ones((SIZE,4,3),dtype=np.uint8)*180,
        heat_img
    ])
    grid_rows.append(row)
    if idx<len(seq_list)-1:
        grid_rows.append(sep_h)
    print(f'seq{seq_idx:03d}  Baseline={err_b.mean():.1f}mm  SFLNet={err_t.mean():.1f}mm')

grid=np.vstack(grid_rows)
out_path='/data/data5/zhaoran/paper_code/comparison_vis/fig3_final.jpg'
cv2.imwrite(out_path,grid,[cv2.IMWRITE_JPEG_QUALITY,95])
print(f'Saved: {out_path}  size={grid.shape}')
