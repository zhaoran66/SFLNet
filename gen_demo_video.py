import sys, torch, numpy as np, cv2, yaml, os
sys.path.insert(0, '/data/data5/zhaoran/paper_code/total')
from dataset_dexycb import DexYCBMultiViewReal
from latent_bone_guided import LatentHandPoseModel

device = torch.device('cuda:0')
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]
SIZE = 256; ORIG_W, ORIG_H = 640, 480; sx, sy = SIZE/ORIG_W, SIZE/ORIG_H

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

cfg = yaml.safe_load(open('/data/data5/zhaoran/paper_code/total/config_fourier.yaml'))
dataset = DexYCBMultiViewReal(cfg, split='val')

total = LatentHandPoseModel(
    embed_dim=cfg['model']['hidden_dim'],
    num_interpolation_steps=cfg['model']['interpolate_steps'],
    num_joints=cfg['dataset']['num_joints'],
    num_views=len(cfg['dataset']['exo_views'])
).to(device)
ckpt=torch.load('/data/data5/zhaoran/paper_code/total/checkpoints/best_latent.pth', map_location=device)
total.load_state_dict(ckpt['model_state_dict'])
total.eval()
print('Model loaded')

os.makedirs('/data/data5/zhaoran/paper_code/demo_videos', exist_ok=True)

# 选效果最好的5个序列
best_seqs = [149, 157, 101, 98, 12]

for seq_idx in best_seqs:
    seq = dataset.sequences[seq_idx]
    sample = dataset[seq_idx]
    exo_video = sample['exo_video'].unsqueeze(0).to(device)
    exo_pose  = sample['exo_pose'].unsqueeze(0).to(device)
    ego_pose  = sample['ego_pose'].unsqueeze(0).to(device)
    gt_joints = sample['ego_keypoints'].numpy()  # [T,21,3]

    with torch.no_grad():
        out = total(exo_video, exo_pose=exo_pose, ego_pose=ego_pose)
        if isinstance(out, tuple): out = out[0]
        pred = out[0].cpu().numpy()  # [T,21,3]

    T = pred.shape[0]
    mpjpe = np.mean(np.sqrt(np.sum((pred - gt_joints)**2, axis=-1))) * 1000

    # 视频输出：横排三列 exo | ego+pred | ego+GT
    out_w = SIZE * 3 + 4
    out_h = SIZE + 30  # 底部留标签
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vpath = f'/data/data5/zhaoran/paper_code/demo_videos/seq{seq_idx:03d}_demo.mp4'
    writer = cv2.VideoWriter(vpath, fourcc, 8, (out_w, out_h))

    for t in range(T):
        frame_data = seq['frames'][t]
        raw_ego = dataset.ds[frame_data[dataset.ego_view]]
        raw_exo = dataset.ds[frame_data[dataset.exo_views[0]]]
        data_ego = np.load(raw_ego['label_file'])
        j2d_ego = data_ego['joint_2d'].squeeze()
        j3d_ego = data_ego['joint_3d'].squeeze()
        intr = raw_ego['intrinsics']
        wrist = j3d_ego[0]

        exo_img = cv2.resize(cv2.imread(raw_exo['color_file']), (SIZE,SIZE))
        ego_img = cv2.resize(cv2.imread(raw_ego['color_file']), (SIZE,SIZE))

        # GT
        j2d_s = j2d_ego.copy(); j2d_s[:,0]*=sx; j2d_s[:,1]*=sy
        gt_img = draw_skeleton(ego_img, j2d_s, (0,220,0), (0,255,0))

        # Pred
        kp2d = project_joints(pred[t], wrist, intr)
        pred_img = draw_skeleton(ego_img, kp2d, (255,120,0), (255,180,50))

        # 拼帧
        sep = np.ones((SIZE,2,3), dtype=np.uint8)*180
        frame = np.hstack([exo_img, sep, pred_img, sep, gt_img])

        # 底部标签
        bar = np.ones((30, out_w, 3), dtype=np.uint8) * 30
        cv2.putText(bar, f'Exo Input', (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        cv2.putText(bar, f'SFLNet Pred  MPJPE={mpjpe:.1f}mm', (SIZE+10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,180,50), 1)
        cv2.putText(bar, f'Ground Truth', (SIZE*2+12,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80,220,80), 1)
        cv2.putText(bar, f't={t+1}/{T}', (out_w-70,20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1)

        full = np.vstack([frame, bar])
        writer.write(full)

    writer.release()
    print(f'seq{seq_idx:03d}  MPJPE={mpjpe:.1f}mm  -> {vpath}')

print('All done!')
