import sys, torch, numpy as np, cv2, yaml, os
sys.path.insert(0, '/data/data5/zhaoran/paper_code/total')
from dataset_dexycb import DexYCBMultiViewReal
from latent_bone_guided import LatentHandPoseModel

cfg = yaml.safe_load(open('/data/data5/zhaoran/paper_code/total/config_fourier.yaml'))
device = torch.device('cuda:0')
dataset = DexYCBMultiViewReal(cfg, split='val')

total = LatentHandPoseModel(
    embed_dim=cfg['model']['hidden_dim'],
    num_interpolation_steps=cfg['model']['interpolate_steps'],
    num_joints=cfg['dataset']['num_joints'],
    num_views=len(cfg['dataset']['exo_views'])
).to(device)
ckpt = torch.load('/data/data5/zhaoran/paper_code/total/checkpoints/best_latent.pth', map_location=device)
total.load_state_dict(ckpt['model_state_dict'])
total.eval()

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

os.makedirs('/data/data5/zhaoran/paper_code/pipeline_vis', exist_ok=True)
SIZE = 256
ORIG_W, ORIG_H = 640, 480
sx, sy = SIZE/ORIG_W, SIZE/ORIG_H

for seq_idx in [10, 11, 23, 3, 15]:
    seq = dataset.sequences[seq_idx]
    sample = dataset[seq_idx]
    exo_video = sample['exo_video'].unsqueeze(0).to(device)
    exo_pose = sample['exo_pose'].unsqueeze(0).to(device)
    ego_pose = sample['ego_pose'].unsqueeze(0).to(device)

    with torch.no_grad():
        out = total(exo_video, exo_pose=exo_pose, ego_pose=ego_pose)
        if isinstance(out, tuple): out = out[0]
        pred_kp = out[0].cpu().numpy()  # [T,21,3]

    t = 8
    frame_data = seq['frames'][t]
    raw_ego = dataset.ds[frame_data[dataset.ego_view]]
    raw_exo = dataset.ds[frame_data[dataset.exo_views[0]]]
    data_ego = np.load(raw_ego['label_file'])
    j2d_ego = data_ego['joint_2d'].squeeze()
    j3d_ego = data_ego['joint_3d'].squeeze()
    intr = raw_ego['intrinsics']
    fx,fy,ppx,ppy = intr['fx'],intr['fy'],intr['ppx'],intr['ppy']
    wrist = j3d_ego[0]

    ego_img = cv2.imread(raw_ego['color_file'])
    ego_img = cv2.resize(ego_img, (SIZE, SIZE))
    exo_img = cv2.imread(raw_exo['color_file'])
    exo_img = cv2.resize(exo_img, (SIZE, SIZE))

    def draw_skeleton(img, kp2d, color, thickness=2):
        for i,j in HAND_CONNECTIONS:
            x1,y1=int(kp2d[i,0]),int(kp2d[i,1])
            x2,y2=int(kp2d[j,0]),int(kp2d[j,1])
            if 0<=x1<SIZE and 0<=y1<SIZE and 0<=x2<SIZE and 0<=y2<SIZE:
                cv2.line(img,(x1,y1),(x2,y2),color,thickness)
        for i in range(21):
            x,y=int(kp2d[i,0]),int(kp2d[i,1])
            if 0<=x<SIZE and 0<=y<SIZE:
                cv2.circle(img,(x,y),2,color,-1)

    # GT骨架在ego图上
    gt_img = ego_img.copy()
    if j3d_ego[0,0] != -1:
        j2d_s = j2d_ego.copy()
        j2d_s[:,0] *= sx; j2d_s[:,1] *= sy
        draw_skeleton(gt_img, j2d_s, (0,255,0))

    # 预测骨架在ego图上
    pred_img = ego_img.copy()
    kp_abs = pred_kp[t] + wrist
    Z = np.clip(kp_abs[:,2], 0.1, 10.0)
    u = (fx*kp_abs[:,0]/Z+ppx)*sx
    v = (fy*kp_abs[:,1]/Z+ppy)*sy
    kp2d = np.stack([u,v],axis=1)
    draw_skeleton(pred_img, kp2d, (180,160,255))

    # 保存各图
    cv2.imwrite(f'/data/data5/zhaoran/paper_code/pipeline_vis/seq{seq_idx:02d}_exo.jpg', exo_img)
    cv2.imwrite(f'/data/data5/zhaoran/paper_code/pipeline_vis/seq{seq_idx:02d}_ego.jpg', ego_img)
    cv2.imwrite(f'/data/data5/zhaoran/paper_code/pipeline_vis/seq{seq_idx:02d}_gt.jpg', gt_img)
    cv2.imwrite(f'/data/data5/zhaoran/paper_code/pipeline_vis/seq{seq_idx:02d}_pred.jpg', pred_img)

    # 拼接：exo | ego | GT | pred
    combined = np.hstack([exo_img, ego_img, gt_img, pred_img])
    cv2.imwrite(f'/data/data5/zhaoran/paper_code/pipeline_vis/seq{seq_idx:02d}_combined.jpg', combined)
    print(f'Saved seq{seq_idx:02d}')
