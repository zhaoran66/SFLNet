import sys, torch, numpy as np, cv2, yaml, os
sys.path.insert(0, '/data/data5/zhaoran/paper_code/total')
sys.path.insert(0, '/data/data5/zhaoran/paper_code/spl')

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

def add_label(img, text, color=(255,255,255), pos=(8,22)):
    img=img.copy()
    cv2.putText(img,text,pos,cv2.FONT_HERSHEY_SIMPLEX,0.52,(0,0,0),3)
    cv2.putText(img,text,pos,cv2.FONT_HERSHEY_SIMPLEX,0.52,color,1)
    return img

# 加载数据集
from dataset_dexycb import DexYCBMultiViewReal
cfg = yaml.safe_load(open('/data/data5/zhaoran/paper_code/total/config_fourier.yaml'))
dataset = DexYCBMultiViewReal(cfg, split='val')

# 加载 SFLNet
from latent_bone_guided import LatentHandPoseModel
total = LatentHandPoseModel(
    embed_dim=cfg['model']['hidden_dim'],
    num_interpolation_steps=cfg['model']['interpolate_steps'],
    num_joints=cfg['dataset']['num_joints'],
    num_views=len(cfg['dataset']['exo_views'])
).to(device)
ckpt = torch.load('/data/data5/zhaoran/paper_code/total/checkpoints/best_latent.pth', map_location=device)
total.load_state_dict(ckpt['model_state_dict'])
total.eval()
print('SFLNet loaded')

# 加载 Baseline
sys.path.insert(0, '/data/data5/zhaoran')
from baseline_model import BaselineModel
baseline = BaselineModel().to(device)
ckpt_b = torch.load('/data/data5/zhaoran/paper_code/baseline/checkpoints/best_model.pth', map_location=device)
baseline.load_state_dict(ckpt_b['model_state_dict'], strict=False)
baseline.eval()
print('Baseline loaded')

os.makedirs('/data/data5/zhaoran/paper_code/comparison_vis', exist_ok=True)

# 选效果对比明显的序列：好/中/难
# best: 149(14mm), 101(15mm), 157(14mm)
# mid:  10(19mm),  23(?)
# hard: 选MPJPE较大的
seq_list = [149, 10, 23]
t = 8

rows = []
for seq_idx in seq_list:
    seq = dataset.sequences[seq_idx]
    sample = dataset[seq_idx]
    exo_video = sample['exo_video'].unsqueeze(0).to(device)
    exo_pose  = sample['exo_pose'].unsqueeze(0).to(device)
    ego_pose  = sample['ego_pose'].unsqueeze(0).to(device)
    gt_joints = sample['ego_keypoints'].numpy()

    with torch.no_grad():
        out = total(exo_video, exo_pose=exo_pose, ego_pose=ego_pose)
        if isinstance(out, tuple): out = out[0]
        pred_total = out[0].cpu().numpy()

        out_b = baseline(exo_video)
        if isinstance(out_b, tuple): out_b = out_b[0]
        pred_base = out_b[0].cpu().numpy()

    frame_data = seq['frames'][t]
    raw_ego = dataset.ds[frame_data[dataset.ego_view]]
    raw_exo = dataset.ds[frame_data[dataset.exo_views[0]]]
    data_ego = np.load(raw_ego['label_file'])
    j2d_ego  = data_ego['joint_2d'].squeeze()
    j3d_ego  = data_ego['joint_3d'].squeeze()
    intr = raw_ego['intrinsics']
    wrist = j3d_ego[0]

    exo_img = cv2.resize(cv2.imread(raw_exo['color_file']), (SIZE,SIZE))
    ego_img = cv2.resize(cv2.imread(raw_ego['color_file']), (SIZE,SIZE))

    # GT
    j2d_s = j2d_ego.copy(); j2d_s[:,0]*=sx; j2d_s[:,1]*=sy
    gt_img = draw_skeleton(ego_img, j2d_s, (0,200,0), (0,255,0), 2, 3)

    # SFLNet
    kp2d_total = project_joints(pred_total[t], wrist, intr)
    total_img  = draw_skeleton(ego_img, kp2d_total, (255,120,0), (255,180,50), 2, 3)

    # Baseline
    kp2d_base = project_joints(pred_base[t], wrist, intr)
    base_img  = draw_skeleton(ego_img, kp2d_base, (80,80,200), (130,130,230), 2, 3)

    # MPJPE per sequence
    err_total = np.mean(np.sqrt(np.sum((pred_total - gt_joints)**2, axis=-1)))*1000
    err_base  = np.mean(np.sqrt(np.sum((pred_base  - gt_joints)**2, axis=-1)))*1000

    rows.append((seq_idx, exo_img, base_img, total_img, gt_img, err_base, err_total))
    print(f'seq{seq_idx:03d}  Baseline={err_base:.1f}mm  SFLNet={err_total:.1f}mm')

# 拼图：4列（exo | baseline | SFLNet | GT），3行
title_h = 28
title_row = np.ones((title_h, SIZE*4+6, 3), dtype=np.uint8)*245
for ci,(txt,col) in enumerate(zip(
    ['Exo Input','Baseline (62.20 mm)','SFLNet Ours (34.42 mm)','Ground Truth'],
    [(50,50,50),(100,80,200),(180,100,20),(40,140,40)])):
    cv2.putText(title_row, txt, (ci*(SIZE+2)+6, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1)

grid_rows = [title_row]
sep_v = np.ones((SIZE, 2, 3), dtype=np.uint8)*160
sep_h = np.ones((2, SIZE*4+6, 3), dtype=np.uint8)*160

for idx,(seq_idx,exo_img,base_img,total_img,gt_img,err_b,err_t) in enumerate(rows):
    row = np.hstack([
        add_label(exo_img,   f'seq{seq_idx:03d}',    (255,255,255)), sep_v,
        add_label(base_img,  f'{err_b:.1f}mm',       (160,160,255)), sep_v,
        add_label(total_img, f'{err_t:.1f}mm',       (255,180,80)),  sep_v,
        add_label(gt_img,    'GT',                    (80,220,80))
    ])
    grid_rows.append(row)
    if idx < len(rows)-1:
        grid_rows.append(sep_h)

grid = np.vstack(grid_rows)
out_path = '/data/data5/zhaoran/paper_code/comparison_vis/fig3_comparison.jpg'
cv2.imwrite(out_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 95])
print(f'\nSaved: {out_path}  size={grid.shape}')
