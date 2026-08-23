import sys, torch, numpy as np, cv2, yaml, os
sys.path.insert(0, '/data/data5/zhaoran/paper_code/total')
sys.path.insert(0, '/data/data5/zhaoran')

device = torch.device('cuda:0')
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]
SIZE = 256

def add_label(img, text, color=(255,255,255), pos=(8,22)):
    img = img.copy()
    cv2.putText(img,text,pos,cv2.FONT_HERSHEY_SIMPLEX,0.52,(0,0,0),3)
    cv2.putText(img,text,pos,cv2.FONT_HERSHEY_SIMPLEX,0.52,color,1)
    return img

def to2d(kp, ref_pts, size=256, margin=0.12):
    xy = kp[:,:2].copy()
    xmin,xmax = ref_pts[:,0].min(), ref_pts[:,0].max()
    ymin,ymax = ref_pts[:,1].min(), ref_pts[:,1].max()
    rng = max(xmax-xmin, ymax-ymin) + 1e-6
    xy[:,0] = (kp[:,0]-xmin)/rng*(1-2*margin)+margin
    xy[:,1] = (kp[:,1]-ymin)/rng*(1-2*margin)+margin
    xy[:,1] = 1 - xy[:,1]
    return (xy*size).astype(np.float32)

def draw_skel_2d(pred_kp2d, gt_kp2d, pred_color, size=256):
    img = np.ones((size,size,3),dtype=np.uint8)*255
    # GT绿色
    for i,j in HAND_CONNECTIONS:
        x1,y1=int(gt_kp2d[i,0]),int(gt_kp2d[i,1])
        x2,y2=int(gt_kp2d[j,0]),int(gt_kp2d[j,1])
        if 0<=x1<size and 0<=y1<size and 0<=x2<size and 0<=y2<size:
            cv2.line(img,(x1,y1),(x2,y2),(0,180,0),2)
    for i in range(21):
        x,y=int(gt_kp2d[i,0]),int(gt_kp2d[i,1])
        if 0<=x<size and 0<=y<size:
            cv2.circle(img,(x,y),4,(0,220,0),-1)
            cv2.circle(img,(x,y),4,(0,80,0),1)
    # 预测骨架
    for i,j in HAND_CONNECTIONS:
        x1,y1=int(pred_kp2d[i,0]),int(pred_kp2d[i,1])
        x2,y2=int(pred_kp2d[j,0]),int(pred_kp2d[j,1])
        if 0<=x1<size and 0<=y1<size and 0<=x2<size and 0<=y2<size:
            cv2.line(img,(x1,y1),(x2,y2),pred_color,2)
    for i in range(21):
        x,y=int(pred_kp2d[i,0]),int(pred_kp2d[i,1])
        if 0<=x<size and 0<=y<size:
            cv2.circle(img,(x,y),4,pred_color,-1)
            cv2.circle(img,(x,y),4,(0,0,0),1)
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
ckpt = torch.load('/data/data5/zhaoran/paper_code/total/checkpoints/best_latent.pth', map_location=device)
total.load_state_dict(ckpt['model_state_dict']); total.eval()

baseline = BaselineModel().to(device)
ckpt_b = torch.load('/data/data5/zhaoran/paper_code/baseline/checkpoints/best_model.pth', map_location=device)
baseline.load_state_dict(ckpt_b['model_state_dict'], strict=False); baseline.eval()
print('Models loaded')

os.makedirs('/data/data5/zhaoran/paper_code/comparison_vis', exist_ok=True)
seq_list = [149, 10, 23]; t = 8

sep_v = np.ones((SIZE, 2, 3), dtype=np.uint8)*180
sep_h = np.ones((3, SIZE*3+4, 3), dtype=np.uint8)*180

grid_rows = []
for idx, seq_idx in enumerate(seq_list):
    seq = dataset.sequences[seq_idx]
    sample = dataset[seq_idx]
    exo_video = sample['exo_video'].unsqueeze(0).to(device)
    exo_pose  = sample['exo_pose'].unsqueeze(0).to(device)
    ego_pose  = sample['ego_pose'].unsqueeze(0).to(device)
    gt_joints = sample['ego_keypoints'].numpy()

    with torch.no_grad():
        out = total(exo_video, exo_pose=exo_pose, ego_pose=ego_pose)
        if isinstance(out, tuple): out = out[0]
        pred_t = out[0].cpu().numpy()
        pred_b = baseline(exo_video)[0].cpu().numpy()

    gt_t = gt_joints[t]
    pt   = pred_t[t]
    pb   = pred_b[t]
    err_t = np.mean(np.sqrt(np.sum((pred_t-gt_joints)**2, axis=-1)))*1000
    err_b = np.mean(np.sqrt(np.sum((pred_b-gt_joints)**2, axis=-1)))*1000

    # exo图
    frame_data = seq['frames'][t]
    raw_exo = dataset.ds[frame_data[dataset.exo_views[0]]]
    exo_img = cv2.resize(cv2.imread(raw_exo['color_file']), (SIZE,SIZE))

    # 统一坐标归一化
    all_pts = np.vstack([gt_t, pt, pb])
    gt2d = to2d(gt_t, all_pts)
    pt2d = to2d(pt,   all_pts)
    pb2d = to2d(pb,   all_pts)

    sfl_img  = draw_skel_2d(pt2d, gt2d, (220,100,20))  # SFLNet橙+GT绿
    base_img = draw_skel_2d(pb2d, gt2d, (60,60,210))   # Baseline蓝+GT绿

    exo_l  = add_label(exo_img,  f'seq{seq_idx:03d}',       (255,255,255))
    sfl_l  = add_label(sfl_img,  f'SFLNet {err_t:.1f}mm',   (220,100,20))
    base_l = add_label(base_img, f'Baseline {err_b:.1f}mm', (60,60,210))

    row = np.hstack([exo_l, sep_v, sfl_l, sep_v, base_l])
    grid_rows.append(row)
    if idx < len(seq_list)-1:
        grid_rows.append(sep_h)
    print(f'seq{seq_idx:03d}  Baseline={err_b:.1f}mm  SFLNet={err_t:.1f}mm')

# 标题行（用第一行row的宽度）
W = grid_rows[0].shape[1]
title_h = 30
title_row = np.ones((title_h, W, 3), dtype=np.uint8)*245
cv2.putText(title_row,'Exo Input',(6,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(50,50,50),1)
cv2.putText(title_row,'SFLNet (Ours) + GT',(SIZE+6,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(20,120,20),1)
cv2.putText(title_row,'Baseline + GT',(SIZE*2+8,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(60,60,200),1)

# 图例行
legend_h = 30
legend = np.ones((legend_h, W, 3), dtype=np.uint8)*245
cv2.circle(legend,(16,15),6,(0,200,0),-1)
cv2.putText(legend,'Ground Truth',(26,19),cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,100,0),1)
cv2.circle(legend,(160,15),6,(220,100,20),-1)
cv2.putText(legend,'SFLNet Pred',(172,19),cv2.FONT_HERSHEY_SIMPLEX,0.42,(160,80,0),1)
cv2.circle(legend,(290,15),6,(60,60,210),-1)
cv2.putText(legend,'Baseline Pred',(302,19),cv2.FONT_HERSHEY_SIMPLEX,0.42,(40,40,160),1)

final = np.vstack([
    title_row,
    np.ones((2,W,3),dtype=np.uint8)*180,
    *grid_rows,
    np.ones((2,W,3),dtype=np.uint8)*180,
    legend
])

out_path = '/data/data5/zhaoran/paper_code/comparison_vis/fig3_skeleton.jpg'
cv2.imwrite(out_path, final, [cv2.IMWRITE_JPEG_QUALITY,95])
print(f'Saved: {out_path}  size={final.shape}')
