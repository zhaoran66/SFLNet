import sys
sys.path.insert(0, '/data/data5/zhaoran/paper_code/total')
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from model import EgoNet
from dataset_dexycb import DexYCBMultiViewReal

class EgoDatasetWrapper(torch.utils.data.Dataset):
    """把DexYCB数据集包装成ego输入"""
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        # 用ego_video作为输入（需要数据集支持）
        return {
            'ego_video': sample.get('ego_video', sample['exo_video'][:, 0]),
            'ego_keypoints': sample['ego_keypoints']
        }

def train():
    cfg = yaml.safe_load(open('/data/data5/zhaoran/paper_code/total/config_fourier.yaml'))
    
    # 修改配置：用ego视角
    cfg['dataset']['exo_views'] = [5]  # ego相机作为输入
    
    train_dataset = DexYCBMultiViewReal(cfg, split='train')
    val_dataset = DexYCBMultiViewReal(cfg, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
    
    model = EgoNet(hidden_dim=256, num_joints=21).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = nn.MSELoss()
    
    best_mpjpe = float('inf')
    
    for epoch in range(50):
        model.train()
        for batch in train_loader:
            ego_video = batch['exo_video'].cuda()  # [B,T,1,3,H,W]
            ego_video = ego_video.squeeze(2)        # [B,T,3,H,W]
            gt_kp = batch['ego_keypoints'].cuda()
            
            pred_kp = model(ego_video)
            loss = criterion(pred_kp, gt_kp) + 0.5 * nn.L1Loss()(pred_kp, gt_kp)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # 验证
        model.eval()
        mpjpe_list = []
        with torch.no_grad():
            for batch in val_loader:
                ego_video = batch['exo_video'].cuda().squeeze(2)
                gt_kp = batch['ego_keypoints'].cuda()
                pred_kp = model(ego_video)
                # wrist-centering
                pred_wrist = pred_kp[:, :, 0:1, :]
                pred_centered = pred_kp - pred_wrist
                mpjpe = torch.norm(pred_centered - gt_kp, dim=-1).mean() * 1000
                mpjpe_list.append(mpjpe.item())
        
        avg_mpjpe = sum(mpjpe_list) / len(mpjpe_list)
        print(f"Epoch {epoch+1}/50 | Val MPJPE: {avg_mpjpe:.2f}mm")
        
        if avg_mpjpe < best_mpjpe:
            best_mpjpe = avg_mpjpe
            torch.save(model.state_dict(), '/data/data5/zhaoran/paper_code/ego_estimator/best_ego_net.pth')
            print(f"  → Best model saved: {best_mpjpe:.2f}mm")
        
        scheduler.step()
    
    print(f"\nTraining complete! Best MPJPE: {best_mpjpe:.2f}mm")

if __name__ == '__main__':
    train()
