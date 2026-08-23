import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import os
sys.path.insert(0, '/data/data5/zhaoran/paper_code/compare1')
sys.path.insert(0, '/data/data5/zhaoran/paper_code/total')
from dataset_dexycb import DexYCBMultiViewReal
from model_syn2seq import Syn2SeqForcing
import torchvision.transforms.functional as TF

def build_dataset(cfg, split):
    """构建同时加载exo和ego视频的数据集"""
    # 修改配置加载ego视频
    ego_cfg = cfg.copy()
    ego_cfg['dataset'] = cfg['dataset'].copy()
    ego_cfg['dataset']['exo_views'] = cfg['dataset']['ego_views'][:1]
    
    exo_dataset = DexYCBMultiViewReal(cfg, split=split)
    ego_dataset = DexYCBMultiViewReal(ego_cfg, split=split)
    return exo_dataset, ego_dataset

class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, exo_ds, ego_ds, img_size):
        self.exo_ds = exo_ds
        self.ego_ds = ego_ds
        self.img_size = img_size
        assert len(exo_ds) == len(ego_ds)

    def __len__(self):
        return len(self.exo_ds)

    def __getitem__(self, idx):
        exo = self.exo_ds[idx]['exo_video'].squeeze(1)  # [T,3,H,W]
        ego = self.ego_ds[idx]['exo_video'].squeeze(1)  # [T,3,H,W]
        gt_kp = self.exo_ds[idx]['ego_keypoints']       # [T,21,3]
        H = self.img_size
        # 批量resize更快
        exo = torch.nn.functional.interpolate(exo, size=(H,H), mode='bilinear', align_corners=False)
        ego = torch.nn.functional.interpolate(ego, size=(H,H), mode='bilinear', align_corners=False)
        return exo, ego, gt_kp

def train():
    cfg = yaml.safe_load(open('/data/data5/zhaoran/paper_code/compare2/config_syn2seq.yaml'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_size = cfg['dataset']['image_size'][0]
    os.makedirs(cfg['training']['output_dir'], exist_ok=True)

    # 数据集
    exo_train, ego_train = build_dataset(cfg, 'train')
    exo_val, ego_val = build_dataset(cfg, 'val')
    train_ds = PairedDataset(exo_train, ego_train, img_size)
    val_ds = PairedDataset(exo_val, ego_val, img_size)
    train_loader = DataLoader(train_ds, batch_size=cfg['training']['batch_size'],
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg['training']['batch_size'],
                            shuffle=False, num_workers=0)

    # 模型
    model = Syn2SeqForcing(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"模型参数量: {total_params:.2f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['training']['lr'],
        weight_decay=cfg['training']['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['training']['num_epochs'])

    best_loss = float('inf')
    for epoch in range(cfg['training']['num_epochs']):
        # 训练
        model.train()
        train_loss = 0
        for batch_idx, (exo, ego, _) in enumerate(train_loader):
            if batch_idx >= 500:
                break
            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch+1} Batch {batch_idx}/500", flush=True)
            exo, ego = exo.to(device), ego.to(device)
            _, loss = model(exo, ego)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for exo, ego, _ in val_loader:
                exo, ego = exo.to(device), ego.to(device)
                model.train()  # 临时切回训练模式以获得loss
                _, loss = model(exo, ego)
                model.eval()
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{cfg['training']['num_epochs']} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch},
                      f"{cfg['training']['output_dir']}/best_model.pth")
            print(f"  → Best model saved")

        if (epoch+1) % cfg['training']['save_interval'] == 0:
            torch.save({'model_state_dict': model.state_dict()},
                      f"{cfg['training']['output_dir']}/epoch_{epoch+1}.pth")

        scheduler.step()

    print(f"训练完成！Best Val Loss: {best_loss:.4f}")

if __name__ == '__main__':
    train()
