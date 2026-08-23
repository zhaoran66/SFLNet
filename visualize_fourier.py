import sys
import torch
import numpy as np
import cv2
import os
sys.path.insert(0, '/data/data5/zhaoran/paper_code/total')
from dataset_dexycb import DexYCBMultiViewReal
import yaml

def temporal_fourier(video):
    """
    video: [T, 3, H, W] tensor, normalized
    return: orig, low_freq, high_freq, all [H, W, 3] uint8
    """
    T, C, H, W = video.shape
    # 转numpy [T,H,W,C]
    v = video.permute(0,2,3,1).numpy()  # [T,H,W,3]
    
    # 沿T轴做FFT
    fft = np.fft.fft(v, axis=0)  # [T,H,W,3]
    
    # 低频mask：中心±r个频率
    r = max(1, int(T * 0.1))
    mask_low = np.zeros(T, dtype=bool)
    mask_low[:r+1] = True
    mask_low[-(r):] = True
    
    # 高频mask
    mask_high = ~mask_low
    
    # 低频重建
    fft_low = fft.copy()
    fft_low[~mask_low] = 0
    low = np.fft.ifft(fft_low, axis=0).real  # [T,H,W,3]
    
    # 高频重建
    fft_high = fft.copy()
    fft_high[~mask_high] = 0
    high = np.fft.ifft(fft_high, axis=0).real  # [T,H,W,3]
    
    # 取中间帧
    t = T // 2
    orig = v[t]
    low_t = low[t]
    high_t = high[t]
    
    # 反归一化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    orig_img = np.clip((orig * std + mean) * 255, 0, 255).astype(np.uint8)
    
    # 低频：归一化到0-255
    low_img = low_t * std + mean
    low_img = np.clip(low_img * 255, 0, 255).astype(np.uint8)
    
    # 高频：归一化到0-255（放大对比度）
    high_norm = (high_t - high_t.min()) / (high_t.max() - high_t.min() + 1e-8)
    high_img = (high_norm * 255).astype(np.uint8)
    
    return orig_img, low_img, high_img

def run():
    cfg = yaml.safe_load(open('/data/data5/zhaoran/paper_code/total/config_fourier.yaml'))
    dataset = DexYCBMultiViewReal(cfg, split='val')
    
    os.makedirs('/data/data5/zhaoran/paper_code/fourier_vis', exist_ok=True)
    
    # 找手部清晰的序列
    good_seqs = [10, 11, 23, 3, 15]
    
    for demo_idx, seq_idx in enumerate(good_seqs):
        sample = dataset[seq_idx]
        exo_video = sample['exo_video'].squeeze(1)  # [T,3,H,W] squeeze掉view维度
        
        orig, low, high = temporal_fourier(exo_video)
        
        # BGR转换
        orig_bgr = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)
        low_bgr = cv2.cvtColor(low, cv2.COLOR_RGB2BGR)
        high_bgr = cv2.cvtColor(high, cv2.COLOR_RGB2BGR)
        
        # 保存单独图片
        cv2.imwrite(f'/data/data5/zhaoran/paper_code/fourier_vis/seq{seq_idx:02d}_orig.jpg', orig_bgr)
        cv2.imwrite(f'/data/data5/zhaoran/paper_code/fourier_vis/seq{seq_idx:02d}_low.jpg', low_bgr)
        cv2.imwrite(f'/data/data5/zhaoran/paper_code/fourier_vis/seq{seq_idx:02d}_high.jpg', high_bgr)
        
        # 拼接横排：原始 | 低频 | 高频
        H, W = orig.shape[:2]
        # 加标签
        def add_label(img, label, color):
            img = img.copy()
            cv2.putText(img, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, color, 2)
            return img
        
        orig_l = add_label(orig_bgr, 'Original (3ch)', (255,255,255))
        low_l = add_label(low_bgr, 'Low-freq: bg (3ch)', (50,200,50))
        high_l = add_label(high_bgr, 'High-freq: hand (3ch)', (100,100,255))
        
        combined = np.hstack([orig_l, low_l, high_l])
        cv2.imwrite(f'/data/data5/zhaoran/paper_code/fourier_vis/seq{seq_idx:02d}_combined.jpg', combined)
        print(f"Saved seq{seq_idx:02d}")

if __name__ == '__main__':
    run()
