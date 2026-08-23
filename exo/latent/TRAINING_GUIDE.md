# Latent Syn2Seq - 训练说明

## 版本对比

| 版本 | 脚本 | Epoch | 学习率 | Interpolator | Scheduler |
|------|------|-------|--------|--------------|-----------|
| **v1** | `train.py` | 100 | 1e-4 | 64 dim, 3 blocks | ? No |
| **v2** | `train_v2.py` | 500 | 2e-4 | 128 dim, 5 blocks + skip | ? Cosine |

---

## 当前指标 (v1 跑了 100 epoch):

| 方法 | PSNR | SSIM | 说明 |
|------|------|------|------|
| **Latent (v1)** | 17.76 dB | -0.0187 | ? 训练不足 |
| **Baseline (像素)** | 18.89 dB | 0.00xx | ? Syn2Seq 原版 |

---

## 运行训练

### v2 (推荐，目标追上 Baseline):

```bash
# 默认 500 epoch
python train_v2.py --output_dir ./outputs_latent_v2

# 更快的选项 (200 epoch)
python train_v2.py --num_epochs 200 --lr 3e-4 --output_dir ./outputs_latent_v2_200e

# 禁用 temporal loss
python train_v2.py --no_temporal --num_epochs 300
```

---

## 运行评测

```bash
# CPU 快速跑少量样本
python quick_eval_latent.py --checkpoint ./outputs_latent_v2/checkpoints/checkpoint_best.pt --num_batches 5

# GPU 跑全量验证 (200 samples)
python evaluate_latent.py --checkpoint ./outputs_latent_v2/checkpoints/checkpoint_best.pt --device cuda
```

---

## 改进点说明

### 1. 学习率调度器 (Cosine Annealing)
- 从 2e-4 降到 2e-6，有助于后期精细优化
- 总步数 = 800 batches × 500 epoch = 400k steps

### 2. 更大的 Interpolator
- hidden_dim: 64 → 128
- res_blocks: 3 → 5
- 加了 skip connection: `h + skip` 的融合
- 加了 GroupNorm 在 out_conv 中间

### 3. Pixel-level 重建损失
- latent 算 diffusion 损失时，decode 回 RGB 再额外加一层 MSE
- 抵消 VAE 量化/压缩带来的信息损失
- 权重 0.5，平衡 latent 级损失和像素级损失

### 4. Checkpoint 策略
- `checkpoint_best.pt`: 验证 loss 最低的 ckpt
- `checkpoint_{epoch}.pt`: 每 50 epoch 存一次
- 含 optimizer / scheduler / scaler state，支持断点续训

---

## 预期效果 (500 epoch 后)

目标: **PSNR > 18.5 dB**, **SSIM > 0**

估计需要 **3-4 小时** 在单卡 A100 / V100 上跑完。

---

## 文件结构

```
latent/
├── train_v2.py                    # v2 训练入口
├── evaluate_latent.py             # 全量评测
├── quick_eval_latent.py           # 快速评测
├── configs/
│   └── default_config.py          # 默认配置
├── models/
│   ├── dfot.py                    # Latent Diffusion Transformer
│   ├── interpolator.py            # V1 + V2 interpolator
│   └── vae.py                     # FrameVAE wrapper
└── training/
    ├── trainer.py                 # v1 trainer
    └── trainer_v2.py              # v2 trainer (改进版)
```
