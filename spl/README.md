# 3D Hand Pose Estimation via View Transformation in Latent Space

基于隐空间视角变换的3D手部姿态估计

## 安装依赖

```bash
pip install -r requirements.txt
```

或者单独安装：

```bash
pip install pyyaml numpy torch torchvision tensorboard pillow
```

## 快速开始

### 1. 使用1号GPU训练（默认）

```bash
python train.py --data_root /data/data3/kuanghaohong/datasets/DexYCB --gpu 1
```

### 2. 使用指定GPU训练

```bash
# 使用0号GPU
python train.py --data_root /data/data3/kuanghaohong/datasets/DexYCB --gpu 0

# 使用CPU
python train.py --data_root /data/data3/kuanghaohong/datasets/DexYCB --gpu -1
```

## 模型架构

### 1. 图像编码器（DINOv2 ViT-S/14）
- 输入: 3 × 224 × 224 RGB图像
- 输出: 图像tokens (N × 384) + 全局特征
- 优势: 预训练特征具有良好的泛化能力

### 2. Transformer解码器（6层）
- 隐藏维度: 384
- 自注意力: 建模5个视角查询间的依赖关系
- 交叉注意力: 关联图像特征与每个视角查询

### 3. 可学习姿态查询（5个）
- 5个可学习嵌入向量 (5 × 384)
- 每个查询对应一个中间视角
- 网络学习从侧视图到主视图的"轨迹"

### 4. 几何投影损失
- **MSE损失**: 仅监督最后一步的3D坐标
- **投影损失**: 对5步都进行2D投影监督
- **3D坐标缩放**: ×1000（米→毫米）
- **Lambda调度**: 前10个epoch=1000，之后=5000

## 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--embed_dim` | 384 | 隐藏维度 |
| `--num_layers` | 6 | Transformer解码器层数 |
| `--num_heads` | 8 | 注意力头数 |
| `--num_sequence_steps` | 5 | 视角序列步数 |
| `--batch_size` | 16 | 批次大小 |
| `--lr` | 1e-4 | 学习率 |
| `--num_epochs` | 100 | 训练轮数 |
| `--freeze_dino` | True | 冻结DINOv2编码器 |
| `--gpu` | 1 | GPU设备ID（-1表示使用CPU） |

## 文件说明

- `model.py`: 模型架构定义
- `dataset.py`: DexYCB数据集加载器
- `train.py`: 训练脚本
- `verify_model.py`: 模型验证和架构说明
- `requirements.txt`: Python依赖列表

## 常见问题

### Q1: Projection Loss 数值过大
**A**: 3D坐标（米）和相机内参（像素）的单位不匹配。已修复：
- 3D坐标 × 1000 缩放（米→毫米）
- Lambda 权重从 0.05 增加到 1000/5000

### Q2: YAML解析错误
```
could not determine a constructor for the tag 'tag:yaml.org,2002:python/tuple'
```
**A**: 已修复，使用 `yaml.load(..., Loader=yaml.FullLoader)` 替代 `yaml.safe_load()`。

### Q3: xFormers警告
```
xFormers can't load C++/CUDA extensions.
```
**A**: 这只是警告，不影响训练。DINOv2会使用标准的注意力机制。如果想优化，可以安装匹配版本的xFormers。

### Q4: 数据集路径不存在
**A**: 确认DexYCB数据集路径正确，默认路径是 `/data/data3/kuanghaohong/datasets/DexYCB`。
