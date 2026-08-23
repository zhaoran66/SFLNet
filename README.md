# SFLNet

本仓库包含论文 **SFLNet** 的 PyTorch 实现代码，研究从 Exo-centric（第三人称）视角到 Ego-centric（第一人称）视角的手部姿态与图像转换。

---

## 仓库结构

```
.
├── total/              # 主方法：基于傅里叶分解与骨骼引导的 Exo→Ego 生成网络
├── total2/             # 主方法改进/备份版本
├── back/               # 主方法早期版本
├── spl/                # 序列端点预测与基线对比
├── compare1/           # Exo2Ego 对比方法（两阶段：Layout Transformer + Diffusion）
├── compare2/           # Syn2Seq-Forcing 对比方法（插值 + 自回归）
├── baseline/           # 基线模型
├── ego_estimator/      # Ego 视角 3D 手部关节点估计器
├── exo/                # Exo 视角相关实验与数据处理
├── benchmark/          # 评测相关代码
├── gen_*.py            # 论文图表/可视化生成脚本
└── visualize_*.py      # 结果可视化脚本
```

---

## 环境依赖

```bash
conda create -n sflnet python=3.11
conda activate sflnet
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt  # 如存在
```

主要依赖：
- PyTorch >= 2.0
- torchvision
- numpy
- opencv-python
- PyYAML
- tqdm
- scikit-image
- einops

---

## 数据准备

本代码基于 [DexYCB](https://dex-ycb.github.io/) 数据集进行训练与评估。请将数据集按如下结构放置：

```
/data/data5/zhaoran/paper_code/data/
└── dexycb/
    ├── 20200709-subject-01/
    ├── 20200709-subject-02/
    └── ...
```

并在各 `config_*.yaml` 中修改 `data_root` 路径。

---

## 训练

### 主方法（total）

```bash
cd total
python train_bone_guided.py --config config_fourier.yaml
```

### 对比方法

```bash
# Exo2Ego
cd ../compare1
python train_exo2ego.py --config config_exo2ego.yaml

# Syn2Seq-Forcing
cd ../compare2
python train_syn2seq.py --config config_syn2seq.yaml
```

---

## 评估

### 主方法评估

```bash
cd total
python eval_bone_guided.py --config config_fourier.yaml
python eval_geodesic_interp.py --config config_fourier.yaml
```

### Geometry Baseline

审稿人要求的刚性变换基线：

```bash
cd total
python eval_geometry_baseline.py --config config_fourier.yaml
```

### 对比方法评估

```bash
cd ../compare1
python evaluate_exo2ego.py

cd ../compare2
python inference_pipeline.py
```

---

## 可视化

生成论文图表：

```bash
python gen_fig3_final.py
python gen_pipeline_vis.py
python visualize_comparison.py
```

---

## 预训练模型

由于 GitHub 文件大小限制，预训练权重（`.pth` / `.pt`）未包含在本仓库中。请将训练好的 checkpoint 放置到对应目录：

- `total/checkpoints/best_latent.pth`
- `baseline/checkpoints/best_model.pth`
- `ego_estimator/best_ego_net.pth`
- `compare2/outputs/syn2seq/best_model.pth`

---

## 主要对比方法说明

| 方法 | 路径 | 核心思想 |
|------|------|---------|
| SFLNet (Ours) | `total/` | 傅里叶分解 + 骨骼引导的潜在扩散模型 |
| Exo2Ego | `compare1/` | 先预测 Layout/关键点，再用 Diffusion 生成 Ego 图像 |
| Syn2Seq-Forcing | `compare2/` | 插值 + 自回归序列建模 |
| Geometry Baseline | `total/eval_geometry_baseline.py` | Exo 3D 姿态经相机外参刚性变换到 Ego 空间 |

---

## 备注

- 代码中部分绝对路径已根据仓库结构调整，统一指向 `/data/data5/zhaoran/paper_code/`。
- 如需在其他机器上运行，请修改各 `config_*.yaml` 中的数据路径与 checkpoint 路径。

---

## 引用

```bibtex
@article{sflnet2025,
  title={SFLNet: Exo-to-Ego Hand Pose and View Synthesis},
  author={Anonymous},
  journal={},
  year={2025}
}
```
