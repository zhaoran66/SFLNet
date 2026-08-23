# FASR: Frequency-Adaptive Structural Routing

## From Synchrony to Sequence: Exo-to-Ego Generation via Interpolation

### 核心方法概述

本方案针对原始硬频率分解的问题，提出了**结构引导的动态频率路由**机制。所有改进都在 `new/` 目录下实现，未修改原始代码。

---

## 1. 完整数学公式定义

### 1.1 Spectral Decomposition (谱分解)
使用高斯基函数进行软频率分解：

```math
F_l = S_l(F)  \quad \text{# low-frequency component}
```
```math
F_h = S_h(F)  \quad \text{# high-frequency component}
```

其中谱滤波器定义为：
```math
S_l(\omega) = \exp\left(-\frac{\|\omega\|^2}{2\sigma^2}\right)
```
```math
S_h(\omega) = 1 - S_l(\omega)
```

支持 **DCT** (默认) 和 **FFT** 两种变换。

---

### 1.2 Pose-Guided Frequency Routing (姿态引导频率路由)
采用残差形式进行动态路由，确保训练稳定：

```math
G = \sigma\left(\Phi\left(\left[F_h; \phi(K)\right]\right)\right)
```
```math
F_r = F_l + G \odot (F_h - F_l)
```

其中：
- `φ(K)`: 关键点特征投影
- `Φ`: 3D 卷积投影网络
- `[;]`: channel-wise concatenation
- `⊙`: element-wise multiplication

**关键改进**: 从直接相加改为 concat，避免特征语义冲突。

---

### 1.3 Structure-Weighted Asymmetric Loss (结构加权非对称损失)

```math
L_{struct} = \lambda_{fg} \cdot M \odot L_{fg} + \lambda_{bg} \cdot (1 - M) \odot L_{bg}
```

其中高斯姿态掩码 `M` 定义为：
```math
M = \exp\left(-\frac{\|x - k\|^2}{2\sigma_k^2}\right)
```

**设计理念**:
- **Foreground branch (`M`)**: 专注于手部动态和高频细节
- **Background branch (`1-M`)**: 保持场景一致性和低频结构

---

### 1.4 Motion-Aware Temporal Smoothness (运动感知时序平滑)

```math
L_{temp} = \lambda_{temp} \cdot \sum_t w_t \|G_t - G_{t-1}\|_1
```

其中运动自适应权重：
```math
w_t = \exp\left(-\frac{\|v_t\|}{\tau}\right)
```

- `v_t`: 关键点在 `t` 时刻的速度
- `τ`: temperature 参数（稳定权重缩放）

**效果**:
- 运动小 → 强平滑（稳定背景）
- 运动大 → 弱平滑（保留动作）

---

### 1.5 Routing Sparsity Regularization (路由稀疏正则化)

从 **Entropy Maximization** 改为 **Sparsity Regularization**：

```math
L_{sparse} = \lambda_{sparse} \cdot \|G \odot (1 - G)\|_1
```

**目的**: 鼓励 gate 值接近 0 或 1，避免均匀不确定状态。

---

### 1.6 Identity Preservation Constraint (身份保持约束)

使用冻结的 DINOv2 编码器保持语义结构连续性：

```math
L_{id} = \lambda_{id} \cdot \|E(I_{pred}) - E(I_{gt})\|_1
```

其中 `E` 是冻结的预训练编码器。

**作用**: 防止 latent drift，保持跨视角生成的语义一致性。

---

### 1.7 Cross-View Latent Alignment (跨视角潜空间对齐)

```math
L_{align} = \lambda_{align} \cdot \|z_{exo} - z_{ego}\|_1
```

应用于 pose mask 加权的共享语义区域，确保 exo 和 ego 潜空间对齐。

---

### 1.8 Total Objective (总目标函数)

```math
L_{total} = L_{diff} + L_{struct} + L_{temp} + L_{sparse} + L_{id} + L_{align}
```

各项权重（默认配置）:

| Loss Term | λ Value | Purpose |
|-----------|---------|---------|
| `L_diff` | 1.0 | 扩散重建损失 |
| `L_struct` | fg=2.0, bg=1.0 | 结构加权非对称损失 |
| `L_temp` | 0.1 | 运动感知时序平滑 |
| `L_sparse` | 0.01 | 路由稀疏正则化 |
| `L_id` | 0.1 | 身份保持约束 |
| `L_align` | 0.05 | 跨视角潜空间对齐 |

---

## 2. Gate Temporal Variance (可解释性指标)

用于量化路由门控的动态行为：

```math
Var_{fg} = Var(G_t) \quad \text{where } G_t > 0.7
```
```math
Var_{bg} = Var(G_t) \quad \text{where } G_t < 0.3
```

**预期行为**:
- **Foreground variance**: 较高（手部动态）
- **Background variance**: 较低（场景稳定）

这些指标会自动记录在 `gate_vis/gate_stats.txt` 中。

---

## 3. 文件结构

```
new/
├── configs/
│   └── default_config.py          # 更新配置（含所有新参数）
├── models/
│   ├── __init__.py                # 模块导出
│   ├── freq_routing.py            # 核心频率路由模块
│   ├── fasr_model.py              # FASR 主模型（含数学公式）
│   └── dino_extractor.py          # DINOv2 特征提取器
├── training/
│   ├── __init__.py
│   └── fasr_trainer.py            # 训练器（含 gate 可视化）
├── train.py                       # 训练入口
├── test_model.py                  # 模型测试脚本
└── README_IMPROVEMENTS.md         # 本文档（完整数学定义）
```

---

## 4. 必须修改的关键改进总结

| 项目 | 原方法 | 改进后 |
|------|--------|-------|
| Frequency Decomposition | 硬阈值切分 | 高斯谱基（DCT/FFT） |
| Routing Formulation | 直接融合 | Residual routing (更稳定) |
| Pose-Conditioning | Add | Concat + Projection |
| Regularization | Entropy Max | Sparsity Loss |
| Temporal Smoothness | 固定权重 | 运动感知 + Temperature τ |
| Identity Constraint | 无 | DINOv2 latent matching |
| Cross-View Alignment | 无 | M-weighted L1 loss |

---

## 5. 消融实验设计

| 实验设置 | 验证假设 |
|---------|---------|
| Baseline (no routing) | 原始模型性能 |
| + Hard FFT cut-off | 证明硬频率切分效果差 |
| + Soft Spectral Routing (no pose) | 软路由本身的作用 |
| + Pose Guidance (完整) | 结构先验的必要性 |
| + Asymmetric Struct Loss | 非对称监督的价值 |
| + Motion-Aware Temp Smooth | 运动感知平滑的效果 |
| + Identity Loss | 语义保持的效果 |
| + Cross-View Alignment | 跨视角对齐的价值 |
| + Sparsity Regularization | 稀疏正则化的效果 |

---

## 6. 论文主线包装

### 方法名称
**FASR: Frequency-Adaptive Structural Routing**

### 核心贡献点
1. 我们提出了一种结构引导的动态频率路由机制，通过人体关键点先验在潜空间中自适应调度高低频特征流
2. 我们引入了运动感知的时序平滑正则化，稳定跨视角插值轨迹
3. 我们设计了完整的稀疏性和身份保持约束，防止语义漂移和训练不稳定性
4. 我们在 Exo-to-Ego 生成任务上验证了 FASR 能显著提升动态一致性和生成稳定性

### 关键图表建议
1. **架构图**: 展示完整 pipeline，突出 residual routing 设计
2. **Gate 可视化热力图**: 手部区域高响应，背景低响应
3. **Gate Variance 统计**: FG vs BG 方差对比
4. **定性生成对比**: 与 baseline 的闪烁对比
5. **消融柱状图**: 各组件贡献分析

---

## 7. 训练策略建议

### Progressive Training (渐进式训练)

**Phase 1 (Epochs 0-20)**:
- 只启用 `lambda_diff` + `lambda_struct_fg` + `lambda_struct_bg`
- 目标: 让 gate 学习基本的手部响应
- 检查: `gate_vis/` 中的可视化

**Phase 2 (Epochs 20-50)**:
- 添加 `lambda_route_temp`
- 目标: 稳定时序行为

**Phase 3 (Epochs 50-100)**:
- 启用所有 loss (`lambda_sparse`, `lambda_id`, `lambda_align`)
- 目标: 精细调优，防止语义漂移

### 超参数调优提示
- `temp_tau`: 根据数据集运动幅度调整（小动作→调小，大动作→调大）
- `lambda_id`: 如果出现身份漂移可增大
- `spectral_sigma`: 控制高低频分离程度

---

## 8. 注意事项

1. **不修改原始文件**: 所有改进都在 `new/` 目录下，保持原始代码完整
2. **可视化检查**: 定期检查 `gate_vis/` 目录，确保 gate 学到了有意义的手部响应
3. **监控指标**: 关注 `fg_gate_variance` vs `bg_gate_variance`，验证时序平滑效果
4. **DCT vs FFT**: 默认 DCT 更稳定，可与 FFT 做 ablation 对比
5. **Residual Design**: 残差形式确保初始训练稳定，gate 初始为 0 时不改变特征

---

## 9. 最终评估指标

### 定量指标
- 标准视频生成指标 (PSNR, SSIM, LPIPS)
- 手部关键点精度
- 时序一致性指标 (TC)
- Gate variance ratio (FG/BG)

### 定性指标
- 手部结构保持
- 背景闪烁程度
- 动作平滑性
- Gate 响应可解释性

---

本方案已经从 "简单频率技巧" 升级为具有完整数学定义、可解释性分析和多维度约束的结构化方法。所有组件围绕 **Routing** 统一设计，形成了完整的方法论框架。
