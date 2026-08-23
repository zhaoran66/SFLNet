# Hand Pose Estimation - 4 Experiments

All methods use **SAME interface**:
- **Input**: `(B, 3, T, 128, 128)` RGB video
- **Output**: `(B*T, 51)` MANO hand pose parameters

---

## Experiment 1: Baseline - Direct RGB Regression

**Folder**: `1_baseline/`

**GPU**: 1

**Pipeline**:
```
RGB video -> PixelEncoder (3D conv) -> RegressorHead
```

**Run**:
```bash
CUDA_VISIBLE_DEVICES=1 python 1_baseline/train.py
```

---

## Experiment 2: Baseline + Frequency Decomposition

**Folder**: `2_freq_decomp/`

**GPU**: 2

**Pipeline**:
```
RGB video -> DINOv2 feature extractor (frozen)
           -> SoftSpectralDecomposition (freq split)
           -> DinoFeatureEncoder x2 (low, high freq)
           -> RegressorHead
```

**Run**:
```bash
CUDA_VISIBLE_DEVICES=2 python 2_freq_decomp/train.py
```

---

## Experiment 3: Baseline + Latent (SD-VAE)

**Folder**: `3_latent_only/`

**GPU**: 4

**Pipeline**:
```
RGB video -> Pre-trained VAE encoder (frozen)
           -> LatentEncoder (3D conv)
           -> RegressorHead
```

**Run**:
```bash
CUDA_VISIBLE_DEVICES=4 python 3_latent_only/train.py
```

---

## Experiment 4: Baseline + Latent + Frequency Decomposition

**Folder**: `4_latent_freq/`

**GPU**: 5

**Pipeline**:
```
RGB video -> Pre-trained VAE encoder (frozen)
           -> SoftSpectralDecomposition (freq split)
           -> LatentEncoder x2 (low, high freq)
           -> RegressorHead
```

**Run**:
```bash
CUDA_VISIBLE_DEVICES=5 python 4_latent_freq/train.py
```

---

## Common Files

- `train_common.py` - Shared training utilities for all experiments

---

## Comparison Table

| Experiment | Method | GPU | Key Features |
|------------|--------|-----|--------------|
| 1 | Baseline | 1 | Direct RGB to pose |
| 2 | +Freq Decomp | 2 | DINOv2 + frequency split |
| 3 | +Latent | 4 | VAE latent space |
| 4 | +Latent +Freq | 5 | VAE + frequency split |
