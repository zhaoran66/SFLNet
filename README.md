# SFLNet: Temporal Fourier Decomposition and Geodesic Latent Interpolation for Exo-to-Ego 3D Hand Pose Estimation

<p align="center">
  <b>Official PyTorch implementation of SFLNet</b>
</p>

<p align="center">
  <a href="https://doi.org/10.1109/LSP.2026.3733358">
    <img src="https://img.shields.io/badge/IEEE%20SPL-2026-blue.svg" alt="IEEE SPL 2026">
  </a>
  <a href="https://doi.org/10.1109/LSP.2026.3733358">
    <img src="https://img.shields.io/badge/Paper-IEEE%20Xplore-blue.svg" alt="Paper">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/Framework-PyTorch-orange.svg" alt="PyTorch">
  </a>
</p>

## 📖 Introduction

This repository provides the official implementation of:

**SFLNet: Temporal Fourier Decomposition and Geodesic Latent Interpolation for Exo-to-Ego 3D Hand Pose Estimation**

published in **IEEE Signal Processing Letters (SPL), 2026**.

Exo-to-ego 3D hand pose estimation aims to infer the 3D hand pose observed from an egocentric viewpoint using information captured from an exocentric viewpoint. This task is challenging because large viewpoint changes can lead to severe appearance variation, occlusion, and temporal inconsistency.

SFLNet addresses these challenges with two key components:

- **Temporal Fourier Decomposition (TFD)**  
  Decomposes temporal hand-motion representations in the frequency domain to preserve stable motion patterns while suppressing unstable high-frequency perturbations.

- **Geodesic Latent Interpolation (GLI)**  
  Performs interpolation along the geodesic path in latent space, enabling smoother and more geometrically meaningful exo-to-ego feature transformation.

By combining temporal frequency modeling with geometry-aware latent interpolation, SFLNet improves the robustness and consistency of cross-view 3D hand pose estimation.

---

## 🔥 News & Updates

- **2026-09**: Our paper has been officially published in **IEEE Signal Processing Letters**.
- **2026-09**: Official SFLNet repository released.
- More checkpoints, visualization examples, and documentation will be added progressively.

---

## 🧠 Method Overview

SFLNet follows a cross-view hand-pose estimation pipeline:

1. Extract temporal hand representations from the exocentric input sequence.
2. Apply **Temporal Fourier Decomposition** to separate stable motion information from temporal perturbations.
3. Perform **Geodesic Latent Interpolation** to bridge the exocentric and egocentric latent representations.
4. Decode the transformed representation into the target egocentric 3D hand pose.

The method is designed to explicitly model both:

- **temporal consistency**, and
- **cross-view geometric transition**.

### Main Contributions

1. We introduce a **Temporal Fourier Decomposition** strategy for modeling exo-to-ego hand motion in the frequency domain.
2. We propose **Geodesic Latent Interpolation** to achieve smooth and geometry-aware cross-view feature transformation.
3. The resulting SFLNet framework provides an effective solution for exo-to-ego 3D hand pose estimation on challenging hand-object interaction datasets.

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/zhaoran66/SFLNet.git
cd SFLNet
```

### 2. Create a conda environment

```bash
conda create -n sflnet python=3.8
conda activate sflnet
```

### 3. Install PyTorch

Please install a PyTorch version compatible with your CUDA environment.

For example:

```bash
pip install torch torchvision torchaudio
```

### 4. Install dependencies

If a `requirements.txt` file is provided:

```bash
pip install -r requirements.txt
```

Otherwise, please install the dependencies required by the corresponding training and evaluation scripts.

> The experiments in the paper were conducted on an NVIDIA RTX 3090 GPU.

---

## 📂 Dataset Preparation

SFLNet is evaluated on **DexYCB** and **H2O**.

Please download the datasets from their official sources and follow their corresponding licenses and usage agreements.

### DexYCB

DexYCB is a large-scale hand-object interaction dataset containing synchronized multi-view RGB-D sequences and 3D hand annotations.

Official project page:

https://dex-ycb.github.io/

### H2O

H2O is a dataset for egocentric hand-object interaction understanding with synchronized multi-view observations.

Official project page:

https://github.com/taeinkwon/h2o

### Recommended directory structure

A recommended dataset organization is:

```text
SFLNet/
├── data/
│   ├── DexYCB/
│   └── H2O/
├── ...
└── README.md
```

Please update the dataset paths in the corresponding configuration files before training or evaluation.

> Do not use machine-specific absolute paths such as `/data/...` in public configuration files.  
> We recommend using relative paths or user-defined configuration entries instead.

---

## 🚀 Usage

The exact training and evaluation commands depend on the configuration files included in this repository.

Before running an experiment, please make sure that:

1. the dataset path is correctly configured;
2. the GPU device is correctly specified;
3. the dataset split matches the setting used in the paper;
4. the checkpoint path is valid when performing evaluation.

### Training

Run the training entry provided in the repository with the desired configuration.

A typical workflow is:

```bash
python train.py
```

or, if the project uses configuration arguments:

```bash
python train.py --config <config_file>
```

### Evaluation

After training, evaluate a saved checkpoint using the evaluation script provided in the repository.

A typical workflow is:

```bash
python test.py
```

or:

```bash
python test.py --checkpoint <checkpoint_path>
```

> Please refer to the actual scripts and configuration files in this repository for the final command-line arguments.

---

## 📊 Experimental Results

### DexYCB

SFLNet achieves an **MPJPE of 30.42 mm** on DexYCB under the evaluation protocol used in the paper.

| Method | MPJPE ↓ |
|---|---:|
| **SFLNet** | **30.42 mm** |

### Efficiency

The reported inference time of SFLNet is approximately:

| GPU | Inference Time |
|---|---:|
| NVIDIA RTX 3090 | **4.49 ms / frame** |

Additional quantitative comparisons and ablation results are available in the paper.

---

## ⚙️ Important Settings

The following settings correspond to the experiments reported in the paper:

- Temporal Fourier threshold:  
  \(\alpha \in [0.05, 0.15]\)

- Geodesic interpolation steps:  
  \(K = 4\)

- Training sequence length:  
  **16 frames**

- H2O evaluation setting:  
  **single-frame input**

These settings may be adjusted depending on the dataset and experimental configuration.

---

## 🎨 Visualization

We recommend adding qualitative visualization results to an `assets/` directory, for example:

```text
SFLNet/
├── assets/
│   ├── framework.png
│   ├── qualitative_results.png
│   └── ...
```

They can then be shown in this README as:

```markdown
<p align="center">
  <img src="assets/framework.png" width="900">
</p>
```

A framework figure and qualitative exo-to-ego hand-pose visualization can help readers understand the method more quickly.

---

## 📄 Paper

**SFLNet: Temporal Fourier Decomposition and Geodesic Latent Interpolation for Exo-to-Ego 3D Hand Pose Estimation**

Published in **IEEE Signal Processing Letters, 2026**.

DOI: [10.1109/LSP.2026.3733358](https://doi.org/10.1109/LSP.2026.3733358)

---

## 🖊️ Citation

If you find this repository useful for your research, please cite our paper:

```bibtex
@article{zhao2026sflnet,
  author  = {Ran Zhao and Changlong Jiang and Ran Wang and Xiaofeng Yue and Lijun Zhu and Yang Xiao},
  title   = {SFLNet: Temporal Fourier Decomposition and Geodesic Latent Interpolation for Exo-to-Ego 3D Hand Pose Estimation},
  journal = {IEEE Signal Processing Letters},
  year    = {2026},
  doi     = {10.1109/LSP.2026.3733358}
}
```

---

## ⚠️ License

This repository is released for **academic research purposes**.

Please check the repository license file for detailed terms of use.  
The datasets used in this project are subject to their respective licenses.

---

## 📬 Contact

For questions about the code or paper, please open an issue in this repository.

Repository:

https://github.com/zhaoran66/SFLNet

---

## 🙏 Acknowledgements

We thank the authors and maintainers of the public datasets and open-source projects used in this work.

If you use code or models derived from other repositories, please also follow their licenses and citation requirements.
