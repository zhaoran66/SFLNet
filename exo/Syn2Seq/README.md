# Syn2Seq: From Synchrony to Sequence

This is an implementation of the paper **"From Synchrony to Sequence: Exo-to-Ego Generation via Interpolation"** using the DexYCB dataset.


## Overview

The paper proposes a novel approach for exocentric-to-egocentric (exo-to-ego) video generation by:

1. **Identifying the synchronization problem**: Directly generating ego videos from synchronized exo videos introduces spatial-temporal discontinuities
2. **Interpolating between views**: Creating a continuous sequence by interpolating between exo and ego frames
3. **Using Diffusion Forcing Transformers (DFoT)**: Modeling the sequence generation as a diffusion process

## Architecture

- **VideoInterpolator**: Generates intermediate frames between exo and ego views
- **DiffusionForcingTransformer**: The core diffusion model for sequence generation
- **Syn2SeqTrainer**: Implements the training logic with history conditioning

## Requirements

- Python 3.8+
- PyTorch 1.13+
- torchvision
- numpy
- PyYAML
- tqdm
- pillow
- scikit-image

## Dataset

Uses the **DexYCB** dataset located at:
```
/data/data2/kuanghaohong/Multiview/POEM/data/DexYCB/
```

The dataset contains multi-view RGB videos with camera poses.

## Training

```bash
cd Syn2Seq
python train.py
```

## Evaluation

```bash
python evaluate.py
```

## Configuration

Edit `configs/default_config.py` to adjust:
- Data paths and parameters
- Model architecture
- Training hyperparameters

## Project Structure

```
Syn2Seq/
?????? configs/
??   ?????? default_config.py     # Configuration
?????? data/
??   ?????? dataset.py            # DexYCB dataset loader
?????? models/
??   ?????? dfot.py               # Diffusion Forcing Transformer
??   ?????? interpolator.py       # Video interpolation module
?????? training/
??   ?????? trainer.py            # Training loop
?????? utils/
??   ?????? pose_utils.py         # Pose processing utilities
?????? train.py                  # Training entry point
?????? evaluate.py               # Evaluation script
```

## Key Features

1. **3D Patch Embedding**: For spatio-temporal tokenization
2. **Pose Conditioning**: Camera pose embeddings for geometric guidance
3. **Classifier-Free Guidance**: Optional conditioning dropout
4. **Mixed Precision Training**: For memory efficiency
5. **Visualization**: Generated samples during training

## Reference

```bibtex
@article{syn2seq2026,
  title={From Synchrony to Sequence: Exo-to-Ego Generation via Interpolation},
  author={Mohammad Mahdi and Nedko Savov and Danda Pani Paudel and Luc Van Gool},
  journal={arXiv preprint arXiv:2604.13793},
  year={2026}
}
```
