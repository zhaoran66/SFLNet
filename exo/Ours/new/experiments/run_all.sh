#!/bin/bash
# Run all 4 experiments on different GPUs

echo "========================================"
echo "Hand Pose Estimation - 4 Experiments"
echo "========================================"
echo ""

cd "$(dirname "$0")"

# Experiment 1: Baseline
echo "Starting Experiment 1: Baseline (GPU 1)"
echo "----------------------------------------"
CUDA_VISIBLE_DEVICES=1 python 1_baseline/train.py

# Experiment 2: Freq Decomp
echo ""
echo "Starting Experiment 2: Frequency Decomposition (GPU 2)"
echo "----------------------------------------"
CUDA_VISIBLE_DEVICES=2 python 2_freq_decomp/train.py

# Experiment 3: Latent Only
echo ""
echo "Starting Experiment 3: Latent Only (GPU 4)"
echo "----------------------------------------"
CUDA_VISIBLE_DEVICES=4 python 3_latent_only/train.py

# Experiment 4: Latent + Freq
echo ""
echo "Starting Experiment 4: Latent + Frequency (GPU 5)"
echo "----------------------------------------"
CUDA_VISIBLE_DEVICES=5 python 4_latent_freq/train.py

echo ""
echo "========================================"
echo "All experiments completed!"
echo "Results saved to outputs_hand_pose/"
echo "========================================"
