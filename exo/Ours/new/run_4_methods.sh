#!/bin/bash
#
# 4 Hand Pose Estimation Methods - Run on Different GPUs
# =====================================================
# Method 1: Baseline (Direct RGB regression)       -> GPU 1
# Method 2: Baseline + Frequency Decomposition       -> GPU 2
# Method 3: Baseline + Latent (SD-VAE)               -> GPU 4
# Method 4: Baseline + Latent + Frequency Decomp     -> GPU 5
#

set -e

echo "=========================================="
echo "Hand Pose Estimation - 4 Methods"
echo "=========================================="
echo ""

OUTPUT_DIR="./outputs_hand_pose"
EPOCHS=100
BATCH_SIZE=4
LR=1e-4

cd /data/data5/zhaoran/paper_code/exo/Ours/new

echo "Experiment 1: Baseline - Direct RGB Regression"
echo "Running on GPU 1..."
echo "------------------------------------------"
CUDA_VISIBLE_DEVICES=1 python train_hand_pose_final.py \
    --method baseline \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --output_dir $OUTPUT_DIR

echo ""
echo "Experiment 2: Baseline + Background Frequency Decomposition"
echo "Running on GPU 2..."
echo "------------------------------------------"
CUDA_VISIBLE_DEVICES=2 python train_hand_pose_final.py \
    --method freq_decomp \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --output_dir $OUTPUT_DIR

echo ""
echo "Experiment 3: Baseline + Latent (SD-VAE)"
echo "Running on GPU 4..."
echo "------------------------------------------"
cd /data/data5/zhaoran/paper_code/exo/latent
CUDA_VISIBLE_DEVICES=4 python train_hand_pose_latent.py \
    --method latent_only \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --output_dir /data/data5/zhaoran/paper_code/exo/Ours/new/$OUTPUT_DIR

echo ""
echo "Experiment 4: Baseline + Latent + Background Frequency Decomposition"
echo "Running on GPU 5..."
echo "------------------------------------------"
CUDA_VISIBLE_DEVICES=5 python train_hand_pose_latent.py \
    --method latent_freq \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --output_dir /data/data5/zhaoran/paper_code/exo/Ours/new/$OUTPUT_DIR

echo ""
echo "=========================================="
echo "All 4 experiments completed!"
echo "Results saved to: /data/data5/zhaoran/paper_code/exo/Ours/new/$OUTPUT_DIR"
echo "=========================================="
